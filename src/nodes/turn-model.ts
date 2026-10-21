import type { IOHandle, NodeDef } from '../core/node';
import type { Json, Msg } from '../core/types';
import { isAudioMsg } from '../core/types';
import { bytesFromF32, Framer, PlayedTrack } from './audio';
import { BcHead, type BcWeights } from './bc-head';
import bcWeights from './bc-head.json';

/** The model's block: 80 ms at 16 kHz. */
const BLOCK = 1280;
const RATE = 16000;
const FPS = 12.5;
const WHO = ['user', 'agent'] as const;

/**
 * TurnBench commit rule: a rising edge above the threshold fires, at most once
 * per refractory period (counting committed fires only). With `recommitFrames`,
 * a score that stays above the threshold that long after a fire fires again once.
 */
export class Committer {
  private prev = false;
  private last = -1e9;
  private runFrom = -1;

  constructor(
    private refractoryFrames: number,
    private recommitFrames = 0,
  ) {}

  step(i: number, p: number, theta: number): 'fire' | 'recommit' | null {
    const above = p > theta;
    let ev: 'fire' | 'recommit' | null = null;
    if (above && !this.prev && i - this.last >= this.refractoryFrames) {
      ev = 'fire';
      this.last = i;
      this.runFrom = i;
    }
    if (!above) this.runFrom = -1;
    else if (this.recommitFrames && this.runFrom >= 0 && i - this.runFrom === this.recommitFrames) {
      ev ??= 'recommit';
      this.runFrom = -1;
    }
    this.prev = above;
    return ev;
  }
}

/** What the 'turn-model' channel is opened with; the page preloads the same thing. */
export function turnModelRequest(p: { modelUrl: string; encoder: string; ep: string }) {
  return { encoderUrl: p.modelUrl + p.encoder, headUrl: `${p.modelUrl}head.onnx`, frames: Number(/_k(\d+)_/.exec(p.encoder)?.[1] ?? 2), ep: p.ep };
}

/** Stage state for a loading message from the model worker: a status line and, while downloading, the progress bar's numbers. */
export function loadingState(d: { type?: string; label?: string; done?: number; total?: number; cached?: boolean; text?: string }): { status: string; download: Json } {
  if (d.type === 'progress' && !d.cached) {
    const mb = (n = 0) => Math.round(n / 2 ** 20);
    const fraction = d.total ? Math.round(((d.done ?? 0) / d.total) * 100) / 100 : null;
    return { status: `downloading ${d.label}`, download: { label: d.label ?? '', done: mb(d.done), total: mb(d.total), fraction } };
  }
  return { status: d.type === 'progress' ? `${d.label} cached` : (d.text ?? 'loading'), download: null };
}

interface TurnModelParams {
  modelUrl: string;
  encoder: string;
  ep: string;
  eotThreshold: number;
  intThreshold: number;
  refractoryMs: number;
  eotRecommitMs: number;
  bcThreshold: number;
  bcGapMs: number;
}

interface Frame {
  t: number;
  eot: number[];
  int: number[];
  silent: number[];
  /** Per speaker, the head's frame classes (sessions recorded before backchannels lack it). */
  fine?: number[][];
}

const r4 = (v: number) => Math.round(v * 1e4) / 1e4;

const ports = {
  inputs: [
    { name: 'audio', type: 'audio', description: 'The user: cleaned microphone audio' },
    { name: 'played', type: 'played', description: 'The agent: what the speaker played, with its schedule' },
  ],
  outputs: [{ name: 'scores', type: 'turnscores', description: 'Per-frame end-of-turn / take-the-floor scores for both speakers, and the events they commit' }],
};

/**
 * The ssl_turn turn-taking model (streaming FastConformer + cross-speaker turn
 * head). Every 80 ms it scores both speakers, the user (mic) and the agent
 * (speaker output placed on the same clock), for end of turn and for taking
 * the floor. Inference runs in the browser behind the 'turn-model' channel, so
 * the scores replay exactly as they arrived; the thresholds are applied here,
 * on the recorded scores, so they can be retuned on a replay.
 */
export const turnModelNode: NodeDef<TurnModelParams> = {
  type: 'turnmodel.ssl',
  title: 'Turn model (ssl_turn)',
  category: 'control',
  description: 'Streaming turn-taking model (onnxruntime-node on the server CPU): end-of-turn and take-the-floor scores per 80 ms for the user and the agent.',
  ...ports,
  params: [
    {
      key: 'modelUrl',
      label: 'Model files URL',
      type: 'string',
      default: 'https://huggingface.co/spaces/shinohararin/turn-playground/resolve/main/models/',
      description: 'Folder holding the encoder and head.onnx. Downloaded once, then cached by the server.',
    },
    {
      key: 'encoder',
      label: 'Encoder',
      type: 'select',
      options: ['encoder_k2_w16.onnx', 'encoder_k4_w16.onnx'],
      default: 'encoder_k2_w16.onnx',
      description: 'k2 runs every 160 ms, k4 every 320 ms (cheaper, later).',
    },
    { key: 'ep', label: 'Backend', type: 'select', options: ['auto', 'cpu'], default: 'auto', description: 'The server runs it on the CPU (onnxruntime-node).' },
    { key: 'eotThreshold', label: 'End-of-turn threshold', type: 'number', default: 0.64, description: 'TurnBench dev operating point.' },
    { key: 'intThreshold', label: 'Take-the-floor threshold', type: 'number', default: 0.18, description: 'TurnBench dev operating point.' },
    { key: 'refractoryMs', label: 'Refractory (ms)', type: 'number', default: 500, description: 'Minimum time between two events of the same kind.' },
    { key: 'eotRecommitMs', label: 'End-of-turn re-commit (ms)', type: 'number', default: 1000, description: 'Fire end of turn again once if the score stays high this long (0 = off).' },
    {
      key: 'bcThreshold',
      label: 'Backchannel threshold',
      type: 'number',
      default: 0.4,
      description: "Backchannel head (a small MLP on this model's outputs): fire when the agent could say mm-hmm. TurnBench dev operating point.",
    },
    { key: 'bcGapMs', label: 'Backchannel refractory (ms)', type: 'number', default: 1000, description: 'Minimum time between two backchannel events.' },
  ],
  channels: [{ name: 'turn-model', description: '80 ms stereo blocks (user, agent) in, per-frame scores out' }],
  create(ctx) {
    const p = ctx.params;
    const framer = new Framer(BLOCK);
    const played = new PlayedTrack(RATE);
    let io: IOHandle | undefined;
    let ready = false;
    let unsupported = false;
    /** Session time of mic sample 0, and the mic sample the model's stream starts at. */
    let t0: number | null = null;
    let base = -1;
    let track = '';
    const refr = Math.max(1, Math.round(p.refractoryMs / (1000 / FPS)));
    const commit = WHO.map(() => ({
      eot: new Committer(refr, Math.round(p.eotRecommitMs / (1000 / FPS))),
      int: new Committer(refr),
    }));

    const bcHead = new BcHead(bcWeights as BcWeights);
    const bcCommit = new Committer(Math.max(1, Math.round(p.bcGapMs / (1000 / FPS))));
    const onFrames = (outs: Frame[]) => {
      for (const o of outs) {
        // Frame t has heard the audio up to the end of block t - 1.
        const at = base + BLOCK * o.t;
        const bc = o.fine ? r4(bcHead.step({ eot: o.eot, int: o.int, silent: o.silent, fine: o.fine })) : undefined;
        ctx.emit('scores', { type: 'turnmodel.frame', track, at, eot: o.eot.map(r4), int: o.int.map(r4), silent: o.silent.map(r4), ...(bc !== undefined ? { bc } : {}) });
        // The agent could backchannel the user here.
        if (bc !== undefined && bcCommit.step(o.t, bc, p.bcThreshold)) ctx.emit('scores', { type: 'turnmodel.bc', who: 'agent', track, at, p: bc });
        WHO.forEach((who, c) => {
          for (const kind of ['eot', 'int'] as const) {
            const ev = commit[c][kind].step(o.t, o[kind][c], kind === 'eot' ? p.eotThreshold : p.intThreshold);
            if (ev) ctx.emit('scores', { type: `turnmodel.${kind}`, who, track, at, p: r4(o[kind][c]), ...(ev === 'recommit' ? { recommit: true } : {}) });
          }
        });
      }
    };

    return {
      // Open the model at session start: it may already be loaded (see preloadTurnModel).
      start() {
        ctx.setState('status', 'loading');
        io = ctx.io.open('turn-model', turnModelRequest(p), {
          onRecv(data) {
            const d = data as { type: string; [k: string]: Json };
            if (d.type === 'progress' || d.type === 'status') {
              // What the diagram's progress bar and the lane show while the model loads.
              const l = loadingState(d as Parameters<typeof loadingState>[0]);
              ctx.setState('status', l.status);
              ctx.setState('download', l.download);
            } else if (d.type === 'ready') {
              ready = true;
              ctx.setState('download', null);
              ctx.setState('status', `running (${d.ep})`);
              ctx.emit('scores', { type: 'turnmodel.status', status: 'ready', ep: d.ep });
            } else if (d.type === 'frames') {
              ctx.setState('lag', `${d.backlog} blocks`);
              onFrames(d.outs as unknown as Frame[]);
            }
          },
          onClose(data) {
            ready = false;
            const err = (data as { error?: Json } | null)?.error;
            ctx.setState('download', null);
            ctx.setState('status', err ? 'failed' : 'closed');
            if (err) {
              ctx.log('error', `turn model: ${typeof err === 'string' ? err : JSON.stringify(err)}`, data);
              ctx.emit('scores', { type: 'turnmodel.status', status: 'failed' });
            }
          },
        });
      },
      onInput(port, msg: Msg) {
        if (port === 'played') {
          if (isAudioMsg(msg) && typeof msg.at === 'number') played.add(msg.at, msg.samples, msg.sampleRate);
          else if (msg.type === 'played.flush' && typeof msg.at === 'number') played.truncate(msg.at);
          return;
        }
        if (!isAudioMsg(msg)) return;
        if (msg.sampleRate !== RATE) {
          if (!unsupported) ctx.log('error', `the turn model needs 16 kHz audio, got ${msg.sampleRate}`);
          unsupported = true;
          return;
        }
        // Mic sample 0 is placed at the arrival of the first frame minus its length (as the AEC does).
        t0 ??= ctx.now() - (msg.end / msg.sampleRate) * 1000;
        // Audio before the model is ready isn't scored; its stream starts at the first block after.
        if (!ready || !io) return;
        framer.push(msg, (frame, tr, start, end) => {
          if (base < 0) base = start;
          track = tr;
          const agent = played.read(Math.round((t0! * RATE) / 1000) + start, new Float32Array(BLOCK));
          const both = new Float32Array(2 * BLOCK);
          both.set(frame);
          both.set(agent, BLOCK);
          io!.send(null, { bin: bytesFromF32(both), media: { track: tr, start, end }, logBin: false });
        });
      },
      stop() {
        io?.close();
      },
    };
  },
};

/** No turn model: turn-taking falls back to its own rules (VAD duration, ASR words, ASR endpointing). */
export const noTurnModelNode: NodeDef<Record<string, never>> = {
  type: 'turnmodel.off',
  title: 'No turn model',
  category: 'control',
  description: 'Turn-taking uses its built-in rules: VAD duration and backchannel words for barge-in, ASR endpointing for the end of a turn.',
  ...ports,
  params: [],
  create() {
    return {};
  },
};
