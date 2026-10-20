import type { NodeContext, NodeDef, ParamDef } from '../core/node';
import type { Msg } from '../core/types';
import { isAudioMsg } from '../core/types';
import { bytesFromF32, Framer } from './audio';

interface SegmenterParams {
  threshold: number;
  negThreshold: number;
  minSpeechMs: number;
  redemptionMs: number;
  emitFrames: boolean;
}

const segmenterParams: ParamDef[] = [
  { key: 'threshold', label: 'Speech threshold', type: 'number', default: 0.5 },
  { key: 'negThreshold', label: 'Silence threshold', type: 'number', default: 0.35 },
  { key: 'minSpeechMs', label: 'Min speech (ms)', type: 'number', default: 96 },
  { key: 'redemptionMs', label: 'Hangover before end (ms)', type: 'number', default: 320 },
  { key: 'emitFrames', label: 'Emit per-frame probabilities', type: 'boolean', default: true },
];

/**
 * Turns per-frame speech probabilities into speech_start / speech_end events
 * with hysteresis. Boundaries are reported as sample positions, so downstream
 * nodes know exactly which audio the decision refers to.
 */
class Segmenter {
  private speaking = false;
  private candidateStart = -1;
  private speechFrames = 0;
  private silenceStart = -1;
  private silenceFrames = 0;

  constructor(
    private ctx: NodeContext<SegmenterParams & { sampleRate?: number }>,
    private frameMs: number,
  ) {}

  push(p: number, track: string, start: number, end: number): void {
    const prm = this.ctx.params;
    if (prm.emitFrames) this.ctx.emit('vad', { type: 'vad.frame', track, start, end, p });
    const minSpeech = Math.max(1, Math.round(prm.minSpeechMs / this.frameMs));
    const redemption = Math.max(1, Math.round(prm.redemptionMs / this.frameMs));

    if (!this.speaking) {
      if (p >= prm.threshold) {
        if (this.candidateStart < 0) this.candidateStart = start;
        this.speechFrames++;
        if (this.speechFrames >= minSpeech) {
          this.speaking = true;
          this.silenceFrames = 0;
          this.ctx.setState('speaking', true);
          this.ctx.emit('vad', { type: 'vad.speech_start', track, at: this.candidateStart, p });
        }
      } else if (p < prm.negThreshold) {
        this.candidateStart = -1;
        this.speechFrames = 0;
      }
      return;
    }
    if (p < prm.negThreshold) {
      if (this.silenceFrames === 0) this.silenceStart = start;
      this.silenceFrames++;
      if (this.silenceFrames >= redemption) {
        this.speaking = false;
        this.ctx.setState('speaking', false);
        this.ctx.emit('vad', { type: 'vad.speech_end', track, at: this.silenceStart, from: this.candidateStart, p });
        this.candidateStart = -1;
        this.speechFrames = 0;
        this.silenceFrames = 0;
      }
    } else if (p >= prm.threshold) {
      this.silenceFrames = 0;
    }
  }
}

/** FireRedVAD defaults: the shared segmenter settings, plus the 5-frame smoothing FireRedVAD's own postprocessor uses. Not tuned yet. */
const FIRERED_DEFAULTS = { smoothFrames: 5, threshold: 0.5, negThreshold: 0.35, minSpeechMs: 96, redemptionMs: 320 };

interface SileroParams extends SegmenterParams {
  model: string;
}

/** Silero VAD. Inference runs in the browser behind the 'silero' channel, so its outputs replay as-is. */
export const sileroVadNode: NodeDef<SileroParams> = {
  type: 'vad.silero',
  title: 'Silero VAD',
  category: 'vad',
  description: 'Silero voice activity detection (ONNX, in-browser), 512-sample frames at 16 kHz.',
  inputs: [{ name: 'audio', type: 'audio' }],
  outputs: [{ name: 'vad', type: 'vad' }],
  params: [{ key: 'model', label: 'Model', type: 'select', options: ['v5', 'v6'], default: 'v5' }, ...segmenterParams],
  channels: [{ name: 'silero', description: 'Frames in, speech probabilities out' }],
  create(ctx) {
    const FRAME = 512;
    const framer = new Framer(FRAME);
    let seg: Segmenter | undefined;
    let io: ReturnType<typeof ctx.io.open> | undefined;
    const pending: { track: string; start: number; end: number }[] = [];
    return {
      onInput(_port, msg: Msg) {
        if (!isAudioMsg(msg)) return;
        if (!io) {
          if (msg.sampleRate !== 16000) ctx.log('error', `Silero expects 16 kHz audio, got ${msg.sampleRate}`);
          seg = new Segmenter(ctx, (FRAME / msg.sampleRate) * 1000);
          io = ctx.io.open('silero', { model: ctx.params.model, sampleRate: msg.sampleRate, frameSamples: FRAME }, {
            onRecv(data) {
              const r = data as { p: number };
              const f = pending.shift();
              if (f) seg!.push(r.p, f.track, f.start, f.end);
            },
            onClose(data) {
              ctx.setState('status', 'closed');
              if (data) ctx.log('warn', 'VAD model closed', data);
            },
          });
          ctx.setState('status', 'running');
        }
        framer.push(msg, (frame, track, start, end) => {
          pending.push({ track, start, end });
          io!.send(null, { bin: bytesFromF32(frame), media: { track, start, end }, logBin: false });
        });
      },
      stop() {
        io?.close();
      },
    };
  },
};

interface FireRedParams extends SegmenterParams {
  smoothFrames: number;
}

/** Shared segmenter params with defaults suited to a model, e.g. one with a different frame rate. */
const withDefaults = (d: Partial<Record<keyof SegmenterParams, number | boolean>>): ParamDef[] =>
  segmenterParams.map((p) => (d[p.key as keyof SegmenterParams] !== undefined ? { ...p, default: d[p.key as keyof SegmenterParams]! } : p));

/**
 * FireRedVAD (streaming DFSMN, FireRedTeam/FireRedVAD, Apache-2.0). Inference runs in the browser behind
 * the 'firered' channel: 30 ms chunks in, one probability per 10 ms frame out, so its outputs replay as-is.
 * Frame k of the stream covers samples [160k, 160k + 400); its probability is smoothed over the last
 * `smoothFrames` frames (as FireRedVAD's own postprocessor does) before the segmenter sees it.
 */
export const fireredVadNode: NodeDef<FireRedParams> = {
  type: 'vad.firered',
  title: 'FireRedVAD',
  category: 'vad',
  description: 'FireRedVAD streaming voice activity detection (ONNX, in-browser): Kaldi fbank, causal DFSMN, 10 ms frames at 16 kHz.',
  inputs: [{ name: 'audio', type: 'audio' }],
  outputs: [{ name: 'vad', type: 'vad' }],
  params: [
    { key: 'smoothFrames', label: 'Smoothing (frames of 10 ms)', type: 'number', default: FIRERED_DEFAULTS.smoothFrames },
    ...withDefaults(FIRERED_DEFAULTS),
  ],
  channels: [{ name: 'firered', description: 'Audio chunks in, per-frame speech probabilities out' }],
  create(ctx) {
    const CHUNK = 480;
    const HOP = 160;
    const WINDOW = 400;
    const framer = new Framer(CHUNK);
    let seg: Segmenter | undefined;
    let io: ReturnType<typeof ctx.io.open> | undefined;
    /** Chunks sent and not yet answered, with where each sits in the stream the model has seen. */
    const pending: { track: string; start: number; v: number }[] = [];
    let sent = 0;
    let frame = 0;
    const recent: number[] = [];
    let sum = 0;
    return {
      onInput(_port, msg: Msg) {
        if (!isAudioMsg(msg)) return;
        if (!io) {
          if (msg.sampleRate !== 16000) ctx.log('error', `FireRedVAD expects 16 kHz audio, got ${msg.sampleRate}`);
          seg = new Segmenter(ctx, (HOP / msg.sampleRate) * 1000);
          io = ctx.io.open('firered', { model: 'stream-vad', sampleRate: msg.sampleRate, chunkSamples: CHUNK }, {
            onRecv(data) {
              const { p } = data as { p: number[] };
              const c = pending.shift();
              if (!c) return;
              const n = Math.max(1, Math.round(ctx.params.smoothFrames));
              for (const x of p) {
                // This frame's window ends inside the chunk that completed it.
                const end = c.start + (frame++ * HOP + WINDOW - c.v);
                recent.push(x);
                sum += x;
                while (recent.length > n) sum -= recent.shift()!;
                seg!.push(Math.round((sum / recent.length) * 1e4) / 1e4, c.track, end - WINDOW, end);
              }
            },
            onClose(data) {
              ctx.setState('status', 'closed');
              if (data) ctx.log('warn', 'VAD model closed', data);
            },
          });
          ctx.setState('status', 'running');
        }
        framer.push(msg, (chunk, track, start, end) => {
          pending.push({ track, start, v: sent });
          sent += chunk.length;
          io!.send(null, { bin: bytesFromF32(chunk), media: { track, start, end }, logBin: false });
        });
      },
      stop() {
        io?.close();
      },
    };
  },
};

interface EnergyParams extends SegmenterParams {
  thresholdDb: number;
  frameSamples: number;
}

/** Energy-based VAD: no model, no IO, fully deterministic. Useful as a baseline and for tests. */
export const energyVadNode: NodeDef<EnergyParams> = {
  type: 'vad.energy',
  title: 'Energy VAD',
  category: 'vad',
  description: 'RMS-energy voice activity detection mapped to a pseudo-probability.',
  inputs: [{ name: 'audio', type: 'audio' }],
  outputs: [{ name: 'vad', type: 'vad' }],
  params: [
    { key: 'thresholdDb', label: 'Threshold (dBFS)', type: 'number', default: -40 },
    { key: 'frameSamples', label: 'Frame size (samples)', type: 'number', default: 512 },
    ...segmenterParams,
  ],
  create(ctx) {
    const framer = new Framer(ctx.params.frameSamples);
    let seg: Segmenter | undefined;
    return {
      onInput(_port, msg) {
        if (!isAudioMsg(msg)) return;
        seg ??= new Segmenter(ctx, (ctx.params.frameSamples / msg.sampleRate) * 1000);
        framer.push(msg, (frame, track, start, end) => {
          let sum = 0;
          for (const s of frame) sum += s * s;
          const db = 10 * Math.log10(sum / frame.length + 1e-12);
          const p = 1 / (1 + Math.exp(-(db - ctx.params.thresholdDb) / 3));
          seg!.push(Math.round(p * 1e4) / 1e4, track, start, end);
        });
      },
    };
  },
};
