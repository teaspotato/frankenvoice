import type { IOHandle, NodeDef } from '../core/node';
import type { AudioMsg, Json, Msg } from '../core/types';
import { isAudioMsg } from '../core/types';
import { bytesFromF32, f32FromBytes, PlayedTrack } from './audio';

// Noise / echo cancellation sits between the mic and everything that listens
// to it. Every variant emits the audio VAD and ASR should hear on 'audio', and
// per-frame levels on 'stats' so the dashboard can show what it removed.

const db = (energy: number) => (energy > 1e-12 ? 10 * Math.log10(energy) : -120);
const meanSquare = (x: Float32Array) => {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return x.length ? s / x.length : 0;
};
const round1 = (v: number) => Math.round(v * 10) / 10;

const audioIn = { name: 'audio', type: 'audio', description: 'Microphone audio' };
const audioOut = { name: 'audio', type: 'audio', description: 'Cleaned audio for VAD and ASR' };
const statsOut = { name: 'stats', type: 'stats', description: 'Per-frame levels in and out (dB)' };

/** Browser processing (or none) happens in getUserMedia; the stage just passes the audio on. */
function passthrough(type: string, title: string, description: string, params: NodeDef['params']): NodeDef {
  return {
    type,
    title,
    category: 'cleanup',
    description,
    inputs: [audioIn],
    outputs: [audioOut],
    params,
    create(ctx) {
      return {
        onInput(_port, msg) {
          if (isAudioMsg(msg)) ctx.emit('audio', msg);
        },
      };
    },
  };
}

export const browserCleanupNode = passthrough(
  'cleanup.browser',
  'Browser AEC/NS',
  "The browser's built-in echo cancellation, noise suppression and gain control (WebRTC), applied at capture. The mic delivers the processed audio, so there is nothing separate to compare against.",
  [
    { key: 'echoCancellation', label: 'Echo cancellation', type: 'boolean', default: true },
    { key: 'noiseSuppression', label: 'Noise suppression', type: 'boolean', default: true },
    { key: 'autoGainControl', label: 'Auto gain', type: 'boolean', default: true },
  ],
);

export const noCleanupNode = passthrough('cleanup.off', 'No cleanup', 'Raw microphone audio: all browser processing off, nothing applied.', []);

// ---- echo canceller -----------------------------------------------------------

export interface EchoCancellerParams {
  /** Length of the echo path the filter can model. */
  filterMs: number;
  /** NLMS step size (0..2; smaller converges slower but steadier). */
  stepSize: number;
  /** Extra delay applied to the reference before the filter. */
  delayMs: number;
  /** Geigel double-talk detector: near-end louder than this × the far-end peak freezes adaptation. */
  doubleTalk: number;
  /** Extra attenuation of the residual while only the far end talks (0 = off). */
  suppressionDb: number;
  adapt: boolean;
}

/**
 * Time-domain NLMS echo canceller. The reference is what the speaker played,
 * placed on the session clock; mic samples are placed on the same clock by
 * counting samples from the first frame. Pure arithmetic, so it replays exactly.
 */
export class EchoCanceller {
  readonly taps: number;
  private w: Float64Array;
  /** What the speaker played, on the session clock. */
  private ref: PlayedTrack;
  private dtHold = 0;
  private offset: number | null = null;

  constructor(
    readonly rate: number,
    private p: EchoCancellerParams,
    seconds = 60,
  ) {
    this.taps = Math.max(16, Math.round((p.filterMs * rate) / 1000));
    this.w = new Float64Array(this.taps);
    this.ref = new PlayedTrack(rate, seconds, this.taps * 4);
  }

  private at(k: number): number {
    return this.ref.get(k);
  }

  /** Add played audio starting at session time `atMs` (resampled to the mic rate). */
  addReference(atMs: number, samples: Float32Array, sampleRate: number) {
    this.ref.add(atMs, samples, sampleRate);
  }

  /** Playback was cut at `atMs`: nothing after that will be heard. */
  truncateReference(atMs: number) {
    this.ref.truncate(atMs);
  }

  /**
   * Cancel echo in mic samples [start, start + d.length), where mic sample 0
   * was captured at `t0Ms` on the session clock.
   */
  process(d: Float32Array, start: number, t0Ms: number): { out: Float32Array; refDb: number; doubleTalk: boolean } {
    this.offset ??= Math.round(((t0Ms - this.p.delayMs) * this.rate) / 1000);
    const L = this.taps;
    const w = this.w;
    const out = new Float32Array(d.length);
    const r0 = this.offset + start; // reference index aligned with d[0]
    // Far-end peak over the window this frame can see, for double-talk detection.
    let peak = 0;
    let refE = 0;
    for (let k = r0 - L + 1; k < r0 + d.length; k++) {
      const v = Math.abs(this.at(k));
      if (v > peak) peak = v;
      if (k >= r0) refE += v * v;
    }
    let dt = false;
    if (peak < 1e-4) {
      out.set(d);
      return { out, refDb: db(refE / d.length), doubleTalk: false };
    }
    // xs[L - 1 + n - i] is the reference sample i taps behind mic sample n.
    const xs = new Float64Array(L - 1 + d.length);
    for (let j = 0; j < xs.length; j++) xs[j] = this.at(r0 - L + 1 + j);
    let norm = 0;
    for (let j = 0; j < L; j++) norm += xs[j] * xs[j];
    const gain = Math.pow(10, -this.p.suppressionDb / 20);
    for (let n = 0; n < d.length; n++) {
      if (n > 0) {
        const gone = xs[n - 1];
        const added = xs[L - 1 + n];
        norm += added * added - gone * gone;
      }
      const base = L - 1 + n;
      let y = 0;
      for (let i = 0; i < L; i++) y += w[i] * xs[base - i];
      const e = d[n] - y;
      if (Math.abs(d[n]) > this.p.doubleTalk * peak) this.dtHold = Math.round(this.rate * 0.03);
      const talking = this.dtHold > 0;
      if (talking) {
        this.dtHold--;
        dt = true;
      }
      if (this.p.adapt && !talking && norm > 1e-8) {
        const g = (this.p.stepSize * e) / (norm + 1e-6);
        for (let i = 0; i < L; i++) w[i] += g * xs[base - i];
      }
      out[n] = talking ? e : e * gain;
    }
    return { out, refDb: db(refE / d.length), doubleTalk: dt };
  }
}

export const aecNode: NodeDef<EchoCancellerParams> = {
  type: 'cleanup.aec',
  title: 'Echo canceller (NLMS)',
  category: 'cleanup',
  description:
    'Adaptive echo canceller running in the page: subtracts an estimate of the agent audio (from the speaker) from the mic. Browser processing is off so you see exactly what it does.',
  inputs: [audioIn, { name: 'ref', type: 'played', description: 'What the speaker played, with its schedule' }],
  outputs: [audioOut, statsOut],
  params: [
    { key: 'filterMs', label: 'Filter length (ms)', type: 'number', default: 128, description: 'Longest echo path it can cancel. Cost grows with it.' },
    { key: 'stepSize', label: 'Step size (μ)', type: 'number', default: 0.2, description: 'NLMS adaptation rate, 0–2.' },
    { key: 'delayMs', label: 'Reference delay (ms)', type: 'number', default: 0, description: 'Bulk delay of the echo path not covered by the filter (output + input latency).' },
    { key: 'doubleTalk', label: 'Double-talk threshold', type: 'number', default: 0.6, description: 'Freeze adaptation when the mic exceeds this × the far-end peak.' },
    { key: 'suppressionDb', label: 'Residual suppression (dB)', type: 'number', default: 0, description: 'Extra attenuation while only the agent is talking.' },
    { key: 'adapt', label: 'Adapt', type: 'boolean', default: true },
  ],
  create(ctx) {
    let aec: EchoCanceller | null = null;
    let t0: number | null = null;
    const pendingRef: { at: number; samples: Float32Array; sampleRate: number }[] = [];
    return {
      onInput(port, msg: Msg) {
        if (port === 'ref') {
          if (isAudioMsg(msg) && typeof msg.at === 'number') {
            if (aec) aec.addReference(msg.at, msg.samples, msg.sampleRate);
            else pendingRef.push({ at: msg.at, samples: msg.samples, sampleRate: msg.sampleRate });
          } else if (msg.type === 'played.flush' && typeof msg.at === 'number') aec?.truncateReference(msg.at);
          return;
        }
        if (!isAudioMsg(msg)) return;
        if (!aec) {
          aec = new EchoCanceller(msg.sampleRate, ctx.params);
          for (const r of pendingRef.splice(0)) aec.addReference(r.at, r.samples, r.sampleRate);
          ctx.setState('taps', aec.taps);
        }
        // Mic sample 0 is placed at the arrival of the first frame minus its length.
        t0 ??= ctx.now() - ((msg.end - 0) / msg.sampleRate) * 1000;
        const { out, refDb, doubleTalk } = aec.process(msg.samples, msg.start, t0);
        const ref = ctx.media.append('audio', out, msg.sampleRate);
        ctx.emit('audio', { type: 'audio', ...ref, sampleRate: msg.sampleRate, samples: out });
        ctx.emit('stats', {
          type: 'cleanup.frame',
          track: msg.track,
          start: msg.start,
          end: msg.end,
          inDb: round1(db(meanSquare(msg.samples))),
          outDb: round1(db(meanSquare(out))),
          refDb: round1(refDb),
          doubleTalk,
        });
      },
    };
  },
};

// ---- RNNoise -----------------------------------------------------------------

interface RnnoiseParams {
  mix: number;
  gateBelow: number;
  gateDb: number;
  echoCancellation: boolean;
}

/**
 * RNNoise (neural noise suppression) runs behind the 'rnnoise' channel, so its
 * output replays as recorded. Mix and gate are applied here, on the recorded
 * output, so they can be retuned on a replay without diverging upstream.
 */
export const rnnoiseNode: NodeDef<RnnoiseParams> = {
  type: 'cleanup.rnnoise',
  title: 'RNNoise',
  category: 'cleanup',
  description: 'RNNoise neural noise suppression (WASM, in-browser, 48 kHz internally). Also reports its own voice probability.',
  inputs: [audioIn],
  outputs: [audioOut, statsOut],
  params: [
    { key: 'mix', label: 'Wet/dry mix', type: 'number', default: 1, description: '1 = fully denoised, 0 = original.' },
    { key: 'gateBelow', label: 'Gate below voice prob.', type: 'number', default: 0, description: 'Attenuate frames whose RNNoise voice probability is below this (0 = off).' },
    { key: 'gateDb', label: 'Gate attenuation (dB)', type: 'number', default: 20 },
    { key: 'echoCancellation', label: 'Browser echo cancellation', type: 'boolean', default: true, description: 'Keep the browser AEC on in front of RNNoise.' },
  ],
  channels: [{ name: 'rnnoise', description: 'Frames in, denoised frames (same length) and voice probability out' }],
  create(ctx) {
    const p = ctx.params;
    let io: IOHandle | undefined;
    const sent: AudioMsg[] = [];
    return {
      onInput(_port, msg: Msg) {
        if (!isAudioMsg(msg)) return;
        if (!io) {
          io = ctx.io.open('rnnoise', { sampleRate: msg.sampleRate }, {
            onRecv(data, bin) {
              const src = sent.shift();
              if (!src || !bin) return;
              const den = f32FromBytes(bin);
              const vad = (data as { vad?: number } | null)?.vad ?? 1;
              const gate = p.gateBelow > 0 && vad < p.gateBelow ? Math.pow(10, -p.gateDb / 20) : 1;
              const out = new Float32Array(src.samples.length);
              for (let i = 0; i < out.length; i++) out[i] = gate * (p.mix * (den[i] ?? 0) + (1 - p.mix) * src.samples[i]);
              const ref = ctx.media.append('audio', out, src.sampleRate);
              ctx.emit('audio', { type: 'audio', ...ref, sampleRate: src.sampleRate, samples: out });
              ctx.emit('stats', {
                type: 'cleanup.frame',
                track: src.track,
                start: src.start,
                end: src.end,
                inDb: round1(db(meanSquare(src.samples))),
                outDb: round1(db(meanSquare(out))),
                vad: Math.round(vad * 1000) / 1000,
                gated: gate < 1,
              });
            },
            onClose(data) {
              if (data && (data as { error?: Json }).error) ctx.log('error', 'RNNoise failed', data);
            },
          });
          ctx.setState('status', 'running');
        }
        sent.push(msg);
        io.send(null, { bin: bytesFromF32(msg.samples), media: { track: msg.track, start: msg.start, end: msg.end }, logBin: false });
      },
      stop() {
        io?.close();
      },
    };
  },
};

/** getUserMedia processing each cleanup variant wants in front of it. */
export function micProcessing(type: string, params: Record<string, Json>): { echoCancellation: boolean; noiseSuppression: boolean; autoGainControl: boolean } {
  const off = { echoCancellation: false, noiseSuppression: false, autoGainControl: false };
  if (type === 'cleanup.browser')
    return { echoCancellation: params.echoCancellation !== false, noiseSuppression: params.noiseSuppression !== false, autoGainControl: params.autoGainControl !== false };
  if (type === 'cleanup.rnnoise') return { ...off, echoCancellation: params.echoCancellation !== false };
  return off;
}
