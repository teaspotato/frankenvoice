// Derives what the dashboard draws from a trajectory's events, incrementally:
// call update() whenever new events may have arrived. Works the same for a
// live session, a replay in progress, and a finished recording.

import type { Trajectory } from '../core/trajectory';
import type { Json, TrajEvent } from '../core/types';

/** Samples per waveform peak (10 ms at 16 kHz). */
const PEAK_MS = 10;

export interface Span {
  t0: number;
  t1?: number;
  text: string;
}

export interface AsrResult {
  /** When the result arrived. */
  t: number;
  /** Audio it covers, in session time. */
  a0: number;
  a1: number;
  text: string;
  final: boolean;
}

export interface LlmSpan extends Span {
  rid: string;
  firstToken?: number;
  status?: string;
}

export interface AudioBlock {
  t: number;
  peaks: Float32Array;
}

export interface CleanupFrame {
  t: number;
  inDb: number;
  outDb: number;
  refDb?: number;
  doubleTalk?: boolean;
  vad?: number;
  gated?: boolean;
}

export interface Marker {
  t: number;
  text: string;
}

export class TimelineModel {
  private next = 0;
  /** mic sample index -> session time, from the arrival of each mic frame. */
  private micSamples: number[] = [];
  private micTimes: number[] = [];
  private micRate = 16000;
  private micPeakSamples = 0;
  micPeaks: number[] = [];
  /** Cleaned audio (what VAD/ASR hear), same sample indexing as the mic. */
  private cleanPeakSamples = 0;
  cleanPeaks: number[] = [];
  cleanup: CleanupFrame[] = [];
  vad: { t: number; p: number }[] = [];
  vadThreshold?: number;
  vadNegThreshold?: number;
  speech: Span[] = [];
  /** Latest ASR result per segment (keyed by its first audio sample). */
  asr = new Map<number, AsrResult>();
  asrArrivals: { t: number; final: boolean }[] = [];
  floor: { t: number; state: string }[] = [];
  userTurns: Marker[] = [];
  llm: LlmSpan[] = [];
  ttsRequests: number[] = [];
  tts: (Span & { rid: string; seg: number })[] = [];
  playback: (Span & { interrupted?: boolean })[] = [];
  playbackSegments: Marker[] = [];
  boundaries: number[] = [];
  agentAudio: AudioBlock[] = [];
  errors: (Marker & { node: string })[] = [];
  /** Latest state per node. */
  state: Record<string, Record<string, Json>> = {};
  duration = 0;

  constructor(readonly traj: Trajectory) {}

  get peakMs() {
    return PEAK_MS;
  }

  /** Consume events appended since the last call. Returns whether anything changed. */
  update(): boolean {
    const ev = this.traj.events;
    if (this.next >= ev.length) return false;
    for (; this.next < ev.length; this.next++) this.consume(ev[this.next]);
    this.duration = ev[ev.length - 1].t;
    return true;
  }

  /** Session time at which mic sample `s` had been captured (interpolated). */
  sampleToT(s: number): number {
    const xs = this.micSamples;
    const ts = this.micTimes;
    if (!xs.length) return 0;
    if (s <= xs[0]) return ts[0] - ((xs[0] - s) / this.micRate) * 1000;
    if (s >= xs[xs.length - 1]) return ts[ts.length - 1] + ((s - xs[xs.length - 1]) / this.micRate) * 1000;
    let lo = 0;
    let hi = xs.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] <= s) lo = mid;
      else hi = mid;
    }
    const f = (s - xs[lo]) / (xs[hi] - xs[lo] || 1);
    return ts[lo] + f * (ts[hi] - ts[lo]);
  }

  /** Session time of mic peak `i` (peaks are equally spaced in audio time). */
  peakToT(i: number): number {
    return this.sampleToT((i + 0.5) * this.micRate * (PEAK_MS / 1000));
  }

  private peaks(samples: Float32Array, rate: number): Float32Array {
    const per = Math.max(1, Math.round((rate * PEAK_MS) / 1000));
    const out = new Float32Array(Math.ceil(samples.length / per));
    for (let i = 0; i < samples.length; i++) {
      const v = Math.abs(samples[i]);
      const k = (i / per) | 0;
      if (v > out[k]) out[k] = v;
    }
    return out;
  }

  private consume(e: TrajEvent) {
    const d = (e.data ?? {}) as Record<string, any>;
    if (e.kind === 'state') {
      this.state[e.node] = { ...this.state[e.node], ...(d as Record<string, Json>) };
      if (typeof d.floor === 'string') this.floor.push({ t: e.t, state: d.floor });
      return;
    }
    if (e.kind === 'node.start' && typeof d.type === 'string' && d.type.startsWith('vad.')) {
      this.vadThreshold = d.params?.threshold;
      this.vadNegThreshold = d.params?.negThreshold;
      return;
    }
    if (e.kind === 'log' && d.level === 'error') {
      this.errors.push({ t: e.t, node: e.node, text: String(d.message) });
      return;
    }
    if (e.kind === 'io.open' && e.node === 'tts') {
      this.ttsRequests.push(e.t);
      return;
    }
    if (e.kind === 'io.send' && d.payload?.cmd === 'play' && e.media) {
      const samples = this.traj.tracks.readF32Range(e.media.track, e.media.start, e.media.end);
      this.agentAudio.push({ t: d.payload.at, peaks: this.peaks(samples, d.payload.sampleRate) });
      return;
    }
    if (e.kind !== 'msg') return;
    switch (d.type) {
      case 'audio':
        if (e.node === 'cleanup' && e.media && e.port === 'audio') {
          const per = Math.round((this.micRate * PEAK_MS) / 1000);
          const upTo = Math.floor(e.media.end / per) * per;
          if (upTo > this.cleanPeakSamples) {
            const s = this.traj.tracks.readF32Range(e.media.track, this.cleanPeakSamples, upTo);
            this.cleanPeaks.push(...this.peaks(s, this.micRate));
            this.cleanPeakSamples = upTo;
          }
        }
        if (e.node === 'mic' && e.media) {
          this.micRate = d.sampleRate ?? this.micRate;
          this.micSamples.push(e.media.end);
          this.micTimes.push(e.t);
          const per = Math.round((this.micRate * PEAK_MS) / 1000);
          // Peaks over whole PEAK_MS blocks; a partial block waits for the next frame.
          const upTo = Math.floor(e.media.end / per) * per;
          if (upTo > this.micPeakSamples) {
            const s = this.traj.tracks.readF32Range(e.media.track, this.micPeakSamples, upTo);
            this.micPeaks.push(...this.peaks(s, this.micRate));
            this.micPeakSamples = upTo;
          }
        }
        break;
      case 'cleanup.frame':
        this.cleanup.push({ t: this.sampleToT(d.end), inDb: d.inDb, outDb: d.outDb, refDb: d.refDb, doubleTalk: d.doubleTalk, vad: d.vad, gated: d.gated });
        break;
      case 'vad.frame':
        this.vad.push({ t: this.sampleToT(d.end), p: d.p });
        break;
      case 'vad.speech_start':
        this.speech.push({ t0: this.sampleToT(d.at), text: '' });
        break;
      case 'vad.speech_end': {
        const s = this.speech[this.speech.length - 1];
        if (s && s.t1 === undefined) s.t1 = this.sampleToT(d.at);
        break;
      }
      case 'asr.partial':
      case 'asr.final': {
        const final = d.type === 'asr.final';
        this.asrArrivals.push({ t: e.t, final });
        if (typeof d.start === 'number' && typeof d.end === 'number')
          this.asr.set(d.start, { t: e.t, a0: this.sampleToT(d.start), a1: this.sampleToT(d.end), text: d.text ?? '', final });
        break;
      }
      case 'turn.user':
        this.userTurns.push({ t: e.t, text: d.text ?? '' });
        break;
      case 'llm.start':
        this.llm.push({ rid: d.rid, t0: e.t, text: '' });
        break;
      case 'llm.delta': {
        const l = this.llm.find((x) => x.rid === d.rid);
        if (l) {
          l.firstToken ??= e.t;
          l.text += d.text ?? '';
        }
        break;
      }
      case 'llm.end': {
        const l = this.llm.find((x) => x.rid === d.rid);
        if (l) {
          l.t1 = e.t;
          l.status = d.status;
        }
        break;
      }
      case 'tts.segment_start':
        this.tts.push({ rid: d.rid, seg: d.seg, t0: e.t, text: d.text ?? '' });
        break;
      case 'tts.segment_end': {
        const s = this.tts.find((x) => x.rid === d.rid && x.seg === d.seg);
        if (s) s.t1 = e.t;
        break;
      }
      case 'playback.start':
        this.playback.push({ t0: d.at ?? e.t, text: d.rid ?? '' });
        break;
      case 'playback.segment':
        this.playbackSegments.push({ t: d.at ?? e.t, text: d.text ?? '' });
        break;
      case 'playback.boundary':
        this.boundaries.push(d.at ?? e.t);
        break;
      case 'playback.end':
      case 'playback.interrupted': {
        const p = this.playback[this.playback.length - 1];
        if (p && p.t1 === undefined) {
          // The speaker reports the end when its timer fires; the audio itself ended at its schedule.
          p.t1 = d.at ?? e.t;
          p.interrupted = d.type === 'playback.interrupted';
        }
        break;
      }
    }
  }
}
