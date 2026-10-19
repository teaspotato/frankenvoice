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
  /** Events that make up this span, for following causes and effects. */
  seqs: number[];
}

export interface AsrResult {
  /** When the result arrived. */
  t: number;
  /** Its words, in session time: first word start to last word end (the window when there are no words). */
  a0: number;
  a1: number;
  text: string;
  final: boolean;
  /** Every partial and final result for this audio. */
  seqs: number[];
}

export interface LlmSpan extends Span {
  rid: string;
  firstToken?: number;
  status?: string;
  /** Streamed chunks as they arrived. */
  chunks: { t: number; text: string; seq: number }[];
  /** The reply was started early on a draft of the user's words, this long before the turn was committed. */
  headStartMs?: number;
  /** When the model's first reasoning arrived. */
  thinkingAt?: number;
}

/** A reply started on the user's words before their turn was committed. */
export interface Draft extends Span {
  rid: string;
  /** kept (became the reply) or dropped. */
  status?: string;
  why?: string;
}

/** A filler the opener stage said while the model was thinking: from the reply's start to when it was said. */
export interface Opener extends Span {
  rid: string;
  why?: string;
}

export interface AudioBlock {
  t: number;
  peaks: Float32Array;
  /** A backchannel: on its own output, so cutting the reply off doesn't cut it. */
  cue?: boolean;
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
  seq: number;
}

/** What led to a set of events and what they led to, through each event's recorded causes. */
export interface Related {
  up: Set<number>;
  down: Set<number>;
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
  /** Turn model scores per frame, [user, agent], placed on the audio they have heard. */
  turnScores: { t: number; eot: number[]; int: number[]; bc?: number }[] = [];
  /** End-of-turn / take-the-floor events the turn model committed. */
  turnEvents: (Marker & { kind: 'eot' | 'int' | 'bc'; who: string; p: number; arrived: number })[] = [];
  turnEotThreshold?: number;
  turnIntThreshold?: number;
  /** Latest ASR result per segment (keyed by its first audio sample). */
  asr = new Map<number, AsrResult>();
  asrArrivals: { t: number; final: boolean; text: string; seq: number; a0?: number; a1?: number }[] = [];
  floor: { t: number; state: string; seq: number }[] = [];
  userTurns: (Marker & { a0?: number; a1?: number; replaces?: string })[] = [];
  /** User speech over the agent and what turn-taking made of it. */
  bargeIns: { t0: number; t1?: number; decision?: string; why?: string; text: string; seqs: number[] }[] = [];
  llm: LlmSpan[] = [];
  drafts: Draft[] = [];
  openers: Opener[] = [];
  ttsRequests: { t: number; seq: number }[] = [];
  tts: (Span & { rid: string; seg: number })[] = [];
  playback: (Span & { rid: string; interrupted?: boolean; heard?: string })[] = [];
  playbackSegments: Marker[] = [];
  /** The agent was cut off (paused) or played quieter (ducked) while turn-taking judged the user's speech over it. */
  holds: Span[] = [];
  /** Backchannels the agent said while the user talked, and proposals that were gated off. */
  backchannels: (Span & { id: string; why?: string })[] = [];
  bcSkips: (Marker & { why: string })[] = [];
  boundaries: number[] = [];
  agentAudio: AudioBlock[] = [];
  errors: (Marker & { node: string })[] = [];
  /** Events caused by each event (the inverse of TrajEvent.cause). */
  private children = new Map<number, number[]>();
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

  private closeHold(at: number, seq: number) {
    const open = this.holds.at(-1);
    if (!open || open.t1 !== undefined) return;
    open.t1 = Math.max(open.t0, at);
    open.seqs.push(seq);
  }

  private clipAgentAudio(at: number) {
    this.playbackSegments = this.playbackSegments.filter((s) => s.t < at);
    this.agentAudio = this.agentAudio.flatMap((b) => {
      if (b.cue) return [b];
      if (b.t >= at) return [];
      const keep = Math.ceil((at - b.t) / PEAK_MS);
      return [keep < b.peaks.length ? { ...b, peaks: b.peaks.slice(0, keep) } : b];
    });
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

  /**
   * Everything upstream of `seqs` (what triggered them) and downstream (what
   * they produced), following recorded causes only: up through causes, down
   * through effects, never sideways, so a shared connection or the session
   * start doesn't relate everything to everything.
   */
  related(seqs: number[]): Related {
    const self = new Set(seqs);
    const walk = (next: (s: number) => readonly number[] | undefined) => {
      const out = new Set<number>();
      const stack = [...seqs];
      while (stack.length) {
        for (const n of next(stack.pop()!) ?? []) {
          if (out.has(n) || self.has(n)) continue;
          out.add(n);
          stack.push(n);
        }
      }
      return out;
    };
    return { up: walk((s) => this.event(s)?.cause), down: walk((s) => this.children.get(s)) };
  }

  private event(seq: number): TrajEvent | undefined {
    const ev = this.traj.events;
    return ev[seq]?.seq === seq ? ev[seq] : ev.find((e) => e.seq === seq);
  }

  private consume(e: TrajEvent) {
    const d = (e.data ?? {}) as Record<string, any>;
    for (const c of e.cause ?? []) {
      const l = this.children.get(c);
      if (l) l.push(e.seq);
      else this.children.set(c, [e.seq]);
    }
    if (e.kind === 'state') {
      this.state[e.node] = { ...this.state[e.node], ...(d as Record<string, Json>) };
      if (typeof d.floor === 'string') this.floor.push({ t: e.t, state: d.floor, seq: e.seq });
      return;
    }
    if (e.kind === 'node.start' && typeof d.type === 'string' && d.type.startsWith('vad.')) {
      this.vadThreshold = d.params?.threshold;
      this.vadNegThreshold = d.params?.negThreshold;
      return;
    }
    if (e.kind === 'node.start' && d.type === 'turnmodel.ssl') {
      this.turnEotThreshold = d.params?.eotThreshold;
      this.turnIntThreshold = d.params?.intThreshold;
      return;
    }
    if (e.kind === 'log' && d.level === 'error') {
      this.errors.push({ t: e.t, node: e.node, text: String(d.message), seq: e.seq });
      return;
    }
    // A request per sentence, or on the streaming socket, each flush (a sentence sent to be spoken).
    const flush = e.kind === 'io.send' && e.node === 'tts' && (e.data as { payload?: { type?: string } })?.payload?.type === 'Flush';
    if ((e.kind === 'io.open' && e.node === 'tts') || flush) {
      this.ttsRequests.push({ t: e.t, seq: e.seq });
      return;
    }
    if (e.kind === 'io.send' && d.payload?.cmd === 'cue' && e.media) {
      const samples = this.traj.tracks.readF32Range(e.media.track, e.media.start, e.media.end);
      this.agentAudio.push({ t: d.payload.at, peaks: this.peaks(samples, d.payload.sampleRate), cue: true });
      return;
    }
    if (e.kind === 'io.send' && d.payload?.cmd === 'play' && e.media) {
      const samples = this.traj.tracks.readF32Range(e.media.track, e.media.start, e.media.end);
      this.agentAudio.push({ t: d.payload.at, peaks: this.peaks(samples, d.payload.sampleRate), cue: true });
      this.playback[this.playback.length - 1]?.seqs.push(e.seq);
      return;
    }
    if (e.kind !== 'msg') return;
    switch (d.type) {
      case 'audio':
        if (e.node === 'tts' && d.rid !== undefined) this.tts.find((x) => x.rid === d.rid && x.seg === d.seg)?.seqs.push(e.seq);
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
      case 'turnmodel.frame':
        this.turnScores.push({ t: this.sampleToT(d.at), eot: d.eot, int: d.int, bc: d.bc });
        break;
      case 'turnmodel.eot':
      case 'turnmodel.int':
      case 'turnmodel.bc':
        this.turnEvents.push({ t: this.sampleToT(d.at), arrived: e.t, kind: d.type.slice('turnmodel.'.length), who: d.who, p: d.p, text: '', seq: e.seq });
        break;
      case 'vad.speech_start':
        this.speech.push({ t0: this.sampleToT(d.at), text: '', seqs: [e.seq] });
        break;
      case 'vad.speech_end': {
        const s = this.speech[this.speech.length - 1];
        if (s && s.t1 === undefined) {
          s.t1 = this.sampleToT(d.at);
          s.seqs.push(e.seq);
        }
        break;
      }
      case 'asr.partial':
      case 'asr.final': {
        const final = d.type === 'asr.final';
        // The provider's window (start/end) begins where its previous final ended, so it takes in the
        // silence before the words; the words' own times say where the speech is.
        const words: { start: number; end: number }[] = Array.isArray(d.words) ? d.words : [];
        const span = words.length ? { start: words[0].start, end: words[words.length - 1].end } : { start: d.start, end: d.end };
        const covers = typeof span.start === 'number' && typeof span.end === 'number' ? { a0: this.sampleToT(span.start), a1: this.sampleToT(span.end) } : {};
        this.asrArrivals.push({ t: e.t, final, text: d.text ?? '', seq: e.seq, ...covers });
        if (typeof d.start === 'number' && typeof d.end === 'number') {
          const seqs = [...(this.asr.get(d.start)?.seqs ?? []), e.seq];
          this.asr.set(d.start, { t: e.t, a0: covers.a0!, a1: covers.a1!, text: d.text ?? '', final, seqs });
        }
        break;
      }
      case 'turn.user': {
        const audio = typeof d.from === 'number' && d.from >= 0 ? { a0: this.sampleToT(d.from), a1: this.sampleToT(d.to) } : {};
        this.userTurns.push({ t: e.t, text: d.text ?? '', seq: e.seq, ...audio, ...(d.replaces ? { replaces: d.replaces } : {}) });
        break;
      }
      case 'turn.decision': {
        if (d.decision === 'candidate') {
          this.bargeIns.push({ t0: e.t, text: '', seqs: [e.seq] });
          break;
        }
        const b = this.bargeIns[this.bargeIns.length - 1];
        if (b && b.t1 === undefined) {
          b.t1 = e.t;
          b.decision = d.decision;
          b.why = d.why;
          b.text = d.text ?? '';
          b.seqs.push(e.seq);
        }
        break;
      }
      case 'turn.heard': {
        const p = this.playback[this.playback.length - 1];
        if (p) {
          p.heard = d.text ?? '';
          p.seqs.push(e.seq);
        }
        break;
      }
      case 'llm.start':
        this.llm.push({ rid: d.rid, t0: e.t, text: '', chunks: [], seqs: [e.seq], ...(typeof d.headStartMs === 'number' ? { headStartMs: d.headStartMs } : {}) });
        break;
      case 'llm.draft':
        this.drafts.push({ rid: d.rid, t0: e.t, text: d.text ?? '', seqs: [e.seq] });
        break;
      case 'opener.say': {
        const l = this.llm.find((x) => x.rid === d.rid);
        this.openers.push({ rid: d.rid, t0: l?.t0 ?? e.t, t1: e.t, text: d.line ?? '', why: d.reason, seqs: [e.seq] });
        break;
      }
      case 'llm.thinking': {
        const l = this.llm.find((x) => x.rid === d.rid);
        if (l) {
          l.thinkingAt ??= e.t;
          l.seqs.push(e.seq);
        }
        break;
      }
      case 'llm.draft_end': {
        const r = this.drafts.find((x) => x.rid === d.rid);
        if (r) {
          r.t1 = e.t;
          r.status = d.status;
          r.why = d.why;
          r.seqs.push(e.seq);
        }
        break;
      }
      case 'llm.delta': {
        const l = this.llm.find((x) => x.rid === d.rid);
        if (l) {
          l.firstToken ??= e.t;
          l.text += d.text ?? '';
          l.chunks.push({ t: e.t, text: d.text ?? '', seq: e.seq });
          l.seqs.push(e.seq);
        }
        break;
      }
      case 'llm.end': {
        const l = this.llm.find((x) => x.rid === d.rid);
        if (l) {
          l.t1 = e.t;
          l.status = d.status;
          l.seqs.push(e.seq);
        }
        break;
      }
      case 'tts.segment_start':
        this.tts.push({ rid: d.rid, seg: d.seg, t0: e.t, text: d.text ?? '', seqs: [e.seq] });
        break;
      case 'tts.segment_end': {
        const s = this.tts.find((x) => x.rid === d.rid && x.seg === d.seg);
        if (s) {
          s.t1 = e.t;
          s.seqs.push(e.seq);
        }
        break;
      }
      case 'bc.say':
        this.backchannels.push({ t0: e.t, text: d.line ?? '', id: d.id ?? '', why: d.why, seqs: [e.seq] });
        break;
      case 'bc.skip':
        this.bcSkips.push({ t: e.t, text: d.reason ?? '', why: d.why ?? '', seq: e.seq });
        break;
      case 'playback.cue': {
        const b = this.backchannels.find((x) => x.id === d.rid);
        if (b) {
          b.t0 = d.at ?? b.t0;
          b.t1 = d.end;
          b.seqs.push(e.seq);
        }
        break;
      }
      case 'playback.start':
        this.playback.push({ t0: d.at ?? e.t, text: d.rid ?? '', rid: d.rid ?? '', seqs: [e.seq] });
        break;
      case 'playback.segment':
        this.playbackSegments.push({ t: d.at ?? e.t, text: d.text ?? '', seq: e.seq });
        break;
      case 'playback.gain':
        this.closeHold(d.at ?? e.t, e.seq);
        if (typeof d.gain === 'number' && d.gain < 1)
          this.holds.push({ t0: d.at ?? e.t, text: `ducked ${(20 * Math.log10(d.gain)).toFixed(0)} dB while the user spoke`, seqs: [e.seq] });
        break;
      case 'playback.paused':
        this.closeHold(d.at ?? e.t, e.seq);
        this.holds.push({ t0: d.at ?? e.t, text: 'cut off while the user spoke', seqs: [e.seq] });
        // Held audio is sent again, at its new times, on resume.
        this.clipAgentAudio(d.at ?? e.t);
        break;
      case 'playback.resumed':
        this.closeHold(d.at ?? e.t, e.seq);
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
          p.seqs.push(e.seq);
          this.closeHold(p.t1!, e.seq);
          // Audio already scheduled past the cut never played: don't draw it.
          if (p.interrupted) this.clipAgentAudio(p.t1!);
        }
        break;
      }
    }
  }
}
