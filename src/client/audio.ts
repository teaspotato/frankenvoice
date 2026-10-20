// Page audio: the mic capture the server asks for, and playback of what the server's speaker
// schedules (and of a replay run in the page). Nothing else runs in the page.

import captureWorkletUrl from './capture-worklet.js?url';
import type { LiveDriver } from '../core/world';
import type { Json } from '../core/types';
import { toJson } from '../core/json';
import { level as loudness, soundEnd, type Span } from '../nodes/cutpoint';
import { sustain } from '../nodes/tts/tail';

/** How the server's speaker cuts the reply off; the page gate cuts the same way. */
export interface CutParams {
  cutoff: 'syllable' | 'hard' | string;
  maxFinishMs: number;
  tailMs: number;
  releaseMs: number;
}

export interface MicRequest {
  sampleRate: number;
  frameSamples: number;
  echoCancellation: boolean;
  noiseSuppression: boolean;
  autoGainControl: boolean;
}

/**
 * Capture the mic with the browser's own processing as requested (echo cancellation works here
 * because the agent plays in this page). `onFrame` gets each frame of `frameSamples` at the target rate.
 */
export function captureMic(
  r: MicRequest,
  on: { ready(device: Json): void; frame(samples: Float32Array, frame: number, ctxTime: number): void; error(e: unknown): void },
) {
  let closed = false;
  let stream: MediaStream | undefined;
  // Capture at the target rate so the browser resamples with a proper filter (the worklet's own
  // resampler has none). Browsers that can't feed a mic into a context at another rate get the default.
  let ctx = new AudioContext({ sampleRate: r.sampleRate });
  (async () => {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: r.echoCancellation, noiseSuppression: r.noiseSuppression, autoGainControl: r.autoGainControl },
    });
    if (closed) return stream.getTracks().forEach((t) => t.stop());
    let src: MediaStreamAudioSourceNode;
    try {
      src = ctx.createMediaStreamSource(stream);
    } catch {
      void ctx.close();
      ctx = new AudioContext();
      src = ctx.createMediaStreamSource(stream);
    }
    await ctx.audioWorklet.addModule(captureWorkletUrl);
    const node = new AudioWorkletNode(ctx, 'frankenvoice-capture', { processorOptions: { targetRate: r.sampleRate, frameSamples: r.frameSamples } });
    const sinkGain = ctx.createGain();
    sinkGain.gain.value = 0;
    const track = stream.getAudioTracks()[0];
    on.ready(toJson({ label: track.label, contextSampleRate: ctx.sampleRate, settings: track.getSettings() }));
    node.port.onmessage = (e: MessageEvent<{ samples: Float32Array; frame: number; ctxTime: number }>) => {
      if (!closed) on.frame(e.data.samples, e.data.frame, e.data.ctxTime);
    };
    src.connect(node).connect(sinkGain).connect(ctx.destination);
  })().catch((e) => !closed && on.error(e));
  return () => {
    closed = true;
    stream?.getTracks().forEach((t) => t.stop());
    void ctx.close();
  };
}

/** Output audio contexts, so a paused replay can freeze what is already scheduled. */
const outputContexts = new Set<AudioContext>();

/** While set, outputs drop what they are asked to play: a replay is skipping ahead. */
let outputSuppressed = false;
const playingSources = new Set<AudioBufferSourceNode>();
/** Outputs holding audio scheduled during a skip; each plays the part still ahead once the skip ends. */
const heldOutputs = new Set<() => void>();

/**
 * Silence outputs while a replay fast-forwards, cutting off whatever is playing.
 * Audio scheduled to play at a session time is held instead of dropped, so when
 * the skip lands in the middle of the agent speaking, the rest of it still plays.
 */
export function setOutputSuppressed(on: boolean) {
  outputSuppressed = on;
  if (!on) {
    for (const release of heldOutputs) release();
    heldOutputs.clear();
    return;
  }
  for (const s of playingSources) s.stop();
  playingSources.clear();
}

/** Muted outputs: the agent's voice and the recorded mic during replay. */
const muted = { agent: false, mic: false };
const gains = { agent: new Set<GainNode>(), mic: new Set<GainNode>() };

export function setOutputMuted(which: 'agent' | 'mic', on: boolean) {
  muted[which] = on;
  for (const g of gains[which]) g.gain.value = on ? 0 : 1;
}

/** An output context whose audio goes through a gain that follows the mute switch for `which`. */
function outputContext(which: 'agent' | 'mic', sampleRate?: number) {
  let ctx: AudioContext;
  try {
    ctx = new AudioContext({ latencyHint: 'interactive', sampleRate });
  } catch {
    ctx = new AudioContext({ latencyHint: 'interactive' });
  }
  const gain = ctx.createGain();
  gain.gain.value = muted[which] ? 0 : 1;
  gain.connect(ctx.destination);
  outputContexts.add(ctx);
  gains[which].add(gain);
  return {
    ctx,
    out: gain,
    close() {
      outputContexts.delete(ctx);
      gains[which].delete(gain);
      void ctx.close();
    },
  };
}

/**
 * Plays a stream of PCM chunks. The context runs at the audio's own rate, so no chunk is resampled on
 * its own (that leaves a click at every chunk edge), and a chunk that continues the previous one starts
 * exactly where it ended on the audio clock instead of at a time converted from the session clock,
 * which jitters by a millisecond or so and left tiny gaps and overlaps: a steady crackle.
 */
function pcmOutput(which: 'agent' | 'mic', clock?: () => number) {
  let o: ReturnType<typeof outputContext> | null = null;
  let next = 0; // audio-clock end of the last chunk
  let nextAt = NaN; // and its session time, when chunks carry one
  const playing = new Set<AudioBufferSourceNode>();
  // Ducking, ahead of the mute switch; kept across a context change.
  let level = 1;
  let duck: GainNode | null = null;
  /** The sources' fade-out at a cut; each cut gets a fresh one, so its ramp never reaches later audio. */
  let fade: GainNode | null = null;
  /** When each source plays: context time, and session time when it was given one. */
  const when = new WeakMap<AudioBufferSourceNode, { t: number; at?: number; dur: number }>();
  /** Timed chunks scheduled lately (session time), for the page gate to find where the sound ends. */
  let recent: { at: number; samples: Float32Array; sampleRate: number }[] = [];
  /** Timed chunks that arrived while a replay was skipping ahead (needs `clock`). */
  let held: { samples: Float32Array<ArrayBuffer>; sampleRate: number; at: number }[] = [];
  const release = () => {
    const chunks = held;
    held = [];
    const now = clock!();
    for (const c of chunks) {
      const cut = Math.max(0, Math.round(((now - c.at) / 1000) * c.sampleRate));
      if (cut >= c.samples.length) continue;
      play(c.samples.subarray(cut), c.sampleRate, Math.max(c.at, now), now);
    }
  };
  /**
   * Stop everything: now, or at session time `at` (ms, with the session clock `now`) after fading
   * out over `releaseMs`, or with `tail` taking over at `at` (sample-exact, in the same context).
   */
  const stop = (at?: number, now?: number, releaseMs = 0, tail?: { samples: Float32Array<ArrayBuffer>; sampleRate: number; stop: number }) => {
    // A tail this context can't play (another sample rate): fade out where the sound ends instead.
    if (tail && o && o.ctx.sampleRate !== tail.sampleRate) [at, tail] = [tail.stop, undefined];
    // While a replay skips ahead nothing is heard: no fade, no tail.
    if (o && duck && fade && at !== undefined && now !== undefined && !outputSuppressed) {
      // The context time of `at`, from the source playing then, so the cut lands on its samples.
      let t = o.ctx.currentTime + Math.max(0, (at - now) / 1000);
      for (const src of playing) {
        const w = when.get(src);
        if (w?.at !== undefined && w.at <= at && at < w.at + w.dur * 1000) t = Math.max(o.ctx.currentTime, w.t + (at - w.at) / 1000);
      }
      if (tail) {
        for (const src of playing) src.stop(t);
        const buf = o.ctx.createBuffer(1, tail.samples.length, tail.sampleRate);
        buf.copyToChannel(tail.samples, 0);
        const src = o.ctx.createBufferSource();
        src.buffer = buf;
        // Past the fade node: a later cut doesn't touch it.
        src.connect(duck);
        src.onended = () => playingSources.delete(src);
        playingSources.add(src);
        src.start(t);
      } else {
        const r = Math.max(0.001, releaseMs / 1000);
        fade.gain.setValueAtTime(1, t);
        fade.gain.linearRampToValueAtTime(0, t + r);
        for (const src of playing) src.stop(t + r);
        fade = o.ctx.createGain();
        fade.connect(duck);
      }
    } else for (const src of playing) src.stop();
    playing.clear();
    nextAt = NaN;
    held = [];
    recent = [];
  };
  /**
   * Play `samples`; at session time `at` (ms, with the session clock `now`), or else right after the previous chunk.
   * Returns, for a timed chunk that starts a run of playback (nothing was playing), how late it starts (ms).
   */
  function play(samples: Float32Array<ArrayBuffer>, sampleRate: number, at?: number, now?: number): { lateMs: number } | undefined {
    if (!samples.length) return;
    if (outputSuppressed) {
      if (at === undefined || !clock) return;
      held.push({ samples, sampleRate, at });
      heldOutputs.add(release);
      return;
    }
    if (!o || o.ctx.sampleRate !== sampleRate) {
      stop();
      o?.close();
      o = outputContext(which, sampleRate);
      duck = o.ctx.createGain();
      duck.gain.value = level;
      duck.connect(o.out);
      fade = o.ctx.createGain();
      fade.connect(duck);
    }
    const { ctx } = o;
    const out = fade!;
    const buf = ctx.createBuffer(1, samples.length, sampleRate);
    buf.copyToChannel(samples, 0);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(out);
    src.onended = () => {
      playing.delete(src);
      playingSources.delete(src);
    };
    /** Nothing scheduled or playing: this chunk starts a run of playback. */
    const idle = playing.size === 0;
    playing.add(src);
    playingSources.add(src);
    const live = next >= ctx.currentTime;
    let t: number;
    if (at === undefined) t = live && next <= ctx.currentTime + 0.5 ? next : ctx.currentTime + 0.08;
    else if (live && Math.abs(at - nextAt) < 2) t = next;
    else t = Math.max(ctx.currentTime, ctx.currentTime + (at - now!) / 1000);
    src.start(t);
    when.set(src, { t, at, dur: buf.duration });
    if (at !== undefined) {
      recent = recent.filter((c) => c.sampleRate === sampleRate && c.at + (c.samples.length / sampleRate) * 1000 > now! - 2000);
      recent.push({ at, samples, sampleRate });
    }
    next = t + buf.duration;
    nextAt = at === undefined ? NaN : at + buf.duration * 1000;
    return at !== undefined && idle ? { lateMs: Math.max(0, now! - at) } : undefined;
  }
  /** The scheduled audio over [t0, t1) of session time, silence where there is none. */
  const spanOf = (t0: number, t1: number): Span | null => {
    const rate = recent[0]?.sampleRate;
    if (!rate) return null;
    const samples = new Float32Array(Math.max(0, Math.round(((t1 - t0) / 1000) * rate)));
    for (const c of recent) {
      const o = Math.round(((c.at - t0) / 1000) * rate);
      for (let i = Math.max(0, -o); i < c.samples.length && o + i < samples.length; i++) samples[o + i] = c.samples[i];
    }
    return { at: t0, sampleRate: rate, samples };
  };
  /**
   * Cut off at session time `now` as the server's speaker would (see its cutPoint and flushAt):
   * finish the sound playing, then sustain its end over a tail or fade over the release.
   * Returns where the sound stops (what was heard ends there) and when it is silent.
   */
  function cut(now: number, p: CutParams): { stop: number; silent: number } {
    const a = spanOf(now - 1500, now + p.maxFinishMs + 50);
    if (!a || loudness(a, now, now + 10) < 1e-3) {
      stop();
      return { stop: now, silent: now };
    }
    const end = p.cutoff === 'syllable' ? soundEnd(a, now, p.maxFinishMs) : now;
    const release = p.cutoff === 'hard' ? 0 : Math.max(0, p.releaseMs);
    const t0 = Math.max(now + 5, end - 90);
    if (p.cutoff === 'syllable' && p.tailMs > 0 && end - t0 >= 65) {
      const i0 = Math.round(((t0 - a.at) / 1000) * a.sampleRate);
      const i1 = Math.round(((end - a.at) / 1000) * a.sampleRate);
      const samples = sustain(a.samples.subarray(i0, i1), a.sampleRate, p.tailMs);
      stop(t0, now, release, { samples, sampleRate: a.sampleRate, stop: end });
      return { stop: end, silent: t0 + (samples.length / a.sampleRate) * 1000 };
    }
    stop(end, now, release);
    return { stop: end, silent: end + release };
  }
  return {
    play,
    stop,
    cut,
    /** Fade to `g` (linear) over a few tens of ms, so the step doesn't click. */
    setGain(g: number) {
      level = g;
      if (o && duck) duck.gain.setTargetAtTime(g, o.ctx.currentTime, 0.015);
    },
    close() {
      stop();
      o?.close();
      o = null;
      duck = null;
      fade = null;
    },
  };
}

/** Plays the recorded mic while a replay runs in real time, so both sides of the conversation are heard. */
export function micMonitor() {
  const out = pcmOutput('mic');
  return { play: (samples: Float32Array<ArrayBuffer>, sampleRate: number) => out.play(samples, sampleRate), close: out.close };
}

/** Pause or resume every playing output. */
export function setOutputPaused(paused: boolean) {
  for (const ctx of outputContexts) void (paused ? ctx.suspend() : ctx.resume());
}

/** The agent's two outputs: the reply, and backchannels on their own so they neither wait for nor shift it. */
export function speakerOutputs(clock?: () => number) {
  const out = pcmOutput('agent', clock);
  const cues = pcmOutput('agent', clock);
  /** Stopped by the page gate: the reply's audio sent meanwhile is dropped until the server's flush says it knows. */
  let gated = false;
  return {
    /** Run one speaker command (see the speaker node), with the session clock `now` when it is run. */
    command(d: { cmd: string; at?: number; stop?: number; sampleRate?: number; gain?: number; releaseMs?: number }, bin: Uint8Array | undefined, now: number) {
      if (d.cmd === 'cue' && bin) return cues.play(new Float32Array(bin.slice().buffer), d.sampleRate!, d.at!, now);
      if (gated && (d.cmd === 'play' || d.cmd === 'flush')) return void (gated = d.cmd === 'play');
      if (d.cmd === 'flush') {
        const tail = bin && d.sampleRate ? { samples: new Float32Array(bin.slice().buffer), sampleRate: d.sampleRate, stop: d.stop ?? d.at! } : undefined;
        return void (d.at === undefined ? out.stop() : out.stop(d.at, now, d.releaseMs, tail));
      }
      if (d.cmd === 'gain') return void out.setGain(d.gain ?? 1);
      if (d.cmd === 'play' && bin) return out.play(new Float32Array(bin.slice().buffer), d.sampleRate!, d.at!, now);
    },
    /** The page gate heard the user at session time `now`: cut the reply off as the server would. */
    gateCut(now: number, p: CutParams) {
      gated = true;
      return out.cut(now, p);
    },
    close() {
      out.close();
      cues.close();
    },
  };
}

/** The speaker for a replay run in the page: plays the recorded agent audio on the replay's clock. */
export const replaySpeaker: LiveDriver = (_req, sink, env) => {
  const o = speakerOutputs(env.now);
  setTimeout(() => sink.recv({ type: 'ready' }), 0);
  return {
    send: (data, bin) => void o.command(data as { cmd: string }, bin, env.now()),
    close: () => o.close(),
  };
};
