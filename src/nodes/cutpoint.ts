/**
 * Where to stop agent speech that is cut off, and where to pick it up again, so neither falls in
 * the middle of a sound. Works on the audio alone (Aura gives no word timings): energy in 10 ms
 * frames, a "quiet point" being a frame well below the speech level around it.
 */

/** Audio over a stretch of session time: `samples[0]` plays at `at`. */
export interface Span {
  at: number;
  sampleRate: number;
  samples: Float32Array;
}

const FRAME_MS = 10;
const HOP_MS = 5;
/** A frame this far below the surrounding speech level counts as a quiet point. */
const QUIET = 0.18;
/** Below this the audio is silence, and any point is as good as another. */
const SILENT = 1e-3;

const rmsAt = (a: Span, i: number) => {
  const n = Math.round((FRAME_MS / 1000) * a.sampleRate);
  let s = 0;
  for (let k = i; k < i + n; k++) s += (a.samples[k] ?? 0) ** 2;
  return Math.sqrt(s / n);
};
const idx = (a: Span, t: number) => Math.round(((t - a.at) / 1000) * a.sampleRate);
const time = (a: Span, i: number) => a.at + (i / a.sampleRate) * 1000;

/** RMS over [t0, t1). */
export function level(a: Span, t0: number, t1: number): number {
  const i0 = Math.max(0, idx(a, t0));
  const i1 = Math.min(a.samples.length, idx(a, t1));
  let s = 0;
  for (let k = i0; k < i1; k++) s += a.samples[k] ** 2;
  return i1 > i0 ? Math.sqrt(s / (i1 - i0)) : 0;
}

/** The speech level around `t`, which quiet points are measured against. */
const reference = (a: Span, t: number) => level(a, t - 300, t + 100) * 1.4;

/**
 * The first quiet point at or after `at`, within `maxMs`: where the sound playing at `at` ends.
 * Without one, the quietest frame in that window. Silence at `at` stops right there.
 */
export function soundEnd(a: Span, at: number, maxMs: number): number {
  const ref = reference(a, at);
  if (ref < SILENT || level(a, at, at + FRAME_MS) < SILENT) return at;
  const hop = Math.max(1, Math.round((HOP_MS / 1000) * a.sampleRate));
  const end = Math.min(a.samples.length - 1, idx(a, at + maxMs));
  let best = { e: Infinity, i: idx(a, at) };
  for (let i = Math.max(0, idx(a, at)); i <= end; i += hop) {
    const e = rmsAt(a, i);
    if (e < QUIET * ref) return time(a, i);
    if (e < best.e) best = { e, i };
  }
  return Math.min(time(a, best.i), at + maxMs);
}

/**
 * Where the word playing at `at` starts. `estimate` is a rough guess (ASR word timings are in 80 ms
 * steps); the onset is the quietest frame from 80 ms before it to 40 ms after. Without an estimate,
 * the last quiet point before `at` within `maxMs`.
 */
export function wordStart(a: Span, at: number, maxMs: number, estimate?: number): number {
  const ref = reference(a, at);
  if (ref < SILENT) return at;
  const hop = Math.max(1, Math.round((HOP_MS / 1000) * a.sampleRate));
  const lo = Math.max(0, idx(a, Math.max(at - maxMs, estimate !== undefined ? estimate - 80 : -Infinity)));
  const hi = idx(a, estimate !== undefined ? Math.min(at, estimate + 40) : at);
  if (hi <= lo) return Math.max(time(a, lo), Math.min(at, estimate ?? at));
  if (estimate === undefined) {
    for (let i = hi; i >= lo; i -= hop) if (rmsAt(a, i) < QUIET * ref) return time(a, i);
  }
  let best = { e: Infinity, i: hi };
  for (let i = hi; i >= lo; i -= hop) {
    const e = rmsAt(a, i);
    if (e < best.e) best = { e, i };
  }
  return time(a, best.i);
}

/** A copy of `x` fading in over its first `n` samples (raised cosine). */
export function fadeIn(x: Float32Array, n: number): Float32Array {
  const y = Float32Array.from(x);
  for (let i = 0; i < Math.min(n, y.length); i++) y[i] *= 0.5 - 0.5 * Math.cos((Math.PI * i) / n);
  return y;
}
