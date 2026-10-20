/**
 * A tail for canned lines: TTS tends to stop a short line like "mm-hmm" dead, so the last sound is
 * sustained (WSOLA time-stretching, which keeps the pitch) and faded out.
 */

/** Resample by `factor` (> 1 = higher and shorter), linear interpolation. */
function resample(x: Float32Array, factor: number): Float32Array {
  const n = Math.max(1, Math.floor(x.length / factor));
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = i * factor;
    const j = Math.floor(p);
    const f = p - j;
    y[i] = (x[j] ?? 0) * (1 - f) + (x[j + 1] ?? x[j] ?? 0) * f;
  }
  return y;
}

/**
 * WSOLA: stretch `x` to `outLen` samples without changing its pitch. Hann frames at a fixed output
 * hop; each input frame is taken where it best continues the previous one (within ±tolerance).
 */
export function stretch(x: Float32Array, outLen: number, rate: number): Float32Array {
  const N = Math.round(rate * 0.03); // 30 ms frames
  const hop = N >> 1;
  const tol = Math.round(rate * 0.008);
  const y = new Float32Array(outLen + N);
  if (x.length < N + 2 * tol + hop) {
    // Too short to stretch: plain resampling.
    y.set(resample(x, x.length / Math.max(1, outLen)).subarray(0, outLen));
    return y.subarray(0, outLen);
  }
  const win = new Float32Array(N);
  for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);
  const ratio = (x.length - N) / Math.max(1, outLen - N);
  let prev = 0; // input position of the previous frame
  for (let out = 0; out + N <= y.length; out += hop) {
    const target = Math.min(x.length - N, Math.round(out * ratio));
    let best = target;
    if (out > 0) {
      // The natural continuation of the previous frame is at prev + hop; find the frame around
      // `target` that looks most like it.
      const nat = prev + hop;
      let bestC = -Infinity;
      for (let k = Math.max(0, target - tol); k <= Math.min(x.length - N, target + tol); k++) {
        let c = 0;
        for (let i = 0; i < N; i += 2) c += (x[k + i] ?? 0) * (x[nat + i] ?? 0);
        if (c > bestC) [bestC, best] = [c, k];
      }
    }
    // The first frame starts at full level instead of fading in.
    for (let i = 0; i < N; i++) y[out + i] += x[best + i] * (out === 0 && i < hop ? 1 : win[i]);
    prev = best;
  }
  return y.subarray(0, outLen);
}

/** `seg` played as is and then sustained for `tailMs` more, fading to silence over the sustain. */
export function sustain(seg: Float32Array, rate: number, tailMs: number): Float32Array<ArrayBuffer> {
  const tail = Math.round((tailMs / 1000) * rate);
  const ext = stretch(seg, seg.length + tail, rate);
  const out = new Float32Array(ext.length);
  for (let i = 0; i < ext.length; i++) out[i] = ext[i] * (i < seg.length ? 1 : 0.5 + 0.5 * Math.cos((Math.PI * (i - seg.length)) / tail));
  return out;
}

export function addTail(y: Float32Array, rate: number, tailMs: number): Float32Array<ArrayBuffer> {
  const tail = Math.round((tailMs / 1000) * rate);
  if (tail > 0 && y.length > rate * 0.08) {
    // Sustain the last 80 ms over the tail, fading from where the line ends to silence.
    const seg = Math.round(rate * 0.08);
    const last = y.subarray(y.length - seg);
    const ext = stretch(last, seg + tail, rate);
    const out = new Float32Array(y.length - seg + ext.length);
    out.set(y.subarray(0, y.length - seg));
    for (let i = 0; i < ext.length; i++) out[y.length - seg + i] = ext[i] * (i < seg ? 1 : 0.5 + 0.5 * Math.cos((Math.PI * (i - seg)) / tail));
    y = out;
  }
  return Float32Array.from(y);
}
