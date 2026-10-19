import type { AudioMsg } from '../core/types';

export function f32FromBytes(bin: Uint8Array): Float32Array {
  const n = Math.floor(bin.byteLength / 4);
  return new Float32Array(bin.buffer.slice(bin.byteOffset, bin.byteOffset + n * 4));
}

export function bytesFromF32(samples: Float32Array): Uint8Array {
  return new Uint8Array(samples.buffer.slice(samples.byteOffset, samples.byteOffset + samples.byteLength));
}

export function int16BytesFromF32(samples: Float32Array): Uint8Array {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return new Uint8Array(out.buffer);
}

/** Re-chunks a contiguous audio stream into fixed-size frames, tracking the sample range of each. */
export class Framer {
  private buf: Float32Array;
  private fill = 0;
  private bufStart = -1;
  private track = '';

  constructor(readonly frameSamples: number) {
    this.buf = new Float32Array(frameSamples);
  }

  push(msg: AudioMsg, onFrame: (frame: Float32Array, track: string, start: number, end: number) => void): void {
    if (this.track !== msg.track || (this.bufStart >= 0 && this.bufStart + this.fill !== msg.start)) {
      // New or discontinuous stream: drop the partial frame.
      this.fill = 0;
      this.track = msg.track;
    }
    let off = 0;
    while (off < msg.samples.length) {
      if (this.fill === 0) this.bufStart = msg.start + off;
      const take = Math.min(this.frameSamples - this.fill, msg.samples.length - off);
      this.buf.set(msg.samples.subarray(off, off + take), this.fill);
      this.fill += take;
      off += take;
      if (this.fill === this.frameSamples) {
        const frame = this.buf.slice();
        this.fill = 0;
        onFrame(frame, this.track, this.bufStart, this.bufStart + this.frameSamples);
        this.bufStart += this.frameSamples;
      }
    }
  }
}

/**
 * Audio the speaker played, placed on the session clock at one sample rate:
 * a ring buffer indexed by absolute sample number (session ms × rate / 1000).
 * Overlapping chunks add up; positions never written read as silence.
 */
export class PlayedTrack {
  private ring: Float32Array;
  private hi = 0;

  constructor(
    readonly rate: number,
    seconds = 60,
    minSamples = 0,
  ) {
    this.ring = new Float32Array(Math.max(minSamples, Math.round(seconds * rate)));
  }

  /** Sample `k`, or 0 outside what is held. */
  get(k: number): number {
    if (k < 0 || k >= this.hi || k < this.hi - this.ring.length) return 0;
    return this.ring[k % this.ring.length];
  }

  /** Copy samples [k0, k0 + out.length) into `out`. */
  read(k0: number, out: Float32Array): Float32Array {
    for (let i = 0; i < out.length; i++) out[i] = this.get(k0 + i);
    return out;
  }

  /** Ensure ring positions up to `k` exist (zeroed). */
  private extend(k: number) {
    const n = this.ring.length;
    if (k - this.hi >= n) {
      this.ring.fill(0);
      this.hi = k - n;
    }
    for (; this.hi < k; this.hi++) this.ring[this.hi % n] = 0;
  }

  /** Add played audio starting at session time `atMs` (linearly resampled to `rate`). */
  add(atMs: number, samples: Float32Array, sampleRate: number) {
    const k0 = Math.round((atMs * this.rate) / 1000);
    const n = Math.floor((samples.length * this.rate) / sampleRate);
    this.extend(k0 + n);
    const step = sampleRate / this.rate;
    for (let i = 0; i < n; i++) {
      const pos = i * step;
      const j = Math.floor(pos);
      const f = pos - j;
      const v = samples[j] * (1 - f) + (j + 1 < samples.length ? samples[j + 1] : samples[j]) * f;
      const k = k0 + i;
      if (k >= this.hi - this.ring.length) this.ring[k % this.ring.length] += v;
    }
  }

  /** Playback was cut at `atMs`: nothing after that will be heard. */
  truncate(atMs: number) {
    const k0 = Math.round((atMs * this.rate) / 1000);
    for (let k = Math.max(k0, this.hi - this.ring.length); k < this.hi; k++) this.ring[k % this.ring.length] = 0;
  }
}
