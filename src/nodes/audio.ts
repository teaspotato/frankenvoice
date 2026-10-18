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
