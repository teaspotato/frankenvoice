import type { BinRef, MediaRef } from './types';

export interface TrackInfo {
  kind: 'f32' | 'bytes';
  /** Samples per second, for f32 media tracks. */
  sampleRate?: number;
  /** Total length: samples for f32, bytes for bytes. */
  length: number;
}

interface Track extends TrackInfo {
  chunks: Uint8Array[];
  byteLength: number;
  merged?: Uint8Array;
}

/**
 * Append-only binary storage for a session: media tracks (float32 PCM addressed
 * by sample index) and byte tracks (raw IO payloads addressed by byte offset).
 */
export class TrackStore {
  private tracks = new Map<string, Track>();

  appendF32(name: string, samples: Float32Array, sampleRate: number): MediaRef {
    const tr = this.get(name, 'f32', sampleRate);
    const start = tr.length;
    this.push(tr, new Uint8Array(samples.buffer.slice(samples.byteOffset, samples.byteOffset + samples.byteLength)));
    tr.length += samples.length;
    return { track: name, start, end: tr.length };
  }

  appendBytes(name: string, bytes: Uint8Array): BinRef {
    const tr = this.get(name, 'bytes');
    const offset = tr.byteLength;
    this.push(tr, bytes.slice());
    tr.length = tr.byteLength;
    return { track: name, offset, length: bytes.byteLength };
  }

  readBytes(ref: BinRef): Uint8Array {
    const tr = this.mustGet(ref.track);
    return this.merged(tr).slice(ref.offset, ref.offset + ref.length);
  }

  readF32(ref: MediaRef): Float32Array {
    const tr = this.mustGet(ref.track);
    const bytes = this.merged(tr).slice(ref.start * 4, ref.end * 4);
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  }

  info(): Record<string, TrackInfo> {
    const out: Record<string, TrackInfo> = {};
    for (const [name, tr] of this.tracks) {
      out[name] = { kind: tr.kind, length: tr.length, ...(tr.sampleRate ? { sampleRate: tr.sampleRate } : {}) };
    }
    return out;
  }

  names(): string[] {
    return [...this.tracks.keys()];
  }

  /** Whole track as one byte array (for serialization). */
  bytes(name: string): Uint8Array {
    return this.merged(this.mustGet(name));
  }

  /** Restore a track from serialized bytes. */
  load(name: string, info: TrackInfo, bytes: Uint8Array): void {
    this.tracks.set(name, {
      kind: info.kind,
      sampleRate: info.sampleRate,
      length: info.length,
      chunks: [bytes],
      byteLength: bytes.byteLength,
      merged: bytes,
    });
  }

  private get(name: string, kind: 'f32' | 'bytes', sampleRate?: number): Track {
    let tr = this.tracks.get(name);
    if (!tr) {
      tr = { kind, sampleRate, length: 0, chunks: [], byteLength: 0 };
      this.tracks.set(name, tr);
    } else if (tr.kind !== kind || (sampleRate && tr.sampleRate !== sampleRate)) {
      throw new Error(`track ${name}: kind/sampleRate mismatch`);
    }
    return tr;
  }

  private mustGet(name: string): Track {
    const tr = this.tracks.get(name);
    if (!tr) throw new Error(`unknown track ${name}`);
    return tr;
  }

  private push(tr: Track, bytes: Uint8Array) {
    tr.chunks.push(bytes);
    tr.byteLength += bytes.byteLength;
    tr.merged = undefined;
  }

  private merged(tr: Track): Uint8Array {
    if (!tr.merged) {
      const out = new Uint8Array(tr.byteLength);
      let o = 0;
      for (const c of tr.chunks) {
        out.set(c, o);
        o += c.byteLength;
      }
      tr.chunks = [out];
      tr.merged = out;
    }
    return tr.merged;
  }
}
