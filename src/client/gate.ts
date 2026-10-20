// The page gate (turn.duplex pageGate): the page runs its own FireRedVAD on the mic so that, while
// the server has armed it, playback stops the moment the user starts talking instead of a network
// round trip later. Same model and segmenter defaults as the server's vad.firered, onsets only.

import type { FireRedStream } from '../../server/drivers/firered/stream';

const HOP = 160;
const SMOOTH = 5;
const THRESHOLD = 0.5;
const NEG_THRESHOLD = 0.35;
/** Frames of 10 ms: 96 ms of speech starts it, 320 ms of silence ends it. */
const MIN_SPEECH = 10;
const REDEMPTION = 32;

export class PageGate {
  private loading: Promise<() => FireRedStream> | null = null;
  private make: (() => FireRedStream) | null = null;
  private stream: FireRedStream | null = null;
  private chain: Promise<void> = Promise.resolve();
  /** Mic sample the stream started at, frames scored, and the segmenter's state. */
  private base = 0;
  private frame = 0;
  private recent: number[] = [];
  private speaking = false;
  private start = -1;
  private speech = 0;
  private silence = 0;
  private epoch = 0;

  /** `onset(pos)`: the user started talking at mic sample `pos`. */
  constructor(private onset: (pos: number) => void) {}

  get ready() {
    return !!this.make;
  }

  /** Fetch onnxruntime and the model (once). */
  load(): void {
    this.loading ??= import('./gate-vad')
      .then((m) => m.loadFireRed())
      .then((make) => (this.make = make))
      .catch((e) => {
        console.warn('page gate: FireRedVAD did not load', e);
        this.loading = null;
        throw e;
      });
    this.loading.catch(() => {});
  }

  /** A new mic stream: forget the old one. */
  reset(): void {
    this.epoch++;
    this.stream = null;
    this.chain = Promise.resolve();
  }

  /** Mic samples (16 kHz) starting at mic sample `pos`. */
  push(samples: Float32Array, pos: number): void {
    if (!this.make) return;
    if (!this.stream) {
      this.stream = this.make();
      this.base = pos;
      this.frame = 0;
      this.recent = [];
      this.speaking = false;
      this.start = -1;
      this.speech = this.silence = 0;
    }
    const stream = this.stream;
    const epoch = this.epoch;
    this.chain = this.chain
      .then(() => stream.push(samples))
      .then((ps) => {
        if (epoch === this.epoch) for (const p of ps) this.score(p);
      })
      .catch(() => {});
  }

  private score(p: number) {
    const at = this.base + this.frame++ * HOP;
    this.recent.push(p);
    if (this.recent.length > SMOOTH) this.recent.shift();
    const s = this.recent.reduce((a, b) => a + b, 0) / this.recent.length;
    if (!this.speaking) {
      if (s >= THRESHOLD) {
        if (this.start < 0) this.start = at;
        if (++this.speech >= MIN_SPEECH) {
          this.speaking = true;
          this.silence = 0;
          this.onset(this.start);
        }
      } else if (s < NEG_THRESHOLD) {
        this.start = -1;
        this.speech = 0;
      }
    } else if (s < NEG_THRESHOLD) {
      if (++this.silence >= REDEMPTION) {
        this.speaking = false;
        this.start = -1;
        this.speech = this.silence = 0;
      }
    } else if (s >= THRESHOLD) this.silence = 0;
  }
}
