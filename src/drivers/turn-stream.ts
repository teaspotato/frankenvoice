// Streaming ssl_turn model: causal FastConformer encoder + cross-speaker turn head, run with
// onnxruntime-web. Ported from the turn-playground Space (ssl_turn/web/turn_stream.js).
//
//   const ts = await TurnStream.create(ort, { encoder, head, frames: 2, ep: 'webgpu' });
//   for each 80 ms block: const outs = await ts.push(user, agent)   // two Float32Array(1280), 16 kHz
//   -> [{ t, eot: [c0, c1], int: [c0, c1], silent: [c0, c1], post, fine: [c0, c1] }, ...]
//
// `frames` (K) must match the encoder file: the encoder runs once per K blocks and returns K
// frames. Frame t's newest audio is block t - 1, so with K > 1 the earlier frames of a chunk wait
// for the last block of the chunk. The first push also returns frame 0 (its feature is zeros).

import type * as Ort from 'onnxruntime-web';

export const FRAME = 1280; // 80 ms at 16 kHz
const LAYERS = 17;
const CACHE = 70;
const D = 512;
const CONV = 8;
const FEAT = 1024;
const HEAD_SHAPE = [6, 2, 4, 249, 48]; // [2 x head layers, channels, heads, W - 1, head dim]
const HEAD_W = 250;
const STATE = ['cache_len', ...[...Array(LAYERS).keys()].map((i) => `cache_ch_${i}`), ...[...Array(LAYERS).keys()].map((i) => `cache_time_${i}`)];
const win = (K: number) => 3073 + 1280 * (K - 1);

export interface TurnFrame {
  t: number;
  eot: [number, number];
  int: [number, number];
  silent: [number, number];
  post: number[];
  /** Per speaker, the head's 18-way frame-class distribution (what the backchannel head reads). */
  fine: [number[], number[]];
}

type OrtModule = typeof Ort;
type Tensor = Ort.Tensor;

export class TurnStream {
  private ort!: OrtModule;
  private K = 1;
  private enc!: Ort.InferenceSession;
  private head!: Ort.InferenceSession;
  private back = 0;
  private buf: Float32Array[] = [];
  private pending = 0;
  private t = 1;
  private state: Record<string, Tensor> = {};
  private kc!: Tensor;
  private vc!: Tensor;
  private started = false;

  /** ep: encoder backend ('webgpu' or 'wasm'). The head (3M params) runs on WASM. */
  static async create(ort: OrtModule, o: { encoder: Uint8Array; head: Uint8Array; frames: number; ep: string }): Promise<TurnStream> {
    const self = new TurnStream();
    self.ort = ort;
    self.K = o.frames;
    const base = { executionProviders: [o.ep], graphOptimizationLevel: 'all' as const };
    // Caches stay on the GPU between steps; only the features are read back.
    self.enc = await ort.InferenceSession.create(
      o.encoder,
      o.ep === 'webgpu' ? { ...base, preferredOutputLocation: Object.fromEntries(STATE.slice(1).map((n) => [`${n}_out`, 'gpu-buffer' as const])) } : base,
    );
    self.head = await ort.InferenceSession.create(o.head, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    self.reset();
    return self;
  }

  reset() {
    const T = this.ort.Tensor;
    for (const s of Object.values(this.state)) if (s.location === 'gpu-buffer') s.dispose();
    this.back = win(this.K) + 1024; // samples kept per channel: the window plus the 1024 it skips
    this.buf = [new Float32Array(this.back), new Float32Array(this.back)];
    this.pending = 0;
    this.t = 1;
    this.state = { cache_len: new T('int64', new BigInt64Array(2), [2]) };
    for (let i = 0; i < LAYERS; i++) {
      this.state[`cache_ch_${i}`] = new T('float32', new Float32Array(2 * CACHE * D), [2, CACHE, D]);
      this.state[`cache_time_${i}`] = new T('float32', new Float32Array(2 * D * CONV), [2, D, CONV]);
    }
    const n = HEAD_SHAPE.reduce((a, b) => a * b);
    this.kc = new T('float32', new Float32Array(n), HEAD_SHAPE);
    this.vc = new T('float32', new Float32Array(n), HEAD_SHAPE);
    this.started = false;
  }

  /** feat: [2, K, 1024] for frames t0 .. t0 + K - 1 */
  private async headStep(feat: Float32Array, t0: number, K: number): Promise<TurnFrame[]> {
    const T = this.ort.Tensor;
    const n = new T('int64', BigInt64Array.from([BigInt(Math.min(t0, HEAD_W - 1))]), [1]);
    const o = await this.head.run({ feat: new T('float32', feat, [2, K, FEAT]), n, kcache: this.kc, vcache: this.vc });
    this.kc = o.kcache_out;
    this.vc = o.vcache_out;
    const [eot, intr, silent, post, fine] = (await Promise.all(['eot', 'int', 'silent', 'post', 'fine'].map((k) => o[k].getData()))) as Float32Array[];
    const out: TurnFrame[] = [];
    for (let j = 0; j < K; j++) {
      out.push({
        t: t0 + j,
        eot: [eot[2 * j], eot[2 * j + 1]],
        int: [intr[2 * j], intr[2 * j + 1]],
        silent: [silent[2 * j], silent[2 * j + 1]],
        post: [...post.subarray(16 * j, 16 * j + 16)],
        fine: [[...fine.subarray(36 * j, 36 * j + 18)], [...fine.subarray(36 * j + 18, 36 * j + 36)]],
      });
    }
    return out;
  }

  /** Feed the next 80 ms block per channel; returns the frames that became available (maybe none). */
  async push(ch0: Float32Array, ch1: Float32Array): Promise<TurnFrame[]> {
    const T = this.ort.Tensor;
    let out: TurnFrame[] = [];
    if (!this.started) {
      this.started = true;
      out = await this.headStep(new Float32Array(2 * FEAT), 0, 1);
    }
    for (const [c, x] of [ch0, ch1].entries()) {
      this.buf[c].copyWithin(0, FRAME);
      this.buf[c].set(x, this.back - FRAME);
    }
    if (++this.pending < this.K) return out;
    this.pending = 0;
    const K = this.K;
    const W = win(K);
    const audio = new Float32Array(2 * W);
    audio.set(this.buf[0].subarray(0, W), 0);
    audio.set(this.buf[1].subarray(0, W), W);
    const o = await this.enc.run({
      audio: new T('float32', audio, [2, W]),
      t: new T('int64', BigInt64Array.from([BigInt(this.t)]), [1]),
      ...this.state,
    });
    for (const name of STATE) {
      const prev = this.state[name];
      if (prev.location === 'gpu-buffer') prev.dispose();
      this.state[name] = o[`${name}_out`];
    }
    const feat = (await o.feat.getData(true)) as Float32Array; // [2, K, 1024]; true releases a GPU buffer
    out = out.concat(await this.headStep(feat, this.t, K));
    this.t += K;
    return out;
  }
}
