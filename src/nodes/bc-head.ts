/**
 * Backchannel head: a small MLP on the turn model's own per-frame outputs that
 * scores "the agent could say mm-hmm here" while the user talks. It reads what
 * the turn head already computes (end-of-turn, take-the-floor and silence scores
 * and the 18-way frame classes, for user and agent), so it costs nothing on the
 * GPU and runs on recorded frames: a replay can swap or retune it.
 *
 * Inputs per frame, user then agent for each: eot, int, silent, fine[18]; then
 * the mean of the previous 4 frames' inputs. Weights: bc-head.json, trained in
 * research/backchannel-timing (TurnBench dev).
 */

export interface BcWeights {
  inputs: number;
  hidden: number;
  mu: number[];
  sd: number[];
  w1: number[];
  b1: number[];
  w2: number[];
  b2: number[];
}

export interface BcFrame {
  eot: number[];
  int: number[];
  silent: number[];
  fine: number[][];
}

const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));

export class BcHead {
  private prev: number[][] = [];

  constructor(private w: BcWeights) {}

  /** The frame's own inputs: user first, then agent (the listener). */
  static inputs(f: BcFrame): number[] {
    return [f.eot[0], f.eot[1], f.int[0], f.int[1], f.silent[0], f.silent[1], ...f.fine[0], ...f.fine[1]];
  }

  step(f: BcFrame): number {
    const x = BcHead.inputs(f);
    const n = x.length;
    const mean = new Array<number>(n).fill(0);
    for (const p of this.prev) for (let i = 0; i < n; i++) mean[i] += p[i] / this.prev.length;
    this.prev.push(x);
    if (this.prev.length > 4) this.prev.shift();
    const v = [...x, ...mean];
    const { w } = this;
    let out = w.b2[0];
    for (let h = 0; h < w.hidden; h++) {
      let a = w.b1[h];
      const row = h * w.inputs;
      for (let i = 0; i < w.inputs; i++) a += w.w1[row + i] * ((v[i] - w.mu[i]) / w.sd[i]);
      out += w.w2[h] * gelu(a);
    }
    return 1 / (1 + Math.exp(-out));
  }
}
