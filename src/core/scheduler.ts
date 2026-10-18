// Clocks. The runtime never reads wall time or calls setTimeout directly; it
// goes through a Scheduler, so the same graph can run against real time (live)
// or against a virtual clock driven by a recorded trajectory (replay).

export interface Scheduler {
  /** Milliseconds since session start. */
  now(): number;
  /**
   * Run `fn` at session time `t`. `order` breaks ties between items due at the
   * same time (lower first); replay passes the original event seq here.
   */
  at(t: number, fn: () => void, order?: number): () => void;
}

export class LiveScheduler implements Scheduler {
  private readonly t0: number;
  constructor(private readonly perf: { now(): number } = performance) {
    this.t0 = perf.now();
  }
  now(): number {
    return this.perf.now() - this.t0;
  }
  at(t: number, fn: () => void): () => void {
    const h = setTimeout(fn, Math.max(0, t - this.now()));
    return () => clearTimeout(h);
  }
}

interface Item {
  t: number;
  order: number;
  id: number;
  fn: () => void;
}

const UNORDERED = 2 ** 52;

/**
 * Discrete-event virtual clock. `runFast()` jumps from item to item;
 * `runRealtime(speed)` paces items against wall time (for audible replay).
 */
export class SimScheduler implements Scheduler {
  private heap: Item[] = [];
  private cur = 0;
  private nextId = 0;
  private stopped = false;
  private realtimeTimer: ReturnType<typeof setTimeout> | undefined;

  now(): number {
    return this.cur;
  }

  at(t: number, fn: () => void, order?: number): () => void {
    const item: Item = { t: Math.max(t, this.cur), order: order ?? UNORDERED + this.nextId, id: this.nextId++, fn };
    this.push(item);
    return () => {
      item.fn = noop;
    };
  }

  get pending(): number {
    return this.heap.length;
  }

  /** Process every scheduled item (including ones scheduled while running) until the queue is empty or `until`. */
  runFast(until = Infinity): void {
    while (this.heap.length && !this.stopped) {
      if (this.heap[0].t > until) break;
      const item = this.pop()!;
      this.cur = item.t;
      item.fn();
    }
    if (until !== Infinity && this.cur < until) this.cur = until;
  }

  /** Pace items against wall time. Resolves when the queue drains or stop() is called. */
  runRealtime(speed = 1, perf: { now(): number } = performance): Promise<void> {
    return new Promise((resolve) => {
      const wall0 = perf.now();
      const sim0 = this.cur;
      const tick = () => {
        if (this.stopped) return resolve();
        const target = sim0 + (perf.now() - wall0) * speed;
        while (this.heap.length && this.heap[0].t <= target && !this.stopped) {
          const item = this.pop()!;
          this.cur = item.t;
          item.fn();
        }
        if (!this.heap.length || this.stopped) return resolve();
        this.cur = Math.max(this.cur, Math.min(target, this.heap[0].t));
        const wait = (this.heap[0].t - target) / speed;
        this.realtimeTimer = setTimeout(tick, Math.max(0, Math.min(wait, 50)));
      };
      tick();
    });
  }

  stop(): void {
    this.stopped = true;
    if (this.realtimeTimer) clearTimeout(this.realtimeTimer);
  }

  private less(a: Item, b: Item) {
    return a.t < b.t || (a.t === b.t && (a.order < b.order || (a.order === b.order && a.id < b.id)));
  }
  private push(item: Item) {
    const h = this.heap;
    h.push(item);
    let i = h.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!this.less(h[i], h[p])) break;
      [h[i], h[p]] = [h[p], h[i]];
      i = p;
    }
  }
  private pop(): Item | undefined {
    const h = this.heap;
    if (!h.length) return undefined;
    const top = h[0];
    const last = h.pop()!;
    if (h.length) {
      h[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < h.length && this.less(h[l], h[m])) m = l;
        if (r < h.length && this.less(h[r], h[m])) m = r;
        if (m === i) break;
        [h[i], h[m]] = [h[m], h[i]];
        i = m;
      }
    }
    return top;
  }
}

function noop() {}
