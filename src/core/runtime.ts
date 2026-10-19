import type { GraphSpec, EdgeSpec } from './graph';
import { runtimeView } from './graph';
import type { IOHandle, IOHandlers, LogLevel, NodeContext, NodeImpl, NodeRegistry, SendOptions, TimerHandle } from './node';
import { resolveParams } from './node';
import type { Scheduler } from './scheduler';
import { Trajectory, TRAJECTORY_FORMAT, TRAJECTORY_FORMAT_VERSION } from './trajectory';
import type { Json, Msg, TrajEvent } from './types';
import { isAudioMsg } from './types';
import { hashJson, jsonEqual, toJson } from './json';
import type { Conn, World } from './world';

export interface RuntimeOptions {
  graph: GraphSpec;
  registry: NodeRegistry;
  scheduler: Scheduler;
  world: World;
  mode: 'live' | 'replay';
  replayOf?: string;
  sessionId?: string;
  env?: Record<string, Json>;
}

interface Delivery {
  node: string;
  port: string;
  msg: Msg;
  cause: number;
}

interface NodeSlot {
  id: string;
  impl: NodeImpl;
  state: Record<string, Json>;
  ioOrdinal: Map<string, number>;
  timerOrdinal: number;
  startSeq: number;
}

export const RUNTIME = '$runtime';

/**
 * Runs a graph. All work happens in "turns": something external happens (an IO
 * message arrives, a timer fires), the runtime stamps it with the scheduler's
 * time, hands it to the node, and then drains every message that results,
 * synchronously and in FIFO order, all at that same timestamp. Given the same
 * external events at the same times, a run produces exactly the same log.
 */
export class Runtime {
  readonly trajectory: Trajectory;
  private slots = new Map<string, NodeSlot>();
  private edgesFrom = new Map<string, EdgeSpec[]>();
  private queue: Delivery[] = [];
  private pendingTurns: (() => void)[] = [];
  private inTurn = false;
  private seq = 0;
  private t = 0;
  private cause: number | undefined;
  private running = false;
  private openConns = new Map<string, { conn: Conn; handle: IOHandle & { closed: boolean } }>();
  private stateListeners = new Set<(node: string, state: Record<string, Json>) => void>();
  /** Live timers not yet fired or cleared, by id: a busy event loop can run them late. */
  private liveTimers = new Map<string, { due: number; fire: () => void }>();

  constructor(private opts: RuntimeOptions) {
    const g = opts.graph;
    this.trajectory = new Trajectory({
      format: TRAJECTORY_FORMAT,
      formatVersion: TRAJECTORY_FORMAT_VERSION,
      id: opts.sessionId ?? newSessionId(),
      createdAt: new Date().toISOString(),
      mode: opts.mode,
      replayOf: opts.replayOf,
      graph: g,
      graphHash: hashJson(runtimeView(g)),
      channelModes: opts.world.channelModes(),
      env: opts.env,
    });
    for (const e of g.edges) {
      const k = `${e.from.node}\u0000${e.from.port}`;
      if (!this.edgesFrom.has(k)) this.edgesFrom.set(k, []);
      this.edgesFrom.get(k)!.push(e);
    }
    opts.world.onDiverge = (d) => this.log(RUNTIME, 'io.diverge', { data: toJson(d) });
  }

  get isRunning(): boolean {
    return this.running;
  }

  nodeState(id: string): Record<string, Json> | undefined {
    return this.slots.get(id)?.state;
  }

  onState(fn: (node: string, state: Record<string, Json>) => void): () => void {
    this.stateListeners.add(fn);
    return () => this.stateListeners.delete(fn);
  }

  start(): void {
    if (this.running) throw new Error('already running');
    this.running = true;
    this.turn(() => {
      this.log(RUNTIME, 'session.start', { data: { graphHash: this.trajectory.meta.graphHash } });
      for (const spec of this.opts.graph.nodes) {
        const def = this.opts.registry[spec.type];
        if (!def) throw new Error(`unknown node type "${spec.type}" (node ${spec.id})`);
        const params = resolveParams(def, spec.params);
        const slot: NodeSlot = { id: spec.id, impl: {}, state: {}, ioOrdinal: new Map(), timerOrdinal: 0, startSeq: -1 };
        this.slots.set(spec.id, slot);
        this.cause = undefined;
        slot.startSeq = this.log(spec.id, 'node.start', { data: { type: spec.type, params } });
        this.cause = slot.startSeq;
        slot.impl = def.create(this.makeContext(slot, params));
      }
      for (const spec of this.opts.graph.nodes) {
        const slot = this.slots.get(spec.id)!;
        this.cause = slot.startSeq;
        this.guard(slot, () => slot.impl.start?.());
      }
    });
  }

  stop(): void {
    if (!this.running) return;
    this.turn(() => {
      const stopSeq = new Map<string, number>();
      for (const slot of this.slots.values()) {
        this.cause = undefined;
        this.cause = this.log(slot.id, 'node.stop');
        stopSeq.set(slot.id, this.cause);
        this.guard(slot, () => slot.impl.stop?.());
      }
      for (const [xid, { handle }] of [...this.openConns.entries()]) {
        this.cause = stopSeq.get(xid.slice(0, xid.indexOf('/')));
        handle.close({ reason: 'session-stop' });
      }
      this.cause = undefined;
      this.log(RUNTIME, 'session.stop');
    });
    this.running = false;
  }

  // ---- turn machinery -------------------------------------------------------

  /** Run `fn` as one turn at the scheduler's current time, then drain. Re-entrant calls queue up. */
  private turn(fn: () => void): void {
    if (this.inTurn) {
      // Arrived while a turn is draining (e.g. a driver delivered synchronously).
      // Stamp it with its own arrival time, so live and replay order it identically.
      const t = this.opts.scheduler.now();
      this.pendingTurns.push(() => {
        this.fireOverdue(t);
        this.t = Math.max(this.t, t);
        this.cause = undefined;
        fn();
      });
      return;
    }
    this.inTurn = true;
    try {
      const now = this.opts.scheduler.now();
      this.fireOverdue(now);
      this.t = Math.max(this.t, now);
      this.cause = undefined;
      fn();
      this.drain();
      while (this.pendingTurns.length) {
        this.cause = undefined;
        this.pendingTurns.shift()!();
        this.drain();
      }
    } finally {
      this.inTurn = false;
      this.cause = undefined;
    }
  }

  /**
   * Live only: run the timers that were due before `now` but haven't fired,
   * each at the time it was due, before whatever arrived at `now`. A replay
   * fires a timer at its time; without this, a live timer running late behind
   * a busy event loop could be cleared by an event that came after its time,
   * and the replay, firing it first, would not match.
   */
  private fireOverdue(now: number): void {
    for (;;) {
      let next: { id: string; due: number; fire: () => void } | undefined;
      for (const [id, x] of this.liveTimers) if (x.due <= now && (!next || x.due < next.due)) next = { id, ...x };
      if (!next) return;
      this.liveTimers.delete(next.id);
      this.t = Math.max(this.t, next.due);
      this.cause = undefined;
      next.fire();
      this.drain();
    }
  }

  private drain(): void {
    while (this.queue.length) {
      const d = this.queue.shift()!;
      const slot = this.slots.get(d.node);
      if (!slot) continue;
      this.cause = d.cause;
      this.guard(slot, () => slot.impl.onInput?.(d.port, d.msg));
    }
  }

  private guard(slot: NodeSlot, fn: () => void): void {
    try {
      fn();
    } catch (e) {
      this.log(slot.id, 'log', { data: { level: 'error', message: String(e), stack: e instanceof Error ? (e.stack ?? null) : null } });
    }
  }

  private log(node: string, kind: TrajEvent['kind'], extra: Partial<TrajEvent> = {}): number {
    const ev: TrajEvent = { seq: this.seq++, t: this.t, node, kind, ...extra };
    if (ev.cause === undefined && this.cause !== undefined) ev.cause = [this.cause];
    this.trajectory.append(ev);
    return ev.seq;
  }

  // ---- node context -----------------------------------------------------------

  private makeContext(slot: NodeSlot, params: Record<string, Json>): NodeContext {
    const rt = this;
    const id = slot.id;
    return {
      id,
      params,
      now: () => rt.t,
      get cause() {
        return rt.cause;
      },
      emit(port, msg, cause) {
        let data: Json;
        let media;
        if (isAudioMsg(msg)) {
          const { samples: _s, track, start, end, ...rest } = msg;
          data = toJson(rest);
          media = { track, start, end };
        } else {
          data = toJson(msg);
        }
        const seq = rt.log(id, 'msg', { port, data, media, ...(cause ? { cause } : {}) });
        for (const e of rt.edgesFrom.get(`${id}\u0000${port}`) ?? []) {
          rt.queue.push({ node: e.to.node, port: e.to.port, msg, cause: seq });
        }
        return seq;
      },
      setState(key, value) {
        if (jsonEqual(slot.state[key], value)) return;
        slot.state = { ...slot.state, [key]: value };
        rt.log(id, 'state', { data: { [key]: value } });
        for (const l of rt.stateListeners) l(id, slot.state);
      },
      log(level: LogLevel, message: string, data?: Json) {
        rt.log(id, 'log', { data: { level, message, ...(data !== undefined ? { data } : {}) } });
      },
      setTimer(delayMs, fn, label): TimerHandle {
        const timerId = `${id}:${slot.timerOrdinal++}`;
        const setSeq = rt.log(id, 'timer.set', { data: { id: timerId, delay: delayMs, ...(label ? { label } : {}) } });
        let done = false;
        const fireAt = rt.opts.world.timerFireAt(timerId, rt.t, delayMs);
        const fire = () => {
          rt.liveTimers.delete(timerId);
          if (done || !rt.running) return;
          done = true;
          rt.cause = rt.log(id, 'timer.fire', { data: { id: timerId }, cause: [setSeq] });
          rt.guard(slot, fn);
        };
        const cancel = rt.opts.scheduler.at(fireAt.t, () => rt.turn(fire), fireAt.order);
        if (rt.opts.mode === 'live') rt.liveTimers.set(timerId, { due: fireAt.t, fire });
        return {
          clear() {
            if (done) return;
            done = true;
            rt.liveTimers.delete(timerId);
            cancel();
            rt.log(id, 'timer.clear', { data: { id: timerId } });
          },
        };
      },
      io: {
        open: (channel, request, handlers) => rt.openExchange(slot, channel, request, handlers),
      },
      media: {
        append(track, samples, sampleRate) {
          return rt.trajectory.tracks.appendF32(`${id}.${track}`, samples, sampleRate);
        },
      },
    };
  }

  private openExchange(slot: NodeSlot, channel: string, request: Json, handlers: IOHandlers): IOHandle {
    const n = slot.ioOrdinal.get(channel) ?? 0;
    slot.ioOrdinal.set(channel, n + 1);
    const xid = `${slot.id}/${channel}#${n}`;
    const node = slot.id;
    const ioTrack = `io/${node}/${channel}`;
    const openSeq = this.log(node, 'io.open', { data: { xid, channel, request } });
    const rt = this;

    const handle = {
      xid,
      closed: false,
      send(data: Json, o: SendOptions = {}) {
        if (handle.closed) return;
        const ev: Partial<TrajEvent> = { data: { xid, payload: data } };
        if (o.media) ev.media = o.media;
        if (o.bin && o.logBin !== false) ev.bin = rt.trajectory.tracks.appendBytes(`${ioTrack}/send`, o.bin);
        rt.log(node, 'io.send', ev);
        conn.send(data, o.bin);
      },
      close(reason?: Json) {
        if (handle.closed) return;
        handle.closed = true;
        rt.openConns.delete(xid);
        rt.log(node, 'io.close', { data: { xid, by: 'node', payload: reason ?? null } });
        conn.close();
      },
    };

    const conn = this.opts.world.open(
      { xid, node, channel, request, t: this.t },
      {
        recv: (data, bin) =>
          rt.turn(() => {
            if (handle.closed || !rt.running) return;
            const ev: Partial<TrajEvent> = { data: { xid, payload: data }, cause: [openSeq] };
            if (bin) ev.bin = rt.trajectory.tracks.appendBytes(ioTrack, bin);
            rt.cause = rt.log(node, 'io.recv', ev);
            rt.guard(slot, () => handlers.onRecv?.(data, bin));
          }),
        close: (data) =>
          rt.turn(() => {
            if (handle.closed || !rt.running) return;
            handle.closed = true;
            rt.openConns.delete(xid);
            rt.cause = rt.log(node, 'io.close', { data: { xid, by: 'remote', payload: data ?? null }, cause: [openSeq] });
            rt.guard(slot, () => handlers.onClose?.(data ?? null));
          }),
      },
    );
    this.openConns.set(xid, { conn, handle });
    return handle;
  }
}

function newSessionId(): string {
  const d = new Date();
  const p = (n: number, w = 2) => String(n).padStart(w, '0');
  const stamp = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  return `${stamp}-${Math.random().toString(36).slice(2, 6)}`;
}
