// The "world" is everything outside the graph: microphones, model APIs,
// speakers, and the timing of timers. A live world talks to real drivers; a
// replay world serves the same exchanges from a recorded trajectory, at the
// same offsets they originally arrived at.

import type { Json, TrajEvent } from './types';
import type { Scheduler } from './scheduler';
import type { Trajectory } from './trajectory';
import { jsonEqual } from './json';

export interface Exchange {
  xid: string;
  node: string;
  channel: string;
  request: Json;
  /** Session time the exchange was opened. */
  t: number;
}

/** Callbacks the world uses to hand data to the graph. */
export interface Delivery {
  recv(data: Json, bin?: Uint8Array): void;
  close(data?: Json): void;
}

export interface Conn {
  send(data: Json, bin?: Uint8Array): void;
  close(): void;
}

export interface DriverEnv {
  /** Session clock, so output drivers can schedule playback in session time. */
  now(): number;
}

/** Talks to one kind of external thing. Must call `sink` asynchronously, never from inside send(). */
export type LiveDriver = (request: Json, sink: Delivery, env: DriverEnv) => Conn;
export type DriverRegistry = Record<string, LiveDriver>;

export interface Divergence {
  xid: string;
  reason: string;
  detail?: Json;
}

export interface World {
  open(ex: Exchange, sink: Delivery): Conn;
  /** Session time at which a timer set at `tSet` actually fires. */
  timerFireAt(timerId: string, tSet: number, delay: number): number;
  /** Report replay divergences into the log. Set by the runtime. */
  onDiverge?: (d: Divergence) => void;
  channelModes(): Record<string, 'live' | 'replay'>;
}

const nullConn: Conn = { send() {}, close() {} };

export class LiveWorld implements World {
  onDiverge?: (d: Divergence) => void;
  constructor(
    private drivers: DriverRegistry,
    private scheduler: Scheduler,
  ) {}

  open(ex: Exchange, sink: Delivery): Conn {
    const driver = this.drivers[ex.channel];
    if (!driver) {
      queueMicrotask(() => sink.close({ error: `no live driver for channel "${ex.channel}"` }));
      return nullConn;
    }
    try {
      return driver(ex.request, sink, { now: () => this.scheduler.now() });
    } catch (e) {
      queueMicrotask(() => sink.close({ error: String(e) }));
      return nullConn;
    }
  }

  timerFireAt(_id: string, tSet: number, delay: number): number {
    return tSet + delay;
  }

  channelModes(): Record<string, 'live' | 'replay'> {
    const out: Record<string, 'live' | 'replay'> = {};
    for (const k of Object.keys(this.drivers)) out[k] = 'live';
    return out;
  }
}

interface RecordedExchange {
  open: TrajEvent;
  /** io.recv events and the remote io.close, in order. */
  incoming: TrajEvent[];
}

export interface ReplayOptions {
  /** Channels to serve live instead of from the recording (e.g. re-run the LLM). */
  liveChannels?: string[];
  /** Output-only channels whose live driver also runs during replay (e.g. the speaker, for audible replay). */
  monitorChannels?: string[];
  drivers?: DriverRegistry;
  /** What to do when a replayed exchange's request differs from the recording. */
  onRequestMismatch?: 'replay' | 'live';
}

export class ReplayWorld implements World {
  onDiverge?: (d: Divergence) => void;
  private exchanges = new Map<string, RecordedExchange>();
  private timers = new Map<string, { set: number; fire: number }>();
  private live: LiveWorld;

  constructor(
    private source: Trajectory,
    private scheduler: Scheduler,
    private opts: ReplayOptions = {},
  ) {
    this.live = new LiveWorld(opts.drivers ?? {}, scheduler);
    const timerSet = new Map<string, number>();
    for (const ev of source.events) {
      const d = ev.data as { xid?: string; id?: string; by?: string } | undefined;
      if (ev.kind === 'io.open' && d?.xid) this.exchanges.set(d.xid, { open: ev, incoming: [] });
      else if (ev.kind === 'io.recv' && d?.xid) this.exchanges.get(d.xid)?.incoming.push(ev);
      else if (ev.kind === 'io.close' && d?.xid && d.by === 'remote') this.exchanges.get(d.xid)?.incoming.push(ev);
      else if (ev.kind === 'timer.set' && d?.id) timerSet.set(d.id, ev.t);
      else if (ev.kind === 'timer.fire' && d?.id && timerSet.has(d.id)) this.timers.set(d.id, { set: timerSet.get(d.id)!, fire: ev.t });
    }
  }

  open(ex: Exchange, sink: Delivery): Conn {
    const monitor =
      this.opts.monitorChannels?.includes(ex.channel) && this.opts.drivers?.[ex.channel]
        ? this.live.open(ex, { recv() {}, close() {} })
        : undefined;

    if (this.opts.liveChannels?.includes(ex.channel)) return this.live.open(ex, sink);

    const rec = this.exchanges.get(ex.xid);
    if (!rec) {
      this.onDiverge?.({ xid: ex.xid, reason: 'not-recorded' });
      return monitor ?? nullConn;
    }
    const recData = rec.open.data as { channel: string; request: Json };
    if (recData.channel !== ex.channel || !jsonEqual(recData.request, ex.request)) {
      this.onDiverge?.({ xid: ex.xid, reason: 'request-mismatch', detail: { recorded: recData.request } });
      if (this.opts.onRequestMismatch === 'live') return this.live.open(ex, sink);
    }

    // Re-deliver everything the outside world sent, at the same offset from open.
    const cancels: (() => void)[] = [];
    for (const ev of rec.incoming) {
      const at = shift(ev.t, rec.open.t, ex.t);
      const payload = (ev.data as { payload?: Json }).payload ?? null;
      if (ev.kind === 'io.recv') {
        const bin = ev.bin ? this.source.tracks.readBytes(ev.bin) : undefined;
        cancels.push(this.scheduler.at(at, () => sink.recv(payload, bin), ev.seq));
      } else {
        cancels.push(this.scheduler.at(at, () => sink.close(payload), ev.seq));
      }
    }
    return {
      send: (data, bin) => monitor?.send(data, bin),
      close: () => {
        for (const c of cancels) c();
        monitor?.close();
      },
    };
  }

  timerFireAt(timerId: string, tSet: number, delay: number): number {
    const rec = this.timers.get(timerId);
    if (!rec) return tSet + delay;
    return shift(rec.fire, rec.set, tSet);
  }

  channelModes(): Record<string, 'live' | 'replay'> {
    const out: Record<string, 'live' | 'replay'> = {};
    for (const { open } of this.exchanges.values()) {
      const ch = (open.data as { channel: string }).channel;
      out[ch] = this.opts.liveChannels?.includes(ch) ? 'live' : 'replay';
    }
    return out;
  }
}

/**
 * Move a recorded time from a recorded anchor to a new anchor. When the anchors
 * match (the replay has not diverged) the recorded time is returned untouched,
 * so floating-point drift can never reorder a faithful replay.
 */
function shift(recorded: number, recordedAnchor: number, anchor: number): number {
  return anchor === recordedAnchor ? recorded : anchor + (recorded - recordedAnchor);
}
