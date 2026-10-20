// The page's end of /api/live (see src/live-protocol.ts): captures the mic when the server's mic
// node asks, plays what the server's speaker schedules, and mirrors the run's trajectory.

import { Trajectory, type TrajectoryMeta } from '../core/trajectory';
import type { GraphSpec } from '../core/graph';
import type { Json, TrajEvent } from '../core/types';
import { f32Of, frame, unframe } from '../live-protocol';
import { captureMic, speakerOutputs, type MicRequest } from './audio';
import { PageGate } from './gate';
import { resolveParams } from '../core/node';
import { duplexTurnNode } from '../nodes/turn';

/** The graph has turn-taking that may arm the page gate. */
function wantsPageGate(graph: GraphSpec): boolean {
  return graph.nodes.some((n) => {
    if (n.type !== duplexTurnNode.type) return false;
    const p = resolveParams(duplexTurnNode, n.params ?? {});
    return !!p.pageGate && !!p.bargeIn && Number(p.yieldWithinMs) > 0;
  });
}

export interface LiveHandlers {
  /** A run started on the server; `traj` fills as its events arrive. */
  session(traj: Trajectory): void;
  states(states: Record<string, Record<string, Json>>): void;
  /** The run stopped; its full recording follows in `bundle`. */
  ended(info: { id: string; stoppedAt?: number }): void;
  bundle(bytes: Uint8Array): void;
  saved(id: string, path: string): void;
  renamed(from: string, to: string, path: string): void;
  notice(text: string): void;
  /** The socket closed (another page took over, or the server went away). */
  closed(reason: string): void;
}

export class LiveClient {
  private ws: WebSocket | null = null;
  private opening: Promise<void> | null = null;
  private traj: Trajectory | null = null;
  /** Server clock minus page clock (performance.now), from the ping with the smallest round trip. */
  private offset = 0;
  private pings: { rtt: number; offset: number }[] = [];
  private pingTimer: ReturnType<typeof setInterval> | undefined;
  /** The session clock at server time `s`. */
  private clock = { now: 0, s: 0, paused: true };
  private stopMic: (() => void) | null = null;
  private speakers = new Map<number, ReturnType<typeof speakerOutputs>>();
  /** Speakers the server lets the page stop on its own VAD, until a session time, with the fade to use. */
  private gateArms = new Map<number, { until: number; releaseMs: number }>();
  /** The mic the page gate listens to: 16 kHz with browser echo cancellation (else it would hear the agent). */
  private gateMic: { frameSamples: number } | null = null;
  private gate = new PageGate((pos) => this.gateOnset(pos));

  constructor(private on: LiveHandlers) {}

  /** Session time now, as the server's clock has it. */
  now = (): number => {
    if (this.clock.paused) return this.clock.now;
    return this.clock.now + Math.max(0, performance.now() + this.offset - this.clock.s);
  };

  /** Round trip of the best clock sample (ms), for display. */
  get rtt(): number | undefined {
    return this.pings.length ? Math.min(...this.pings.map((p) => p.rtt)) : undefined;
  }

  connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) return Promise.resolve();
    return (this.opening ??= new Promise<void>((resolve, reject) => {
      const u = new URL('/api/live', location.href);
      u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(u);
      ws.binaryType = 'arraybuffer';
      this.ws = ws;
      ws.onopen = () => {
        this.opening = null;
        this.pings = [];
        // A burst to sync the clocks at once, then one now and then to follow drift.
        for (let i = 0; i < 8; i++) setTimeout(() => this.ping(), i * 40);
        this.pingTimer = setInterval(() => this.ping(), 2000);
        setTimeout(resolve, 8 * 40 + 50);
      };
      ws.onerror = () => {
        this.opening = null;
        reject(new Error('could not connect to the server'));
      };
      ws.onclose = (e) => {
        this.opening = null;
        clearInterval(this.pingTimer);
        if (this.ws !== ws) return;
        this.ws = null;
        this.teardown();
        this.on.closed(e.reason || `connection closed (${e.code})`);
      };
      ws.onmessage = (e) => {
        if (typeof e.data === 'string') this.onMessage(JSON.parse(e.data) as { type: string; [k: string]: Json });
        else this.onBinary(new Uint8Array(e.data as ArrayBuffer));
      };
    }));
  }

  private send(m: Json) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(m));
  }

  private ping() {
    this.send({ type: 'ping', id: 0, t: performance.now() });
  }

  start(graph: GraphSpec, o: { env: Record<string, Json>; autosave: boolean; title: string }) {
    // Load the page gate's VAD now, so the first reply finds it ready.
    if (wantsPageGate(graph)) this.gate.load();
    this.send({ type: 'start', graph: graph as unknown as Json, env: o.env, autosave: o.autosave, title: o.title });
  }

  /** Replay a recording on the server, with `liveChannels` served live. */
  replay(bundle: Uint8Array, o: { graph?: GraphSpec; liveChannels: string[]; audible: boolean }) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    this.ws.send(frame({ k: 'replay', ...(o.graph ? { graph: o.graph as unknown as Json } : {}), liveChannels: o.liveChannels, audible: o.audible }, bundle));
  }

  stop() {
    this.send({ type: 'stop' });
  }
  pause() {
    this.send({ type: 'pause' });
  }
  resume() {
    this.send({ type: 'resume' });
  }

  private teardown() {
    this.stopMic?.();
    this.stopMic = null;
    for (const s of this.speakers.values()) s.close();
    this.speakers.clear();
    this.gateArms.clear();
  }

  private onMessage(m: { type: string; [k: string]: Json }) {
    switch (m.type) {
      case 'pong': {
        const rtt = performance.now() - (m.t as number);
        this.pings = [...this.pings.slice(-19), { rtt, offset: (m.s as number) - ((m.t as number) + rtt / 2) }];
        this.offset = this.pings.reduce((a, b) => (b.rtt < a.rtt ? b : a)).offset;
        return;
      }
      case 'session':
        this.traj = new Trajectory(m.meta as unknown as TrajectoryMeta);
        this.clock = { now: 0, s: performance.now() + this.offset, paused: false };
        return this.on.session(this.traj);
      case 'events':
        this.clock = m.clock as typeof this.clock;
        for (const e of m.events as unknown as TrajEvent[]) this.traj?.append(e);
        return;
      case 'state':
        return this.on.states(m.states as Record<string, Record<string, Json>>);
      case 'mic.open':
        return this.openMic(m.request as unknown as MicRequest);
      case 'mic.close':
        this.stopMic?.();
        this.stopMic = null;
        this.gateMic = null;
        return;
      case 'speaker.close':
        this.speakers.get(m.id as number)?.close();
        this.speakers.delete(m.id as number);
        this.gateArms.delete(m.id as number);
        return;
      case 'ended':
        this.clock = { ...this.clock, paused: true };
        return this.on.ended({ id: m.id as string, ...(typeof m.stoppedAt === 'number' ? { stoppedAt: m.stoppedAt } : {}) });
      case 'saved':
        return this.on.saved(m.id as string, m.path as string);
      case 'renamed':
        return this.on.renamed(m.from as string, m.to as string, m.path as string);
      case 'notice':
        return this.on.notice(m.text as string);
    }
  }

  private onBinary(buf: Uint8Array) {
    const { header: h, payload } = unframe(buf);
    if (h.k === 'track') this.traj?.tracks.appendF32(h.name as string, f32Of(payload), h.sampleRate as number);
    else if (h.k === 'bundle') this.on.bundle(payload.slice());
    else if (h.k === 'speaker') {
      const id = h.id as number;
      if (h.cmd === 'gate') {
        if (typeof h.until !== 'number') return void this.gateArms.delete(id);
        this.gate.load();
        return void this.gateArms.set(id, { until: h.until, releaseMs: (h.releaseMs as number) ?? 40 });
      }
      let s = this.speakers.get(id);
      if (!s) this.speakers.set(id, (s = speakerOutputs(this.now)));
      // The session time now: the clock when the command was sent, plus the time since (server clock).
      const now = (h.now as number) + Math.max(0, performance.now() + this.offset - (h.s as number));
      const r = s.command(h as unknown as { cmd: string }, payload.byteLength ? payload : undefined, now);
      // When a run of playback starts after its time (the audio arrived late), tell the speaker node.
      if (r) this.send({ type: 'speaker.late', id, at: h.at as number, ms: Math.round(r.lateMs) });
    }
  }

  /** The page gate heard the user start: stop each armed speaker now and tell the server when, and where in the mic. */
  private gateOnset(pos: number) {
    const now = this.now();
    for (const [id, arm] of this.gateArms) {
      this.gateArms.delete(id);
      if (now >= arm.until) continue;
      this.speakers.get(id)?.gateCut(now, arm.releaseMs);
      this.send({ type: 'speaker.gate', id, at: now, pos });
    }
  }

  private openMic(r: MicRequest) {
    this.stopMic?.();
    this.gate.reset();
    this.gateMic = r.sampleRate === 16000 && r.echoCancellation ? { frameSamples: r.frameSamples } : null;
    this.stopMic = captureMic(r, {
      ready: (device) => this.send({ type: 'mic.ready', ...(device as Record<string, Json>) }),
      frame: (samples, n, ctxTime) => {
        if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(frame({ k: 'mic', frame: n, ctxTime }, new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength)));
        if (this.gateMic) this.gate.push(samples, n * this.gateMic.frameSamples);
      },
      error: (e) => this.send({ type: 'mic.error', error: e instanceof Error ? `${e.name}: ${e.message}` : String(e) }),
    });
  }
}
