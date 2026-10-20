// The pipeline runs here. A page connects one socket (/api/live, see src/live-protocol.ts): it
// sends mic frames and controls, and gets the agent's audio to play plus the run's trajectory to
// show. Each page has its own run (session, recording, autosave). Up to `maxSessions` (default 1) use the
// GPU box at once; a page that asks for another waits in a FIFO queue and starts when its turn comes.
// Every page is told how many pages are connected, how many sessions run and how many wait.

import { mkdir, rename, stat, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { WebSocket } from 'ws';
import { registry } from '../src/nodes';
import { Runtime } from '../src/core/runtime';
import { LiveScheduler, type SimScheduler } from '../src/core/scheduler';
import { LiveWorld, type Delivery, type DriverRegistry, type LiveDriver } from '../src/core/world';
import { prepareReplay } from '../src/core/replay';
import { decodeBundle, encodeBundle } from '../src/core/bundle';
import { checkGraph, type GraphSpec } from '../src/core/graph';
import type { Trajectory } from '../src/core/trajectory';
import type { Json, TrajEvent } from '../src/core/types';
import { frame, unframe, type Header } from '../src/live-protocol';
import { serverDrivers, sessionTitle, type Upstreams } from './drivers';

export interface LiveOptions {
  upstreams: Upstreams;
  sessionsDir(): string;
  modelsDir: string;
  /** Canned lines' audio, rendered once per box (default: tts-cache in modelsDir's folder). */
  cannedDir?: string;
  /** How many sessions may use the GPU box at once; more wait in a queue. Default 1. */
  maxSessions?: () => number;
  /** A live session is stopped after this long (ms; 0 = never), so a stuck page can't hold the box. */
  maxSessionMs?: () => number;
  /** How often the server pings each page (ms); a page that misses two pongs is dropped. */
  heartbeatMs?: number;
  /**
   * Public demo: what a visitor's live session runs (the graph the host started last, else the default). Visitors'
   * own graphs are ignored and they can't replay on the box.
   */
  demoGraph?: () => GraphSpec;
  /** The host started a live session with this graph (the next demo graph). */
  hostGraph?: (graph: GraphSpec) => void;
  log?: (msg: string) => void;
}

/** What was said in a session, for naming it: user turns and the agent's replies, in order. */
function transcriptOf(traj: Trajectory): string {
  const lines: string[] = [];
  for (const e of traj.events) {
    if (e.kind !== 'msg') continue;
    const d = e.data as { type?: string; text?: string };
    if (d.type === 'turn.user' && d.text) lines.push(`User: ${d.text}`);
    else if (d.type === 'llm.end' && d.text) lines.push(`Agent: ${d.text}`);
  }
  return lines.join('\n').slice(0, 6000);
}

/** A title made safe as a session (file) name. */
function titleToName(title: string): string {
  return title
    .replace(/[\/\\:*?"<>|\x00-\x1f]/g, ' ')
    .replace(/^[\s.]+/, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 60)
    .trim();
}

/** A connected page: its mic and speakers, and the run it started (at most one). */
class Page {
  run: Run | null = null;
  micSink: Delivery | undefined;
  speakers = new Map<number, Delivery>();
  private speakerSeq = 0;
  /** The server's drivers plus this page's mic and speaker. */
  readonly drivers: DriverRegistry;

  constructor(
    readonly ws: WebSocket,
    base: DriverRegistry,
    /** The host (token or local), not a demo visitor. */
    readonly host: boolean,
  ) {
    this.drivers = { ...base, mic: this.mic, speaker: this.speaker };
  }
  send(m: Json) {
    if (this.ws.readyState === this.ws.OPEN) this.ws.send(JSON.stringify(m));
  }
  sendBin(h: Header, payload?: Uint8Array) {
    if (this.ws.readyState === this.ws.OPEN) this.ws.send(frame(h, payload), { binary: true });
  }
  get open() {
    return this.ws.readyState === this.ws.OPEN;
  }

  private mic: LiveDriver = (req, sink) => {
    if (!this.open) {
      setTimeout(() => sink.close({ error: 'no page connected' }), 0);
      return { send() {}, close() {} };
    }
    this.micSink = sink;
    this.send({ type: 'mic.open', request: req });
    return {
      send() {},
      close: () => {
        if (this.micSink === sink) this.micSink = undefined;
        this.send({ type: 'mic.close' });
      },
    };
  };

  /** Renders the speaker's schedule in the page; each command carries the session clock when sent. */
  private speaker: LiveDriver = (_req, sink, env) => {
    const id = ++this.speakerSeq;
    this.speakers.set(id, sink);
    setTimeout(() => sink.recv({ type: 'ready' }), 0);
    return {
      send: (data, bin) => this.sendBin({ k: 'speaker', id, ...(data as Record<string, Json>), now: env.now(), s: performance.now() }, bin),
      close: () => {
        this.speakers.delete(id);
        this.send({ type: 'speaker.close', id });
      },
    };
  };
}

interface Run {
  rt: Runtime;
  sim?: SimScheduler;
  now(): number;
  autosave: boolean;
  title: string;
  /** Uses the GPU box (a live session, or a replay with live channels): counts against the cap. */
  box: boolean;
  /** When the session limit stops it (Date.now() ms), for live sessions under a limit. */
  endsAt?: number;
  /** Stops the run; resolves once it is finished and handed over. */
  stop(): Promise<void>;
  detach(): void;
}

export class LiveHub {
  private pages = new Set<Page>();
  private drivers: DriverRegistry;
  private lastUsers = '';
  /** Pages waiting for the box, first in line first, with what to run when their turn comes. */
  private queue: { page: Page; go: () => void; told?: number }[] = [];
  /** Re-sends each waiting page its place and wait while anyone waits. */
  private queueTick?: ReturnType<typeof setInterval>;

  constructor(private o: LiveOptions) {
    this.drivers = serverDrivers(o.upstreams, { modelsDir: o.modelsDir, cannedDir: o.cannedDir ?? join(o.modelsDir, '..', 'fv-tts-cache') });
  }

  private log(m: string) {
    (this.o.log ?? console.log)(`[live] ${m}`);
  }

  private get maxSessions() {
    return Math.max(1, Math.floor(this.o.maxSessions?.() ?? 1));
  }

  /** Runs in progress that use the GPU box (live sessions, replays with live channels). */
  get sessions() {
    let n = 0;
    for (const p of this.pages) if (p.run?.box) n++;
    return n;
  }

  /** Who is here, for every page: connected pages, running sessions, the cap, pages waiting. Sent when it changes (and to `to`). */
  private broadcastUsers(to?: Page) {
    const m = { type: 'users', pages: this.pages.size, sessions: this.sessions, max: this.maxSessions, queued: this.queue.length };
    const key = JSON.stringify(m);
    if (key !== this.lastUsers) {
      this.lastUsers = key;
      for (const p of this.pages) p.send(m);
    } else to?.send(m);
  }

  /** A page connected. Each page runs its own sessions; pages don't take over each other, they queue. */
  handle(ws: WebSocket, { host = true }: { host?: boolean } = {}) {
    const page = new Page(ws, this.drivers, host);
    this.pages.add(page);
    this.broadcastUsers(page);
    // A tab closed behind a tunnel may never send a close: without pongs the socket is dead.
    let alive = true;
    ws.on('pong', () => (alive = true));
    const beat = setInterval(() => {
      if (!alive) {
        this.log('a page stopped answering pings: dropping it');
        return ws.terminate();
      }
      alive = false;
      ws.ping();
    }, this.o.heartbeatMs ?? 10_000);
    ws.on('message', (data: Buffer, binary) => {
      try {
        if (binary) this.onBinary(page, unframe(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)));
        else this.onMessage(page, JSON.parse(data.toString('utf8')) as { type: string; [k: string]: Json });
      } catch (e) {
        this.log(`bad message from the page: ${String(e)}`);
      }
    });
    ws.on('close', () => {
      clearInterval(beat);
      this.pages.delete(page);
      this.leaveQueue(page);
      // Its mic and speaker are gone: its run can't go on (its end lets the next page in).
      void page.run?.stop();
      this.broadcastUsers();
    });
  }

  private onMessage(page: Page, m: { type: string; [k: string]: Json }) {
    switch (m.type) {
      case 'ping':
        return page.send({ type: 'pong', id: m.id, t: m.t, s: performance.now() });
      case 'start':
        return this.startLive(page, m as unknown as { graph: GraphSpec; env?: Record<string, Json>; autosave?: boolean; title?: string });
      case 'stop':
        // Stop the run, or leave the queue.
        if (this.leaveQueue(page)) return page.send({ type: 'queue', position: 0 });
        return void page.run?.stop();
      case 'pause':
        return page.run?.sim?.pause();
      case 'resume':
        return page.run?.sim?.resume();
      case 'mic.ready':
        return page.micSink?.recv({ type: 'ready', label: m.label ?? null, contextSampleRate: m.contextSampleRate ?? null, settings: m.settings ?? null });
      case 'mic.error':
        return page.micSink?.close({ error: m.error ?? 'microphone failed' });
      case 'speaker.late':
        return page.speakers.get(m.id as number)?.recv({ type: 'late', at: m.at, ms: m.ms });
      case 'speaker.gate':
        return page.speakers.get(m.id as number)?.recv({ type: 'gate', at: m.at, pos: m.pos ?? null, stop: m.stop ?? null, silent: m.silent ?? null });
    }
  }

  private onBinary(page: Page, { header, payload }: { header: Header; payload: Uint8Array }) {
    if (header.k === 'mic') page.micSink?.recv({ frame: header.frame, ctxTime: header.ctxTime }, payload.slice());
    else if (header.k === 'replay') {
      if (!page.host) return page.send({ type: 'notice', text: 'replays on the GPU box are for the host only' });
      this.startReplay(page, header, payload);
    }
  }

  /**
   * Run `go` for `page` now if the box has room (or the run doesn't use it), else queue it: the page hears its
   * place in line (`queue`, position from 1) and `go` runs when its turn comes.
   */
  private request(page: Page, box: boolean, go: () => void) {
    if (page.run || this.queue.some((q) => q.page === page)) return page.send({ type: 'notice', text: 'a session is already running or waiting' });
    if (!box || (this.sessions < this.maxSessions && !this.queue.length)) return go();
    this.queue.push({ page, go });
    this.log(`a page waits for the box (${this.queue.length} in line)`);
    this.tellQueue();
    this.broadcastUsers();
  }

  /** Take `page` out of the queue; true if it was waiting. */
  private leaveQueue(page: Page): boolean {
    const n = this.queue.length;
    this.queue = this.queue.filter((q) => q.page !== page);
    if (this.queue.length === n) return false;
    this.tellQueue();
    this.broadcastUsers();
    return true;
  }

  private get maxSessionMs() {
    return Math.max(0, this.o.maxSessionMs?.() ?? 0);
  }

  /**
   * The longest each waiting page may wait (ms, in line order), from the session limit: the box's runs end by their
   * limit, and each page ahead may use a whole one. Null without a limit.
   */
  private waits(): (number | null)[] {
    const limit = this.maxSessionMs;
    if (!limit) return this.queue.map(() => null);
    const now = Date.now();
    const slots: number[] = [];
    for (const p of this.pages) if (p.run?.box) slots.push(Math.max(0, (p.run.endsAt ?? now + limit) - now));
    while (slots.length < this.maxSessions) slots.push(0);
    return this.queue.map(() => {
      slots.sort((a, b) => a - b);
      const wait = slots[0];
      slots[0] += limit;
      return wait;
    });
  }

  /** Tell waiting pages their place in line and how long they may wait; `all`: also those whose place didn't change. */
  private tellQueue(all = false) {
    const waits = this.waits();
    this.queue.forEach((q, i) => {
      if (all || q.told !== i + 1) q.page.send({ type: 'queue', position: (q.told = i + 1), ahead: i, waitMs: waits[i] });
    });
    if (!this.queue.length) {
      clearInterval(this.queueTick);
      this.queueTick = undefined;
    } else if (!this.queueTick) {
      this.queueTick = setInterval(() => this.tellQueue(true), 5000);
      this.queueTick.unref?.();
    }
  }

  /** The box has room: start the pages first in line. */
  private next() {
    while (this.queue.length && this.sessions < this.maxSessions) {
      const q = this.queue.shift()!;
      if (!q.page.open) continue;
      q.page.send({ type: 'queue', position: 0, started: true });
      this.log('the next page in line starts');
      q.go();
    }
    this.tellQueue();
    this.broadcastUsers();
  }

  // ---- runs -------------------------------------------------------------------

  /** Stream a run's trajectory and node states to the page, in order. */
  private stream(page: Page, rt: Runtime, now: () => number, paused: () => boolean) {
    let events: TrajEvent[] = [];
    let states: Record<string, Json> = {};
    let hasStates = false;
    let scheduled = false;
    const clock = () => ({ now: now(), s: performance.now(), paused: paused() });
    const flush = () => {
      scheduled = false;
      if (events.length) page.send({ type: 'events', events: events as unknown as Json, clock: clock() });
      if (hasStates) page.send({ type: 'state', states });
      events = [];
      states = {};
      hasStates = false;
    };
    const later = () => {
      if (!scheduled) {
        scheduled = true;
        setImmediate(flush);
      }
    };
    page.send({ type: 'session', meta: rt.trajectory.meta as unknown as Json });
    const offEv = rt.trajectory.subscribe((e) => {
      events.push(e);
      later();
    });
    const offTrack = rt.trajectory.tracks.onAppendF32((name, bytes, sampleRate) => {
      // Before the events that refer to these samples.
      flush();
      page.sendBin({ k: 'track', name, sampleRate }, bytes);
    });
    const offState = rt.onState((node, state) => {
      states[node] = state;
      hasStates = true;
      later();
    });
    const tick = setInterval(() => page.send({ type: 'events', events: [], clock: clock() }), 1000);
    return {
      flush,
      detach() {
        flush();
        offEv();
        offTrack();
        offState();
        clearInterval(tick);
      },
    };
  }

  private startLive(page: Page, m: { graph: GraphSpec; env?: Record<string, Json>; autosave?: boolean; title?: string }) {
    if (!page.host && this.o.demoGraph) m = { graph: this.o.demoGraph(), env: m.env, autosave: true, title: '' };
    this.request(page, true, () => this.runLive(page, m));
  }

  private runLive(page: Page, m: { graph: GraphSpec; env?: Record<string, Json>; autosave?: boolean; title?: string }) {
    let graph: GraphSpec;
    try {
      graph = checkGraph(m.graph);
    } catch (e) {
      return page.send({ type: 'notice', text: `bad graph: ${String(e)}` });
    }
    if (page.host) this.o.hostGraph?.(graph);
    const scheduler = new LiveScheduler();
    const rt = new Runtime({
      graph,
      registry,
      scheduler,
      world: new LiveWorld(page.drivers, scheduler),
      mode: 'live',
      env: { ...m.env, server: `node ${process.version}` },
    });
    const s = this.stream(page, rt, () => scheduler.now(), () => false);
    let stopping: Promise<void> | undefined;
    const limit = this.maxSessionMs;
    const timer =
      limit > 0
        ? setTimeout(() => {
            page.send({ type: 'notice', text: `the session reached the ${Math.round(limit / 1000)} s limit and was stopped` });
            this.log(`live session ${rt.trajectory.meta.id} hit the time limit`);
            void run.stop();
          }, limit)
        : undefined;
    const run: Run = {
      rt,
      now: () => scheduler.now(),
      autosave: m.autosave !== false,
      title: m.title ?? '',
      box: true,
      endsAt: limit > 0 ? Date.now() + limit : undefined,
      detach: s.detach,
      stop: () =>
        (stopping ??= (async () => {
          clearTimeout(timer);
          try {
            rt.stop();
          } finally {
            await this.finish(page, run);
          }
        })()),
    };
    page.run = run;
    if (limit > 0) page.send({ type: 'limit', leftMs: limit });
    this.broadcastUsers();
    this.tellQueue(true);
    this.log(`live session ${rt.trajectory.meta.id} started`);
    rt.start();
  }

  private startReplay(page: Page, h: Header, bundle: Uint8Array) {
    const liveChannels = (h.liveChannels as string[]) ?? [];
    this.request(page, liveChannels.length > 0, () => this.runReplay(page, h, bundle, liveChannels));
  }

  private runReplay(page: Page, h: Header, bundle: Uint8Array, liveChannels: string[]) {
    let source: Trajectory;
    try {
      source = decodeBundle(bundle);
    } catch (e) {
      return page.send({ type: 'notice', text: `replay: ${String(e)}` });
    }
    const outputChannels = Object.values(registry).flatMap((d) => (d.channels ?? []).filter((c) => c.output).map((c) => c.name));
    const { runtime: rt, sim } = prepareReplay(source, {
      registry,
      graph: h.graph ? checkGraph(h.graph as unknown as GraphSpec) : source.meta.graph,
      drivers: page.drivers,
      liveChannels,
      monitorChannels: h.audible ? outputChannels : [],
    });
    const s = this.stream(page, rt, () => sim.now(), () => sim.isPaused);
    let stoppedAt: number | undefined;
    const done = sim.runRealtime(1).then(async () => {
      stoppedAt = rt.isRunning ? sim.now() : undefined;
      if (rt.isRunning) rt.stop();
      await this.finish(page, run, stoppedAt);
    });
    const run: Run = {
      rt,
      sim,
      now: () => sim.now(),
      autosave: false,
      title: '',
      box: liveChannels.length > 0,
      detach: s.detach,
      stop: () => {
        sim.stop();
        return done;
      },
    };
    page.run = run;
    this.broadcastUsers();
    this.log(`replay of ${source.meta.id} started (live: ${liveChannels.join(',') || 'none'})`);
  }

  /** Hand a stopped run to the page (its full bundle), then save and name it. */
  private async finish(page: Page, run: Run, stoppedAt?: number) {
    run.detach();
    if (page.run === run) page.run = null;
    page.micSink = undefined;
    this.next();
    const traj = run.rt.trajectory;
    const bytes = encodeBundle(traj);
    page.send({ type: 'ended', id: traj.meta.id, ...(stoppedAt !== undefined ? { stoppedAt } : {}) });
    page.sendBin({ k: 'bundle' }, bytes);
    this.log(`${traj.meta.mode} session ${traj.meta.id} stopped (${(traj.duration / 1000).toFixed(1)} s, ${traj.events.length} events)`);
    if (traj.meta.mode !== 'live' || !run.autosave) return;
    const dir = this.o.sessionsDir();
    let id = traj.meta.id;
    try {
      await mkdir(dir, { recursive: true });
      await writeFile(join(dir, `${id}.fvs`), bytes);
      page.send({ type: 'saved', id, path: join(dir, `${id}.fvs`) });
    } catch (e) {
      return page.send({ type: 'notice', text: `Session not saved (${String(e)}).` });
    }
    const transcript = run.title ? transcriptOf(traj) : '';
    if (!transcript) return;
    let name: string;
    try {
      name = titleToName(await sessionTitle(this.o.upstreams, run.title, transcript));
    } catch (e) {
      return page.send({ type: 'notice', text: `Could not title the session with ${run.title}: ${String(e).slice(0, 200)}` });
    }
    if (!name) return;
    for (let n = 1; n <= 9; n++) {
      const to = n === 1 ? name : `${name} ${n}`;
      const dest = join(dir, `${to}.fvs`);
      if (await stat(dest).catch(() => null)) continue;
      await rename(join(dir, `${id}.fvs`), dest);
      page.send({ type: 'renamed', from: id, to, path: dest });
      id = to;
      return;
    }
  }
}
