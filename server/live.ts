// The pipeline runs here. A page connects one socket (/api/live, see src/live-protocol.ts): it
// sends mic frames and controls, and gets the agent's audio to play plus the run's trajectory to
// show. One page at a time: a page that connects takes over, and a run of the previous page stops.

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

class Page {
  constructor(readonly ws: WebSocket) {}
  send(m: Json) {
    if (this.ws.readyState === this.ws.OPEN) this.ws.send(JSON.stringify(m));
  }
  sendBin(h: Header, payload?: Uint8Array) {
    if (this.ws.readyState === this.ws.OPEN) this.ws.send(frame(h, payload), { binary: true });
  }
}

interface Run {
  rt: Runtime;
  sim?: SimScheduler;
  now(): number;
  autosave: boolean;
  title: string;
  /** Stops the run; resolves once it is finished and handed over. */
  stop(): Promise<void>;
  detach(): void;
}

export class LiveHub {
  private page: Page | null = null;
  private run: Run | null = null;
  private micSink: Delivery | undefined;
  private speakers = new Map<number, Delivery>();
  private speakerSeq = 0;
  private drivers: DriverRegistry;

  constructor(private o: LiveOptions) {
    this.drivers = { ...serverDrivers(o.upstreams, { modelsDir: o.modelsDir, cannedDir: o.cannedDir ?? join(o.modelsDir, '..', 'fv-tts-cache') }), mic: this.mic, speaker: this.speaker };
  }

  private log(m: string) {
    (this.o.log ?? console.log)(`[live] ${m}`);
  }

  /** A page connected. */
  handle(ws: WebSocket) {
    const prev = this.page;
    const page = new Page(ws);
    this.page = page;
    if (prev) {
      this.log('a new page took over');
      prev.ws.close(4000, 'another page took over');
    }
    void this.run?.stop();
    ws.on('message', (data: Buffer, binary) => {
      if (this.page !== page) return;
      try {
        if (binary) this.onBinary(page, unframe(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)));
        else this.onMessage(page, JSON.parse(data.toString('utf8')) as { type: string; [k: string]: Json });
      } catch (e) {
        this.log(`bad message from the page: ${String(e)}`);
      }
    });
    ws.on('close', () => {
      if (this.page !== page) return;
      this.page = null;
      // Its mic and speaker are gone: a run can't go on.
      void this.run?.stop();
    });
  }

  private onMessage(page: Page, m: { type: string; [k: string]: Json }) {
    switch (m.type) {
      case 'ping':
        return page.send({ type: 'pong', id: m.id, t: m.t, s: performance.now() });
      case 'start':
        return this.startLive(page, m as unknown as { graph: GraphSpec; env?: Record<string, Json>; autosave?: boolean; title?: string });
      case 'stop':
        return void this.run?.stop();
      case 'pause':
        return this.run?.sim?.pause();
      case 'resume':
        return this.run?.sim?.resume();
      case 'mic.ready':
        return this.micSink?.recv({ type: 'ready', label: m.label ?? null, contextSampleRate: m.contextSampleRate ?? null, settings: m.settings ?? null });
      case 'mic.error':
        return this.micSink?.close({ error: m.error ?? 'microphone failed' });
      case 'speaker.late':
        return this.speakers.get(m.id as number)?.recv({ type: 'late', at: m.at, ms: m.ms });
      case 'speaker.gate':
        return this.speakers.get(m.id as number)?.recv({ type: 'gate', at: m.at, pos: m.pos ?? null });
    }
  }

  private onBinary(page: Page, { header, payload }: { header: Header; payload: Uint8Array }) {
    if (header.k === 'mic') this.micSink?.recv({ frame: header.frame, ctxTime: header.ctxTime }, payload.slice());
    else if (header.k === 'replay') this.startReplay(page, header, payload);
  }

  // ---- the page's devices -------------------------------------------------------

  private mic: LiveDriver = (req, sink) => {
    const page = this.page;
    if (!page) {
      setTimeout(() => sink.close({ error: 'no page connected' }), 0);
      return { send() {}, close() {} };
    }
    this.micSink = sink;
    page.send({ type: 'mic.open', request: req });
    return {
      send() {},
      close: () => {
        if (this.micSink === sink) this.micSink = undefined;
        page.send({ type: 'mic.close' });
      },
    };
  };

  /** Renders the speaker's schedule in the page; each command carries the session clock when sent. */
  private speaker: LiveDriver = (_req, sink, env) => {
    const id = ++this.speakerSeq;
    const page = this.page;
    this.speakers.set(id, sink);
    setTimeout(() => sink.recv({ type: 'ready' }), 0);
    return {
      send: (data, bin) => page?.sendBin({ k: 'speaker', id, ...(data as Record<string, Json>), now: env.now(), s: performance.now() }, bin),
      close: () => {
        this.speakers.delete(id);
        page?.send({ type: 'speaker.close', id });
      },
    };
  };

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
    if (this.run) return page.send({ type: 'notice', text: 'a session is already running' });
    let graph: GraphSpec;
    try {
      graph = checkGraph(m.graph);
    } catch (e) {
      return page.send({ type: 'notice', text: `bad graph: ${String(e)}` });
    }
    const scheduler = new LiveScheduler();
    const rt = new Runtime({
      graph,
      registry,
      scheduler,
      world: new LiveWorld(this.drivers, scheduler),
      mode: 'live',
      env: { ...m.env, server: `node ${process.version}` },
    });
    const s = this.stream(page, rt, () => scheduler.now(), () => false);
    let stopping: Promise<void> | undefined;
    const run: Run = {
      rt,
      now: () => scheduler.now(),
      autosave: m.autosave !== false,
      title: m.title ?? '',
      detach: s.detach,
      stop: () =>
        (stopping ??= (async () => {
          rt.stop();
          await this.finish(page, run);
        })()),
    };
    this.run = run;
    this.log(`live session ${rt.trajectory.meta.id} started`);
    rt.start();
  }

  private startReplay(page: Page, h: Header, bundle: Uint8Array) {
    if (this.run) return page.send({ type: 'notice', text: 'a session is already running' });
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
      drivers: this.drivers,
      liveChannels: (h.liveChannels as string[]) ?? [],
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
      detach: s.detach,
      stop: () => {
        sim.stop();
        return done;
      },
    };
    this.run = run;
    this.log(`replay of ${source.meta.id} started (live: ${((h.liveChannels as string[]) ?? []).join(',') || 'none'})`);
  }

  /** Hand a stopped run to the page (its full bundle), then save and name it. */
  private async finish(page: Page, run: Run, stoppedAt?: number) {
    run.detach();
    if (this.run === run) this.run = null;
    this.micSink = undefined;
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
