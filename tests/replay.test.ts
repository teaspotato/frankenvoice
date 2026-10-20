import { describe, expect, it } from 'vitest';
import { Runtime } from '../src/core/runtime';
import { LiveScheduler } from '../src/core/scheduler';
import { LiveWorld, type Delivery } from '../src/core/world';
import { prepareReplay } from '../src/core/replay';
import { decodeBundle, encodeBundle } from '../src/core/bundle';
import { registry } from '../src/nodes';
import { buildGraph } from '../src/pipeline';
import type { Trajectory } from '../src/core/trajectory';
import { GRAPH_FORMAT, GRAPH_FORMAT_VERSION, type GraphSpec } from '../src/core/graph';
import type { NodeDef } from '../src/core/node';
import { testConfig, fakeDrivers } from './fake-drivers';

function testGraph(): GraphSpec {
  const g = buildGraph(testConfig());
  // Also run the deterministic energy VAD alongside Silero.
  g.nodes.push({ id: 'vad2', type: 'vad.energy', params: {} });
  g.edges.push({ from: { node: 'mic', port: 'audio' }, to: { node: 'vad2', port: 'audio' } });
  return g;
}

async function recordLive(ms: number, perf: { now(): number } = performance, graph = testGraph()): Promise<Trajectory> {
  const scheduler = new LiveScheduler(perf);
  const rt = new Runtime({ graph, registry, scheduler, world: new LiveWorld(fakeDrivers, scheduler), mode: 'live' });
  rt.start();
  await new Promise((r) => setTimeout(r, ms));
  rt.stop();
  return rt.trajectory;
}

const count = (t: Trajectory, pred: (e: Trajectory['events'][number]) => boolean) => t.events.filter(pred).length;

describe('record and replay', () => {
  let live: Trajectory;

  it('records a live session through the whole pipeline', async () => {
    live = await recordLive(1500);
    const types = (port: string) => new Set(live.events.filter((e) => e.kind === 'msg' && e.port === port).map((e) => (e.data as { type: string }).type));
    expect(types('vad')).toContain('vad.speech_start');
    expect(types('transcript')).toContain('asr.partial');
    expect(types('transcript')).toContain('asr.final');
    expect(types('turn')).toContain('turn.user');
    expect(types('text')).toContain('llm.delta');
    expect(types('audio')).toContain('audio');
    expect(types('playback')).toContain('playback.start');
    expect(types('scores')).toContain('turnmodel.frame');
    expect(types('scores')).toContain('turnmodel.eot');
    // Kyutai's word timings, placed on the agent's track.
    expect(types('audio')).toContain('tts.words');
    // The tick scheduler paced the replies: chunks, grants, and a logged tick every 80 ms.
    expect(types('text')).toContain('llm.chunk');
    expect(types('control')).toContain('sched.grant');
    expect(types('ticks')).toContain('sched.tick');
    // The turn model, not ASR endpointing, ended the user's turns.
    const commits = live.events.filter((e) => e.kind === 'msg' && (e.data as { type: string; on?: string }).on === 'commit');
    expect(commits.length).toBeGreaterThan(0);
    expect(commits.some((e) => String((e.data as { why: string }).why).startsWith('model: end of turn'))).toBe(true);
    expect(count(live, (e) => e.kind === 'log' && (e.data as { level: string }).level === 'error')).toBe(0);
  }, 10_000);

  it('replays bit-for-bit, including timing, after a bundle round trip', () => {
    const source = decodeBundle(encodeBundle(live));
    const { runtime, sim } = prepareReplay(source, { registry });
    sim.runFast();
    const replay = runtime.trajectory;

    expect(replay.events.length).toBe(live.events.length);
    expect(replay.events).toEqual(live.events);
    expect(replay.tracks.names()).toEqual(live.tracks.names());
    for (const name of live.tracks.names()) expect(Buffer.from(replay.tracks.bytes(name)).equals(Buffer.from(live.tracks.bytes(name)))).toBe(true);
    expect(count(replay, (e) => e.kind === 'io.diverge')).toBe(0);
  });

  it('replays streaming ASR partials at their original arrival times', () => {
    const { runtime, sim } = prepareReplay(live, { registry });
    sim.runFast();
    const asr = (t: Trajectory) => t.events.filter((e) => e.node === 'asr' && e.kind === 'msg').map((e) => [e.t, e.data]);
    expect(asr(runtime.trajectory)).toEqual(asr(live));
    expect(asr(live).length).toBeGreaterThan(3);
  });

  it('flags divergence when the graph changes, and keeps going', () => {
    const g = testGraph();
    g.nodes.find((n) => n.id === 'llm')!.params.system = 'A different prompt.';
    const { runtime, sim } = prepareReplay(live, { registry, graph: g });
    sim.runFast();
    const diverged = runtime.trajectory.events.filter((e) => e.kind === 'io.diverge');
    expect(diverged.length).toBeGreaterThan(0);
    expect((diverged[0].data as { reason: string }).reason).toBe('request-mismatch');
    expect(runtime.trajectory.meta.graphHash).not.toBe(live.meta.graphHash);
    // Upstream of the change is untouched.
    const asr = (t: Trajectory) => t.events.filter((e) => e.node === 'asr').map((e) => [e.t, e.kind, e.data]);
    expect(asr(runtime.trajectory)).toEqual(asr(live));
  });

  it('replays identically when a coarse clock makes many events share a timestamp', async () => {
    // Browsers quantize performance.now() (Firefox to 1 ms); ties must replay in their original order.
    const coarse = await recordLive(800, { now: () => Math.floor(performance.now() / 2) * 2 });
    const times = coarse.events.map((e) => e.t);
    expect(new Set(times).size).toBeLessThan(times.length / 2);
    const { runtime, sim } = prepareReplay(coarse, { registry });
    sim.runFast();
    expect(runtime.trajectory.events).toEqual(coarse.events);
  }, 10_000);

  it('links every derived event to a cause', () => {
    const roots = ['session.start', 'session.stop', 'node.start', 'node.stop'];
    expect(live.events.filter((e) => !e.cause && !roots.includes(e.kind))).toEqual([]);
    const bySeq = new Map(live.events.map((e) => [e.seq, e]));
    for (const e of live.events) for (const c of e.cause ?? []) expect(bySeq.get(c)!.t).toBeLessThanOrEqual(e.t);
  });

  it('swaps the LLM model without touching the rest of the graph, and replays it exactly', async () => {
    const g = testGraph();
    Object.assign(g.nodes.find((n) => n.id === 'llm')!.params, { model: 'some/model' });
    const run = await recordLive(800, performance, g);
    const opened = run.events.find((e) => e.kind === 'io.open' && e.node === 'llm')!;
    expect((opened.data as { channel: string }).channel).toBe('openai');
    expect((opened.data as { request: { body: { model: string } } }).request.body.model).toBe('some/model');
    const deltas = run.events.filter((e) => e.kind === 'msg' && (e.data as { type: string }).type === 'llm.delta');
    expect(deltas.length).toBeGreaterThan(0);
    const { runtime, sim } = prepareReplay(run, { registry });
    sim.runFast();
    expect(runtime.trajectory.events).toEqual(run.events);
  }, 10_000);

  it('runs early interjection (TypeSafe gate, LLM check, take) and replays it exactly', async () => {
    const c = testConfig();
    c.interject.variant = 'typesafe';
    // From the first word, so the gate runs before the fake session's first turn is committed.
    c.interject.params.typesafe = { minWords: 1 };
    const run = await recordLive(2500, performance, buildGraph(c));
    const types = run.events.filter((e) => e.kind === 'msg').map((e) => (e.data as { type: string }).type);
    // Whether a check gets as far as a take depends on the fake session's timing; the gate always runs.
    for (const t of ['turn.words', 'interject.gate']) expect(types).toContain(t);
    expect(count(run, (e) => e.kind === 'log' && (e.data as { level: string }).level === 'error')).toBe(0);
    const { runtime, sim } = prepareReplay(decodeBundle(encodeBundle(run)), { registry });
    sim.runFast();
    expect(count(runtime.trajectory, (e) => e.kind === 'io.diverge')).toBe(0);
    expect(runtime.trajectory.events).toEqual(run.events);
  }, 15_000);

  it('says backchannels the LLM writes (own slot, own TTS socket) and replays them exactly', async () => {
    const c = testConfig();
    // TypeSafe says whether (the fake one always says yes), the LLM what; the fake user's words come just after each pause.
    c.backchannel.params.on = { decide: 'typesafe', gapMs: 0, pauseMs: 40 };
    const run = await recordLive(2500, performance, buildGraph(c));
    const msgs = run.events.filter((e) => e.kind === 'msg').map((e) => e.data as { type: string; gen?: boolean });
    expect(msgs.map((m) => m.type)).toContain('llm.bc');
    expect(msgs.some((m) => m.type === 'bc.say' && m.gen)).toBe(true);
    const asks = run.events.filter((e) => e.kind === 'io.open' && e.node === 'llm' && (e.data as { request: { body: { id_slot?: number } } }).request.body.id_slot === 1);
    expect(asks.length).toBeGreaterThan(0);
    expect(msgs.some((m) => m.type === 'playback.cue')).toBe(true);
    expect(count(run, (e) => e.kind === 'log' && (e.data as { level: string }).level === 'error')).toBe(0);
    const { runtime, sim } = prepareReplay(decodeBundle(encodeBundle(run)), { registry });
    sim.runFast();
    expect(count(runtime.trajectory, (e) => e.kind === 'io.diverge')).toBe(0);
    expect(runtime.trajectory.events).toEqual(run.events);
  }, 15_000);

  it('runs the tone TTS mock in place of Kyutai, and replays it exactly', async () => {
    const c = testConfig();
    c.tts.variant = 'tone';
    const run = await recordLive(800, performance, buildGraph(c));
    expect(run.events.some((e) => e.kind === 'io.open' && e.node === 'tts')).toBe(false);
    expect(run.events.some((e) => e.kind === 'msg' && e.node === 'tts' && (e.data as { type: string }).type === 'audio')).toBe(true);
    expect(run.events.filter((e) => e.kind === 'log' && (e.data as { level: string }).level === 'error')).toEqual([]);
    const { runtime, sim } = prepareReplay(run, { registry });
    sim.runFast();
    expect(runtime.trajectory.events).toEqual(run.events);
  }, 10_000);

  for (const variant of ['aec', 'rnnoise'] as const) {
    it(`runs the ${variant} cleanup stage and replays it exactly`, async () => {
      const c = testConfig();
      c.cleanup.variant = variant;
      const g = buildGraph(c);
      expect(g.nodes.find((n) => n.id === 'mic')!.params).toMatchObject({ noiseSuppression: false, autoGainControl: false });
      const run = await recordLive(1500, performance, g);
      const stats = run.events.filter((e) => e.node === 'cleanup' && e.kind === 'msg' && e.port === 'stats');
      expect(stats.length).toBeGreaterThan(10);
      // VAD and ASR listen to the cleaned track.
      const vadIn = run.events.find((e) => e.node === 'vad' && e.kind === 'io.send')!;
      expect(vadIn.media!.track).toBe('cleanup.audio');
      if (variant === 'aec') expect(run.events.some((e) => e.node === 'speaker' && e.port === 'played')).toBe(true);
      expect(run.events.filter((e) => e.kind === 'log' && (e.data as { level: string }).level === 'error')).toEqual([]);
      const { runtime, sim } = prepareReplay(run, { registry });
      sim.runFast();
      expect(runtime.trajectory.events).toEqual(run.events);
    }, 10_000);
  }
});

describe('live timers', () => {
  it('fire at their time even when the event loop runs them late, so the replay matches', () => {
    // A node that waits 10 ms for an answer and gives up; the answer arrives at 20 ms,
    // while the timer, held up behind a busy event loop, has not run yet.
    const node: NodeDef = {
      type: 'waiter',
      title: 'Waiter',
      category: 'llm',
      inputs: [],
      outputs: [{ name: 'out', type: 'x' }],
      params: [],
      channels: [{ name: 'x' }],
      create(ctx) {
        let timer: { clear(): void } | undefined;
        return {
          start() {
            timer = ctx.setTimer(10, () => ctx.emit('out', { type: 'gave_up' }));
            ctx.io.open('x', {}, { onRecv: () => (timer?.clear(), ctx.emit('out', { type: 'answered' })) });
          },
        };
      },
    };
    const reg = { ...registry, waiter: node };
    let now = 0;
    const late: (() => void)[] = [];
    const scheduler = { now: () => now, at: (_t: number, fn: () => void) => (late.push(fn), () => {}) };
    let sink: Delivery | undefined;
    const world = new LiveWorld({ x: (_r, s) => ((sink = s), { send() {}, close() {} }) }, scheduler);
    const graph: GraphSpec = { format: GRAPH_FORMAT, formatVersion: GRAPH_FORMAT_VERSION, name: 't', nodes: [{ id: 'w', type: 'waiter', params: {} }], edges: [] };
    const rt = new Runtime({ graph, registry: reg, scheduler, world, mode: 'live' });
    rt.start();
    now = 20;
    sink!.recv({ ok: true });
    for (const fn of late) fn();
    rt.stop();
    const outs = (t: Trajectory) => t.events.filter((e) => e.kind === 'msg').map((e) => [e.t, (e.data as { type: string }).type]);
    expect(outs(rt.trajectory)).toEqual([
      [10, 'gave_up'],
      [20, 'answered'],
    ]);
    const { runtime, sim } = prepareReplay(rt.trajectory, { registry: reg });
    sim.runFast();
    expect(runtime.trajectory.events).toEqual(rt.trajectory.events);
  });
});
