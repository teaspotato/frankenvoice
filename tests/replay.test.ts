import { describe, expect, it } from 'vitest';
import { Runtime } from '../src/core/runtime';
import { LiveScheduler } from '../src/core/scheduler';
import { LiveWorld } from '../src/core/world';
import { prepareReplay } from '../src/core/replay';
import { decodeBundle, encodeBundle } from '../src/core/bundle';
import { defaultGraph, registry } from '../src/nodes';
import type { Trajectory } from '../src/core/trajectory';
import type { GraphSpec } from '../src/core/graph';
import { fakeDrivers } from './fake-drivers';

function testGraph(): GraphSpec {
  const g = defaultGraph();
  // Also run the deterministic energy VAD alongside Silero.
  g.nodes.push({ id: 'vad2', type: 'vad.energy', params: {} });
  g.edges.push({ from: { node: 'mic', port: 'audio' }, to: { node: 'vad2', port: 'audio' } });
  return g;
}

async function recordLive(ms: number, perf: { now(): number } = performance): Promise<Trajectory> {
  const scheduler = new LiveScheduler(perf);
  const rt = new Runtime({ graph: testGraph(), registry, scheduler, world: new LiveWorld(fakeDrivers, scheduler), mode: 'live' });
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
});
