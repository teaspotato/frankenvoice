import type { GraphSpec } from './graph';
import type { NodeRegistry } from './node';
import { Runtime } from './runtime';
import { SimScheduler } from './scheduler';
import type { Trajectory } from './trajectory';
import { ReplayWorld, type ReplayOptions } from './world';

export interface ReplaySetup extends ReplayOptions {
  registry: NodeRegistry;
  /** Run a different graph against the recorded world (default: the recorded graph). */
  graph?: GraphSpec;
}

/**
 * Prepare a replay of `source`: a runtime on a virtual clock whose world serves
 * every recorded IO exchange and timer at its original time, and which stops
 * when the original session stopped. Call `sim.runFast()` or `sim.runRealtime()`.
 */
export function prepareReplay(source: Trajectory, setup: ReplaySetup): { runtime: Runtime; sim: SimScheduler } {
  const sim = new SimScheduler();
  const world = new ReplayWorld(source, sim, setup);
  const runtime = new Runtime({
    graph: setup.graph ?? source.meta.graph,
    registry: setup.registry,
    scheduler: sim,
    world,
    mode: 'replay',
    replayOf: source.meta.id,
  });
  const startEv = source.events.find((e) => e.kind === 'session.start');
  const stopEv = [...source.events].reverse().find((e) => e.kind === 'session.stop');
  sim.at(startEv?.t ?? 0, () => runtime.start(), -1);
  sim.at(stopEv ? stopEv.t : source.duration, () => runtime.stop(), stopEv ? stopEv.seq : undefined);
  return { runtime, sim };
}
