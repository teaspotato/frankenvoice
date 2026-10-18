import { stableStringify } from './json';
import type { Trajectory } from './trajectory';
import type { TrajEvent } from './types';

export interface Comparison {
  identical: boolean;
  /** Events compared before the first difference. */
  matched: number;
  total: [number, number];
  firstDiff?: { index: number; a?: TrajEvent; b?: TrajEvent };
  tracksIdentical: boolean;
}

/** Compare two trajectories event by event (time, node, payload, causes) and track by track. */
export function compareTrajectories(a: Trajectory, b: Trajectory): Comparison {
  const n = Math.min(a.events.length, b.events.length);
  let i = 0;
  while (i < n && stableStringify(a.events[i]) === stableStringify(b.events[i])) i++;
  const sameLength = a.events.length === b.events.length;
  const names = new Set([...a.tracks.names(), ...b.tracks.names()]);
  let tracksIdentical = true;
  for (const name of names) {
    if (!a.tracks.names().includes(name) || !b.tracks.names().includes(name)) {
      tracksIdentical = false;
      break;
    }
    const x = a.tracks.bytes(name);
    const y = b.tracks.bytes(name);
    if (x.length !== y.length || x.some((v, k) => v !== y[k])) {
      tracksIdentical = false;
      break;
    }
  }
  const identical = i === n && sameLength;
  return {
    identical: identical && tracksIdentical,
    matched: i,
    total: [a.events.length, b.events.length],
    ...(identical ? {} : { firstDiff: { index: i, a: a.events[i], b: b.events[i] } }),
    tracksIdentical,
  };
}
