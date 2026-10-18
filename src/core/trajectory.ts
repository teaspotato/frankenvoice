import { TrackStore } from './tracks';
import type { Json, TrajEvent } from './types';
import type { GraphSpec } from './graph';

export const TRAJECTORY_FORMAT = 'frankenvoice-trajectory';
export const TRAJECTORY_FORMAT_VERSION = 1;

export interface TrajectoryMeta {
  format: typeof TRAJECTORY_FORMAT;
  formatVersion: number;
  id: string;
  createdAt: string;
  mode: 'live' | 'replay';
  /** Session this one replayed, when mode is 'replay'. */
  replayOf?: string;
  graph: GraphSpec;
  graphHash: string;
  /** How each IO channel was served: live, or replayed from the recording. */
  channelModes?: Record<string, 'live' | 'replay'>;
  env?: Record<string, Json>;
}

/** Everything a session produced: metadata, the ordered event log, and binary tracks. */
export class Trajectory {
  readonly events: TrajEvent[] = [];
  readonly tracks = new TrackStore();
  private listeners = new Set<(ev: TrajEvent) => void>();

  constructor(public meta: TrajectoryMeta) {}

  append(ev: TrajEvent): void {
    this.events.push(ev);
    for (const l of this.listeners) l(ev);
  }

  subscribe(fn: (ev: TrajEvent) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  get duration(): number {
    return this.events.length ? this.events[this.events.length - 1].t : 0;
  }
}
