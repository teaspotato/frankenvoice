// A session bundle is a zip:
//   meta.json          TrajectoryMeta plus the track table
//   events.jsonl       one TrajEvent per line, in seq order
//   tracks/<name>.bin  raw track bytes (float32 LE PCM for media, payload bytes for IO)

import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';
import { Trajectory, TRAJECTORY_FORMAT, type TrajectoryMeta } from './trajectory';
import type { TrackInfo } from './tracks';
import { checkGraph } from './graph';
import type { TrajEvent } from './types';

interface BundleMeta extends TrajectoryMeta {
  tracks: Record<string, TrackInfo & { file: string }>;
  eventCount: number;
}

export function encodeBundle(traj: Trajectory): Uint8Array {
  const files: Record<string, Uint8Array> = {};
  const tracks: BundleMeta['tracks'] = {};
  const info = traj.tracks.info();
  traj.tracks.names().forEach((name, i) => {
    const file = `tracks/${String(i).padStart(3, '0')}.bin`;
    tracks[name] = { ...info[name], file };
    files[file] = traj.tracks.bytes(name);
  });
  const meta: BundleMeta = { ...traj.meta, tracks, eventCount: traj.events.length };
  files['meta.json'] = strToU8(JSON.stringify(meta, null, 2));
  files['events.jsonl'] = strToU8(traj.events.map((e) => JSON.stringify(e)).join('\n') + '\n');
  return zipSync(files, { level: 1 });
}

export function decodeBundle(bytes: Uint8Array): Trajectory {
  const files = unzipSync(bytes);
  if (!files['meta.json'] || !files['events.jsonl']) throw new Error('not a frankenvoice session bundle');
  const meta = JSON.parse(strFromU8(files['meta.json'])) as BundleMeta;
  if (meta.format !== TRAJECTORY_FORMAT) throw new Error(`unexpected format ${meta.format}`);
  checkGraph(meta.graph);
  const { tracks, eventCount: _n, ...trajMeta } = meta;
  const traj = new Trajectory(trajMeta);
  for (const [name, t] of Object.entries(tracks)) {
    const { file, ...info } = t;
    traj.tracks.load(name, info, files[file] ?? new Uint8Array());
  }
  for (const line of strFromU8(files['events.jsonl']).split('\n')) {
    if (line) traj.events.push(JSON.parse(line) as TrajEvent);
  }
  return traj;
}
