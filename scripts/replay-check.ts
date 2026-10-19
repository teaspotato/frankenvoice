// Replays recorded sessions as the app does (recorded graph and io) and reports
// io divergences and message counts that differ from the recording.
// Usage: pnpm replay-check path/to/*.fvs
import { readFileSync } from 'node:fs';
import { decodeBundle } from '../src/core/bundle';
import { prepareReplay } from '../src/core/replay';
import { registry } from '../src/nodes';
const key = (e: any) => (e.kind === 'msg' ? `${e.node}:${e.data?.type}` : '');
for (const f of process.argv.slice(2)) {
  const src = decodeBundle(new Uint8Array(readFileSync(f)));
  const r = prepareReplay(src, { registry });
  r.sim.runFast();
  const ev = r.runtime.trajectory.events;
  const div = ev.filter((e) => e.kind === 'io.diverge');
  const count = (es: any[]) => es.reduce((c, e) => { const k = key(e); if (k) c[k] = (c[k] ?? 0) + 1; return c; }, {} as Record<string, number>);
  const a = count(src.events), b = count(ev);
  const diff = [...new Set([...Object.keys(a), ...Object.keys(b)])].filter((k) => a[k] !== b[k]).map((k) => `${k} ${a[k] ?? 0}->${b[k] ?? 0}`);
  console.log(f.split('/').pop(), 'v' + src.meta.graph.formatVersion, 'io divergences', div.length, div.slice(0, 3).map((d: any) => `${d.data.xid} ${d.data.reason}`).join(', '), '| msg count diffs:', diff.join('; ') || 'none');
}
