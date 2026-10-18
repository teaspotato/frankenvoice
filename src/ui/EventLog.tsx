import { useEffect, useMemo, useRef, useState } from 'react';
import type { Trajectory } from '../core/trajectory';
import type { TrajEvent } from '../core/types';

const NOISY = (e: TrajEvent) => {
  const type = (e.data as { type?: string } | null)?.type;
  if (e.kind === 'msg' && (type === 'audio' || type === 'vad.frame')) return true;
  if (e.kind === 'io.send' && e.media) return true;
  if (e.kind === 'io.recv' && (e.bin || /\/(mic|silero)#/.test((e.data as { xid?: string }).xid ?? ''))) return true;
  if (e.kind.startsWith('timer.')) return true;
  return false;
};

function summary(e: TrajEvent): string {
  const d = e.data as Record<string, unknown> | null | undefined;
  if (!d) return '';
  const pick = (o: Record<string, unknown>) => {
    const { xid: _x, ...rest } = o;
    return JSON.stringify(rest);
  };
  let s = e.kind === 'io.recv' || e.kind === 'io.send' ? pick(d) : JSON.stringify(d);
  if (e.media) s += `  ⟨${e.media.track} ${e.media.start}–${e.media.end}⟩`;
  if (e.bin) s += `  ⟨${e.bin.length} B⟩`;
  return s.length > 220 ? `${s.slice(0, 219)}…` : s;
}

export function EventLog({ trajectory, live }: { trajectory?: Trajectory; live: boolean }) {
  const [, setTick] = useState(0);
  const [hideNoisy, setHideNoisy] = useState(true);
  const [node, setNode] = useState('');
  const [filter, setFilter] = useState('');
  const [selected, setSelected] = useState<TrajEvent | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  useEffect(() => {
    if (!trajectory || !live) return;
    let raf = 0;
    const unsub = trajectory.subscribe(() => {
      if (!raf) raf = requestAnimationFrame(() => ((raf = 0), setTick((t) => t + 1)));
    });
    return () => {
      unsub();
      cancelAnimationFrame(raf);
    };
  }, [trajectory, live]);

  const nodes = useMemo(() => trajectory?.meta.graph.nodes.map((n) => n.id) ?? [], [trajectory]);
  const events = trajectory?.events ?? [];
  const shown = events.filter(
    (e) => (!hideNoisy || !NOISY(e)) && (!node || e.node === node) && (!filter || `${e.kind} ${e.port ?? ''} ${JSON.stringify(e.data)}`.includes(filter)),
  );
  const visible = shown.slice(-600);

  useEffect(() => {
    if (stick.current && bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
  });

  return (
    <div className="event-log">
      <div className="event-log-bar">
        <strong>Events</strong>
        <span className="muted">
          {shown.length} shown / {events.length} total
        </span>
        <select value={node} onChange={(e) => setNode(e.target.value)}>
          <option value="">all nodes</option>
          <option value="$runtime">$runtime</option>
          {nodes.map((n) => (
            <option key={n}>{n}</option>
          ))}
        </select>
        <input placeholder="filter text" value={filter} onChange={(e) => setFilter(e.target.value)} />
        <label>
          <input type="checkbox" checked={hideNoisy} onChange={(e) => setHideNoisy(e.target.checked)} /> hide frames & timers
        </label>
      </div>
      <div
        className="event-log-body"
        ref={bodyRef}
        onScroll={(e) => {
          const el = e.currentTarget;
          stick.current = el.scrollTop + el.clientHeight >= el.scrollHeight - 8;
        }}
      >
        {visible.map((e) => (
          <div key={e.seq} className={`ev k-${e.kind.replace('.', '-')}${selected?.seq === e.seq ? ' sel' : ''}`} onClick={() => setSelected(e)}>
            <span className="t">{e.t.toFixed(1)}</span>
            <span className="seq">#{e.seq}</span>
            <span className="node">{e.node}</span>
            <span className="kind">
              {e.kind}
              {e.port ? `:${e.port}` : ''}
            </span>
            <span className="data">{summary(e)}</span>
          </div>
        ))}
      </div>
      {selected && (
        <pre className="event-detail" onClick={() => setSelected(null)}>
          {JSON.stringify(selected, null, 2)}
        </pre>
      )}
    </div>
  );
}
