import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
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

export function EventLog({
  trajectory,
  live,
  open,
  onToggle,
  focus,
  tabs,
}: {
  trajectory?: Trajectory;
  live: boolean;
  /** Events of a clicked timeline item: scroll to and select the first, mark the rest. `n` changes on every click. */
  focus?: { seqs: number[]; n: number } | null;
  /** Collapsed, only the header bar shows and live updates are skipped. */
  open: boolean;
  onToggle(): void;
  /** Switches between the bottom panel's views, shown in place of the title. */
  tabs?: ReactNode;
}) {
  const [, setTick] = useState(0);
  const [hideNoisy, setHideNoisy] = useState(true);
  const [node, setNode] = useState('');
  const [filter, setFilter] = useState('');
  const [selected, setSelected] = useState<TrajEvent | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const scrollTo = useRef<number | null>(null);
  const focusSet = useMemo(() => new Set(focus?.seqs ?? []), [focus]);

  // Jump to a clicked timeline item's events, clearing filters that would hide them.
  useEffect(() => {
    if (!focus || !open || !trajectory) return;
    const first = trajectory.events.find((e) => e.seq === focus.seqs[0]);
    if (!first) return;
    setNode('');
    setFilter('');
    if (NOISY(first)) setHideNoisy(false);
    setSelected(first);
    stick.current = false;
    scrollTo.current = first.seq;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus?.n]);

  useEffect(() => {
    if (!trajectory || !live || !open) return;
    let raf = 0;
    const unsub = trajectory.subscribe(() => {
      if (!raf) raf = requestAnimationFrame(() => ((raf = 0), setTick((t) => t + 1)));
    });
    return () => {
      unsub();
      cancelAnimationFrame(raf);
    };
  }, [trajectory, live, open]);

  const nodes = useMemo(() => trajectory?.meta.graph.nodes.map((n) => n.id) ?? [], [trajectory]);
  const events = trajectory?.events ?? [];
  const shown = !open
    ? []
    : events.filter(
        (e) => (!hideNoisy || !NOISY(e)) && (!node || e.node === node) && (!filter || `${e.kind} ${e.port ?? ''} ${JSON.stringify(e.data)}`.includes(filter)),
      );
  // The newest 600 rows, or 600 around the selected event when it is older.
  const selIdx = selected ? shown.indexOf(selected) : -1;
  const visible = selIdx >= 0 && selIdx < shown.length - 600 ? shown.slice(Math.max(0, selIdx - 300), selIdx + 300) : shown.slice(-600);

  useEffect(() => {
    if (scrollTo.current !== null && bodyRef.current) {
      // The row may only appear on the next render, once cleared filters apply.
      const row = bodyRef.current.querySelector(`[data-seq="${scrollTo.current}"]`);
      if (row) {
        row.scrollIntoView({ block: 'center' });
        scrollTo.current = null;
      }
    } else if (stick.current && bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
  });

  return (
    <div className="event-log">
      <div className="event-log-bar">
        <button className="link event-log-toggle" onClick={onToggle} title={open ? 'Collapse' : 'Expand'}>
          {open ? '▾' : '▸'} {!tabs && <strong>Events</strong>}
        </button>
        {tabs}
        <span className="muted">{open ? `${shown.length} shown / ${events.length} total` : `${events.length} events`}</span>
        {open && (
          <>
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
          </>
        )}
      </div>
      {open && (
        <div
          className="event-log-body"
          ref={bodyRef}
          onScroll={(e) => {
            const el = e.currentTarget;
            stick.current = el.scrollTop + el.clientHeight >= el.scrollHeight - 8;
          }}
        >
          {visible.map((e) => (
            <div
              key={e.seq}
              data-seq={e.seq}
              className={`ev k-${e.kind.replace('.', '-')}${selected?.seq === e.seq ? ' sel' : ''}${focusSet.has(e.seq) ? ' focus' : ''}`}
              onClick={() => setSelected(e)}
            >
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
      )}
      {open && selected && (
        <pre className="event-detail" onClick={() => setSelected(null)}>
          {JSON.stringify(selected, null, 2)}
        </pre>
      )}
    </div>
  );
}
