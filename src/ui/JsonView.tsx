import { useState } from 'react';
import type { Json } from '../core/types';

/** Strings longer than this start folded to their first line. */
const LONG_STRING = 160;

/**
 * A JSON value as a foldable tree: keys, strings, numbers and literals coloured apart, strings shown
 * with their real line breaks, and each object or array folded to a one-line preview past `open` levels.
 */
export function JsonView({ value, open = 2, label }: { value: Json | undefined; open?: number; label?: string }) {
  return (
    <div className="json">
      <Node k={label} v={value === undefined ? null : value} depth={0} open={open} />
    </div>
  );
}

function Node({ k, v, depth, open, last = true }: { k?: string; v: Json; depth: number; open: number; last?: boolean }) {
  const key = k !== undefined && <span className="j-key">{k}: </span>;
  const comma = last ? '' : ',';
  if (v === null || typeof v !== 'object') {
    return (
      <div className="j-row">
        {key}
        <Leaf v={v} />
        {comma}
      </div>
    );
  }
  return <Branch k={k} v={v} depth={depth} open={open} comma={comma} keyEl={key} />;
}

function Branch({
  v,
  depth,
  open,
  comma,
  keyEl,
}: {
  k?: string;
  v: Json[] | { [key: string]: Json };
  depth: number;
  open: number;
  comma: string;
  keyEl: React.ReactNode;
}) {
  const isArr = Array.isArray(v);
  const entries: [string | undefined, Json][] = isArr ? v.map((x) => [undefined, x]) : Object.entries(v);
  const [expanded, setExpanded] = useState(depth < open && entries.length > 0);
  const [o, c] = isArr ? ['[', ']'] : ['{', '}'];
  if (!entries.length)
    return (
      <div className="j-row">
        {keyEl}
        <span className="j-punct">
          {o}
          {c}
        </span>
        {comma}
      </div>
    );
  return (
    <div className="j-branch">
      <div className="j-row j-toggle" onClick={() => setExpanded(!expanded)}>
        <span className="j-caret">{expanded ? '▾' : '▸'}</span>
        {keyEl}
        <span className="j-punct">{o}</span>
        {!expanded && (
          <>
            <span className="j-preview">{preview(v)}</span>
            <span className="j-punct">{c}</span>
            {comma}
          </>
        )}
      </div>
      {expanded && (
        <>
          <div className="j-children">
            {entries.map(([ek, ev], i) => (
              <Node key={i} k={ek} v={ev} depth={depth + 1} open={open} last={i === entries.length - 1} />
            ))}
          </div>
          <div className="j-row">
            <span className="j-punct">{c}</span>
            {comma}
          </div>
        </>
      )}
    </div>
  );
}

function Leaf({ v }: { v: string | number | boolean | null }) {
  const [full, setFull] = useState(false);
  if (typeof v !== 'string') return <span className={v === null ? 'j-null' : typeof v === 'number' ? 'j-num' : 'j-bool'}>{String(v)}</span>;
  const long = v.length > LONG_STRING || v.includes('\n');
  if (!long || full)
    return (
      <span className="j-str" onClick={long ? () => setFull(false) : undefined}>
        "{v}"
      </span>
    );
  return (
    <span className="j-str j-folded" onClick={() => setFull(true)} title="Click to show all">
      "{v.split('\n')[0].slice(0, LONG_STRING)}…" <span className="j-more">+{v.length.toLocaleString()} chars</span>
    </span>
  );
}

/** One line standing in for a folded object: its type/role/text fields first, then the other keys. */
function preview(v: Json[] | { [key: string]: Json }): string {
  if (Array.isArray(v)) {
    // Content blocks and the like: show what they say.
    const texts = v.flatMap((x) => {
      if (typeof x === 'string') return [x];
      const o = x && typeof x === 'object' && !Array.isArray(x) ? x : {};
      const t = o.text ?? o.thinking ?? o.transcript;
      return typeof t === 'string' ? [o.type && o.type !== 'text' ? `[${String(o.type)}] ${t}` : t] : o.type ? [`[${String(o.type)}]`] : [];
    });
    if (texts.length) {
      const t = texts.join(' ');
      return ` "${t.length > 80 ? t.slice(0, 80) + '…' : t}"${v.length > 1 ? ` (${v.length} items)` : ''} `;
    }
    return ` ${v.length} item${v.length === 1 ? '' : 's'} `;
  }
  const parts: string[] = [];
  for (const f of ['type', 'role', 'name', 'text', 'thinking', 'transcript']) {
    const x = v[f];
    if (typeof x === 'string' && x) parts.push(f === 'type' || f === 'role' ? x : `"${x.length > 60 ? x.slice(0, 60) + '…' : x}"`);
  }
  const rest = Object.keys(v).filter((x) => !['type', 'role', 'name', 'text', 'thinking', 'transcript'].includes(x));
  if (rest.length) parts.push(rest.slice(0, 4).join(', ') + (rest.length > 4 ? ', …' : ''));
  return ` ${parts.join(' · ')} `;
}
