import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';

export interface Layout {
  /** Height of the pipeline diagram, px. */
  diagram: number;
  /** Height of the event log, px. */
  log: number;
  /** Width of the side panel, px. */
  side: number;
  /** Whether the event log is expanded; collapsed by default. */
  logOpen: boolean;
  /** Whether the side panel (stage settings, sessions) is shown. */
  sideOpen: boolean;
}

const LAYOUT_KEY = 'frankenvoice.layout';
export const DEFAULT_LAYOUT: Layout = {
  diagram: 220,
  log: 220,
  side: 330,
  logOpen: false,
  sideOpen: true,
};
type Size = 'diagram' | 'log' | 'side';
const MIN: Record<Size, number> = { diagram: 80, log: 60, side: 220 };

/** Panel sizes, remembered in the browser. */
export function useLayout(): [Layout, (l: Layout) => void] {
  const [layout, setLayout] = useState<Layout>(() => {
    try {
      return {
        ...DEFAULT_LAYOUT,
        ...(JSON.parse(localStorage.getItem(LAYOUT_KEY) ?? '{}') as Partial<Layout>),
      };
    } catch {
      return DEFAULT_LAYOUT;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout));
    } catch {
      /* storage unavailable */
    }
  }, [layout]);
  return [layout, setLayout];
}

/**
 * A drag handle on one edge of a panel. Dragging changes `key` of the layout;
 * `sign` is +1 when dragging right/down grows the panel, -1 when it shrinks it.
 * Double-click restores the default size.
 */
export function Splitter({
  layout,
  onChange,
  field,
  axis,
  sign,
  max,
}: {
  layout: Layout;
  onChange(l: Layout): void;
  field: Size;
  axis: 'x' | 'y';
  sign: 1 | -1;
  /** Largest size allowed, given the window; read at drag start. */
  max(): number;
}) {
  const drag = useRef<{ start: number; size: number; max: number } | null>(null);
  const pos = (e: ReactPointerEvent) => (axis === 'x' ? e.clientX : e.clientY);
  return (
    <div
      className={`splitter splitter-${axis} splitter-${field}`}
      title="Drag to resize · double-click to reset"
      onPointerDown={(e) => {
        e.preventDefault();
        e.currentTarget.setPointerCapture(e.pointerId);
        drag.current = { start: pos(e), size: layout[field], max: max() };
      }}
      onPointerMove={(e) => {
        const d = drag.current;
        if (!d) return;
        const size = Math.round(Math.min(d.max, Math.max(MIN[field], d.size + sign * (pos(e) - d.start))));
        if (size !== layout[field]) onChange({ ...layout, [field]: size });
      }}
      onPointerUp={() => (drag.current = null)}
      onPointerCancel={() => (drag.current = null)}
      onDoubleClick={() => onChange({ ...layout, [field]: DEFAULT_LAYOUT[field] })}
    />
  );
}
