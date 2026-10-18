import { useEffect, useRef, useState, type ReactNode } from 'react';
import type { Trajectory } from '../core/trajectory';
import { TimelineModel } from './timeline';

export interface Lane {
  id: string;
  label: string;
  height: number;
}

/** One lane per pipeline stage, top to bottom in pipeline order. */
export const LANES: Lane[] = [
  { id: 'mic', label: 'Mic', height: 48 },
  { id: 'cleanup', label: 'Cleaned', height: 52 },
  { id: 'vad', label: 'VAD', height: 60 },
  { id: 'asr', label: 'ASR', height: 46 },
  { id: 'turn', label: 'Turn', height: 36 },
  { id: 'llm', label: 'LLM', height: 60 },
  { id: 'tts', label: 'TTS', height: 36 },
  { id: 'speaker', label: 'Playback', height: 48 },
];

const AXIS = 20;
const LABEL_W = 78;
const MIN_SPAN = 200;

interface Hit {
  lane: string;
  x0: number;
  x1: number;
  text: string;
  /** Vertical extent within the canvas, when only part of the lane counts. */
  y0?: number;
  y1?: number;
}

interface View {
  /** Right edge of the window in session ms; null follows the newest data. */
  end: number | null;
  span: number;
}

function css(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
}

function niceStep(span: number, px: number): number {
  const target = (span / px) * 90;
  const steps = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 15000, 30000, 60000, 120000, 300000];
  return steps.find((s) => s >= target) ?? 600000;
}

function fmt(ms: number, step: number) {
  const s = ms / 1000;
  return step < 1000 ? `${s.toFixed(step < 100 ? 2 : 1)}s` : s >= 60 ? `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}` : `${s.toFixed(0)}s`;
}

/**
 * Time-aligned view of every stage: mic waveform, VAD probability, ASR results
 * (placed on the audio they cover, with ticks where they arrived), turn-taking
 * state, LLM generation, TTS segments, and what was played.
 */
export function Dashboard({
  trajectory,
  running,
  now,
  selected,
  onSelect,
  transport,
}: {
  trajectory?: Trajectory;
  running: boolean;
  /** Current session time while running (between events). */
  now?: () => number;
  selected: string | null;
  onSelect(lane: string | null): void;
  /** Play/pause/stop for the shown session. */
  transport: ReactNode;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<TimelineModel | null>(null);
  const hitsRef = useRef<Hit[]>([]);
  const [view, setView] = useState<View>({ end: null, span: 15000 });
  const viewRef = useRef(view);
  viewRef.current = view;
  const [size, setSize] = useState({ w: 800, h: 300 });
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null);
  const [tick, setTick] = useState(0);
  const drag = useRef<{ x: number; end: number } | null>(null);

  const totalH = AXIS + LANES.reduce((a, l) => a + l.height, 0);

  if (trajectory && modelRef.current?.traj !== trajectory) modelRef.current = new TimelineModel(trajectory);
  if (!trajectory) modelRef.current = null;

  // A new session starts following again.
  useEffect(() => setView((v) => ({ ...v, end: null })), [trajectory]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Redraw every frame while running; otherwise when something changes.
  useEffect(() => {
    if (!running) return;
    let raf = 0;
    const loop = () => {
      setTick((t) => t + 1);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [running]);

  const window_ = (m: TimelineModel | null): [number, number] => {
    const head = Math.max(m?.duration ?? 0, running && now ? now() : 0);
    // Never show time before the session started.
    const end = Math.max(view.end ?? head, view.span);
    return [end - view.span, end];
  };

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const m = modelRef.current;
    m?.update();
    const dpr = window.devicePixelRatio || 1;
    const W = size.w;
    const H = Math.max(totalH, 0);
    if (c.width !== W * dpr || c.height !== H * dpr) {
      c.width = W * dpr;
      c.height = H * dpr;
      c.style.width = `${W}px`;
      c.style.height = `${H}px`;
    }
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const col = {
      bg: css('--bg'),
      panel: css('--panel'),
      text: css('--text'),
      muted: css('--muted'),
      border: css('--border'),
      accent: css('--accent'),
      danger: css('--danger'),
      ok: css('--ok'),
      warn: css('--warn'),
      mic: css('--cat-source'),
      cleanup: css('--cat-cleanup'),
      vad: css('--cat-vad'),
      asr: css('--cat-asr'),
      turn: css('--cat-control'),
      llm: css('--cat-llm'),
      tts: css('--cat-tts'),
      sink: css('--cat-sink'),
    };
    g.clearRect(0, 0, W, H);
    g.fillStyle = col.panel;
    g.fillRect(0, 0, W, H);
    const hits: Hit[] = [];
    hitsRef.current = hits;
    const [t0, t1] = window_(m);
    const plotW = Math.max(1, W - LABEL_W);
    const x = (t: number) => LABEL_W + ((t - t0) / (t1 - t0)) * plotW;
    const tAt = (px: number) => t0 + ((px - LABEL_W) / plotW) * (t1 - t0);
    g.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    g.textBaseline = 'middle';

    // lanes
    const laneY: Record<string, [number, number]> = {};
    let y = AXIS;
    for (const l of LANES) {
      laneY[l.id] = [y, y + l.height];
      if (l.id === selected) {
        g.fillStyle = col.bg;
        g.fillRect(0, y, W, l.height);
      }
      g.strokeStyle = col.border;
      g.beginPath();
      g.moveTo(0, y + l.height + 0.5);
      g.lineTo(W, y + l.height + 0.5);
      g.stroke();
      g.fillStyle = l.id === selected ? col.accent : col.muted;
      g.fillText(l.label, 8, y + l.height / 2);
      y += l.height;
    }

    // time grid
    const step = niceStep(t1 - t0, plotW);
    g.fillStyle = col.muted;
    g.strokeStyle = col.border;
    for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
      const px = Math.round(x(t)) + 0.5;
      g.beginPath();
      g.moveTo(px, AXIS - 4);
      g.lineTo(px, H);
      g.globalAlpha = 0.5;
      g.stroke();
      g.globalAlpha = 1;
      if (t >= 0) g.fillText(fmt(t, step), px + 3, AXIS / 2);
    }
    g.strokeStyle = col.border;
    g.beginPath();
    g.moveTo(LABEL_W + 0.5, 0);
    g.lineTo(LABEL_W + 0.5, H);
    g.stroke();

    g.save();
    g.beginPath();
    g.rect(LABEL_W + 1, 0, plotW, H);
    g.clip();

    const label = (text: string, x0: number, x1: number, yc: number, color: string) => {
      const room = x1 - x0 - 6;
      if (room < 12 || !text) return;
      let s = text;
      while (s.length > 1 && g.measureText(s).width > room) s = s.slice(0, Math.max(1, Math.floor(s.length * (room / g.measureText(s).width)) - 1));
      if (s !== text) s = s.slice(0, -1) + '…';
      g.fillStyle = color;
      g.fillText(s, Math.max(x0, LABEL_W) + 3, yc);
    };
    const span = (lane: string, a: number, b: number, color: string, alpha: number, text: string, top = 0.2, bot = 0.8) => {
      const [ly0, ly1] = laneY[lane];
      const h = ly1 - ly0;
      const x0 = x(a);
      const x1 = Math.max(x(b), x0 + 2);
      g.globalAlpha = alpha;
      g.fillStyle = color;
      g.fillRect(x0, ly0 + h * top, x1 - x0, h * (bot - top));
      g.globalAlpha = 1;
      label(text, x0, x1, ly0 + (h * (top + bot)) / 2, col.text);
      hits.push({ lane, x0, x1, text });
    };
    const vline = (lane: string, t: number, color: string, from = 0, to = 1) => {
      const [ly0, ly1] = laneY[lane];
      const px = Math.round(x(t)) + 0.5;
      g.strokeStyle = color;
      g.beginPath();
      g.moveTo(px, ly0 + (ly1 - ly0) * from);
      g.lineTo(px, ly0 + (ly1 - ly0) * to);
      g.stroke();
    };
    const head = Math.max(m?.duration ?? 0, running && now ? now() : 0);

    if (m) {
      // speech regions behind mic + vad
      for (const s of m.speech) {
        const a = x(s.t0);
        const b = x(s.t1 ?? head);
        if (b < LABEL_W || a > W) continue;
        g.fillStyle = col.vad;
        g.globalAlpha = 0.12;
        g.fillRect(a, laneY.mic[0], b - a, laneY.vad[1] - laneY.mic[0]);
        g.globalAlpha = 1;
        hits.push({ lane: 'vad', x0: a, x1: b, text: `speech ${((((s.t1 ?? head) - s.t0) / 1000) as number).toFixed(2)} s` });
      }

      // mic waveform, and what the noise/echo stage made of it
      const wave = (lane: string, peaks: number[], color: string) => {
        const [ly0, ly1] = laneY[lane];
        const mid = (ly0 + ly1) / 2;
        const half = (ly1 - ly0) / 2 - 3;
        const n = peaks.length;
        let lo = 0;
        let hi = n;
        while (lo < hi) {
          const k = (lo + hi) >> 1;
          if (m.peakToT(k) < t0) lo = k + 1;
          else hi = k;
        }
        const cols = new Float32Array(Math.ceil(plotW) + 2);
        for (let i = Math.max(0, lo - 1); i < n; i++) {
          const t = m.peakToT(i);
          if (t > t1) break;
          const c = Math.floor(x(t) - LABEL_W);
          if (c >= 0 && c < cols.length) cols[c] = Math.max(cols[c], peaks[i]);
        }
        g.fillStyle = color;
        for (let c = 0; c < cols.length; c++) {
          if (!cols[c]) continue;
          const h = Math.max(1, Math.min(1, Math.sqrt(cols[c])) * half);
          g.fillRect(LABEL_W + c, mid - h, 1, h * 2);
        }
      };
      wave('mic', m.micPeaks, col.mic);
      wave('cleanup', m.cleanPeaks, col.cleanup);
      // reduction (in − out, 0–30 dB) and double-talk / gating
      {
        const [ly0, ly1] = laneY.cleanup;
        const yr = (dbr: number) => ly1 - 3 - (Math.max(0, Math.min(30, dbr)) / 30) * (ly1 - ly0 - 6);
        g.strokeStyle = col.text;
        g.globalAlpha = 0.7;
        g.beginPath();
        let started = false;
        for (const f of m.cleanup) {
          if (f.t < t0 - 100 || f.t > t1 + 100) continue;
          const px = x(f.t);
          const py = yr(f.inDb - f.outDb);
          if (!started) g.moveTo(px, py);
          else g.lineTo(px, py);
          started = true;
        }
        g.stroke();
        g.globalAlpha = 1;
        g.fillStyle = col.danger;
        for (const f of m.cleanup) if ((f.doubleTalk || f.gated) && f.t >= t0 && f.t <= t1) g.fillRect(x(f.t) - 1, ly1 - 4, 2, 3);
      }

      // VAD probability
      {
        const [ly0, ly1] = laneY.vad;
        const yv = (p: number) => ly1 - 4 - p * (ly1 - ly0 - 8);
        g.setLineDash([3, 3]);
        g.strokeStyle = col.muted;
        for (const th of [m.vadThreshold, m.vadNegThreshold]) {
          if (typeof th !== 'number') continue;
          g.beginPath();
          g.moveTo(LABEL_W, yv(th) + 0.5);
          g.lineTo(W, yv(th) + 0.5);
          g.stroke();
        }
        g.setLineDash([]);
        g.strokeStyle = col.vad;
        g.lineWidth = 1.5;
        g.beginPath();
        let started = false;
        for (const f of m.vad) {
          if (f.t < t0 - 100 || f.t > t1 + 100) continue;
          if (!started) g.moveTo(x(f.t), yv(f.p));
          else g.lineTo(x(f.t), yv(f.p));
          started = true;
        }
        g.stroke();
        g.lineWidth = 1;
      }

      // ASR: results on the audio they cover; ticks where they arrived
      for (const r of m.asr.values()) {
        if (r.a1 < t0 || r.a0 > t1) {
          if (r.t < t0 || r.t > t1) continue;
        }
        const [ly0, ly1] = laneY.asr;
        if (r.final) span('asr', r.a0, r.a1, col.asr, 0.3, r.text, 0.08, 0.62);
        else {
          const x0 = x(r.a0);
          const x1 = Math.max(x(r.a1), x0 + 2);
          g.strokeStyle = col.asr;
          g.setLineDash([3, 2]);
          g.strokeRect(x0 + 0.5, ly0 + (ly1 - ly0) * 0.08 + 0.5, x1 - x0 - 1, (ly1 - ly0) * 0.54);
          g.setLineDash([]);
          label(r.text, x0, x1, ly0 + (ly1 - ly0) * 0.35, col.muted);
          hits.push({ lane: 'asr', x0, x1, text: `${r.text} (partial)` });
        }
        // latency: end of the audio to arrival of this result
        g.strokeStyle = col.asr;
        g.globalAlpha = 0.5;
        g.beginPath();
        g.moveTo(x(r.a1), ly0 + (ly1 - ly0) * 0.82);
        g.lineTo(x(r.t), ly0 + (ly1 - ly0) * 0.82);
        g.stroke();
        g.globalAlpha = 1;
        hits.push({ lane: 'asr', x0: x(r.a1), x1: x(r.t), text: `${r.final ? 'final' : 'partial'} arrived ${(r.t - r.a1).toFixed(0)} ms after its audio ended` });
      }
      for (const a of m.asrArrivals) {
        if (a.t < t0 || a.t > t1) continue;
        vline('asr', a.t, a.final ? col.asr : col.muted, a.final ? 0.68 : 0.74, 0.96);
      }

      // turn-taking state
      const floorColor: Record<string, string> = { thinking: col.warn, speaking: col.accent };
      m.floor.forEach((f, i) => {
        const end = m.floor[i + 1]?.t ?? head;
        if (end < t0 || f.t > t1) return;
        if (floorColor[f.state]) span('turn', f.t, end, floorColor[f.state], 0.25, f.state, 0.15, 0.85);
        else hits.push({ lane: 'turn', x0: x(f.t), x1: x(end), text: f.state });
      });
      for (const u of m.userTurns) {
        if (u.t < t0 || u.t > t1) continue;
        vline('turn', u.t, col.turn);
        hits.push({ lane: 'turn', x0: x(u.t) - 3, x1: x(u.t) + 3, text: `user turn: ${u.text}` });
      }

      // LLM: waiting for the first token, then generating; below, each chunk at its arrival
      for (const l of m.llm) {
        const end = l.t1 ?? head;
        if (end < t0 || l.t0 > t1) continue;
        const [ly0, ly1] = laneY.llm;
        const h = ly1 - ly0;
        const ft = l.firstToken ?? end;
        const dur = (end - ft) / 1000;
        const stats = `${l.firstToken ? `TTFT ${(l.firstToken - l.t0).toFixed(0)} ms` : 'waiting…'} · ${l.chunks.length} chunks${dur > 0.05 ? ` · ${(l.text.length / dur).toFixed(0)} ch/s` : ''}${l.status ? ` · ${l.status}` : ''}`;
        span('llm', l.t0, ft, col.llm, 0.15, '', 0.06, 0.4);
        hits.pop();
        span('llm', ft, end, col.llm, 0.35, stats, 0.06, 0.4);
        hits.pop();
        hits.push({ lane: 'llm', x0: x(l.t0), x1: x(end), text: `${stats}\n${l.text}` });
        let so = '';
        l.chunks.forEach((c, i) => {
          so += c.text;
          const cx = x(c.t);
          const nx = i + 1 < l.chunks.length ? x(l.chunks[i + 1].t) : x(end) + 200;
          if (nx < LABEL_W || cx > W) return;
          vline('llm', c.t, col.llm, 0.46, 0.56);
          if (nx - cx > 8) {
            g.save();
            g.beginPath();
            g.rect(cx + 1, ly0 + h * 0.5, nx - cx - 2, h * 0.5);
            g.clip();
            g.fillStyle = col.text;
            g.fillText(c.text.replace(/\s+/g, ' '), cx + 2, ly0 + h * 0.76);
            g.restore();
          }
          hits.push({ lane: 'llm', y0: ly0 + h * 0.44, y1: ly1, x0: cx - 1, x1: Math.max(cx + 3, nx - 1), text: `+${(c.t - l.t0).toFixed(0)} ms  ${JSON.stringify(c.text)}\n${so}` });
        });
      }

      // TTS: request sent, then audio streaming per sentence
      for (const t of m.ttsRequests) if (t >= t0 && t <= t1) vline('tts', t, col.muted, 0.75, 1);
      for (const s of m.tts) {
        const end = s.t1 ?? head;
        if (end < t0 || s.t0 > t1) continue;
        span('tts', s.t0, end, col.tts, 0.35, s.text, 0.15, 0.7);
      }

      // playback: what was heard
      for (const p of m.playback) {
        const end = p.t1 ?? head;
        if (end < t0 || p.t0 > t1) continue;
        span('speaker', p.t0, end, col.sink, 0.1, '', 0, 1);
        hits.pop();
        if (p.interrupted) vline('speaker', end, col.danger);
      }
      {
        const [ly0, ly1] = laneY.speaker;
        const mid = (ly0 + ly1) / 2;
        const half = (ly1 - ly0) / 2 - 3;
        g.fillStyle = col.sink;
        const dt = m.peakMs;
        for (const b of m.agentAudio) {
          if (b.t > t1 || b.t + b.peaks.length * dt < t0) continue;
          for (let i = 0; i < b.peaks.length; i++) {
            const px = x(b.t + i * dt);
            if (px < LABEL_W || px > W) continue;
            const h = Math.max(1, Math.min(1, Math.sqrt(b.peaks[i])) * half);
            g.fillRect(px, mid - h, Math.max(1, x(b.t + (i + 1) * dt) - px), h * 2);
          }
        }
      }
      for (const s of m.playbackSegments) {
        if (s.t < t0 || s.t > t1) continue;
        vline('speaker', s.t, col.text, 0, 0.3);
        hits.push({ lane: 'speaker', x0: x(s.t) - 3, x1: x(s.t) + 60, text: s.text });
      }
      for (const b of m.boundaries) if (b >= t0 && b <= t1) vline('speaker', b, col.sink, 0.7, 1);

      for (const err of m.errors) {
        const lane = laneY[err.node] ? err.node : 'mic';
        if (err.t < t0 || err.t > t1) continue;
        const [ly0] = laneY[lane];
        g.fillStyle = col.danger;
        g.beginPath();
        g.arc(x(err.t), ly0 + 6, 4, 0, Math.PI * 2);
        g.fill();
        hits.push({ lane, x0: x(err.t) - 5, x1: x(err.t) + 5, text: `error: ${err.text}` });
      }

      // live head
      if (running) {
        g.strokeStyle = col.danger;
        g.globalAlpha = 0.6;
        g.beginPath();
        g.moveTo(Math.round(x(head)) + 0.5, AXIS);
        g.lineTo(Math.round(x(head)) + 0.5, H);
        g.stroke();
        g.globalAlpha = 1;
      }
    }

    // cursor
    if (hover && hover.x > LABEL_W) {
      g.strokeStyle = col.text;
      g.globalAlpha = 0.35;
      g.beginPath();
      g.moveTo(hover.x + 0.5, AXIS);
      g.lineTo(hover.x + 0.5, H);
      g.stroke();
      g.globalAlpha = 1;
      // Opaque backing so the cursor's time doesn't overprint the axis labels.
      const label = fmt(tAt(hover.x), 10);
      g.fillStyle = col.panel;
      g.fillRect(hover.x + 1, 0, g.measureText(label).width + 8, AXIS - 1);
      g.fillStyle = col.text;
      g.fillText(label, hover.x + 4, AXIS / 2);
    }
    g.restore();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick, view, size, hover, selected, trajectory, running, totalH]);

  // ---- interaction --------------------------------------------------------

  const plotW = () => Math.max(1, size.w - LABEL_W);
  const currentEnd = () => window_(modelRef.current)[1];

  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    const rect = canvasRef.current!.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const end = currentEnd();
    if (e.ctrlKey || e.metaKey || Math.abs(e.deltaY) > Math.abs(e.deltaX) * 2) {
      // zoom around the cursor
      const f = Math.exp(e.deltaY * 0.0015);
      const span = Math.max(MIN_SPAN, Math.min(3_600_000, view.span * f));
      const tCursor = end - view.span + ((px - LABEL_W) / plotW()) * view.span;
      const frac = (px - LABEL_W) / plotW();
      const newEnd = tCursor + (1 - frac) * span;
      setView({ span, end: view.end === null && frac > 0.95 ? null : newEnd });
    } else {
      setView({ span: view.span, end: end + (e.deltaX / plotW()) * view.span });
    }
  };

  // React registers wheel listeners as passive; zooming must stop the page from scrolling.
  const wheelRef = useRef(onWheel);
  wheelRef.current = onWheel;
  useEffect(() => {
    const c = canvasRef.current!;
    const h = (e: WheelEvent) => wheelRef.current(e);
    c.addEventListener('wheel', h, { passive: false });
    return () => c.removeEventListener('wheel', h);
  }, []);

  const onMouseDown = (e: React.MouseEvent) => {
    drag.current = { x: e.clientX, end: currentEnd() };
  };
  const onMouseMove = (e: React.MouseEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    setHover({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    if (drag.current && e.buttons & 1) {
      const dx = e.clientX - drag.current.x;
      if (Math.abs(dx) > 2) setView({ span: view.span, end: drag.current.end - (dx / plotW()) * view.span });
    }
  };
  const onMouseUp = (e: React.MouseEvent) => {
    const moved = drag.current && Math.abs(e.clientX - drag.current.x) > 2;
    drag.current = null;
    if (moved) return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const py = e.clientY - rect.top;
    let y = AXIS;
    for (const l of LANES) {
      if (py >= y && py < y + l.height) return onSelect(l.id === selected ? null : l.id);
      y += l.height;
    }
  };
  const fit = () => {
    const d = modelRef.current?.duration ?? 0;
    setView({ span: Math.max(MIN_SPAN, d * 1.02), end: running ? null : d * 1.01 });
  };

  let tip: Hit | undefined;
  let tipLane: string | undefined;
  if (hover) {
    let y = AXIS;
    for (const l of LANES) {
      if (hover.y >= y && hover.y < y + l.height) tipLane = l.id;
      y += l.height;
    }
    tip = [...hitsRef.current]
      .reverse()
      .find((h) => h.lane === tipLane && hover.x >= h.x0 - 1 && hover.x <= h.x1 + 1 && (h.y0 === undefined || (hover.y >= h.y0 && hover.y <= h.y1!)));
  }
  let cleanAt: string | undefined;
  if (hover && tipLane === 'cleanup' && modelRef.current) {
    const t = currentEnd() - view.span + ((hover.x - LABEL_W) / plotW()) * view.span;
    let best: (typeof modelRef.current.cleanup)[number] | undefined;
    for (const f of modelRef.current.cleanup) if (!best || Math.abs(f.t - t) < Math.abs(best.t - t)) best = f;
    if (best && Math.abs(best.t - t) < 100) {
      const r = best.inDb - best.outDb;
      cleanAt = `in ${best.inDb} dB → out ${best.outDb} dB (${r >= 0 ? '−' : '+'}${Math.abs(r).toFixed(1)} dB)`;
      if (best.refDb !== undefined) cleanAt += `\necho reference ${best.refDb} dB${best.doubleTalk ? ' · double talk: not adapting' : ''}`;
      if (best.vad !== undefined) cleanAt += `\nRNNoise voice prob. ${best.vad}${best.gated ? ' · gated' : ''}`;
    }
  }
  let vadAt: string | undefined;
  if (hover && tipLane === 'vad' && modelRef.current) {
    const t = currentEnd() - view.span + ((hover.x - LABEL_W) / plotW()) * view.span;
    const fr = modelRef.current.vad;
    let best = -1;
    let bd = Infinity;
    for (let i = 0; i < fr.length; i++) {
      const dd = Math.abs(fr[i].t - t);
      if (dd < bd) {
        bd = dd;
        best = i;
      }
    }
    if (best >= 0 && bd < 100) vadAt = `p = ${fr[best].p.toFixed(3)}`;
  }

  return (
    <div className="dashboard">
      <div className="dashboard-bar">
        {transport}
        <span className="muted">
          {trajectory ? (trajectory.meta.mode === 'replay' ? `replay of ${trajectory.meta.replayOf}` : running ? 'live' : trajectory.meta.id) : 'no session yet: press Run live'}
        </span>
        <span className="spacer" />
        <small className="muted">scroll to zoom · drag to pan · click a lane to select its stage</small>
        <button onClick={() => setView({ ...view, span: Math.max(MIN_SPAN, view.span / 1.5) })}>+</button>
        <button onClick={() => setView({ ...view, span: Math.min(3_600_000, view.span * 1.5) })}>−</button>
        <button onClick={fit}>Fit</button>
        <button className={view.end === null ? 'on' : ''} onClick={() => setView({ ...view, end: view.end === null ? currentEnd() : null })}>
          Follow
        </button>
      </div>
      <div className="dashboard-body" ref={wrapRef}>
        <canvas
          ref={canvasRef}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={() => {
            setHover(null);
            drag.current = null;
          }}
          onDoubleClick={fit}
        />
        {hover && (tip || vadAt || cleanAt) && (
          <div className="dashboard-tip" style={{ left: Math.min(hover.x + 12, size.w - 320), top: hover.y + 14 }}>
            {vadAt && <div>{vadAt}</div>}
            {cleanAt && <div>{cleanAt}</div>}
            {tip && <div>{tip.text}</div>}
          </div>
        )}
      </div>
    </div>
  );
}
