import { useEffect, useLayoutEffect, useRef, useState, type ReactNode, type RefObject } from 'react';
import type { Trajectory } from '../core/trajectory';
import { TimelineModel, type Related } from './timeline';

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
  { id: 'turn', label: 'Turn', height: 52 },
  { id: 'llm', label: 'LLM', height: 60 },
  { id: 'tts', label: 'TTS', height: 36 },
  { id: 'speaker', label: 'Playback', height: 48 },
];

/** What each lane's marks mean, shown when hovering the lane's name. */
const LEGEND: Record<string, string> = {
  mic: 'Microphone input: waveform peaks every 10 ms, placed when each frame arrived. Green shading: speech the VAD detected.',
  cleanup: 'Audio after noise/echo cleanup: what VAD and ASR hear. Small ticks under it: AEC double talk or the RNNoise gate closing. Hover for levels.',
  vad: 'Speech probability per frame (line) against the start and stop thresholds (dashed). Shaded: detected speech.',
  asr:
    'Bar: a final transcript, over the audio it covers. Dashed box: the latest partial.\nBottom row: small grey dots are partial results arriving, purple dots are finals arriving. The purple horizontal line runs from the end of the audio to when its result arrived: the ASR latency.',
  turn: 'Top: floor state, thinking (amber) or speaking (blue); listening is blank. Vertical line: a user turn handed to the LLM.\nBottom: the user speaking while the agent had the floor, ending in the decision: red diamond = interrupted the agent, amber diamond = the user kept talking while the agent was thinking (its reply was dropped), grey dot = backchannel or noise, ignored.',
  llm: 'Top: waiting for the first token (light), then generating (solid), labelled with TTFT, chunks and chars/s. Bottom: a dot for each streamed chunk at the moment it arrived, with its text from there to the next chunk.',
  tts: 'One bar per sentence, from its request until its audio finished arriving. Diamonds at the bottom: requests sent to the provider.',
  speaker: 'Agent audio as scheduled to play. Shaded: one playback; a red line ends an interrupted one. Diamonds at the top: sentence starts; dots at the bottom: word boundaries.',
};

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
  /** Events behind this item, for highlighting what caused it and what it caused. */
  seqs?: number[];
  /** User audio it covers, in session time. */
  audio?: [number, number];
}

type Relation = 'self' | 'up' | 'down' | 'audio';

interface RelInfo {
  up: string[];
  down: string[];
}

/** How `h` relates to the hovered item. */
function relationOf(h: Hit, tip: Hit, rel: Related, audio: [number, number][]): Relation | undefined {
  if (!h.seqs?.length) return undefined;
  if (h.seqs[0] === tip.seqs?.[0] && h.lane === tip.lane) return 'self';
  if (h.seqs.some((s) => rel.up.has(s))) return 'up';
  if (h.seqs.some((s) => rel.down.has(s))) return 'down';
  if (h.audio && audio.some(([a, b]) => h.audio![0] < b && h.audio![1] > a)) return 'audio';
  return undefined;
}

/** Find the item under the pointer, topmost first. */
function hitAt(hits: Hit[], lane: string | undefined, px: number, py: number): Hit | undefined {
  for (let i = hits.length - 1; i >= 0; i--) {
    const h = hits[i];
    if (h.lane === lane && px >= h.x0 - 1 && px <= h.x1 + 1 && (h.y0 === undefined || (py >= h.y0 && py <= h.y1!))) return h;
  }
  return undefined;
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
  onFocusEvents,
  onSeek,
  playhead,
  transport,
  muted,
  onMute,
}: {
  trajectory?: Trajectory;
  running: boolean;
  /** Current session time while running (between events). */
  now?: () => number;
  selected: string | null;
  /** A lane was clicked; `item` when the click was on one of its items rather than empty space. */
  onSelect(lane: string | null, item?: boolean): void;
  /** An item was clicked: the events behind it. */
  onFocusEvents(seqs: number[]): void;
  /** Empty timeline space or the time axis was clicked: seek there. */
  onSeek(t: number): void;
  /** Where ▶ will start while nothing runs; while a recorded session plays back, the current playback time. */
  playhead: number | (() => number) | null;
  /** Play/pause/stop for the shown session. */
  transport: ReactNode;
  /** Mute switches on the Mic lane (recorded mic, heard in replays) and the Playback lane (the agent). */
  muted: { mic: boolean; agent: boolean };
  onMute(which: 'mic' | 'agent'): void;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<TimelineModel | null>(null);
  const hitsRef = useRef<Hit[]>([]);
  const [view, setView] = useState<View>({ end: null, span: 15000 });
  /** Causes and effects of the hovered item, as found by the last draw. */
  const [relInfo, setRelInfo] = useState<RelInfo | null>(null);
  const relKey = useRef('null');
  const viewRef = useRef(view);
  viewRef.current = view;
  const [size, setSize] = useState({ w: 800, h: 300 });
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null);
  const [tick, setTick] = useState(0);
  const drag = useRef<{ x: number; end: number } | null>(null);

  const totalH = AXIS + LANES.reduce((a, l) => a + l.height, 0);

  if (trajectory && modelRef.current?.traj !== trajectory) modelRef.current = new TimelineModel(trajectory);
  if (!trajectory) modelRef.current = null;

  // A new session starts following again, except a replay of the one on screen: the view stays put.
  const prevTraj = useRef(trajectory);
  useEffect(() => {
    const prev = prevTraj.current;
    prevTraj.current = trajectory;
    if (trajectory?.meta.mode === 'replay' && trajectory.meta.replayOf === prev?.meta.id) return;
    setView((v) => ({ ...v, end: null }));
  }, [trajectory]);

  // Playing back a recorded session shows all of it; following turns pages only when the playhead leaves the view.
  const playing = typeof playhead === 'function';
  const page = useRef<number | null>(null);
  if (!playing) page.current = null;

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
    if (playing && view.end === null) {
      const t = (playhead as () => number)();
      let end = page.current ?? Math.max(m?.duration ?? 0, view.span);
      if (t > end - view.span * 0.02 || t < end - view.span) end = t - view.span * 0.05 + view.span;
      page.current = end;
      return [Math.max(end, view.span) - view.span, Math.max(end, view.span)];
    }
    const head = Math.max(m?.duration ?? 0, running && !playing && now ? now() : 0);
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
      if (LEGEND[l.id]) {
        // hint that hovering the name explains the lane
        g.globalAlpha = hover && hover.x <= LABEL_W && hover.y >= y && hover.y < y + l.height ? 1 : 0.4;
        g.fillStyle = col.muted;
        g.fillText('ⓘ', LABEL_W - 16, y + l.height / 2);
        g.globalAlpha = 1;
      }
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
    /** A point-in-time marker: dots for things arriving, diamonds for things sent or starting. */
    const mark = (lane: string, t: number, color: string, at: number, shape: 'dot' | 'diamond', r = 3) => {
      const [ly0, ly1] = laneY[lane];
      const px = x(t);
      const py = ly0 + (ly1 - ly0) * at;
      g.fillStyle = color;
      g.beginPath();
      if (shape === 'dot') g.arc(px, py, r, 0, Math.PI * 2);
      else {
        g.moveTo(px, py - r - 0.5);
        g.lineTo(px + r + 0.5, py);
        g.lineTo(px, py + r + 0.5);
        g.lineTo(px - r - 0.5, py);
        g.closePath();
      }
      g.fill();
    };
    const head = Math.max(m?.duration ?? 0, running && now ? now() : 0);
    /** Attach the events (and user audio) behind the item just added to `hits`. */
    const tag = (seqs: number[], audio?: [number, number]) => {
      const h = hits[hits.length - 1];
      h.seqs = seqs;
      h.audio = audio;
    };

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
        hits.push({ lane: 'vad', x0: a, x1: b, text: `speech ${((((s.t1 ?? head) - s.t0) / 1000) as number).toFixed(2)} s`, seqs: s.seqs, audio: [s.t0, s.t1 ?? head] });
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
        if (r.final) {
          span('asr', r.a0, r.a1, col.asr, 0.3, r.text, 0.08, 0.62);
          tag(r.seqs, [r.a0, r.a1]);
          Object.assign(hits[hits.length - 1], { y0: ly0, y1: ly0 + (ly1 - ly0) * 0.64, text: `final transcript: ${r.text}` });
        } else {
          const x0 = x(r.a0);
          const x1 = Math.max(x(r.a1), x0 + 2);
          g.strokeStyle = col.asr;
          g.setLineDash([3, 2]);
          g.strokeRect(x0 + 0.5, ly0 + (ly1 - ly0) * 0.08 + 0.5, x1 - x0 - 1, (ly1 - ly0) * 0.54);
          g.setLineDash([]);
          label(r.text, x0, x1, ly0 + (ly1 - ly0) * 0.35, col.muted);
          hits.push({ lane: 'asr', x0, x1, y0: ly0, y1: ly0 + (ly1 - ly0) * 0.64, text: `partial transcript (may still change): ${r.text}`, seqs: r.seqs, audio: [r.a0, r.a1] });
        }
        // latency: end of the audio to arrival of this result
        g.strokeStyle = col.asr;
        g.globalAlpha = 0.5;
        g.beginPath();
        g.moveTo(x(r.a1), ly0 + (ly1 - ly0) * 0.82);
        g.lineTo(x(r.t), ly0 + (ly1 - ly0) * 0.82);
        g.stroke();
        g.globalAlpha = 1;
        hits.push({
          lane: 'asr',
          x0: x(r.a1),
          x1: x(r.t),
          y0: ly0 + (ly1 - ly0) * 0.64,
          y1: ly1,
          text: `ASR latency: the ${r.final ? 'final' : 'latest partial'} result arrived ${(r.t - r.a1).toFixed(0)} ms after the end of the audio it covers`,
          seqs: r.seqs,
        });
      }
      for (const a of m.asrArrivals) {
        if (a.t < t0 || a.t > t1) continue;
        mark('asr', a.t, a.final ? col.asr : col.muted, 0.82, 'dot', a.final ? 3.5 : 2.5);
        const [ly0, ly1] = laneY.asr;
        hits.push({
          lane: 'asr',
          x0: x(a.t) - 3,
          x1: x(a.t) + 3,
          y0: ly0 + (ly1 - ly0) * 0.64,
          y1: ly1,
          text: `${a.final ? 'final' : 'partial'} result arrived${a.a1 !== undefined ? ` ${(a.t - a.a1).toFixed(0)} ms after its audio ended` : ''}: ${a.text || '(empty)'}`,
          seqs: [a.seq],
          audio: a.a0 === undefined ? undefined : [a.a0, a.a1!],
        });
      }

      // turn-taking state
      const floorColor: Record<string, string> = { thinking: col.warn, speaking: col.accent };
      m.floor.forEach((f, i) => {
        const end = m.floor[i + 1]?.t ?? head;
        if (end < t0 || f.t > t1) return;
        if (floorColor[f.state]) span('turn', f.t, end, floorColor[f.state], 0.25, f.state, 0.06, 0.56);
        else hits.push({ lane: 'turn', x0: x(f.t), x1: x(end), text: f.state });
        tag([f.seq]);
      });
      for (const u of m.userTurns) {
        if (u.t < t0 || u.t > t1) continue;
        vline('turn', u.t, col.turn);
        hits.push({
          lane: 'turn',
          x0: x(u.t) - 3,
          x1: x(u.t) + 3,
          text: `user turn${u.replaces ? ' (resent: the user kept talking)' : ''}: ${u.text}`,
          seqs: [u.seq],
          audio: u.a0 === undefined ? undefined : [u.a0, u.a1!],
        });
      }
      // Barge-in: user speech while the agent had the floor, and the decision on it
      const verdict: Record<string, [string, string]> = {
        interrupt: [col.danger, 'interrupted the agent'],
        continue: [col.warn, 'user kept talking: reply dropped, turn resent'],
        backchannel: [col.muted, 'backchannel or noise: agent kept going'],
      };
      m.bargeIns.forEach((b, i) => {
        const end = b.t1 ?? head;
        if (end < t0 || b.t0 > t1) return;
        const [ly0, ly1] = laneY.turn;
        const h = ly1 - ly0;
        const [c, what] = verdict[b.decision ?? ''] ?? [col.text, 'deciding…'];
        const x0 = x(b.t0);
        const x1 = Math.max(x(end), x0 + 2);
        g.globalAlpha = 0.25;
        g.fillStyle = c;
        g.fillRect(x0, ly0 + h * 0.66, x1 - x0, h * 0.26);
        g.globalAlpha = 1;
        if (b.decision) mark('turn', end, c, 0.79, b.decision === 'backchannel' ? 'dot' : 'diamond', 4);
        const next = m.bargeIns[i + 1];
        label(b.decision ?? '', x1 + 4, Math.min(x1 + 120, next ? x(next.t0) : Infinity), ly0 + h * 0.79, c);
        hits.push({
          lane: 'turn',
          x0,
          x1: x1 + 6,
          y0: ly0 + h * 0.6,
          y1: ly1,
          text: `user spoke over the agent for ${(end - b.t0).toFixed(0)} ms: ${what}${b.why ? ` (${b.why})` : ''}${b.text ? `\nASR so far: ${b.text}` : ''}`,
          seqs: b.seqs,
        });
      });

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
        hits.push({ lane: 'llm', x0: x(l.t0), x1: x(end), text: `${stats}\n${l.text}`, seqs: l.seqs });
        let so = '';
        l.chunks.forEach((c, i) => {
          so += c.text;
          const cx = x(c.t);
          const nx = i + 1 < l.chunks.length ? x(l.chunks[i + 1].t) : x(end) + 200;
          if (nx < LABEL_W || cx > W) return;
          mark('llm', c.t, col.llm, 0.5, 'dot', 2.5);
          if (nx - cx > 8) {
            g.save();
            g.beginPath();
            g.rect(cx + 1, ly0 + h * 0.5, nx - cx - 2, h * 0.5);
            g.clip();
            g.fillStyle = col.text;
            g.fillText(c.text.replace(/\s+/g, ' '), cx + 2, ly0 + h * 0.76);
            g.restore();
          }
          hits.push({ lane: 'llm', y0: ly0 + h * 0.44, y1: ly1, x0: cx - 1, x1: Math.max(cx + 3, nx - 1), text: `+${(c.t - l.t0).toFixed(0)} ms  ${JSON.stringify(c.text)}\n${so}`, seqs: [c.seq] });
        });
      }

      // TTS: request sent, then audio streaming per sentence
      for (const r of m.ttsRequests) {
        if (r.t < t0 || r.t > t1) continue;
        mark('tts', r.t, col.muted, 0.86, 'diamond');
        const [ly0, ly1] = laneY.tts;
        hits.push({ lane: 'tts', x0: x(r.t) - 3, x1: x(r.t) + 3, y0: ly0 + (ly1 - ly0) * 0.72, y1: ly1, text: 'request sent to the TTS provider', seqs: [r.seq] });
      }
      for (const s of m.tts) {
        const end = s.t1 ?? head;
        if (end < t0 || s.t0 > t1) continue;
        span('tts', s.t0, end, col.tts, 0.35, s.text, 0.15, 0.7);
        tag(s.seqs);
      }

      // playback: what was heard
      for (const p of m.playback) {
        const end = p.t1 ?? head;
        if (end < t0 || p.t0 > t1) continue;
        span('speaker', p.t0, end, col.sink, 0.1, '', 0, 1);
        hits[hits.length - 1].text = `${p.interrupted ? 'interrupted' : 'played'} ${((end - p.t0) / 1000).toFixed(2)} s${p.heard !== undefined ? `\nheard: ${p.heard || '(nothing)'}` : ''}`;
        tag(p.seqs);
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
        mark('speaker', s.t, col.text, 0.14, 'diamond');
        hits.push({ lane: 'speaker', x0: x(s.t) - 3, x1: x(s.t) + 60, text: s.text, seqs: [s.seq] });
      }
      for (const b of m.boundaries) if (b >= t0 && b <= t1) mark('speaker', b, col.sink, 0.88, 'dot', 2);

      for (const err of m.errors) {
        const lane = laneY[err.node] ? err.node : 'mic';
        if (err.t < t0 || err.t > t1) continue;
        const [ly0] = laneY[lane];
        g.fillStyle = col.danger;
        g.beginPath();
        g.arc(x(err.t), ly0 + 6, 4, 0, Math.PI * 2);
        g.fill();
        hits.push({ lane, x0: x(err.t) - 5, x1: x(err.t) + 5, text: `error: ${err.text}`, seqs: [err.seq] });
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

    // causes and effects of the item under the pointer
    let info: RelInfo | null = null;
    if (m && hover && hover.x > LABEL_W) {
      const lane = Object.keys(laneY).find((k) => hover.y >= laneY[k][0] && hover.y < laneY[k][1]);
      const tipHit = hitAt(hits, lane, hover.x, hover.y);
      if (tipHit) {
        // Glow the hovered item so it's clear what the tooltip describes.
        const [ly0, ly1] = laneY[tipHit.lane];
        g.save();
        g.shadowColor = col.accent;
        g.shadowBlur = 10;
        g.strokeStyle = col.text;
        g.lineWidth = 2;
        g.strokeRect(tipHit.x0 - 1, (tipHit.y0 ?? ly0 + 2) - 1, Math.max(4, tipHit.x1 - tipHit.x0 + 2), (tipHit.y1 ?? ly1 - 2) - (tipHit.y0 ?? ly0 + 2) + 2);
        g.restore();
      }
      if (tipHit?.seqs?.length) {
        const rel = m.related(tipHit.seqs);
        let rels = hits.map((h) => relationOf(h, tipHit, rel, []));
        // User audio behind the item and everything causally tied to it; VAD/ASR on the same audio count too.
        const audio = hits.filter((h, i) => rels[i] && h.audio).map((h) => h.audio!);
        rels = hits.map((h, i) => rels[i] ?? relationOf(h, tipHit, rel, audio));
        g.fillStyle = col.accent;
        g.globalAlpha = 0.12;
        for (const [a, b] of audio)
          for (const l of ['mic', 'cleanup']) g.fillRect(x(a), laneY[l][0], Math.max(2, x(b) - x(a)), laneY[l][1] - laneY[l][0]);
        g.globalAlpha = 1;
        const stroke: Record<Relation, string> = { self: col.text, up: col.warn, down: col.accent, audio: col.muted };
        g.lineWidth = 2;
        hits.forEach((h, i) => {
          const r = rels[i];
          if (!r || h === tipHit) return;
          const [ly0, ly1] = laneY[h.lane];
          g.strokeStyle = stroke[r];
          g.setLineDash(r === 'audio' ? [4, 3] : []);
          g.strokeRect(h.x0, (h.y0 ?? ly0 + 2) + 1, Math.max(3, h.x1 - h.x0), (h.y1 ?? ly1 - 2) - (h.y0 ?? ly0 + 2) - 2);
        });
        g.setLineDash([]);
        g.lineWidth = 1;
        const list = (want: Relation) => {
          const out: string[] = [];
          hits
            .map((h, i) => [h, rels[i]] as const)
            .filter(([h, r]) => r === want && h.y0 === undefined)
            .sort((a, b) => a[0].x0 - b[0].x0)
            .forEach(([h]) => {
              const s = `${LANES.find((l) => l.id === h.lane)?.label ?? h.lane}: ${h.text.split('\n')[0].slice(0, 60)}`;
              if (!out.includes(s)) out.push(s);
            });
          return out;
        };
        info = { up: list('up'), down: list('down') };
      }
    }
    const key = JSON.stringify(info);
    if (key !== relKey.current) {
      relKey.current = key;
      setRelInfo(info);
    }

    // where ▶ starts, or where playback is
    const cursor = typeof playhead === 'function' ? playhead() : playhead;
    if (cursor !== null && (playing || cursor > 0) && cursor >= t0 && cursor <= t1) {
      const px = Math.round(x(cursor)) + 0.5;
      g.strokeStyle = col.accent;
      g.fillStyle = col.accent;
      g.beginPath();
      g.moveTo(px, AXIS);
      g.lineTo(px, H);
      g.stroke();
      g.beginPath();
      g.moveTo(px - 5, 2);
      g.lineTo(px + 5, 2);
      g.lineTo(px, AXIS - 6);
      g.closePath();
      g.fill();
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
  }, [tick, view, size, hover, selected, trajectory, running, totalH, playhead]);

  // ---- interaction --------------------------------------------------------

  const plotW = () => Math.max(1, size.w - LABEL_W);
  const currentEnd = () => window_(modelRef.current)[1];

  // Pinch (which browsers report as ctrl+wheel) or ctrl/cmd+wheel zooms; a sideways swipe pans;
  // an up/down swipe or plain wheel scrolls the lanes as usual.
  const onWheel = (e: WheelEvent) => {
    const zoom = e.ctrlKey || e.metaKey;
    if (!zoom && Math.abs(e.deltaY) >= Math.abs(e.deltaX)) return;
    e.preventDefault();
    const rect = canvasRef.current!.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const end = currentEnd();
    if (zoom) {
      // zoom around the cursor
      // Trackpad pinches send many small deltas; mouse wheels a few large ones.
      const f = Math.exp(e.deltaY * (Math.abs(e.deltaY) < 40 ? 0.01 : 0.0015));
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
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const t = currentEnd() - view.span + ((px - LABEL_W) / plotW()) * view.span;
    if (py < AXIS && px > LABEL_W) return onSeek(t);
    let y = AXIS;
    for (const l of LANES) {
      if (py >= y && py < y + l.height) {
        // A lane's name toggles its stage; an item shows its events; empty space seeks.
        if (px <= LABEL_W) return onSelect(l.id === selected ? null : l.id);
        const h = hitAt(hitsRef.current, l.id, px, py);
        if (h?.seqs?.length) {
          onFocusEvents(h.seqs);
          return onSelect(l.id, true);
        }
        return onSeek(t);
      }
      y += l.height;
    }
  };
  const fit = () => {
    const d = modelRef.current?.duration ?? 0;
    setView({ span: Math.max(MIN_SPAN, d * 1.02), end: running && !playing ? null : d * 1.01 });
  };

  let tip: Hit | undefined;
  let tipLane: string | undefined;
  if (hover) {
    let y = AXIS;
    for (const l of LANES) {
      if (hover.y >= y && hover.y < y + l.height) tipLane = l.id;
      y += l.height;
    }
    tip = hover.x > LABEL_W ? hitAt(hitsRef.current, tipLane, hover.x, hover.y) : undefined;
  }
  const legend = hover && hover.x <= LABEL_W && tipLane ? LEGEND[tipLane] : undefined;
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
          {trajectory ? (trajectory.meta.mode === 'replay' ? `replay of ${trajectory.meta.replayOf}` : playing ? `playing ${trajectory.meta.id}` : running ? 'live' : trajectory.meta.id) : 'no session yet: press Run live'}
        </span>
        <span className="spacer" />
        <small className="muted">click to seek · pinch or ⌘/ctrl+scroll to zoom · swipe sideways or drag to pan · click a lane's name to select its stage</small>
        <button onClick={() => setView({ ...view, span: Math.max(MIN_SPAN, view.span / 1.5) })}>+</button>
        <button onClick={() => setView({ ...view, span: Math.min(3_600_000, view.span * 1.5) })}>−</button>
        <button onClick={fit}>Fit</button>
        <button className={view.end === null ? 'on' : ''} onClick={() => setView({ ...view, end: view.end === null ? currentEnd() : null })}>
          Follow
        </button>
      </div>
      <div className="dashboard-body" ref={wrapRef}>
        {LANES.map((l, i) => {
          const which = l.id === 'mic' ? 'mic' : l.id === 'speaker' ? 'agent' : null;
          if (!which) return null;
          const top = AXIS + LANES.slice(0, i).reduce((a, x) => a + x.height, 0) + l.height / 2 + 7;
          return (
            <button
              key={l.id}
              className={`lane-mute${muted[which] ? ' on' : ''}`}
              style={{ top }}
              onClick={() => onMute(which)}
              title={muted[which] ? `Unmute ${which === 'mic' ? 'the recorded mic' : 'the agent'}` : `Mute ${which === 'mic' ? 'the recorded mic (heard in replays)' : 'the agent'}`}
            >
              {muted[which] ? '🔇' : '🔊'}
            </button>
          );
        })}
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
        {hover && legend && (
          <Tip className="dashboard-tip legend" x={LABEL_W - 8} y={hover.y} canvas={canvasRef} frame={wrapRef}>
            {legend}
          </Tip>
        )}
        {hover && !legend && (tip || vadAt || cleanAt) && (
          <Tip className="dashboard-tip" x={hover.x} y={hover.y} canvas={canvasRef} frame={wrapRef}>
            {vadAt && <div>{vadAt}</div>}
            {cleanAt && <div>{cleanAt}</div>}
            {tip && <div>{tip.text}</div>}
            {relInfo && relInfo.up.length > 0 && (
              <div className="rel up">
                <b>↑ triggered by</b>
                {relInfo.up.slice(-6).map((s) => (
                  <div key={s}>{s}</div>
                ))}
              </div>
            )}
            {relInfo && relInfo.down.length > 0 && (
              <div className="rel down">
                <b>↓ produced</b>
                {relInfo.down.slice(0, 8).map((s) => (
                  <div key={s}>{s}</div>
                ))}
                {relInfo.down.length > 8 && <div className="muted">… {relInfo.down.length - 8} more</div>}
              </div>
            )}
          </Tip>
        )}
      </div>
    </div>
  );
}

/**
 * A hover tooltip kept inside the dashboard's visible area: below-right of the pointer by default,
 * flipped left or above when it would cross the edge. Positioned against the viewport so the
 * scrolling dashboard can't clip it; if it is taller than the dashboard it may extend past it, never past the window.
 */
function Tip(props: { x: number; y: number; canvas: RefObject<HTMLCanvasElement | null>; frame: RefObject<HTMLDivElement | null>; className: string; children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    const c = props.canvas.current?.getBoundingClientRect();
    const f = props.frame.current?.getBoundingClientRect();
    if (!el || !c || !f) return;
    const pad = 4;
    const w = el.offsetWidth;
    const h = el.offsetHeight;
    const px = c.left + props.x;
    const py = c.top + props.y;
    let left = px + 12;
    if (left + w > f.right - pad) left = px - 12 - w;
    left = Math.max(Math.max(f.left, 0) + pad, Math.min(left, window.innerWidth - w - pad));
    let top = py + 14;
    if (top + h > f.bottom - pad) top = py - 10 - h;
    if (top < f.top + pad) top = Math.max(pad, Math.min(f.top + pad, window.innerHeight - h - pad));
    el.style.left = `${left}px`;
    el.style.top = `${top}px`;
    el.style.visibility = 'visible';
  });
  return (
    <div ref={ref} className={props.className} style={{ position: 'fixed', left: 0, top: 0, visibility: 'hidden' }}>
      {props.children}
    </div>
  );
}
