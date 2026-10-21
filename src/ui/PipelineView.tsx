import { registry } from '../nodes';
import type { Json } from '../core/types';
import type { Activity } from './activity';
import type { TtsEngineLoad, TurnModelLoad } from '../live-protocol';
import { loadingState } from '../nodes/turn-model';
import { STAGES, stageType, type PipelineConfig, type StageId } from '../pipeline';

const W = 150;
const H = 64;

/**
 * Fixed layout: one column per step, VAD, turn model and ASR stacked in the third. The
 * playback feedback runs above the main row and the echo reference below it,
 * where nothing else is drawn.
 */
const POS: Record<StageId, [number, number]> = {
  mic: [10, 88],
  cleanup: [210, 88],
  vad: [410, 10],
  turnmodel: [410, 88],
  asr: [410, 166],
  turn: [610, 88],
  interject: [610, 206],
  backchannel: [810, 206],
  llm: [810, 88],
  opener: [1010, 88],
  tts: [1210, 88],
  speaker: [1410, 88],
  sched: [1210, 206],
};
const VIEW_W = 1570;
const VIEW_H = 410;
const FEEDBACK_Y = 60;
/** The echo reference runs under everything, below ASR. */
const REFERENCE_Y = 300;
/** Barge-in cancels run between the row and the echo reference. */
const CONTROL_Y = 190;

function summarize(v: Json): string {
  if (v === null) return '–';
  if (typeof v === 'object') return Array.isArray(v) ? `[${v.length}]` : '{…}';
  const s = String(v);
  return s.length > 22 ? `${s.slice(0, 21)}…` : s;
}

/** The turn model's preload, in the shape the turn model stage reports while a session runs. */
function preloadState(l: TurnModelLoad): Record<string, Json> {
  const m = l.last as { type?: string; label?: string; done?: number; total?: number; cached?: boolean; text?: string } | undefined;
  if (l.state === 'failed') return { status: 'failed', error: l.error ?? null };
  if (l.state === 'ready') return { status: `preloaded (${(m as { ep?: string } | undefined)?.ep ?? '?'})` };
  return m ? loadingState(m) : { status: 'loading' };
}

/** The box's TTS engine while no session runs: a swap shows as a progress bar (seconds against the last swap's). */
function engineState(l: TtsEngineLoad): Record<string, Json> {
  if (l.state === 'failed') return { engine: l.engine, status: 'failed', error: l.error ?? null };
  if (l.state === 'ready') return { engine: l.engine, status: 'ready' };
  const done = Math.round(l.elapsed_s ?? 0);
  const total = Math.round(l.estimate_s ?? 60);
  return { download: { label: `switching to ${l.engine}`, done, total, fraction: Math.min(0.97, done / Math.max(1, total)), unit: 's' } };
}

interface Edge {
  /** The output port feeding this connection, `stage.port`. */
  src: string;
  d: string;
  label: string;
  lx: number;
  ly: number;
}

const right = (id: StageId): [number, number] => [POS[id][0] + W, POS[id][1] + H / 2];
const left = (id: StageId): [number, number] => [POS[id][0], POS[id][1] + H / 2];

function link(src: string, a: [number, number], b: [number, number], label: string): Edge {
  if (a[1] === b[1]) return { src, d: `M${a[0]},${a[1]} L${b[0] - 2},${b[1]}`, label, lx: (a[0] + b[0]) / 2, ly: a[1] - 6 };
  const mx = (a[0] + b[0]) / 2;
  return { src, d: `M${a[0]},${a[1]} C${mx},${a[1]} ${mx},${b[1]} ${b[0] - 2},${b[1]}`, label, lx: mx, ly: (a[1] + b[1]) / 2 - 6 };
}

/** Read-only diagram of the fixed pipeline with each stage's live state. */
export function PipelineView({
  config,
  turnModel,
  ttsEngine,
  states,
  stateTypes,
  activity,
  running,
  selected,
  onSelect,
}: {
  config: PipelineConfig;
  /** The preloaded turn model, shown on its stage while no session runs. */
  turnModel?: TurnModelLoad;
  /** The GPU box's TTS engine, shown on the TTS stage while no session runs. */
  ttsEngine?: TtsEngineLoad;
  /** Latest state per stage; only shown for stages that still run the same node type. */
  states: Record<string, Record<string, Json>>;
  stateTypes: Record<string, string>;
  /** What is streaming or switched on right now; empty when nothing runs. */
  activity: Activity;
  running: boolean;
  selected: StageId | null;
  onSelect(id: StageId | null): void;
}) {
  const stages = STAGES;
  const r = 10;
  const edges: Edge[] = [
    link('mic.audio', right('mic'), left('cleanup'), 'audio'),
    link('cleanup.audio', right('cleanup'), left('vad'), 'clean'),
    link('cleanup.audio', right('cleanup'), left('asr'), 'clean'),
    link('cleanup.audio', right('cleanup'), left('turnmodel'), ''),
    link('vad.vad', right('vad'), left('turn'), 'speech'),
    ...(registry[stageType(config, 'turn')].inputs.some((p) => p.name === 'scores') ? [link('turnmodel.scores', right('turnmodel'), left('turn'), 'scores')] : []),
    link('asr.transcript', right('asr'), left('turn'), 'transcript'),
    link('turn.turn', right('turn'), left('llm'), 'turn'),
    // The opener watches the LLM's stream and adds its own lines to what TTS says.
    link('llm.text', right('llm'), left('opener'), 'text'),
    link('opener.text', right('opener'), left('tts'), 'text'),
    // Backchannels hear the user (speech, words, turn model) through turn-taking's view of the floor.
    link('turn.turn', right('turn'), left('backchannel'), 'floor'),
    link('backchannel.text', right('backchannel'), left('tts'), 'lines'),
    link('tts.audio', right('tts'), left('speaker'), 'audio'),
  ];
  // Early interjection, under turn-taking: the words so far down to it, the "interrupt?" check up into the LLM, and the take back up.
  {
    const [ix, iy] = POS.interject;
    const ty = POS.turn[1] + H;
    edges.push({ src: 'turn.turn', d: `M${ix + 30},${ty} L${ix + 30},${iy - 2}`, label: 'words', lx: ix + 30, ly: iy - 30 });
    if (registry[stageType(config, 'turn')].inputs.some((p) => p.name === 'interject'))
      edges.push({ src: 'interject.take', d: `M${ix + 70},${iy} L${ix + 70},${ty + 2}`, label: 'take', lx: ix + 86, ly: iy - 30 });
    // Out of its top right and up into the LLM's bottom left, clear of the backchannel box beside it.
    if (registry[stageType(config, 'llm')].inputs.some((p) => p.name === 'interject')) {
      const [sx, lx, ky] = [ix + W - 20, POS.llm[0] + 20, iy - 26];
      edges.push({ src: 'interject.check', d: `M${sx},${iy} L${sx},${ky + 8} Q${sx},${ky} ${sx + 8},${ky} L${lx - 8},${ky} Q${lx},${ky} ${lx},${ky - 8} L${lx},${POS.llm[1] + H + 2}`, label: 'interrupt?', lx: lx + 34, ly: POS.llm[1] + H + 12 });
    }
  }
  // The tick scheduler under TTS: the TTS queue down into it, and the pace out of its left up into the LLM.
  {
    const [sx, sy] = POS.sched;
    edges.push({ src: 'tts.audio', d: `M${sx + 40},${POS.tts[1] + H} L${sx + 40},${sy - 2}`, label: 'queue', lx: sx + 20, ly: sy - 26 });
    if (registry[stageType(config, 'llm')].inputs.some((p) => p.name === 'sched')) {
      const [gx, ky, lx] = [POS.opener[0] - 25, CONTROL_Y - 14, POS.llm[0] + W - 20];
      edges.push({
        src: 'sched.control',
        d: `M${sx},${sy + H / 2} L${gx + r},${sy + H / 2} Q${gx},${sy + H / 2} ${gx},${sy + H / 2 - r} L${gx},${ky + r} Q${gx},${ky} ${gx - r},${ky} L${lx + r},${ky} Q${lx},${ky} ${lx},${ky - r} L${lx},${POS.llm[1] + H + 2}`,
        label: 'pace',
        lx: gx + 22,
        ly: sy + H / 2 - 6,
      });
    }
  }
  // Playback feedback: up from the speaker, back over the row, down into turn-taking.
  const [ox, oy] = [POS.speaker[0] + W / 2, POS.speaker[1]];
  const [tx, ty] = [POS.turn[0] + W / 2, POS.turn[1]];
  edges.push({
    src: 'speaker.playback',
    d: `M${ox},${oy} L${ox},${FEEDBACK_Y + r} Q${ox},${FEEDBACK_Y} ${ox - r},${FEEDBACK_Y} L${tx + r},${FEEDBACK_Y} Q${tx},${FEEDBACK_Y} ${tx},${FEEDBACK_Y + r} L${tx},${ty - 2}`,
    label: 'playback (what was heard)',
    lx: (ox + tx) / 2,
    ly: FEEDBACK_Y - 6,
  });

  // Barge-in: turn-taking can cancel the reply in flight, below the row.
  if (registry[stageType(config, 'turn')].outputs.some((p) => p.name === 'control')) {
    const sx = POS.turn[0] + W / 2 + 30;
    const by = POS.turn[1] + H;
    const targets = (['llm', 'tts', 'speaker'] as StageId[]).map((id) => POS[id][0] + W / 2 + 30);
    const last = targets[targets.length - 1];
    edges.push({
      src: 'turn.control',
      d: `M${sx},${by} L${sx},${CONTROL_Y - r} Q${sx},${CONTROL_Y} ${sx + r},${CONTROL_Y} L${last - r},${CONTROL_Y} Q${last},${CONTROL_Y} ${last},${CONTROL_Y - r} L${last},${by + 2}`,
      label: 'cancel (barge-in)',
      lx: (sx + targets[0]) / 2,
      ly: CONTROL_Y + 12,
    });
    for (const tx of targets.slice(0, -1)) edges.push({ src: 'turn.control', d: `M${tx},${CONTROL_Y} L${tx},${by + 2}`, label: '', lx: 0, ly: 0 });
  }

  if (registry[stageType(config, 'cleanup')].inputs.some((p) => p.name === 'ref')) {
    const [sx, sy] = [POS.speaker[0] + W / 2, POS.speaker[1] + H];
    const [cx, cy] = [POS.cleanup[0] + W / 2, POS.cleanup[1] + H];
    edges.push({
      src: 'speaker.played',
      d: `M${sx},${sy} L${sx},${REFERENCE_Y - r} Q${sx},${REFERENCE_Y} ${sx - r},${REFERENCE_Y} L${cx + r},${REFERENCE_Y} Q${cx},${REFERENCE_Y} ${cx},${REFERENCE_Y - r} L${cx},${cy + 2}`,
      label: 'echo reference (what was played)',
      lx: (sx + cx) / 2,
      ly: REFERENCE_Y - 6,
    });
  }

  // The agent's channel for the turn model: what the speaker played, along the bottom and up the gap left of the column.
  {
    const [sx, sy] = [POS.speaker[0] + W / 2 + 20, POS.speaker[1] + H];
    const gx = POS.turnmodel[0] - 16;
    const ey = POS.turnmodel[1] + H - 12;
    const by = REFERENCE_Y + 12;
    edges.push({
      src: 'speaker.played',
      d: `M${sx},${sy} L${sx},${by - r} Q${sx},${by} ${sx - r},${by} L${gx + r},${by} Q${gx},${by} ${gx},${by - r} L${gx},${ey + r} Q${gx},${ey} ${gx + r},${ey} L${POS.turnmodel[0] - 2},${ey}`,
      label: 'agent audio (to the turn model)',
      lx: (sx + gx) / 2 + 120,
      ly: by + 12,
    });
  }

  return (
    <svg className={`pipeline${running ? ' running' : ''}`} viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} preserveAspectRatio="xMidYMid meet" onClick={() => onSelect(null)}>
      <defs>
        <marker id="fv-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="context-stroke" />
        </marker>
      </defs>
      {edges.map((e) => (
        <g key={e.d} className={activity.edges.has(e.src) ? 'flowing' : undefined}>
          <path d={e.d} className="pipeline-edge" markerEnd="url(#fv-arrow)" />
          <text x={e.lx} y={e.ly} className="pipeline-edge-label" textAnchor="middle">
            {e.label}
          </text>
        </g>
      ))}
      {stages.map((s) => {
        const type = stageType(config, s.id);
        const def = registry[type];
        const [x, y] = POS[s.id];
        const variant = config[s.id].variant;
        const impl = Object.keys(s.variants).length > 1 ? variant : '';
        const state =
          s.id === 'turnmodel' && !running && turnModel
            ? preloadState(turnModel)
            : s.id === 'tts' && !running && ttsEngine
              ? engineState(ttsEngine)
              : stateTypes[s.id] === type
                ? states[s.id]
                : undefined;
        const on = activity.stages.get(s.id);
        const dl = state?.download as { label: string; done: number; total: number; fraction: number | null; unit?: string } | null | undefined;
        const lines = Object.entries(state ?? {})
          .filter(([k, v]) => k !== 'download' && (typeof v !== 'object' || v === null))
          .slice(0, 2);
        return (
          <g
            key={s.id}
            className={`pipeline-stage cat-${def.category}${selected === s.id ? ' selected' : ''}${on ? ' on' : ''}`}
            transform={`translate(${x},${y})`}
            onClick={(e) => {
              e.stopPropagation();
              onSelect(s.id);
            }}
          >
            <title>{def.description}</title>
            <rect width={W} height={H} rx={8} className="pipeline-box" />
            <rect width={4} height={H - 12} x={0} y={6} rx={2} className="pipeline-accent" />
            <text x={12} y={18} className="pipeline-title">
              {s.label}
            </text>
            {impl && (
              <text x={W - 8} y={18} className="pipeline-impl" textAnchor="end">
                {impl}
              </text>
            )}
            {on && (
              <text x={W / 2} y={H + 13} className="pipeline-on" textAnchor="middle">
                ● {on}
              </text>
            )}
            {dl && (
              <g className="pipeline-progress">
                <title>{dl.unit === 's' ? `${dl.label}: ${dl.done} s, the last one took ${dl.total} s` : `downloading the ${dl.label}: ${dl.done} of ${dl.total} MB`}</title>
                <text x={12} y={38} className="pipeline-state">
                  <tspan className="k">{dl.label}</tspan>{' '}
                  {dl.unit === 's' ? `${dl.done} s of ~${dl.total} s` : dl.fraction !== null ? `${Math.round(dl.fraction * 100)}% of ${dl.total} MB` : `${dl.done} MB`}
                </text>
                <rect x={12} y={46} width={W - 24} height={5} rx={2} className="track" />
                <rect x={12} y={46} width={(W - 24) * (dl.fraction ?? 0)} height={5} rx={2} className="bar" />
              </g>
            )}
            {(dl ? [] : lines).map(([k, v], i) => (
              <text key={k} x={12} y={38 + i * 15} className="pipeline-state">
                <tspan className="k">{k}</tspan> {summarize(v)}
              </text>
            ))}
          </g>
        );
      })}
    </svg>
  );
}
