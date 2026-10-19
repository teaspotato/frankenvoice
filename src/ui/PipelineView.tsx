import { registry } from '../nodes';
import type { Json } from '../core/types';
import type { Activity } from './activity';
import type { TurnModelLoad } from '../drivers';
import { loadingState } from '../nodes/turn-model';
import { activeStages, STAGES, stageParams, stageType, type PipelineConfig, type StageId } from '../pipeline';

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
  backchannel: [810, 206],
  llm: [810, 88],
  opener: [1010, 88],
  workers: [1010, 330],
  tts: [1210, 88],
  speaker: [1410, 88],
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
  /** Latest state per stage; only shown for stages that still run the same node type. */
  states: Record<string, Record<string, Json>>;
  stateTypes: Record<string, string>;
  /** What is streaming or switched on right now; empty when nothing runs. */
  activity: Activity;
  running: boolean;
  selected: StageId | null;
  onSelect(id: StageId | null): void;
}) {
  const stages = activeStages(config);
  const out: StageId = stages.some((s) => s.id === 'speaker') ? 'speaker' : 'tts';
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
    ...(stages.some((s) => s.id === 'backchannel') ? [link('turn.turn', right('turn'), left('backchannel'), 'floor'), link('backchannel.text', right('backchannel'), left('tts'), 'lines')] : []),
    ...(out === 'speaker' ? [link('tts.audio', right('tts'), left('speaker'), 'audio')] : []),
  ];
  // Delegation: tasks out of the LLM's right side, down the gap beside the opener to the workers, and their news back.
  {
    const [rx, ry] = [POS.llm[0] + W, POS.llm[1] + H / 2];
    const [wx, wy] = POS.workers;
    const gap = (POS.opener[0] - rx) / 2;
    edges.push({ src: 'llm.workers', d: `M${rx},${ry - 8} L${rx + gap + 8},${ry - 8} L${rx + gap + 8},${wy + H * 0.35} L${wx - 2},${wy + H * 0.35}`, label: 'tasks', lx: rx + gap + 30, ly: wy - 60 });
    edges.push({ src: 'workers.updates', d: `M${wx},${wy + H * 0.7} L${rx + gap - 8},${wy + H * 0.7} L${rx + gap - 8},${ry + 8} L${rx + 2},${ry + 8}`, label: 'results', lx: rx + gap - 44, ly: wy - 60 });
  }
  // Playback feedback: up from the output stage, back over the row, down into turn-taking.
  const [ox, oy] = [POS[out][0] + W / 2, POS[out][1]];
  const [tx, ty] = [POS.turn[0] + W / 2, POS.turn[1]];
  const r = 10;
  edges.push({
    src: `${out}.playback`,
    d: `M${ox},${oy} L${ox},${FEEDBACK_Y + r} Q${ox},${FEEDBACK_Y} ${ox - r},${FEEDBACK_Y} L${tx + r},${FEEDBACK_Y} Q${tx},${FEEDBACK_Y} ${tx},${FEEDBACK_Y + r} L${tx},${ty - 2}`,
    label: 'playback (what was heard)',
    lx: (ox + tx) / 2,
    ly: FEEDBACK_Y - 6,
  });

  // Barge-in: turn-taking can cancel the reply in flight, below the row.
  if (registry[stageType(config, 'turn')].outputs.some((p) => p.name === 'control')) {
    const sx = POS.turn[0] + W / 2 + 30;
    const by = POS.turn[1] + H;
    const targets = (['llm', 'tts', ...(out === 'speaker' ? ['speaker'] : [])] as StageId[]).map((id) => POS[id][0] + W / 2 + 30);
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

  if (out === 'speaker' && registry[stageType(config, 'cleanup')].inputs.some((p) => p.name === 'ref')) {
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
  if (out === 'speaker') {
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
        const params = stageParams(config, s.id);
        const variant = config[s.id].variant;
        const impl = typeof params.provider === 'string' ? params.provider : Object.keys(STAGES.find((d) => d.id === s.id)!.variants).length > 1 ? variant : '';
        const state = s.id === 'turnmodel' && !running && turnModel ? preloadState(turnModel) : stateTypes[s.id] === type ? states[s.id] : undefined;
        const on = activity.stages.get(s.id);
        const dl = state?.download as { label: string; done: number; total: number; fraction: number | null } | null | undefined;
        const lines = Object.entries(state ?? {})
          .filter(([k, v]) => k !== 'provider' && k !== 'download' && (typeof v !== 'object' || v === null))
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
                <title>{`downloading the ${dl.label}: ${dl.done} of ${dl.total} MB`}</title>
                <text x={12} y={38} className="pipeline-state">
                  <tspan className="k">{dl.label}</tspan> {dl.fraction !== null ? `${Math.round(dl.fraction * 100)}% of ${dl.total} MB` : `${dl.done} MB`}
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
