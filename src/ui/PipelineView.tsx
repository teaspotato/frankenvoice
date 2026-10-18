import { registry } from '../nodes';
import type { Json } from '../core/types';
import { activeStages, STAGES, stageParams, stageType, type PipelineConfig, type StageId } from '../pipeline';

const W = 150;
const H = 64;

/**
 * Fixed layout: one column per step, VAD and ASR stacked in the third. The
 * playback feedback runs above the main row and the echo reference below it,
 * where nothing else is drawn.
 */
const POS: Record<StageId, [number, number]> = {
  mic: [10, 88],
  cleanup: [210, 88],
  vad: [410, 10],
  asr: [410, 166],
  turn: [610, 88],
  llm: [810, 88],
  tts: [1010, 88],
  speaker: [1210, 88],
};
const VIEW_W = 1370;
const VIEW_H = 272;
const FEEDBACK_Y = 60;
/** The echo reference runs under everything, below ASR. */
const REFERENCE_Y = 250;

function summarize(v: Json): string {
  if (v === null) return '–';
  if (typeof v === 'object') return Array.isArray(v) ? `[${v.length}]` : '{…}';
  const s = String(v);
  return s.length > 22 ? `${s.slice(0, 21)}…` : s;
}

interface Edge {
  d: string;
  label: string;
  lx: number;
  ly: number;
}

const right = (id: StageId): [number, number] => [POS[id][0] + W, POS[id][1] + H / 2];
const left = (id: StageId): [number, number] => [POS[id][0], POS[id][1] + H / 2];

function link(a: [number, number], b: [number, number], label: string): Edge {
  if (a[1] === b[1]) return { d: `M${a[0]},${a[1]} L${b[0] - 2},${b[1]}`, label, lx: (a[0] + b[0]) / 2, ly: a[1] - 6 };
  const mx = (a[0] + b[0]) / 2;
  return { d: `M${a[0]},${a[1]} C${mx},${a[1]} ${mx},${b[1]} ${b[0] - 2},${b[1]}`, label, lx: mx, ly: (a[1] + b[1]) / 2 - 6 };
}

/** Read-only diagram of the fixed pipeline with each stage's live state. */
export function PipelineView({
  config,
  states,
  stateTypes,
  running,
  selected,
  onSelect,
}: {
  config: PipelineConfig;
  /** Latest state per stage; only shown for stages that still run the same node type. */
  states: Record<string, Record<string, Json>>;
  stateTypes: Record<string, string>;
  running: boolean;
  selected: StageId | null;
  onSelect(id: StageId | null): void;
}) {
  const stages = activeStages(config);
  const out: StageId = stages.some((s) => s.id === 'speaker') ? 'speaker' : 'tts';
  const edges: Edge[] = [
    link(right('mic'), left('cleanup'), 'audio'),
    link(right('cleanup'), left('vad'), 'clean'),
    link(right('cleanup'), left('asr'), 'clean'),
    link(right('vad'), left('turn'), 'speech'),
    link(right('asr'), left('turn'), 'transcript'),
    link(right('turn'), left('llm'), 'turn'),
    link(right('llm'), left('tts'), 'text'),
    ...(out === 'speaker' ? [link(right('tts'), left('speaker'), 'audio')] : []),
  ];
  // Playback feedback: up from the output stage, back over the row, down into turn-taking.
  const [ox, oy] = [POS[out][0] + W / 2, POS[out][1]];
  const [tx, ty] = [POS.turn[0] + W / 2, POS.turn[1]];
  const r = 10;
  edges.push({
    d: `M${ox},${oy} L${ox},${FEEDBACK_Y + r} Q${ox},${FEEDBACK_Y} ${ox - r},${FEEDBACK_Y} L${tx + r},${FEEDBACK_Y} Q${tx},${FEEDBACK_Y} ${tx},${FEEDBACK_Y + r} L${tx},${ty - 2}`,
    label: 'playback (what was heard)',
    lx: (ox + tx) / 2,
    ly: FEEDBACK_Y - 6,
  });

  if (out === 'speaker' && registry[stageType(config, 'cleanup')].inputs.some((p) => p.name === 'ref')) {
    const [sx, sy] = [POS.speaker[0] + W / 2, POS.speaker[1] + H];
    const [cx, cy] = [POS.cleanup[0] + W / 2, POS.cleanup[1] + H];
    edges.push({
      d: `M${sx},${sy} L${sx},${REFERENCE_Y - r} Q${sx},${REFERENCE_Y} ${sx - r},${REFERENCE_Y} L${cx + r},${REFERENCE_Y} Q${cx},${REFERENCE_Y} ${cx},${REFERENCE_Y - r} L${cx},${cy + 2}`,
      label: 'echo reference (what was played)',
      lx: (sx + cx) / 2,
      ly: REFERENCE_Y - 6,
    });
  }

  return (
    <svg className={`pipeline${running ? ' running' : ''}`} viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} preserveAspectRatio="xMidYMid meet" onClick={() => onSelect(null)}>
      <defs>
        <marker id="fv-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" className="pipeline-arrowhead" />
        </marker>
      </defs>
      {edges.map((e) => (
        <g key={e.d}>
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
        const state = stateTypes[s.id] === type ? states[s.id] : undefined;
        const lines = Object.entries(state ?? {})
          .filter(([k, v]) => k !== 'provider' && (typeof v !== 'object' || v === null))
          .slice(0, 2);
        return (
          <g
            key={s.id}
            className={`pipeline-stage cat-${def.category}${selected === s.id ? ' selected' : ''}`}
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
            {lines.map(([k, v], i) => (
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
