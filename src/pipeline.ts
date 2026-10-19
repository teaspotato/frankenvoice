// The voice pipeline is fixed here, in code. What can be chosen at runtime
// (from the web UI) is which implementation each stage uses and its params;
// changing the topology means editing this file.

import type { EdgeSpec, GraphSpec, NodeSpec } from './core/graph';
import { GRAPH_FORMAT, GRAPH_FORMAT_VERSION } from './core/graph';
import { resolveParams } from './core/node';
import type { Json } from './core/types';
import { migrateGraph, registry } from './nodes';
import { micProcessing } from './nodes/cleanup';

export type StageId = 'mic' | 'cleanup' | 'vad' | 'turnmodel' | 'asr' | 'turn' | 'llm' | 'tts' | 'speaker';

export interface StageDef {
  id: StageId;
  label: string;
  /** Implementation name -> node type. The first one is the default. */
  variants: Record<string, string>;
  /** Params set by another stage, hidden from this stage's panel. */
  managed?: string[];
}

export const STAGES: StageDef[] = [
  { id: 'mic', label: 'Microphone', variants: { mic: 'mic' }, managed: ['echoCancellation', 'noiseSuppression', 'autoGainControl'] },
  {
    id: 'cleanup',
    label: 'Noise / echo',
    variants: { browser: 'cleanup.browser', aec: 'cleanup.aec', rnnoise: 'cleanup.rnnoise', off: 'cleanup.off' },
  },
  { id: 'vad', label: 'VAD', variants: { silero: 'vad.silero', energy: 'vad.energy' } },
  { id: 'turnmodel', label: 'Turn model', variants: { ssl_turn: 'turnmodel.ssl', off: 'turnmodel.off' } },
  { id: 'asr', label: 'ASR', variants: { streaming: 'asr' } },
  { id: 'turn', label: 'Turn-taking', variants: { 'full-duplex': 'turn.duplex', 'half-duplex': 'turn.half-duplex' } },
  { id: 'llm', label: 'LLM', variants: { api: 'llm', echo: 'llm.echo' } },
  { id: 'tts', label: 'TTS', variants: { api: 'tts', browser: 'tts.browser', tone: 'tts.tone' } },
  { id: 'speaker', label: 'Speaker', variants: { speaker: 'speaker' } },
];

export interface StageConfig {
  variant: string;
  /** Params per variant, so switching back and forth keeps what was set. */
  params: Record<string, Record<string, Json>>;
}

export type PipelineConfig = Record<StageId, StageConfig>;

export const CONFIG_FORMAT = 'frankenvoice-pipeline';

export function defaultConfig(): PipelineConfig {
  const c = {} as PipelineConfig;
  for (const s of STAGES) c[s.id] = { variant: Object.keys(s.variants)[0], params: {} };
  c.asr.params.streaming = { provider: 'deepgram' };
  c.llm.params.api = { provider: 'anthropic' };
  c.tts.params.api = { provider: 'cartesia' };
  return c;
}

/** Fill in stages a saved config is missing (e.g. after a stage was added in code). */
export function normalizeConfig(c: Partial<PipelineConfig> | undefined): PipelineConfig {
  const d = defaultConfig();
  for (const s of STAGES) {
    const sc = c?.[s.id];
    if (sc && sc.variant in s.variants) d[s.id] = { variant: sc.variant, params: { ...d[s.id].params, ...sc.params } };
  }
  // Configs saved before full-duplex existed move to it once; picking half-duplex again sticks.
  if (d.turn.variant === 'half-duplex' && !('full-duplex' in d.turn.params)) d.turn = { variant: 'full-duplex', params: { ...d.turn.params, 'full-duplex': {} } };
  return d;
}

export const stageType = (c: PipelineConfig, id: StageId) => STAGES.find((s) => s.id === id)!.variants[c[id].variant];
export const stageParams = (c: PipelineConfig, id: StageId) => c[id].params[c[id].variant] ?? {};

/** The speaker stage only exists when TTS produces audio (the browser TTS plays it itself). */
export function activeStages(c: PipelineConfig): StageDef[] {
  const ttsPlays = registry[stageType(c, 'tts')].outputs.some((p) => p.name === 'playback');
  return STAGES.filter((s) => s.id !== 'speaker' || !ttsPlays);
}

const COLUMN: Record<StageId, [number, number]> = {
  mic: [0, 110],
  cleanup: [210, 110],
  vad: [420, 20],
  turnmodel: [420, 110],
  asr: [420, 200],
  turn: [630, 110],
  llm: [840, 110],
  tts: [1050, 110],
  speaker: [1260, 110],
};

const e = (from: string, fp: string, to: string, tp: string): EdgeSpec => ({ from: { node: from, port: fp }, to: { node: to, port: tp } });

/**
 * The fixed topology: mic → noise/echo cleanup → VAD + turn model + ASR → turn-taking → LLM → TTS → speaker,
 * with playback fed back to turn-taking and what was played to the turn model (the agent's channel).
 */
export function buildGraph(c: PipelineConfig): GraphSpec {
  const stages = activeStages(c);
  const cleanupType = stageType(c, 'cleanup');
  const nodes: NodeSpec[] = stages.map((s) => ({
    id: s.id,
    type: stageType(c, s.id),
    params:
      s.id === 'mic'
        ? // The cleanup stage decides what the browser does at capture.
          { ...stageParams(c, 'mic'), ...micProcessing(cleanupType, resolveParams(registry[cleanupType], stageParams(c, 'cleanup'))) }
        : stageParams(c, s.id),
    ui: { x: COLUMN[s.id][0], y: COLUMN[s.id][1] },
  }));
  const hasSpeaker = stages.some((s) => s.id === 'speaker');
  const wantsReference = registry[cleanupType].inputs.some((p) => p.name === 'ref');
  const edges = [
    e('mic', 'audio', 'cleanup', 'audio'),
    e('cleanup', 'audio', 'vad', 'audio'),
    e('cleanup', 'audio', 'asr', 'audio'),
    e('cleanup', 'audio', 'turnmodel', 'audio'),
    ...(hasSpeaker ? [e('speaker', 'played', 'turnmodel', 'played')] : []),
    ...(registry[stageType(c, 'turn')].inputs.some((p) => p.name === 'scores') ? [e('turnmodel', 'scores', 'turn', 'scores')] : []),
    e('vad', 'vad', 'turn', 'vad'),
    e('asr', 'transcript', 'turn', 'transcript'),
    e('turn', 'turn', 'llm', 'turn'),
    e('llm', 'text', 'tts', 'text'),
    ...(hasSpeaker ? [e('tts', 'audio', 'speaker', 'audio'), e('speaker', 'playback', 'turn', 'playback')] : [e('tts', 'playback', 'turn', 'playback')]),
    ...(hasSpeaker && wantsReference ? [e('speaker', 'played', 'cleanup', 'ref')] : []),
    // Barge-in: turn-taking can stop the reply in flight.
    ...(registry[stageType(c, 'turn')].outputs.some((p) => p.name === 'control')
      ? [e('turn', 'control', 'llm', 'control'), e('turn', 'control', 'tts', 'control'), ...(hasSpeaker ? [e('turn', 'control', 'speaker', 'control')] : [])]
      : []),
  ];
  return { format: GRAPH_FORMAT, formatVersion: GRAPH_FORMAT_VERSION, name: 'Cascade', nodes, edges };
}

/** Recover the stage config a recorded session ran with (stages it lacks keep their defaults). */
export function configFromGraph(graph: GraphSpec): PipelineConfig {
  const g = migrateGraph(graph);
  const c = defaultConfig();
  for (const s of STAGES) {
    const n = g.nodes.find((x) => x.id === s.id) ?? g.nodes.find((x) => Object.values(s.variants).includes(x.type));
    const variant = n && Object.entries(s.variants).find(([, t]) => t === n.type)?.[0];
    if (n && variant) c[s.id] = { variant, params: { ...c[s.id].params, [variant]: n.params } };
  }
  // Sessions from before the cleanup stage set browser processing on the mic.
  const mic = g.nodes.find((x) => x.type === 'mic');
  if (mic && !g.nodes.some((x) => x.type.startsWith('cleanup.'))) {
    const keys = ['echoCancellation', 'noiseSuppression', 'autoGainControl'];
    const flags = Object.fromEntries(keys.map((k) => [k, mic.params[k] !== false]));
    c.cleanup = Object.values(flags).some(Boolean) ? { variant: 'browser', params: { browser: flags } } : { variant: 'off', params: {} };
  }
  return c;
}

/** Proxy upstreams (API keys) the configured pipeline needs. */
export function requiredUpstreams(c: PipelineConfig): string[] {
  const out = new Set<string>();
  for (const s of activeStages(c)) {
    const def = registry[stageType(c, s.id)];
    for (const u of def.requires?.(resolveParams(def, stageParams(c, s.id))) ?? []) out.add(u);
  }
  return [...out];
}
