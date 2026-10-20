// The voice pipeline is fixed here, in code. What can be chosen at runtime
// (from the web UI) is which implementation each stage uses and its params;
// changing the topology means editing this file.

import type { EdgeSpec, GraphSpec, NodeSpec } from './core/graph';
import { checkGraph, GRAPH_FORMAT, GRAPH_FORMAT_VERSION } from './core/graph';
import { resolveParams } from './core/node';
import type { Json } from './core/types';
import { registry } from './nodes';
import { micProcessing } from './nodes/cleanup';

export type StageId = 'mic' | 'cleanup' | 'vad' | 'turnmodel' | 'asr' | 'turn' | 'interject' | 'backchannel' | 'llm' | 'opener' | 'workers' | 'tts' | 'speaker';

export interface StageDef {
  id: StageId;
  label: string;
  /** Implementation name -> node type. The first one is the default. */
  variants: Record<string, string>;
  /** Params set by another stage, hidden from this stage's panel. */
  managed?: string[];
  /** Says which stage sets the managed params. */
  managedNote?: string;
}

export const STAGES: StageDef[] = [
  { id: 'mic', label: 'Microphone', variants: { mic: 'mic' },
    managed: ['echoCancellation', 'noiseSuppression', 'autoGainControl'],
    managedNote: 'Browser echo cancellation, noise suppression and gain are set by the Noise / echo stage.',
  },
  {
    id: 'cleanup',
    label: 'Noise / echo',
    variants: { browser: 'cleanup.browser', aec: 'cleanup.aec', rnnoise: 'cleanup.rnnoise', off: 'cleanup.off' },
  },
  { id: 'vad', label: 'VAD', variants: { silero: 'vad.silero', energy: 'vad.energy' } },
  { id: 'turnmodel', label: 'Turn model', variants: { ssl_turn: 'turnmodel.ssl', off: 'turnmodel.off' } },
  { id: 'asr', label: 'ASR', variants: { streaming: 'asr' } },
  { id: 'turn', label: 'Turn-taking', variants: { 'full-duplex': 'turn.duplex', 'half-duplex': 'turn.half-duplex' } },
  { id: 'interject', label: 'Interjection', variants: { off: 'interject.off', typesafe: 'interject' } },
  { id: 'backchannel', label: 'Backchannels', variants: { on: 'backchannel', off: 'backchannel.off' } },
  { id: 'llm', label: 'LLM', variants: { api: 'llm', echo: 'llm.echo' }, managed: ['delegate'], managedNote: 'Delegation (the worker tools) is set by the Workers stage.' },
  { id: 'opener', label: 'Opener', variants: { filler: 'opener.filler', typesafe: 'opener.choice', off: 'opener.off' } },
  { id: 'workers', label: 'Workers', variants: { acp: 'workers', mock: 'workers.mock', off: 'workers.off' } },
  { id: 'tts', label: 'TTS', variants: { api: 'tts', 'deepgram-stream': 'tts.stream', browser: 'tts.browser', tone: 'tts.tone' } },
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
  interject: [630, 200],
  backchannel: [840, 20],
  llm: [840, 110],
  opener: [840, 200],
  workers: [840, 290],
  tts: [1050, 110],
  speaker: [1260, 110],
};

const e = (from: string, fp: string, to: string, tp: string): EdgeSpec => ({ from: { node: from, port: fp }, to: { node: to, port: tp } });

/**
 * The fixed topology: mic → noise/echo cleanup → VAD + turn model + ASR → turn-taking → LLM → TTS → speaker,
 * the opener watching the LLM and speaking through TTS too, workers taking the LLM's delegated tasks, with playback fed back to turn-taking and what was played to the turn model (the agent's channel).
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
        : s.id === 'llm'
          ? // The workers stage decides whether the LLM can delegate, and how it hears back.
            { ...stageParams(c, 'llm'), delegate: delegation(c) }
          : s.id === 'tts' && stageType(c, 'tts') === 'tts.stream' && !speakerWantsWords(c)
            ? // Word timings cost a Deepgram ASR stream; only a speaker that resumes from a word's start uses them.
              { ...stageParams(c, 'tts'), alignWords: false }
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
    // Speech end: ASR flushes its final words now rather than at its own endpointing.
    e('vad', 'vad', 'asr', 'vad'),
    e('asr', 'transcript', 'turn', 'transcript'),
    e('turn', 'turn', 'llm', 'turn'),
    // Early interjection: TypeSafe gates the words so far, the LLM is asked whether to cut in, turn-taking commits at the next pause.
    e('turn', 'turn', 'interject', 'turn'),
    e('vad', 'vad', 'interject', 'vad'),
    ...(registry[stageType(c, 'llm')].inputs.some((p) => p.name === 'interject') ? [e('interject', 'check', 'llm', 'interject'), e('llm', 'text', 'interject', 'text')] : []),
    ...(registry[stageType(c, 'turn')].inputs.some((p) => p.name === 'interject') ? [e('interject', 'take', 'turn', 'interject')] : []),
    // Backchannels: when, from the user's speech and the floor; their lines are rendered and played by TTS.
    e('vad', 'vad', 'backchannel', 'vad'),
    e('turnmodel', 'scores', 'backchannel', 'scores'),
    e('turn', 'turn', 'backchannel', 'turn'),
    e('asr', 'transcript', 'backchannel', 'transcript'),
    e('backchannel', 'text', 'tts', 'text'),
    e('llm', 'text', 'tts', 'text'),
    e('llm', 'text', 'opener', 'text'),
    // The opener may pick its line from the user's words.
    e('turn', 'turn', 'opener', 'turn'),
    e('opener', 'text', 'tts', 'text'),
    ...(registry[stageType(c, 'turn')].outputs.some((p) => p.name === 'text') ? [e('turn', 'text', 'tts', 'text')] : []),
    // Delegation: the LLM hands tasks to workers and hears back; the VAD tells it when the user pauses.
    e('llm', 'workers', 'workers', 'workers'),
    e('workers', 'updates', 'llm', 'updates'),
    e('vad', 'vad', 'llm', 'vad'),
    ...(hasSpeaker ? [e('tts', 'audio', 'speaker', 'audio'), e('speaker', 'playback', 'turn', 'playback')] : [e('tts', 'playback', 'turn', 'playback')]),
    ...(hasSpeaker && wantsReference ? [e('speaker', 'played', 'cleanup', 'ref')] : []),
    // Barge-in: turn-taking can stop the reply in flight.
    ...(registry[stageType(c, 'turn')].outputs.some((p) => p.name === 'control')
      ? [e('turn', 'control', 'llm', 'control'), e('turn', 'control', 'tts', 'control'), ...(hasSpeaker ? [e('turn', 'control', 'speaker', 'control')] : [])]
      : []),
  ];
  return { format: GRAPH_FORMAT, formatVersion: GRAPH_FORMAT_VERSION, name: 'Cascade', nodes, edges };
}

/** Whether the speaker resumes a cut-off reply from the start of a word (needs tts.words). */
function speakerWantsWords(c: PipelineConfig): boolean {
  if (!activeStages(c).some((s) => s.id === 'speaker')) return false;
  return resolveParams(registry[stageType(c, 'speaker')], stageParams(c, 'speaker')).resumeFrom === 'word';
}

/** What the LLM needs to delegate to workers, or null when the stage is off. */
function delegation(c: PipelineConfig): Json {
  const type = stageType(c, 'workers');
  if (type === 'workers.off') return null;
  const a = resolveParams(registry[type], stageParams(c, 'workers'));
  return { prompt: a.delegatePrompt, progress: a.progress, quietMs: a.quietMs, maxWorkers: a.maxWorkers };
}

/** Recover the stage config a recorded session ran with (stages it lacks keep their defaults). */
export function configFromGraph(graph: GraphSpec): PipelineConfig {
  const g = checkGraph(graph);
  const c = defaultConfig();
  for (const s of STAGES) {
    const n = g.nodes.find((x) => x.id === s.id) ?? g.nodes.find((x) => Object.values(s.variants).includes(x.type));
    const variant = n && Object.entries(s.variants).find(([, t]) => t === n.type)?.[0];
    if (n && variant) c[s.id] = { variant, params: { ...c[s.id].params, [variant]: n.params } };
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
