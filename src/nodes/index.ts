import type { NodeRegistry } from '../core/node';
import type { GraphSpec } from '../core/graph';
import type { Json } from '../core/types';
import { GRAPH_FORMAT, GRAPH_FORMAT_VERSION } from '../core/graph';
import { micNode } from './mic';
import { energyVadNode, sileroVadNode } from './vad';
import { asrNode } from './asr';
import { halfDuplexTurnNode } from './turn';
import { llmNode } from './llm';
import { ttsNode } from './tts';
import { speakerNode } from './speaker';
import { echoLlmNode, toneTtsNode } from './mocks';

export const registry: NodeRegistry = Object.fromEntries(
  [micNode, sileroVadNode, energyVadNode, asrNode, halfDuplexTurnNode, llmNode, ttsNode, speakerNode, echoLlmNode, toneTtsNode].map(
    (d) => [d.type, d],
  ),
);

const e = (from: string, fp: string, to: string, tp: string) => ({ from: { node: from, port: fp }, to: { node: to, port: tp } });

/** Upgrade graphs saved by older versions (format 1 had one node type per provider). */
export function migrateGraph(g: GraphSpec): GraphSpec {
  if (g.formatVersion >= GRAPH_FORMAT_VERSION) return g;
  const prefix = (provider: string, params: Record<string, Json>, rename: Record<string, string> = {}) => ({
    provider,
    ...Object.fromEntries(Object.entries(params).map(([k, v]) => [`${provider}.${rename[k] ?? k}`, v])),
  });
  const nodes = g.nodes.map((n) => {
    switch (n.type) {
      case 'llm.claude':
        return { ...n, type: 'llm', params: { provider: 'anthropic', ...n.params } };
      case 'asr.deepgram':
        return { ...n, type: 'asr', params: prefix('deepgram', n.params) };
      case 'tts.cartesia': {
        const { minSegmentChars, ...rest } = n.params;
        return {
          ...n,
          type: 'tts',
          params: { ...prefix('cartesia', rest, { modelId: 'model' }), ...(minSegmentChars !== undefined ? { minSegmentChars } : {}) },
        };
      }
      default:
        return n;
    }
  });
  return { ...g, formatVersion: GRAPH_FORMAT_VERSION, nodes };
}

/** Mic → VAD + ASR → turns → Claude → TTS → speaker. */
export function defaultGraph(): GraphSpec {
  return {
    format: GRAPH_FORMAT,
    formatVersion: GRAPH_FORMAT_VERSION,
    name: 'Half-duplex cascade',
    nodes: [
      { id: 'mic', type: 'mic', params: {}, ui: { x: 0, y: 120 } },
      { id: 'vad', type: 'vad.silero', params: {}, ui: { x: 260, y: 0 } },
      { id: 'asr', type: 'asr', params: { provider: 'deepgram' }, ui: { x: 260, y: 220 } },
      { id: 'turn', type: 'turn.half-duplex', params: {}, ui: { x: 540, y: 120 } },
      { id: 'llm', type: 'llm', params: { provider: 'anthropic' }, ui: { x: 800, y: 120 } },
      { id: 'tts', type: 'tts', params: { provider: 'cartesia' }, ui: { x: 1060, y: 120 } },
      { id: 'speaker', type: 'speaker', params: {}, ui: { x: 1320, y: 120 } },
    ],
    edges: [
      e('mic', 'audio', 'vad', 'audio'),
      e('mic', 'audio', 'asr', 'audio'),
      e('vad', 'vad', 'turn', 'vad'),
      e('asr', 'transcript', 'turn', 'transcript'),
      e('turn', 'turn', 'llm', 'turn'),
      e('llm', 'text', 'tts', 'text'),
      e('tts', 'audio', 'speaker', 'audio'),
      e('speaker', 'playback', 'turn', 'playback'),
    ],
  };
}
