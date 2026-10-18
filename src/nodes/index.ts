import type { NodeRegistry } from '../core/node';
import type { GraphSpec } from '../core/graph';
import { GRAPH_FORMAT, GRAPH_FORMAT_VERSION } from '../core/graph';
import { micNode } from './mic';
import { energyVadNode, sileroVadNode } from './vad';
import { deepgramAsrNode } from './asr-deepgram';
import { halfDuplexTurnNode } from './turn';
import { claudeLlmNode } from './llm-claude';
import { cartesiaTtsNode } from './tts-cartesia';
import { speakerNode } from './speaker';
import { echoLlmNode, toneTtsNode } from './mocks';

export const registry: NodeRegistry = Object.fromEntries(
  [micNode, sileroVadNode, energyVadNode, deepgramAsrNode, halfDuplexTurnNode, claudeLlmNode, cartesiaTtsNode, speakerNode, echoLlmNode, toneTtsNode].map(
    (d) => [d.type, d],
  ),
);

const e = (from: string, fp: string, to: string, tp: string) => ({ from: { node: from, port: fp }, to: { node: to, port: tp } });

/** Mic → VAD + ASR → turns → Claude → TTS → speaker. */
export function defaultGraph(): GraphSpec {
  return {
    format: GRAPH_FORMAT,
    formatVersion: GRAPH_FORMAT_VERSION,
    name: 'Half-duplex cascade',
    nodes: [
      { id: 'mic', type: 'mic', params: {}, ui: { x: 0, y: 120 } },
      { id: 'vad', type: 'vad.silero', params: {}, ui: { x: 260, y: 0 } },
      { id: 'asr', type: 'asr.deepgram', params: {}, ui: { x: 260, y: 220 } },
      { id: 'turn', type: 'turn.half-duplex', params: {}, ui: { x: 540, y: 120 } },
      { id: 'llm', type: 'llm.claude', params: {}, ui: { x: 800, y: 120 } },
      { id: 'tts', type: 'tts.cartesia', params: {}, ui: { x: 1060, y: 120 } },
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
