import type { NodeRegistry } from '../core/node';
import type { GraphSpec } from '../core/graph';
import type { Json } from '../core/types';
import { GRAPH_FORMAT_VERSION } from '../core/graph';
import { micNode } from './mic';
import { aecNode, browserCleanupNode, noCleanupNode, rnnoiseNode } from './cleanup';
import { energyVadNode, sileroVadNode } from './vad';
import { asrNode } from './asr';
import { halfDuplexTurnNode } from './turn';
import { llmNode } from './llm';
import { ttsNode } from './tts';
import { speakerNode } from './speaker';
import { browserTtsNode } from './browser-tts';
import { echoLlmNode, toneTtsNode } from './mocks';

export const registry: NodeRegistry = Object.fromEntries(
  [micNode, browserCleanupNode, aecNode, rnnoiseNode, noCleanupNode, sileroVadNode, energyVadNode, asrNode, halfDuplexTurnNode, llmNode, ttsNode, browserTtsNode, speakerNode, echoLlmNode, toneTtsNode].map(
    (d) => [d.type, d],
  ),
);

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
