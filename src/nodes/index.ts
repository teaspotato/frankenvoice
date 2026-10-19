import type { NodeRegistry } from '../core/node';
import type { GraphSpec, NodeSpec } from '../core/graph';
import type { Json } from '../core/types';
import { GRAPH_FORMAT_VERSION } from '../core/graph';
import { micNode } from './mic';
import { aecNode, browserCleanupNode, noCleanupNode, rnnoiseNode } from './cleanup';
import { energyVadNode, sileroVadNode } from './vad';
import { asrNode } from './asr';
import { duplexTurnNode, halfDuplexTurnNode } from './turn';
import { noTurnModelNode, turnModelNode } from './turn-model';
import { llmNode } from './llm';
import { ttsNode } from './tts';
import { streamingTtsNode } from './tts/stream';
import { speakerNode } from './speaker';
import { browserTtsNode } from './browser-tts';
import { echoLlmNode, toneTtsNode } from './mocks';

export const registry: NodeRegistry = Object.fromEntries(
  [micNode, browserCleanupNode, aecNode, rnnoiseNode, noCleanupNode, sileroVadNode, energyVadNode, asrNode, turnModelNode, noTurnModelNode, duplexTurnNode, halfDuplexTurnNode, llmNode, ttsNode, streamingTtsNode, browserTtsNode, speakerNode, echoLlmNode, toneTtsNode].map(
    (d) => [d.type, d],
  ),
);

/**
 * Upgrade graphs saved by older versions, so they run as they did: format 1 had
 * one node type per provider; before format 3, full-duplex turns did not draft replies early.
 */
export function migrateGraph(g: GraphSpec): GraphSpec {
  if (g.formatVersion >= GRAPH_FORMAT_VERSION) return g;
  const older = (n: NodeSpec) => (g.formatVersion < 3 ? noDrafts(noOpener(n)) : noOpener(n));
  if (g.formatVersion >= 2) return { ...g, formatVersion: GRAPH_FORMAT_VERSION, nodes: g.nodes.map(older) };
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
  return { ...g, formatVersion: GRAPH_FORMAT_VERSION, nodes: nodes.map(older) };
}

/** Before format 4 the LLM had no opener, sent no thinking setting and defaulted to Opus. */
const noOpener = (n: NodeSpec): NodeSpec => {
  if (n.type !== 'llm') return n;
  const p = { ...n.params };
  p.opener ??= false;
  p.thinking ??= 'model default';
  if ((p.provider ?? 'anthropic') === 'anthropic' && !p.model) p.model = 'claude-opus-5-5';
  return { ...n, params: p };
};
const noDrafts = (n: NodeSpec): NodeSpec => (n.type === 'turn.duplex' && n.params.speculate === undefined ? { ...n, params: { ...n.params, speculate: false } } : n);
