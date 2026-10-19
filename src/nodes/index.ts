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
import { fillerOpenerNode, noOpenerNode } from './opener';
import { ttsNode } from './tts';
import { streamingTtsNode } from './tts/stream';
import { speakerNode } from './speaker';
import { browserTtsNode } from './browser-tts';
import { echoLlmNode, toneTtsNode } from './mocks';

export const registry: NodeRegistry = Object.fromEntries(
  [micNode, browserCleanupNode, aecNode, rnnoiseNode, noCleanupNode, sileroVadNode, energyVadNode, asrNode, turnModelNode, noTurnModelNode, duplexTurnNode, halfDuplexTurnNode, llmNode, fillerOpenerNode, noOpenerNode, ttsNode, streamingTtsNode, browserTtsNode, speakerNode, echoLlmNode, toneTtsNode].map(
    (d) => [d.type, d],
  ),
);

/**
 * Upgrade graphs saved by older versions, so they run as they did: format 1 had
 * one node type per provider; before format 3, full-duplex turns did not draft replies early.
 */
export function migrateGraph(g: GraphSpec): GraphSpec {
  if (g.formatVersion >= GRAPH_FORMAT_VERSION) return g;
  const older = (n: NodeSpec) => {
    if (g.formatVersion < 3) n = noDrafts(n);
    if (g.formatVersion < 4) n = noThinking(n);
    if (g.formatVersion < 5) n = noOpener(n);
    else if (g.formatVersion < 7) n = cannedOpener(n);
    if (g.formatVersion < 6) n = oldModelRules(n);
    return n;
  };
  if (g.formatVersion === 7) return fillerStage({ ...g, formatVersion: GRAPH_FORMAT_VERSION });
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

/** Before format 4 the LLM sent no thinking setting and defaulted to Opus. */
const noThinking = (n: NodeSpec): NodeSpec => {
  if (n.type !== 'llm') return n;
  const p = { ...n.params };
  p.thinking ??= 'model default';
  if ((p.provider ?? 'anthropic') === 'anthropic' && !p.model) p.model = 'claude-opus-5-5';
  return { ...n, params: p };
};
/** Before format 5 there was no canned opener (format 4's opener asked an LLM, and is gone). */
const noOpener = (n: NodeSpec): NodeSpec => (n.type === 'llm' ? { ...n, params: { ...n.params, opener: 'off' } } : n);
/** Formats 5 and 6 defaulted to the canned opener (format 7: the thinking filler). */
const cannedOpener = (n: NodeSpec): NodeSpec => (n.type === 'llm' ? { ...n, params: { opener: 'canned', ...n.params } } : n);
/**
 * Format 7 said the thinking filler from inside the LLM (its default opener, 'thinking').
 * It is the opener stage now: the filler moves there, between the LLM and TTS.
 */
function fillerStage(g: GraphSpec): GraphSpec {
  const llm = g.nodes.find((n) => n.type === 'llm');
  const tts = g.edges.find((e) => e.from.node === llm?.id && e.from.port === 'text');
  if (!llm) return g;
  const { opener, fillerLines, fillerAfterMs, ...params } = llm.params;
  const nodes = g.nodes.map((n) => (n === llm ? { ...n, params: opener === undefined || opener === 'thinking' ? params : { ...params, opener } } : n));
  if (!tts || (opener !== undefined && opener !== 'thinking')) return { ...g, nodes };
  const id = 'opener';
  nodes.push({
    id,
    type: 'opener.filler',
    params: { ...(fillerLines !== undefined ? { lines: fillerLines } : {}), ...(fillerAfterMs !== undefined ? { afterMs: fillerAfterMs } : {}) },
    ...(llm.ui ? { ui: { ...llm.ui, y: llm.ui.y + 90 } } : {}),
  });
  const edges = [...g.edges, { from: { node: llm.id, port: 'text' }, to: { node: id, port: 'text' } }, { from: { node: id, port: 'text' }, to: tts.to }];
  return { ...g, nodes, edges };
}
/** Before format 6 the turn model's floor takes counted without VAD speech, and silence never committed. */
const oldModelRules = (n: NodeSpec): NodeSpec =>
  n.type === 'turn.duplex' ? { ...n, params: { modelNeedsVad: false, maxSilenceMs: 0, ...n.params } } : n;
const noDrafts = (n: NodeSpec): NodeSpec => (n.type === 'turn.duplex' && n.params.speculate === undefined ? { ...n, params: { ...n.params, speculate: false } } : n);
