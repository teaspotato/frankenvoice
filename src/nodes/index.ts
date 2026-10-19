import type { NodeRegistry } from '../core/node';
import { micNode } from './mic';
import { aecNode, browserCleanupNode, noCleanupNode, rnnoiseNode } from './cleanup';
import { energyVadNode, sileroVadNode } from './vad';
import { asrNode } from './asr';
import { duplexTurnNode, halfDuplexTurnNode } from './turn';
import { noTurnModelNode, turnModelNode } from './turn-model';
import { llmNode } from './llm';
import { fillerOpenerNode, noOpenerNode } from './opener';
import { agentNode, noAgentNode } from './agent';
import { ttsNode } from './tts';
import { streamingTtsNode } from './tts/stream';
import { speakerNode } from './speaker';
import { browserTtsNode } from './browser-tts';
import { echoLlmNode, toneTtsNode } from './mocks';

export const registry: NodeRegistry = Object.fromEntries(
  [micNode, browserCleanupNode, aecNode, rnnoiseNode, noCleanupNode, sileroVadNode, energyVadNode, asrNode, turnModelNode, noTurnModelNode, duplexTurnNode, halfDuplexTurnNode, llmNode, fillerOpenerNode, noOpenerNode, agentNode, noAgentNode, ttsNode, streamingTtsNode, browserTtsNode, speakerNode, echoLlmNode, toneTtsNode].map(
    (d) => [d.type, d],
  ),
);
