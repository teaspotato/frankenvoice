import type { NodeRegistry } from '../core/node';
import { micNode } from './mic';
import { aecNode, browserCleanupNode, noCleanupNode, rnnoiseNode } from './cleanup';
import { energyVadNode, fireredVadNode, sileroVadNode } from './vad';
import { asrNode } from './asr';
import { duplexTurnNode, halfDuplexTurnNode } from './turn';
import { noTurnModelNode, turnModelNode } from './turn-model';
import { llmNode } from './llm';
import { kyutaiTtsNode } from './tts/stream';
import { speakerNode } from './speaker';
import { echoLlmNode, toneTtsNode } from './mocks';
import { backchannelNode, noBackchannelNode } from './backchannel';
import { interjectNode, noInterjectNode } from './interject';
import { schedNode } from './sched';
import { tapeTurnNode } from './turn-tape';

export const registry: NodeRegistry = Object.fromEntries(
  [micNode, browserCleanupNode, aecNode, rnnoiseNode, noCleanupNode, sileroVadNode, fireredVadNode, energyVadNode, asrNode, turnModelNode, noTurnModelNode, duplexTurnNode, halfDuplexTurnNode, tapeTurnNode, llmNode, kyutaiTtsNode, speakerNode, echoLlmNode, toneTtsNode, backchannelNode, noBackchannelNode, interjectNode, noInterjectNode, schedNode].map(
    (d) => [d.type, d],
  ),
);
