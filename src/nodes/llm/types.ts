import type { Json, JsonObject } from '../../core/types';

/**
 * Conversation history kept by the LLM node. `text` is provider-neutral; for
 * assistant turns `native` keeps the provider's own content (e.g. Claude
 * thinking blocks) so the same provider gets it back verbatim.
 */
export interface ChatTurn {
  role: 'user' | 'assistant';
  text: string;
  native?: { provider: string; content: Json };
  /** The user turn this belongs to (an assistant turn: the one it answers). */
  turnId?: string;
}

/** What every provider stream is reduced to. */
export type LlmStreamEvent =
  | { type: 'delta'; text: string }
  /** The model is reasoning (a thinking block opened, or reasoning tokens arrived); no text for the user. */
  | { type: 'thinking' }
  | { type: 'end'; stopReason: string | null }
  | { type: 'error'; error: Json };

/** Parses one response stream. Created per request; fed raw recorded payloads. */
export interface LlmStreamParser {
  recv(data: Json): LlmStreamEvent[];
  close(data: Json): LlmStreamEvent[];
  /** Provider-native assistant content produced so far, for the history. */
  native(): Json;
}

/**
 * A provider adapter is pure: it builds the request that is recorded and sent,
 * and parses the raw payloads that come back. All network code lives in the
 * driver for `channel`, so replay feeds the same payloads into the same parser.
 */
export interface LlmAdapter {
  provider: string;
  channel: string;
  /** Used when the node's model param is empty. */
  defaultModel?: string;
  buildRequest(history: ChatTurn[], params: JsonObject): Json;
  parser(): LlmStreamParser;
}
