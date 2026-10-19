import type { Json, JsonObject } from '../../core/types';
import type { ChatTurn, LlmAdapter, LlmStreamEvent } from './types';

/**
 * OpenAI-compatible Chat Completions (OpenRouter, OpenAI, vLLM, Ollama, LM
 * Studio, ...). Payloads are the parsed `data:` lines of the SSE stream;
 * `{"$done": true}` stands for the literal `[DONE]`.
 */
export const openaiCompatAdapter: LlmAdapter = {
  provider: 'openai-compatible',
  channel: 'openai',
  buildRequest(history: ChatTurn[], p: JsonObject): Json {
    const messages: JsonObject[] = [];
    if (p.system) messages.push({ role: 'system', content: p.system });
    for (const t of history) {
      const last = messages[messages.length - 1];
      if (last?.role === t.role && t.role === 'user') last.content = `${last.content as string}\n${t.text}`;
      else messages.push({ role: t.role, content: t.text });
    }
    let extra: JsonObject = {};
    if (typeof p.extraBody === 'string' && p.extraBody.trim()) extra = JSON.parse(p.extraBody) as JsonObject;
    return {
      endpoint: p.endpoint,
      body: {
        model: p.model,
        messages,
        stream: true,
        ...(p.maxTokens ? { max_tokens: p.maxTokens } : {}),
        ...extra,
      },
    };
  },
  parser() {
    let stopReason: string | null = null;
    let ended = false;
    const end = (): LlmStreamEvent[] => {
      if (ended) return [];
      ended = true;
      return [{ type: 'end', stopReason }];
    };
    return {
      recv(data): LlmStreamEvent[] {
        const c = data as JsonObject;
        if (c.$done) return end();
        if (c.error) return [{ type: 'error', error: c.error }];
        const out: LlmStreamEvent[] = [];
        for (const choice of (c.choices as JsonObject[] | undefined) ?? []) {
          if ((choice.index ?? 0) !== 0) continue;
          const delta = choice.delta as JsonObject | undefined;
          // Reasoning models stream their thinking as `reasoning` (OpenRouter) or `reasoning_content` (DeepSeek, vLLM).
          const reasoning = delta?.reasoning ?? delta?.reasoning_content;
          if (typeof reasoning === 'string' && reasoning) out.push({ type: 'thinking' });
          const text = delta?.content;
          if (typeof text === 'string' && text) out.push({ type: 'delta', text });
          if (choice.finish_reason) stopReason = choice.finish_reason as string;
        }
        return out;
      },
      close(data) {
        const d = data as JsonObject | null;
        return d?.error ? [{ type: 'error', error: d.error }] : end();
      },
      native() {
        return null;
      },
    };
  },
};
