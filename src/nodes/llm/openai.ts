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

/**
 * OpenAI-compatible Responses API (`/responses`: OpenAI, OpenRouter, vLLM, ...).
 * Stateless: the whole history goes in `input` every time (`store: false`), as
 * with Chat Completions, so a request is all a replay needs. Payloads are the
 * parsed `data:` lines; every one carries its event `type`.
 */
export const openaiResponsesAdapter: LlmAdapter = {
  provider: 'openai-compatible',
  channel: 'openai-responses',
  buildRequest(history: ChatTurn[], p: JsonObject): Json {
    const input: JsonObject[] = [];
    for (const t of history) {
      const last = input[input.length - 1];
      if (last?.role === t.role && t.role === 'user') last.content = `${last.content as string}\n${t.text}`;
      else input.push({ role: t.role, content: t.text });
    }
    let extra: JsonObject = {};
    if (typeof p.extraBody === 'string' && p.extraBody.trim()) extra = JSON.parse(p.extraBody) as JsonObject;
    return {
      endpoint: p.endpoint,
      body: {
        model: p.model,
        ...(p.system ? { instructions: p.system } : {}),
        input,
        stream: true,
        store: false,
        ...(p.maxTokens ? { max_output_tokens: p.maxTokens } : {}),
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
    /** One error ends the reply; a later error or close adds nothing. */
    const fail = (error: Json): LlmStreamEvent[] => {
      if (ended) return [];
      ended = true;
      return [{ type: 'error', error }];
    };
    return {
      recv(data): LlmStreamEvent[] {
        const e = data as JsonObject;
        switch (e.type) {
          case 'response.output_text.delta':
          // A refusal is spoken like any other reply.
          case 'response.refusal.delta':
            return typeof e.delta === 'string' && e.delta ? [{ type: 'delta', text: e.delta }] : [];
          case 'response.output_item.added':
            return (e.item as JsonObject | undefined)?.type === 'reasoning' ? [{ type: 'thinking' }] : [];
          case 'response.reasoning_text.delta':
          case 'response.reasoning_summary_text.delta':
            return [{ type: 'thinking' }];
          case 'response.completed':
          case 'response.done': // OpenRouter's documented example
            stopReason = 'completed';
            return end();
          case 'response.incomplete': {
            const why = ((e.response as JsonObject | undefined)?.incomplete_details as JsonObject | undefined)?.reason;
            stopReason = typeof why === 'string' ? why : 'incomplete';
            return end();
          }
          case 'response.failed':
            return fail(((e.response as JsonObject | undefined)?.error as Json) ?? e);
          case 'error':
            return fail((e.error as Json) ?? e);
          default:
            if (e.$done) return end();
            // Servers that answer with a bare error object instead of an event.
            return e.error ? fail(e.error) : [];
        }
      },
      close(data) {
        const d = data as JsonObject | null;
        if (d?.error) return fail(d.error);
        // A stream cut off before its terminal event (e.g. a server-side generation error) is not a finished reply.
        return ended ? [] : fail({ message: 'stream ended without response.completed' });
      },
      native() {
        return null;
      },
    };
  },
};
