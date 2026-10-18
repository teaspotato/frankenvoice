import type { NodeDef } from '../../core/node';
import type { Json, JsonObject } from '../../core/types';
import { anthropicAdapter } from './anthropic';
import { openaiCompatAdapter } from './openai';
import type { ChatTurn, LlmAdapter, LlmStreamParser } from './types';

export const llmAdapters: Record<string, LlmAdapter> = {
  [anthropicAdapter.provider]: anthropicAdapter,
  [openaiCompatAdapter.provider]: openaiCompatAdapter,
};

const DEFAULT_SYSTEM =
  'You are a voice assistant in a live spoken conversation. Your words are converted to speech, so answer in short, natural spoken sentences. No markdown, lists, or emoji.';

const anthropic = { provider: ['anthropic'] };
const openai = { provider: ['openai-compatible'] };

/**
 * Streams a reply for each user turn and keeps the conversation. The provider
 * is a parameter: swapping it changes only the request/stream format, never
 * the node's ports or messages. The request (full history included) is the
 * io.open payload and every raw stream payload is an io.recv, so a replay sees
 * the identical token stream with identical timing.
 */
export const llmNode: NodeDef = {
  type: 'llm',
  title: 'LLM',
  category: 'llm',
  description: 'Streaming chat model. Providers: Anthropic (Claude), any OpenAI-compatible endpoint (OpenRouter, OpenAI, local servers).',
  inputs: [
    { name: 'turn', type: 'turn' },
    { name: 'control', type: 'control' },
  ],
  outputs: [{ name: 'text', type: 'text' }],
  params: [
    { key: 'provider', label: 'Provider', type: 'select', options: Object.keys(llmAdapters), default: 'anthropic' },
    {
      key: 'endpoint',
      label: 'Endpoint',
      type: 'select',
      options: ['openrouter', 'openai', 'custom'],
      default: 'openrouter',
      when: openai,
      description: 'Named endpoint configured in the proxy (.env): base URL and key stay server-side.',
    },
    {
      key: 'model',
      label: 'Model',
      type: 'string',
      default: '',
      description: 'Empty uses the provider default (Anthropic: claude-opus-5-5). OpenAI-compatible endpoints need a model name, e.g. an OpenRouter slug.',
    },
    { key: 'effort', label: 'Effort', type: 'select', options: ['low', 'medium', 'high', 'xhigh', 'max'], default: 'low', when: anthropic },
    { key: 'maxTokens', label: 'Max tokens', type: 'number', default: 4096 },
    { key: 'fallbacks', label: 'Server-side refusal fallback', type: 'boolean', default: true, when: anthropic },
    { key: 'extraBody', label: 'Extra request body (JSON)', type: 'text', default: '', when: openai, description: 'Merged into the request, e.g. {"temperature":0.7}' },
    { key: 'system', label: 'System prompt', type: 'text', default: DEFAULT_SYSTEM },
  ],
  channels: [
    { name: 'anthropic', description: 'Messages API stream (raw SDK stream events)' },
    { name: 'openai', description: 'Chat Completions SSE (parsed data lines)' },
  ],
  requires: (p) => (p.provider === 'openai-compatible' ? [`openai/${String(p.endpoint)}`] : [String(p.provider)]),
  create(ctx) {
    const p = ctx.params as JsonObject;
    const adapter = llmAdapters[p.provider as string];
    const reqParams: JsonObject = { ...p, model: (p.model as string) || adapter?.defaultModel || '' };
    const history: ChatTurn[] = [];
    let n = 0;
    let active:
      | { io: ReturnType<typeof ctx.io.open>; rid: string; turnId?: string; parser: LlmStreamParser; text: string; stopReason: string | null }
      | undefined;

    const finish = (status: 'done' | 'cancelled' | 'error', error?: Json) => {
      if (!active) return;
      const a = active;
      active = undefined;
      if (!a.io.closed) a.io.close({ status });
      // Keep whatever the model produced, so the next request continues from it.
      const native = a.parser.native();
      if (a.text || native)
        history.push({ role: 'assistant', text: a.text, turnId: a.turnId, ...(native ? { native: { provider: adapter.provider, content: native } } : {}) });
      ctx.emit('text', { type: 'llm.end', rid: a.rid, status, stopReason: a.stopReason, text: a.text, ...(error ? { error } : {}) });
      ctx.setState('status', status === 'error' ? 'error' : 'idle');
      ctx.setState('turns', history.length);
      if (error) ctx.log('error', 'LLM request failed', error);
    };

    const handle = (events: ReturnType<LlmStreamParser['recv']>) => {
      for (const ev of events) {
        if (!active) return;
        if (ev.type === 'delta') {
          active.text += ev.text;
          ctx.emit('text', { type: 'llm.delta', rid: active.rid, text: ev.text });
        } else if (ev.type === 'end') {
          active.stopReason = ev.stopReason;
          finish('done');
        } else finish('error', ev.error);
      }
    };

    if (!adapter) ctx.log('error', `unknown LLM provider "${String(p.provider)}"`);
    ctx.setState('provider', p.provider);

    return {
      onInput(port, msg) {
        if (port === 'control') {
          if (msg.type === 'cancel') finish('cancelled');
          return;
        }
        if (msg.type === 'turn.heard') {
          // The user cut the reply off: keep only what they heard, marked as cut off.
          const i = history.findIndex((t) => t.role === 'assistant' && t.turnId === msg.turnId);
          if (i < 0) return;
          const heard = ((msg.text as string) ?? '').trim();
          if (!heard) history.splice(i, 1);
          else history[i] = { role: 'assistant', text: `${heard}—`, turnId: history[i].turnId };
          ctx.setState('turns', history.length);
          ctx.log('info', heard ? 'reply truncated to what was heard' : 'reply dropped: nothing was heard', { turnId: msg.turnId as string, heard });
          return;
        }
        if (msg.type !== 'turn.user' || !adapter) return;
        if (!reqParams.model) {
          ctx.log('error', `set a model for provider "${adapter.provider}"`);
          return;
        }
        finish('cancelled');
        if (msg.replaces) {
          // The user wasn't done: this turn replaces an earlier one and whatever followed it.
          const i = history.findIndex((t) => t.role === 'user' && t.turnId === msg.replaces);
          if (i >= 0) history.splice(i);
        }
        history.push({ role: 'user', text: msg.text as string, turnId: msg.turnId as string });

        const rid = `${ctx.id}-r${n++}`;
        let request: Json;
        try {
          request = adapter.buildRequest(history, reqParams);
        } catch (e) {
          ctx.log('error', `could not build request: ${String(e)}`);
          return;
        }
        ctx.emit('text', { type: 'llm.start', rid, turnId: (msg.turnId as string) ?? null });
        ctx.setState('status', 'generating');
        const parser = adapter.parser();
        const io = ctx.io.open(adapter.channel, request, {
          onRecv(data) {
            if (active?.io === io) handle(parser.recv(data));
          },
          onClose(data) {
            if (active?.io === io) handle(parser.close(data));
          },
        });
        active = { io, rid, turnId: msg.turnId as string, parser, text: '', stopReason: null };
      },
      stop() {
        finish('cancelled');
      },
    };
  },
};
