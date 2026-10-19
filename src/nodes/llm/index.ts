import type { NodeDef } from '../../core/node';
import type { Json, JsonObject, Msg } from '../../core/types';
import { anthropicAdapter } from './anthropic';
import { openaiCompatAdapter } from './openai';
import type { ChatTurn, LlmAdapter, LlmStreamParser } from './types';

const words = (s: string) => s.split(/\s+/).filter(Boolean);
const norm = (w: string) => w.toLowerCase().replace(/[^\p{L}\p{N}']/gu, '');

/**
 * The part of `heard` that is the start of `reply`: what was said before the
 * reply's own words (a filler from the opener stage) is cut off. Heard text is
 * whole words of what was spoken, so a suffix of it matches a prefix of the reply
 * (empty when none of the reply was heard).
 */
export function heardOfReply(heard: string, reply: string): string {
  const h = words(heard);
  const r = words(reply).map(norm);
  for (let i = 0; i <= h.length; i++) {
    const rest = h.slice(i);
    if (rest.length <= r.length && rest.every((w, j) => norm(w) === r[j])) return rest.join(' ');
  }
  return '';
}

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
 *
 * A `turn.draft` (the user's words before their turn is committed) starts a
 * request early, without touching the history or emitting anything but
 * `llm.draft` / `llm.draft_end`: the stream is held back. When the committed
 * `turn.user` names the draft, the held stream is emitted at once as an
 * ordinary reply (llm.start with `draft`, then its deltas) and the rest
 * streams as usual. Any other turn, a new draft or a drop discards it.
 *
 * The first sign of reasoning in a reply (a thinking block, reasoning tokens)
 * is emitted as `llm.thinking`, for the opener stage to fill the wait with.
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
      description: 'Empty uses the provider default (Anthropic: claude-haiku-5-5). OpenAI-compatible endpoints need a model name, e.g. an OpenRouter slug.',
    },
    { key: 'effort', label: 'Effort', type: 'select', options: ['low', 'medium', 'high', 'xhigh', 'max'], default: 'low', when: anthropic },
    {
      key: 'thinking',
      label: 'Thinking',
      type: 'select',
      options: ['model default', 'off'],
      default: 'off',
      when: anthropic,
      description: 'Off sends thinking {type: "disabled"} (Haiku) or {type: "between_tools"} (Sonnet). Opus cannot turn it off.',
    },
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
    type Io = ReturnType<typeof ctx.io.open>;
    type Events = ReturnType<LlmStreamParser['recv']>;
    /**
     * One reply: started on a draft (not live: its stream is held back) or on a
     * committed turn (live).
     */
    interface Reply {
      rid: string;
      user: ChatTurn;
      replaces?: string;
      draftId?: string;
      turnId?: string;
      t0: number;
      live: boolean;
      io?: Io;
      parser?: LlmStreamParser;
      held: Events;
      text: string;
      stopReason: string | null;
      /** llm.thinking was emitted for this reply. */
      thinking?: boolean;
    }
    let active: Reply | undefined;
    let draft: Reply | undefined;

    const finish = (status: 'done' | 'cancelled' | 'error', error?: Json) => {
      if (!active) return;
      const a = active;
      active = undefined;
      if (a.io && !a.io.closed) a.io.close({ status });
      // Keep whatever the model produced, so the next request continues from it.
      const native = a.parser ? a.parser.native() : null;
      if (a.text || native)
        history.push({ role: 'assistant', text: a.text, turnId: a.turnId, ...(native ? { native: { provider: adapter.provider, content: native } } : {}) });
      ctx.emit('text', { type: 'llm.end', rid: a.rid, status, stopReason: a.stopReason, text: a.text, ...(error ? { error } : {}) });
      ctx.setState('status', status === 'error' ? 'error' : 'idle');
      ctx.setState('turns', history.length);
      if (error) ctx.log('error', 'LLM request failed', error);
    };

    const handle = (events: Events) => {
      for (const ev of events) {
        if (!active) return;
        if (ev.type === 'delta') {
          active.text += ev.text;
          ctx.emit('text', { type: 'llm.delta', rid: active.rid, text: ev.text });
        } else if (ev.type === 'end') {
          active.stopReason = ev.stopReason;
          finish('done');
        } else if (ev.type === 'error') finish('error', ev.error);
        else if (ev.type === 'thinking' && !active.thinking) {
          active.thinking = true;
          ctx.emit('text', { type: 'llm.thinking', rid: active.rid });
        }
      }
    };

    /** The history a request for `user` sees: up to the turn it replaces, if any. */
    const historyFor = (user: ChatTurn, replaces?: unknown) => {
      const i = replaces ? history.findIndex((t) => t.role === 'user' && t.turnId === replaces) : -1;
      return [...(i >= 0 ? history.slice(0, i) : history), user];
    };

    /** Send the request for `r`. */
    const launch = (r: Reply, turns: ChatTurn[]) => {
      let request: Json;
      try {
        request = adapter.buildRequest(turns, reqParams);
      } catch (e) {
        ctx.log('error', `could not build request: ${String(e)}`);
        if (r === draft) draft = undefined;
        else if (r === active) finish('error', String(e));
        return;
      }
      const parser = adapter.parser();
      r.io = ctx.io.open(adapter.channel, request, {
        onRecv(data) {
          route(r, parser.recv(data));
        },
        onClose(data) {
          route(r, parser.close(data));
        },
      });
      r.parser = parser;
    };
    const route = (r: Reply, events: Events) => {
      if (r === active) handle(events);
      else if (r === draft) r.held.push(...events);
    };
    const dropDraft = (why: string) => {
      if (!draft) return;
      const d = draft;
      draft = undefined;
      if (d.io && !d.io.closed) d.io.close({ status: 'cancelled' });
      const tokens = d.held.filter((e) => e.type === 'delta').length;
      ctx.emit('text', { type: 'llm.draft_end', rid: d.rid, draftId: d.draftId!, status: 'dropped', why, chunks: tokens });
      ctx.log('debug', 'draft dropped', { draftId: d.draftId!, why, chunks: tokens });
    };
    const reply = (user: ChatTurn, extra: Partial<Reply>): Reply => ({ rid: `${ctx.id}-r${n++}`, user, t0: ctx.now(), live: false, held: [], text: '', stopReason: null, ...extra });
    const startDraft = (msg: Msg) => {
      dropDraft('redrafted');
      const user: ChatTurn = { role: 'user', text: msg.text as string };
      const d = reply(user, { draftId: msg.draftId as string, ...(msg.replaces ? { replaces: msg.replaces as string } : {}) });
      ctx.emit('text', { type: 'llm.draft', rid: d.rid, draftId: d.draftId!, text: user.text });
      draft = d;
      launch(d, historyFor(user, msg.replaces));
    };

    if (!adapter) ctx.log('error', `unknown LLM provider "${String(p.provider)}"`);
    ctx.setState('provider', p.provider);

    return {
      onInput(port, msg) {
        if (port === 'control') {
          if (msg.type === 'cancel') {
            dropDraft('cancelled');
            finish('cancelled');
          }
          return;
        }
        if (msg.type === 'turn.draft') {
          if (adapter && reqParams.model) startDraft(msg);
          return;
        }
        if (msg.type === 'turn.draft_drop') {
          if (draft?.draftId === msg.draftId) dropDraft((msg.why as string) ?? 'dropped');
          return;
        }
        if (msg.type === 'turn.heard') {
          // The user cut the reply off: keep only what they heard, marked as cut off.
          const i = history.findIndex((t) => t.role === 'assistant' && t.turnId === msg.turnId);
          if (i < 0) return;
          // Anything said before the reply's own words (an opener the model never knew of) is not part of it.
          const heard = heardOfReply(((msg.text as string) ?? '').trim(), history[i].text);
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
        const kept = msg.draft && draft?.draftId === msg.draft ? draft : undefined;
        if (!kept) dropDraft('turn committed without it');
        if (msg.replaces) {
          // The user wasn't done: this turn replaces an earlier one and whatever followed it.
          const i = history.findIndex((t) => t.role === 'user' && t.turnId === msg.replaces);
          if (i >= 0) history.splice(i);
        }
        const turnId = msg.turnId as string;
        if (kept) {
          // The draft's words stand: keep its reply. The history records what the request said.
          draft = undefined;
          history.push({ ...kept.user, turnId });
          const held = kept.held;
          kept.held = [];
          ctx.emit('text', { type: 'llm.draft_end', rid: kept.rid, draftId: kept.draftId!, status: 'kept', chunks: held.filter((e) => e.type === 'delta').length });
          ctx.emit('text', { type: 'llm.start', rid: kept.rid, turnId: turnId ?? null, draft: kept.draftId!, headStartMs: Math.round(ctx.now() - kept.t0) });
          ctx.setState('status', 'generating');
          kept.turnId = turnId;
          active = kept;
          kept.live = true;
          handle(held);
          return;
        }
        const user: ChatTurn = { role: 'user', text: msg.text as string, turnId };
        history.push(user);
        const r = reply(user, { turnId });
        ctx.emit('text', { type: 'llm.start', rid: r.rid, turnId: turnId ?? null });
        ctx.setState('status', 'generating');
        active = r;
        r.live = true;
        launch(r, [...history]);
      },
      stop() {
        dropDraft('stopped');
        finish('cancelled');
      },
    };
  },
};
