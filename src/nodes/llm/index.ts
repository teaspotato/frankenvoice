import type { NodeDef } from '../../core/node';
import type { Json, JsonObject, Msg } from '../../core/types';
import { anthropicAdapter } from './anthropic';
import { openaiCompatAdapter } from './openai';
import type { ChatTurn, LlmAdapter, LlmStreamParser } from './types';
import {
  choiceRequest,
  DEFAULT_CHOICE_INSTRUCTIONS,
  DEFAULT_OPENER_CHOICES,
  DEFAULT_OPENER_LINES,
  parseOpenerChoices,
  parseOpenerLines,
  pickOpener,
  readChoice,
  type OpenerPick,
} from './opener';

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
 * With the opener on, each reply first gets a canned line ("Hmm, let me
 * think.") picked locally: the user's words are embedded by a small sentence
 * encoder (channel `embed`, recorded like any io) and matched against example
 * turns for each line. The request goes out once the pick is known (a few ms)
 * and tells the model what was already said; when the reply goes live, the
 * line is emitted first as `llm.delta` with `opener`, before any token.
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
    {
      key: 'opener',
      label: 'Opener',
      type: 'select',
      options: ['canned', 'choice', 'off'],
      default: 'canned',
      description: 'Say a canned line the moment the turn commits; the model is told it was said. canned: picked in the browser from the user\'s words by a sentence encoder. choice: picked by TypeSafe Choice from the whole conversation.',
    },
    {
      key: 'openerChoices',
      label: 'Opener options',
      type: 'text',
      default: DEFAULT_OPENER_CHOICES,
      when: { opener: ['choice'] },
      description: 'line = when it fits ("-" says nothing). Sent to TypeSafe as the Choice criteria.',
    },
    { key: 'openerInstructions', label: 'Opener question', type: 'text', default: DEFAULT_CHOICE_INSTRUCTIONS, when: { opener: ['choice'] } },
    { key: 'openerMinConfidence', label: 'Opener min confidence', type: 'number', default: 0.5, when: { opener: ['choice'] } },
    { key: 'choiceModel', label: 'TypeSafe model', type: 'string', default: 'jev-latest', when: { opener: ['choice'] } },
    {
      key: 'openerWaitMs',
      label: 'Wait for the opener (ms)',
      type: 'number',
      default: 400,
      when: { opener: ['choice'] },
      description: 'The request waits this long at most for the pick; past it, the reply goes without an opener.',
    },
    {
      key: 'openerLines',
      label: 'Opener lines',
      type: 'text',
      default: DEFAULT_OPENER_LINES,
      when: { opener: ['canned'] },
      description: 'line = example | example | … ("-" says nothing). The nearest example by meaning picks the line.',
    },
    { key: 'openerMinScore', label: 'Opener min similarity', type: 'number', default: 0.3, when: { opener: ['canned'] } },
    {
      key: 'embedModel',
      label: 'Opener encoder',
      type: 'string',
      default: 'Xenova/all-MiniLM-L6-v2',
      when: { opener: ['canned'] },
      description: 'Hugging Face repo with onnx/model_quantized.onnx and vocab.txt (BERT-style, mean pooling). Runs in the browser.',
    },
  ],
  channels: [
    { name: 'anthropic', description: 'Messages API stream (raw SDK stream events)' },
    { name: 'openai', description: 'Chat Completions SSE (parsed data lines)' },
    { name: 'embed', description: 'Sentence encoder in the browser: texts out, unit vectors back' },
    { name: 'typesafe', description: 'TypeSafe primitives: a Choice question over the conversation, one JSON answer' },
  ],
  requires: (p) => [
    ...(p.provider === 'openai-compatible' ? [`openai/${String(p.endpoint)}`] : [String(p.provider)]),
    ...(p.opener === 'choice' ? ['typesafe'] : []),
  ],
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
     * committed turn (live). The request goes out once the opener is picked.
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
      /** undefined until picked; null: no opener. */
      pick?: OpenerPick | null;
      held: Events;
      text: string;
      stopReason: string | null;
      /** The opener was said as the start of this reply. */
      opened?: boolean;
    }
    let active: Reply | undefined;
    let draft: Reply | undefined;

    // Opener. canned: example turns are embedded once at start, each user text once.
    // choice: TypeSafe reads the conversation up to the user's words and picks an option.
    const useOpener = p.opener === 'canned' || p.opener === 'choice';
    const examples = p.opener === 'canned' ? parseOpenerLines(String(p.openerLines ?? '')) : [];
    const choices = p.opener === 'choice' ? parseOpenerChoices(String(p.openerChoices ?? '')) : [];
    let exampleVecs: number[][] | undefined;
    const picks = new Map<string, OpenerPick | null>();
    const waiting = new Map<string, ((pick: OpenerPick | null) => void)[]>();
    const embed = (texts: string[], done: (vecs: number[][] | null) => void) => {
      let got = false;
      ctx.io.open(
        'embed',
        { model: p.embedModel as string, texts },
        {
          onRecv(data) {
            const v = (data as { vectors?: number[][] } | null)?.vectors;
            if (got || !v) return;
            got = true;
            done(v);
          },
          onClose(data) {
            if (got) return;
            got = true;
            if (data && (data as { error?: unknown }).error) ctx.log('warn', 'opener encoder failed', data);
            done(null);
          },
        },
      );
    };
    /** Asks for the opener for `turns` (the conversation ending with the user's words). */
    const ask = (turns: ChatTurn[], done: (pick: OpenerPick | null) => void) => {
      if (p.opener === 'canned') {
        const text = turns[turns.length - 1].text;
        return embed([text], (vecs) => done(vecs?.[0] ? pickOpener(vecs[0], examples, exampleVecs!, Number(p.openerMinScore)) : null));
      }
      let got = false;
      ctx.io.open(
        'typesafe',
        { path: '/v1/systemone', body: choiceRequest(turns, choices, String(p.openerInstructions ?? ''), String(p.choiceModel ?? 'jev-latest')) },
        {
          onRecv(data) {
            if (got) return;
            got = true;
            done(readChoice(data, Number(p.openerMinConfidence)));
          },
          onClose(data) {
            if (got) return;
            got = true;
            if (data && (data as { error?: unknown }).error) ctx.log('warn', 'opener choice failed', data);
            done(null);
          },
        },
      );
    };
    const ready = () => (p.opener === 'canned' ? !!exampleVecs : p.opener === 'choice' && choices.length > 0);
    /**
     * Calls back with the opener for the conversation `turns`: at once if it is known or can't be
     * known, else when it is picked, or with none once `openerWaitMs` has passed.
     */
    const whenPicked = (turns: ChatTurn[], go: (pick: OpenerPick | null) => void) => {
      if (!useOpener || !ready()) return go(null);
      const text = turns[turns.length - 1].text;
      // The encoder only reads the user's words; Choice reads the conversation.
      const key = p.opener === 'canned' ? text : turns.map((t) => `${t.role}:${t.text}`).join('\n');
      if (picks.has(key)) return go(picks.get(key)!);
      let called = false;
      const once = (pick: OpenerPick | null) => {
        if (called) return;
        called = true;
        go(pick);
      };
      // Only the network pick is capped: the local encoder answers in milliseconds.
      const wait = p.opener === 'choice' ? Number(p.openerWaitMs) : -1;
      if (wait >= 0) ctx.setTimer(wait, () => once(null));
      const w = waiting.get(key);
      if (w) return void w.push(once);
      waiting.set(key, [once]);
      const t0 = ctx.now();
      ask(turns, (pick) => {
        picks.set(key, pick);
        ctx.emit('text', { type: 'llm.opener', text, ...(pick ?? { line: null }), ms: Math.round(ctx.now() - t0) });
        const cbs = waiting.get(key) ?? [];
        waiting.delete(key);
        for (const cb of cbs) cb(pick);
      });
    };

    const isCurrent = (r: Reply) => r === active || r === draft;

    const finish = (status: 'done' | 'cancelled' | 'error', error?: Json) => {
      if (!active) return;
      const a = active;
      active = undefined;
      if (a.io && !a.io.closed) a.io.close({ status });
      // Keep whatever the model produced, so the next request continues from it.
      // The opener isn't in the model's own content blocks, so a reply that began with one is kept as plain text.
      const native = a.opened || !a.parser ? null : a.parser.native();
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
        } else finish('error', ev.error);
      }
    };

    /** Say the reply's opener, if it has one: before anything the model wrote. */
    const speakOpener = (r: Reply) => {
      const line = r.pick?.line;
      if (!line || r.opened || r.text) return;
      r.opened = true;
      r.text = `${line} `;
      ctx.emit('text', { type: 'llm.delta', rid: r.rid, text: `${line} `, opener: true });
    };

    /** The history a request for `user` sees: up to the turn it replaces, if any. */
    const historyFor = (user: ChatTurn, replaces?: unknown) => {
      const i = replaces ? history.findIndex((t) => t.role === 'user' && t.turnId === replaces) : -1;
      return [...(i >= 0 ? history.slice(0, i) : history), user];
    };

    /** Pick the opener, then send the request (told what was already said). */
    const launch = (r: Reply, turns: ChatTurn[]) => {
      whenPicked(turns, (pick) => {
        if (!isCurrent(r) || r.io) return;
        r.pick = pick;
        const line = pick?.line;
        const params = line
          ? { ...reqParams, system: `${String(reqParams.system ?? '')}\n\nYou have already said aloud: "${line}" Continue your reply from there, without repeating it or opening with another acknowledgement.`.trim() }
          : reqParams;
        let request: Json;
        try {
          request = adapter.buildRequest(turns, params);
        } catch (e) {
          ctx.log('error', `could not build request: ${String(e)}`);
          if (r === draft) draft = undefined;
          else if (r === active) finish('error', String(e));
          return;
        }
        const parser = adapter.parser();
        const io = ctx.io.open(adapter.channel, request, {
          onRecv(data) {
            route(r, parser.recv(data));
          },
          onClose(data) {
            route(r, parser.close(data));
          },
        });
        r.io = io;
        r.parser = parser;
        if (r.live) speakOpener(r);
      });
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
      start() {
        const lines = [...examples, ...choices].map((e) => e.line).filter((l): l is string => !!l);
        // Tell TTS the lines, so it can have them ready.
        if (lines.length) ctx.emit('text', { type: 'llm.opener_lines', lines: [...new Set(lines)] });
        if (!examples.length) return;
        embed(
          examples.map((e) => e.text),
          (vecs) => {
            if (vecs && vecs.length === examples.length) exampleVecs = vecs;
            ctx.setState('opener', exampleVecs ? 'ready' : 'unavailable');
          },
        );
      },
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
          kept.live = true;
          kept.turnId = turnId;
          active = kept;
          if (kept.io) speakOpener(kept);
          handle(held);
          return;
        }
        const user: ChatTurn = { role: 'user', text: msg.text as string, turnId };
        history.push(user);
        const r = reply(user, { live: true, turnId });
        ctx.emit('text', { type: 'llm.start', rid: r.rid, turnId: turnId ?? null });
        ctx.setState('status', 'generating');
        active = r;
        launch(r, [...history]);
      },
      stop() {
        dropDraft('stopped');
        finish('cancelled');
      },
    };
  },
};
