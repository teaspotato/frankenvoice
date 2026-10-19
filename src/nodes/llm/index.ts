import type { NodeDef } from '../../core/node';
import type { Json, JsonObject, Msg } from '../../core/types';
import { anthropicAdapter } from './anthropic';
import { openaiCompatAdapter, openaiResponsesAdapter } from './openai';
import type { ChatTurn, LlmAdapter, LlmStreamParser } from './types';
import { DEFAULT_DELEGATE_PROMPT, WORKER_TOOLS, toolCalls, workerStatus, workerUpdate, type ToolCall, type WorkerInfo } from './delegate';

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

/**
 * Cut an assistant reply (possibly several history turns, when it called tools)
 * down to the `heard` words of its text. Tool calls stay, so their results keep
 * a call to answer; a turn left with nothing is dropped. Returns the history.
 */
export function cutToHeard(history: ChatTurn[], indexes: number[], heard: string): ChatTurn[] {
  const said = words(heard);
  let pos = 0;
  const texts = indexes.map((i) => {
    const k = Math.min(words(history[i].text).length, said.length - pos);
    const text = said.slice(pos, pos + k).join(' ');
    pos += k;
    return text;
  });
  // Marked as cut off where the user stopped hearing it.
  const last = texts.map((t) => !!t).lastIndexOf(true);
  if (last >= 0) texts[last] += '—';
  const out = new Map<number, ChatTurn | null>();
  indexes.forEach((i, j) => {
    const t = history[i];
    const text = texts[j];
    const calls = Array.isArray(t.native?.content) ? (t.native!.content as JsonObject[]).filter((b) => b?.type === 'tool_use') : [];
    if (!text && !calls.length) out.set(i, null);
    else
      out.set(i, {
        role: 'assistant',
        text,
        turnId: t.turnId,
        rid: t.rid,
        ...(calls.length ? { native: { provider: t.native!.provider, content: [...(text ? [{ type: 'text', text }] : []), ...calls] } } : {}),
      });
  });
  return history.flatMap((t, i) => (out.has(i) ? (out.get(i) ? [out.get(i)!] : []) : [t]));
}

export const llmAdapters: Record<string, LlmAdapter> = {
  [anthropicAdapter.provider]: anthropicAdapter,
  [openaiCompatAdapter.provider]: openaiCompatAdapter,
};

const DEFAULT_SYSTEM =
  'You are a voice assistant in a live spoken conversation. Your words are converted to speech, so answer in short, natural spoken sentences. No markdown, lists, or emoji.';

/** The adapter for these params: OpenAI-compatible endpoints speak Chat Completions or Responses. */
export function llmAdapter(p: JsonObject): LlmAdapter {
  if (p.provider === 'openai-compatible' && p.api === 'responses') return openaiResponsesAdapter;
  return llmAdapters[p.provider as string];
}

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
    { name: 'updates', type: 'workers', description: "The workers' progress, answers and results" },
    { name: 'vad', type: 'vad', description: 'When the user speaks, so worker results wait for a pause' },
  ],
  outputs: [
    { name: 'text', type: 'text' },
    { name: 'workers', type: 'workers', description: 'worker.start / worker.send / worker.stop for the workers' },
  ],
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
      key: 'api',
      label: 'API',
      type: 'select',
      options: ['chat', 'responses'],
      default: 'chat',
      when: openai,
      description: 'chat: /chat/completions, which every compatible server has. responses: /responses (OpenAI, OpenRouter, vLLM, ...), sent stateless with the whole history.',
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
      key: 'delegate',
      label: 'Delegation',
      type: 'text',
      default: null,
      description:
        'Set by the Workers stage: {prompt, progress, quietMs, maxWorkers}. With it, the model gets the worker tools and the workers\' updates (Anthropic only).',
    },
  ],
  channels: [
    { name: 'anthropic', description: 'Messages API stream (raw SDK stream events)' },
    { name: 'openai', description: 'Chat Completions SSE (parsed data lines)' },
    { name: 'openai-responses', description: 'Responses API SSE (parsed data lines)' },
  ],
  requires: (p) => (p.provider === 'openai-compatible' ? [`openai/${String(p.endpoint)}`] : [String(p.provider)]),
  create(ctx) {
    const p = ctx.params as JsonObject;
    const adapter = llmAdapter(p);
    const reqParams: JsonObject = { ...p, model: (p.model as string) || adapter?.defaultModel || '' };
    // Delegation: the worker tools, and how the workers' updates come back.
    const delegation = p.delegate && typeof p.delegate === 'object' && adapter?.provider === 'anthropic' ? (p.delegate as JsonObject) : null;
    if (p.delegate && !delegation) ctx.log('warn', 'delegation needs the Anthropic provider: the worker tools are off');
    if (delegation) {
      reqParams.tools = WORKER_TOOLS;
      reqParams.system = `${String(p.system ?? '')}\n\n${String(delegation.prompt ?? DEFAULT_DELEGATE_PROMPT)}`.trim();
    }
    const progressPolicy = String(delegation?.progress ?? 'context');
    const quietMs = Number(delegation?.quietMs ?? 800);
    const maxWorkers = Number(delegation?.maxWorkers ?? 4);
    const maxToolRounds = 4;
    /** What each worker is doing, from its updates (for check_worker). */
    const workers = new Map<string, WorkerInfo>();
    let workerNo = 0;
    /** Worker updates not yet in the history; `speak`: one of them wants a reply of its own. */
    let notes: { turns: ChatTurn[]; workers: string[]; speak: boolean } = { turns: [], workers: [], speak: false };
    let floor = 'listening';
    /** Turn-taking holds words of the user's it hasn't committed yet. */
    let userWords = false;
    /** Set while a committed turn is being handled: news waits for its reply. */
    let committing = false;
    let userSpeaking = false;
    let lastSpeech = -Infinity;
    let deliverTimer: { clear(): void } | undefined;
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
      /** Text of the current request (a reply that calls tools makes several). */
      round: string;
      rounds: number;
      /** A reply to workers' news, started on its own rather than for a user turn: the news it carried. */
      news?: { turns: ChatTurn[]; workers: string[] };
    }
    let active: Reply | undefined;
    let draft: Reply | undefined;

    /** End the active reply. `unheard`: nothing of it was played, so it leaves no trace in the history. */
    const finish = (status: 'done' | 'cancelled' | 'error', error?: Json, unheard = false) => {
      if (!active) return;
      const a = active;
      active = undefined;
      if (a.io && !a.io.closed) a.io.close({ status });
      if (unheard) a.round = '';
      // Keep whatever the model produced, so the next request continues from it.
      // Tool calls that were never run (cut off, or past the round limit) are left out: they have no results.
      let native = a.parser && !unheard ? a.parser.native() : null;
      if (Array.isArray(native)) native = (native as JsonObject[]).filter((b) => b?.type !== 'tool_use');
      if (Array.isArray(native) && !native.length) native = null;
      if (a.round || native)
        history.push({ role: 'assistant', text: a.round, turnId: a.turnId, rid: a.rid, ...(native ? { native: { provider: adapter.provider, content: native } } : {}) });
      ctx.emit('text', { type: 'llm.end', rid: a.rid, status, stopReason: a.stopReason, text: a.text, ...(error ? { error } : {}) });
      ctx.setState('status', status === 'error' ? 'error' : 'idle');
      ctx.setState('turns', history.length);
      if (error) ctx.log('error', 'LLM request failed', error);
      deliver();
    };

    /** Run a reply's worker tool calls: each is answered at once, the workers do the work. */
    const runTool = (r: Reply, c: ToolCall): JsonObject => {
      const result = (content: string, isError = false): JsonObject => ({ type: 'tool_result', tool_use_id: c.id, content, ...(isError ? { is_error: true } : {}) });
      const id = String(c.input.worker_id ?? '');
      const w = workers.get(id);
      const running = [...workers.values()].filter((x) => x.status === 'running').length;
      let out: JsonObject;
      if (c.name === 'spawn_worker') {
        if (running >= maxWorkers) out = result(`${running} workers are already running, the most at once. Wait for one to finish, or stop one.`, true);
        else {
          const workerId = `w${++workerNo}`;
          workers.set(workerId, { status: 'running', task: String(c.input.task ?? ''), rid: r.rid, t0: ctx.now(), steps: [], plan: '' });
          ctx.emit('workers', { type: 'worker.start', workerId, task: String(c.input.task ?? ''), rid: r.rid });
          out = result(`Started worker ${workerId}. Its result will come as a worker_update; keep talking with the user meanwhile.`);
        }
      } else if (['message_worker', 'check_worker', 'stop_worker'].includes(c.name) && !w) out = result(`There is no worker "${id}".`, true);
      else if (w && c.name === 'check_worker') out = result(workerStatus(id, w, ctx.now()));
      else if (w && c.name === 'message_worker' && w.status !== 'running' && running >= maxWorkers)
        out = result(`${running} workers are already running, the most at once. Wait for one to finish, or stop one.`, true);
      else if (w && c.name === 'message_worker') {
        const was = w.status;
        w.status = 'running';
        if (was !== 'running') w.t0 = ctx.now();
        ctx.emit('workers', { type: 'worker.send', workerId: id, message: String(c.input.message ?? ''), rid: r.rid });
        out = result(was === 'running' ? 'Sent to the worker.' : 'Sent: the worker picks up again with it.');
      } else if (w && c.name === 'stop_worker') {
        w.status = 'stopped';
        ctx.emit('workers', { type: 'worker.stop', workerId: id, rid: r.rid });
        out = result('Stopped.');
      } else out = result(`Unknown tool "${c.name}".`, true);
      ctx.emit('text', { type: 'llm.tool', rid: r.rid, name: c.name, input: c.input, result: out.content as string, ...(out.is_error ? { error: true } : {}) });
      return out;
    };
    /** The reply stopped to call tools: run them and continue it with their results. */
    const toolRound = (r: Reply): boolean => {
      const native = r.parser?.native() ?? null;
      const calls = toolCalls(native);
      if (!delegation || !calls.length || r.rounds >= maxToolRounds) return false;
      r.rounds++;
      if (r.io && !r.io.closed) r.io.close({ status: 'done' });
      history.push({ role: 'assistant', text: r.round, turnId: r.turnId, rid: r.rid, native: { provider: adapter.provider, content: native } });
      const results = calls.map((c) => runTool(r, c));
      history.push({ role: 'user', text: '', native: { provider: adapter.provider, content: results } });
      ctx.setState('turns', history.length);
      r.round = '';
      r.thinking = false;
      send(r, [...history]);
      return true;
    };

    /** Is the conversation free for the workers' news: no reply, no draft, the user quiet for a moment? */
    const deliver = () => {
      deliverTimer?.clear();
      deliverTimer = undefined;
      if (!notes.speak || active || draft || committing || userWords || floor !== 'listening' || userSpeaking || !adapter || !reqParams.model) return;
      const wait = lastSpeech + quietMs - ctx.now();
      if (wait > 0) return void (deliverTimer = ctx.setTimer(wait, deliver, 'worker-update'));
      const n = notes;
      notes = { turns: [], workers: [], speak: false };
      history.push(...n.turns);
      const r = reply(n.turns[n.turns.length - 1], {});
      r.news = { turns: n.turns, workers: n.workers };
      ctx.emit('text', { type: 'llm.start', rid: r.rid, turnId: null, workers: n.workers });
      ctx.setState('status', 'generating');
      active = r;
      r.live = true;
      send(r, [...history]);
    };
    /** Updates the model hasn't seen go into the history ahead of the user's turn. */
    const flushNotes = () => {
      if (!notes.turns.length) return;
      history.push(...notes.turns);
      notes = { turns: [], workers: [], speak: false };
    };

    const handle = (events: Events) => {
      for (const ev of events) {
        if (!active) return;
        if (ev.type === 'delta') {
          active.text += ev.text;
          active.round += ev.text;
          ctx.emit('text', { type: 'llm.delta', rid: active.rid, text: ev.text });
        } else if (ev.type === 'end') {
          active.stopReason = ev.stopReason;
          if (ev.stopReason === 'tool_use' && toolRound(active)) return;
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
      send(r, turns);
    };
    /** Send one request for `r` and route its stream to it. */
    const send = (r: Reply, turns: ChatTurn[], params: JsonObject = reqParams): boolean => {
      let request: Json;
      try {
        request = adapter.buildRequest(turns, params);
      } catch (e) {
        ctx.log('error', `could not build request: ${String(e)}`);
        if (r === draft) draft = undefined;
        else if (r === active) finish('error', String(e));
        return false;
      }
      const parser = adapter.parser();
      // Only the reply's current request counts: an earlier one (before its tool results) may still close.
      r.parser = parser;
      r.io = ctx.io.open(adapter.channel, request, {
        onRecv(data) {
          if (r.parser === parser) route(r, parser.recv(data));
        },
        onClose(data) {
          if (r.parser === parser) route(r, parser.close(data));
        },
      });
      return true;
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
      deliver();
    };
    const reply = (user: ChatTurn, extra: Partial<Reply>): Reply => ({
      rid: `${ctx.id}-r${n++}`,
      user,
      t0: ctx.now(),
      live: false,
      held: [],
      text: '',
      round: '',
      rounds: 0,
      stopReason: null,
      ...extra,
    });
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

    /** Replies taken back (their turn was replaced): the workers they started go too, with any news from them. */
    const stopWorkersOf = (removed: ChatTurn[]) => {
      const rids = new Set(removed.filter((t) => t.role === 'assistant' && t.rid).map((t) => t.rid));
      const gone = [...workers].filter(([, w]) => rids.has(w.rid) && w.status !== 'stopped');
      for (const [id, w] of gone) {
        if (w.status === 'running') ctx.emit('workers', { type: 'worker.stop', workerId: id, rid: w.rid });
        w.status = 'stopped';
      }
      const ids = new Set(gone.map(([id]) => id));
      notes.turns = notes.turns.filter((t) => !gone.some(([id]) => t.text.startsWith(`<worker_update id="${id}"`)));
      notes.workers = notes.workers.filter((id) => !ids.has(id));
      notes.speak = notes.speak && notes.turns.length > 0;
    };

    /** A committed user turn: answer it (with the draft made for it, if it still stands). */
    const commitTurn = (msg: Msg) => {
      finish('cancelled');
      // A draft went out without a worker result that has come in since: the turn is sent again with it.
      const kept = msg.draft && draft?.draftId === msg.draft && !notes.speak ? draft : undefined;
      if (!kept) dropDraft(notes.speak && msg.draft ? 'a worker result came in' : 'turn committed without it');
      if (msg.replaces) {
        // The user wasn't done: this turn replaces an earlier one and whatever followed it.
        const i = history.findIndex((t) => t.role === 'user' && t.turnId === msg.replaces);
        if (i >= 0) stopWorkersOf(history.splice(i));
      }
      const turnId = msg.turnId as string;
      // A kept draft's request went out without the workers' progress notes: those wait for the next request.
      if (!kept) flushNotes();
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
    };

    return {
      onInput(port, msg) {
        if (port === 'vad') {
          if (msg.type === 'vad.speech_start') {
            userSpeaking = true;
            // The user spoke up before a reply to news began to play: drop it, and keep the news for later.
            const r = active;
            if (r?.news && floor === 'listening') {
              finish('cancelled', undefined, true);
              const at = history.length - r.news.turns.length;
              if (at >= 0 && r.news.turns.every((t, i) => history[at + i] === t)) history.splice(at);
              notes = { turns: [...r.news.turns, ...notes.turns], workers: [...new Set([...r.news.workers, ...notes.workers])], speak: true };
            }
          } else if (msg.type === 'vad.speech_end') {
            userSpeaking = false;
            lastSpeech = ctx.now();
          } else return;
          deliver();
          return;
        }
        if (port === 'updates') {
          const kind = { 'worker.done': 'result', 'worker.error': 'error', 'worker.progress': 'progress' }[msg.type];
          const workerId = msg.workerId as string;
          const w = workers.get(workerId);
          if (!w || !delegation || w.status === 'stopped') return;
          if (msg.type === 'worker.tool' && msg.title) w.steps = [...w.steps, String(msg.title)].slice(-5);
          if (msg.type === 'worker.progress') w.plan = String(msg.text ?? '');
          if (msg.type === 'worker.done' || msg.type === 'worker.error') w.status = msg.type === 'worker.done' ? 'done' : 'error';
          if (!kind) return;
          if (kind === 'progress' && progressPolicy === 'off') return;
          notes.turns.push({ role: 'user', text: workerUpdate({ ...(msg as JsonObject), kind }) });
          if (!notes.workers.includes(workerId)) notes.workers.push(workerId);
          if (kind !== 'progress' || progressPolicy === 'speak') notes.speak = true;
          deliver();
          return;
        }
        if (msg.type === 'turn.transition') {
          floor = String(msg.to);
          deliver();
          return;
        }
        if (msg.type === 'turn.pending') {
          userWords = !!msg.words;
          deliver();
          return;
        }
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
          // By the reply when known (it may span tool calls, or answer no user turn), else the first answer to the turn.
          const first = history.findIndex((t) => t.role === 'assistant' && t.turnId === msg.turnId);
          const at = msg.rid
            ? history.flatMap((t, i) => (t.role === 'assistant' && t.rid === msg.rid ? [i] : []))
            : first >= 0
              ? [first]
              : [];
          if (!at.length) return;
          // Anything said before the reply's own words (an opener the model never knew of) is not part of it.
          const heard = heardOfReply(((msg.text as string) ?? '').trim(), at.map((i) => history[i].text).join(' '));
          const cut = cutToHeard(history, at, heard);
          history.splice(0, history.length, ...cut);
          ctx.setState('turns', history.length);
          ctx.log('info', heard ? 'reply truncated to what was heard' : 'reply dropped: nothing was heard', { turnId: msg.turnId as string, heard });
          return;
        }
        if (msg.type !== 'turn.user' || !adapter) return;
        userWords = false;
        if (!reqParams.model) {
          ctx.log('error', `set a model for provider "${adapter.provider}"`);
          return;
        }
        committing = true;
        try {
          commitTurn(msg);
        } finally {
          committing = false;
        }
      },
      stop() {
        deliverTimer?.clear();
        dropDraft('stopped');
        finish('cancelled');
      },
    };
  },
};
