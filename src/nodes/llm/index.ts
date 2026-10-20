import type { NodeDef } from '../../core/node';
import type { Json, JsonObject, Msg } from '../../core/types';
import { openaiCompatAdapter as adapter } from './openai';
import type { ChatTurn, LlmStreamParser } from './types';

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
 * Cut an assistant reply (possibly several history turns) down to the `heard`
 * words of its text; a turn left with nothing is dropped. Returns the history.
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
    out.set(i, text ? { role: 'assistant', text, turnId: t.turnId, rid: t.rid } : null);
  });
  return history.flatMap((t, i) => (out.has(i) ? (out.get(i) ? [out.get(i)!] : []) : [t]));
}

/** Enough for the one or two sentences an interjection is. */
const CHECK_MAX_TOKENS = 300;

const DEFAULT_SYSTEM =
  'You are a voice assistant in a live spoken conversation. Your words are converted to speech, so answer in short, natural spoken sentences. No markdown, lists, or emoji.';

/**
 * Streams a reply for each user turn and keeps the conversation, over an
 * OpenAI-compatible endpoint (llama.cpp on the GPU box). The request (full
 * history included) is the io.open payload and every raw stream payload is an io.recv, so a replay sees
 * the identical token stream with identical timing.
 *
 * A `turn.draft` (the user's words before their turn is committed) starts a
 * request early, without touching the history or emitting anything but
 * `llm.draft` / `llm.draft_end`: the stream is held back. When the committed
 * `turn.user` names the draft, the held stream is emitted at once as an
 * ordinary reply (llm.start with `draft`, then its deltas) and the rest
 * streams as usual. Any other turn, a new draft or a drop discards it.
 *
 * The first sign of reasoning in a reply (reasoning tokens) is emitted as `llm.thinking`, for the opener stage to fill the wait with.
 *
 * An `interject.check` (the user's words so far, while they still talk) asks
 * whether to cut in: a request like a draft's, with the check's prompt after
 * the words, its stream held back, with a short token limit so it answers fast. As soon as the reply is clearly not the
 * pass reply, `llm.check_verdict` says `interject` (else `pass`, and the check
 * ends). A committed turn that names the check (`interject`) keeps it the way
 * a kept draft is kept; the history records the committed words, marked as
 * cut off, without the prompt.
 *
 * A `bc.ask` (from the backchannel stage) has the model write a backchannel now: the history as a
 * reply would send it, then the user's words so far with the ask's prompt as the last user turn,
 * and its prefill as the start of the answer, on its own llama.cpp slot (`bcSlot`; replies run on
 * `slot`), so the reply's cached prefix stays. Neither the ask nor the answer goes into the history.
 * The answer comes back whole as `llm.bc`; `bc.drop` cancels it.
 *
 * Paced (the scheduler's `sched.config` says so): a reply, and a draft, first asks for `firstTokens`
 * tokens. A request that stops at its token limit leaves the reply open (`llm.chunk`), and each
 * `sched.grant` continues it: the same messages plus the text so far as the assistant's turn, which
 * llama.cpp continues; with its prompt cache that prefills only a token or two.
 */
export const llmNode: NodeDef = {
  type: 'llm',
  title: 'LLM',
  category: 'llm',
  description: 'Streaming chat model on an OpenAI-compatible endpoint (llama.cpp on the GPU box, or any local server), set in the proxy.',
  inputs: [
    { name: 'turn', type: 'turn' },
    { name: 'control', type: 'control' },
    { name: 'interject', type: 'interject', description: 'interject.check / interject.drop: whether to cut in on the words so far' },
    { name: 'sched', type: 'control', description: 'The scheduler: sched.config, sched.grant (write this many more tokens)' },
    { name: 'bc', type: 'backchannel', description: 'bc.ask / bc.drop: write a backchannel from the conversation so far' },
  ],
  outputs: [{ name: 'text', type: 'text' }],
  params: [
    { key: 'model', label: 'Model', type: 'string', default: 'gemma-4-12b-it', description: 'The model name the endpoint serves (the GPU box: gemma-4-12b-it).' },
    { key: 'maxTokens', label: 'Max tokens', type: 'number', default: 4096 },
    { key: 'extraBody', label: 'Extra request body (JSON)', type: 'text', default: '', description: 'Merged into the request, e.g. {"temperature":0.7}' },
    { key: 'system', label: 'System prompt', type: 'text', default: DEFAULT_SYSTEM },
    {
      key: 'slot',
      label: 'Reply slot',
      type: 'number',
      default: 0,
      description: 'llama.cpp slot (id_slot) for replies and drafts, so backchannels never evict their prompt cache. -1: any.',
    },
    { key: 'bcSlot', label: 'Backchannel slot', type: 'number', default: 1, description: 'llama.cpp slot for backchannels. -1: any.' },
  ],
  channels: [{ name: 'openai', description: 'Chat Completions SSE (parsed data lines)' }],
  requires: () => ['llm'],
  create(ctx) {
    const p = ctx.params as JsonObject;
    /** Replies and drafts run on their slot; interjection checks on any. */
    const reqParams: JsonObject = { ...p, idSlot: Number(p.slot ?? -1) };
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
      /** Paced: its requests' token limit, tokens so far, chunks so far, and whether it waits for a grant. */
      cap?: number;
      tokens: number;
      chunks: number;
      waiting?: boolean;
      /** The text so far ends in whitespace the continuation's prompt left out: drop it from the next delta. */
      trimNext?: boolean;
      /** An interjection check: its id, the pass reply, and the verdict once known. */
      checkId?: string;
      pass?: string;
      verdict?: 'interject' | 'pass';
    }
    let active: Reply | undefined;
    /** Set by the scheduler: write replies in chunks it grants. */
    let pace: { firstTokens: number } | undefined;
    let draft: Reply | undefined;
    let check: Reply | undefined;

    /** End the active reply. */
    const finish = (status: 'done' | 'cancelled' | 'error', error?: Json) => {
      if (!active) return;
      const a = active;
      active = undefined;
      if (a.io && !a.io.closed) a.io.close({ status });
      // Keep whatever the model produced, so the next request continues from it.
      if (a.text) history.push({ role: 'assistant', text: a.text, turnId: a.turnId, rid: a.rid });
      ctx.emit('text', { type: 'llm.end', rid: a.rid, status, stopReason: a.stopReason, text: a.text, ...(error ? { error } : {}) });
      ctx.setState('status', status === 'error' ? 'error' : 'idle');
      ctx.setState('turns', history.length);
      if (error) ctx.log('error', 'LLM request failed', error);
    };

    const handle = (events: Events) => {
      for (const ev of events) {
        if (!active) return;
        if (ev.type === 'delta') {
          let text = ev.text;
          if (active.trimNext) {
            text = text.replace(/^\s+/, '');
            if (!text) continue;
            active.trimNext = false;
          }
          active.text += text;
          active.tokens++;
          ctx.emit('text', { type: 'llm.delta', rid: active.rid, text });
        } else if (ev.type === 'end') {
          active.stopReason = ev.stopReason;
          if (active.cap !== undefined) {
            const more = ev.stopReason === 'length' && active.tokens < (Number(reqParams.maxTokens) || Infinity);
            ctx.emit('text', { type: 'llm.chunk', rid: active.rid, n: active.chunks++, tokens: active.tokens, stop: more ? 'length' : (ev.stopReason ?? 'stop') });
            if (more) {
              active.waiting = true;
              ctx.setState('status', 'paced');
              continue;
            }
          }
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

    /** Send one request for `r` and route its stream to it. */
    const send = (r: Reply, turns: ChatTurn[], params: JsonObject = reqParams, cap?: number): boolean => {
      if (cap !== undefined) {
        r.cap = cap;
        params = { ...params, maxTokens: Math.min(cap, Number(params.maxTokens) || cap) };
      }
      let request: Json;
      try {
        request = adapter.buildRequest(turns, params);
      } catch (e) {
        ctx.log('error', `could not build request: ${String(e)}`);
        if (r === draft) draft = undefined;
        else if (r === check) endCheck('error', String(e));
        else if (r === active) finish('error', String(e));
        return false;
      }
      const parser = adapter.parser();
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
      else if (r === check) {
        r.held.push(...events);
        judgeCheck(r);
      }
    };
    /** Pass replies, compared without case, spaces or wrapping quotes and emphasis. */
    const squash = (s: string) => s.toLowerCase().replace(/\s+/g, '').replace(/^["'`*_“‘]+/, '');
    /** Decide a check as soon as its reply is clearly the pass reply or not; a pass ends it. */
    const judgeCheck = (r: Reply) => {
      if (r.verdict) return;
      const text = r.held.flatMap((e) => (e.type === 'delta' ? [e.text] : [])).join('');
      const said = squash(text);
      const pass = squash(r.pass ?? '');
      const ended = r.held.some((e) => e.type === 'end' || e.type === 'error');
      const passing = !said || (pass && (pass.startsWith(said) || said.startsWith(pass)));
      if (!ended && passing) return;
      const error = r.held.find((e) => e.type === 'error');
      if (error && passing) {
        ctx.log('warn', 'interjection check failed', error.error);
        return endCheck('error', 'request failed');
      }
      r.verdict = ended && passing ? 'pass' : 'interject';
      ctx.emit('text', { type: 'llm.check_verdict', rid: r.rid, checkId: r.checkId!, verdict: r.verdict, text: text.trim(), ms: Math.round(ctx.now() - r.t0) });
      if (r.verdict === 'pass') endCheck('pass');
    };
    const endCheck = (status: string, why?: string) => {
      if (!check) return;
      const c = check;
      check = undefined;
      if (c.io && !c.io.closed) c.io.close({ status: status === 'pass' ? 'done' : 'cancelled' });
      ctx.emit('text', { type: 'llm.check_end', rid: c.rid, checkId: c.checkId!, status, ...(why ? { why } : {}) });
    };
    const startCheck = (msg: Msg) => {
      endCheck('dropped', 'checked again');
      const text = String(msg.text ?? '');
      const user: ChatTurn = { role: 'user', text: `${text}\n\n${String(msg.prompt ?? '')}`.trim() };
      const r = reply(user, {
        checkId: msg.checkId as string,
        pass: String(msg.pass ?? ''),
        ...(msg.replaces ? { replaces: msg.replaces as string } : {}),
      });
      ctx.emit('text', { type: 'llm.check', rid: r.rid, checkId: r.checkId!, text });
      check = r;
      // A check is a reflex: a short answer.
      send(r, historyFor(user, msg.replaces), { ...reqParams, idSlot: -1, maxTokens: Math.min(Number(reqParams.maxTokens) || CHECK_MAX_TOKENS, CHECK_MAX_TOKENS) });
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
    const reply = (user: ChatTurn, extra: Partial<Reply>): Reply => ({
      rid: `${ctx.id}-r${n++}`,
      user,
      t0: ctx.now(),
      live: false,
      held: [],
      text: '',
      stopReason: null,
      tokens: 0,
      chunks: 0,
      ...extra,
    });
    const startDraft = (msg: Msg) => {
      dropDraft('redrafted');
      const user: ChatTurn = { role: 'user', text: msg.text as string };
      const d = reply(user, { draftId: msg.draftId as string, ...(msg.replaces ? { replaces: msg.replaces as string } : {}) });
      ctx.emit('text', { type: 'llm.draft', rid: d.rid, draftId: d.draftId!, text: user.text });
      draft = d;
      send(d, historyFor(user, msg.replaces), reqParams, pace?.firstTokens);
    };

    /** A committed user turn: answer it (with the draft made for it, if it still stands). */
    const commitTurn = (msg: Msg) => {
      finish('cancelled');
      // An interjection the turn was committed for: its held reply is the answer.
      const cut = msg.interject && check?.checkId === msg.interject && check.verdict === 'interject' ? check : undefined;
      if (!cut) endCheck('dropped', 'turn committed without it');
      const kept = cut ?? (msg.draft && draft?.draftId === msg.draft ? draft : undefined);
      if (!kept || cut) dropDraft('turn committed without it');
      if (msg.replaces) {
        // The user wasn't done: this turn replaces an earlier one and whatever followed it.
        const i = history.findIndex((t) => t.role === 'user' && t.turnId === msg.replaces);
        if (i >= 0) history.splice(i);
      }
      const turnId = msg.turnId as string;
      if (kept) {
        // The draft's words stand: keep its reply. The history records what the request said;
        // for an interjection, all the words committed (some may have come after the check), cut off.
        if (cut) check = undefined;
        else draft = undefined;
        history.push(cut ? { role: 'user', text: `${String(msg.text)}—`, turnId } : { ...kept.user, turnId });
        const held = kept.held;
        kept.held = [];
        const chunks = held.filter((e) => e.type === 'delta').length;
        if (cut) ctx.emit('text', { type: 'llm.check_end', rid: kept.rid, checkId: kept.checkId!, status: 'kept', chunks });
        else ctx.emit('text', { type: 'llm.draft_end', rid: kept.rid, draftId: kept.draftId!, status: 'kept', chunks });
        ctx.emit('text', {
          type: 'llm.start',
          rid: kept.rid,
          turnId: turnId ?? null,
          ...(cut ? { interject: kept.checkId! } : { draft: kept.draftId! }),
          headStartMs: Math.round(ctx.now() - kept.t0),
        });
        ctx.setState('status', 'generating');
        kept.turnId = turnId;
        active = kept;
        kept.live = true;
        handle(held);
        return;
      }
      // Committed early for an interjection that couldn't be kept: the words are still cut off.
      const user: ChatTurn = { role: 'user', text: msg.interject ? `${String(msg.text)}—` : (msg.text as string), turnId };
      history.push(user);
      const r = reply(user, { turnId });
      ctx.emit('text', { type: 'llm.start', rid: r.rid, turnId: turnId ?? null });
      ctx.setState('status', 'generating');
      active = r;
      r.live = true;
      send(r, [...history], reqParams, pace?.firstTokens);
    };

    /** The scheduler lets a paced reply write `tokens` more: continue its text. */
    const grant = (tokens: number) => {
      const r = active;
      if (!r?.waiting) return;
      r.waiting = false;
      ctx.setState('status', 'generating');
      const prefix = r.text.trimEnd();
      r.trimNext = prefix !== r.text;
      send(r, [...history, { role: 'assistant', text: prefix }], reqParams, tokens);
    };

    /** The backchannel being written, if any. */
    let bc: { id: string; io: Io; t0: number; text: string; first?: number; done?: boolean } | undefined;
    const dropBc = (why: string) => {
      const b = bc;
      if (!b) return;
      bc = undefined;
      if (!b.io.closed) b.io.close({ status: 'cancelled' });
      ctx.log('debug', 'backchannel request dropped', { id: b.id, why });
    };
    const askBc = (msg: Msg) => {
      dropBc('asked again');
      const user: ChatTurn = { role: 'user', text: `${String(msg.text ?? '')}\n\n${String(msg.prompt ?? '')}`.trim() };
      const prefill = String(msg.prefill ?? '');
      // The history exactly as a reply would send it, so the prefix matches the reply's.
      const turns: ChatTurn[] = [...history, user, ...(prefill ? [{ role: 'assistant' as const, text: prefill }] : [])];
      let request: Json;
      try {
        const built = adapter.buildRequest(turns, { ...reqParams, idSlot: Number(p.bcSlot ?? -1) }) as JsonObject;
        request = { ...built, body: { ...(built.body as JsonObject), ...((msg.body as JsonObject) ?? {}) } };
      } catch (e) {
        ctx.emit('text', { type: 'llm.bc', id: msg.id as string, status: 'error', text: '', error: String(e) });
        return;
      }
      const parser = adapter.parser();
      const b: NonNullable<typeof bc> = { id: msg.id as string, t0: ctx.now(), text: '', io: undefined as unknown as Io };
      bc = b;
      const route = (events: Events) => {
        for (const ev of events) {
          if (bc !== b || b.done) return;
          if (ev.type === 'delta') {
            b.first ??= ctx.now();
            b.text += ev.text;
          } else if (ev.type === 'end' || ev.type === 'error') {
            b.done = true;
            bc = undefined;
            if (!b.io.closed) b.io.close({ status: 'done' });
            const status = ev.type === 'end' ? 'done' : 'error';
            ctx.emit('text', {
              type: 'llm.bc',
              id: b.id,
              status,
              text: b.text,
              ms: Math.round(ctx.now() - b.t0),
              firstMs: b.first === undefined ? null : Math.round(b.first - b.t0),
              ...(ev.type === 'error' ? { error: ev.error } : { stopReason: ev.stopReason }),
            });
          }
        }
      };
      b.io = ctx.io.open(adapter.channel, request, {
        onRecv: (data) => route(parser.recv(data)),
        onClose: (data) => route(parser.close(data)),
      });
    };

    return {
      onInput(port, msg) {
        if (port === 'bc') {
          if (msg.type === 'bc.ask') askBc(msg);
          else if (msg.type === 'bc.drop' && bc?.id === msg.id) dropBc(String(msg.why ?? 'dropped'));
          return;
        }
        if (port === 'control') {
          if (msg.type === 'cancel') {
            dropDraft('cancelled');
            endCheck('dropped', 'cancelled');
            finish('cancelled');
          }
          return;
        }
        if (port === 'sched') {
          if (msg.type === 'sched.config') pace = msg.pace ? { firstTokens: Number(msg.firstTokens) || 16 } : undefined;
          else if (msg.type === 'sched.grant' && msg.rid === active?.rid) grant(Number(msg.tokens));
          return;
        }
        if (port === 'interject') {
          if (msg.type === 'interject.check') {
            if (reqParams.model) startCheck(msg);
          } else if (msg.type === 'interject.drop' && check?.checkId === msg.checkId) endCheck('dropped', (msg.why as string) ?? 'dropped');
          return;
        }
        if (msg.type === 'turn.draft') {
          if (reqParams.model) startDraft(msg);
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
        if (msg.type !== 'turn.user') return;
        if (!reqParams.model) {
          ctx.log('error', 'set a model for the LLM');
          return;
        }
        commitTurn(msg);
      },
      stop() {
        dropBc('stopped');
        dropDraft('stopped');
        endCheck('dropped', 'stopped');
        finish('cancelled');
      },
    };
  },
};
