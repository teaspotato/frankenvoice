import type { IOHandle, NodeDef, TimerHandle } from '../core/node';
import type { Json, JsonObject, Msg } from '../core/types';
import { heardOfReply, llmNode } from './llm';
import { openaiCompatAdapter as adapter } from './llm/openai';
import type { ChatTurn, LlmStreamParser } from './llm/types';
import { cutDecision, DECISION_GRAMMAR, DecisionParser, DEFAULT_AGENT_SYSTEM, saidWords, stamp, taskLabel, type ActionLine, type DecisionEvent } from './agent-format';

/** Why the agent thinks: the user's words (high priority), the session's start, System 2's news, or its own timers. */
type Wake = 'user' | 'start' | 'news' | 'timer';
type EventKind = 'start' | 'user' | 'cut' | 'news' | 'progress' | 'silence' | 'wake' | 'note';

interface AgentParams {
  model: string;
  system: string;
  maxTokens: number;
  extraBody: string;
  deadAirMs: number;
  progress: string;
  progressMs: number;
  maxResultChars: number;
  maxWorkers: number;
  maxSteps: number;
  slot: number;
  bcSlot: number;
}

/** One think step: a request whose decision streams in. */
interface Call {
  n: number;
  wake: Wake;
  io: IOHandle;
  parser: LlmStreamParser;
  dp: DecisionParser;
  t0: number;
  firstAt?: number;
  /** The reply its say lines make, once it says something. */
  rid?: string;
  turnId: string | null;
  /** Its complete lines, as written. */
  kept: string[];
  /** The say line being written: its tag and the words so far. */
  saying: string;
  utterances: number;
  stopReason: string | null;
  actions: JsonObject[];
  wait?: number;
  quiet?: boolean;
}

const words = (s: string) => s.split(/\s+/).filter(Boolean);
const clip = (s: string, n: number) => {
  const t = s.replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
};
const RANK: Record<Wake, number> = { timer: 0, news: 1, start: 2, user: 3 };
const DEFAULT_MODEL = llmNode.params.find((x) => x.key === 'model')!.default;

/**
 * The voice agent on its own clock (System 1). It keeps one append-only log of what happens (the
 * user's turns, their cutting in, System 2's progress, results and questions, silences, its own
 * wake-ups: agent-format.ts) and thinks whenever something worth considering arrives. A think step
 * is one streamed request under a grammar; its say lines go to TTS as they are written (llm.start,
 * llm.utterance, llm.delta, llm.end, like the reply LLM), and it can hand tasks to System 2 (the
 * workers stage), wait, stay silent or go quiet. Nothing forces it to speak, and it can speak with
 * nobody asking: at the session's start, when news comes in, after a silence.
 *
 * The user's words are the high-priority event: they cancel a think step in flight (and its speech,
 * if none of it has played yet) and the next step sees them. Other news waits for the step in
 * flight, and for the user to stop talking. While quiet, only the user's words wake it. After
 * `deadAirMs` of silence on both sides it gets a silence event (the next one after twice as long, up
 * to 8×, until someone speaks). At most `maxSteps` steps in a row run without the user.
 *
 * Every event is emitted as agent.event, every step as agent.think and agent.decide, so the
 * dashboard shows the log; agent.pass tells turn-taking a step said nothing (the floor is free).
 */
export const agentLoopNode: NodeDef<AgentParams> = {
  type: 'agent.loop',
  title: 'Voice agent (System 1)',
  category: 'llm',
  description:
    'The voice agent on its own clock: an event log the LLM reads and acts on whenever something happens, speaking, waiting, staying quiet or handing work to System 2. The user’s words are a high-priority event, not the only trigger.',
  inputs: [
    { name: 'turn', type: 'turn', description: 'turn.user (the user finished), turn.heard (what they heard of a cut-off reply), turn.words' },
    { name: 'control', type: 'control', description: 'cancel: the user cut in' },
    { name: 'vad', type: 'vad', description: 'Whether the user is talking, for silences' },
    { name: 'playback', type: 'playback', description: 'Whether the agent is audible, for silences' },
    { name: 'transcript', type: 'transcript', description: 'asr.open: the session starts' },
    { name: 'updates', type: 'workers', description: 'System 2: progress, results, questions' },
    { name: 'bc', type: 'backchannel', description: 'bc.ask / bc.drop: write a backchannel from the log so far' },
  ],
  outputs: [
    { name: 'text', type: 'text', description: 'What it says (llm.start, llm.utterance, llm.delta, llm.end) and backchannels (llm.bc)' },
    { name: 'workers', type: 'workers', description: 'worker.start / worker.send / worker.stop, to System 2' },
    { name: 'agent', type: 'agent', description: 'agent.event, agent.think, agent.decide, agent.pass' },
    { name: 'control', type: 'control', description: 'cancel: its own speech, when the user’s words overtake a step that has not been heard yet' },
  ],
  params: [
    { key: 'model', label: 'Model', type: 'string', default: DEFAULT_MODEL, description: 'The model name the endpoint serves.' },
    { key: 'system', label: 'Instructions', type: 'text', default: DEFAULT_AGENT_SYSTEM },
    { key: 'maxTokens', label: 'Max tokens per step', type: 'number', default: 200 },
    { key: 'extraBody', label: 'Extra request body (JSON)', type: 'text', default: '', description: 'Merged into each request, e.g. {"temperature":0.7}' },
    { key: 'deadAirMs', label: 'Silence event after (ms)', type: 'number', default: 6000, description: 'Nobody has spoken this long: a silence event wakes the agent (not while quiet). Each one in a row waits twice as long, up to 8×. 0: off.' },
    {
      key: 'progress',
      label: 'System 2 progress',
      type: 'select',
      options: ['wake', 'log', 'off'],
      default: 'wake',
      description: 'A worker’s steps, at most one line per worker every so often. wake: on the log, and the agent thinks about it. log: on the log, read at the next step. off: only results and questions.',
    },
    { key: 'progressMs', label: 'Progress at most every (ms)', type: 'number', default: 10000, description: 'Per worker.' },
    { key: 'maxResultChars', label: 'Longest result on the log (chars)', type: 'number', default: 600 },
    { key: 'maxWorkers', label: 'Most workers at once', type: 'number', default: 4 },
    { key: 'maxSteps', label: 'Most steps without the user', type: 'number', default: 12, description: 'Steps in a row woken by news, silences or its own wake-ups; more wait for the user to speak.' },
    { key: 'slot', label: 'llama.cpp slot', type: 'number', default: 0, description: 'id_slot for think steps (their prompt cache). -1: any.' },
    { key: 'bcSlot', label: 'Backchannel slot', type: 'number', default: 1, description: 'llama.cpp slot for generated backchannels. -1: any.' },
  ],
  channels: [{ name: 'openai', description: 'Chat Completions SSE (parsed data lines): one per think step or backchannel' }],
  requires: () => ['llm'],
  create(ctx) {
    const p = ctx.params;
    const reqParams: JsonObject = { model: p.model, system: p.system, maxTokens: p.maxTokens, extraBody: p.extraBody, idSlot: Number(p.slot) };
    const history: ChatTurn[] = [];
    /** Event lines that came while a step was in flight: they go after its decision. */
    let queued: string[] = [];
    let t0 = 0;
    let started = false;
    let call: Call | null = null;
    let callNo = 0;
    let ridNo = 0;
    let pending: Wake | null = null;
    let thinkTimer: TimerHandle | null = null;
    let quiet = false;
    /** Steps in a row not woken by the user. */
    let autoSteps = 0;
    let lastTurnId: string | null = null;
    const turnText = new Map<string, string>();
    /** The user's words not committed yet (turn.words). */
    let lastWords = '';
    let userSpeaking = false;
    const playing = new Set<string>();
    /** Replies whose audio started playing. */
    const heardRids = new Set<string>();
    let wakeTimer: TimerHandle | null = null;
    let deadAir: TimerHandle | null = null;
    let silences = 0;
    const workers = new Map<string, { task: string; status: string }>();
    let workerNo = 0;
    const progress = new Map<string, { last: number; text?: string; timer?: TimerHandle }>();

    const setStatus = () => {
      ctx.setState('status', call ? 'thinking' : 'idle');
      ctx.setState('quiet', quiet);
      ctx.setState('workers', [...workers.values()].filter((w) => w.status === 'running').length);
    };
    setStatus();

    const agentMsg = (type: string, data: JsonObject = {}) => ctx.emit('agent', { type, ...data });

    /** Put an event on the log; `wake` asks for a step. */
    const append = (kind: EventKind, line: string, wake?: Wake) => {
      const text = `${stamp(ctx.now() - t0)} ${line}`;
      if (call) queued.push(text);
      else history.push({ role: 'user', text });
      agentMsg('agent.event', { kind, line: text, wake: wake ?? null });
      if (wake) want(wake);
    };

    const want = (wake: Wake) => {
      if (wake === 'user') {
        autoSteps = 0;
        if (call) overtake(call, 'the user spoke');
      } else if (quiet && wake !== 'start') return void ctx.log('debug', 'quiet: not woken', { wake });
      else if (autoSteps >= Math.max(1, p.maxSteps)) return void ctx.log('info', `${autoSteps} steps without the user: waiting for them`, { wake });
      if (!pending || RANK[wake] > RANK[pending]) pending = wake;
      schedule();
    };

    /** Think as soon as nothing is in flight (and, unless the user woke it, the user isn't talking). */
    const schedule = () => {
      if (!pending || call || thinkTimer) return;
      if (userSpeaking && pending !== 'user') return;
      deadAir?.clear();
      deadAir = null;
      thinkTimer = ctx.setTimer(0, () => {
        thinkTimer = null;
        think();
      }, 'think');
    };

    const think = () => {
      const wake = pending;
      pending = null;
      if (!wake || call) return;
      if (history.at(-1)?.role !== 'user') return;
      if (wake === 'news' || wake === 'timer') autoSteps++;
      // Its own wake-up was for a quiet moment; something else came first.
      wakeTimer?.clear();
      wakeTimer = null;
      let request: Json;
      try {
        const built = adapter.buildRequest(history, reqParams) as JsonObject;
        request = { ...built, body: { ...(built.body as JsonObject), grammar: DECISION_GRAMMAR } };
      } catch (e) {
        ctx.log('error', `could not build the request: ${String(e)}`);
        return;
      }
      const parser = adapter.parser();
      const c: Call = {
        n: callNo++,
        wake,
        parser,
        dp: new DecisionParser(),
        t0: ctx.now(),
        turnId: wake === 'user' ? lastTurnId : null,
        kept: [],
        saying: '',
        utterances: 0,
        stopReason: null,
        actions: [],
        io: undefined as unknown as IOHandle,
      };
      call = c;
      agentMsg('agent.think', { n: c.n, wake, turns: history.length });
      c.io = ctx.io.open(adapter.channel, request, {
        onRecv: (data) => route(c, parser.recv(data)),
        onClose: (data) => route(c, parser.close(data)),
      });
      setStatus();
      armDeadAir();
    };

    const route = (c: Call, events: ReturnType<LlmStreamParser['recv']>) => {
      for (const ev of events) {
        if (call !== c) return;
        if (ev.type === 'delta') {
          c.firstAt ??= ctx.now();
          take(c, c.dp.feed(ev.text));
        } else if (ev.type === 'end') {
          c.stopReason = ev.stopReason;
          take(c, c.dp.end());
          finish(c, 'done');
        } else if (ev.type === 'error') {
          take(c, c.dp.end());
          finish(c, 'error', ev.error);
        }
      }
    };

    /** What the decision says, as it streams. */
    const take = (c: Call, des: DecisionEvent[]) => {
      for (const de of des) {
        if (de.type === 'say') {
          if (!c.rid) {
            c.rid = `${ctx.id}-r${ridNo++}`;
            ctx.emit('text', { type: 'llm.start', rid: c.rid, turnId: c.turnId });
          }
          ctx.emit('text', { type: 'llm.utterance', rid: c.rid, n: c.utterances, interruptible: de.interruptible, gate: de.gate });
          c.saying = `[i${de.interruptible ? 1 : 0}g${de.gate ? 1 : 0}] `;
          // A space between utterances, so their words stay apart.
          if (c.utterances++ > 0) ctx.emit('text', { type: 'llm.delta', rid: c.rid, text: ' ' });
          // Talking again ends a quiet spell.
          if (quiet) {
            quiet = false;
            setStatus();
          }
        } else if (de.type === 'words') {
          c.saying += de.text;
          ctx.emit('text', { type: 'llm.delta', rid: c.rid!, text: de.text });
        } else if (de.type === 'junk') {
          c.kept.push(de.raw);
          ctx.log('warn', 'a line not in the format', { line: de.raw });
        } else {
          c.kept.push(de.raw);
          c.saying = '';
          act(c, de.line);
        }
      }
    };

    const act = (c: Call, a: ActionLine) => {
      switch (a.kind) {
        case 'say':
          c.actions.push({ say: a.text });
          break;
        case 'think': {
          const id = `w${++workerNo}`;
          const running = [...workers.values()].filter((w) => w.status === 'running').length;
          c.actions.push({ think: a.text, worker: id });
          if (running >= Math.max(1, p.maxWorkers)) {
            append('note', `${id} not started: ${running} workers are already busy`);
            break;
          }
          workers.set(id, { task: a.text, status: 'running' });
          ctx.emit('workers', { type: 'worker.start', workerId: id, task: a.text });
          break;
        }
        case 'tell': {
          c.actions.push({ tell: a.worker, text: a.text });
          const w = workers.get(a.worker);
          if (!w) {
            append('note', `there is no worker ${a.worker}`);
            break;
          }
          w.status = 'running';
          ctx.emit('workers', { type: 'worker.send', workerId: a.worker, message: a.text });
          break;
        }
        case 'stop': {
          c.actions.push({ stop: a.worker });
          const w = workers.get(a.worker);
          if (w && w.status === 'running') {
            w.status = 'stopped';
            ctx.emit('workers', { type: 'worker.stop', workerId: a.worker });
          }
          break;
        }
        case 'wait':
          c.wait = Math.min(600, Math.max(1, a.s));
          c.actions.push({ wait: c.wait });
          break;
        case 'silent':
          c.actions.push({ silent: true });
          break;
        case 'quiet':
          c.quiet = true;
          c.actions.push({ quiet: true });
          break;
      }
      setStatus();
    };

    /** The decision as it stands: its complete lines and the say line being written. */
    const written = (c: Call) => [...c.kept, ...(c.saying.trim().length > 7 ? [c.saying.trim()] : [])].join('\n');

    /** End step `c`: its decision goes on the log, what came meanwhile after it. */
    const close = (c: Call, raw: string, status: string, extra: JsonObject = {}) => {
      call = null;
      if (raw) history.push({ role: 'assistant', text: raw, ...(c.rid ? { rid: c.rid } : {}) });
      for (const text of queued) history.push({ role: 'user', text });
      queued = [];
      agentMsg('agent.decide', {
        n: c.n,
        status,
        text: raw,
        ms: Math.round(ctx.now() - c.t0),
        firstMs: c.firstAt === undefined ? null : Math.round(c.firstAt - c.t0),
        ...(c.rid ? { rid: c.rid } : {}),
        actions: c.actions as Json,
        ...extra,
      });
      setStatus();
    };

    const finish = (c: Call, status: 'done' | 'error', error?: Json) => {
      if (!c.io.closed) c.io.close({ status });
      const raw = written(c);
      if (c.rid) ctx.emit('text', { type: 'llm.end', rid: c.rid, status, stopReason: c.stopReason, text: saidWords(raw), raw, ...(error ? { error } : {}) });
      if (error) ctx.log('error', 'think step failed', error);
      close(c, raw, status, error ? { error } : {});
      if (c.wait) {
        const at = stamp(ctx.now() - t0);
        const ms = c.wait * 1000;
        wakeTimer?.clear();
        wakeTimer = ctx.setTimer(ms, () => {
          wakeTimer = null;
          append('wake', `wake-up (you asked for it at ${at})`, 'timer');
        }, 'wake');
      }
      if (c.quiet) quiet = true;
      if (!c.rid && !pending) agentMsg('agent.pass', { n: c.n });
      setStatus();
      schedule();
      armDeadAir();
    };

    /**
     * The user's words came while step `c` was in flight: drop it. What it said that is already
     * playing stays (turn-taking cuts it to what was heard); what nobody has heard yet is cancelled
     * and leaves the log. Workers it started keep going.
     */
    const overtake = (c: Call, why: string, cancelled = false) => {
      if (!c.io.closed) c.io.close({ status: 'cancelled' });
      let raw = written(c);
      if (c.rid) {
        if (!cancelled) ctx.emit('control', { type: 'cancel' });
        ctx.emit('text', { type: 'llm.end', rid: c.rid, status: 'cancelled', stopReason: null, text: saidWords(raw) });
        // Not heard yet: none of it was said. Heard: turn-taking says how much, and the log keeps that.
        if (!heardRids.has(c.rid)) raw = cutDecision(raw, 0);
      }
      close(c, raw, 'overtaken', { why });
      if (c.quiet) quiet = true;
    };

    // ---- silences ------------------------------------------------------------------------

    const armDeadAir = () => {
      deadAir?.clear();
      deadAir = null;
      if (!started || quiet || p.deadAirMs <= 0 || userSpeaking || playing.size || call || thinkTimer || pending) return;
      const ms = p.deadAirMs * 2 ** Math.min(silences, 3);
      deadAir = ctx.setTimer(ms, () => {
        deadAir = null;
        silences++;
        const so = lastWords ? `; the user said so far: "${clip(lastWords, 200)}"` : '';
        append('silence', `silence for ${Math.round(ms / 1000)} s${so}`, 'timer');
      }, 'dead-air');
    };

    // ---- System 2 ------------------------------------------------------------------------

    const progressLine = (id: string) => {
      const pr = progress.get(id);
      const w = workers.get(id);
      if (!pr?.text || !w || w.status !== 'running') return;
      pr.last = ctx.now();
      const text = pr.text;
      pr.text = undefined;
      append('progress', `${id} progress: ${text}`, p.progress === 'wake' ? 'news' : undefined);
    };
    const onProgress = (id: string, text: string) => {
      if (p.progress === 'off' || !text) return;
      let pr = progress.get(id);
      if (!pr) progress.set(id, (pr = { last: -Infinity }));
      pr.text = clip(text, 160);
      if (pr.timer) return;
      const left = pr.last + p.progressMs - ctx.now();
      if (left <= 0) return progressLine(id);
      const q = pr;
      q.timer = ctx.setTimer(left, () => {
        q.timer = undefined;
        progressLine(id);
      }, 'progress');
    };
    const ended = (id: string, status: string) => {
      const w = workers.get(id);
      if (w) w.status = status;
      progress.get(id)?.timer?.clear();
      progress.delete(id);
      setStatus();
    };

    const update = (msg: Msg) => {
      const id = String(msg.workerId ?? '');
      const label = workers.has(id) ? ` (${taskLabel(workers.get(id)!.task)})` : '';
      switch (msg.type) {
        case 'worker.tool':
          return onProgress(id, String(msg.title ?? ''));
        case 'worker.progress': {
          const plan = String(msg.text ?? '').split('\n');
          const now = plan.find((l) => l.startsWith('[>]')) ?? plan.at(-1) ?? '';
          return onProgress(id, now.replace(/^\[.\] /, 'working on: '));
        }
        case 'worker.text':
          return onProgress(id, String(msg.text ?? ''));
        case 'worker.done':
          ended(id, 'done');
          return append('news', `${id} result${label}: ${clip(String(msg.text ?? ''), p.maxResultChars)}`, 'news');
        case 'worker.question':
          ended(id, 'waiting');
          return append('news', `${id} asks the user${label}: ${clip(String(msg.text ?? ''), p.maxResultChars)}`, 'news');
        case 'worker.error':
          ended(id, 'error');
          return append('news', `${id} failed${label}: ${clip(String(msg.text ?? ''), 200)}`, 'news');
        case 'worker.stopped':
          return ended(id, 'stopped');
      }
    };

    // ---- generated backchannels ----------------------------------------------------------

    /** The backchannel being written, if any: the log so far, the ask's instruction and the user's words. */
    let bc: { id: string; io: IOHandle; t0: number; text: string; first?: number; done?: boolean } | undefined;
    const dropBc = (why: string) => {
      const b = bc;
      if (!b) return;
      bc = undefined;
      if (!b.io.closed) b.io.close({ status: 'cancelled' });
      ctx.log('debug', 'backchannel request dropped', { id: b.id, why });
    };
    const askBc = (msg: Msg) => {
      dropBc('asked again');
      const prefill = String(msg.prefill ?? '');
      const turns: ChatTurn[] = [
        ...history,
        ...queued.map((text) => ({ role: 'user' as const, text })),
        { role: 'user', text: `${String(msg.prompt ?? '')}\n\n${String(msg.text ?? '')}`.trim() },
        ...(prefill ? [{ role: 'assistant' as const, text: prefill }] : []),
      ];
      let request: Json;
      try {
        const built = adapter.buildRequest(turns, { ...reqParams, idSlot: Number(p.bcSlot) }) as JsonObject;
        request = { ...built, body: { ...(built.body as JsonObject), ...((msg.body as JsonObject) ?? {}) } };
      } catch (e) {
        ctx.emit('text', { type: 'llm.bc', id: msg.id as string, status: 'error', text: '', error: String(e) });
        return;
      }
      const parser = adapter.parser();
      const b: NonNullable<typeof bc> = { id: msg.id as string, t0: ctx.now(), text: '', io: undefined as unknown as IOHandle };
      bc = b;
      const route = (events: ReturnType<LlmStreamParser['recv']>) => {
        for (const ev of events) {
          if (bc !== b || b.done) return;
          if (ev.type === 'delta') {
            b.first ??= ctx.now();
            b.text += ev.text;
          } else if (ev.type === 'end' || ev.type === 'error') {
            b.done = true;
            bc = undefined;
            if (!b.io.closed) b.io.close({ status: 'done' });
            ctx.emit('text', {
              type: 'llm.bc',
              id: b.id,
              status: ev.type === 'end' ? 'done' : 'error',
              text: b.text,
              ms: Math.round(ctx.now() - b.t0),
              firstMs: b.first === undefined ? null : Math.round(b.first - b.t0),
              ...(ev.type === 'error' ? { error: ev.error } : { stopReason: ev.stopReason }),
            });
          }
        }
      };
      b.io = ctx.io.open(adapter.channel, request, { onRecv: (d) => route(parser.recv(d)), onClose: (d) => route(parser.close(d)) });
    };

    // ---- inputs ----------------------------------------------------------------------------

    /** The user cut the reply `rid` off: its decision keeps what they heard. */
    const heard = (rid: string, text: string) => {
      const i = history.findLastIndex((t) => t.role === 'assistant' && t.rid === rid);
      if (i < 0) return;
      const said = heardOfReply(text.trim(), saidWords(history[i].text));
      const n = words(said).length;
      const cut = cutDecision(history[i].text.replace(/—$/, ''), n);
      // Only what they heard of what it said: nothing else of the log changes.
      history[i] = { ...history[i], text: cut };
      const tail = words(said).slice(-6).join(' ');
      append('cut', n ? `user cut in; they heard you up to "${tail}"` : 'user cut in before they heard anything of it');
    };

    return {
      start() {
        t0 = ctx.now();
      },
      onInput(port, msg) {
        switch (port) {
          case 'transcript':
            if (msg.type === 'asr.open' && !started) {
              started = true;
              append('start', 'session start: the user is on the line and can hear you', 'start');
            }
            return;
          case 'vad':
            if (msg.type === 'vad.speech_start') {
              userSpeaking = true;
              silences = 0;
            } else if (msg.type === 'vad.speech_end') {
              userSpeaking = false;
              schedule();
            }
            return armDeadAir();
          case 'playback':
            if (msg.type === 'playback.start' && msg.rid) {
              playing.add(msg.rid as string);
              heardRids.add(msg.rid as string);
              silences = 0;
            } else if ((msg.type === 'playback.end' || msg.type === 'playback.interrupted') && msg.rid) playing.delete(msg.rid as string);
            else return;
            return armDeadAir();
          case 'updates':
            return update(msg);
          case 'bc':
            if (msg.type === 'bc.ask') askBc(msg);
            else if (msg.type === 'bc.drop' && bc?.id === msg.id) dropBc(String(msg.why ?? 'dropped'));
            return;
          case 'control':
            // The user cut in: the step in flight stops, and turn-taking already stopped the audio.
            if (msg.type === 'cancel' && call) overtake(call, 'the user cut in', true);
            return armDeadAir();
          case 'turn':
            if (msg.type === 'turn.words') lastWords = String(msg.text ?? '');
            else if (msg.type === 'turn.heard' && msg.rid) heard(msg.rid as string, String(msg.text ?? ''));
            else if (msg.type === 'turn.user') {
              const text = String(msg.text ?? '').trim();
              const turnId = (msg.turnId as string) ?? null;
              if (turnId) turnText.set(turnId, text);
              lastTurnId = turnId;
              lastWords = '';
              const before = msg.replaces ? turnText.get(msg.replaces as string) : undefined;
              // They went on after a turn the log already has: only the new words.
              if (before !== undefined && text.startsWith(before) && text.length > before.length)
                append('user', `user (went on): ${text.slice(before.length).trim()}`, 'user');
              else append('user', `user: ${text}`, 'user');
            }
            return;
        }
      },
      stop() {
        dropBc('stopped');
        if (call && !call.io.closed) call.io.close({ status: 'cancelled' });
        call = null;
      },
    };
  },
};
