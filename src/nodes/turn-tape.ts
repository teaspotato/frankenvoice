import type { IOHandle, NodeDef, TimerHandle } from '../core/node';
import type { Json, JsonObject, Msg } from '../core/types';
import {
  ACTIONS,
  AGENT_BACKCHANNELS,
  actionOdds,
  EXAMPLE_TAPE,
  PROMPTS,
  WORD_S,
  cutTape,
  grammar,
  headroom,
  line,
  newCell,
  parseAction,
  promptHead,
  timed,
  tapePrompt,
  transcript,
  trimIdle,
  type Cell,
  type SentWord,
  type TapeState,
} from './tape';

interface TapeParams {
  cellMs: number;
  timeScale: number;
  prompt: string;
  system: string;
  example: string;
  maxWords: number;
  ttsLeadMs: number;
  cutAfterMs: number;
  idleCells: number;
  cutOverCells: number;
  specCells: number;
  maxLines: number;
  keepLines: number;
  stableWords: boolean;
  slot: number;
  temperature: number;
  nProbs: number;
  respondBias: number;
  idleKeep: number;
  answerAfterS: number;
  greeting: string;
}

const CHANNEL = 'llama-completion';
const words = (s: string) => s.split(/\s+/).filter(Boolean);
/** A user word without its periods ("if." -> "if", "U.S." -> "US"); a decimal point between digits stays. */
export const noPeriods = (w: string) => w.replace(/\.(?!\d)|(?<!\d)\./g, '');

/** The answer being spoken (or whose audio still plays). */
interface Reply {
  rid: string;
  sent: SentWord[];
  /** llm.end went out (finish, or cut). */
  ended: boolean;
  /** Its audio is playing (playback.start seen, no end yet). */
  playing: boolean;
  /** Cut off: waiting for the speaker to say how much was heard. */
  cutting?: boolean;
}

interface Call {
  cell: Cell;
  state: TapeState;
  allowed: string[];
  maxWords: number;
  t0: number;
  io: IOHandle;
  res?: JsonObject;
  /** A cell written ahead, for this speculation. */
  spec: unknown;
  /** The backstop: only respond was allowed. */
  answer?: boolean;
}

/**
 * Turn-taking by a text LLM writing a tape (tape.ts), zero-shot: once per cell, one llama.cpp
 * `/completion` call continues the tape by the agent's action, under a grammar that allows only what
 * its state permits. The node speaks for the agent itself, as the LLM stage would: respond / resume
 * open a reply (llm.start), continue / finish stream its words (llm.delta, llm.end) to TTS, and a
 * backchannel is a canned cue (bc.say). The LLM stage gets no user turns and stays idle.
 *
 * User voice over the agent's answer (`cutAfterMs` of it) cuts it off as the full-duplex stage does
 * (control cancel); the speaker's playback.interrupted says what was heard, and the words written but
 * never heard leave the tape. Calls are skipped when the grammar would leave no choice (speaking with
 * a full TTS queue), while a call is still out, and in long silences (`idleCells`).
 *
 * Zero-shot, Gemma waits too long at turn ends and, once the tape fills with idle listen lines, copies
 * them and stops answering (research: pilot/prompt-v2.md). So silent cells add `respondBias` to the
 * logit of " respond", the prompt cuts each finished idle run to `idleKeep` lines, and a backstop
 * allows only respond once the user has been silent `answerAfterS` after words left unanswered.
 *
 * Every cell is emitted as `tape.cell` (the line as written, the call's latency, timings, its xid and
 * Gemma's odds for each action), every cut as `tape.cut`, so the dashboard and the Tape inspector show
 * the tape as it is written.
 */
export const tapeTurnNode: NodeDef<TapeParams> = {
  type: 'turn.tape',
  title: 'Tape (LLM decides each cell)',
  category: 'control',
  description:
    'Zero-shot full-duplex text LLM: every cell (640 ms) Gemma continues a tape of what the user said and what the agent did, and writes the agent’s next action and words under a grammar. The LLM stage stays idle; set Backchannels off.',
  inputs: [
    { name: 'transcript', type: 'transcript' },
    { name: 'vad', type: 'vad' },
    { name: 'playback', type: 'playback' },
  ],
  outputs: [
    { name: 'turn', type: 'turn', description: 'tape.cell per cell, tape.cut, turn.greet, turn.heard' },
    { name: 'control', type: 'control', description: 'cancel: the user cut the agent off' },
    { name: 'text', type: 'text', description: 'The agent’s words (llm.start / llm.delta / llm.end) and backchannel cues (bc.lines, bc.say), to TTS' },
  ],
  params: [
    { key: 'cellMs', label: 'Cell (ms)', type: 'number', default: 640, description: 'One decision per cell, in real time.' },
    {
      key: 'timeScale',
      label: 'Tape time scale',
      type: 'number',
      default: 1,
      description: 'Stretches the time written on the tape: each cell reads as cell × this seconds, in the timestamps, the example and the prompt’s cell length. Above 1, a model that waits a fixed number of seconds waits fewer cells.',
    },
    { key: 'prompt', label: 'Prompt', type: 'select', options: ['base', 'eager', 'custom'], default: 'base', description: 'base: the pilot’s. eager: turn-end rules rewritten against dead air. custom: the text below.' },
    { key: 'system', label: 'Custom instructions', type: 'text', default: PROMPTS.base, when: { prompt: ['custom'] } },
    { key: 'example', label: 'Example tape', type: 'text', default: EXAMPLE_TAPE, description: 'Shown as the user turn, so the live tape reads as the model’s own continuation.' },
    { key: 'maxWords', label: 'Max words per cell', type: 'number', default: 6, description: 'Cap on the TTS headroom (two cells ahead at 0.3 s a word).' },
    { key: 'ttsLeadMs', label: 'TTS lead (ms)', type: 'number', default: 400, description: 'From writing a word to hearing it, for the headroom.' },
    { key: 'cutAfterMs', label: 'Cut after user voice (ms)', type: 'number', default: 200, description: 'User voice over the agent’s answer this long stops it (backchannel cues are never cut).' },
    {
      key: 'respondBias',
      label: 'Respond bias',
      type: 'number',
      default: 2.2,
      description: 'Added to the logit of “ respond” in silent cells (llama.cpp logit_bias): zero-shot, Gemma waits too long once the user is done. Pilot: +2.2 took the median answer from 2.8 to 2.0 s with no cut-ins; +3 cut into pauses. 0: off.',
    },
    {
      key: 'idleKeep',
      label: 'Idle lines kept',
      type: 'number',
      default: 2,
      description: 'Each finished run of idle cells (nobody talking, the agent listening) is cut to this many lines in the prompt: a tape full of listen lines makes Gemma copy them and stop answering. 0: keep them all.',
    },
    {
      key: 'answerAfterS',
      label: 'Answer after silence (s)',
      type: 'number',
      default: 3,
      description: 'Backstop: once the user has said something and then stayed silent this long with no answer, the next call may only respond. 0: off.',
    },
    { key: 'idleCells', label: 'Stop calling after silent cells', type: 'number', default: 6, description: 'No calls (listen) once this many cells in a row had no user words or voice, unless the agent is speaking. 0: always call.' },
    {
      key: 'cutOverCells',
      label: 'Cut off ends after silent cells',
      type: 'number',
      default: 3,
      description: 'After a cut, this many cells without user words or voice (and no resume or answer) end the cut-off state: the next lines say silent. 0: it lasts until Gemma resumes or answers.',
    },
    {
      key: 'specCells',
      label: 'Write ahead (cells)',
      type: 'number',
      default: 6,
      description:
        'Once the user stops (VAD silent, ASR caught up), write up to this many cells ahead as if they stay silent. A reply started there is rendered by TTS and held; its audio plays the moment a real cell confirms the silence. User voice or new words drop it. 0: off.',
    },
    { key: 'maxLines', label: 'Max tape lines', type: 'number', default: 72, description: 'Longer tapes are cut to the last lines below, the rest kept as a plain transcript (one full prefill).' },
    { key: 'keepLines', label: 'Lines kept when cut', type: 'number', default: 40 },
    { key: 'stableWords', label: 'Commit stable partial words', type: 'boolean', default: true, description: 'Put ASR partial words on the tape once a later partial repeats them (else only finals).' },
    { key: 'slot', label: 'llama.cpp slot', type: 'number', default: 0, description: 'id_slot for the tape’s calls (its prompt cache). -1: any.' },
    { key: 'temperature', label: 'Temperature', type: 'number', default: 0 },
    {
      key: 'nProbs',
      label: 'Odds (top tokens)',
      type: 'number',
      default: 8,
      description: 'llama.cpp n_probs: the alternatives Gemma weighed for each token it wrote, for the Tape inspector (its odds for each action). 0: off.',
    },
    {
      key: 'greeting',
      label: 'Opening line',
      type: 'string',
      default: "Hi! What's on your mind?",
      description: 'Said as soon as the ASR is connected, and written on the tape. Empty: wait for the user.',
    },
  ],
  channels: [{ name: CHANNEL, description: 'llama.cpp /completion: one JSON request (prompt, grammar), one JSON answer' }],
  requires: () => ['llm'],
  create(ctx) {
    const p = ctx.params;
    const cellMs = Math.max(50, Number(p.cellMs) || 640);
    const timeScale = Number(p.timeScale) > 0 ? Number(p.timeScale) : 1;
    const { system, example } = timed(
      p.prompt === 'custom' ? String(p.system) : (PROMPTS[p.prompt] ?? PROMPTS.base),
      String(p.example).replace(/\s+$/, ''),
      (cellMs * timeScale) / 1000,
    );
    let head = promptHead(system, example);
    let earlier: string[] = [];

    /** The agent as the tape has it: live, or as a speculation imagines it. */
    interface Side {
      state: TapeState;
      reply: Reply | null;
      /** Session ms when the words sent so far will have been said. */
      queueEnd: number;
      /** Cells in a row without user words or voice. */
      idle: number;
      /** The user said something the agent hasn't answered yet. */
      owed: boolean;
    }
    const live: Side = { state: 'silent', reply: null, queueEnd: 0, idle: 0, owed: false };
    const cells: Cell[] = [];
    let t0 = 0;
    let cellNo = 0;
    let winWords: string[] = [];
    let winVoice = false;
    /** The VAD hears the user. */
    let speaking = false;
    let call: Call | null = null;
    let ridNo = 0;
    let cutTimer: TimerHandle | null = null;
    let greet: string[] | null = null;
    let greetFinish = false;
    let greeted = false;
    /** ASR: words of the current segment already on the tape, and the last partial's words. */
    let segN = 0;
    let prevPartial: string[] = [];

    /**
     * A speculation: the cells after the live tape, written ahead on the assumption that the user stays
     * silent. A reply it starts is rendered by TTS but held; each real cell that matches its cell (no
     * words, no voice) adopts what it decided, and the reply's audio is released by the first one.
     */
    interface Spec {
      id: number;
      side: Side;
      /** Decided cells, oldest first, with what they did to the side. */
      cells: (Cell & { after: TapeState; stats: JsonObject })[];
      /** Its reply's audio was released (a real cell adopted its start). */
      released: boolean;
      /** Nothing more to write ahead. */
      stopped?: string;
      /** The real cell waiting for the speculative call in flight for it. */
      awaiting?: Cell;
    }
    let spec: Spec | null = null;
    let specNo = 0;

    const tOf = (n: number) => Math.round(n * cellMs * timeScale) / 1000;
    const setFloor = () => {
      ctx.setState('state', live.state);
      ctx.setState('floor', live.state === 'speaking' || live.reply?.playing ? 'speaking' : 'listening');
    };
    setFloor();

    const emitCell = (c: Cell, extra: JsonObject = {}) => {
      const l = line(c);
      ctx.setState('line', l);
      ctx.emit('turn', { type: 'tape.cell', n: c.n, t: c.t, at: t0 + c.n * cellMs, state: c.state, action: c.action, say: c.say, user: c.user, voice: c.voice, line: l, ...extra });
    };

    /** The tape gets too long for the slot: keep the last lines, the rest as a transcript in the head. */
    const trim = () => {
      if (cells.length <= Math.max(2, p.maxLines)) return;
      const drop = cells.splice(0, cells.length - Math.max(1, p.keepLines));
      earlier = [...earlier, ...transcript(drop)];
      // About the last 800 characters of it.
      let chars = 0;
      let from = earlier.length;
      while (from > 0 && chars + earlier[from - 1].length < 800) chars += earlier[--from].length;
      earlier = earlier.slice(from);
      head = promptHead(system, example, earlier.join('\n'));
      ctx.log('info', 'tape shortened', { dropped: drop.length, kept: cells.length });
    };

    /** `s` sends `ws` (in cell `c`, at session time `at`). */
    const say = (s: Side, c: Cell, ws: string[], at: number) => {
      const r = s.reply;
      if (!ws.length || !r) return;
      c.say.push(...ws);
      for (const w of ws) r.sent.push({ n: c.n, word: w });
      ctx.emit('text', { type: 'llm.delta', rid: r.rid, text: `${ws.join(' ')} ` });
      s.queueEnd = Math.max(at + p.ttsLeadMs, s.queueEnd) + ws.length * WORD_S * 1000;
    };
    const endReply = (r: Reply, status: string) => {
      if (r.ended) return;
      r.ended = true;
      ctx.emit('text', { type: 'llm.end', rid: r.rid, status, stopReason: status === 'done' ? 'stop' : null, text: r.sent.map((w) => w.word).join(' '), tape: true });
    };

    /**
     * Carry out the action written for cell `c` on side `s` (at session time `at`). A speculation's
     * reply is held in TTS, and its backchannel waits for a real cell to adopt it.
     */
    const act = (s: Side, c: Cell, action: string, ws: string[], at: number, held: boolean) => {
      c.action = action;
      if (action === 'backchannel') {
        c.say = ws;
        if (!held) sayBackchannel(c);
      } else if (action === 'respond' || action === 'resume' || (action === 'continue' && !s.reply)) {
        if (!ws.length) return;
        if (s.reply && !s.reply.ended) endReply(s.reply, 'cancelled');
        s.reply = { rid: `${ctx.id}-r${ridNo++}`, sent: [], ended: false, playing: false };
        s.owed = false;
        ctx.emit('text', { type: 'llm.start', rid: s.reply.rid, turnId: null, tape: action, ...(held ? { held: true } : {}) });
        s.queueEnd = Math.max(s.queueEnd, at);
        say(s, c, ws, at);
        s.state = 'speaking';
      } else if (action === 'continue') {
        say(s, c, ws, at);
      } else if (action === 'finish') {
        const r = s.reply;
        if (r) {
          say(s, c, ws, at);
          endReply(r, 'done');
          if (!r.playing && !r.sent.length) s.reply = null;
        }
        s.state = 'silent';
      }
    };
    const sayBackchannel = (c: Cell) => {
      const text = c.say.join(' ');
      ctx.emit('text', { type: 'bc.say', id: `${ctx.id}-bc${c.n}`, line: text, take: text, tailMs: 150, why: 'tape' });
    };

    const defaultAction = (s: TapeState) => (s === 'speaking' ? 'continue' : 'listen');
    /** The backstop: the user said something, then stayed silent `answerAfterS` without an answer. */
    const mustAnswer = (s: Side) => s.state === 'silent' && s.owed && p.answerAfterS > 0 && s.idle >= Math.max(1, Math.round((p.answerAfterS * 1000) / cellMs));

    /**
     * Cell `n` of side `s`: its line up to the action, and whether a call decides it. The same rules
     * write the live cell at its boundary and a speculative one ahead of time (at `at`).
     */
    const plan = (s: Side, n: number, user: string[], voice: boolean, at: number) => {
      // Cut off, and the user has been quiet for a while without Gemma resuming or answering: the turn is over.
      if (s.state === 'cut' && p.cutOverCells > 0 && s.idle >= p.cutOverCells) s.state = 'silent';
      const c = newCell(n, tOf(n), s.state, user, voice);
      s.idle = user.length || voice ? 0 : s.idle + 1;
      if (user.length) s.owed = true;
      const room = headroom(at / 1000, s.queueEnd / 1000, p.ttsLeadMs / 1000, cellMs / 1000, Math.max(1, p.maxWords));
      const forced = s.state === 'speaking' && room === 0 ? 'TTS queue full' : s.state !== 'speaking' && p.idleCells > 0 && s.idle > p.idleCells && !mustAnswer(s) ? 'idle' : null;
      return { c, forced, maxWords: Math.max(1, room) };
    };

    const ask = (c: Cell, tape: Cell[], maxWords: number, sp: Spec | null) => {
      const answer = mustAnswer(sp ? sp.side : live);
      const allowed = answer ? ['respond'] : ACTIONS[c.state];
      const bias: JsonObject = c.state === 'silent' && !answer && p.respondBias ? { logit_bias: [[' respond', p.respondBias]] } : {};
      const body: JsonObject = {
        prompt: tapePrompt(head, trimIdle(tape, p.idleKeep)),
        grammar: grammar(allowed, maxWords),
        n_predict: 32,
        temperature: p.temperature,
        cache_prompt: true,
        stop: ['\n'],
        ...(p.slot >= 0 ? { id_slot: p.slot } : {}),
        ...(p.nProbs > 0 ? { n_probs: p.nProbs } : {}),
        ...bias,
      };
      const k: Call = { cell: c, state: c.state, allowed, maxWords, t0: ctx.now(), io: undefined as unknown as Call['io'], spec: sp, ...(answer ? { answer } : {}) };
      call = k;
      ctx.setState('calling', sp ? 'ahead' : 'live');
      k.io = ctx.io.open(CHANNEL, { body }, {
        onRecv(data) {
          if (data && typeof data === 'object' && !Array.isArray(data)) k.res = data as JsonObject;
        },
        onClose(data) {
          if (call !== k) return;
          call = null;
          ctx.setState('calling', false);
          const error = (data as JsonObject | null)?.error ?? null;
          if (k.spec) specAnswered(k, error);
          else answered(k, error);
        },
      });
    };

    /** What a call wrote, or null (and why) if it failed. */
    const read = (k: Call, error: Json) => {
      const timings = (k.res?.timings ?? null) as JsonObject | null;
      const odds = actionOdds(k.res, k.allowed);
      const stats: JsonObject = {
        ms: Math.round(ctx.now() - k.t0),
        maxWords: k.maxWords,
        allowed: k.allowed,
        xid: k.io.xid,
        ...(odds ? { odds } : {}),
        ...(k.answer ? { backstop: 'answer after silence' } : {}),
        ...(timings ? { promptN: timings.prompt_n ?? null, promptMs: timings.prompt_ms ?? null, predictedN: timings.predicted_n ?? null, predictedMs: timings.predicted_ms ?? null } : {}),
      };
      ctx.setState('lastMs', stats.ms);
      const text = typeof k.res?.content === 'string' ? k.res.content : null;
      const d = text === null ? null : parseAction(text);
      if (error || !d || !k.allowed.includes(d.action)) {
        if (error || text === null) ctx.log('warn', 'tape call failed', { error, cell: k.cell.n });
        else ctx.log('warn', 'tape call wrote no action', { text, cell: k.cell.n });
        return { d: null, text, stats };
      }
      return { d, text, stats };
    };

    const answered = (k: Call, error: Json) => {
      const c = k.cell;
      const { d, text, stats } = read(k, error);
      if (!d) {
        c.action = defaultAction(k.state);
        emitCell(c, { ...stats, failed: true, ...(text !== null ? { wrote: text } : {}) });
      } else if (live.state !== k.state) {
        // The agent was cut off while Gemma wrote: what it wrote no longer applies.
        c.action = defaultAction(k.state);
        emitCell(c, { ...stats, dropped: `the state changed to ${live.state}`, wrote: text });
      } else {
        act(live, c, d.action, d.words, ctx.now(), false);
        setFloor();
        emitCell(c, { ...stats, late: ctx.now() > t0 + (c.n + 1) * cellMs });
      }
      maybeSpeculate();
    };

    // ---- speculation ---------------------------------------------------------------------------

    const emitSpec = (event: string, data: JsonObject = {}) => ctx.emit('turn', { type: 'tape.spec', event, spec: spec?.id ?? null, ...data });

    /** Give up the speculation: its held reply is dropped, unless it was released (then it is live). */
    const dropSpec = (why: string) => {
      const sp = spec;
      if (!sp) return;
      if (call?.spec === sp) {
        if (!call.io.closed) call.io.close({ status: 'cancelled' });
        call = null;
        ctx.setState('calling', false);
      }
      const r = sp.side.reply;
      if (r && !sp.released) {
        ctx.emit('text', { type: 'llm.drop', rid: r.rid });
        if (!r.ended) endReply(r, 'dropped');
      }
      emitSpec('drop', { why, ahead: sp.cells.length, ...(r && !sp.released ? { rid: r.rid } : {}) });
      spec = null;
      ctx.setState('ahead', 0);
    };

    /** Start writing ahead, if the user is silent, ASR has their words and the agent has nothing in flight. */
    const maybeSpeculate = () => {
      if (spec || call || p.specCells <= 0 || speaking || greet || greetFinish) return;
      if (live.state === 'speaking' || live.reply) return;
      // An ASR partial with words not on the tape yet: they are still to come.
      if (prevPartial.length > segN) return;
      if (p.idleCells > 0 && live.idle >= p.idleCells) return;
      spec = { id: specNo++, side: { ...live }, cells: [], released: false };
      emitSpec('start', { from: cellNo + 1 });
      specStep();
    };

    /** Write the next cell ahead. */
    const specStep = () => {
      const sp = spec;
      if (!sp || call || sp.stopped) return;
      const n = (sp.cells.at(-1)?.n ?? cellNo) + 1;
      if (n - cellNo > p.specCells) return;
      // The cell in progress has what the user did so far; later ones are silent.
      const now = n === cellNo + 1;
      const { c, forced, maxWords } = plan(sp.side, n, now ? [...winWords] : [], now ? winVoice || speaking : false, t0 + n * cellMs);
      if (forced === 'idle') {
        sp.stopped = 'idle';
        return;
      }
      if (forced) {
        sp.cells.push({ ...c, after: sp.side.state, stats: { forced } });
        return specStep();
      }
      ask(c, [...cells, ...sp.cells, c], maxWords, sp);
    };

    const specAnswered = (k: Call, error: Json) => {
      const sp = k.spec as Spec;
      if (spec !== sp) return;
      const c = k.cell;
      const { d, text, stats } = read(k, error);
      if (!d) {
        sp.stopped = 'call failed';
        emitSpec('stop', { why: 'call failed', n: c.n, ...(text !== null ? { wrote: text } : {}) });
        if (sp.awaiting?.n === c.n) {
          const real = sp.awaiting;
          sp.awaiting = undefined;
          real.action = defaultAction(real.state);
          emitCell(real, { ...stats, failed: true });
        }
        return;
      }
      act(sp.side, c, d.action, d.words, Math.max(ctx.now(), t0 + c.n * cellMs), !sp.released);
      const done = { ...c, after: sp.side.state, stats };
      sp.cells.push(done);
      emitSpec('cell', { n: c.n, action: c.action, line: line(c), ...stats });
      ctx.setState('ahead', sp.cells.length);
      if (d.action === 'finish' || d.action === 'backchannel') sp.stopped = d.action;
      if (sp.awaiting?.n === c.n) {
        // The real cell came while this was written: it takes it now.
        const real = sp.awaiting;
        sp.awaiting = undefined;
        adopt(sp, real, true);
      }
      specStep();
    };

    /** Real cell `c` matches the speculation's first cell: take what it decided. */
    const adopt = (sp: Spec, c: Cell, late = false) => {
      const s = sp.cells.shift()!;
      c.action = s.action;
      c.say = [...s.say];
      // The words were sent under the speculative cell: they are this cell's.
      const r = sp.side.reply;
      if (s.action === 'backchannel') sayBackchannel(c);
      if ((s.action === 'respond' || s.action === 'resume') && r && !sp.released) {
        sp.released = true;
        ctx.emit('text', { type: 'llm.release', rid: r.rid });
        emitSpec('release', { n: c.n, rid: r.rid });
      }
      if (sp.released || s.action === 'listen' || s.action === 'backchannel') {
        live.state = s.after;
        if (sp.released) {
          live.reply = r;
          live.queueEnd = Math.max(live.queueEnd, sp.side.queueEnd);
        }
      }
      setFloor();
      emitCell(c, { ...s.stats, spec: late ? 'adopted late' : 'adopted' });
      ctx.setState('ahead', sp.cells.length);
      if (!sp.cells.length && sp.stopped) {
        spec = null;
        emitSpec('done', { n: c.n });
        maybeSpeculate();
      } else specStep();
    };

    /** The real cell `c` differs from the speculation (the user spoke): what was released stays live. */
    const breakSpec = (c: Cell, why: string) => {
      const sp = spec!;
      if (sp.released && sp.side.reply) {
        // Words already sent for cells ahead belong to this cell now.
        const r = sp.side.reply;
        const rest = sp.cells.filter((x) => x.n >= c.n);
        const ws = rest.flatMap((x) => x.say);
        for (const w of r.sent) if (w.n >= c.n) w.n = c.n;
        if (ws.length) {
          c.say = ws;
          c.action = rest.some((x) => x.action === 'finish') ? 'finish' : 'continue';
        }
        live.state = rest.length ? rest[rest.length - 1].after : live.state;
        live.reply = r;
        sp.released = true;
      }
      dropSpec(why);
    };

    // ---- the live tape -------------------------------------------------------------------------

    /** One cell ends: write its line up to the action and decide it (or take the speculation's). */
    const boundary = () => {
      const n = ++cellNo;
      const user = winWords;
      const voice = winVoice || speaking;
      winWords = [];
      winVoice = speaking;
      ctx.setTimer(Math.max(0, t0 + (n + 1) * cellMs - ctx.now()), boundary, 'cell');
      if (spec) {
        const sp = spec;
        const ahead = sp.cells[0]?.n === n ? sp.cells[0] : null;
        const pending = !ahead && call?.spec === sp && call.cell.n === n ? call.cell : null;
        const expect = ahead ?? pending;
        const same = !!expect && expect.voice === voice && expect.user.join(' ') === user.join(' ');
        if (same) {
          // The real cell is the one written ahead.
          const { c } = plan(live, n, user, voice, ctx.now());
          cells.push(c);
          trim();
          if (ahead) return adopt(sp, c);
          sp.awaiting = c;
          return;
        }
        const planned = plan(live, n, user, voice, ctx.now());
        const c = planned.c;
        breakSpec(c, expect ? 'the user spoke' : 'nothing written for this cell');
        if (c.say.length) {
          cells.push(c);
          trim();
          setFloor();
          return emitCell(c, { forced: 'sent ahead' });
        }
        return liveCell(c, planned.forced, planned.maxWords);
      }
      if (greet) {
        const { c } = plan(live, n, user, voice, ctx.now());
        // The opening line, said as a cue: on the tape as an answer.
        c.action = 'respond';
        c.say = greet;
        greet = null;
        greetFinish = true;
        live.state = 'speaking';
        cells.push(c);
        return emitCell(c, { forced: 'greeting' });
      }
      if (greetFinish && live.state === 'speaking' && !live.reply) {
        const { c } = plan(live, n, user, voice, ctx.now());
        greetFinish = false;
        c.action = 'finish';
        live.state = 'silent';
        cells.push(c);
        setFloor();
        emitCell(c, { forced: 'greeting' });
        return maybeSpeculate();
      }
      greetFinish = false;
      const { c, forced, maxWords } = plan(live, n, user, voice, ctx.now());
      liveCell(c, forced, maxWords);
    };
    const liveCell = (c: Cell, forced: string | null, maxWords: number) => {
      cells.push(c);
      trim();
      if (call) {
        c.action = defaultAction(live.state);
        return emitCell(c, { forced: 'a call is still out' });
      }
      if (forced) {
        emitCell(c, { forced });
        return maybeSpeculate();
      }
      ask(c, cells, maxWords, null);
    };

    /** User voice over the answer: stop it. */
    const audible = () => !!live.reply && !live.reply.cutting && (live.state === 'speaking' || live.reply.playing);
    const armCut = () => {
      cutTimer?.clear();
      cutTimer = null;
      if (!audible()) return;
      cutTimer = ctx.setTimer(Math.max(0, p.cutAfterMs), () => {
        cutTimer = null;
        if (speaking && audible()) cut();
      }, 'cut');
    };
    const cut = () => {
      if (spec) dropSpec('the agent was cut off');
      const r = live.reply!;
      ctx.emit('control', { type: 'cancel' });
      endReply(r, 'cancelled');
      live.state = 'cut';
      live.queueEnd = ctx.now();
      if (r.playing) r.cutting = true;
      else heardOf(r, '');
      setFloor();
    };
    const heardOf = (r: Reply, heard: string) => {
      const k = words(heard).length;
      const gone = cutTape(cells, r.sent, k);
      ctx.emit('turn', { type: 'tape.cut', rid: r.rid, heard, said: Math.min(k, r.sent.length), unsaid: gone });
      ctx.emit('turn', { type: 'turn.heard', rid: r.rid, text: heard });
      if (live.reply === r) live.reply = null;
      setFloor();
      maybeSpeculate();
    };

    const commitWords = (ws: string[]) => {
      // No periods on the user's words: the ASR ends every final with one, also at a pause mid-sentence.
      const add = ws.map(noPeriods).filter(Boolean);
      if (!add.length) return;
      winWords.push(...add);
      // New words: the cell in progress isn't what the speculation assumed.
      if (spec && !spec.released) dropSpec('new words');
    };

    return {
      start() {
        t0 = ctx.now();
        const greeting = p.greeting.trim();
        ctx.emit('text', { type: 'bc.lines', lines: [...AGENT_BACKCHANNELS, ...(greeting ? [greeting] : [])] });
        ctx.setTimer(cellMs, boundary, 'cell');
      },
      onInput(port, msg: Msg) {
        if (port === 'vad') {
          if (msg.type === 'vad.speech_start') {
            speaking = true;
            winVoice = true;
            ctx.setState('userSpeaking', true);
            if (spec && !spec.released) dropSpec('user voice');
            armCut();
          } else if (msg.type === 'vad.speech_end') {
            speaking = false;
            ctx.setState('userSpeaking', false);
            cutTimer?.clear();
            cutTimer = null;
            maybeSpeculate();
          }
          return;
        }
        if (port === 'playback') {
          const r = live.reply;
          if (!r || msg.rid !== r.rid) return;
          if (msg.type === 'playback.start') {
            r.playing = true;
            if (speaking) armCut();
          } else if (msg.type === 'playback.end') {
            r.playing = false;
            live.queueEnd = Math.min(live.queueEnd, ctx.now());
            if (r.ended) live.reply = null;
          } else if (msg.type === 'playback.interrupted') {
            r.playing = false;
            if (r.cutting) heardOf(r, String(msg.heard ?? ''));
          }
          setFloor();
          maybeSpeculate();
          return;
        }
        // transcript
        if (msg.type === 'asr.open') {
          const line = p.greeting.trim();
          if (greeted || !line) return;
          greeted = true;
          greet = words(line);
          ctx.emit('turn', { type: 'turn.greet', text: line });
          ctx.emit('text', { type: 'bc.say', id: `${ctx.id}-greet`, line, take: line, tailMs: 150, why: 'greeting' });
          return;
        }
        const ws = words(String(msg.text ?? ''));
        if (msg.type === 'asr.final') {
          commitWords(ws.slice(segN));
          segN = 0;
          prevPartial = [];
          maybeSpeculate();
        } else if (msg.type === 'asr.partial' && p.stableWords) {
          let same = 0;
          while (same < ws.length && same < prevPartial.length && ws[same] === prevPartial[same]) same++;
          const stable = Math.min(same, ws.length - 1);
          if (stable > segN) {
            commitWords(ws.slice(segN, stable));
            segN = stable;
          }
          prevPartial = ws;
        }
      },
      stop() {
        cutTimer?.clear();
        if (call && !call.io.closed) call.io.close({ status: 'cancelled' });
        call = null;
      },
    };
  },
};
