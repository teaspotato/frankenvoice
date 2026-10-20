import type { IOHandle, NodeDef, TimerHandle } from '../core/node';
import type { Json, JsonObject, Msg } from '../core/types';
import {
  ACTIONS,
  AGENT_BACKCHANNELS,
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
  tapePrompt,
  transcript,
  type Cell,
  type SentWord,
  type TapeState,
} from './tape';

interface TapeParams {
  cellMs: number;
  prompt: string;
  system: string;
  example: string;
  maxWords: number;
  ttsLeadMs: number;
  cutAfterMs: number;
  idleCells: number;
  cutOverCells: number;
  maxLines: number;
  keepLines: number;
  stableWords: boolean;
  slot: number;
  temperature: number;
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
 * Every cell is emitted as `tape.cell` (the line as written, the call's latency and timings), every
 * cut as `tape.cut`, so the dashboard shows the tape as it is written.
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
    { key: 'cellMs', label: 'Cell (ms)', type: 'number', default: 640, description: 'One decision per cell. The prompt says 0.64 s.' },
    { key: 'prompt', label: 'Prompt', type: 'select', options: ['base', 'eager', 'custom'], default: 'base', description: 'base: the pilot’s. eager: turn-end rules rewritten against dead air. custom: the text below.' },
    { key: 'system', label: 'Custom instructions', type: 'text', default: PROMPTS.base, when: { prompt: ['custom'] } },
    { key: 'example', label: 'Example tape', type: 'text', default: EXAMPLE_TAPE, description: 'Shown as the user turn, so the live tape reads as the model’s own continuation.' },
    { key: 'maxWords', label: 'Max words per cell', type: 'number', default: 6, description: 'Cap on the TTS headroom (two cells ahead at 0.3 s a word).' },
    { key: 'ttsLeadMs', label: 'TTS lead (ms)', type: 'number', default: 400, description: 'From writing a word to hearing it, for the headroom.' },
    { key: 'cutAfterMs', label: 'Cut after user voice (ms)', type: 'number', default: 200, description: 'User voice over the agent’s answer this long stops it (backchannel cues are never cut).' },
    { key: 'idleCells', label: 'Stop calling after silent cells', type: 'number', default: 6, description: 'No calls (listen) once this many cells in a row had no user words or voice, unless the agent is speaking. 0: always call.' },
    {
      key: 'cutOverCells',
      label: 'Cut off ends after silent cells',
      type: 'number',
      default: 3,
      description: 'After a cut, this many cells without user words or voice (and no resume or answer) end the cut-off state: the next lines say silent. 0: it lasts until Gemma resumes or answers.',
    },
    { key: 'maxLines', label: 'Max tape lines', type: 'number', default: 72, description: 'Longer tapes are cut to the last lines below, the rest kept as a plain transcript (one full prefill).' },
    { key: 'keepLines', label: 'Lines kept when cut', type: 'number', default: 40 },
    { key: 'stableWords', label: 'Commit stable partial words', type: 'boolean', default: true, description: 'Put ASR partial words on the tape once a later partial repeats them (else only finals).' },
    { key: 'slot', label: 'llama.cpp slot', type: 'number', default: 0, description: 'id_slot for the tape’s calls (its prompt cache). -1: any.' },
    { key: 'temperature', label: 'Temperature', type: 'number', default: 0 },
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
    const system = p.prompt === 'custom' ? String(p.system) : (PROMPTS[p.prompt] ?? PROMPTS.base);
    const example = String(p.example).replace(/\s+$/, '');
    let head = promptHead(system, example);
    let earlier: string[] = [];

    let state: TapeState = 'silent';
    const cells: Cell[] = [];
    let t0 = 0;
    let cellNo = 0;
    let winWords: string[] = [];
    let winVoice = false;
    /** The VAD hears the user. */
    let speaking = false;
    let idle = 0;
    let call: Call | null = null;
    let reply: Reply | null = null;
    let ridNo = 0;
    /** Session ms when the words sent so far will have been said. */
    let queueEnd = 0;
    let cutTimer: TimerHandle | null = null;
    let greet: string[] | null = null;
    let greetFinish = false;
    let greeted = false;
    /** ASR: words of the current segment already on the tape, and the last partial's words. */
    let segN = 0;
    let prevPartial: string[] = [];

    const setFloor = () => {
      ctx.setState('state', state);
      ctx.setState('floor', state === 'speaking' || reply?.playing ? 'speaking' : 'listening');
    };
    setFloor();

    const emitCell = (c: Cell, extra: JsonObject = {}) => {
      const l = line(c);
      ctx.setState('line', l);
      ctx.emit('turn', { type: 'tape.cell', n: c.n, t: c.t, at: t0 + c.n * cellMs, state: c.state, action: c.action, say: c.say, user: c.user, voice: c.voice, line: l, ...extra });
    };

    const commitWords = (ws: string[]) => {
      // No periods on the user's words: the ASR ends every final with one, also at a pause mid-sentence.
      winWords.push(...ws.map(noPeriods).filter(Boolean));
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

    const queueWords = (n: number) => {
      queueEnd = Math.max(ctx.now() + p.ttsLeadMs, queueEnd) + n * WORD_S * 1000;
    };

    const say = (r: Reply, c: Cell, ws: string[]) => {
      if (!ws.length) return;
      c.say.push(...ws);
      for (const w of ws) r.sent.push({ n: c.n, word: w });
      ctx.emit('text', { type: 'llm.delta', rid: r.rid, text: `${ws.join(' ')} ` });
      queueWords(ws.length);
    };
    const endReply = (r: Reply, status: string) => {
      if (r.ended) return;
      r.ended = true;
      ctx.emit('text', { type: 'llm.end', rid: r.rid, status, stopReason: status === 'done' ? 'stop' : null, text: r.sent.map((w) => w.word).join(' '), tape: true });
    };

    /** Carry out the action Gemma wrote for cell `c`. */
    const act = (c: Cell, action: string, ws: string[]) => {
      c.action = action;
      if (action === 'backchannel') {
        const text = ws.join(' ');
        c.say = ws;
        ctx.emit('text', { type: 'bc.say', id: `${ctx.id}-bc${c.n}`, line: text, take: text, tailMs: 150, why: 'tape' });
      } else if (action === 'respond' || action === 'resume' || (action === 'continue' && !reply)) {
        if (!ws.length) return;
        if (reply && !reply.ended) endReply(reply, 'cancelled');
        const r: Reply = { rid: `${ctx.id}-r${ridNo++}`, sent: [], ended: false, playing: false };
        reply = r;
        ctx.emit('text', { type: 'llm.start', rid: r.rid, turnId: null, tape: action });
        queueEnd = Math.max(queueEnd, ctx.now());
        say(r, c, ws);
        state = 'speaking';
      } else if (action === 'continue' && reply) {
        say(reply, c, ws);
      } else if (action === 'finish') {
        if (reply) {
          say(reply, c, ws);
          endReply(reply, 'done');
          if (!reply.playing && !reply.sent.length) reply = null;
        }
        state = 'silent';
      }
      setFloor();
    };

    const defaultAction = (s: TapeState) => (s === 'speaking' ? 'continue' : 'listen');

    const ask = (c: Cell, maxWords: number) => {
      const allowed = ACTIONS[c.state];
      const body: JsonObject = {
        prompt: tapePrompt(head, cells),
        grammar: grammar(allowed, maxWords),
        n_predict: 32,
        temperature: p.temperature,
        cache_prompt: true,
        stop: ['\n'],
        ...(p.slot >= 0 ? { id_slot: p.slot } : {}),
      };
      const k: Call = { cell: c, state: c.state, allowed, maxWords, t0: ctx.now(), io: undefined as unknown as Call['io'] };
      call = k;
      ctx.setState('calling', true);
      k.io = ctx.io.open(CHANNEL, { body }, {
        onRecv(data) {
          if (data && typeof data === 'object' && !Array.isArray(data)) k.res = data as JsonObject;
        },
        onClose(data) {
          if (call !== k) return;
          call = null;
          ctx.setState('calling', false);
          answered(k, (data as JsonObject | null)?.error ?? null);
        },
      });
    };

    const answered = (k: Call, error: Json) => {
      const c = k.cell;
      const ms = Math.round(ctx.now() - k.t0);
      ctx.setState('lastMs', ms);
      const timings = (k.res?.timings ?? null) as JsonObject | null;
      const stats: JsonObject = {
        ms,
        maxWords: k.maxWords,
        allowed: k.allowed,
        ...(timings ? { promptN: timings.prompt_n ?? null, promptMs: timings.prompt_ms ?? null, predictedN: timings.predicted_n ?? null, predictedMs: timings.predicted_ms ?? null } : {}),
      };
      const text = typeof k.res?.content === 'string' ? k.res.content : null;
      const d = text === null ? null : parseAction(text);
      if (error || !d || !k.allowed.includes(d.action)) {
        if (error || text === null) ctx.log('warn', 'tape call failed', { error, cell: c.n });
        else ctx.log('warn', 'tape call wrote no action', { text, cell: c.n });
        c.action = defaultAction(k.state);
        return emitCell(c, { ...stats, failed: true, ...(text !== null ? { wrote: text } : {}) });
      }
      if (state !== k.state) {
        // The agent was cut off while Gemma wrote: what it wrote no longer applies.
        c.action = defaultAction(k.state);
        return emitCell(c, { ...stats, dropped: `the state changed to ${state}`, wrote: text });
      }
      act(c, d.action, d.words);
      emitCell(c, { ...stats, late: ctx.now() > t0 + (c.n + 1) * cellMs });
    };

    /** One cell ends: write its line up to the action and decide it. */
    const boundary = () => {
      const n = ++cellNo;
      // Cut off, and the user has been quiet for a while without Gemma resuming or answering: the turn is over.
      if (state === 'cut' && p.cutOverCells > 0 && idle >= p.cutOverCells) {
        state = 'silent';
        setFloor();
      }
      const c = newCell(n, Math.round(n * cellMs) / 1000, state, winWords, winVoice || speaking);
      winWords = [];
      winVoice = speaking;
      idle = c.user.length || c.voice ? 0 : idle + 1;
      cells.push(c);
      trim();
      ctx.setTimer(Math.max(0, t0 + (n + 1) * cellMs - ctx.now()), boundary, 'cell');
      if (greet) {
        // The opening line, said as a cue: on the tape as an answer.
        c.action = 'respond';
        c.say = greet;
        greet = null;
        greetFinish = true;
        state = 'speaking';
        return emitCell(c, { forced: 'greeting' });
      }
      if (greetFinish && state === 'speaking' && !reply) {
        greetFinish = false;
        c.action = 'finish';
        state = 'silent';
        setFloor();
        return emitCell(c, { forced: 'greeting' });
      }
      greetFinish = false;
      if (call) {
        c.action = defaultAction(state);
        return emitCell(c, { forced: 'a call is still out' });
      }
      const room = headroom(ctx.now() / 1000, queueEnd / 1000, p.ttsLeadMs / 1000, cellMs / 1000, Math.max(1, p.maxWords));
      if (state === 'speaking' && room === 0) return emitCell(c, { forced: 'TTS queue full' });
      if (state !== 'speaking' && p.idleCells > 0 && idle > p.idleCells) return emitCell(c, { forced: 'idle' });
      ask(c, Math.max(1, room));
    };

    /** User voice over the answer: stop it. */
    const audible = () => !!reply && !reply.cutting && (state === 'speaking' || reply.playing);
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
      const r = reply!;
      ctx.emit('control', { type: 'cancel' });
      endReply(r, 'cancelled');
      state = 'cut';
      queueEnd = ctx.now();
      if (r.playing) r.cutting = true;
      else heardOf(r, '');
      setFloor();
    };
    const heardOf = (r: Reply, heard: string) => {
      const k = words(heard).length;
      const gone = cutTape(cells, r.sent, k);
      ctx.emit('turn', { type: 'tape.cut', rid: r.rid, heard, said: Math.min(k, r.sent.length), unsaid: gone });
      ctx.emit('turn', { type: 'turn.heard', rid: r.rid, text: heard });
      if (reply === r) reply = null;
      setFloor();
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
            armCut();
          } else if (msg.type === 'vad.speech_end') {
            speaking = false;
            ctx.setState('userSpeaking', false);
            cutTimer?.clear();
            cutTimer = null;
          }
          return;
        }
        if (port === 'playback') {
          const r = reply;
          if (!r || msg.rid !== r.rid) return;
          if (msg.type === 'playback.start') {
            r.playing = true;
            if (speaking) armCut();
          } else if (msg.type === 'playback.end') {
            r.playing = false;
            queueEnd = Math.min(queueEnd, ctx.now());
            if (r.ended) reply = null;
          } else if (msg.type === 'playback.interrupted') {
            r.playing = false;
            if (r.cutting) heardOf(r, String(msg.heard ?? ''));
          }
          setFloor();
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
