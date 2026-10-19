import type { NodeDef, TimerHandle } from '../core/node';
import type { Json } from '../core/types';

interface TurnParams {
  commitOn: string;
}

type Floor = 'listening' | 'thinking' | 'speaking';

/**
 * Half-duplex turn taking: the agent takes the floor when the user's utterance
 * is final and gives it back when playback ends. User speech while the agent
 * talks is ignored (and logged). Step 2 replaces this with a full-duplex floor controller.
 */
export const halfDuplexTurnNode: NodeDef<TurnParams> = {
  type: 'turn.half-duplex',
  title: 'Half-duplex turns',
  category: 'control',
  description: 'Commits a user turn on ASR endpointing; ignores the user while the agent is speaking.',
  inputs: [
    { name: 'transcript', type: 'transcript' },
    { name: 'vad', type: 'vad' },
    { name: 'playback', type: 'playback' },
  ],
  outputs: [{ name: 'turn', type: 'turn' }],
  params: [{ key: 'commitOn', label: 'Commit on', type: 'select', options: ['speech_final', 'utterance_end'], default: 'speech_final' }],
  create(ctx) {
    let floor: Floor = 'listening';
    let parts: string[] = [];
    let from = -1;
    let to = -1;
    let turnNo = 0;
    const setFloor = (f: Floor, why: string) => {
      floor = f;
      ctx.setState('floor', f);
      ctx.log('info', `floor -> ${f}`, { why });
    };
    const commit = (why: string) => {
      const text = parts.join(' ').trim();
      if (!text) return;
      const turnId = `${ctx.id}-u${turnNo++}`;
      ctx.emit('turn', { type: 'turn.user', turnId, text, from, to });
      parts = [];
      from = to = -1;
      ctx.setState('pending', '');
      setFloor('thinking', why);
    };
    ctx.setState('floor', floor);

    return {
      onInput(port, msg) {
        if (port === 'playback') {
          if (msg.type === 'playback.start') setFloor('speaking', 'playback started');
          else if (msg.type === 'playback.end' && floor === 'speaking') setFloor('listening', 'playback ended');
          return;
        }
        if (port === 'vad') {
          if (msg.type === 'vad.speech_start' || msg.type === 'vad.speech_end') ctx.setState('userSpeaking', msg.type === 'vad.speech_start');
          return;
        }
        if (floor === 'speaking') {
          if (msg.type === 'asr.final' && msg.text) ctx.log('info', 'ignored user speech while agent speaking', { text: msg.text as string });
          return;
        }
        if (msg.type === 'asr.final') {
          const text = (msg.text as string).trim();
          if (text) {
            parts.push(text);
            if (from < 0) from = msg.start as number;
            to = msg.end as number;
            ctx.setState('pending', parts.join(' '));
          }
          if (msg.speechFinal && ctx.params.commitOn === 'speech_final') commit('speech_final');
        } else if (msg.type === 'asr.utterance_end') {
          commit('utterance_end');
        }
      },
    };
  },
};

interface DuplexParams {
  useModel: boolean;
  asrGraceMs: number;
  commitOn: string;
  bargeIn: boolean;
  minSpeechMs: number;
  backchannels: string;
  maxBackchannelMs: number;
}

/** Who holds the floor, and whether the user is talking over the agent while it does. */
export type FloorState = 'listening' | 'thinking' | 'speaking' | 'thinking+user' | 'speaking+user';
export type FloorEvent = 'commit' | 'playback_start' | 'playback_end' | 'user_start' | 'backchannel' | 'take_floor';

/**
 * The floor as an explicit state machine. These are the only transitions; an
 * event a state has no entry for is ignored (and logged). Every transition is
 * emitted as a `turn.transition` message with what triggered it.
 */
export const FLOOR: Record<FloorState, Partial<Record<FloorEvent, FloorState>>> = {
  listening: { commit: 'thinking' },
  thinking: { playback_start: 'speaking', user_start: 'thinking+user' },
  speaking: { playback_end: 'listening', user_start: 'speaking+user' },
  'thinking+user': { playback_start: 'speaking+user', backchannel: 'thinking', take_floor: 'listening' },
  'speaking+user': { playback_end: 'listening', backchannel: 'speaking', take_floor: 'listening' },
};

const holder = (s: FloorState): Floor => s.replace('+user', '') as Floor;

type Candidate = {
  /** Sample where the user's speech started (from the VAD, or the model). */
  at: number;
  since: number;
  timer?: TimerHandle;
  /** Latest ASR text for this speech, partial or final. */
  text: string;
  finals: { text: string; start: number; end: number }[];
};

const words = (s: string) => s.toLowerCase().match(/[\p{L}\p{N}']+/gu) ?? [];

/**
 * Full-duplex turn taking, as the state machine in FLOOR. The user can take
 * the floor back while the agent is thinking or speaking; while the agent
 * speaks, the speaker reports how much of the reply was heard, and that is
 * passed on so the LLM's history keeps only what the user actually heard.
 * Speech while the agent is still thinking means the user wasn't done: the
 * pending reply is dropped and their earlier turn is resent with the new words.
 *
 * What decides, when the turn model is running: its take-the-floor event
 * (user) stops the agent, user speech that ends without one is a backchannel,
 * and its end-of-turn event (user) commits the turn, once ASR has the words.
 * Without it: speech that lasts `minSpeechMs` and isn't just a backchannel
 * word stops the agent, and ASR endpointing commits the turn.
 *
 * Every barge-in decision is also emitted as a `turn.decision` message
 * (candidate, backchannel, interrupt, continue) so the timeline can show it.
 */
export const duplexTurnNode: NodeDef<DuplexParams> = {
  type: 'turn.duplex',
  title: 'Full-duplex turns',
  category: 'control',
  description: 'Floor state machine: commits user turns, lets the user interrupt the agent (barge-in) and lets backchannels through. Uses the turn model when it runs, else VAD/ASR rules.',
  inputs: [
    { name: 'transcript', type: 'transcript' },
    { name: 'vad', type: 'vad' },
    { name: 'playback', type: 'playback' },
    { name: 'scores', type: 'turnscores' },
  ],
  outputs: [
    { name: 'turn', type: 'turn' },
    { name: 'control', type: 'control', description: 'cancel: stop the reply in flight (LLM, TTS, playback)' },
  ],
  params: [
    {
      key: 'useModel',
      label: 'Use the turn model',
      type: 'boolean',
      default: true,
      description: 'Decide on the turn model’s end-of-turn and take-the-floor events while it runs. The rules below apply when it is off or not loaded yet.',
    },
    {
      key: 'asrGraceMs',
      label: 'Wait for ASR after end of turn (ms)',
      type: 'number',
      default: 600,
      description: 'When the model ends the turn before ASR has finalised the last words, wait this long for the final, then commit with the partial.',
      when: { useModel: [true] },
    },
    {
      key: 'commitOn',
      label: 'Commit on (without the model)',
      type: 'select',
      options: ['speech_final', 'utterance_end'],
      default: 'speech_final',
      description: 'ASR endpointing used when the turn model is not running. utterance_end always commits, as a safety net.',
    },
    { key: 'bargeIn', label: 'Barge-in', type: 'boolean', default: true, description: 'Let the user interrupt the agent' },
    {
      key: 'minSpeechMs',
      label: 'Min speech to interrupt (ms)',
      type: 'number',
      default: 400,
      description: 'Without the model: how long the user must talk over the agent before it stops. Shorter speech is treated as a backchannel or noise.',
      when: { bargeIn: [true] },
    },
    {
      key: 'backchannels',
      label: 'Backchannel words',
      type: 'string',
      default: 'yeah, yes, yep, uh-huh, uh huh, mhm, mm, hmm, ok, okay, right, alright, all right, sure, i see, got it, cool, nice, wow',
      description: 'Without the model: if what the ASR heard so far is only these, keep talking (up to the max below)',
      when: { bargeIn: [true] },
    },
    {
      key: 'maxBackchannelMs',
      label: 'Max backchannel (ms)',
      type: 'number',
      default: 1200,
      description: 'Speech over the agent longer than this interrupts, whatever the words or the model say',
      when: { bargeIn: [true] },
    },
  ],
  create(ctx) {
    const p = ctx.params;
    let state: FloorState = 'listening';
    let parts: string[] = [];
    let from = -1;
    let to = -1;
    let turnNo = 0;
    /** The last committed turn, for resending it when the user wasn't done. */
    let last: { id: string; text: string; from: number; to: number } | null = null;
    let replaces: string | null = null;
    let candidate: Candidate | null = null;
    /** Latest ASR partial since the last final, while listening. */
    let partial: { text: string; start: number; end: number } | null = null;
    /** The model ended the turn, but ASR hasn't finalised its last words yet. */
    let eotWait: { why: string; timer: TimerHandle } | null = null;
    /** The turn model is loaded and scoring. */
    let model = false;
    const useModel = () => p.useModel && model;
    /**
     * The model decides barge-in only over the agent's speech: while the agent is
     * thinking its channel is silent, so take-the-floor rarely fires and the user
     * adding to their turn is judged by the rules.
     */
    const modelDecides = () => useModel() && holder(state) === 'speaking';
    const backchannel = new Set(
      String(p.backchannels)
        .split(',')
        .map((s) => words(s).join(' '))
        .filter(Boolean),
    );
    const isBackchannel = (text: string) => {
      const w = words(text);
      if (!w.length) return true; // nothing recognisable yet
      // Any run of listed phrases, e.g. "yeah yeah" or "oh okay".
      let i = 0;
      while (i < w.length) {
        let matched = 0;
        for (let n = Math.min(3, w.length - i); n > 0 && !matched; n--) if (backchannel.has(w.slice(i, i + n).join(' '))) matched = n;
        if (!matched) return false;
        i += matched;
      }
      return true;
    };

    /** Fire `on`: take the transition FLOOR has for it in the current state, if any. */
    const go = (on: FloorEvent, why: string): boolean => {
      const next = FLOOR[state][on];
      if (!next) {
        ctx.log('debug', `ignored ${on} in ${state}`, { why });
        return false;
      }
      const prev = state;
      state = next;
      ctx.emit('turn', { type: 'turn.transition', from: prev, to: next, on, why });
      ctx.setState('state', next);
      if (holder(next) !== holder(prev)) {
        ctx.setState('floor', holder(next));
        ctx.log('info', `floor -> ${holder(next)}`, { why });
      }
      return true;
    };
    const decide = (decision: string, data: Record<string, Json> = {}) => {
      ctx.emit('turn', { type: 'turn.decision', decision, floor: holder(state), ...data });
      ctx.setState('lastDecision', decision);
    };
    const collect = (text: string, start: number, end: number) => {
      text = text.trim();
      if (!text) return;
      parts.push(text);
      if (from < 0) from = start;
      to = end;
      ctx.setState('pending', parts.join(' '));
    };
    const commit = (why: string) => {
      eotWait?.timer.clear();
      eotWait = null;
      const text = parts.join(' ').trim();
      if (!text || state !== 'listening') return;
      const turnId = `${ctx.id}-u${turnNo++}`;
      ctx.emit('turn', { type: 'turn.user', turnId, text, from, to, ...(replaces ? { replaces } : {}) });
      last = { id: turnId, text, from, to };
      replaces = null;
      parts = [];
      partial = null;
      from = to = -1;
      ctx.setState('pending', '');
      go('commit', why);
    };
    /** The model says the user's turn is over: commit once ASR has the words. */
    const endOfTurn = (why: string) => {
      if (eotWait) return;
      if (!partial && parts.length) return commit(why);
      const timer = ctx.setTimer(
        p.asrGraceMs,
        () => {
          const w = eotWait!;
          eotWait = null;
          if (partial) collect(partial.text, partial.start, partial.end);
          partial = null;
          commit(`${w.why}; no ASR final in time`);
        },
        'eot-wait',
      );
      eotWait = { why, timer };
    };

    /** The user takes the floor: stop the reply and start listening. */
    const yieldFloor = (c: Candidate, why: string) => {
      c.timer?.clear();
      candidate = null;
      const wasThinking = state === 'thinking+user';
      decide(wasThinking ? 'continue' : 'interrupt', { why, ms: ctx.now() - c.since, text: c.text });
      ctx.emit('control', { type: 'cancel' });
      if (wasThinking && last) {
        // They weren't done: resend their turn together with what they say now.
        replaces = last.id;
        parts = [last.text];
        from = last.from;
        to = last.to;
        ctx.setState('pending', parts.join(' '));
      }
      go('take_floor', why);
      for (const f of c.finals) collect(f.text, f.start, f.end);
    };

    /** Rules without the model: enough speech that isn't only backchannel words. */
    const consider = (c: Candidate) => {
      if (modelDecides()) return;
      const ms = ctx.now() - c.since;
      if (ms < p.minSpeechMs) return;
      if (c.text && isBackchannel(c.text) && ms < p.maxBackchannelMs) return;
      yieldFloor(c, c.text && !isBackchannel(c.text) ? 'words' : 'sustained speech');
    };

    /** The user starts talking while the agent holds the floor. */
    const overlap = (at: number, why: string): Candidate | null => {
      if (!p.bargeIn || candidate || !go('user_start', why)) return candidate;
      const c: Candidate = { at, since: ctx.now(), text: '', finals: [] };
      candidate = c;
      decide('candidate', { at });
      const check = () => {
        c.timer = undefined;
        if (candidate !== c) return;
        if (modelDecides()) {
          // The model may never call it a floor take; this much speech is one anyway.
          if (ctx.now() - c.since >= p.maxBackchannelMs) yieldFloor(c, 'sustained speech');
          else c.timer = ctx.setTimer(p.maxBackchannelMs - (ctx.now() - c.since), check, 'barge-in');
          return;
        }
        consider(c);
        // Still a backchannel: look again when it would be too long for one.
        if (candidate === c) c.timer = ctx.setTimer(Math.max(1, p.maxBackchannelMs - (ctx.now() - c.since)), check, 'barge-in');
      };
      c.timer = ctx.setTimer(modelDecides() ? p.maxBackchannelMs : p.minSpeechMs, check, 'barge-in');
      return c;
    };

    ctx.setState('floor', holder(state));
    ctx.setState('state', state);

    return {
      onInput(port, msg) {
        if (port === 'scores') {
          if (msg.type === 'turnmodel.status') {
            model = msg.status === 'ready';
            if (p.useModel) ctx.setState('using', model ? 'turn model' : 'rules');
            return;
          }
          if (!useModel() || msg.who !== 'user') return;
          const why = `model: ${msg.type === 'turnmodel.eot' ? 'end of turn' : 'take the floor'} p=${msg.p}`;
          if (msg.type === 'turnmodel.int' && p.bargeIn && state !== 'listening') {
            const c = overlap(msg.at as number, why);
            if (c) yieldFloor(c, why);
          } else if (msg.type === 'turnmodel.eot' && state === 'listening') endOfTurn(why);
          return;
        }
        if (port === 'playback') {
          if (msg.type === 'playback.start') go('playback_start', 'playback started');
          else if (msg.type === 'playback.end') {
            const c = candidate;
            if (!go('playback_end', 'playback ended')) return;
            if (c) {
              // The agent finished while the user was already talking: just listen.
              c.timer?.clear();
              candidate = null;
              for (const f of c.finals) collect(f.text, f.start, f.end);
            }
          } else if (msg.type === 'playback.interrupted' && last)
            // What the user heard before we stopped, for the LLM's history.
            ctx.emit('turn', { type: 'turn.heard', turnId: last.id, text: (msg.heard as string) ?? '' });
          return;
        }
        if (port === 'vad') {
          if (msg.type === 'vad.speech_start') {
            ctx.setState('userSpeaking', true);
            if (state === 'thinking' || state === 'speaking') overlap(msg.at as number, 'user speech (VAD)');
            else if (eotWait) {
              // They went on after the model called the end of their turn: wait for the next one.
              eotWait.timer.clear();
              eotWait = null;
            }
          } else if (msg.type === 'vad.speech_end') {
            ctx.setState('userSpeaking', false);
            if (candidate) {
              const c = candidate;
              c.timer?.clear();
              candidate = null;
              const why = modelDecides() ? 'speech ended without the model calling a floor take' : 'speech ended before it counted';
              decide('backchannel', { ms: ctx.now() - c.since, text: c.text, why });
              go('backchannel', why);
            }
          }
          return;
        }
        const text = ((msg.text as string) ?? '').trim();
        if (candidate && (msg.type === 'asr.partial' || msg.type === 'asr.final') && (msg.end as number) > candidate.at - 4800) {
          // Speech over the agent: decide on what it says.
          if (text) candidate.text = msg.type === 'asr.final' ? [...candidate.finals.map((f) => f.text), text].join(' ') : text;
          if (msg.type === 'asr.final' && text) candidate.finals.push({ text, start: msg.start as number, end: msg.end as number });
          consider(candidate);
          return;
        }
        if (state !== 'listening') {
          if (msg.type === 'asr.final' && text) ctx.log('info', 'ignored short speech while the agent has the floor', { text });
          return;
        }
        if (msg.type === 'asr.partial') {
          partial = text ? { text, start: msg.start as number, end: msg.end as number } : null;
        } else if (msg.type === 'asr.final') {
          collect(text, msg.start as number, msg.end as number);
          partial = null;
          if (eotWait) commit(eotWait.why);
          else if (!useModel() && msg.speechFinal && p.commitOn === 'speech_final') commit('speech_final');
        } else if (msg.type === 'asr.utterance_end') {
          commit('utterance_end');
        }
      },
    };
  },
};
