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
  commitOn: string;
  bargeIn: boolean;
  minSpeechMs: number;
  backchannels: string;
  maxBackchannelMs: number;
}

type Candidate = {
  /** Sample where the user's speech started (from the VAD). */
  at: number;
  since: number;
  timer?: TimerHandle;
  /** Latest ASR text for this speech, partial or final. */
  text: string;
  finals: { text: string; start: number; end: number }[];
};

const words = (s: string) => s.toLowerCase().match(/[\p{L}\p{N}']+/gu) ?? [];

/**
 * Full-duplex turn taking. The user can take the floor back while the agent is
 * thinking or speaking: speech that lasts `minSpeechMs` (and isn't just a
 * backchannel like "uh-huh") cancels the agent's reply. While the agent
 * speaks, the speaker reports how much of the reply was heard, and that is
 * passed on so the LLM's history keeps only what the user actually heard.
 * Speech while the agent is still thinking means the user wasn't done: the
 * pending reply is dropped and their earlier turn is resent with the new words.
 *
 * Every decision is emitted as a `turn.decision` message (candidate,
 * backchannel, interrupt, continue) so the timeline can show it.
 */
export const duplexTurnNode: NodeDef<DuplexParams> = {
  type: 'turn.duplex',
  title: 'Full-duplex turns',
  category: 'control',
  description: 'Commits a user turn on ASR endpointing; the user can interrupt the agent (barge-in), backchannels are let through.',
  inputs: [
    { name: 'transcript', type: 'transcript' },
    { name: 'vad', type: 'vad' },
    { name: 'playback', type: 'playback' },
  ],
  outputs: [
    { name: 'turn', type: 'turn' },
    { name: 'control', type: 'control', description: 'cancel: stop the reply in flight (LLM, TTS, playback)' },
  ],
  params: [
    { key: 'commitOn', label: 'Commit on', type: 'select', options: ['speech_final', 'utterance_end'], default: 'speech_final' },
    { key: 'bargeIn', label: 'Barge-in', type: 'boolean', default: true, description: 'Let the user interrupt the agent' },
    {
      key: 'minSpeechMs',
      label: 'Min speech to interrupt (ms)',
      type: 'number',
      default: 400,
      description: 'How long the user must talk over the agent before it stops. Shorter speech is treated as a backchannel or noise.',
      when: { bargeIn: [true] },
    },
    {
      key: 'backchannels',
      label: 'Backchannel words',
      type: 'string',
      default: 'yeah, yes, yep, uh-huh, uh huh, mhm, mm, hmm, ok, okay, right, alright, all right, sure, i see, got it, cool, nice, wow',
      description: 'If what the ASR heard so far is only these, keep talking (up to the max below)',
      when: { bargeIn: [true] },
    },
    {
      key: 'maxBackchannelMs',
      label: 'Max backchannel (ms)',
      type: 'number',
      default: 1200,
      description: 'Speech longer than this interrupts even if it sounds like a backchannel',
      when: { bargeIn: [true] },
    },
  ],
  create(ctx) {
    const p = ctx.params;
    let floor: Floor = 'listening';
    let parts: string[] = [];
    let from = -1;
    let to = -1;
    let turnNo = 0;
    /** The last committed turn, for resending it when the user wasn't done. */
    let last: { id: string; text: string; from: number; to: number } | null = null;
    let replaces: string | null = null;
    let candidate: Candidate | null = null;
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

    const setFloor = (f: Floor, why: string) => {
      if (f === floor) return;
      floor = f;
      ctx.setState('floor', f);
      ctx.log('info', `floor -> ${f}`, { why });
    };
    const decide = (decision: string, data: Record<string, Json> = {}) => {
      ctx.emit('turn', { type: 'turn.decision', decision, floor, ...data });
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
      const text = parts.join(' ').trim();
      if (!text) return;
      const turnId = `${ctx.id}-u${turnNo++}`;
      ctx.emit('turn', { type: 'turn.user', turnId, text, from, to, ...(replaces ? { replaces } : {}) });
      last = { id: turnId, text, from, to };
      replaces = null;
      parts = [];
      from = to = -1;
      ctx.setState('pending', '');
      setFloor('thinking', why);
    };

    /** The user takes the floor: stop the reply and start listening. */
    const yieldFloor = (c: Candidate, why: string) => {
      c.timer?.clear();
      candidate = null;
      const wasThinking = floor === 'thinking';
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
      setFloor('listening', why);
      for (const f of c.finals) collect(f.text, f.start, f.end);
    };

    const consider = (c: Candidate) => {
      const ms = ctx.now() - c.since;
      if (ms < p.minSpeechMs) return;
      if (c.text && isBackchannel(c.text) && ms < p.maxBackchannelMs) return;
      yieldFloor(c, c.text && !isBackchannel(c.text) ? 'words' : 'sustained speech');
    };

    ctx.setState('floor', floor);

    return {
      onInput(port, msg) {
        if (port === 'playback') {
          if (msg.type === 'playback.start') setFloor('speaking', 'playback started');
          else if (msg.type === 'playback.end' && floor === 'speaking') {
            setFloor('listening', 'playback ended');
            if (candidate) {
              // The agent finished while the user was already talking: just listen.
              const c = candidate;
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
            if (p.bargeIn && floor !== 'listening' && !candidate) {
              const c: Candidate = { at: msg.at as number, since: ctx.now(), text: '', finals: [] };
              candidate = c;
              decide('candidate', { at: c.at });
              const check = () => {
                c.timer = undefined;
                if (candidate !== c) return;
                consider(c);
                // Still a backchannel: look again when it would be too long for one.
                if (candidate === c) c.timer = ctx.setTimer(Math.max(1, p.maxBackchannelMs - (ctx.now() - c.since)), check, 'barge-in');
              };
              c.timer = ctx.setTimer(p.minSpeechMs, check, 'barge-in');
            }
          } else if (msg.type === 'vad.speech_end') {
            ctx.setState('userSpeaking', false);
            if (candidate) {
              const c = candidate;
              c.timer?.clear();
              candidate = null;
              decide('backchannel', { ms: ctx.now() - c.since, text: c.text });
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
        if (floor !== 'listening') {
          if (msg.type === 'asr.final' && text) ctx.log('info', 'ignored short speech while the agent has the floor', { text });
          return;
        }
        if (msg.type === 'asr.final') {
          collect(text, msg.start as number, msg.end as number);
          if (msg.speechFinal && p.commitOn === 'speech_final') commit('speech_final');
        } else if (msg.type === 'asr.utterance_end') {
          commit('utterance_end');
        }
      },
    };
  },
};
