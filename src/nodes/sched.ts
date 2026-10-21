import type { NodeDef, TimerHandle } from '../core/node';
import type { Msg } from '../core/types';
import { isAudioMsg } from '../core/types';

interface SchedParams {
  pace: boolean;
  tickMs: number;
  firstTokens: number;
  chunkTokens: number;
  leadWords: number;
  maxQueuedWords: number;
  minBufferedMs: number;
  holdUntilAudio: boolean;
}

/** What the scheduler knows of the reply being spoken, from the messages it sees. */
interface Reply {
  rid: string;
  t0: number;
  /** Words sent to TTS; the words its model had yet to read when it last said (tts.consumed), and the words sent by then. */
  sent: number;
  unread?: number;
  sentThen: number;
  /** When its first audio came out of TTS. */
  firstAudio?: number;
  llmDone: boolean;
  ttsDone: boolean;
  /** The LLM stopped at the end of a chunk and waits for a grant. */
  waiting: boolean;
  /** Tokens granted and not arrived yet. */
  owed: number;
}

/**
 * The tick scheduler: the pipeline's view of the GPU as 80 ms frames (one Kyutai frame each). Every
 * tick, TTS comes first (its queue and how far its audio runs ahead of playback), then the LLM
 * gets tokens in the slack, then ASR. Each tick is logged as `sched.tick`, with what ran during it
 * and the queue depths, so a replay shows the same ticks and the same decisions.
 *
 * The backends are separate processes today, so what it governs is when the LLM may generate:
 * with `pace` on, the LLM writes a reply in chunks (`sched.grant`, a short continuation of the
 * text so far). A reply starts with a burst of `firstTokens`; until TTS has its lookahead
 * (`leadWords`) more is granted at once, then held until the first audio is out. After that a
 * chunk is granted while TTS has fewer than `maxQueuedWords` words left to read and its audio runs
 * at least `minBufferedMs` ahead of playback, or whenever TTS is down to its lookahead or playback
 * is about to run dry. Words left to read come from Kyutai (Consumed). Drafts are
 * capped at `firstTokens` and paced once they are committed. With `pace` off the LLM runs free
 * and the ticks are only logged.
 *
 * Only messages and timers decide anything, so a future engine running the models in one process
 * can take grants over its own channel and report its ticks, without changing this node or replay.
 */
export const schedNode: NodeDef<SchedParams> = {
  type: 'sched',
  title: 'Scheduler',
  category: 'control',
  description: 'Tick scheduler: 80 ms frames, TTS first, LLM tokens in the slack (lead-buffer pacing), every tick logged.',
  inputs: [
    { name: 'text', type: 'text', description: 'The LLM: llm.start, llm.delta, llm.chunk, llm.end' },
    { name: 'audio', type: 'audio', description: 'TTS: tts.sent, tts.consumed, tts.words, audio, tts.response_end' },
    { name: 'played', type: 'played', description: 'The speaker: when each chunk plays' },
    { name: 'transcript', type: 'transcript', description: 'ASR partials and finals' },
  ],
  outputs: [
    { name: 'control', type: 'control', description: 'sched.config, then sched.grant: tokens the LLM may write next' },
    { name: 'ticks', type: 'ticks', description: 'sched.tick, one per tick' },
  ],
  params: [
    { key: 'pace', label: 'Pace the LLM', type: 'boolean', default: true, description: 'Off: the LLM writes freely; ticks are still logged.' },
    { key: 'tickMs', label: 'Tick (ms)', type: 'number', default: 80, description: 'One Kyutai frame.' },
    { key: 'firstTokens', label: 'First burst (tokens)', type: 'number', default: 16, description: "A reply's (and a draft's) first request." },
    { key: 'chunkTokens', label: 'Chunk (tokens)', type: 'number', default: 8 },
    { key: 'leadWords', label: 'Lookahead (words)', type: 'number', default: 3, description: 'Kyutai needs this many words queued to speak: below it, the LLM always gets more.' },
    { key: 'maxQueuedWords', label: 'Max words queued', type: 'number', default: 6, description: 'No more tokens while TTS has this many words not yet spoken.' },
    { key: 'minBufferedMs', label: 'Min audio ahead (ms)', type: 'number', default: 300, description: 'Between the two, tokens only while TTS audio runs this far ahead of playback.' },
    { key: 'holdUntilAudio', label: 'Hold until first audio', type: 'boolean', default: true, description: "Once TTS has its lookahead, the reply's first audio gets the GPU alone." },
  ],
  create(ctx) {
    const p = ctx.params;
    let reply: Reply | undefined;
    let timer: TimerHandle | undefined;
    let n = 0;
    let t0 = 0;
    /** Session time the reply's audio is scheduled to play until. */
    let playEnd = 0;
    /** Since the last tick. */
    let acc = { tokens: 0, audioMs: 0, words: 0, partials: 0, finals: 0 };

    const buffered = () => Math.max(0, Math.round(playEnd - ctx.now()));
    /** Words TTS has to go on: what it last said it had unread, plus what was sent since. */
    const queued = (r: Reply) => (r.unread ?? 0) + r.sent - r.sentThen;

    const decide = (r: Reply): { decision: string; why: string; grant: number } => {
      if (!p.pace) return { decision: 'free', why: 'pacing off', grant: 0 };
      if (r.llmDone) return { decision: 'done', why: 'reply written', grant: 0 };
      if (!r.waiting) return { decision: 'busy', why: 'chunk in flight', grant: 0 };
      if (r.ttsDone) return { decision: 'grant', why: 'TTS is done with it', grant: p.chunkTokens };
      if (r.firstAudio !== undefined && buffered() < p.tickMs) return { decision: 'grant', why: 'playback about to run dry', grant: p.chunkTokens };
      const q = queued(r);
      if (q < p.leadWords) return { decision: 'grant', why: r.firstAudio === undefined ? 'burst to the lookahead' : 'TTS short of words', grant: p.chunkTokens };
      if (r.firstAudio === undefined && p.holdUntilAudio) return { decision: 'hold', why: 'first audio first', grant: 0 };
      if (q >= p.maxQueuedWords) return { decision: 'hold', why: 'TTS has enough words', grant: 0 };
      if (buffered() < p.minBufferedMs) return { decision: 'hold', why: 'TTS not far enough ahead', grant: 0 };
      return { decision: 'grant', why: 'TTS ahead', grant: p.chunkTokens };
    };

    const tick = () => {
      timer = undefined;
      const r = reply;
      if (!r) return;
      const d = decide(r);
      if (d.grant > 0) {
        r.waiting = false;
        r.owed += d.grant;
        ctx.emit('control', { type: 'sched.grant', rid: r.rid, tokens: d.grant });
      }
      ctx.emit('ticks', {
        type: 'sched.tick',
        n: n++,
        rid: r.rid,
        ms: Math.round(ctx.now() - r.t0),
        // In order: TTS frames first, then the LLM, then ASR.
        tts: { audioMs: Math.round(acc.audioMs), words: acc.words, queuedWords: queued(r), bufferedMs: buffered(), done: r.ttsDone },
        llm: { tokens: acc.tokens, pending: p.pace ? r.owed : null, decision: d.decision, why: d.why, granted: d.grant },
        asr: { partials: acc.partials, finals: acc.finals },
      });
      acc = { tokens: 0, audioMs: 0, words: 0, partials: 0, finals: 0 };
      // Done once the reply is written and spoken, and its audio has played.
      if (r.llmDone && r.ttsDone && buffered() === 0) {
        reply = undefined;
        ctx.setState('phase', 'idle');
        return;
      }
      ctx.setState('phase', d.decision);
      // On the tick grid from the reply's start, whatever the timer's lateness.
      timer = ctx.setTimer(Math.max(0, t0 + n * p.tickMs - ctx.now()), tick, 'tick');
    };

    const begin = (rid: string) => {
      timer?.clear();
      reply = { rid, t0: ctx.now(), sent: 0, sentThen: 0, llmDone: false, ttsDone: false, waiting: false, owed: 0 };
      t0 = ctx.now();
      n = 0;
      acc = { tokens: 0, audioMs: 0, words: 0, partials: 0, finals: 0 };
      timer = ctx.setTimer(0, tick, 'tick');
    };

    return {
      start() {
        ctx.emit('control', { type: 'sched.config', pace: p.pace, firstTokens: p.firstTokens });
        ctx.setState('phase', 'idle');
      },
      onInput(port, msg: Msg) {
        const r = reply;
        if (port === 'text') {
          if (msg.type === 'llm.start') return begin(msg.rid as string);
          if (!r || msg.rid !== r.rid) return;
          if (msg.type === 'llm.delta') {
            acc.tokens++;
            r.owed = Math.max(0, r.owed - 1);
          } else if (msg.type === 'llm.chunk') {
            r.owed = 0;
            r.waiting = msg.stop === 'length';
          } else if (msg.type === 'llm.end') r.llmDone = true;
          return;
        }
        if (port === 'played') {
          if (isAudioMsg(msg) && !msg.cue) playEnd = Math.max(playEnd, (msg.at as number) + (msg.samples.length / msg.sampleRate) * 1000);
          else if (msg.type === 'played.flush') playEnd = Math.min(playEnd, msg.at as number);
          return;
        }
        if (port === 'transcript') {
          if (msg.type === 'asr.partial') acc.partials++;
          else if (msg.type === 'asr.final') acc.finals++;
          return;
        }
        if (!r || msg.rid !== r.rid) return;
        if (isAudioMsg(msg)) {
          if (msg.cue) return;
          r.firstAudio ??= ctx.now();
          acc.audioMs += (msg.samples.length / msg.sampleRate) * 1000;
        } else if (msg.type === 'tts.sent') r.sent += msg.words as number;
        else if (msg.type === 'tts.consumed') {
          r.unread = msg.queued as number;
          r.sentThen = r.sent;
        } else if (msg.type === 'tts.words') acc.words += (msg.words as unknown[]).length;
        else if (msg.type === 'tts.response_end' || msg.type === 'tts.cancelled') r.ttsDone = true;
      },
      stop() {
        timer?.clear();
      },
    };
  },
};
