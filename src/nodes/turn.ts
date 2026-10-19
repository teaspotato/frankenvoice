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
  speculate: boolean;
  draftAt: number;
  modelNeedsVad: boolean;
  maxSilenceMs: number;
  overlap: string;
  onset: string;
  duckDb: number;
  floorWords: string;
  asides: boolean;
  classifyWaitMs: number;
}

/** Who holds the floor, and whether the user is talking over the agent while it does. */
export type FloorState = 'listening' | 'thinking' | 'speaking' | 'thinking+user' | 'speaking+user';
export type FloorEvent = 'commit' | 'playback_start' | 'playback_end' | 'user_start' | 'backchannel' | 'take_floor' | 'aside';

/**
 * The floor as an explicit state machine. These are the only transitions; an
 * event a state has no entry for is ignored (and logged). Every transition is
 * emitted as a `turn.transition` message with what triggered it.
 */
export const FLOOR: Record<FloorState, Partial<Record<FloorEvent, FloorState>>> = {
  listening: { commit: 'thinking' },
  thinking: { playback_start: 'speaking', user_start: 'thinking+user' },
  // aside: the agent finished the sentence the user spoke over, and stops to take in what they said.
  speaking: { playback_end: 'listening', user_start: 'speaking+user', aside: 'listening' },
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
  /** When the speech ended (and its last sample), while its words are still awaited (classify mode). */
  ended?: number;
  endAt?: number;
  wait?: TimerHandle;
  check?: () => void;
};

/** How long before the VAD's speech end a held end of turn still counts (the VAD's hangover and some). */
const EOT_HOLD_MS = 600;

type Partial_ = { text: string; start: number; end: number };

const words = (s: string) => s.toLowerCase().match(/[\p{L}\p{N}']+/gu) ?? [];
const FILLERS = new Set(['um', 'umm', 'uh', 'uhh', 'er', 'erm', 'ah']);
/** What a draft must match: the words, without case, punctuation or fillers. */
export const draftKey = (s: string) =>
  words(s)
    .filter((w) => !FILLERS.has(w))
    .join(' ');

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
 * (candidate, backchannel, noise, aside, interrupt, continue) so the timeline can show it.
 *
 * Speech over the agent, `overlap: classify` (the default): the agent is cut
 * off at once (`onset: cut`, a pause the speaker can resume from), so an
 * interruption lands, while the words come in (`duck` plays it `duckDb`
 * quieter instead; `none` lets it talk on). The reply is cancelled on a
 * floor-taking opener ("wait", "so", "actually"…), on speech longer than
 * `maxBackchannelMs`, or on the model's take-the-floor. When the speech ends,
 * its words decide: nothing (noise) or only backchannel words are dropped and
 * the agent goes on from where it stopped; anything else is an aside (a
 * confirmation, a quick addition): the agent goes on to the end of its
 * sentence, stops there and gets the aside as the user's turn. `overlap: legacy` is the earlier rule set:
 * `minSpeechMs` of speech that isn't only backchannel words stops the agent.
 *
 * Speculative replies: while the user is silent and their turn looks like it
 * is ending (the model's end-of-turn score reaches `draftAt`, or just silence
 * without the model), and ASR has caught up (a result arrived during the
 * silence, or the words are all final), the words so far go out as a
 * `turn.draft` so the LLM can start early. A draft is redone whenever the
 * words change during the silence and dropped (`turn.draft_drop`) if the user goes on. At commit the turn
 * names the draft (`draft`) when the words still match, and the LLM keeps the
 * reply it already started; otherwise the draft is dropped and the turn is
 * sent as usual.
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
      key: 'modelNeedsVad',
      label: 'Model floor takes need VAD speech',
      type: 'boolean',
      default: true,
      description: 'Ignore the model’s take-the-floor events while the VAD hears no speech (it fires them in silence too).',
      when: { useModel: [true] },
    },
    {
      key: 'maxSilenceMs',
      label: 'Commit after silence (ms)',
      type: 'number',
      default: 1500,
      description: 'With the model: commit the user’s words after this much silence even if the model hasn’t called the end of their turn (0 = never).',
      when: { useModel: [true] },
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
      description:
        'Without the model: how long the user must talk while the agent is thinking (and, in legacy mode, speaking) before it stops. Shorter speech is treated as a backchannel or noise.',
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
      description: 'Speech over the agent longer than this interrupts, whatever the words or the model say. In classify mode it also ends an aside: longer speech takes the floor.',
      when: { bargeIn: [true] },
    },
    {
      key: 'speculate',
      label: 'Draft replies early',
      type: 'boolean',
      default: true,
      description:
        'Start the LLM on the words so far while the user is silent, before the turn is committed. Kept if the final words match, else dropped and resent. Costs the tokens of dropped drafts.',
    },
    {
      key: 'draftAt',
      label: 'Draft at end-of-turn score',
      type: 'number',
      default: 0,
      description: 'With the turn model: draft only once its end-of-turn score for the user reaches this (it commits at its own threshold). 0 drafts on any silence once ASR has the words, as without the model.',
      when: { speculate: [true], useModel: [true] },
    },
    {
      key: 'overlap',
      label: 'Speech over the agent',
      type: 'select',
      options: ['classify', 'legacy'],
      default: 'classify',
      description:
        'classify: cut the agent off, then decide from the words: floor take (drop the reply), aside (go on to the end of the sentence, then answer it) or backchannel/noise (go on). legacy: stop after min speech unless it is only backchannel words.',
      when: { bargeIn: [true] },
    },
    {
      key: 'onset',
      label: 'While deciding',
      type: 'select',
      options: ['cut', 'duck', 'none'],
      default: 'cut',
      description:
        'What the agent does from the moment the user speaks over it until their words decide. cut: stop at once, and pick up where it stopped if it was only a backchannel or noise. duck: play quieter. none: talk on.',
      when: { bargeIn: [true], overlap: ['classify'] },
    },
    {
      key: 'duckDb',
      label: 'Duck the agent by (dB)',
      type: 'number',
      default: 12,
      description: 'While deciding on speech over the agent, play the agent this much quieter.',
      when: { bargeIn: [true], overlap: ['classify'], onset: ['duck'] },
    },
    {
      key: 'floorWords',
      label: 'Floor-taking openers',
      type: 'string',
      default: 'wait, stop, hold on, hang on, sorry, excuse me, actually, no, so, but, hey, well',
      description: 'Speech over the agent that starts with one of these stops it at once, however short.',
      when: { bargeIn: [true], overlap: ['classify'] },
    },
    {
      key: 'asides',
      label: 'Asides',
      type: 'boolean',
      default: true,
      description:
        'Short speech over the agent with real words (not backchannel words) is an aside: the agent finishes its sentence, then stops and answers it. Off: it is ignored like a backchannel.',
      when: { bargeIn: [true], overlap: ['classify'] },
    },
    {
      key: 'classifyWaitMs',
      label: 'Wait for the words (ms)',
      type: 'number',
      default: 1500,
      description: 'After speech over the agent ends, how long to wait for its ASR final before deciding on what was heard so far.',
      when: { bargeIn: [true], overlap: ['classify'] },
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
    let last: { id: string; text: string; from: number; to: number; tail?: Partial_; base?: string[] } | null = null;
    let replaces: string | null = null;
    let candidate: Candidate | null = null;
    /** Latest ASR partial since the last final, while listening. */
    let partial: Partial_ | null = null;
    /** The model ended the turn while the VAD still heard speech: acted on if the speech ends right after. */
    let eotHeld: { why: string; at: number } | null = null;
    /** The model ended the turn, but ASR hasn't finalised its last words yet. */
    let eotWait: { why: string; timer: TimerHandle } | null = null;
    /** The turn model is loaded and scoring. */
    let model = false;
    /** With the model: commits the words if the user stays silent this long without its end of turn. */
    let silence: TimerHandle | null = null;
    /** A Finalize'd final was held back because the user went on: their next silence commits it, ASR final or not. */
    let heldFinal = false;
    /** The user is talking (VAD). */
    let speaking = false;
    /** ASR has caught up with the silence: a result arrived during it, or the words before it were all final. */
    let caughtUp = false;
    /** The model's latest end-of-turn score for the user. */
    let eotUser = 0;
    /** The reply being drafted on the words so far, before the turn is committed. */
    let draft: { id: string; key: string } | null = null;
    let draftNo = 0;
    /** The agent's audio is cut off (or ducked) while speech over it is judged. */
    let held = false;
    /** Words said over the agent that it should take in, held until its current sentence ends. */
    let aside: { text: string; from: number; to: number; timer?: TimerHandle; cutting?: boolean } | null = null;
    /** When each sentence of the reply playing now starts (session time), from the speaker. */
    let segRid: string | null = null;
    let segStarts: number[] = [];
    const useModel = () => p.useModel && model;
    /** Speech over the agent's voice is judged by its words (classify mode); over its thinking, by the rules. */
    const classifies = () => p.bargeIn && p.overlap === 'classify' && holder(state) === 'speaking';
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
    const floorWords = new Set(
      String(p.floorWords)
        .split(',')
        .map((s) => words(s).join(' '))
        .filter(Boolean),
    );
    /** The speech opens with a floor-taking word ("wait", "so", "actually"…). */
    const opensFloor = (text: string) => {
      const w = words(text);
      for (let n = Math.min(3, w.length); n > 0; n--) if (floorWords.has(w.slice(0, n).join(' '))) return true;
      return false;
    };
    const hold = () => {
      if (held) return;
      if (p.onset === 'cut') ctx.emit('control', { type: 'pause' });
      else if (p.onset === 'duck' && p.duckDb > 0) ctx.emit('control', { type: 'duck', db: p.duckDb });
      else return;
      held = true;
    };
    const release = () => {
      if (!held) return;
      held = false;
      ctx.emit('control', { type: p.onset === 'cut' ? 'resume' : 'unduck' });
    };
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
    const dropDraft = (why: string) => {
      if (!draft) return;
      ctx.emit('turn', { type: 'turn.draft_drop', draftId: draft.id, why });
      draft = null;
      ctx.setState('draft', '');
    };
    /** Draft a reply on the words so far, if the turn looks like it is ending and they changed. */
    const maybeDraft = (why: string) => {
      if (!p.speculate || state !== 'listening' || speaking || !caughtUp) return;
      if (useModel() && eotUser < p.draftAt) return;
      const text = [...parts, partial?.text ?? ''].join(' ').trim();
      const key = draftKey(text);
      if (!key || draft?.key === key) return;
      dropDraft('words changed');
      draft = { id: `${ctx.id}-d${draftNo++}`, key };
      const span = partial ? { from: from < 0 ? partial.start : from, to: partial.end } : { from, to };
      ctx.emit('turn', { type: 'turn.draft', draftId: draft.id, text, ...span, why, ...(replaces ? { replaces } : {}) });
      ctx.setState('draft', text);
    };
    /** The words changed: redo the draft during silence, drop it if the user is still talking. */
    const refreshDraft = (why: string) => {
      if (draft && speaking && draftKey([...parts, partial?.text ?? ''].join(' ')) !== draft.key) return dropDraft('user went on');
      maybeDraft(why);
    };
    /** Commit the turn. `tail`: its last words are an ASR partial whose final may still come. */
    const commit = (why: string, tail?: Partial_, extra: Record<string, Json> = {}) => {
      heldFinal = false;
      eotWait?.timer.clear();
      eotWait = null;
      silence?.clear();
      silence = null;
      const text = parts.join(' ').trim();
      if (!text || state !== 'listening') return;
      const turnId = `${ctx.id}-u${turnNo++}`;
      let kept: string | null = null;
      if (draft && draft.key === draftKey(text)) {
        kept = draft.id;
        draft = null;
        ctx.setState('draft', '');
      } else dropDraft('final words differ');
      ctx.emit('turn', { type: 'turn.user', turnId, text, from, to, ...(replaces ? { replaces } : {}), ...(kept ? { draft: kept } : {}), ...extra });
      last = { id: turnId, text, from, to, ...(tail ? { tail, base: parts.slice(0, -1) } : {}) };
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
          const tail = partial ?? undefined;
          if (partial) collect(partial.text, partial.start, partial.end);
          partial = null;
          commit(`${w.why}; no ASR final in time`, tail);
        },
        'eot-wait',
      );
      eotWait = { why, timer };
    };

    /** The model may never call the end of the turn: don't let silence go on forever. */
    const armSilence = () => {
      silence?.clear();
      silence = null;
      if ((!useModel() && !heldFinal) || p.maxSilenceMs <= 0 || state !== 'listening' || speaking) return;
      silence = ctx.setTimer(
        p.maxSilenceMs,
        () => {
          silence = null;
          if (state === 'listening' && !speaking && (parts.length || partial)) endOfTurn(`${p.maxSilenceMs} ms of silence without ${useModel() ? "the model's end of turn" : 'another ASR final'}`);
        },
        'silence',
      );
    };

    /** The user takes the floor: stop the reply and start listening. */
    const yieldFloor = (c: Candidate, why: string) => {
      c.timer?.clear();
      c.wait?.clear();
      candidate = null;
      // Cancelling drops the held audio; the speaker restores its volume for the next reply.
      held = false;
      const wasThinking = state === 'thinking+user';
      decide(wasThinking ? 'continue' : 'interrupt', { why, ms: ctx.now() - c.since, text: c.text });
      ctx.emit('control', { type: 'cancel' });
      if (wasThinking && last) {
        // They weren't done: resend their turn together with what they say now.
        replaces = last.id;
        // A turn committed on a partial: its final is still to come, so the partial stays a partial.
        parts = last.tail ? [...last.base!] : [last.text];
        partial = last.tail ?? null;
        from = last.from;
        to = last.to;
        ctx.setState('pending', parts.join(' '));
      }
      go('take_floor', why);
      takeAside();
      for (const f of c.finals) {
        if (partial && f.start <= partial.end) partial = null;
        collect(f.text, f.start, f.end);
      }
      armSilence();
    };

    /** A held aside becomes the start of the user's next turn. */
    const takeAside = () => {
      const a = aside;
      if (!a) return;
      a.timer?.clear();
      aside = null;
      collect(a.text, a.from, a.to);
    };
    /** The agent stopped at the end of its sentence (or finished): the aside is the user's turn now. */
    const finishAside = (why: string) => {
      if (!aside) return;
      if (state === 'speaking') go('aside', why);
      if (state !== 'listening') return;
      takeAside();
      commit(why, undefined, { aside: true });
    };
    /** Stop at the start of the reply's next sentence, if it is known yet (else when the speaker announces it). */
    const scheduleAside = () => {
      const a = aside;
      if (!a || a.timer || a.cutting) return;
      const next = segStarts.find((t) => t > ctx.now());
      if (next === undefined) return;
      a.timer = ctx.setTimer(
        next - ctx.now(),
        () => {
          a.timer = undefined;
          if (aside !== a) return;
          // The user is talking again: wait for the sentence after.
          if (candidate) return scheduleAside();
          a.cutting = true;
          ctx.emit('control', { type: 'cancel' });
          // The speaker answers with playback.interrupted; if it had nothing playing, go on without it.
          a.timer = ctx.setTimer(300, () => finishAside('aside at sentence end'), 'aside');
        },
        'aside',
      );
    };
    const holdAside = (c: Candidate) => {
      const fin = c.finals.length ? c.finals : null;
      const text = fin ? fin.map((f) => f.text).join(' ') : c.text;
      const span = fin ? { from: fin[0].start, to: fin[fin.length - 1].end } : { from: -1, to: -1 };
      if (aside) {
        aside.text = `${aside.text} ${text}`;
        if (aside.from < 0) aside.from = span.from;
        if (span.to >= 0) aside.to = span.to;
      } else aside = { text, ...span };
      scheduleAside();
    };

    /** Speech over the agent ended and its words are in (or late): decide what it was. */
    const classify = (c: Candidate, why: string) => {
      if (candidate !== c) return;
      if (c.text && opensFloor(c.text)) return yieldFloor(c, 'floor-taking opener');
      c.timer?.clear();
      c.wait?.clear();
      candidate = null;
      release();
      const kind = !words(c.text).length ? 'noise' : isBackchannel(c.text) || !p.asides ? 'backchannel' : 'aside';
      decide(kind, { ms: (c.ended ?? ctx.now()) - c.since, text: c.text, why });
      go('backchannel', kind === 'aside' ? 'aside: held until the end of the sentence' : why);
      if (kind === 'aside') holdAside(c);
    };

    /** Rules without the model: enough speech that isn't only backchannel words. */
    const consider = (c: Candidate) => {
      if (classifies()) {
        if (c.text && opensFloor(c.text)) yieldFloor(c, 'floor-taking opener');
        return;
      }
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
        if (classifies()) {
          // Speech still going this long takes the floor, whatever it says.
          if (c.ended !== undefined) return;
          if (ctx.now() - c.since >= p.maxBackchannelMs) yieldFloor(c, 'sustained speech');
          else c.timer = ctx.setTimer(p.maxBackchannelMs - (ctx.now() - c.since), check, 'barge-in');
          return;
        }
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
      if (classifies()) hold();
      c.check = check;
      c.timer = ctx.setTimer(classifies() || modelDecides() ? p.maxBackchannelMs : p.minSpeechMs, check, 'barge-in');
      return c;
    };

    ctx.setState('floor', holder(state));
    ctx.setState('state', state);

    return {
      onInput(port, msg) {
        if (port === 'scores') {
          if (msg.type === 'turnmodel.frame') {
            eotUser = (msg.eot as number[])[0];
            if (useModel() && eotUser >= p.draftAt) maybeDraft(`model: end of turn likely p=${eotUser}`);
            return;
          }
          if (msg.type === 'turnmodel.status') {
            model = msg.status === 'ready';
            if (p.useModel) ctx.setState('using', model ? 'turn model' : 'rules');
            return;
          }
          if (!useModel() || msg.who !== 'user') return;
          const why = `model: ${msg.type === 'turnmodel.eot' ? 'end of turn' : 'take the floor'} p=${msg.p}`;
          if (msg.type === 'turnmodel.int' && p.bargeIn && state !== 'listening') {
            if (p.modelNeedsVad && !speaking) return void ctx.log('debug', 'model floor take ignored: the VAD hears no speech', { why });
            const c = overlap(msg.at as number, why);
            if (c) yieldFloor(c, why);
          } else if (msg.type === 'turnmodel.eot' && state === 'listening') {
            // Not while the VAD still hears them; if their speech ends right after, it stands.
            if (speaking) eotHeld = { why, at: ctx.now() };
            else endOfTurn(why);
          }
          return;
        }
        if (port === 'playback') {
          if (msg.type === 'playback.start') {
            go('playback_start', 'playback started');
            // They started talking while it was thinking: don't talk over them while their words are judged.
            if (candidate && classifies()) hold();
          } else if (msg.type === 'playback.segment') {
            if (msg.rid !== segRid) {
              segRid = (msg.rid as string) ?? null;
              segStarts = [];
            }
            segStarts.push(msg.at as number);
            scheduleAside();
          } else if (msg.type === 'playback.paused') {
            // The sentences still to come are announced again, with their new times, on resume.
            segStarts = segStarts.filter((t) => t < (msg.at as number));
            if (aside?.timer) {
              aside.timer.clear();
              aside.timer = undefined;
            }
          } else if (msg.type === 'playback.end') {
            const c = candidate;
            if (!go('playback_end', 'playback ended')) return;
            held = false;
            if (c) {
              // The agent finished while the user was already talking: just listen (a bare backchannel is dropped).
              c.timer?.clear();
              c.wait?.clear();
              candidate = null;
              if (!(p.overlap === 'classify' && c.text && isBackchannel(c.text))) for (const f of c.finals) collect(f.text, f.start, f.end);
            }
            if (aside) {
              takeAside();
              if (!c) commit('aside, after the agent finished', undefined, { aside: true });
            }
          } else if (msg.type === 'playback.interrupted') {
            // What the user heard before we stopped, for the LLM's history.
            if (last) ctx.emit('turn', { type: 'turn.heard', turnId: last.id, text: (msg.heard as string) ?? '' });
            if (aside?.cutting) finishAside('aside at sentence end');
          }
          return;
        }
        if (port === 'vad') {
          if (msg.type === 'vad.speech_start') {
            ctx.setState('userSpeaking', true);
            speaking = true;
            caughtUp = false;
            eotHeld = null;
            silence?.clear();
            silence = null;
            if (state === 'thinking' || state === 'speaking') overlap(msg.at as number, 'user speech (VAD)');
            else if (candidate?.ended !== undefined) {
              // They went on before the words of the last bit came in: one stretch of speech.
              const c = candidate;
              c.ended = undefined;
              c.wait?.clear();
              c.timer?.clear();
              if (c.check) c.timer = ctx.setTimer(Math.max(1, p.maxBackchannelMs - (ctx.now() - c.since)), c.check, 'barge-in');
            } else if (eotWait) {
              // They went on after the model called the end of their turn: wait for the next one.
              eotWait.timer.clear();
              eotWait = null;
            }
          } else if (msg.type === 'vad.speech_end') {
            ctx.setState('userSpeaking', false);
            speaking = false;
            // A partial from before the silence usually lacks the last words: wait for ASR to catch up.
            caughtUp = !partial;
            if (state === 'listening') maybeDraft('silence, words final');
            armSilence();
            if (eotHeld && state === 'listening' && ctx.now() - eotHeld.at <= EOT_HOLD_MS) endOfTurn(`${eotHeld.why}, then VAD speech end`);
            eotHeld = null;
            if (candidate && classifies()) {
              // Decide once its words are in.
              const c = candidate;
              c.ended = ctx.now();
              c.endAt = msg.at as number;
              c.wait?.clear();
              c.wait = ctx.setTimer(p.classifyWaitMs, () => classify(c, 'no ASR final in time'), 'classify');
            } else if (candidate) {
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
          const c = candidate;
          // Classify judges all the words so far; the legacy rules, a partial on its own.
          if (text) c.text = p.overlap === 'legacy' && msg.type === 'asr.partial' ? text : [...c.finals.map((f) => f.text), text].join(' ');
          if (msg.type === 'asr.final' && text) c.finals.push({ text, start: msg.start as number, end: msg.end as number });
          consider(c);
          // Decide on the final that reaches the end of the speech (earlier finals, often empty, cover audio before it).
          if (msg.type === 'asr.final' && c.ended !== undefined && (msg.end as number) >= (c.endAt ?? 0)) classify(c, 'ASR final');
          return;
        }
        if (candidate?.ended !== undefined && msg.type === 'asr.utterance_end') return classify(candidate, 'ASR utterance end');
        if (state !== 'listening') {
          if (msg.type === 'asr.final' && text) ctx.log('info', 'ignored short speech while the agent has the floor', { text });
          return;
        }
        if (!speaking) caughtUp = true;
        if (msg.type === 'asr.partial') {
          partial = text ? { text, start: msg.start as number, end: msg.end as number } : null;
          if (text) refreshDraft('ASR partial');
        } else if (msg.type === 'asr.final') {
          collect(text, msg.start as number, msg.end as number);
          partial = null;
          if (eotWait) commit(eotWait.why);
          // A final flushed because the VAD heard the speech end is as good as ASR endpointing (and sooner).
          // But not if the VAD hears them again by the time it lands: they only paused and went on (our
          // Finalize marks even a short pause speech_final), and the next speech end commits it all.
          else if (!useModel() && p.commitOn === 'speech_final' && (msg.fromFinalize ? !speaking : msg.speechFinal))
            commit(msg.speechFinal ? 'speech_final' : 'finalized on VAD speech end');
          else {
            if (!useModel() && msg.fromFinalize && speaking) heldFinal = true;
            refreshDraft('ASR final');
          }
        } else if (msg.type === 'asr.utterance_end') {
          if (speaking) ctx.log('debug', 'utterance end ignored: the VAD still hears speech');
          else commit('utterance_end');
        }
      },
    };
  },
};
