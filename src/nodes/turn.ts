import type { NodeDef, TimerHandle } from '../core/node';
import type { Json, Msg } from '../core/types';
import { seededRandom, takes } from './backchannel';
import { askChoice, choiceRequest, parseChoiceOptions, readChoice } from './typesafe';

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
      if (f !== floor) ctx.emit('turn', { type: 'turn.transition', from: floor, to: f, why });
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
            // Words are waiting to be committed: the LLM holds its workers' news until they are.
            if (parts.length === 1) ctx.emit('turn', { type: 'turn.pending', words: true });
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
  holdAfterMs: number;
  yieldWithinMs: number;
  duckDb: number;
  floorWords: string;
  asides: boolean;
  classifyWaitMs: number;
  eotCheck: string;
  eotYield: number;
  eotHoldMs: number;
  eotOptions: string;
  eotInstructions: string;
  backoffAfter: number;
  backoffLines: string;
  backoffWords: number;
  backoffSilenceMs: number;
}

export const DEFAULT_EOT_OPTIONS = `# The first option means the user is done; "-" means they are probably going on.
yield = The speaker has finished and is handing the turn over: they asked the listener a question, invited a reply ("your turn", "what about you"), or completed a point and are clearly waiting for a response.
- = The speaker is likely to keep talking: mid-sentence or mid-phrase, partway through a list, story or explanation, a short reaction or acknowledgement (yeah, exactly, for real, I hear that, got you) that usually leads into more, or a statement that sets up something further.`;

export const DEFAULT_EOT_INSTRUCTIONS =
  'The text is a speech-recognition transcript of what a user has said so far in their turn, in a live spoken conversation with a voice agent; they have just paused. Judge whether the pause is the end of their turn or just a breath before they go on. When unsure, pick "-".';

/** Who holds the floor, and whether the user is talking over the agent while it does. */
export type FloorState = 'listening' | 'thinking' | 'speaking' | 'thinking+user' | 'speaking+user';
export type FloorEvent = 'commit' | 'playback_start' | 'playback_end' | 'user_start' | 'backchannel' | 'take_floor' | 'aside';

/**
 * The floor as an explicit state machine. These are the only transitions; an
 * event a state has no entry for is ignored (and logged). Every transition is
 * emitted as a `turn.transition` message with what triggered it.
 */
export const FLOOR: Record<FloorState, Partial<Record<FloorEvent, FloorState>>> = {
  // playback_start: the agent speaks unprompted (a background task's result).
  listening: { commit: 'thinking', playback_start: 'speaking' },
  thinking: { playback_start: 'speaking', user_start: 'thinking+user' },
  // aside: the agent stopped to answer what the user said over it.
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
  /** Cuts the agent off (or ducks it) once the speech has gone on for `holdAfterMs`. */
  holdTimer?: TimerHandle;
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
 * Speech over the agent, `overlap: classify` (the default): once the VAD has
 * heard `holdAfterMs` of it the agent is cut off (`onset: cut`, a pause the
 * speaker can resume from), so an interruption lands, while the words come in
 * (`duck` plays it `duckDb` quieter instead; `none` lets it talk on). Shorter
 * speech never stops it unless its words do. The reply is cancelled on a
 * floor-taking opener ("wait", "so", "actually"…), on speech longer than
 * `maxBackchannelMs`, or on the model's take-the-floor. When the speech ends,
 * its words decide: nothing (noise) or only backchannel words are dropped and
 * the agent goes on from where it stopped; anything else is an aside (a
 * confirmation, a quick addition): the agent stops at once and the aside is
 * the user's turn, answered right away. The reply stays in the history only up
 * to where the user started the aside, so the aside follows the words it was
 * said over, not the ones the agent said after. `overlap: legacy` is the earlier rule set:
 * `minSpeechMs` of speech that isn't only backchannel words stops the agent.
 *
 * Backing off: after `backoffAfter` interruptions in a row (replies cut off
 * by the user, with none played to the end in between) the agent stops
 * competing for the floor. It says a canned line ("You go first.") once the
 * user pauses, and commits nothing until they say something coherent (at least
 * `backoffWords` words that aren't only backchannel words) or `backoffSilenceMs`
 * of silence passes; then it takes what they said as their turn and carries on.
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
 *
 * Early interjection: the user's uncommitted words are emitted as `turn.words`
 * whenever they change while listening. An `interject.take` commits them at
 * once (its turn names the check in `interject`, and a draft out at the same
 * pause is dropped), unless the agent has the floor or it is backed off and
 * the words aren't coherent; `turn.interject` says which.
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
    { name: 'interject', type: 'interject', description: 'interject.take: commit the words so far now, for an early interjection' },
  ],
  outputs: [
    { name: 'turn', type: 'turn' },
    { name: 'control', type: 'control', description: 'cancel: stop the reply in flight (LLM, TTS, playback)' },
    { name: 'text', type: 'text', description: 'Backing off: the "you go first" lines, rendered ahead (bc.lines) and played as a cue (bc.say) by TTS' },
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
      default: 'yeah, yes, yep, uh-huh, uh huh, mhm, mm, hmm, ok, okay, right, alright, all right, sure, i see, i hear you, i hear that, got it, cool, nice, wow',
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
        'classify: cut the agent off, then decide from the words: floor take (drop the reply), aside (stop and answer it at once) or backchannel/noise (go on). legacy: stop after min speech unless it is only backchannel words.',
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
      key: 'yieldWithinMs',
      label: 'Yield if it just started (ms)',
      type: 'number',
      default: 1000,
      description:
        'User speech within this long of the agent starting to talk stops it at once and gives the user the floor: both started into a brief silence, and talking on turns into a fight over the floor. 0: off.',
      when: { bargeIn: [true] },
    },
    {
      key: 'holdAfterMs',
      label: 'Cut after (ms)',
      type: 'number',
      default: 1000,
      description:
        'How long the VAD must hear the user over the agent before it is cut off (or ducked). Shorter speech, a cough or a backchannel, leaves it talking; a floor-taking opener still stops it at once. 0: at the first sound.',
      when: { bargeIn: [true], overlap: ['classify'], onset: ['cut', 'duck'] },
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
        'Short speech over the agent with real words (not backchannel words) is an aside: the agent stops and answers it at once. Off: it is ignored like a backchannel.',
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
    {
      key: 'eotCheck',
      label: 'Done or pausing?',
      type: 'select',
      options: ['typesafe', 'off'],
      default: 'typesafe',
      description:
        'Without the turn model, ASR endpointing ends the turn at the first pause. typesafe: TypeSafe Choice reads the words when the user pauses; unless it thinks they are handing the turn over, the commit waits for more silence (a draft still starts, so the reply is ready).',
    },
    {
      key: 'eotYield',
      label: 'Commit at once from p(done)',
      type: 'number',
      default: 0.5,
      description: "TypeSafe's probability that the user is done at or above which the turn is committed at the pause; below it, after the hold.",
      when: { eotCheck: ['typesafe'] },
    },
    { key: 'eotHoldMs', label: 'Hold (ms)', type: 'number', default: 1500, description: 'More silence needed before committing words TypeSafe thinks are not done.', when: { eotCheck: ['typesafe'] } },
    { key: 'eotOptions', label: 'Done/pausing options', type: 'text', default: DEFAULT_EOT_OPTIONS, description: 'Sent to TypeSafe as the Choice criteria; the first option means done.', when: { eotCheck: ['typesafe'] } },
    { key: 'eotInstructions', label: 'Done/pausing question', type: 'text', default: DEFAULT_EOT_INSTRUCTIONS, when: { eotCheck: ['typesafe'] } },
    {
      key: 'backoffAfter',
      label: 'Back off after interruptions',
      type: 'number',
      default: 2,
      description: 'After this many replies in a row cut off by the user, the agent says a "you go first" line and waits for them. 0: never.',
      when: { bargeIn: [true] },
    },
    {
      key: 'backoffLines',
      label: 'Back-off lines',
      type: 'string',
      default: 'You go first. | Sorry, go ahead. | Go ahead.',
      description: 'What the agent says when it backs off, separated by |. Needs a TTS that plays canned lines (Deepgram stream).',
      when: { bargeIn: [true] },
    },
    {
      key: 'backoffWords',
      label: 'Coherent speech (words)',
      type: 'number',
      default: 4,
      description: 'While backed off, the user\'s turn is taken once it has this many words that aren\'t only backchannel words.',
      when: { bargeIn: [true] },
    },
    {
      key: 'backoffSilenceMs',
      label: 'Back off until silence (ms)',
      type: 'number',
      default: 4000,
      description: 'While backed off, this much silence ends it: whatever the user said is taken as their turn.',
      when: { bargeIn: [true] },
    },
  ],
  channels: [{ name: 'typesafe', description: "TypeSafe Choice on the user's words at a pause: done, or going on?" }],
  requires: (p) => (p.eotCheck === 'typesafe' ? ['typesafe'] : []),
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
    /** TypeSafe's p(done) for the words at a pause, by their key: undefined while it answers, null if it failed. */
    const eotAnswers = new Map<string, number | null | undefined>();
    const eotOpts = p.eotCheck === 'typesafe' ? parseChoiceOptions(String(p.eotOptions)) : [];
    /** The rules ended the turn but TypeSafe thinks the user is going on: commit after more silence. */
    let eotHold: { why: string; key: string; timer: TimerHandle } | null = null;
    /** The turn model is loaded and scoring. */
    let model = false;
    /** With the model: commits the words if the user stays silent this long without its end of turn. */
    let silence: TimerHandle | null = null;
    /** A Finalize'd final was held back because the user went on: their next silence commits it, ASR final or not. */
    let heldFinal = false;
    /** Replies cut off by the user since one last played to its end. */
    let interrupts = 0;
    /** Backed off after repeated interruptions: waiting for the user to say something coherent, or for silence. */
    let backoff: { said: boolean; timer?: TimerHandle } | null = null;
    let backoffNo = 0;
    const yieldLines = String(p.backoffLines)
      .split('|')
      .map((l) => l.trim())
      .filter(Boolean);
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
    /** When the agent's current reply started playing. */
    let playingSince: number | null = null;
    /** Words said over the agent that it should take in, held until its current sentence ends. */
    let aside: { text: string; from: number; to: number; timer?: TimerHandle; cutting?: boolean } | null = null;
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
    /** Hold once the user's speech `c` has gone on for `holdAfterMs`. */
    const holdWhenSustained = (c: Candidate) => {
      c.holdTimer?.clear();
      c.holdTimer = undefined;
      if (held || !classifies() || c.ended !== undefined) return;
      const left = p.holdAfterMs - (ctx.now() - c.since);
      if (left <= 0) return hold();
      c.holdTimer = ctx.setTimer(
        left,
        () => {
          c.holdTimer = undefined;
          if (candidate === c && c.ended === undefined && classifies()) hold();
        },
        'hold',
      );
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
    /** Enough words, and not only backchannel words, to take as the user's turn while backed off. */
    const coherent = (text: string) => words(text).length >= p.backoffWords && !isBackchannel(text);
    const pendingText = () => [...parts, partial?.text ?? ''].join(' ').trim();
    const sayYield = () => {
      if (!backoff || backoff.said || !yieldLines.length) return;
      backoff.said = true;
      const id = `${ctx.id}-yield${backoffNo}`;
      const line = yieldLines[backoffNo % yieldLines.length];
      const all = takes(line, 3);
      ctx.emit('text', { type: 'bc.say', id, line, take: all[Math.floor(seededRandom(id)() * all.length)], tailMs: 150, why: 'backing off' });
    };
    const enterBackoff = () => {
      backoff = { said: false };
      interrupts = 0;
      decide('backoff', { why: `${p.backoffAfter} interruptions in a row` });
      ctx.setState('backoff', true);
      if (!speaking) {
        sayYield();
        armBackoffSilence();
      }
    };
    const recover = (why: string) => {
      if (!backoff) return;
      backoff.timer?.clear();
      backoff = null;
      backoffNo++;
      decide('recover', { why });
      ctx.setState('backoff', false);
    };
    /** While backed off, enough silence ends it: take what they said, if anything. */
    const armBackoffSilence = () => {
      if (!backoff) return;
      backoff.timer?.clear();
      backoff.timer = ctx.setTimer(
        p.backoffSilenceMs,
        () => {
          recover(`${p.backoffSilenceMs} ms of silence`);
          if (state === 'listening' && (parts.length || partial)) endOfTurn('silence after backing off');
        },
        'backoff',
      );
    };

    /** Commit the turn. `tail`: its last words are an ASR partial whose final may still come. */
    const commit = (why: string, tail?: Partial_, extra: Record<string, Json> = {}) => {
      heldFinal = false;
      eotWait?.timer.clear();
      eotWait = null;
      eotHold?.timer.clear();
      eotHold = null;
      silence?.clear();
      silence = null;
      const text = parts.join(' ').trim();
      if (!text || state !== 'listening') return;
      if (backoff) {
        if (!coherent(text)) return void ctx.log('debug', 'backed off: waiting for coherent speech or silence', { text, why });
        recover('coherent speech');
      }
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
      eotAnswers.clear();
      go('commit', why);
      // Commits from timers too: the words are gone.
      syncWords();
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

    const eotOn = () => !useModel() && eotOpts.length > 0;
    /** Ask TypeSafe whether these words end the user's turn (once per wording). */
    const askEot = (text: string) => {
      const key = draftKey(text);
      if (!key || eotAnswers.has(key) || !eotOn()) return;
      eotAnswers.set(key, undefined);
      const t0 = ctx.now();
      const done = eotOpts[0].key;
      askChoice(ctx, choiceRequest(text, eotOpts, String(p.eotInstructions), 'jev-latest'), (res) => {
        const got = res === null ? null : readChoice(res, eotOpts, 0);
        const pDone = got ? Math.round((got.probabilities[done] ?? (got.choice === done ? got.confidence : 1 - got.confidence)) * 1000) / 1000 : null;
        eotAnswers.set(key, pDone);
        const hold = pDone !== null && pDone < p.eotYield;
        ctx.emit('turn', { type: 'turn.eot_check', text, ms: Math.round(ctx.now() - t0), ...(pDone === null ? { failed: true } : { p: pDone, hold }) });
        if (eotHold?.key === key && !hold) settleEot();
      });
    };
    /** Commit a held turn now, if TypeSafe has said the user is done (or failed). */
    const settleEot = () => {
      const h = eotHold;
      if (!h) return;
      const a = eotAnswers.get(h.key);
      if (a === undefined || (a !== null && a < p.eotYield)) return;
      commit(`${h.why}; TypeSafe ${a === null ? 'failed' : `p(done)=${a}`}`);
    };
    /**
     * The rules (an ASR endpoint) end the turn. Without the turn model these fire
     * at the first short pause, so TypeSafe reads the words: unless it thinks the
     * user is done, the commit waits for `eotHoldMs` more silence. A draft still
     * goes out in the pause, so the reply is ready if it does commit.
     */
    const endpoint = (why: string) => {
      const text = parts.join(' ').trim();
      if (!eotOn() || !text || state !== 'listening' || backoff) return commit(why);
      const key = draftKey(text);
      // Already holding these words (an utterance end after the final): keep counting from the first.
      if (eotHold?.key === key) return;
      askEot(text);
      eotHold?.timer.clear();
      const h: { why: string; key: string; timer: TimerHandle } = {
        why,
        key,
        timer: ctx.setTimer(
          p.eotHoldMs,
          () => {
            if (eotHold !== h) return;
            const a = eotAnswers.get(key);
            commit(`${why}; TypeSafe ${a === undefined ? 'did not answer' : `p(done)=${a}`}, then ${p.eotHoldMs} ms more silence`);
          },
          'eot-hold',
        ),
      };
      eotHold = h;
      settleEot();
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

    /** A quick yield counts toward backing off only if the user heard some of the reply (told by playback.interrupted). */
    let countIfHeard = false;
    /** The user takes the floor: stop the reply and start listening. */
    const yieldFloor = (c: Candidate, why: string, quick = false) => {
      c.timer?.clear();
      c.wait?.clear();
      candidate = null;
      // Cancelling drops the held audio; the speaker restores its volume for the next reply.
      held = false;
      const wasThinking = state === 'thinking+user';
      decide(wasThinking ? 'continue' : 'interrupt', { why, ms: ctx.now() - c.since, text: c.text });
      if (!wasThinking && quick) countIfHeard = p.backoffAfter > 0;
      else if (!wasThinking && p.backoffAfter > 0 && ++interrupts >= p.backoffAfter) enterBackoff();
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
    /** The agent stopped (or finished): the aside is the user's turn now. */
    const finishAside = (why: string) => {
      if (!aside) return;
      if (state === 'speaking') go('aside', why);
      if (state !== 'listening') return;
      takeAside();
      commit(why, undefined, { aside: true });
    };
    /**
     * Speech `c` over the agent is an aside: stop the reply now and answer it. The reply is
     * kept as heard up to where the aside started (`heardAt`), so the history has the aside
     * where it was said. The speaker answers with playback.interrupted; if it had nothing
     * playing, go on without it.
     */
    const cutForAside = (c: Candidate) => {
      const fin = c.finals.length ? c.finals : null;
      const text = fin ? fin.map((f) => f.text).join(' ') : c.text;
      const span = fin ? { from: fin[0].start, to: fin[fin.length - 1].end } : { from: -1, to: -1 };
      if (aside) {
        aside.text = `${aside.text} ${text}`;
        if (aside.from < 0) aside.from = span.from;
        if (span.to >= 0) aside.to = span.to;
      } else aside = { text, ...span };
      if (aside.cutting) return;
      const a = aside;
      a.cutting = true;
      // Cancelling drops the held audio; the speaker restores its volume for the next reply.
      held = false;
      ctx.emit('control', { type: 'cancel', heardAt: c.since });
      a.timer = ctx.setTimer(300, () => finishAside('aside'), 'aside');
    };

    /** Speech over the agent ended and its words are in (or late): decide what it was. */
    const classify = (c: Candidate, why: string) => {
      if (candidate !== c) return;
      if (c.text && opensFloor(c.text)) return yieldFloor(c, 'floor-taking opener');
      c.timer?.clear();
      c.wait?.clear();
      candidate = null;
      const kind = !words(c.text).length ? 'noise' : isBackchannel(c.text) || !p.asides ? 'backchannel' : 'aside';
      if (kind !== 'aside') release();
      decide(kind, { ms: (c.ended ?? ctx.now()) - c.since, text: c.text, why });
      go('backchannel', kind === 'aside' ? 'aside: answered at once' : why);
      if (kind === 'aside') cutForAside(c);
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
      holdWhenSustained(c);
      c.check = check;
      c.timer = ctx.setTimer(classifies() || modelDecides() ? p.maxBackchannelMs : p.minSpeechMs, check, 'barge-in');
      return c;
    };

    ctx.setState('floor', holder(state));
    ctx.setState('state', state);

    /** Whether the user has words not yet committed, told to the LLM when it changes (so news waits for them). */
    let hadWords = false;
    const syncPending = () => {
      const has = parts.length > 0 || !!partial?.text;
      if (has === hadWords) return;
      hadWords = has;
      ctx.emit('turn', { type: 'turn.pending', words: has });
    };

    /** The uncommitted words, told whenever they change while listening (for early interjections). */
    let lastWords = '';
    const syncWords = () => {
      const text = state === 'listening' ? [...parts, partial?.text ?? ''].join(' ').trim() : '';
      if (text === lastWords) return;
      lastWords = text;
      ctx.emit('turn', { type: 'turn.words', text, ...(text && replaces ? { replaces } : {}) });
    };

    const handle = (port: string, msg: Msg) => {
      if (port === 'interject') {
        if (msg.type !== 'interject.take') return;
        const why =
          state !== 'listening'
            ? 'the agent has the floor'
            : !pendingText()
              ? 'no words to commit'
              : backoff && !coherent(pendingText())
                ? 'backed off: waiting for coherent speech'
                : null;
        ctx.emit('turn', { type: 'turn.interject', checkId: msg.checkId as string, taken: !why, ...(why ? { why } : {}) });
        if (why) return;
        eotHeld = null;
        // A draft goes out at the same pause: the interjection wins, the LLM drops the draft.
        dropDraft('early interjection');
        // A partial's final may still come: it stays the turn's tail, as at the model's end of turn.
        const tail = partial ?? undefined;
        if (partial) collect(partial.text, partial.start, partial.end);
        partial = null;
        commit(`early interjection: ${String(msg.why ?? 'the LLM cuts in')}`, tail, { interject: msg.checkId as string });
        return;
      }
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
          playingSince = ctx.now();
          go('playback_start', 'playback started');
          // They started talking while it was thinking: don't talk over them while their words are judged.
          if (candidate) holdWhenSustained(candidate);
        } else if (msg.type === 'playback.end') {
          const c = candidate;
          if (!go('playback_end', 'playback ended')) return;
          held = false;
          interrupts = 0;
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
          // Stopped before a word of it was heard: not a fight over the floor.
          if (countIfHeard) {
            countIfHeard = false;
            if (String(msg.heard ?? '').trim() && ++interrupts >= p.backoffAfter) enterBackoff();
          }
          // What the user heard before we stopped, for the LLM's history.
          if (last) ctx.emit('turn', { type: 'turn.heard', turnId: last.id, ...(msg.rid ? { rid: msg.rid as string } : {}), text: (msg.heard as string) ?? '' });
          if (aside?.cutting) finishAside('aside');
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
          backoff?.timer?.clear();
          if (state === 'speaking' && p.yieldWithinMs > 0 && playingSince !== null && ctx.now() - playingSince < p.yieldWithinMs) {
            // Both started into the same silence: the agent backs off at once rather than fight over the floor.
            const c = overlap(msg.at as number, 'user speech (VAD)');
            if (c && candidate === c) yieldFloor(c, `user spoke ${Math.round(ctx.now() - playingSince)} ms after the agent started`, true);
          } else if (state === 'thinking' || state === 'speaking') overlap(msg.at as number, 'user speech (VAD)');
          else if (candidate?.ended !== undefined) {
            // They went on before the words of the last bit came in: one stretch of speech.
            const c = candidate;
            c.ended = undefined;
            c.wait?.clear();
            c.timer?.clear();
            holdWhenSustained(c);
            if (c.check) c.timer = ctx.setTimer(Math.max(1, p.maxBackchannelMs - (ctx.now() - c.since)), c.check, 'barge-in');
          } else if (eotHold) {
            // TypeSafe was right: they went on. Their next pause ends it (or the silence timeout, whatever ASR does).
            eotHold.timer.clear();
            eotHold = null;
            heldFinal = true;
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
          if (state === 'listening') {
            maybeDraft('silence, words final');
            // Ask ahead: the answer is usually in by the time ASR ends the turn.
            if (!backoff) askEot(pendingText());
          }
          armSilence();
          if (eotHeld && state === 'listening' && ctx.now() - eotHeld.at <= EOT_HOLD_MS) endOfTurn(`${eotHeld.why}, then VAD speech end`);
          eotHeld = null;
          if (backoff) {
            // Back off out loud at their first pause, unless what they said already makes a turn.
            if (!coherent(pendingText())) sayYield();
            armBackoffSilence();
          }
          if (candidate && classifies()) {
            // Decide once its words are in.
            const c = candidate;
            c.ended = ctx.now();
            c.endAt = msg.at as number;
            c.holdTimer?.clear();
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
          endpoint(msg.speechFinal ? 'speech_final' : 'finalized on VAD speech end');
        else {
          if (!useModel() && msg.fromFinalize && speaking) heldFinal = true;
          refreshDraft('ASR final');
        }
      } else if (msg.type === 'asr.utterance_end') {
        if (speaking) ctx.log('debug', 'utterance end ignored: the VAD still hears speech');
        else endpoint('utterance_end');
      }
    };

    return {
      start() {
        // Render the back-off lines ahead, so one plays at once when needed.
        if (p.bargeIn && p.backoffAfter > 0 && yieldLines.length) ctx.emit('text', { type: 'bc.lines', lines: yieldLines.flatMap((l) => takes(l, 3)) });
      },
      onInput(port, msg) {
        handle(port, msg);
        syncPending();
        syncWords();
      },
    };
  },
};
