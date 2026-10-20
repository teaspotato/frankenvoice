// Early interjection: the agent cuts in before the user has finished their
// turn, when cutting in helps (they got something wrong, or it's clear what
// they're after). Asking the conversation LLM on every ASR partial would spam
// it, so TypeSafe Choice gates first: only when the user's words so far end on
// a complete sentence or claim does the LLM get an "interrupt?" check. The
// check is a reply held back like a draft; if the LLM wants to cut in, the
// user's next pause commits their words so far as the turn and the held reply
// plays.

import type { NodeDef, TimerHandle } from '../core/node';
import type { JsonObject } from '../core/types';
import { askChoice, choiceRequest, parseChoiceOptions, readChoice } from './typesafe';
import { draftKey } from './turn';

export const DEFAULT_GATE_OPTIONS = `# One option per line: name = when it fits. Anything but "-" sends the words to the LLM for an "interrupt?" check.
complete = The transcript ends right after a complete sentence, claim, statement of fact, name, number, plan or question that a listener could already react to (correct it, answer it, or object to it).
- = The transcript ends mid-phrase or mid-sentence, or the only thing just completed is filler, a greeting, thanks or an acknowledgement.`;

export const DEFAULT_GATE_INSTRUCTIONS =
  'The text is a live speech-recognition transcript of what a user has said so far to a voice agent. They may still be talking, and it may have recognition errors. Judge only where the transcript ends right now: has the speaker just completed something a listener could react to?';

export const PASS = '[listen]';

export const DEFAULT_CHECK_PROMPT = `[The user is still talking: the message above is a live speech-recognition transcript of what they have said so far, cut off mid-turn. Don't answer it yet unless cutting in right now clearly helps, the way an attentive person would: they just said something wrong (a factual error, a slip, a wrong name or number, or something that contradicts what was said before), they misheard or misremember what you said, they are about to make a mistake, or it is already obvious what they are asking and a quick answer saves them the rest. If so, reply with only what you'd say to cut in: one short spoken sentence. Otherwise reply with exactly ${PASS}]`;

const ports = {
  inputs: [
    { name: 'turn', type: 'turn', description: "The user's uncommitted words (turn.words), drafts and committed turns" },
    { name: 'vad', type: 'vad', description: 'Pauses: an interjection waits for the next one' },
    { name: 'text', type: 'text', description: "The LLM's verdict on a check (llm.check_verdict)" },
  ],
  outputs: [
    { name: 'check', type: 'interject', description: 'interject.check / interject.drop for the LLM; interject.gate, the TypeSafe answers' },
    { name: 'take', type: 'interject', description: 'interject.take: turn-taking commits the words so far, and the held interjection plays' },
  ],
};

const count = (key: string) => (key ? key.split(' ').length : 0);

/**
 * TypeSafe gates the user's words so far (turn.words) while they talk; when
 * they end on something complete, the LLM is asked whether to cut in
 * (interject.check, with `prompt` added to the words). If it wants to, the next
 * pause in the user's speech (VAD speech end) takes the floor
 * (interject.take); if they go on for `maxWaitMs` or `maxExtraWords` more words
 * without one, the interjection is dropped as stale.
 */
export const interjectNode: NodeDef = {
  type: 'interject',
  title: 'Early interjection',
  category: 'control',
  description: "TypeSafe Choice watches the user's words as they talk; when they complete a claim or sentence, the LLM is asked whether to cut in (a correction, a second guess). Off by default.",
  ...ports,
  params: [
    { key: 'options', label: 'Gate options', type: 'text', default: DEFAULT_GATE_OPTIONS, description: 'name = when it fits ("-" = not yet). Sent to TypeSafe as the Choice criteria.' },
    { key: 'instructions', label: 'Gate question', type: 'text', default: DEFAULT_GATE_INSTRUCTIONS, description: "The Choice instructions. The state is only the user's ASR words so far." },
    { key: 'minConfidence', label: 'Gate min confidence', type: 'number', default: 0.6, description: 'Ask the LLM only when TypeSafe is at least this sure.' },
    { key: 'minWords', label: 'Min words', type: 'number', default: 4, description: 'Words the user must have said before the gate is asked.' },
    {
      key: 'prompt',
      label: 'Check prompt',
      type: 'text',
      default: DEFAULT_CHECK_PROMPT,
      description: "Added after the user's words in the LLM check. History, system prompt and tools stay as in a reply, so the prompt cache holds.",
    },
    { key: 'pass', label: 'Pass reply', type: 'string', default: PASS, description: 'What the LLM replies when it lets the user go on (the prompt must ask for it).' },
    { key: 'maxWaitMs', label: 'Max wait for a pause (ms)', type: 'number', default: 2500, description: 'Drop the interjection if the user does not pause this long after the LLM wanted to cut in.' },
    { key: 'maxExtraWords', label: 'Max extra words', type: 'number', default: 8, description: 'Drop the interjection if the user says this many more words than the LLM saw before pausing.' },
    { key: 'model', label: 'TypeSafe model', type: 'string', default: 'jev-latest' },
  ],
  channels: [{ name: 'typesafe', description: "TypeSafe Choice over the user's words so far: one JSON request, one JSON answer" }],
  requires: () => ['typesafe'],
  create(ctx) {
    const p = ctx.params as JsonObject;
    const opts = parseChoiceOptions(String(p.options ?? ''));
    let words = '';
    let replaces: string | null = null;
    /** Key of the words last sent to TypeSafe, and whether it is still answering. */
    let asked = '';
    let asking = false;
    /** Key of the words last checked with the LLM: the same words are not checked twice. */
    let checked = '';
    let draftOut = false;
    let speaking = false;
    let n = 0;
    /** Counts turns, so a TypeSafe answer about an earlier turn's words is ignored. */
    let gen = 0;
    /** The check with the LLM in flight, or the interjection waiting for a pause. */
    let check: { id: string; key: string; interject?: string; taken?: boolean; wait?: TimerHandle } | null = null;

    const drop = (why: string) => {
      const c = check;
      if (!c) return;
      check = null;
      c.wait?.clear();
      ctx.emit('check', { type: 'interject.drop', checkId: c.id, why });
      ctx.setState('check', '');
    };
    const take = (why: string) => {
      const c = check;
      if (!c || c.taken) return;
      c.taken = true;
      c.wait?.clear();
      ctx.emit('take', { type: 'interject.take', checkId: c.id, text: c.interject ?? '', why });
    };
    const startCheck = () => {
      const key = draftKey(words);
      if (!key || key === checked || draftOut || check?.interject) return;
      drop('words went on');
      checked = key;
      check = { id: `${ctx.id}-c${n++}`, key };
      ctx.emit('check', {
        type: 'interject.check',
        checkId: check.id,
        text: words,
        prompt: String(p.prompt ?? ''),
        pass: String(p.pass ?? PASS),
        ...(replaces ? { replaces } : {}),
      });
      ctx.setState('check', words);
    };
    /** Ask TypeSafe about the words so far, if they changed and it isn't busy. */
    const gate = () => {
      if (asking || draftOut || check?.interject || !opts.length) return;
      const key = draftKey(words);
      if (count(key) < Number(p.minWords) || key === asked || key === checked) return;
      asked = key;
      asking = true;
      const text = words;
      const t0 = ctx.now();
      const g = gen;
      askChoice(ctx, choiceRequest(text, opts, String(p.instructions ?? ''), String(p.model ?? 'jev-latest')), (res) => {
        asking = false;
        const got = res === null ? null : readChoice(res, opts, Number(p.minConfidence));
        const go = !!got?.option;
        ctx.emit('check', {
          type: 'interject.gate',
          text,
          ms: Math.round(ctx.now() - t0),
          ...(got ? { choice: got.choice, confidence: got.confidence, probabilities: got.probabilities, go } : { failed: true }),
        });
        // The words it judged were complete: check with the words as they are now (if still the same turn).
        if (go && words && g === gen) startCheck();
        gate();
      });
    };

    return {
      onInput(port, msg) {
        if (port === 'vad') {
          if (msg.type === 'vad.speech_start') speaking = true;
          else if (msg.type === 'vad.speech_end') {
            speaking = false;
            if (check?.interject) take('the user paused');
          }
          return;
        }
        if (port === 'text') {
          const c = check;
          if (msg.type !== 'llm.check_verdict' || !c || msg.checkId !== c.id) return;
          if (msg.verdict !== 'interject') {
            check = null;
            ctx.setState('check', '');
            return;
          }
          c.interject = String(msg.text ?? '');
          // The user talked on while the LLM was deciding.
          if (count(draftKey(words)) - count(c.key) > Number(p.maxExtraWords)) return drop(`the user said more than ${Number(p.maxExtraWords)} words past it`);
          if (!speaking) take('the user was already silent');
          else
            c.wait = ctx.setTimer(
              Number(p.maxWaitMs),
              () => {
                if (check === c && !c.taken) drop(`no pause within ${Number(p.maxWaitMs)} ms`);
              },
              'interject',
            );
          return;
        }
        switch (msg.type) {
          case 'turn.words': {
            words = String(msg.text ?? '');
            replaces = (msg.replaces as string) ?? null;
            if (!words) {
              // Committed, or gone: start over with the next words.
              asked = checked = '';
              gen++;
              if (!check?.taken) drop('the turn ended');
              return;
            }
            if (check?.interject && !check.taken && count(draftKey(words)) - count(check.key) > Number(p.maxExtraWords))
              drop(`the user said more than ${Number(p.maxExtraWords)} words past it`);
            gate();
            return;
          }
          case 'turn.draft':
            draftOut = true;
            return;
          case 'turn.draft_drop':
            draftOut = false;
            gate();
            return;
          case 'turn.user':
            draftOut = false;
            if (check && msg.interject === check.id) {
              check = null;
              ctx.setState('check', '');
            } else drop('the turn was committed without it');
            return;
          case 'turn.interject':
            if (!msg.taken && msg.checkId === check?.id) drop(`turn-taking refused it: ${String(msg.why)}`);
            return;
        }
      },
      stop() {
        check?.wait?.clear();
      },
    };
  },
};

export const noInterjectNode: NodeDef<Record<string, never>> = {
  type: 'interject.off',
  title: 'No early interjection',
  category: 'control',
  description: 'The agent waits for the end of the turn.',
  ...ports,
  params: [],
  create() {
    return {};
  },
};
