import type { NodeDef, TimerHandle } from '../core/node';
import type { JsonObject } from '../core/types';
import { askChoice, choiceRequest, optionLines, parseChoiceOptions, readChoice, rotator, type ChoicePick } from './typesafe';

export const DEFAULT_FILLER_LINES = 'Let me see.\nHmm, let me think.\nOne moment.\nLet me think about that.';

const ports = {
  inputs: [
    { name: 'text', type: 'text', description: "The LLM's reply stream (llm.start, llm.thinking, llm.delta, llm.end)" },
    { name: 'turn', type: 'turn', description: "The user's words (turn.draft, turn.user)" },
  ],
  outputs: [{ name: 'text', type: 'text', description: 'opener.say: a line for TTS to speak on the reply, before its words' }],
};

/**
 * Says a filler ("Let me see.") on a reply while a reasoning model keeps the
 * user waiting: the model is thinking and no words have come `afterMs` after
 * the turn was committed. It only watches the LLM's stream and speaks through
 * TTS on the same reply; the LLM never sees it, so its prompt, history and
 * cache are untouched. Lines rotate and the last one is never said twice in a row.
 */
export const fillerOpenerNode: NodeDef = {
  type: 'opener.filler',
  title: 'Thinking filler',
  category: 'llm',
  description: 'Says a filler line while a reasoning model is still thinking after the turn. The LLM never knows.',
  ...ports,
  params: [
    { key: 'lines', label: 'Filler lines', type: 'text', default: DEFAULT_FILLER_LINES, description: 'One per line, said in turn.' },
    {
      key: 'afterMs',
      label: 'After (ms)',
      type: 'number',
      default: 500,
      description: 'Say a filler if the model is still thinking this long after the turn was committed, with no words yet.',
    },
  ],
  create(ctx) {
    const p = ctx.params as JsonObject;
    const lines = String(p.lines ?? '')
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean);
    let n = 0;
    let last = '';
    /** The reply in flight: when it started, whether the model is thinking, whether anything was said. */
    let cur: { rid: string; t0: number; thinking: boolean; spoken: boolean; timer?: TimerHandle } | undefined;

    const check = () => {
      if (!cur || !cur.thinking || cur.spoken || !lines.length) return;
      const waited = ctx.now() - cur.t0;
      if (waited < Number(p.afterMs)) return;
      let line = lines[n++ % lines.length];
      if (line === last && lines.length > 1) line = lines[n++ % lines.length];
      last = line;
      cur.spoken = true;
      ctx.emit('text', { type: 'opener.say', rid: cur.rid, text: `${line} `, line, reason: `still thinking ${Math.round(waited)} ms after the turn` });
    };
    const end = () => {
      cur?.timer?.clear();
      cur = undefined;
    };

    return {
      start() {
        // Tell TTS the lines, so it can have them ready.
        if (lines.length) ctx.emit('text', { type: 'opener.lines', lines: [...new Set(lines)] });
      },
      onInput(_port, msg) {
        switch (msg.type) {
          case 'llm.start':
            end();
            cur = { rid: msg.rid as string, t0: ctx.now(), thinking: false, spoken: false };
            cur.timer = ctx.setTimer(Number(p.afterMs), check, 'filler');
            break;
          case 'llm.thinking':
            if (!cur || cur.rid !== msg.rid) return;
            cur.thinking = true;
            check();
            break;
          case 'llm.delta':
            // The model is talking: too late for a filler.
            if (cur && cur.rid === msg.rid) cur.spoken = true;
            break;
          case 'llm.end':
            if (cur?.rid === msg.rid) end();
            break;
        }
      },
      stop: end,
    };
  },
};

export const noOpenerNode: NodeDef<Record<string, never>> = {
  type: 'opener.off',
  title: 'No opener',
  category: 'llm',
  description: 'Replies start with the LLM\'s own words.',
  ...ports,
  params: [],
  create() {
    return {};
  },
};

export const DEFAULT_OPENER_OPTIONS = `# One option per line: what to say = when it fits. "a | b = …" says a, then b next time. "-" says nothing.
Hmm, let me think. | Hmm. = The user asked for an opinion, a judgement, a recommendation or a decision that genuinely needs weighing up.
Good question. = The user asked a thoughtful how or why question about how something works.
Sure. = The user asked the agent to do something for them, such as an order, a task or a story.
Oh, nice. = The user shared good news or something they are happy or excited about.
Oh no. = The user shared bad news, a mishap or a frustration from their own life.
Oh, interesting. | Huh. = The user shared a surprising fact, finding or idea.
Ah, I see. = The user corrected the agent or clarified what they meant.
- = Anything else: greetings, small talk, thanks, goodbyes, short answers, acknowledgements or agreement, quick factual questions, plain statements, an unfinished fragment, or anything a quick direct reply answers well.`;

export const DEFAULT_OPENER_INSTRUCTIONS =
  'The text is a speech-recognition transcript of what a user just said to a voice agent in a live spoken conversation (it may have recognition errors). The agent\'s reply is still being prepared, so it can say a few words right away to fill the silence. Pick the opening words a thoughtful person would naturally say first. The opener is said before the agent knows its reply, so it must fit any sensible answer: never agree, disagree, promise or commit to anything, and never sound like more thought is coming than the message needs. Most messages need no opener: pick "-" unless one option clearly fits.';

/**
 * Says an opener ("Good question.", "Oh no.") picked by TypeSafe Choice from
 * the user's ASR words, when a reply has no words of its own `afterMs` after it
 * started. The pick is asked for as soon as the words are known (a draft, then
 * the committed turn), so it is usually ready when the reply starts. Like the
 * filler, it only speaks through TTS: the LLM never sees it.
 */
export const choiceOpenerNode: NodeDef = {
  type: 'opener.choice',
  title: 'TypeSafe opener',
  category: 'llm',
  description: "Says an opener picked by TypeSafe Choice from the user's words when the reply is slow to start. The LLM never knows.",
  ...ports,
  params: [
    { key: 'options', label: 'Options', type: 'text', default: DEFAULT_OPENER_OPTIONS, description: 'line = when it fits ("-" says nothing). Sent to TypeSafe as the Choice criteria.' },
    { key: 'instructions', label: 'Question', type: 'text', default: DEFAULT_OPENER_INSTRUCTIONS, description: "The Choice instructions. The state is only the user's ASR words for the turn." },
    { key: 'minConfidence', label: 'Min confidence', type: 'number', default: 0.5, description: 'Say nothing when TypeSafe is less sure than this of its pick.' },
    {
      key: 'afterMs',
      label: 'After (ms)',
      type: 'number',
      default: 200,
      description: "Say the opener if the reply has no words this long after it started. A reply drafted ahead usually has its words at once, and gets none.",
    },
    { key: 'model', label: 'TypeSafe model', type: 'string', default: 'jev-latest' },
  ],
  channels: [{ name: 'typesafe', description: "TypeSafe Choice over the user's words: one JSON request, one JSON answer" }],
  requires: () => ['typesafe'],
  create(ctx) {
    const p = ctx.params as JsonObject;
    const opts = parseChoiceOptions(String(p.options ?? ''));
    const next = rotator();
    /** Picks by the user's words: the answer, or the callbacks waiting for it. */
    const picks = new Map<string, ChoicePick | null>();
    const waiting = new Map<string, ((pick: ChoicePick | null) => void)[]>();
    const turnText = new Map<string, string>();
    /** The reply in flight. */
    let cur: { rid: string; turnId: string | null; t0: number; words: boolean; done: boolean; pick?: ChoicePick | null; timer?: TimerHandle } | undefined;

    const pick = (text: string, go?: (pick: ChoicePick | null) => void) => {
      if (!text || !opts.length) return go?.(null);
      if (picks.has(text)) return go?.(picks.get(text)!);
      const w = waiting.get(text);
      if (w) return void (go && w.push(go));
      waiting.set(text, go ? [go] : []);
      const t0 = ctx.now();
      askChoice(ctx, choiceRequest(text, opts, String(p.instructions ?? ''), String(p.model ?? 'jev-latest')), (res) => {
        const got = res === null ? null : readChoice(res, opts, Number(p.minConfidence));
        picks.set(text, got);
        ctx.emit('text', {
          type: 'opener.pick',
          text,
          ms: Math.round(ctx.now() - t0),
          ...(got ? { choice: got.choice, confidence: got.confidence, probabilities: got.probabilities, say: got.option !== null } : { failed: true }),
        });
        const cbs = waiting.get(text) ?? [];
        waiting.delete(text);
        for (const cb of cbs) cb(got);
      });
    };

    /** Say the pick if the reply still has no words and has waited long enough. */
    const check = () => {
      const r = cur;
      if (!r || r.done || r.words || r.pick === undefined) return;
      const waited = ctx.now() - r.t0;
      if (waited < Number(p.afterMs)) return;
      r.done = true;
      const opt = r.pick?.option;
      if (!opt) return;
      const line = next(opt);
      ctx.setState('last', line);
      ctx.emit('text', {
        type: 'opener.say',
        rid: r.rid,
        text: `${line} `,
        line,
        reason: `${opt.key === line ? line : `${opt.key} (${line})`} at ${r.pick!.confidence}, no words ${Math.round(waited)} ms after the reply started`,
      });
    };

    const follow = (text: string) => {
      const r = cur;
      if (!r) return;
      pick(text, (got) => {
        if (cur !== r) return;
        r.pick = got;
        check();
      });
    };

    const end = () => {
      cur?.timer?.clear();
      cur = undefined;
    };

    return {
      start() {
        const lines = optionLines(opts);
        if (lines.length) ctx.emit('text', { type: 'opener.lines', lines });
      },
      onInput(port, msg) {
        if (port === 'turn') {
          if (msg.type !== 'turn.draft' && msg.type !== 'turn.user') return;
          const text = String(msg.text ?? '').trim();
          if (msg.type === 'turn.user' && msg.turnId) turnText.set(msg.turnId as string, text);
          if (msg.type === 'turn.user' && cur && cur.turnId === msg.turnId && cur.pick === undefined) follow(text);
          else pick(text);
          return;
        }
        switch (msg.type) {
          case 'llm.start': {
            end();
            // Replies the user didn't ask for (a worker's news) get no opener.
            const turnId = (msg.turnId as string | null) ?? null;
            if (!turnId) return;
            cur = { rid: msg.rid as string, turnId, t0: ctx.now(), words: false, done: false };
            cur.timer = ctx.setTimer(Number(p.afterMs), check, 'opener');
            const text = turnText.get(turnId);
            if (text !== undefined) follow(text);
            break;
          }
          case 'llm.delta':
            // The model is talking: too late for an opener.
            if (cur && cur.rid === msg.rid) cur.words = true;
            break;
          case 'llm.end':
            if (cur?.rid === msg.rid) end();
            break;
        }
      },
      stop: end,
    };
  },
};
