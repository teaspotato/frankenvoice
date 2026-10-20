import type { NodeDef, TimerHandle } from '../core/node';
import type { Json, JsonObject } from '../core/types';
import { askChoice, choiceRequest, optionLines, parseChoiceOptions, readChoice, rotator } from './typesafe';

/**
 * Lines the agent can say while the user keeps talking. A bare line is a
 * continuer, rotated so the same one doesn't repeat; a line with examples is
 * said when the user's last words are near one of them in meaning.
 */
export const DEFAULT_BACKCHANNEL_LINES = `# One line per backchannel. A bare line is a continuer ("go on"), taken in turn.
# "line = example | example" is said instead when the user's recent words are close to an example.
Mm-hmm.
Yeah.
Right.
Oh wow. = and then I won the whole thing | it was the most amazing view | we finally got the offer | I got the job
Oh no. = and then my laptop died | I lost my keys again | the flight got cancelled | it broke on the first day
I see. = so that's why it never worked | it turns out it was the cable all along | that's the reason I left`;

export const DEFAULT_BC_OPTIONS = `# One option per line: what to say = when it fits. "a | b = …" says a, then b next time. "-" says nothing.
Mm-hmm. | Yeah. | Right. = A neutral continuer: the user just finished a statement or a point while telling, describing or explaining something.
Oh wow. = The user just said something surprising, impressive or exciting.
Oh no. = The user just described a mishap, a problem, bad luck or a frustration.
I see. = The user just explained a reason or a cause, or corrected something.
- = Nothing: a question or request, a greeting, thanks or goodbye, a reaction to what the agent said, only a word or two, or a phrase broken off in the middle.`;

export const DEFAULT_BC_INSTRUCTIONS =
  'The text is a live speech-recognition transcript of what a user has said so far in their current turn of a spoken conversation with a voice agent (it may have recognition errors, and the last words may be missing or cut off). The user has just paused. Pick what the agent says right now, quietly, while the user keeps talking. A backchannel only shows the agent is listening; it never answers or takes the turn. Good listeners backchannel often, every sentence or two, whenever the speaker is telling, describing or explaining something of their own and has just finished a statement or a point, even if more is likely to come. Pick "-" when a backchannel would be wrong: the user asked a question or made a request (they want an answer), greeted, thanked or said goodbye, is only reacting to what the agent said ("okay", "that\'s great", "sounds good", "I see"), said only a word or two, or broke off in the middle of a phrase.';

/** Added to the question when the state starts with when the agent last backchannelled. */
export const BC_CONTEXT_NOTE =
  'The first line, in brackets, is not speech: it says when the agent last backchannelled. Only avoid a backchannel within about 2 s of the last one; otherwise judge from the words alone.';

/**
 * The last user turn of a generated backchannel's request starts with this, then the user's words so
 * far: first, so it stays in llama.cpp's prompt cache (from the last request) and only the newest
 * words need a prefill.
 */
export const DEFAULT_BC_PROMPT =
  "(They're still talking. Reply with only a brief backchannel, one to three words, that fits what they just said. No question.)";

/**
 * The start of the assistant's turn for a generated backchannel: the utterance tag with its flags
 * (not interruptible, not gated), so the model writes only the words.
 */
export const DEFAULT_BC_PREFILL = '[i0g0] ';

/** The words Gemma may write after the prefill: 1 to `maxWords` words, then at most one . or !. */
export function bcGrammar(maxWords: number): string {
  const n = Math.max(1, Math.round(maxWords));
  return `root ::= w (","? " " w){0,${n - 1}} [.!]?\nw ::= [A-Za-z'-]+\n`;
}

/** A generated backchannel cleaned up, or null when it isn't one (empty, too long, a question). */
export function cleanBackchannel(raw: string, maxWords: number): string | null {
  let s = raw.replace(/^\s*\[[^\]]*\]\s*/, '').split('\n')[0];
  s = s.replace(/[*_"“”()]/g, '').trim();
  if (!s || s.includes('?')) return null;
  const ws = s.match(/[\p{L}\p{N}']+/gu) ?? [];
  if (!ws.length || ws.length > maxWords) return null;
  if (!/[.!…]$/.test(s)) s = `${s.replace(/[,;:—–-]+$/, '')}.`;
  return s;
}

/** The state TypeSafe reads: the last backchannel (if `context`), then the user's words. */
export function bcState(text: string, last: { line: string; agoMs: number } | null, context: boolean): string {
  if (!context) return text;
  const head = last ? `[Agent's last backchannel: "${last.line}", ${Math.round(last.agoMs / 1000)} s ago]` : "[Agent hasn't backchannelled yet this turn]";
  return `${head}\n${text}`;
}

/** A line with a user utterance it fits. */
export interface LineExample {
  line: string;
  text: string;
}

export interface BackchannelLines {
  continuers: string[];
  examples: LineExample[];
}

/** The example nearest `v` (unit vectors, so the dot product is the cosine); null below `minScore`. */
export function pickLine(v: number[], examples: LineExample[], vecs: number[][], minScore: number) {
  let best = -1;
  let score = -Infinity;
  vecs.forEach((e, i) => {
    let s = 0;
    for (let k = 0; k < e.length; k++) s += e[k] * v[k];
    if (s > score) [best, score] = [i, s];
  });
  const ex = examples[best];
  if (!ex) return null;
  return { line: score >= minScore ? ex.line : null, score: Math.round(score * 1000) / 1000, example: ex.text };
}

export function parseBackchannelLines(src: string): BackchannelLines {
  const continuers: string[] = [];
  const examples: LineExample[] = [];
  for (const raw of src.split('\n')) {
    const s = raw.trim();
    if (!s || s.startsWith('#')) continue;
    const i = s.indexOf('=');
    if (i < 0) {
      continuers.push(s);
      continue;
    }
    const line = s.slice(0, i).trim();
    if (!line) continue;
    for (const ex of s.slice(i + 1).split('|')) if (ex.trim()) examples.push({ line, text: ex.trim() });
  }
  return { continuers, examples };
}

/**
 * The texts TTS renders for one line: the line, then the same words with other punctuation and case,
 * so each take comes out a little different (same words, never a different meaning).
 */
export function takes(line: string, n: number): string[] {
  const base = line.replace(/[.!?,…]+$/u, '');
  const low = base.charAt(0).toLowerCase() + base.slice(1);
  const out = [...new Set([line, `${base},`, `${base}...`, `${low}.`, `${low},`, `${low}...`, `${base}.`, `${base}!`])];
  return out.slice(0, Math.max(1, Math.min(out.length, Math.round(n))));
}

/** Deterministic random numbers in [0, 1) from a string, so a replay draws the same. */
export function seededRandom(seed: string): () => number {
  let h = 2166136261;
  for (let i = 0; i < seed.length; i++) h = Math.imul(h ^ seed.charCodeAt(i), 16777619);
  let a = h >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Everything the agent can say as a backchannel. */
export const allLines = (l: BackchannelLines) => [...new Set([...l.continuers, ...l.examples.map((e) => e.line)])];

interface BackchannelParams {
  source: string;
  say: string;
  prompt: string;
  prefill: string;
  speculate: boolean;
  maxWords: number;
  temperature: number;
  waitMs: number;
  fallback: string;
  decide: string;
  options: string;
  instructions: string;
  minConfidence: number;
  model: string;
  context: boolean;
  gapMs: number;
  pauseMs: number;
  maxEot: number;
  minTurnMs: number;
  minGapMs: number;
  maxSilenceMs: number;
  lines: string;
  minScore: number;
  embedModel: string;
  takes: number;
  tailMs: number;
}

const words = (s: string) => s.toLowerCase().match(/[\p{L}\p{N}']+/gu) ?? [];

/**
 * Backchannels: short lines ("mm-hmm", "oh no") the agent says while
 * the user holds the floor, to show it is listening. Something proposes a
 * moment (the turn model's backchannel score, or a pause in the user's speech),
 * and rules gate it: the user has the floor and has talked for a while, isn't
 * about to hand over (the VAD heard them recently, ASR hasn't endpointed,
 * end-of-turn score low, no draft reply out), and the last
 * backchannel was long enough ago.
 *
 * What to say: generated (the default), Gemma writes it at that moment (`bc.ask` to the LLM stage,
 * which sends the conversation so far plus the user's words with `prompt` as their turn, on its own
 * llama.cpp slot, and never keeps it), while TTS opens a socket for it (`bc.prep`); `bc.say` with
 * `gen` speaks it there. An answer that isn't a backchannel, or none within `waitMs`, says a
 * canned `fallback` line instead. canned: lines rendered ahead by TTS.
 * Either way it plays as a cue that doesn't start a reply: turn-taking keeps listening, and nothing
 * the user does stops it once it sounds. A generated line whose audio hasn't started when the
 * user's turn ends is dropped (`bc.drop`).
 */
export const backchannelNode: NodeDef<BackchannelParams> = {
  type: 'backchannel',
  title: 'Backchannels',
  category: 'control',
  description: 'Says "mm-hmm"-style lines while the user keeps talking: when, from the turn model or pauses; what, written by the LLM there and then, or canned lines.',
  inputs: [
    { name: 'vad', type: 'vad' },
    { name: 'scores', type: 'turnscores' },
    { name: 'turn', type: 'turn' },
    { name: 'transcript', type: 'transcript' },
    { name: 'llm', type: 'text', description: 'llm.bc: the backchannel the LLM wrote' },
  ],
  outputs: [
    { name: 'text', type: 'text', description: 'bc.lines (render these ahead), bc.prep (open a socket), bc.say (play one now) and bc.drop for TTS' },
    { name: 'ask', type: 'backchannel', description: 'bc.ask / bc.drop: have the LLM write a backchannel' },
  ],
  params: [
    {
      key: 'source',
      label: 'When',
      type: 'select',
      options: ['turn model', 'pause'],
      default: 'pause',
      description:
        'turn model: when its backchannel score fires (needs a turn model with a backchannel head). pause: when the user pauses mid-turn and the end-of-turn score (if any) is low.',
    },
    {
      key: 'say',
      label: 'Say',
      type: 'select',
      options: ['generated', 'canned'],
      default: 'generated',
      description:
        "generated: the LLM writes the line when it's time, from the conversation and the user's words so far, and TTS speaks it at once. canned: lines rendered ahead.",
    },
    {
      key: 'prompt',
      label: 'Instruction',
      type: 'text',
      default: DEFAULT_BC_PROMPT,
      when: { say: ['generated'] },
      description: "Appended to the user's words so far, as the request's last user turn (the system prompt and history stay the reply's, so the cached prefix is reused).",
    },
    {
      key: 'prefill',
      label: 'Prefill',
      type: 'string',
      default: DEFAULT_BC_PREFILL,
      when: { say: ['generated'] },
      description: "The start of the LLM's answer, written for it: the utterance tag (not interruptible, not gated), so it writes only the words.",
    },
    {
      key: 'speculate',
      label: 'Write ahead',
      type: 'boolean',
      default: true,
      when: { say: ['generated'] },
      description:
        "While the user talks, have the LLM write a line from their words so far (one at a time, at most every 300 ms, on the backchannel slot). When it's time, a line written from the same words (but for the last two) is said at once, without waiting for the LLM.",
    },
    { key: 'maxWords', label: 'Max words', type: 'number', default: 4, when: { say: ['generated'] }, description: 'A grammar caps the line at this many words (and no question).' },
    { key: 'temperature', label: 'Temperature', type: 'number', default: 0.3, when: { say: ['generated'] } },
    {
      key: 'waitMs',
      label: 'Wait for the LLM (ms)',
      type: 'number',
      default: 500,
      when: { say: ['generated'] },
      description: 'Say a fallback line instead when the LLM has not written one by then.',
    },
    {
      key: 'fallback',
      label: 'Fallback lines',
      type: 'string',
      default: 'Mm-hmm. | Yeah. | Right.',
      when: { say: ['generated'] },
      description: "Canned lines, taken in turn, when the LLM's line is late or isn't a backchannel.",
    },
    {
      key: 'decide',
      label: 'Decide',
      type: 'select',
      options: ['rules', 'typesafe'],
      default: 'rules',
      description:
        "Whether and what to say at a proposed moment. rules: the gates below (talked long enough, end-of-turn score low, no draft, ASR hasn't endpointed), then a continuer or a line picked by meaning. typesafe: TypeSafe Choice reads the user's ASR words for this turn and picks a line or nothing; only the floor, the gap and the silence are checked.",
    },
    { key: 'options', label: 'Options', type: 'text', default: DEFAULT_BC_OPTIONS, when: { decide: ['typesafe'] }, description: 'line = when it fits ("-" says nothing). Sent to TypeSafe as the Choice criteria.' },
    { key: 'instructions', label: 'Question', type: 'text', default: DEFAULT_BC_INSTRUCTIONS, when: { decide: ['typesafe'] }, description: "The Choice instructions. The state is only the user's ASR words for the turn so far." },
    { key: 'minConfidence', label: 'Min confidence', type: 'number', default: 0.4, when: { decide: ['typesafe'] }, description: 'Say nothing when TypeSafe is less sure than this of its pick.' },
    { key: 'model', label: 'TypeSafe model', type: 'string', default: 'jev-latest', when: { decide: ['typesafe'] } },
    {
      key: 'context',
      label: 'Tell it the last backchannel',
      type: 'boolean',
      default: true,
      when: { decide: ['typesafe'] },
      description: 'Start the state with the last line said and how long ago, so TypeSafe can space them out itself.',
    },
    {
      key: 'gapMs',
      label: 'Min gap (ms)',
      type: 'number',
      default: 2000,
      when: { decide: ['typesafe'] },
      description: 'Time since the last backchannel. With the last backchannel in the state, 2 s gave gaps like people leave on TurnBench (median ~8 s, p10 ~3 s).',
    },
    { key: 'pauseMs', label: 'Pause (ms)', type: 'number', default: 0, when: { source: ['pause'] }, description: 'Extra silence after the VAD hears speech end.' },
    { key: 'maxEot', label: 'Max end-of-turn score', type: 'number', default: 0.3, when: { decide: ['rules'] }, description: "Don't backchannel when the user seems to be done: the turn model's end-of-turn score for the user is at least this." },
    { key: 'minTurnMs', label: 'Min turn length (ms)', type: 'number', default: 3000, when: { decide: ['rules'] }, description: 'The user has been talking this long in the current turn.' },
    { key: 'minGapMs', label: 'Min gap (ms)', type: 'number', default: 4000, when: { decide: ['rules'] }, description: 'Time since the last backchannel. People leave about 9 s (median), rarely under 3 s.' },
    {
      key: 'maxSilenceMs',
      label: 'Max silence (ms)',
      type: 'number',
      default: 700,
      when: { decide: ['rules'] },
      description: "The VAD has heard the user within this long: most backchannels go over the user's speech, and a long silence means they may be done, whatever the turn model says.",
    },
    { key: 'lines', label: 'Lines', type: 'text', default: DEFAULT_BACKCHANNEL_LINES, when: { decide: ['rules'], say: ['canned'] }, description: 'Bare lines are rotated; "line = example | …" is picked by meaning from the user\'s recent words.' },
    { key: 'minScore', label: 'Min similarity', type: 'number', default: 0.45, when: { decide: ['rules'], say: ['canned'] }, description: 'A line with examples is said only when the recent words are at least this close; else a continuer.' },
    {
      key: 'takes',
      label: 'Takes per line',
      type: 'number',
      default: 6,
      description: 'TTS renders each line this many times ahead (spelled slightly differently, so the takes differ), and a different take is played each time.',
    },
    { key: 'tailMs', label: 'Tail (ms)', type: 'number', default: 150, description: "Sustain the line's last sound this long and fade it out, since TTS often stops it dead." },
    {
      key: 'embedModel',
      label: 'Encoder',
      type: 'string',
      default: 'Xenova/all-MiniLM-L6-v2',
      when: { decide: ['rules'], say: ['canned'] },
      description: 'Sentence encoder for picking lines. Runs in the browser.',
    },
  ],
  channels: [
    { name: 'embed', description: 'Sentence encoder in the browser: texts out, unit vectors back' },
    { name: 'typesafe', description: "TypeSafe Choice over the user's words: one JSON request, one JSON answer" },
  ],
  requires: (p) => (p.decide === 'typesafe' ? ['typesafe'] : []),
  create(ctx) {
    const p = ctx.params;
    const typesafe = p.decide === 'typesafe';
    const gen = p.say === 'generated';
    const lines = typesafe || gen ? { continuers: [], examples: [] } : parseBackchannelLines(String(p.lines));
    const fallbacks = gen ? String(p.fallback).split('|').map((l) => l.trim()).filter(Boolean) : [];
    let fallbackNo = 0;
    /** A line the LLM is writing: its id, the turn it is for, and the wait for it. */
    let pending: { id: string; why: string; data: JsonObject; t0: number; prevBc: number; timer: TimerHandle } | undefined;
    /** Generated lines said this turn: TTS drops one whose audio hasn't started when the turn ends. */
    const said = new Set<string>();
    const opts = typesafe ? parseChoiceOptions(String(p.options)) : [];
    const nextLine = rotator();
    let exampleVecs: number[][] | undefined;
    /** The agent's floor, from turn-taking's transitions. */
    let floor = 'listening';
    /** The current user turn: when it started and how much speech it has had. */
    let turnStart: number | null = null;
    let speechMs = 0;
    let speechFrom: number | null = null;
    let draftOut = false;
    /** When the VAD last heard the user stop; ASR said they finished (speech_final) since then. */
    let silentSince: number | null = null;
    /** The VAD hears the user now, whoever has the floor. */
    let vadSpeech = false;
    /** Counts user turns, so an encoder answer can tell its turn is over. */
    let turnNo = 0;
    let asrDone = false;
    let eotUser = 0;
    let lastBc = -Infinity;
    let lastLine = '';
    let rot = 0;
    const lastTake = new Map<string, string>();
    let n = 0;
    /** Recent words of this turn (finals + the latest partial), for picking a line. */
    let finals = '';
    let partial = '';
    let pause: TimerHandle | undefined;

    const embed = (texts: string[], done: (vecs: number[][] | null) => void) => {
      let got = false;
      ctx.io.open(
        'embed',
        { model: p.embedModel, texts },
        {
          onRecv(data) {
            const v = (data as { vectors?: number[][] } | null)?.vectors;
            if (got || !v) return;
            got = true;
            done(v);
          },
          onClose() {
            if (!got) done(null);
            got = true;
          },
        },
      );
    };

    const talkedMs = (now: number) => speechMs + (speechFrom !== null ? now - speechFrom : 0);

    /** Why not now, or null when a backchannel may be said. TypeSafe judges the rest from the words. */
    const blocked = (now: number, gap = true): string | null => {
      if (floor !== 'listening') return `agent is ${floor}`;
      if (turnStart === null) return 'no user turn';
      if (gap && now - lastBc < (typesafe ? p.gapMs : p.minGapMs)) return `last backchannel ${Math.round(now - lastBc)} ms ago`;
      // TypeSafe answers in ~200 ms, and judges from the words whether the user is done.
      if (typesafe) return null;
      if (silentSince !== null && now - silentSince > p.maxSilenceMs) return `user silent for ${Math.round(now - silentSince)} ms`;
      if (talkedMs(now) < p.minTurnMs) return `user has talked ${Math.round(talkedMs(now))} ms`;
      if (asrDone) return 'ASR heard the end of the utterance';
      if (draftOut) return 'a reply is being drafted';
      if (eotUser >= p.maxEot) return `end-of-turn score ${eotUser}`;
      return null;
    };

    const say = (line: string, why: string, extra: JsonObject = {}, id = `${ctx.id}-bc${n++}`) => {
      lastBc = ctx.now();
      // A take other than the one this line used last, drawn from the id so a replay draws the same.
      const all = takes(line, p.takes);
      const prev = lastTake.get(line);
      const pool = all.length > 1 ? all.filter((t) => t !== prev) : all;
      const take = extra.gen ? line : pool[Math.floor(seededRandom(id)() * pool.length)];
      lastTake.set(line, take);
      lastLine = line;
      ctx.emit('text', { type: 'bc.say', id, line, take, tailMs: p.tailMs, why, ...extra });
      ctx.setState('last', line);
      ctx.setState('count', n);
    };

    const nextContinuer = () => {
      const c = lines.continuers;
      return c.length ? c[rot++ % c.length] : null;
    };

    /** A moment was proposed: gate it, then pick what to say. */
    const propose = (why: string, data: JsonObject = {}) => {
      const now = ctx.now();
      const no = blocked(now);
      if (no) {
        ctx.emit('text', { type: 'bc.skip', why, reason: no, ...data });
        return;
      }
      // Claim the slot now, so a second proposal during the encoder call is gated.
      const prevBc = lastBc;
      lastBc = now;
      const turn = turnNo;
      if (typesafe) return decide(why, data, prevBc, turn);
      if (gen) return generate(why, data, prevBc);
      const recent = words(`${finals} ${partial}`).slice(-12).join(' ');
      if (!recent || !exampleVecs || !lines.examples.length) {
        const c = nextContinuer();
        if (c) say(c, why, data);
        return;
      }
      embed([recent], (vecs) => {
        const pick = vecs?.[0] ? pickLine(vecs[0], lines.examples, exampleVecs!, Number(p.minScore)) : null;
        const pickData: JsonObject = pick ? { words: recent, score: pick.score, example: pick.example } : { words: recent };
        // The moment may have passed while the encoder ran.
        const late = turn !== turnNo ? 'the turn ended' : blocked(ctx.now(), false);
        if (late) {
          lastBc = prevBc;
          return void ctx.emit('text', { type: 'bc.skip', why, reason: `moment passed: ${late}`, ...data });
        }
        const line = pick?.line ?? nextContinuer();
        if (line) say(line, why, { ...data, ...pickData });
      });
    };

    /** TypeSafe reads the turn's words so far and picks a line, or nothing. */
    const decide = (why: string, data: JsonObject, prevBc: number, turn: number) => {
      // The ASR words as they came, the latest partial included; a long turn is cut to its end.
      const text = `${finals} ${partial}`.trim().split(/\s+/).slice(-80).join(' ');
      if (!text || !opts.length) {
        lastBc = prevBc;
        return void ctx.emit('text', { type: 'bc.skip', why, reason: 'no words yet', ...data });
      }
      const t0 = ctx.now();
      const state = bcState(text, lastLine ? { line: lastLine, agoMs: t0 - prevBc } : null, !!p.context);
      const instructions = p.context ? `${p.instructions} ${BC_CONTEXT_NOTE}` : String(p.instructions);
      askChoice(ctx, choiceRequest(state, opts, instructions, String(p.model)), (res) => {
        const got = res === null ? null : readChoice(res, opts, Number(p.minConfidence));
        const pickData: JsonObject = got
          ? { words: text, choice: got.choice, confidence: got.confidence, probabilities: got.probabilities, ms: Math.round(ctx.now() - t0) }
          : { words: text, failed: true, ms: Math.round(ctx.now() - t0) };
        const late = turn !== turnNo ? 'the turn ended' : blocked(ctx.now(), false);
        const no = !got ? 'TypeSafe gave no answer' : !got.option ? `TypeSafe picked ${got.choice} at ${got.confidence}` : late ? `moment passed: ${late}` : null;
        if (no || !got?.option) {
          lastBc = prevBc;
          return void ctx.emit('text', { type: 'bc.skip', why, reason: no ?? 'nothing picked', ...data, ...pickData });
        }
        if (gen) generate(why, { ...data, ...pickData }, prevBc);
        else say(nextLine(got.option), why, { ...data, ...pickData });
      });
    };

    /** The user's words so far, for the LLM. */
    const userWords = () => `${finals} ${partial}`.trim().split(/\s+/).slice(-80).join(' ');
    const maxWords = () => Math.max(1, Math.round(Number(p.maxWords) || 3));
    const askFor = (id: string, text: string, draft: boolean) =>
      ctx.emit('ask', {
        type: 'bc.ask',
        id,
        text,
        prompt: String(p.prompt),
        prefill: String(p.prefill),
        // A word is a token or two, and the grammar ends it.
        body: { max_tokens: maxWords() * 3 + 3, temperature: Number(p.temperature), grammar: bcGrammar(maxWords()) },
        ...(draft ? { draft: true } : {}),
      });
    /**
     * A line written ahead from the words so far (`speculate`): its words, and the line once written
     * (null: not a backchannel). At the moment, it is said at once if the words haven't moved on.
     */
    let draft: { id: string; text: string; inFlight: boolean; t0: number; line?: string | null; raw?: string; ms?: number } | undefined;
    let draftNo = 0;
    /** The draft was written from (nearly) these words: at most 2 words fewer, the rest the same but for the last 2. */
    const fits = (d: { text: string }, text: string) => {
      const a = words(d.text);
      const b = words(text);
      const k = Math.max(0, a.length - 2);
      return b.length - a.length <= 2 && b.length >= a.length - 1 && a.slice(0, k).join(' ') === b.slice(0, k).join(' ');
    };
    /** Write a line ahead from the words so far: one at a time, at most every 300 ms. */
    const speculate = () => {
      if (!gen || !p.speculate || pending || floor !== 'listening' || turnStart === null) return;
      if (draft && (draft.inFlight || ctx.now() - draft.t0 < 300)) return;
      const text = userWords();
      if (!text || text === draft?.text) return;
      draft = { id: `${ctx.id}-d${draftNo++}`, text, inFlight: true, t0: ctx.now() };
      askFor(draft.id, text, true);
    };
    const drafted = (msg: JsonObject) => {
      if (msg.id !== draft?.id) return;
      draft.inFlight = false;
      draft.raw = String(msg.text ?? '');
      draft.ms = Math.round(ctx.now() - draft.t0);
      draft.line = msg.status === 'done' ? cleanBackchannel(draft.raw, maxWords()) : null;
      // The user talked on meanwhile: write again from where they are.
      speculate();
    };

    /** Have the LLM write the line now, while TTS opens a socket for it. */
    const generate = (why: string, data: JsonObject, prevBc: number) => {
      const text = userWords();
      if (!text) {
        lastBc = prevBc;
        return void ctx.emit('text', { type: 'bc.skip', why, reason: 'no words yet', ...data });
      }
      dropPending('asked again');
      const d = draft;
      draft = undefined;
      if (d && fits(d, text) && !d.inFlight && d.line) {
        // Written ahead from these words: say it now.
        const id = `${ctx.id}-bc${n++}`;
        ctx.emit('text', { type: 'bc.prep', id });
        said.add(id);
        return say(d.line, why, { ...data, gen: true, raw: d.raw ?? '', ms: 0, draft: { text: d.text, ms: d.ms ?? null }, askedAt: ctx.now() }, id);
      }
      // Being written from these words: wait for it. Else write it now.
      const ahead = d && fits(d, text) && d.inFlight;
      const id = ahead ? d.id : `${ctx.id}-bc${n++}`;
      ctx.emit('text', { type: 'bc.prep', id });
      if (!ahead) askFor(id, text, false);
      const timer = ctx.setTimer(Math.max(0, Number(p.waitMs)), () => fallback(`the LLM took over ${p.waitMs} ms`), 'bc-wait');
      pending = { id, why, data, t0: ctx.now(), prevBc, timer };
    };
    /** Stop waiting for the LLM's line: the request and the socket are closed. */
    const dropPending = (why: string) => {
      const q = pending;
      if (!q) return;
      pending = undefined;
      q.timer.clear();
      ctx.emit('ask', { type: 'bc.drop', id: q.id, why });
      ctx.emit('text', { type: 'bc.drop', id: q.id, why });
      return q;
    };
    /** A canned line in place of the LLM's. */
    const fallback = (reason: string, extra: JsonObject = {}) => {
      const q = dropPending(reason);
      if (!q) return;
      const line = fallbacks.length ? fallbacks[fallbackNo++ % fallbacks.length] : null;
      if (!line) return void ctx.emit('text', { type: 'bc.skip', why: q.why, reason, ...q.data, ...extra });
      say(line, q.why, { ...q.data, ...extra, fallback: reason });
    };
    /** The LLM's line: said at once on the socket TTS opened, unless it isn't a backchannel. */
    const written = (msg: JsonObject) => {
      const q = pending;
      if (!q || msg.id !== q.id) return;
      const raw = String(msg.text ?? '');
      const ms = Math.round(ctx.now() - q.t0);
      const line = msg.status === 'done' ? cleanBackchannel(raw, maxWords()) : null;
      if (!line) return fallback(msg.status === 'done' ? `not a backchannel: "${raw}"` : `the LLM failed (${String(msg.status)})`, { raw, ms });
      pending = undefined;
      q.timer.clear();
      said.add(q.id);
      say(line, q.why, { ...q.data, gen: true, raw, ms, askedAt: q.t0 }, q.id);
    };

    const endTurn = () => {
      if (pending) {
        const q = pending;
        dropPending('the turn ended');
        lastBc = q.prevBc;
        ctx.emit('text', { type: 'bc.skip', why: q.why, reason: 'moment passed: the turn ended', ...q.data });
      }
      // Said, but not heard yet: too late now.
      for (const id of said) ctx.emit('text', { type: 'bc.drop', id, why: 'the turn ended', unheard: true });
      said.clear();
      if (draft?.inFlight) ctx.emit('ask', { type: 'bc.drop', id: draft.id, why: 'the turn ended' });
      draft = undefined;
      turnNo++;
      turnStart = null;
      silentSince = null;
      speechMs = 0;
      speechFrom = null;
      finals = partial = '';
      draftOut = false;
      asrDone = false;
      pause?.clear();
    };

    return {
      start() {
        const all = gen ? fallbacks : typesafe ? optionLines(opts) : allLines(lines);
        if (!all.length) return;
        ctx.emit('text', { type: 'bc.lines', lines: all.flatMap((l) => takes(l, p.takes)) });
        if (lines.examples.length)
          embed(
            lines.examples.map((e) => e.text),
            (vecs) => {
              if (vecs && vecs.length === lines.examples.length) exampleVecs = vecs;
              ctx.setState('encoder', exampleVecs ? 'ready' : 'unavailable');
            },
          );
      },
      onInput(port, msg) {
        const now = ctx.now();
        if (port === 'turn') {
          if (msg.type === 'turn.transition') {
            floor = String(msg.to).replace('+user', '');
            if (floor !== 'listening') endTurn();
            else if (vadSpeech && turnStart === null) {
              // They took the floor by talking over the agent: their turn started with that speech.
              turnStart = now;
              speechFrom = now;
            }
          } else if (msg.type === 'turn.draft') draftOut = true;
          else if (msg.type === 'turn.draft_drop') draftOut = false;
          return;
        }
        if (port === 'vad') {
          if (msg.type === 'vad.speech_start') {
            vadSpeech = true;
            pause?.clear();
            if (floor !== 'listening') return;
            turnStart ??= now;
            speechFrom = now;
            silentSince = null;
            asrDone = false;
          } else if (msg.type === 'vad.speech_end') {
            vadSpeech = false;
            if (speechFrom !== null) speechMs += now - speechFrom;
            speechFrom = null;
            silentSince = now;
            if (p.source !== 'pause') return;
            pause?.clear();
            pause = ctx.setTimer(Math.max(0, Number(p.pauseMs)), () => propose('pause'), 'bc-pause');
          }
          return;
        }
        if (port === 'scores') {
          if (msg.type === 'turnmodel.frame') eotUser = (msg.eot as number[] | undefined)?.[0] ?? 0;
          else if (msg.type === 'turnmodel.bc' && p.source === 'turn model') propose('turn model', { p: msg.p as Json });
          return;
        }
        if (port === 'llm') {
          if (msg.type === 'llm.bc') {
            if (msg.id === draft?.id) drafted(msg as JsonObject);
            else written(msg as JsonObject);
          }
          return;
        }
        if (port === 'transcript' && turnStart !== null) {
          if (msg.type === 'asr.final') {
            finals = `${finals} ${String(msg.text ?? '')}`.trim();
            partial = '';
            if (msg.speechFinal && silentSince !== null) asrDone = true;
            speculate();
          } else if (msg.type === 'asr.partial') {
            partial = String(msg.text ?? '');
            speculate();
          }
        }
      },
      stop() {
        pause?.clear();
        pending?.timer.clear();
      },
    };
  },
};

/** No backchannels. */
export const noBackchannelNode: NodeDef<Record<string, never>> = {
  type: 'backchannel.off',
  title: 'No backchannels',
  category: 'control',
  description: 'The agent stays silent while the user talks.',
  inputs: backchannelNode.inputs,
  outputs: backchannelNode.outputs,
  params: [],
  create() {
    return { onInput() {} };
  },
};
