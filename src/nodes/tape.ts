/**
 * The tape: a zero-shot "full-duplex" text LLM as the turn-taking layer (research: duplex-text-llm/pilot).
 *
 * Time runs in cells (640 ms). Each cell is one line of a tape:
 *
 *   ` 12.16 | user: "car?" (silent) | agent (cut off): respond "By car, it's a"`
 *
 * The user column is what the ASR committed in the cell and whether the VAD heard the user; the agent
 * column is the agent's state (silent, speaking, cut off) and what it did. The pipeline writes each
 * line up to `agent (<state>):` and Gemma writes the rest under a GBNF grammar that allows only that
 * state's actions and as many words as the TTS queue has room for. History is what was heard: when the
 * user cuts the agent off, the words it wrote but never said leave the earlier lines and `[cut]` marks
 * where its voice stopped.
 *
 * This file holds the format (a port of the pilot's tape.py / pilot.py); the node is in turn-tape.ts.
 */

export type TapeState = 'silent' | 'speaking' | 'cut';

export const ACTIONS: Record<TapeState, string[]> = {
  silent: ['listen', 'respond', 'backchannel'],
  speaking: ['continue', 'finish'],
  cut: ['listen', 'resume', 'respond'],
};
export const STATE_TEXT: Record<TapeState, string> = { silent: 'silent', speaking: 'speaking', cut: 'cut off' };

export const AGENT_BACKCHANNELS = ['mhm', 'yeah', 'right', 'okay', 'uh-huh', 'I see'];
/** A word's average length with its pause, for the TTS headroom. */
export const WORD_S = 0.3;

export interface Cell {
  /** Absolute cell number (0 = the tape's first). */
  n: number;
  /** End of the cell's window, in seconds from the tape's start: when the agent decides. */
  t: number;
  /** Words the ASR committed in the window. */
  user: string[];
  /** The VAD heard the user in the window. */
  voice: boolean;
  state: TapeState;
  action: string;
  /** Sent to TTS and said (or still to be said). */
  say: string[];
  /** Sent to TTS, then cut before they were said: not on the tape. */
  unsaid: string[];
  /** The agent's voice was cut during this cell's words. */
  cut: boolean;
}

export const newCell = (n: number, t: number, state: TapeState, user: string[] = [], voice = false): Cell => ({
  n,
  t,
  user,
  voice,
  state,
  action: ACTIONS[state][0],
  say: [],
  unsaid: [],
  cut: false,
});

export function userText(c: Cell): string {
  const words = c.user.length ? `"${c.user.join(' ').replace(/"/g, "'")}" ` : '';
  return `${words}(${c.voice ? 'talking' : 'silent'})`;
}

/** The agent's action as heard, e.g. `continue "but they [cut]"`. */
export function agentText(c: Cell): string {
  const words = [...c.say, ...(c.cut ? ['[cut]'] : [])];
  return c.action + (words.length ? ` "${words.join(' ')}"` : '');
}

/** A line up to the agent's action: what the agent has in front of it when it decides. */
export const prefix = (c: Cell) => `${c.t.toFixed(2).padStart(6)} | user: ${userText(c)} | agent (${STATE_TEXT[c.state]}):`;

export const line = (c: Cell) => `${prefix(c)} ${agentText(c)}`;

/** GBNF for one cell's action: the state's actions, at most `n` (>= 1) words. */
export function grammar(allowed: string[], n: number): string {
  const alts = allowed.map((a) =>
    a === 'listen' ? '" listen"' : a === 'backchannel' ? '" backchannel " bc' : a === 'finish' ? '" finish" (" " say)?' : `" ${a} " say`,
  );
  const rep = n > 1 ? `{0,${n - 1}}` : '{0,0}';
  return [
    `root ::= ${alts.join(' | ')}`,
    `say ::= "\\"" word (" " word)${rep} "\\""`,
    "word ::= [A-Za-z0-9$%&'] [A-Za-z0-9$%&'-]{0,18} [,.?!;:]?",
    `bc ::= ${AGENT_BACKCHANNELS.map((b) => `"\\"${b}\\""`).join(' | ')}`,
  ].join('\n');
}

/** One token's alternatives as llama.cpp's `/completion` returns them with `n_probs` (before sampling). */
interface TopToken {
  token: string;
  logprob: number;
}

/**
 * Gemma's odds for each allowed action: the probability of the token it starts with, among the first
 * written token's top alternatives (`n_probs`, taken before logit bias and grammar). An action whose
 * first token isn't among them gets 0. Null when the answer carries no probabilities.
 */
export function actionOdds(res: { completion_probabilities?: unknown } | undefined, allowed: string[]): Record<string, number> | null {
  const first = Array.isArray(res?.completion_probabilities) ? (res.completion_probabilities[0] as { top_logprobs?: TopToken[] }) : null;
  const top = first?.top_logprobs;
  if (!Array.isArray(top)) return null;
  const odds: Record<string, number> = {};
  for (const a of allowed) {
    const word = ` ${a}`;
    // The longest top token the action's word starts with (" respond", or " back" for " backchannel").
    let best: TopToken | null = null;
    for (const t of top) if (t.token.trim() && word.startsWith(t.token) && (!best || t.token.length > best.token.length)) best = t;
    odds[a] = best ? Math.round(Math.exp(best.logprob) * 1e4) / 1e4 : 0;
  }
  return odds;
}

export function parseAction(text: string): { action: string; words: string[] } | null {
  const m = /^\s*(\w+)(?:\s+"([^"]*)")?/.exec(text);
  if (!m) return null;
  return { action: m[1], words: (m[2] ?? '').split(/\s+/).filter(Boolean) };
}

/**
 * How many words the TTS queue has room for: two cells ahead of where the voice will be, at
 * `WORD_S` a word. `queueEnd`: when the words already sent will have been said (seconds, same clock as `now`).
 */
export function headroom(now: number, queueEnd: number, leadS: number, cellS: number, maxWords: number): number {
  const start = Math.max(now + leadS, queueEnd);
  const budget = now + leadS + 2 * cellS - start;
  return Math.max(0, Math.min(maxWords, Math.round(budget / WORD_S + 1e-9)));
}

/** A word of the agent's answer as sent: the cell that sent it. */
export interface SentWord {
  n: number;
  word: string;
}

/**
 * The user cut the agent off after it had said the first `heard` words of its answer (`sent`):
 * the rest leave the cells that sent them (as `unsaid`), and `[cut]` goes on the cell of the last
 * word heard, or of the first word sent if none was. Cells no longer on the tape are skipped.
 * Returns the words taken off.
 */
export function cutTape(cells: Cell[], sent: SentWord[], heard: number): string[] {
  const k = Math.max(0, Math.min(heard, sent.length));
  const byN = new Map(cells.map((c) => [c.n, c]));
  const gone = sent.slice(k);
  // From the end, so each cell loses its last words first.
  for (const w of [...gone].reverse()) {
    const c = byN.get(w.n);
    if (!c) continue;
    const i = c.say.lastIndexOf(w.word);
    if (i >= 0) c.say.splice(i, 1);
    c.unsaid.unshift(w.word);
  }
  const mark = k > 0 ? sent[k - 1] : sent[0];
  const c = mark && byN.get(mark.n);
  if (c) c.cut = true;
  return gone.map((w) => w.word);
}

const SYSTEM_HEAD = `You are the turn-taking layer of a voice assistant on a live call. You hear the user through a speech recognizer and talk through a text-to-speech voice. Time runs in cells of 0.64 seconds, and you act once per cell.

Each line of the tape is one cell:
  <time> | user: "<words>" (talking or silent) | agent (<state>): <action>
- user: the words the speech recognizer confirmed in that cell (they arrive about half a second after the user says them), and whether the user was making sound.
- agent state: silent (you don't have the floor), speaking (you are partway through an answer), cut off (the user started talking over you and your voice was stopped).
- Your words are spoken aloud 0.4 s after you write them, a few words per cell, so carry your sentence on in the next cell.
- [cut] marks where your voice was stopped. Words you wrote after it were never heard, so they are not on the tape.

Actions:
  listen               stay quiet: the user is talking, pausing mid-thought, or you are waiting
  respond "<words>"    start your answer to what the user just said
  backchannel "mhm"    a short acknowledgement while the user keeps the floor
  continue "<words>"   the next words of your answer
  finish "<words>"     the last words of your answer, or just finish when you have said them
  resume "<words>"     go on with an answer the user cut off only to acknowledge you (mhm, yeah, okay)

`;

const TAKE_TURNS = `Take turns like a person:
- Respond once the user has finished a thought: a question or a complete sentence, then silence. Don't start while they are talking, or while they pause in the middle of a sentence (after "and", "um", "the", a comma). Wait for them.
- Once they are done, respond promptly, usually in the first silent cell after their last words.
- If a brief acknowledgement cut you off, resume from where you were cut. If the user is taking the floor (a question, a correction, "wait"), listen until they finish, then respond to what they said.
- Talk like a person on the phone: one or two short sentences, no lists.`;

/** The turn-end rules rewritten against dead air (the base prompt's agent waits about two extra cells). */
const TAKE_TURNS_EAGER = `Take turns like a person, without dead air:
- The speech recognizer is about half a second behind the user. When a line brings the end of a question or a complete sentence, the user has usually stopped already, even if that line still says (talking).
- So respond in that cell, or at the latest in the next one. Every cell you wait is 0.64 s of silence the user sits through, and people find a second of silence after a question slow.
- Don't start while a sentence is plainly unfinished: it ends in "and", "um", "the" or a comma, or more words are still coming. Wait for the rest.
- If a brief acknowledgement cut you off, resume from where you were cut. If the user is taking the floor (a question, a correction, "wait"), listen until they finish, then respond to what they said just as promptly.
- Talk like a person on the phone: one or two short sentences, no lists.`;

export const PROMPTS: Record<string, string> = { base: SYSTEM_HEAD + TAKE_TURNS, eager: SYSTEM_HEAD + TAKE_TURNS_EAGER };

/** The pilot's example tape (written by its rules agent on another call), without periods on the user's words as the live tape has none. */
export const EXAMPLE_TAPE = `  0.64 | user: (silent) | agent (silent): listen
  1.28 | user: (talking) | agent (silent): listen
  1.92 | user: "hey, do you" (talking) | agent (silent): listen
  2.56 | user: "know if the" (talking) | agent (silent): listen
  3.20 | user: "pharmacy" (talking) | agent (silent): listen
  3.84 | user: "on Main Street" (talking) | agent (silent): listen
  4.48 | user: "is," (talking) | agent (silent): listen
  5.12 | user: "um," (silent) | agent (silent): listen
  5.76 | user: (silent) | agent (silent): listen
  6.40 | user: (silent) | agent (silent): backchannel "mhm"
  7.04 | user: (talking) | agent (silent): listen
  7.68 | user: "open on" (talking) | agent (silent): listen
  8.32 | user: "Sundays?" (talking) | agent (silent): listen
  8.96 | user: (silent) | agent (silent): respond "It is, from ten"
  9.60 | user: (silent) | agent (speaking): continue "to four, but [cut]"
 10.24 | user: (silent) | agent (speaking): continue
 10.88 | user: (silent) | agent (speaking): continue
 11.52 | user: (talking) | agent (cut off): listen
 12.16 | user: "yeah" (silent) | agent (cut off): resume "it closes early on [cut]"
 12.80 | user: (silent) | agent (speaking): finish
 13.44 | user: (silent) | agent (silent): listen
 14.08 | user: (talking) | agent (cut off): listen
 14.72 | user: "oh" (talking) | agent (cut off): listen
 15.36 | user: "wait, I" (talking) | agent (cut off): listen
 16.00 | user: "meant the one" (talking) | agent (cut off): listen
 16.64 | user: "on Oak Street" (talking) | agent (cut off): listen
 17.28 | user: (silent) | agent (cut off): respond "Ah, the Oak Street"
 17.92 | user: (silent) | agent (speaking): continue "one is"
 18.56 | user: (silent) | agent (speaking): finish "closed on Sundays."`;

/** Seconds as the prompt writes them: 0.64, 0.96, 1.2, 1. */
const secs = (x: number) => String(Number(x.toFixed(2)));

/**
 * The prompt's cell length follows the tape's time: the instructions' "0.64 seconds" / "0.64 s" become
 * `cellS`, and the example tape (written on 0.64 s cells) has its timestamps stretched to match.
 */
export function timed(system: string, example: string, cellS: number): { system: string; example: string } {
  const k = cellS / 0.64;
  return {
    system: system.replace(/\b0\.64(?= s\b| seconds\b)/g, secs(cellS)),
    example: example.replace(/^( *)(\d+\.\d+)(?= \|)/gm, (_m, _sp, t: string) => (Number(t) * k).toFixed(2).padStart(6)),
  };
}

/**
 * The prompt's fixed head, in Gemma 4's chat format: instructions as the system turn, the example tape
 * as the user turn, then the model's own turn with thinking off. `earlier`: what was said before the
 * lines still on the tape (a plain transcript), when the tape has been shortened.
 */
export function promptHead(system: string, example: string, earlier = ''): string {
  const before = earlier ? ` What was said before the tape below:\n\n${earlier}\n\n` : ' ';
  return (
    `<|turn>system\n${system}<turn|>\n<|turn>user\nAn example tape from another call:\n\n${example}\n\n` +
    `Now the live call.${before}Continue the tape with the agent's action on each new line.<turn|>\n` +
    '<|turn>model\n<|channel>thought\n<channel|>'
  );
}

/** A cell where nothing happened: the user silent with no words, the agent silent and listening. */
const idleCell = (c: Cell) => c.state === 'silent' && !c.voice && !c.user.length && c.action === 'listen' && !c.say.length;

/**
 * The tape as Gemma reads it: each finished run of idle cells cut to its first `keep` lines. A tape
 * full of listen lines makes Gemma copy them and stop answering (pilot/stall-replay: P(respond) after
 * a finished question went 0.10, 0.06, 0.01 as idle lines piled up, and back to 0.6-0.86 trimmed to 2).
 * The run leading up to the cell being decided stays whole. keep <= 0: every line.
 */
export function trimIdle(cells: Cell[], keep: number): Cell[] {
  if (keep <= 0 || cells.length < 2) return cells;
  const out: Cell[] = [];
  let run: Cell[] = [];
  for (const c of cells.slice(0, -1)) {
    if (idleCell(c)) {
      run.push(c);
      continue;
    }
    out.push(...run.slice(0, keep), c);
    run = [];
  }
  return [...out, ...run, cells[cells.length - 1]];
}

/** The whole prompt for the last cell: the head, every earlier line, then the open field. */
export function tapePrompt(head: string, cells: Cell[]): string {
  return head + cells.slice(0, -1).map((c) => `${line(c)}\n`).join('') + prefix(cells[cells.length - 1]);
}

/** Cells as a plain transcript ("user: ..." / "agent: ..."), what was heard, consecutive lines merged. */
export function transcript(cells: Cell[]): string[] {
  const out: { who: string; text: string }[] = [];
  const add = (who: string, words: string[]) => {
    if (!words.length) return;
    const last = out[out.length - 1];
    if (last?.who === who) last.text += ` ${words.join(' ')}`;
    else out.push({ who, text: words.join(' ') });
  };
  for (const c of cells) {
    add('user', c.user);
    add('agent', c.action === 'backchannel' ? [] : [...c.say, ...(c.cut ? ['—'] : [])]);
  }
  return out.map((x) => `${x.who}: ${x.text}`);
}
