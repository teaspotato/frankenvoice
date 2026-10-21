/**
 * The voice agent's log (agent.loop, System 1). The conversation is one append-only log: what
 * happened comes in as event lines (the user's words, a cut-off, System 2's news, silences, its own
 * wake-ups), one compact timestamped line each, and every think step writes its decision as lines
 * after them. The prompt is the log as chat turns (events as user turns, decisions as the model's),
 * so llama.cpp reuses the cached prefix: earlier lines are never rewritten, except that a decision
 * the user cut off keeps only the words they heard (as the reply LLM's history does).
 *
 * A decision, under a GBNF grammar (JSON costs about twice as many forced tokens on llama.cpp
 * b11753, which has no jump-forward), is any number of action lines and then one ending line:
 *
 *   [i1g1] Sure, let me check.     say this (the flags of tagged utterances: interruptible, gated)
 *   [think] Kyoto weather tomorrow  hand a task to System 2: a new worker, w1, w2, ... in order
 *   [tell w1] Make it Osaka.        a message for worker w1 (a follow-up once it is done)
 *   [stop w1]                       stop worker w1
 *   [wait 20]                       end the step; wake up in 20 s unless something happens first
 *   [silent]                        end the step; nothing more to do now
 *   [quiet]                         end the step and stay quiet: only the user's own words wake it
 */

/** Seconds as the log writes them: one decimal. */
export const stamp = (ms: number) => `${(Math.max(0, ms) / 1000).toFixed(1)}s`;

export type ActionLine =
  | { kind: 'say'; interruptible: boolean; gate: boolean; text: string }
  | { kind: 'think'; text: string }
  | { kind: 'tell'; worker: string; text: string }
  | { kind: 'stop'; worker: string }
  | { kind: 'wait'; s: number }
  | { kind: 'silent' }
  | { kind: 'quiet' };

/** The grammar of one decision. */
export const DECISION_GRAMMAR = [
  'root ::= (act "\\n")* end',
  'act ::= say | think | tell | stop',
  'say ::= "[i" [01] "g" [01] "] " t',
  'think ::= "[think] " t',
  'tell ::= "[tell w" [1-9] [0-9]? "] " t',
  'stop ::= "[stop w" [1-9] [0-9]? "]"',
  'end ::= "[wait " [1-9] [0-9]? [0-9]? "]" | "[silent]" | "[quiet]"',
  't ::= [^\\n\\[]+',
].join('\n');

/** One whole line of a decision, or null if it isn't one. */
export function parseLine(line: string): ActionLine | null {
  let m = /^\[i([01])g([01])\] ?(.*)$/.exec(line);
  if (m) return { kind: 'say', interruptible: m[1] === '1', gate: m[2] === '1', text: m[3].trim() };
  if ((m = /^\[think\] ?(.*)$/.exec(line))) return { kind: 'think', text: m[1].trim() };
  if ((m = /^\[tell (w\d+)\] ?(.*)$/.exec(line))) return { kind: 'tell', worker: m[1], text: m[2].trim() };
  if ((m = /^\[stop (w\d+)\]$/.exec(line))) return { kind: 'stop', worker: m[1] };
  if ((m = /^\[wait (\d+)\]$/.exec(line))) return { kind: 'wait', s: Number(m[1]) };
  if (line === '[silent]') return { kind: 'silent' };
  if (line === '[quiet]') return { kind: 'quiet' };
  return null;
}

export type DecisionEvent =
  /** A say line's tag is complete: its words follow as `words`. */
  | { type: 'say'; interruptible: boolean; gate: boolean }
  | { type: 'words'; text: string }
  /** A line is complete. */
  | { type: 'line'; line: ActionLine; raw: string }
  /** A line that is not in the format (no grammar, or a model that ignored it). */
  | { type: 'junk'; raw: string };

const SAY_TAG = /^\[i([01])g([01])\] /;
const isSayPrefix = (s: string) => /^(?:\[(?:i(?:[01](?:g(?:[01](?:\](?: )?)?)?)?)?)?)?$/.test(s);

/**
 * Reads a decision as it streams in. A say line's words stream out as soon as its tag is known;
 * every other line acts once it is complete.
 */
export class DecisionParser {
  private line = '';
  /** The current line is a say line whose words are streaming. */
  private saying = false;

  feed(chunk: string): DecisionEvent[] {
    const out: DecisionEvent[] = [];
    let words = '';
    const flushWords = () => {
      if (words) out.push({ type: 'words', text: words });
      words = '';
    };
    for (const c of chunk) {
      if (c === '\n') {
        flushWords();
        this.endLine(out);
        continue;
      }
      this.line += c;
      if (this.saying) {
        words += c;
        continue;
      }
      if (this.line.length === 7 && SAY_TAG.test(this.line)) {
        const m = SAY_TAG.exec(this.line)!;
        this.saying = true;
        out.push({ type: 'say', interruptible: m[1] === '1', gate: m[2] === '1' });
      } else if (this.line.length > 7 && !isSayPrefix(this.line) && /^\[i[01]g[01]\]\S/.test(this.line)) {
        // A tag without its space: the words start right after it.
        this.saying = true;
        const m = /^\[i([01])g([01])\]/.exec(this.line)!;
        out.push({ type: 'say', interruptible: m[1] === '1', gate: m[2] === '1' });
        words += this.line.slice(6);
      }
    }
    flushWords();
    return out;
  }

  /** The stream ended: a last line without its newline still counts. */
  end(): DecisionEvent[] {
    const out: DecisionEvent[] = [];
    if (this.line) this.endLine(out);
    return out;
  }

  private endLine(out: DecisionEvent[]) {
    const raw = this.line;
    this.line = '';
    this.saying = false;
    if (!raw.trim()) return;
    const line = parseLine(raw.trim());
    if (line) out.push({ type: 'line', line, raw: raw.trim() });
    else out.push({ type: 'junk', raw });
  }

  /** The line being written, not complete yet. */
  get partial(): string {
    return this.line;
  }
}

const words = (s: string) => s.split(/\s+/).filter(Boolean);

/**
 * A decision's text as the user heard it: say lines keep only the first `heard` words of all of
 * them (a line cut short ends in "—"; later ones go), every other line stays.
 */
export function cutDecision(raw: string, heard: number): string {
  let left = heard;
  let cut = false;
  const out: string[] = [];
  for (const line of raw.split('\n')) {
    const m = /^(\[i[01]g[01]\]) ?(.*)$/.exec(line);
    if (!m) {
      out.push(line);
      continue;
    }
    if (cut) continue;
    const ws = words(m[2]);
    if (ws.length <= left) {
      left -= ws.length;
      out.push(line);
      continue;
    }
    cut = true;
    if (left > 0) out.push(`${m[1]} ${ws.slice(0, left).join(' ')}—`);
    left = 0;
  }
  return out.join('\n');
}

/** The words a decision says, without tags. */
export const saidWords = (raw: string) =>
  raw
    .split('\n')
    .flatMap((l) => {
      const m = /^\[i[01]g[01]\] ?(.*)$/.exec(l);
      return m ? [m[1]] : [];
    })
    .join(' ')
    .trim();

export const DEFAULT_AGENT_SYSTEM = `You are the voice of an assistant on a live call: you hear the user through a speech recognizer and talk through a text-to-speech voice. You are fast and you run on your own clock. You don't wait for the user to speak first, you don't have to answer everything at once, and you never make things up. For anything you would have to look up, work out or think through, hand it to your System 2 workers: slower agents that can search and read the web, reason carefully and do work. Keep the user company while they work, and tell the user what they find when it is useful, without being asked.

The log: every line I send you is something that happened, with the time since the call started:
  user: ...             the user finished saying this (speech recognition, may have errors)
  user (went on): ...   they added this to what they said before
  user cut in ...       the user started talking over you; your words they never heard are gone from what you wrote
  w1 progress: ...      a worker's latest step (for you: tell the user only if it helps)
  w1 result: ...        a worker finished (tell the user what matters, in a sentence or two)
  w1 asks the user: ... a worker needs something only the user knows (ask them; give the answer with [tell w1])
  silence for 6 s       nobody has said anything for a while
  wake-up               the time you asked to be woken has come

Answer each time with lines like these, then one line that ends your step:
  [i1g1] text           say this, one sentence or clause per line
  [think] task          hand a task to a new worker, with everything it needs: it can't hear the user. Workers are numbered in order: w1, w2, ...
  [tell w1] message     tell a worker something (change its task, answer its question, or a follow-up)
  [stop w1]             stop a worker the user no longer needs
then one of:
  [wait 20]             end your step, and wake up in 20 s if nothing happens first
  [silent]              end your step: nothing to do now
  [quiet]               end your step and stay quiet: nothing but the user's own words wakes you

How to talk:
- Talk like a person on the phone: short spoken sentences, no lists, no markdown.
- When the user asks for something a worker should do, say a short line ("Let me look that up.") and [think] in the same step. Keep the conversation going while it works.
- When a result comes in, tell the user unprompted, but don't talk over them: if they are in the middle of something, wait for it.
- After a silence, you may say something useful or friendly (the worker's progress, a follow-up question), or stay [silent]. Don't fill every silence.
- If the user asks you to be quiet, stop talking or give them a moment, do it: say at most "Sure." and end with [quiet].
- Almost every line is [i1g1]. i0: the user can't cut this line off (only for something they must hear whole). g0: say it even if the user starts talking just as you begin (only for a one or two word acknowledgement).`;

/** A short form of a worker's task, for its event lines. */
export const taskLabel = (task: string) => {
  const t = task.replace(/\s+/g, ' ').trim();
  return t.length > 40 ? `${t.slice(0, 39)}…` : t;
};
