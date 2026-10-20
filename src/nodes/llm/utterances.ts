/**
 * Tagged replies: the model writes its reply as lines, one utterance (a sentence or a clause) per
 * line, each starting with a tag of its flags: `[i1g0] Text of the utterance.`
 *
 * - i (interruptible): 1 if the user's speech may cut it off, 0 if it plays to its end.
 * - g (gated): 1 if it is subject to the quick yield (the agent stops when the user starts talking
 *   right as it begins), 0 if it is said regardless.
 *
 * The flags come before the words, so the pipeline knows how to treat the audio before any of it
 * is spoken. A tag is 6 tokens for Gemma; JSON objects cost twice that before the first word (see
 * the format bench in the commit). The history keeps the raw text, so the prompt prefix stays exact.
 */

export interface UtteranceFlags {
  interruptible: boolean;
  gate: boolean;
}

export const DEFAULT_FLAGS: UtteranceFlags = { interruptible: true, gate: true };

/** Told to the model once, in the system prompt. */
export const TAG_INSTRUCTIONS =
  'Write your reply as lines, one per sentence or clause, each starting with a tag: [i1g1] text. ' +
  'i: 1 if the listener may cut this line off by talking, 0 if it must be heard to the end (a number, a warning, a very short answer). ' +
  'g: 1 if you should stop when the listener starts talking right as you begin, 0 to say it regardless (a short acknowledgement).';

/** Written for the model at the start of each reply: the first tag's opening, so it only writes the flags. */
export const TAG_PREFILL = '[i';

const TAG = ['"["', '"i"', '[01]', '"g"', '[01]', '"]"', '" "'];
const RULES = 'line ::= "[i" [01] "g" [01] "] " t\nt ::= [^\\n\\[]+\n';

/**
 * The GBNF grammar for what the model writes after `prefix` (the reply so far: a tagged text, maybe
 * cut anywhere): the rest of the line it is in, then more lines.
 */
export function tagGrammar(prefix: string): string {
  const line = prefix.slice(prefix.lastIndexOf('\n') + 1);
  const rest = '("\\n" line)*';
  if (prefix.endsWith('\n') || !prefix) return `root ::= line ${rest}\n${RULES}`;
  // In a tag: its remaining parts, then the words.
  let k = 0;
  while (k < line.length && k < TAG.length && matchesPart(line[k], TAG[k])) k++;
  if (k === line.length && k < TAG.length) return `root ::= ${TAG.slice(k).join(' ')} t ${rest}\n${RULES}`;
  // In the words (or a line without a tag, e.g. a kept interjection): more words, maybe none.
  return `root ::= [^\\n\\[]* ${rest}\n${RULES}`;
}

function matchesPart(c: string, part: string): boolean {
  return part === '[01]' ? c === '0' || c === '1' : part === `"${c}"`;
}

export type UtteranceEvent = { type: 'utterance'; flags: UtteranceFlags; tagged: boolean } | { type: 'text'; text: string };

const isTagPrefix = (s: string) => /^(?:\[(?:i(?:[01](?:g(?:[01](?:\](?: )?)?)?)?)?)?)?$/.test(s);
const FULL = /^\[i([01])g([01])\]/;

/**
 * Streams a tagged reply: each line's tag becomes an `utterance` event (before its words), its words
 * `text` events, and a line break a space. Text without a tag is an utterance with the default flags.
 */
export class UtteranceParser {
  private head = '';
  private lineStart = true;
  private any = false;

  feed(chunk: string): UtteranceEvent[] {
    const out: UtteranceEvent[] = [];
    let text = '';
    const flushText = () => {
      if (text) out.push({ type: 'text', text });
      text = '';
    };
    const startLine = (flags: UtteranceFlags, tagged: boolean) => {
      flushText();
      // A space between utterances, so their words stay apart.
      if (this.any) text = ' ';
      out.push({ type: 'utterance', flags, tagged });
      this.any = true;
      this.lineStart = false;
    };
    for (const c of chunk) {
      if (c === '\n') {
        if (this.lineStart && this.head) {
          startLine(DEFAULT_FLAGS, false);
          text += this.head;
          this.head = '';
        }
        this.lineStart = true;
        continue;
      }
      if (!this.lineStart) {
        text += c;
        continue;
      }
      const h = this.head + c;
      if (h.length === 7 && FULL.test(h)) {
        // "[i1g0]" and the space after it.
        const m = FULL.exec(h)!;
        this.head = '';
        startLine({ interruptible: m[1] === '1', gate: m[2] === '1' }, true);
        if (c !== ' ') text += c;
        continue;
      }
      if (isTagPrefix(h)) {
        this.head = h;
        continue;
      }
      // Not a tag after all.
      if (FULL.test(this.head)) {
        const m = FULL.exec(this.head)!;
        this.head = '';
        startLine({ interruptible: m[1] === '1', gate: m[2] === '1' }, true);
        text += c;
        continue;
      }
      startLine(DEFAULT_FLAGS, false);
      text += h;
      this.head = '';
    }
    flushText();
    return out;
  }
}

/** The words of a (maybe tagged) text, without the tags. */
export function visibleText(raw: string): string {
  const p = new UtteranceParser();
  return p
    .feed(`${raw}\n`)
    .flatMap((e) => (e.type === 'text' ? [e.text] : []))
    .join('');
}

const words = (s: string) => s.split(/\s+/).filter(Boolean);

/** The number of words a (maybe tagged) text has, tags not counted. */
export const visibleWordCount = (raw: string) => words(visibleText(raw)).length;

/** `raw` cut after its `k`th word (tags not counted), each line's tag kept with its words. */
export function rawPrefix(raw: string, k: number): string {
  const out: string[] = [];
  let left = k;
  for (const line of raw.split('\n')) {
    if (left <= 0) break;
    const m = /^\[i[01]g[01]\] ?/.exec(line);
    const tag = m ? m[0].trimEnd() : '';
    const ws = words(m ? line.slice(m[0].length) : line);
    if (!ws.length) continue;
    const take = ws.slice(0, left);
    left -= take.length;
    out.push(tag ? `${tag} ${take.join(' ')}` : take.join(' '));
  }
  return out.join('\n');
}
