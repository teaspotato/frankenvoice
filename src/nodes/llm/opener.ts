/** Canned openers: lines the agent can say at once, each with example user turns it fits. */

export const DEFAULT_OPENER_LINES = `# One line per opener: what to say = example user turns it fits, separated by |
# "-" means say nothing. The nearest example (by meaning) picks the line.
Hmm, let me think. = what do you think about this | what do you think of it | why does that happen | which one would you pick | should I do it or not | what would you recommend | do you think it's a good idea | is it worth it
Let me see. = how do I make this | how many are there | when does it open | where can I find one | what's the best way to do it | can you give me a recipe | tell me how to do it | what should I cook tonight
Oh, good question. = how does that actually work | what's the difference between them | why is the sky blue | how come nobody knows | what makes it so special
Right, okay. = I was thinking about starting a business | so here's my situation | my friend told me something weird | I have a problem with my laptop | let me tell you about my day
- = thanks | thank you so much | okay | yeah | no | hi there | good morning | bye | talk to you later | mhmm | uh-huh | got it | sure | I see | cool | that sounds good | that's great`;

export interface OpenerExample {
  /** null: say nothing. */
  line: string | null;
  text: string;
}

export function parseOpenerLines(src: string): OpenerExample[] {
  const out: OpenerExample[] = [];
  for (const raw of src.split('\n')) {
    const s = raw.trim();
    if (!s || s.startsWith('#')) continue;
    const i = s.indexOf('=');
    if (i < 0) continue;
    const say = s.slice(0, i).trim();
    const line = say === '-' || !say ? null : say;
    for (const ex of s.slice(i + 1).split('|')) if (ex.trim()) out.push({ line, text: ex.trim() });
  }
  return out;
}

export interface OpenerPick {
  line: string | null;
  score: number;
  example: string;
}

/** The example nearest `v` (unit vectors, so the dot product is the cosine). Below `minScore`, say nothing. */
export function pickOpener(v: number[], examples: OpenerExample[], vecs: number[][], minScore: number): OpenerPick {
  let best = -1;
  let score = -Infinity;
  vecs.forEach((e, i) => {
    let s = 0;
    for (let k = 0; k < e.length; k++) s += e[k] * v[k];
    if (s > score) [best, score] = [i, s];
  });
  const ex = examples[best];
  if (!ex) return { line: null, score: 0, example: '' };
  return { line: score >= minScore ? ex.line : null, score: Math.round(score * 1000) / 1000, example: ex.text };
}
