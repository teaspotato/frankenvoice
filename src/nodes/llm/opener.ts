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

// TypeSafe Choice: a hosted classifier that reads the whole conversation and picks one option.

export const DEFAULT_OPENER_CHOICES = `# One option per line: what to say = when it fits. "-" means say nothing.
Hmm, let me think. = The user asked for an opinion, a judgement or a recommendation that genuinely needs weighing up before answering.
Let me see. = The user asked for concrete information, steps or a plan that take a moment to put together.
Oh, good question. = The user asked how or why something works, and the question is a thoughtful one.
Oh? = The user hinted at something or started a story without giving the details yet; the agent should invite them to go on.
- = Anything else: small talk, greetings, thanks, agreement, acknowledgements, or anything a quick direct reply answers well. Also pick this if the user hasn't actually said anything that needs a response yet.`;

export const DEFAULT_CHOICE_INSTRUCTIONS =
  'This is a live spoken conversation between a user and a voice agent. The agent is about to reply to the last User line and can say a few words at once, while its full answer is still being prepared. Pick the opening words that a thoughtful person would naturally say first, given the whole conversation so far. The opening must fit whatever the agent says next and must not promise more thought than the reply needs. When in doubt, pick "-".';

export interface OpenerChoice {
  /** null: say nothing. */
  line: string | null;
  description: string;
}

export function parseOpenerChoices(src: string): OpenerChoice[] {
  const out: OpenerChoice[] = [];
  for (const raw of src.split('\n')) {
    const s = raw.trim();
    if (!s || s.startsWith('#')) continue;
    const i = s.indexOf('=');
    if (i < 0) continue;
    const say = s.slice(0, i).trim();
    out.push({ line: say === '-' || !say ? null : say, description: s.slice(i + 1).trim() });
  }
  return out;
}

const NOTHING = '-';

/** The request body for /v1/systemone: the conversation as `state`, one Choice question. */
export function choiceRequest(turns: { role: string; text: string }[], choices: OpenerChoice[], instructions: string, model: string) {
  const state = turns.map((t) => `${t.role === 'user' ? 'User' : 'Agent'}: ${t.text}`).join('\n');
  return {
    model,
    state,
    questions: {
      opener: {
        type: 'choice',
        instructions,
        criteria: Object.fromEntries(choices.map((c) => [c.line ?? NOTHING, c.description])),
      },
    },
  };
}

/** The pick from a /v1/systemone answer; below `minConfidence`, say nothing. */
export function readChoice(res: unknown, minConfidence: number): OpenerPick | null {
  const a = (res as { answers?: { opener?: { choice?: string; confidence?: number } } } | null)?.answers?.opener;
  if (!a || typeof a.choice !== 'string') return null;
  const score = Math.round((a.confidence ?? 0) * 1000) / 1000;
  const line = a.choice === NOTHING || score < minConfidence ? null : a.choice;
  return { line, score, example: a.choice };
}
