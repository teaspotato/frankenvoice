// TypeSafe Choice (POST /v1/systemone): a hosted classifier that reads a text
// state and picks one of a few named options, with a probability for each.
// The opener and the backchannel stages use it to pick a line from the user's
// ASR words.

import type { NodeContext } from '../core/node';
import type { JsonObject } from '../core/types';

/** "-" as an option: say nothing. */
export const NOTHING = '-';

export interface ChoiceOption {
  /** The option's name sent to TypeSafe: the first line, or "-". */
  key: string;
  /** What can be said for it, taken in turn; empty for "-". */
  lines: string[];
  /** When it fits: the Choice criterion. */
  description: string;
}

/**
 * Options, one per line: `line | other line = when it fits`. The lines of one
 * option are said in turn, so it doesn't repeat itself; `-` says nothing.
 */
export function parseChoiceOptions(src: string): ChoiceOption[] {
  const out: ChoiceOption[] = [];
  for (const raw of src.split('\n')) {
    const s = raw.trim();
    if (!s || s.startsWith('#')) continue;
    const i = s.indexOf('=');
    if (i < 0) continue;
    const lines = s
      .slice(0, i)
      .split('|')
      .map((l) => l.trim())
      .filter(Boolean);
    const description = s.slice(i + 1).trim();
    if (!lines.length || !description) continue;
    const nothing = lines.length === 1 && lines[0] === NOTHING;
    out.push({ key: nothing ? NOTHING : lines[0], lines: nothing ? [] : lines, description });
  }
  return out;
}

/** Every line the options can say, so TTS can render them ahead. */
export const optionLines = (opts: ChoiceOption[]) => [...new Set(opts.flatMap((o) => o.lines))];

/** The request body for /v1/systemone: `state` is the text to judge, one Choice question. */
export function choiceRequest(state: string, opts: ChoiceOption[], instructions: string, model: string): JsonObject {
  return {
    model,
    state,
    questions: { pick: { type: 'choice', instructions, criteria: Object.fromEntries(opts.map((o) => [o.key, o.description])) } },
  };
}

export interface ChoicePick {
  /** The option picked, or null for nothing (picked "-", or not sure enough). */
  option: ChoiceOption | null;
  /** The option TypeSafe picked, whatever its confidence. */
  choice: string;
  confidence: number;
  probabilities: Record<string, number>;
}

/** The pick from a /v1/systemone answer; below `minConfidence`, nothing. Null when the answer has no pick. */
export function readChoice(res: unknown, opts: ChoiceOption[], minConfidence: number): ChoicePick | null {
  const a = (res as { answers?: { pick?: { choice?: string; confidence?: number; probabilities?: Record<string, number> } } } | null)?.answers?.pick;
  if (!a || typeof a.choice !== 'string') return null;
  const confidence = Math.round((a.confidence ?? 0) * 1000) / 1000;
  const opt = opts.find((o) => o.key === a.choice);
  return {
    option: opt && opt.key !== NOTHING && confidence >= minConfidence ? opt : null,
    choice: a.choice,
    confidence,
    probabilities: a.probabilities ?? {},
  };
}

/** Asks one Choice question on the 'typesafe' channel; `done` gets the answer, or null if the request failed. */
export function askChoice(ctx: Pick<NodeContext, 'io' | 'log'>, body: JsonObject, done: (res: unknown | null) => void) {
  let got = false;
  ctx.io.open(
    'typesafe',
    { path: '/v1/systemone', body },
    {
      onRecv(data) {
        if (got) return;
        got = true;
        done(data);
      },
      onClose(data) {
        if (got) return;
        got = true;
        if (data && (data as { error?: unknown }).error) ctx.log('warn', 'TypeSafe choice failed', data);
        done(null);
      },
    },
  );
}

/** Takes an option's lines in turn, never the line said last when there is another. */
export function rotator() {
  const next = new Map<string, number>();
  let last = '';
  return (o: ChoiceOption): string => {
    const n = next.get(o.key) ?? 0;
    let line = o.lines[n % o.lines.length];
    let k = n + 1;
    if (line === last && o.lines.length > 1) line = o.lines[k++ % o.lines.length];
    next.set(o.key, k);
    last = line;
    return line;
  };
}
