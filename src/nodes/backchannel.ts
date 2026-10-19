import type { NodeDef, TimerHandle } from '../core/node';
import type { Json, JsonObject } from '../core/types';

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
 * Backchannels: short canned lines ("mm-hmm", "oh wow") the agent says while
 * the user holds the floor, to show it is listening. Something proposes a
 * moment (the turn model's backchannel score, or a pause in the user's speech),
 * and rules gate it: the user has the floor and has talked for a while, isn't
 * about to hand over (the VAD heard them recently, ASR hasn't endpointed,
 * end-of-turn score low, no draft reply out), and the last
 * backchannel was long enough ago. The line is rendered ahead by TTS and played
 * as a cue that doesn't start a reply, so turn-taking keeps listening.
 */
export const backchannelNode: NodeDef<BackchannelParams> = {
  type: 'backchannel',
  title: 'Backchannels',
  category: 'control',
  description: 'Says "mm-hmm"-style lines while the user keeps talking: when, from the turn model or pauses; what, from canned lines.',
  inputs: [
    { name: 'vad', type: 'vad' },
    { name: 'scores', type: 'turnscores' },
    { name: 'turn', type: 'turn' },
    { name: 'transcript', type: 'transcript' },
  ],
  outputs: [{ name: 'text', type: 'text', description: 'bc.lines (render these ahead) and bc.say (play one now) for TTS' }],
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
    { key: 'pauseMs', label: 'Pause (ms)', type: 'number', default: 0, when: { source: ['pause'] }, description: 'Extra silence after the VAD hears speech end.' },
    { key: 'maxEot', label: 'Max end-of-turn score', type: 'number', default: 0.3, description: "Don't backchannel when the user seems to be done: the turn model's end-of-turn score for the user is at least this." },
    { key: 'minTurnMs', label: 'Min turn length (ms)', type: 'number', default: 3000, description: 'The user has been talking this long in the current turn.' },
    { key: 'minGapMs', label: 'Min gap (ms)', type: 'number', default: 4000, description: 'Time since the last backchannel. People leave about 9 s (median), rarely under 3 s.' },
    {
      key: 'maxSilenceMs',
      label: 'Max silence (ms)',
      type: 'number',
      default: 700,
      description: "The VAD has heard the user within this long: most backchannels go over the user's speech, and a long silence means they may be done, whatever the turn model says.",
    },
    { key: 'lines', label: 'Lines', type: 'text', default: DEFAULT_BACKCHANNEL_LINES, description: 'Bare lines are rotated; "line = example | …" is picked by meaning from the user\'s recent words.' },
    { key: 'minScore', label: 'Min similarity', type: 'number', default: 0.45, description: 'A line with examples is said only when the recent words are at least this close; else a continuer.' },
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
      description: 'Sentence encoder for picking lines. Runs in the browser.',
    },
  ],
  channels: [{ name: 'embed', description: 'Sentence encoder in the browser: texts out, unit vectors back' }],
  create(ctx) {
    const p = ctx.params;
    const lines = parseBackchannelLines(String(p.lines));
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
    let asrDone = false;
    let eotUser = 0;
    let lastBc = -Infinity;
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

    /** Why not now, or null when a backchannel may be said. */
    const blocked = (now: number): string | null => {
      if (floor !== 'listening') return `agent is ${floor}`;
      if (turnStart === null) return 'no user turn';
      if (talkedMs(now) < p.minTurnMs) return `user has talked ${Math.round(talkedMs(now))} ms`;
      if (now - lastBc < p.minGapMs) return `last backchannel ${Math.round(now - lastBc)} ms ago`;
      if (silentSince !== null && now - silentSince > p.maxSilenceMs) return `user silent for ${Math.round(now - silentSince)} ms`;
      if (asrDone) return 'ASR heard the end of the utterance';
      if (draftOut) return 'a reply is being drafted';
      if (eotUser >= p.maxEot) return `end-of-turn score ${eotUser}`;
      return null;
    };

    const say = (line: string, why: string, extra: JsonObject = {}) => {
      const id = `${ctx.id}-bc${n++}`;
      lastBc = ctx.now();
      // A take other than the one this line used last, drawn from the id so a replay draws the same.
      const all = takes(line, p.takes);
      const prev = lastTake.get(line);
      const pool = all.length > 1 ? all.filter((t) => t !== prev) : all;
      const take = pool[Math.floor(seededRandom(id)() * pool.length)];
      lastTake.set(line, take);
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
      const recent = words(`${finals} ${partial}`).slice(-12).join(' ');
      // Claim the slot now, so a second proposal during the encoder call is gated.
      lastBc = now;
      if (!recent || !exampleVecs || !lines.examples.length) {
        const c = nextContinuer();
        if (c) say(c, why, data);
        return;
      }
      embed([recent], (vecs) => {
        const pick = vecs?.[0] ? pickLine(vecs[0], lines.examples, exampleVecs!, Number(p.minScore)) : null;
        const pickData: JsonObject = pick ? { words: recent, score: pick.score, example: pick.example } : { words: recent };
        // The moment may have passed while the encoder ran.
        if (floor !== 'listening' || draftOut) return void ctx.emit('text', { type: 'bc.skip', why, reason: 'moment passed', ...data });
        const line = pick?.line ?? nextContinuer();
        if (line) say(line, why, { ...data, ...pickData });
      });
    };

    const endTurn = () => {
      turnStart = null;
      speechMs = 0;
      speechFrom = null;
      finals = partial = '';
      draftOut = false;
      asrDone = false;
      pause?.clear();
    };

    return {
      start() {
        const all = allLines(lines);
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
          } else if (msg.type === 'turn.draft') draftOut = true;
          else if (msg.type === 'turn.draft_drop') draftOut = false;
          return;
        }
        if (port === 'vad') {
          if (msg.type === 'vad.speech_start') {
            pause?.clear();
            if (floor !== 'listening') return;
            turnStart ??= now;
            speechFrom = now;
            silentSince = null;
            asrDone = false;
          } else if (msg.type === 'vad.speech_end') {
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
        if (port === 'transcript' && turnStart !== null) {
          if (msg.type === 'asr.final') {
            finals = `${finals} ${String(msg.text ?? '')}`.trim();
            partial = '';
            if (msg.speechFinal && silentSince !== null) asrDone = true;
          } else if (msg.type === 'asr.partial') partial = String(msg.text ?? '');
        }
      },
      stop() {
        pause?.clear();
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
