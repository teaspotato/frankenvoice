// What the Tape inspector shows, derived from a trajectory's events incrementally (like timeline.ts):
// the tape as the tape stage (turn.tape) wrote it, cell by cell, with Gemma's odds, each call's request
// and answer, the cells written ahead, and the call in flight. Call update() whenever new events may
// have arrived; it works the same live, in a replay, and on a finished recording.

import type { Trajectory } from '../core/trajectory';
import type { Json, TrajEvent } from '../core/types';

/** One `/completion` call: the request as sent, and the answer once it came. */
export interface TapeCall {
  xid: string;
  /** Session ms when it was sent, and when it closed. */
  t0: number;
  t1?: number;
  /** Written ahead (a speculation) rather than for the live cell. */
  ahead: boolean;
  body: { prompt?: string; grammar?: string; logit_bias?: Json; n_probs?: number; [k: string]: Json | undefined };
  response?: { content?: string; completion_probabilities?: TokenProbs[]; timings?: Record<string, number>; [k: string]: unknown };
  /** Why it closed without an answer (cancelled, an error). */
  closed?: Json;
  seq: number;
}

/** A written token and its top alternatives, as llama.cpp returns them with n_probs. */
export interface TokenProbs {
  token: string;
  logprob: number;
  top_logprobs?: { token: string; logprob: number }[];
}

export interface TapeRow {
  n: number;
  /** Tape time of the cell (s), and session ms when its line was written. */
  t: number;
  at: number;
  written: number;
  line: string;
  state: string;
  action: string;
  say: string[];
  user: string[];
  voice: boolean;
  allowed?: string[];
  odds?: Record<string, number>;
  ms?: number;
  promptN?: number;
  promptMs?: number;
  predictedN?: number;
  predictedMs?: number;
  /** No call was made, and why (idle, TTS queue full, a call is still out, ...). */
  forced?: string;
  dropped?: string;
  failed?: boolean;
  late?: boolean;
  /** Taken from the cells written ahead (adopted, adopted late). */
  spec?: string;
  /** What the call wrote, when it couldn't be used. */
  wrote?: string;
  xid?: string;
  /** The last words of `say` the user never heard (a cut took them off the tape). */
  unsaid: number;
  /** The agent's voice was cut in this cell's words: `[cut]` after the heard ones. */
  cut: boolean;
  seq: number;
}

/** A cell written ahead, as if the user stays silent, not yet taken by a real cell. */
export interface AheadRow {
  n: number;
  line: string;
  action: string;
  odds?: Record<string, number>;
  ms?: number;
  xid?: string;
  seq: number;
}

/** A speculation given up, shown after the row it followed. */
export interface SpecNote {
  afterN: number;
  t: number;
  why: string;
  ahead: number;
  seq: number;
}

export class TapeModel {
  /** The tape stage's node id, if the session ran one. */
  readonly node: string | null;
  rows: TapeRow[] = [];
  ahead: AheadRow[] = [];
  notes: SpecNote[] = [];
  calls = new Map<string, TapeCall>();
  /** The call in flight, if any. */
  pending: TapeCall | null = null;
  /** The node's latest state (state, floor, calling, ahead, lastMs, userSpeaking). */
  state: Record<string, Json> = {};
  /** A cell's length in real time (the call's budget), and in tape seconds. */
  readonly cellMs: number = 640;
  readonly cellS: number = 0.64;
  private next = 0;
  private calling: Json = false;

  constructor(readonly traj: Trajectory) {
    const spec = traj.meta.graph.nodes.find((n) => n.type === 'turn.tape');
    this.node = spec?.id ?? null;
    if (spec) {
      this.cellMs = Math.max(50, Number(spec.params.cellMs) || 640);
      this.cellS = (this.cellMs * (Number(spec.params.timeScale) || 1)) / 1000;
    }
  }

  /** Consume new events; true if anything changed. */
  update(): boolean {
    const ev = this.traj.events;
    if (!this.node || this.next >= ev.length) {
      this.next = ev.length;
      return false;
    }
    let changed = false;
    for (; this.next < ev.length; this.next++) if (ev[this.next].node === this.node && this.consume(ev[this.next])) changed = true;
    return changed;
  }

  private consume(e: TrajEvent): boolean {
    const d = (e.data ?? {}) as Record<string, any>;
    if (e.kind === 'state') {
      Object.assign(this.state, d);
      if ('calling' in d) this.calling = d.calling;
      return true;
    }
    if (e.kind === 'io.open' && d.channel === 'llama-completion') {
      const call: TapeCall = { xid: d.xid, t0: e.t, ahead: this.calling === 'ahead', body: d.request?.body ?? {}, seq: e.seq };
      this.calls.set(call.xid, call);
      this.pending = call;
      return true;
    }
    if (e.kind === 'io.recv' || e.kind === 'io.close') {
      const call = this.calls.get(d.xid);
      if (!call) return false;
      if (e.kind === 'io.recv' && d.payload && typeof d.payload === 'object') call.response = d.payload;
      if (e.kind === 'io.close') {
        call.t1 = e.t;
        if (d.payload) call.closed = d.payload;
        if (this.pending === call) this.pending = null;
      }
      return true;
    }
    if (e.kind !== 'msg') return false;
    switch (d.type) {
      case 'tape.cell': {
        const row: TapeRow = {
          n: d.n,
          t: d.t,
          at: d.at,
          written: e.t,
          line: d.line ?? '',
          state: d.state ?? '',
          action: d.action ?? '',
          say: d.say ?? [],
          user: d.user ?? [],
          voice: !!d.voice,
          unsaid: 0,
          cut: false,
          seq: e.seq,
        };
        for (const k of ['allowed', 'odds', 'ms', 'promptN', 'promptMs', 'predictedN', 'predictedMs', 'forced', 'dropped', 'failed', 'late', 'spec', 'wrote', 'xid'] as const)
          if (d[k] !== undefined && d[k] !== null) (row as unknown as Record<string, unknown>)[k] = d[k];
        // Recorded before cells carried their call's xid: the last call that asked for this line.
        if (!row.xid && row.ms !== undefined) row.xid = this.callFor(row.line);
        // In tape order: a late answer lands after cells that were decided without it.
        let i = this.rows.length;
        while (i > 0 && this.rows[i - 1].n > row.n) i--;
        this.rows.splice(i, 0, row);
        // Written ahead and now real: it leaves the cells ahead.
        this.ahead = this.ahead.filter((a) => a.n > row.n);
        return true;
      }
      case 'tape.spec':
        if (d.event === 'start') this.ahead = [];
        else if (d.event === 'cell') {
          const xid = d.xid ?? this.callFor(d.line ?? '');
          this.ahead.push({ n: d.n, line: d.line ?? '', action: d.action ?? '', seq: e.seq, ...(d.odds ? { odds: d.odds } : {}), ...(typeof d.ms === 'number' ? { ms: d.ms } : {}), ...(xid ? { xid } : {}) });
        }
        else if (d.event === 'drop') {
          this.notes.push({ afterN: this.rows.at(-1)?.n ?? 0, t: e.t, why: d.why ?? '', ahead: d.ahead ?? this.ahead.length, seq: e.seq });
          this.ahead = [];
        }
        return true;
      case 'tape.cut': {
        // The reply's last `unsaid` words were never heard: take them from the newest rows that sent words.
        let k = Array.isArray(d.unsaid) ? d.unsaid.length : 0;
        let marked = false;
        for (let i = this.rows.length - 1; i >= 0; i--) {
          const r = this.rows[i];
          if (r.action === 'backchannel' || !r.say.length) continue;
          const take = Math.min(k, r.say.length - r.unsaid);
          r.unsaid += take;
          k -= take;
          if (k === 0 && r.unsaid < r.say.length) {
            r.cut = true;
            marked = true;
            break;
          }
          if (r.action === 'respond' || r.action === 'resume') break;
        }
        // Nothing of it was heard: the mark goes on the reply's first row.
        if (!marked) {
          const first = [...this.rows].reverse().find((r) => r.action === 'respond' || r.action === 'resume');
          if (first) first.cut = true;
        }
        return true;
      }
    }
    return false;
  }

  /** The newest answered call whose prompt ends in `line`'s cell, up to the agent's action. */
  private callFor(line: string): string | undefined {
    const i = line.indexOf('agent (');
    const open = i < 0 ? '' : line.slice(0, line.indexOf(':', i) + 1);
    const calls = [...this.calls.values()];
    for (let k = calls.length - 1; k >= 0; k--) if (calls[k].response && openLine(calls[k].body.prompt) === open) return calls[k].xid;
    return undefined;
  }

  /** The call that decided a row (or a cell written ahead). */
  callOf(x: { xid?: string } | null | undefined): TapeCall | undefined {
    return x?.xid ? this.calls.get(x.xid) : undefined;
  }
}

/** Actions that leave things as they are: listen, or go on speaking. */
export const HOLD = new Set(['listen', 'continue']);

/** The ways Gemma can act, as the odds chart groups them: respond and resume both start an answer. */
export type SeriesKey = 'respond' | 'backchannel' | 'finish';
export const SERIES_KEYS: SeriesKey[] = ['respond', 'backchannel', 'finish'];
export const SERIES: Record<string, SeriesKey | undefined> = { respond: 'respond', resume: 'respond', backchannel: 'backchannel', finish: 'finish' };

/** Gemma's odds of each way of acting in one cell; a way the cell's state doesn't allow is absent. */
export function seriesOdds(odds: Record<string, number>): Partial<Record<SeriesKey, number>> {
  const out: Partial<Record<SeriesKey, number>> = {};
  for (const [a, p] of Object.entries(odds)) {
    const k = SERIES[a];
    if (k) out[k] = (out[k] ?? 0) + p;
  }
  return out;
}

/** A cell where nothing happened and nothing was asked: no call, nobody talking, the agent holding. */
export const isIdle = (r: TapeRow) => r.forced === 'idle' && HOLD.has(r.action) && !r.user.length && !r.voice && !r.say.length;

/** The tape as the table lists it: a run of at least `min` idle cells is one item. */
export type TapeItem = { kind: 'row'; row: TapeRow } | { kind: 'idle'; rows: TapeRow[] };
export function groupRows(rows: TapeRow[], min = 3): TapeItem[] {
  const out: TapeItem[] = [];
  let run: TapeRow[] = [];
  const flush = () => {
    if (run.length >= min) out.push({ kind: 'idle', rows: run });
    else for (const row of run) out.push({ kind: 'row', row });
    run = [];
  };
  for (const r of rows) {
    if (isIdle(r)) run.push(r);
    else {
      flush();
      out.push({ kind: 'row', row: r });
    }
  }
  flush();
  return out;
}

/** A line's tape time (s): the number it starts with. */
export const lineTime = (line: string) => Number.parseFloat(line);

/** A prompt split into its fixed head (instructions, example) and the live tape. */
export function splitPrompt(prompt: string): { head: string; tape: string } {
  const i = prompt.lastIndexOf('<channel|>');
  return i < 0 ? { head: '', tape: prompt } : { head: prompt.slice(0, i + '<channel|>'.length), tape: prompt.slice(i + '<channel|>'.length) };
}

/** The last line of a prompt: the cell being decided, up to `agent (<state>):`. */
export function openLine(prompt: string | undefined): string {
  const { tape } = splitPrompt(prompt ?? '');
  return tape.slice(tape.lastIndexOf('\n') + 1);
}
