import { useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import type { Trajectory } from '../core/trajectory';
import {
  HOLD,
  SERIES,
  SERIES_KEYS,
  TapeModel,
  groupRows,
  lineTime,
  openLine,
  seriesOdds,
  splitPrompt,
  type AheadRow,
  type SeriesKey,
  type SpecNote,
  type TapeCall,
  type TapeRow,
} from './tape-model';

/** Each way of acting has one colour, here and on the dashboard's tape row; holding (listen, continue) is grey. */
const SERIES_VAR: Record<SeriesKey, string> = { respond: 'var(--act-respond)', backchannel: 'var(--act-backchannel)', finish: 'var(--act-finish)' };
const SERIES_LABEL: Record<SeriesKey, string> = { respond: 'respond / resume', backchannel: 'backchannel', finish: 'finish' };
const actionColor = (a: string) => {
  const k = SERIES[a];
  return k ? SERIES_VAR[k] : 'var(--muted)';
};
const STATE_TEXT: Record<string, string> = { silent: 'silent', speaking: 'speaking', cut: 'cut off', 'cut off': 'cut off' };
/** A probability as text; 0 means the token wasn't among the top alternatives llama.cpp returned. */
const pct = (p: number) => (p >= 0.995 ? '100%' : p >= 0.1 ? `${Math.round(p * 100)}%` : p >= 0.001 ? `${(p * 100).toFixed(1)}%` : p > 0 ? '<0.1%' : '≈0');
const vis = (s: string) => JSON.stringify(s).slice(1, -1);

type Pick = { kind: 'row' | 'ahead'; seq: number };

/** The most likely action Gemma did not write: what it nearly did. */
function runnerUp(odds: Record<string, number> | undefined, action: string): [string, number] | null {
  if (!odds) return null;
  const alt = Object.entries(odds)
    .filter(([a]) => a !== action)
    .sort((a, b) => b[1] - a[1])[0];
  return alt ?? null;
}

/** The pieces of a line up to `agent (<state>):`. */
function parseOpen(line: string): { t: number; words: string; voice: boolean; state: string } | null {
  const m = /^\s*([\d.]+) \| user: (?:"(.*)" )?\((talking|silent)\) \| agent \(([^)]*)\):/.exec(line);
  return m ? { t: Number(m[1]), words: m[2] ?? '', voice: m[3] === 'talking', state: m[4] } : null;
}

function UserCell({ words, voice }: { words: string; voice: boolean }) {
  if (words) return <span title={voice ? 'talking' : 'words committed, silent by now'}>“{words}”</span>;
  return voice ? <span className="muted">talking</span> : null;
}

/** What Gemma wrote: the action (grey when it holds), its words, `[cut]` where the voice stopped, unheard words struck through. */
function AgentCell({ action, say = [], unsaid = 0, cut = false }: { action: string; say?: string[]; unsaid?: number; cut?: boolean }) {
  const heard = say.slice(0, say.length - unsaid);
  const gone = say.slice(say.length - unsaid);
  const hold = HOLD.has(action);
  return (
    <span>
      {!hold && <i className="tape-key" style={{ background: actionColor(action) }} />}
      <span className={hold ? 'muted' : 'tape-act'}>{action}</span>
      {say.length > 0 && (
        <>
          {' “'}
          {heard.join(' ')}
          {cut && <span className="bad">{heard.length ? ' ' : ''}[cut]</span>}
          {gone.length > 0 && (
            <s className="muted" title="sent to TTS, never heard: taken off the tape">
              {` ${gone.join(' ')}`}
            </s>
          )}
          {'”'}
        </>
      )}
    </span>
  );
}

function Notes({ r }: { r: TapeRow }) {
  const out: ReactNode[] = [];
  if (r.forced) out.push(r.forced === 'idle' ? 'idle, no call' : `no call: ${r.forced}`);
  if (r.spec) out.push(r.spec === 'adopted' ? 'written ahead' : `written ahead, ${r.spec}`);
  if (r.late)
    out.push(
      <span key="late" className="warn" title="answered after its cell had ended">
        ⚠ late
      </span>,
    );
  if (r.dropped)
    out.push(
      <span key="drop" className="bad">
        ✕ dropped: {r.dropped}
      </span>,
    );
  if (r.failed)
    out.push(
      <span key="fail" className="bad">
        ✕ failed{r.wrote ? `: wrote ${r.wrote}` : ''}
      </span>,
    );
  return (
    <>
      {out.map((x, i) => (
        <span key={i}>
          {i > 0 && ' · '}
          {x}
        </span>
      ))}
    </>
  );
}

// ---- the odds chart --------------------------------------------------------------------------

/** A column of the chart: a real cell or one written ahead. */
interface ChartCell {
  n: number;
  t: number;
  state: string;
  action: string;
  odds?: Record<string, number>;
  ms?: number;
  ahead: boolean;
  user: string[];
  voice: boolean;
  /** The agent's voice was heard in this cell. */
  said: boolean;
  cut: boolean;
  pick: Pick;
}

const PX = 8;
const ML = 52;
const MR = 12;
/** Rows of the chart, top to bottom: who talked, Gemma's odds (log), call time, the time axis. */
const Y = { user: 4, agent: 17, strip: 8, o0: 40, o1: 118, l0: 132, l1: 164, ax: 177, h: 182 };
const LOG_MIN = -3;
const yOdds = (p: number) => {
  const l = Math.max(LOG_MIN, Math.log10(Math.max(p, 1e-9)));
  return Y.o1 - ((l - LOG_MIN) / -LOG_MIN) * (Y.o1 - Y.o0);
};
/** A column with a rounded top, square on its baseline. */
const colPath = (x: number, w: number, y: number, base: number) => {
  const r = Math.min(2, w / 2, base - y);
  return `M${x},${base}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${base}Z`;
};

function OddsChart({
  m,
  rows,
  selN,
  centre,
  pending,
  onPick,
}: {
  m: TapeModel;
  rows: TapeRow[];
  selN?: number;
  /** Keep the selected cell in view, rather than the newest. */
  centre: boolean;
  pending?: { n: number; ms: number };
  onPick(p: Pick): void;
}) {
  const box = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(640);
  const [hover, setHover] = useState<{ n: number; x: number } | null>(null);
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    setW(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const cells = new Map<number, ChartCell>();
  for (const r of rows)
    cells.set(r.n, {
      n: r.n,
      t: r.t,
      state: r.state,
      action: r.action,
      odds: r.odds,
      ms: r.ms,
      ahead: false,
      user: r.user,
      voice: r.voice,
      said: r.say.length - r.unsaid > 0,
      cut: r.cut,
      pick: { kind: 'row', seq: r.seq },
    });
  for (const a of m.ahead)
    if (!cells.has(a.n))
      cells.set(a.n, { n: a.n, t: lineTime(a.line), state: parseOpen(a.line)?.state ?? 'silent', action: a.action, odds: a.odds, ms: a.ms, ahead: true, user: [], voice: false, said: false, cut: false, pick: { kind: 'ahead', seq: a.seq } });
  const ns = [...cells.keys()];
  const first = Math.min(...ns);
  const last = Math.max(...ns, pending?.n ?? -Infinity);
  const span = Math.max(8, Math.floor((w - ML - MR) / PX));
  let end = centre && selN !== undefined ? Math.min(last, selN + Math.floor(span / 2)) : last;
  const start = Math.max(first, end - span + 1);
  end = Math.min(last, start + span - 1);
  const x = (n: number) => ML + (n - start + 0.5) * PX;
  const ref = rows[0] ?? { n: first, t: cells.get(first)?.t ?? 0 };
  const tOf = (n: number) => ref.t + (n - ref.n) * m.cellS;

  const inView: ChartCell[] = [];
  for (let n = start; n <= end; n++) {
    const c = cells.get(n);
    if (c) inView.push(c);
  }
  const anyOdds = rows.some((r) => r.odds);

  // Gemma's odds of each way of acting: a line through consecutive cells that allow it, dashed where written ahead.
  const lines: { k: SeriesKey; d: string; ahead: boolean }[] = [];
  const dots: { k: SeriesKey; x: number; y: number; ahead: boolean }[] = [];
  for (const k of SERIES_KEYS) {
    let pts: string[] = [];
    let ahead = false;
    let prev: string | null = null;
    const flush = () => {
      if (pts.length > 1) lines.push({ k, d: `M${pts.join('L')}`, ahead });
      pts = [];
    };
    for (let n = start; n <= end; n++) {
      const c = cells.get(n);
      const p = c?.odds ? seriesOdds(c.odds)[k] : undefined;
      if (c && SERIES[c.action] === k) dots.push({ k, x: x(n), y: p === undefined ? Y.o0 : yOdds(p), ahead: c.ahead });
      if (p === undefined || !c) {
        flush();
        prev = null;
        continue;
      }
      const pt = `${x(n).toFixed(1)},${yOdds(p).toFixed(1)}`;
      if (c.ahead !== ahead) {
        flush();
        ahead = c.ahead;
        if (prev) pts.push(prev);
      }
      pts.push(pt);
      prev = pt;
    }
    flush();
  }

  const maxMs = Math.max(m.cellMs * 1.25, ...inView.map((c) => c.ms ?? 0), pending?.ms ?? 0);
  const yMs = (ms: number) => Y.l1 - (Math.min(ms, maxMs) / maxMs) * (Y.l1 - Y.l0);
  const step = (5 / m.cellS) * PX >= 48 ? 5 : (10 / m.cellS) * PX >= 48 ? 10 : 30;
  const ticks: number[] = [];
  for (let T = Math.ceil(tOf(start) / step) * step; T <= tOf(end); T += step) ticks.push(T);

  const at = (clientX: number) => {
    const r = box.current!.getBoundingClientRect();
    const n = start + Math.floor((clientX - r.left - ML) / PX);
    return n >= start && n <= end ? n : null;
  };
  const hc = hover ? cells.get(hover.n) : undefined;
  const bw = PX - 2;

  return (
    <div className="tape-chart">
      <div className="tape-chart-head">
        <h4>Gemma’s odds of acting</h4>
        <span className="tape-legend">
          {SERIES_KEYS.map((k) => (
            <span key={k}>
              <i style={{ background: SERIES_VAR[k] }} />
              {SERIES_LABEL[k]}
            </span>
          ))}
          <span>
            <i className="dashed" />
            written ahead
          </span>
        </span>
        <span className="muted small">
          {anyOdds ? 'per cell, log scale; dot: what it wrote · bars: call time against one cell' : 'No odds recorded: set Odds (top tokens) above 0 in the tape stage. Dots: what it wrote · bars: call time'}
        </span>
      </div>
      <div
        ref={box}
        className="tape-chart-box"
        onPointerMove={(e) => {
          const n = at(e.clientX);
          setHover(n === null ? null : { n, x: x(n) });
        }}
        onPointerLeave={() => setHover(null)}
        onClick={(e) => {
          const n = at(e.clientX);
          const c = n === null ? undefined : cells.get(n);
          if (c) onPick(c.pick);
        }}
      >
        <svg width={w} height={Y.h} role="img" aria-label="Gemma's odds of acting per cell, and each call's time; the table below lists the same cells">
          {selN !== undefined && selN >= start && selN <= end && <rect className="tc-sel" x={x(selN) - PX / 2} y={0} width={PX} height={Y.l1} />}
          <text className="tc-tick" x={ML - 6} y={Y.user + 7} textAnchor="end">
            user
          </text>
          <text className="tc-tick" x={ML - 6} y={Y.agent + 7} textAnchor="end">
            agent
          </text>
          {[0, -1, -2, -3].map((l) => (
            <g key={l}>
              <line className="tc-grid" x1={ML} x2={w - MR} y1={yOdds(10 ** l)} y2={yOdds(10 ** l)} />
              <text className="tc-tick" x={ML - 6} y={yOdds(10 ** l) + 3.5} textAnchor="end">
                {l === -3 ? '≤0.1%' : `${100 * 10 ** l}%`}
              </text>
            </g>
          ))}
          <line className="tc-ref" x1={ML} x2={w - MR} y1={yMs(m.cellMs)} y2={yMs(m.cellMs)} />
          <text className="tc-tick" x={ML - 6} y={yMs(m.cellMs) + 3.5} textAnchor="end">
            {m.cellMs} ms
          </text>
          <line className="tc-grid" x1={ML} x2={w - MR} y1={Y.l1} y2={Y.l1} />
          {ticks.map((T) => {
            const tx = x(ref.n + (T - ref.t) / m.cellS);
            return (
              <text key={T} className="tc-tick" x={tx} y={Y.ax} textAnchor="middle">
                {T} s
              </text>
            );
          })}
          {inView.map((c) => (
            <g key={c.n} opacity={c.ahead ? 0.5 : 1}>
              {(c.voice || c.user.length > 0) && <rect className="tc-strip" x={x(c.n) - PX / 2} y={Y.user} width={PX} height={Y.strip} />}
              {c.said && <rect className="tc-strip" x={x(c.n) - PX / 2} y={Y.agent} width={PX} height={Y.strip} />}
              {c.cut && <path className="tc-cut" d={`M${x(c.n) - 3},${Y.agent}l6,${Y.strip}M${x(c.n) + 3},${Y.agent}l-6,${Y.strip}`} />}
              {c.ms !== undefined && <path className="tc-col" d={colPath(x(c.n) - bw / 2, bw, yMs(c.ms), Y.l1)} />}
            </g>
          ))}
          {pending && pending.n >= start && pending.n <= end && <path className="tc-col pending" d={colPath(x(pending.n) - bw / 2, bw, yMs(pending.ms), Y.l1)} />}
          {lines.map((l, i) => (
            <path key={i} className={`tc-line${l.ahead ? ' ahead' : ''}`} d={l.d} style={{ stroke: SERIES_VAR[l.k] }} />
          ))}
          {dots.map((d, i) => (
            <circle key={i} className="tc-dot" cx={d.x} cy={d.y} r={4} style={{ fill: SERIES_VAR[d.k] }} opacity={d.ahead ? 0.6 : 1} />
          ))}
          {hover && <line className="tc-cross" x1={hover.x} x2={hover.x} y1={0} y2={Y.l1} />}
        </svg>
        {hover && hc && <ChartTip c={hc} x={hover.x} w={w} />}
      </div>
    </div>
  );
}

function ChartTip({ c, x, w }: { c: ChartCell; x: number; w: number }) {
  const odds = Object.entries(c.odds ?? {}).sort((a, b) => b[1] - a[1]);
  return (
    <div className="tape-tip" style={{ left: x + 220 > w ? Math.max(0, x - 226) : x + 12 }}>
      <div>
        <b>{c.t.toFixed(2)} s</b> <span className="muted">{c.ahead ? 'written ahead' : `agent ${STATE_TEXT[c.state] ?? c.state}`}</span>
      </div>
      <div className="muted">
        user {c.user.length ? <span className="tape-tip-v">“{c.user.join(' ')}”</span> : null} {c.voice ? 'talking' : c.user.length ? '' : 'silent'}
      </div>
      {odds.length === 0 && (
        <div className="tape-tip-row">
          <i style={{ background: actionColor(c.action) }} />
          <b>{c.action}</b> <span className="muted">written</span>
        </div>
      )}
      {odds.map(([a, p]) => (
        <div key={a} className="tape-tip-row">
          <i style={{ background: actionColor(a) }} />
          <b className="num">{pct(p)}</b> <span className="muted">{a}</span>
          {a === c.action && <span className="muted"> ✓ written</span>}
        </div>
      ))}
      {c.ms !== undefined && (
        <div>
          <b className="num">{c.ms} ms</b> <span className="muted">call</span>
        </div>
      )}
    </div>
  );
}

// ---- the selected cell ----------------------------------------------------------------------

/** Where a call's time went, against the cell's budget: prefill, writing, the rest (HTTP, waiting). */
function CallTime({ call, cellMs }: { call: TapeCall; cellMs: number }) {
  const t = call.response?.timings ?? {};
  if (call.t1 === undefined) return <p className="small muted">Still out.</p>;
  const total = Math.max(0, call.t1 - call.t0);
  const prefill = Math.min(total, t.prompt_ms ?? 0);
  const write = Math.min(total - prefill, t.predicted_ms ?? 0);
  const other = Math.max(0, total - prefill - write);
  const scale = Math.max(total, cellMs) * 1.04;
  const parts: [string, number, string, string][] = [
    ['prefill', prefill, 'tape-seg1', t.prompt_n !== undefined ? `${t.prompt_n} new tokens${t.cache_n !== undefined ? `, ${t.cache_n} cached` : ''}` : ''],
    ['writing', write, 'tape-seg2', t.predicted_n !== undefined ? `${t.predicted_n} tokens` : ''],
    ['other', other, 'tape-seg3', 'HTTP, waiting'],
  ];
  return (
    <>
      <div className="tape-timebar" title={`${Math.round(total)} ms of a ${cellMs} ms cell`}>
        {parts.map(([k, ms, cls]) => ms > 0 && <span key={k} className={cls} style={{ width: `${(ms / scale) * 100}%` }} />)}
        <span className="tape-budget" style={{ left: `${(cellMs / scale) * 100}%` }}>
          <small>{cellMs} ms</small>
        </span>
      </div>
      <div className="tape-timelegend">
        {parts.map(([k, ms, cls, what]) => (
          <span key={k}>
            <i className={cls} />
            <b className="num">{Math.round(ms)} ms</b> <span className="muted">{k}</span>
            {what && <span className="muted"> ({what})</span>}
          </span>
        ))}
      </div>
      {(t.draft_n || call.ahead) && (
        <p className="small muted">
          {t.draft_n ? `Drafts: ${t.draft_n_accepted ?? 0} of ${t.draft_n} accepted. ` : ''}
          {call.ahead ? 'Written ahead, as if the user stays silent.' : ''}
        </p>
      )}
    </>
  );
}

function CallDetail({ call, odds, allowed, chosen, cellMs }: { call: TapeCall; odds?: Record<string, number>; allowed?: string[]; chosen: string; cellMs: number }) {
  const [copied, setCopied] = useState(false);
  const res = call.response;
  const probs = res?.completion_probabilities ?? [];
  const first = probs[0]?.top_logprobs ?? [];
  const actions = allowed ?? Object.keys(odds ?? {});
  const others = first.filter((t) => !actions.some((a) => ` ${a}`.startsWith(t.token) && t.token.trim()));
  const prompt = call.body.prompt ?? '';
  const { head, tape } = splitPrompt(prompt);
  const lines = tape.split('\n');
  const total = call.t1 !== undefined ? Math.round(call.t1 - call.t0) : null;
  const copy = () =>
    navigator.clipboard?.writeText(prompt).then(
      () => (setCopied(true), setTimeout(() => setCopied(false), 1200)),
      () => {},
    );
  return (
    <>
      {odds && (
        <section>
          <h4>
            Decision <small className="muted">first token, before bias and grammar</small>
          </h4>
          <div className="tape-bars">
            {Object.entries(odds)
              .sort((a, b) => b[1] - a[1])
              .map(([a, p]) => (
                <div key={a} className="tape-barrow">
                  <span>{a}</span>
                  <span className="track">
                    <span className="fill" style={{ width: `${p * 100}%`, background: actionColor(a) }} />
                  </span>
                  <span className="num">{pct(p)}</span>
                  <span className="muted">{a === chosen ? '✓ written' : ''}</span>
                </div>
              ))}
          </div>
          {others.length > 0 && (
            <p className="small muted">
              Outside the grammar: {others.map((t) => `“${vis(t.token)}” ${pct(Math.exp(t.logprob))}`).join(' · ')}
            </p>
          )}
        </section>
      )}
      <section>
        <h4>
          Call time{' '}
          <small className="muted">
            {total !== null ? `${total} ms of a ${cellMs} ms cell` : 'in flight'}
            {call.closed !== undefined && !res ? ` · closed: ${JSON.stringify(call.closed)}` : ''}
          </small>
        </h4>
        <CallTime call={call} cellMs={cellMs} />
      </section>
      {probs.length > 0 && (
        <section>
          <h4>
            Tokens written <small className="muted">each with the alternatives Gemma weighed</small>
          </h4>
          <div className="tape-tokens">
            {probs.map((t, i) => (
              <div key={i} className="tape-token">
                <code>“{vis(t.token)}”</code>
                <span className="num">{pct(Math.exp(t.logprob))}</span>
                <span className="muted">
                  {(t.top_logprobs ?? [])
                    .filter((x) => x.token !== t.token)
                    .slice(0, 4)
                    .map((x) => `“${vis(x.token)}” ${pct(Math.exp(x.logprob))}`)
                    .join(' · ')}
                </span>
              </div>
            ))}
          </div>
        </section>
      )}
      <section>
        <h4>
          Prompt <small className="muted">{prompt.length} chars</small>{' '}
          <button className="link" onClick={copy}>
            {copied ? 'copied' : 'copy'}
          </button>
        </h4>
        {call.body.logit_bias !== undefined && (
          <p className="small">
            Logit bias <code>{JSON.stringify(call.body.logit_bias)}</code>
          </p>
        )}
        <details>
          <summary className="muted small">Instructions and example ({head.length} chars)</summary>
          <pre className="tape-pre">{head}</pre>
        </details>
        <pre className="tape-pre">
          {lines.slice(0, -1).join('\n')}
          {lines.length > 1 && '\n'}
          <mark>{lines[lines.length - 1]}</mark>
        </pre>
        <details>
          <summary className="muted small">Grammar</summary>
          <pre className="tape-pre">{call.body.grammar ?? ''}</pre>
        </details>
        {res && (
          <details>
            <summary className="muted small">Raw answer</summary>
            <pre className="tape-pre">{JSON.stringify(res, null, 2)}</pre>
          </details>
        )}
      </section>
    </>
  );
}

function Detail({ m, sel }: { m: TapeModel; sel: TapeRow | AheadRow }) {
  const row = 'say' in sel ? (sel as TapeRow) : undefined;
  const open = parseOpen(sel.line);
  const call = m.callOf(sel);
  const state = row?.state ?? open?.state ?? '';
  return (
    <>
      <div className="tape-dhead">
        <h4>Cell {(row?.t ?? lineTime(sel.line)).toFixed(2)} s</h4>
        <span className="muted">
          {row ? (
            <>
              user {row.user.length ? `“${row.user.join(' ')}”` : ''} {row.voice ? 'talking' : row.user.length ? '' : 'silent'} · agent {STATE_TEXT[state] ?? state}, wrote{' '}
            </>
          ) : (
            'written ahead, as if the user stays silent: '
          )}
          <span className="tape-act-inline">
            {!HOLD.has(sel.action) && <i className="tape-key" style={{ background: actionColor(sel.action) }} />}
            {sel.action}
          </span>
        </span>
      </div>
      <pre className="tape-pre tape-selline">{sel.line}</pre>
      {row?.forced && <p className="small muted">No call: {row.forced}.</p>}
      {row?.dropped && <p className="small bad">✕ Not used: {row.dropped}.</p>}
      {call ? (
        <CallDetail call={call} odds={sel.odds} allowed={row?.allowed} chosen={sel.action} cellMs={m.cellMs} />
      ) : (
        !row?.forced && <p className="muted small">No call recorded for this cell.</p>
      )}
    </>
  );
}

// ---- the inspector --------------------------------------------------------------------------

/**
 * The tape stage (turn.tape) as it runs: Gemma's odds of acting cell by cell, each call's time, the
 * tape as a table (what the user said, what Gemma wrote and how sure it was), the cells written ahead,
 * the call in flight, and for any cell the exact prompt, grammar and token odds of its call.
 */
export function TapeInspector({
  trajectory,
  live,
  open,
  onToggle,
  now,
  tabs,
  focus,
}: {
  trajectory?: Trajectory;
  live: boolean;
  open: boolean;
  onToggle(): void;
  /** Events of a clicked timeline item: a tape cell among them is selected. `n` changes on every click. */
  focus?: { seqs: number[]; n: number } | null;
  /** Session ms now, for how long the call in flight has taken. */
  now(): number;
  tabs?: ReactNode;
}) {
  const [, setTick] = useState(0);
  const modelRef = useRef<TapeModel | null>(null);
  if (trajectory && modelRef.current?.traj !== trajectory) modelRef.current = new TapeModel(trajectory);
  const m = trajectory ? modelRef.current : null;
  m?.update();
  /** The row whose call is shown; null: the newest call. */
  const [picked, setPicked] = useState<Pick | null>(null);
  /** Idle runs opened in the table, by their first cell. */
  const [unfolded, setUnfolded] = useState<Set<number>>(() => new Set());
  const bodyRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  useEffect(() => {
    setPicked(null);
    setUnfolded(new Set());
  }, [trajectory]);
  const pick = (p: Pick | null, scroll = false) => {
    setPicked(p);
    if (!p) {
      stick.current = true;
      return;
    }
    stick.current = false;
    if (scroll) requestAnimationFrame(() => bodyRef.current?.querySelector(`[data-seq="${p.seq}"]`)?.scrollIntoView({ block: 'nearest' }));
  };
  // A cell clicked on the dashboard's tape row: select it here and scroll to it.
  useEffect(() => {
    const row = focus && modelRef.current?.rows.find((r) => focus.seqs.includes(r.seq));
    if (!row) return;
    pick({ kind: 'row', seq: row.seq }, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus?.n]);
  useEffect(() => {
    if (!trajectory || !live || !open) return;
    let raf = 0;
    const tick = () => {
      raf = 0;
      if (modelRef.current?.update()) setTick((t) => t + 1);
    };
    const unsub = trajectory.subscribe(() => {
      if (!raf) raf = requestAnimationFrame(tick);
    });
    // While a call is out, its timer runs.
    const clock = setInterval(() => modelRef.current?.pending && setTick((t) => t + 1), 100);
    return () => {
      unsub();
      cancelAnimationFrame(raf);
      clearInterval(clock);
    };
  }, [trajectory, live, open]);
  useEffect(() => {
    if (stick.current && bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
  });

  const rows = m?.rows ?? [];
  const visible = rows.slice(-600);
  const notesAfter = new Map<number, SpecNote[]>();
  for (const k of m?.notes ?? []) notesAfter.set(k.afterN, [...(notesAfter.get(k.afterN) ?? []), k]);

  const latest = [...rows].reverse().find((r) => r.xid);
  const sel: TapeRow | AheadRow | undefined = picked
    ? picked.kind === 'row'
      ? rows.find((r) => r.seq === picked.seq)
      : m?.ahead.find((a) => a.seq === picked.seq)
    : latest;
  const st = m?.state ?? {};
  const pending = m?.pending;
  const waitedMs = pending && live ? Math.max(0, Math.round(now() - pending.t0)) : undefined;
  const pendingOpen = pending ? parseOpen(openLine(pending.body.prompt)) : null;
  const ref = rows[0];
  const pendingN = pendingOpen && ref && m ? Math.round(ref.n + (pendingOpen.t - ref.t) / m.cellS) : undefined;

  const keys = (e: KeyboardEvent) => {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    e.preventDefault();
    const i = sel ? visible.findIndex((r) => r === sel) : visible.length;
    const next = visible[Math.max(0, Math.min(visible.length - 1, i + (e.key === 'ArrowDown' ? 1 : -1)))];
    if (next) pick({ kind: 'row', seq: next.seq }, true);
  };
  const rowEl = (r: TapeRow) => {
    const alt = runnerUp(r.odds, r.action);
    return (
      <div key={r.seq} data-seq={r.seq} className={`tape-tr${sel === r ? ' sel' : ''}`} onClick={() => pick({ kind: 'row', seq: r.seq })} title={`written ${(r.written / 1000).toFixed(2)} s into the session`}>
        <span className="num muted">{r.t.toFixed(2)}</span>
        <span>
          <UserCell words={r.user.join(' ')} voice={r.voice} />
        </span>
        <span className="muted">{STATE_TEXT[r.state] ?? r.state}</span>
        <AgentCell action={r.action} say={r.say} unsaid={r.unsaid} cut={r.cut} />
        <span className="num">{r.odds ? pct(r.odds[r.action] ?? 0) : ''}</span>
        <span>
          {alt && (
            <>
              <span className="muted">{alt[0]}</span> <span className="num">{pct(alt[1])}</span>
            </>
          )}
        </span>
        <span className="num">{r.ms ?? <span className="muted">–</span>}</span>
        <span className="small">
          <Notes r={r} />
        </span>
      </div>
    );
  };

  return (
    <div className="event-log tape-inspector">
      <div className="event-log-bar">
        <button className="link event-log-toggle" onClick={onToggle} title={open ? 'Collapse' : 'Expand'}>
          {open ? '▾' : '▸'}
        </button>
        {tabs}
        {!m?.node ? (
          <span className="muted">{trajectory ? 'This session has no tape stage: pick turn-taking › tape.' : 'No session.'}</span>
        ) : (
          <>
            <span className="tape-status">
              <span>
                <span className="k">agent</span>
                {STATE_TEXT[String(st.state)] ?? String(st.state ?? '–')}
              </span>
              <span>
                <span className="k">floor</span>
                {String(st.floor ?? '–')}
              </span>
              <span>
                <span className="k">user</span>
                {st.userSpeaking ? 'talking' : 'quiet'}
              </span>
              <span>
                <span className="k">Gemma</span>
                {pending ? (
                  <>
                    <i className="tape-live" />
                    {pending.ahead ? 'writing ahead' : 'writing'} {waitedMs !== undefined ? <span className="num">{waitedMs} ms</span> : '(no answer recorded)'}
                  </>
                ) : (
                  'between calls'
                )}
              </span>
              {Number(st.ahead) > 0 && (
                <span>
                  <span className="k">ahead</span>
                  {String(st.ahead)} cells
                </span>
              )}
            </span>
            <span className="tape-count">
              {rows.length} cells · {rows.filter((r) => r.xid).length} calls
            </span>
            {open && picked && (
              <button className="link" onClick={() => pick(null)} title="Show the newest call again, and follow the tape">
                Follow newest
              </button>
            )}
          </>
        )}
      </div>
      {open && m?.node && (
        <div className="tape-split">
          <div className="tape-main">
            {rows.length > 0 && (
              <OddsChart
                m={m}
                rows={rows}
                selN={sel?.n}
                centre={!!picked}
                pending={pendingN !== undefined ? { n: pendingN, ms: waitedMs ?? 0 } : undefined}
                onPick={(p) => pick(p, true)}
              />
            )}
            <div
              className="tape-table"
              ref={bodyRef}
              tabIndex={0}
              onKeyDown={keys}
              onScroll={(e) => {
                const el = e.currentTarget;
                stick.current = !picked && el.scrollTop + el.clientHeight >= el.scrollHeight - 8;
              }}
            >
              <div className="tape-tr tape-th">
                <span className="num">Time (s)</span>
                <span>User</span>
                <span>Agent</span>
                <span>Gemma wrote</span>
                <span className="num" title="Gemma's odds for what it wrote (first token, before bias and grammar)">
                  Odds
                </span>
                <span title="The most likely action it did not write">Next best</span>
                <span className="num" title="How long the call took">
                  Call ms
                </span>
                <span>Notes</span>
              </div>
              {rows.length === 0 && <p className="muted tape-empty">Waiting for the first cell.</p>}
              {groupRows(visible).map((it) => {
                if (it.kind === 'row')
                  return (
                    <div key={it.row.seq}>
                      {rowEl(it.row)}
                      {notesAfter.get(it.row.n)?.map((k) => (
                        <div key={k.seq} className="tape-tr tape-fold">
                          <span />
                          <span className="tape-span bad">
                            ✕ {k.ahead ? `${k.ahead} cell${k.ahead === 1 ? '' : 's'} written ahead` : 'Writing ahead'} dropped: {k.why}
                          </span>
                        </div>
                      ))}
                    </div>
                  );
                const a = it.rows[0];
                const b = it.rows[it.rows.length - 1];
                const isOpen = unfolded.has(a.seq);
                return (
                  <div key={a.seq}>
                    <div
                      className="tape-tr tape-fold"
                      onClick={() => {
                        const s = new Set(unfolded);
                        if (isOpen) s.delete(a.seq);
                        else s.add(a.seq);
                        setUnfolded(s);
                      }}
                    >
                      <span className="num muted">{a.t.toFixed(2)}</span>
                      <span className="tape-span muted">
                        {isOpen ? '▾' : '▸'} {it.rows.length} idle cells, to {b.t.toFixed(2)} s: nobody talking, no calls
                      </span>
                    </div>
                    {isOpen && it.rows.map(rowEl)}
                  </div>
                );
              })}
              {m.ahead.map((a) => {
                const alt = runnerUp(a.odds, a.action);
                return (
                  <div key={a.seq} data-seq={a.seq} className={`tape-tr ahead${sel === a ? ' sel' : ''}`} onClick={() => pick({ kind: 'ahead', seq: a.seq })}>
                    <span className="num">{lineTime(a.line).toFixed(2)}</span>
                    <span />
                    <span>{STATE_TEXT[parseOpen(a.line)?.state ?? ''] ?? ''}</span>
                    <AgentCell action={a.action} />
                    <span className="num">{a.odds ? pct(a.odds[a.action] ?? 0) : ''}</span>
                    <span>{alt && `${alt[0]} ${pct(alt[1])}`}</span>
                    <span className="num">{a.ms ?? ''}</span>
                    <span className="small">written ahead, as if the user stays silent</span>
                  </div>
                );
              })}
              {pending && (
                <div className={`tape-tr pending${pending.ahead ? ' ahead' : ''}`}>
                  <span className="num muted">{pendingOpen ? pendingOpen.t.toFixed(2) : ''}</span>
                  <span>{pendingOpen && <UserCell words={pendingOpen.words} voice={pendingOpen.voice} />}</span>
                  <span className="muted">{STATE_TEXT[pendingOpen?.state ?? ''] ?? pendingOpen?.state}</span>
                  <span>
                    <span className="tape-caret">▍</span>
                  </span>
                  <span />
                  <span />
                  <span className="num">{waitedMs ?? ''}</span>
                  <span className="small muted">{pending.ahead ? 'writing ahead' : 'writing'}</span>
                </div>
              )}
            </div>
          </div>
          <div className="tape-detail">{sel ? <Detail m={m} sel={sel} /> : <p className="muted">No call yet.</p>}</div>
        </div>
      )}
    </div>
  );
}
