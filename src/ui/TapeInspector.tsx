import { useEffect, useRef, useState, type ReactNode } from 'react';
import type { Trajectory } from '../core/trajectory';
import { TapeModel, openLine, splitPrompt, type AheadRow, type SpecNote, type TapeCall, type TapeRow } from './tape-model';

/** Each action's colour, as on the dashboard's tape row. */
const ACTION_VAR: Record<string, string> = {
  listen: 'var(--muted)',
  respond: 'var(--accent)',
  resume: 'var(--accent)',
  continue: 'var(--cat-llm)',
  finish: 'var(--cat-llm)',
  backchannel: 'var(--ok)',
};
const pct = (p: number) => (p >= 0.995 ? '100%' : p >= 0.1 ? `${Math.round(p * 100)}%` : p >= 0.001 ? `${(p * 100).toFixed(1)}%` : p > 0 ? '<0.1%' : '0');
const vis = (s: string) => JSON.stringify(s).slice(1, -1);

/** The allowed actions' odds as one stacked bar. */
function OddsBar({ odds, width = 72 }: { odds: Record<string, number>; width?: number }) {
  const entries = Object.entries(odds);
  const title = entries.map(([a, p]) => `${a} ${pct(p)}`).join(' · ');
  return (
    <span className="tape-odds" style={{ width }} title={`Gemma's odds before bias and grammar: ${title}`}>
      {entries.map(([a, p]) => (
        <span key={a} style={{ width: `${p * 100}%`, background: ACTION_VAR[a] ?? 'var(--muted)' }} />
      ))}
    </span>
  );
}

/** The runner-up among the allowed actions: what Gemma nearly did. */
function runnerUp(odds: Record<string, number> | undefined, action: string): string | null {
  if (!odds) return null;
  const alt = Object.entries(odds)
    .filter(([a]) => a !== action)
    .sort((a, b) => b[1] - a[1])[0];
  return alt ? `${alt[0]} ${pct(alt[1])}` : null;
}

/** The agent's side of a row, with the words the user never heard struck through and `[cut]` where the voice stopped. */
function AgentPart({ row }: { row: TapeRow }) {
  const heard = row.say.slice(0, row.say.length - row.unsaid);
  const gone = row.say.slice(row.say.length - row.unsaid);
  const i = row.line.indexOf(' | agent (');
  const state = i >= 0 ? row.line.slice(i + 3, row.line.indexOf(':', i + 3) + 1) : `agent (${row.state}):`;
  return (
    <span className="tape-agent">
      {state} <b style={{ color: ACTION_VAR[row.action] }}>{row.action}</b>
      {row.say.length > 0 && (
        <>
          {' "'}
          {heard.join(' ')}
          {row.cut && <span className="tape-cutmark">{heard.length ? ' ' : ''}[cut]</span>}
          {gone.length > 0 && <s title="sent to TTS, never heard: taken off the tape">{` ${gone.join(' ')}`}</s>}
          {'"'}
        </>
      )}
    </span>
  );
}

function Flags({ row }: { row: TapeRow }) {
  const flags: [string, string][] = [];
  if (row.forced) flags.push([row.forced === 'idle' ? 'idle' : row.forced, `no call: ${row.forced}`]);
  if (row.spec) flags.push([row.spec === 'adopted' ? 'ahead' : row.spec, 'decided in advance, as if the user stayed silent, and confirmed by this cell']);
  if (row.dropped) flags.push(['dropped', row.dropped]);
  if (row.failed) flags.push(['failed', row.wrote ? `wrote ${row.wrote}` : 'the call failed']);
  if (row.late) flags.push(['late', 'answered after the cell had ended']);
  return (
    <>
      {flags.map(([f, title]) => (
        <span key={f} className={`tape-flag${row.failed || row.dropped ? ' bad' : ''}`} title={title}>
          {f}
        </span>
      ))}
    </>
  );
}

function CallDetail({ call, odds, allowed }: { call?: TapeCall; odds?: Record<string, number>; allowed?: string[] }) {
  const [copied, setCopied] = useState(false);
  if (!call) return <p className="muted">No call was made for this cell.</p>;
  const res = call.response;
  const probs = res?.completion_probabilities ?? [];
  const first = probs[0]?.top_logprobs ?? [];
  const actions = allowed ?? Object.keys(odds ?? {});
  const others = first.filter((t) => !actions.some((a) => ` ${a}`.startsWith(t.token) && t.token.trim()));
  const timings = res?.timings ?? {};
  const prompt = call.body.prompt ?? '';
  const { head, tape } = splitPrompt(prompt);
  const lines = tape.split('\n');
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
            Odds for each action <small className="muted">first token, before bias and grammar</small>
          </h4>
          <table className="tape-oddtable">
            <tbody>
              {Object.entries(odds)
                .sort((a, b) => b[1] - a[1])
                .map(([a, p]) => (
                  <tr key={a}>
                    <td style={{ color: ACTION_VAR[a] }}>{a}</td>
                    <td>
                      <span className="tape-bar" style={{ width: `${Math.max(1, p * 100)}%`, background: ACTION_VAR[a] }} />
                    </td>
                    <td className="num">{pct(p)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
          {others.length > 0 && (
            <p className="muted small">
              Outside the grammar: {others.map((t) => `“${vis(t.token)}” ${pct(Math.exp(t.logprob))}`).join(' · ')}
            </p>
          )}
        </section>
      )}
      {probs.length > 0 && (
        <section>
          <h4>
            Written tokens <small className="muted">chosen, then its alternatives</small>
          </h4>
          <div className="tape-tokens">
            {probs.map((t, i) => (
              <div key={i} className="tape-token">
                <span className="tok">“{vis(t.token)}”</span>
                <span className="num">{pct(Math.exp(t.logprob))}</span>
                <span className="muted">
                  {(t.top_logprobs ?? [])
                    .filter((x) => x.token !== t.token)
                    .slice(0, 4)
                    .map((x) => `“${vis(x.token)}” ${pct(Math.exp(x.logprob))}`)
                    .join('  ')}
                </span>
              </div>
            ))}
          </div>
        </section>
      )}
      <section>
        <h4>Call</h4>
        <p className="small">
          {call.t1 !== undefined ? `${Math.round(call.t1 - call.t0)} ms` : 'in flight'}
          {timings.prompt_n !== undefined && ` · prefill ${timings.prompt_n} tokens in ${Math.round(timings.prompt_ms ?? 0)} ms (cached ${timings.cache_n ?? '?'})`}
          {timings.predicted_n !== undefined && ` · wrote ${timings.predicted_n} tokens in ${Math.round(timings.predicted_ms ?? 0)} ms`}
          {timings.draft_n ? ` · drafts ${timings.draft_n_accepted ?? 0}/${timings.draft_n} accepted` : ''}
          {call.ahead && ' · written ahead'}
          {call.closed !== undefined && !res && ` · closed: ${JSON.stringify(call.closed)}`}
        </p>
        {res?.content !== undefined && (
          <p className="small">
            Wrote <code>{vis(res.content)}</code>
          </p>
        )}
        {call.body.logit_bias !== undefined && (
          <p className="small">
            Logit bias <code>{JSON.stringify(call.body.logit_bias)}</code>
          </p>
        )}
      </section>
      <section>
        <h4>
          Prompt <small className="muted">{prompt.length} chars</small>{' '}
          <button className="link" onClick={copy}>
            {copied ? 'copied' : 'copy'}
          </button>
        </h4>
        <details>
          <summary className="muted small">Head: instructions and example ({head.length} chars)</summary>
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

/**
 * The tape stage (turn.tape) as it runs: every line Gemma has in front of it, what it wrote and how
 * sure it was, the cells written ahead, the call in flight, and for any cell the exact prompt, grammar
 * and token odds of its call.
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
  const [picked, setPicked] = useState<{ kind: 'row' | 'ahead'; seq: number } | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  useEffect(() => setPicked(null), [trajectory]);
  // A cell clicked on the dashboard's tape row: select it here and scroll to it.
  useEffect(() => {
    const row = focus && modelRef.current?.rows.find((r) => focus.seqs.includes(r.seq));
    if (!row) return;
    setPicked({ kind: 'row', seq: row.seq });
    stick.current = false;
    requestAnimationFrame(() => bodyRef.current?.querySelector(`[data-seq="${row.seq}"]`)?.scrollIntoView({ block: 'center' }));
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
  const selRow = sel && 'say' in sel ? (sel as TapeRow) : undefined;
  const st = m?.state ?? {};
  const pending = m?.pending;
  const waited = pending ? (live ? `${Math.max(0, Math.round(now() - pending.t0))} ms` : 'no answer recorded') : '';

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
          <span className="muted tape-status">
            {String(st.state ?? '?')} · floor {String(st.floor ?? '?')}
            {st.userSpeaking ? ' · user speaking' : ''}
            {pending ? ` · ${pending.ahead ? 'writing ahead' : 'Gemma writing'} ${waited}` : ''}
            {Number(st.ahead) > 0 ? ` · ${st.ahead} cells ahead` : ''}
            {` · ${rows.length} cells, ${rows.filter((r) => r.xid).length} calls`}
          </span>
        )}
        {open && picked && (
          <button className="link" onClick={() => setPicked(null)} title="Show the newest call again">
            follow newest
          </button>
        )}
      </div>
      {open && m?.node && (
        <div className="tape-split">
          <div
            className="event-log-body tape-rows"
            ref={bodyRef}
            onScroll={(e) => {
              const el = e.currentTarget;
              stick.current = el.scrollTop + el.clientHeight >= el.scrollHeight - 8;
            }}
          >
            {visible.map((r) => {
              const i = r.line.indexOf(' | agent (');
              const alt = runnerUp(r.odds, r.action);
              return (
                <div key={r.seq}>
                  <div
                    data-seq={r.seq}
                    className={`tape-row${sel === r ? ' sel' : ''}${r.xid ? '' : ' nocall'}`}
                    onClick={() => setPicked({ kind: 'row', seq: r.seq })}
                    title={`written ${(r.written / 1000).toFixed(2)} s into the session`}
                  >
                    <span className="tape-line">
                      <span className="tape-user">{i >= 0 ? r.line.slice(0, i + 3) : r.line}</span>
                      {i >= 0 && <AgentPart row={r} />}
                    </span>
                    <span className="tape-meta">
                      {r.odds && <OddsBar odds={r.odds} />}
                      <span className="tape-alt">{alt ?? ''}</span>
                      <span className="num">{r.ms !== undefined ? `${r.ms} ms` : ''}</span>
                      <Flags row={r} />
                    </span>
                  </div>
                  {notesAfter.get(r.n)?.map((k) => (
                    <div key={k.seq} className="tape-note">
                      ✕ written ahead, dropped ({k.why}){k.ahead ? `: ${k.ahead} cell${k.ahead === 1 ? '' : 's'}` : ''}
                    </div>
                  ))}
                </div>
              );
            })}
            {m.ahead.map((a) => {
              const alt = runnerUp(a.odds, a.action);
              return (
                <div key={a.seq} className={`tape-row ahead${sel === a ? ' sel' : ''}`} onClick={() => setPicked({ kind: 'ahead', seq: a.seq })} title="written ahead, as if the user stays silent">
                  <span className="tape-line">{a.line}</span>
                  <span className="tape-meta">
                    {a.odds && <OddsBar odds={a.odds} />}
                    <span className="tape-alt">{alt ?? ''}</span>
                    <span className="num">{a.ms !== undefined ? `${a.ms} ms` : ''}</span>
                    <span className="tape-flag">ahead</span>
                  </span>
                </div>
              );
            })}
            {pending && (
              <div className={`tape-row pending${pending.ahead ? ' ahead' : ''}`}>
                <span className="tape-line">
                  {openLine(pending.body.prompt)} <span className="tape-caret">▍</span>
                </span>
                <span className="tape-meta">
                  <span className="num">{waited}</span>
                  <span className="tape-flag">{pending.ahead ? 'writing ahead' : 'writing'}</span>
                </span>
              </div>
            )}
          </div>
          <div className="tape-detail">
            {sel ? (
              <>
                <pre className="tape-pre tape-selline">{sel.line}</pre>
                {selRow?.forced && <p className="small">No call: {selRow.forced}.</p>}
                {selRow?.dropped && <p className="small warn">Not used: {selRow.dropped}.</p>}
                <CallDetail call={m.callOf(sel)} odds={sel.odds} allowed={selRow?.allowed} />
              </>
            ) : (
              <p className="muted">No call yet.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
