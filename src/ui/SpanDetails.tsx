import { useMemo, useState } from 'react';
import type { Trajectory } from '../core/trajectory';
import type { Json, TrajEvent } from '../core/types';
import { JsonView } from './JsonView';

/** One IO exchange (an API call, a websocket) as recorded: what was sent and what came back. */
interface Exchange {
  xid: string;
  channel: string;
  open?: TrajEvent;
  sends: TrajEvent[];
  recvs: TrajEvent[];
  close?: TrajEvent;
}

type Obj = Record<string, Json>;
const obj = (j: Json | undefined): Obj => (j && typeof j === 'object' && !Array.isArray(j) ? (j as Obj) : {});

/** A long conversation shows only its last this-many messages until expanded. */
const RECENT_TURNS = 12;

/** Streams with this many messages are shown only around the selected item, not whole (an ASR socket, the mic). */
const LONG = 40;

/**
 * Details of the item selected on the timeline: its events, and every IO exchange behind them with the
 * full request and response as recorded. A streamed Anthropic reply is also shown reassembled.
 */
export function SpanDetails({ trajectory, seqs, onClose }: { trajectory: Trajectory; seqs: number[]; onClose(): void }) {
  const { events, exchanges, related } = useMemo(() => collect(trajectory, seqs), [trajectory, seqs]);
  const t0 = events[0]?.t ?? 0;
  const t1 = events[events.length - 1]?.t ?? t0;
  return (
    <section className="span-details">
      <h3>
        Selected <span className="muted">{events[0]?.node}</span>
        <button className="link" onClick={onClose} title="Clear the selection">
          ×
        </button>
      </h3>
      <p className="muted">
        {(t0 / 1000).toFixed(3)} s{t1 > t0 ? ` → ${(t1 / 1000).toFixed(3)} s (${Math.round(t1 - t0)} ms)` : ''} · {events.length} event
        {events.length === 1 ? '' : 's'}
      </p>
      {exchanges.map((x) => (
        <ExchangeView key={x.xid} x={x} related={related} />
      ))}
      <details>
        <summary>Events ({events.length})</summary>
        <pre>{events.map((e) => `${(e.t / 1000).toFixed(3)} #${e.seq} ${e.kind} ${e.node}${e.port ? `.${e.port}` : ''} ${short(e.data)}`).join('\n')}</pre>
      </details>
    </section>
  );
}

function ExchangeView({ x, related }: { x: Exchange; related: Set<number> }) {
  const request = obj(x.open?.data).request;
  const long = x.recvs.length > LONG || x.sends.length > LONG;
  const recvs = long ? x.recvs.filter((e) => related.has(e.seq)) : x.recvs;
  const sends = (long ? x.sends.filter((e) => related.has(e.seq)) : x.sends).filter((e) => obj(e.data).payload != null);
  const payloads = recvs.map((e) => obj(e.data).payload);
  const reply = reassembleAnthropic(payloads) ?? reassembleOpenAI(payloads);
  const chat = chatOf(request);
  const close = obj(x.close?.data);
  const dur = x.open && x.close ? Math.round(x.close.t - x.open.t) : undefined;
  return (
    <div className="exchange">
      <div className="exchange-head">
        <b>{x.channel}</b> <span className="muted">{x.xid}</span>
        {dur !== undefined && <span className="muted"> · {dur} ms</span>}
        {x.close && <span className={close.payload && obj(close.payload).error ? 'warn' : 'muted'}> · closed by {String(close.by)}</span>}
      </div>
      {chat ? (
        <>
          <Chat turns={chat.turns} reply={reply} />
          <div className="muted reply-meta">
            {Object.entries(chat.params)
              .map(([k, v]) => `${k}: ${paramText(k, v)}`)
              .join(' · ')}
          </div>
          {reply && <JsonView value={reply.usage} open={0} label="usage" />}
          <details>
            <summary>Raw request</summary>
            <JsonView value={request} open={1} />
            {sends.map((e) => (
              <JsonView key={e.seq} value={obj(e.data).payload} />
            ))}
          </details>
        </>
      ) : (
        <details open={!long}>
          <summary>Request</summary>
          <JsonView value={request} open={3} />
          {sends.map((e) => (
            <JsonView key={e.seq} value={obj(e.data).payload} />
          ))}
        </details>
      )}
      {reply && !chat && (
        <details open>
          <summary>Response</summary>
          <Chat turns={[]} reply={reply} />
          <JsonView value={reply.usage} open={0} label="usage" />
        </details>
      )}
      <details open={!reply && !long}>
        <summary>
          {reply ? 'Raw response' : 'Response'} ({recvs.length}
          {long ? ` of ${x.recvs.length} messages, those behind the selection` : ` message${recvs.length === 1 ? '' : 's'}`}
          {x.recvs.some((e) => e.bin) ? ', with binary data' : ''})
        </summary>
        {recvs.every((e) => obj(e.data).payload == null && e.bin) ? (
          <p className="muted">
            {recvs.length} binary chunks, {recvs.reduce((a, e) => a + (e.bin?.length ?? 0), 0).toLocaleString()} bytes (audio: see the Playback lane)
          </p>
        ) : (
          recvs.map((e) =>
            obj(e.data).payload == null && e.bin ? (
              <p key={e.seq} className="muted">
                ({e.bin.length} bytes)
              </p>
            ) : (
              <JsonView key={e.seq} value={obj(e.data).payload} open={recvs.length > 3 ? 0 : 2} label={`${((e.t - (x.open?.t ?? e.t)) / 1000).toFixed(3)} s`} />
            ),
          )
        )}
      </details>
      {x.close && close.payload != null && (
        <details open={!!obj(close.payload).error}>
          <summary>Close</summary>
          <JsonView value={close.payload} />
        </details>
      )}
    </div>
  );
}

function collect(traj: Trajectory, seqs: number[]) {
  const bySeq = new Map(traj.events.map((e) => [e.seq, e]));
  const events = seqs
    .map((s) => bySeq.get(s))
    .filter((e): e is TrajEvent => !!e)
    .sort((a, b) => a.seq - b.seq);
  // Exchanges touched by the selected events or by what directly caused them (an LLM delta is caused by a received chunk).
  const related = new Set<number>(seqs);
  const xids = new Set<string>();
  for (const e of events) {
    const own = obj(e.data).xid;
    if (typeof own === 'string') xids.add(own);
    for (const c of e.cause ?? []) {
      related.add(c);
      const xid = obj(bySeq.get(c)?.data).xid;
      if (typeof xid === 'string') xids.add(xid);
    }
  }
  // A request opened while handling the same input as the selection (an LLM call next to llm.start).
  const causes = new Set(events.flatMap((e) => e.cause ?? []));
  for (const e of traj.events) if (e.kind === 'io.open' && e.node === events[0]?.node && e.cause?.some((c) => causes.has(c))) xids.add(String(obj(e.data).xid));
  const exchanges: Exchange[] = [];
  for (const xid of xids) {
    const x: Exchange = { xid, channel: '', sends: [], recvs: [] };
    for (const e of traj.events) {
      if (!e.kind.startsWith('io.') || obj(e.data).xid !== xid) continue;
      if (e.kind === 'io.open') {
        x.open = e;
        x.channel = String(obj(e.data).channel ?? '');
      } else if (e.kind === 'io.send') x.sends.push(e);
      else if (e.kind === 'io.recv') x.recvs.push(e);
      else if (e.kind === 'io.close') x.close = e;
    }
    if (!x.channel) x.channel = xid.split('/')[1]?.split('#')[0] ?? xid;
    exchanges.push(x);
  }
  return { events, exchanges, related };
}

interface Block {
  type: string;
  text: string;
}
interface Reply {
  blocks: Block[];
  stop: string;
  usage: Json;
}

/** The reply an Anthropic message stream adds up to: text, thinking, tool calls, stop reason and usage. */
export function reassembleAnthropic(payloads: Json[]): Reply | null {
  const evs = payloads.map(obj);
  if (!evs.some((p) => p.type === 'message_start')) return null;
  const blocks: Block[] = [];
  let stop = '';
  let usage: Obj = {};
  for (const p of evs) {
    if (p.type === 'message_start') usage = { ...obj(obj(p.message).usage) };
    else if (p.type === 'content_block_start') {
      const b = obj(p.content_block);
      const type = b.type === 'redacted_thinking' ? 'thinking' : String(b.type);
      blocks[Number(p.index)] = { type, text: typeof b.text === 'string' ? b.text : b.type === 'tool_use' ? `${String(b.name)} ` : '' };
    } else if (p.type === 'content_block_delta') {
      const d = obj(p.delta);
      const b = (blocks[Number(p.index)] ??= { type: 'text', text: '' });
      b.text += String(d.text ?? d.thinking ?? d.partial_json ?? '');
    } else if (p.type === 'message_delta') {
      stop = String(obj(p.delta).stop_reason ?? '');
      usage = { ...usage, ...obj(p.usage) };
    } else if (p.type === 'error') {
      blocks.push({ type: 'error', text: clip(pretty(p.error)) });
      stop = 'error';
    }
  }
  return { blocks: blocks.filter(Boolean), stop, usage };
}

/** The reply an OpenAI-compatible chat completion (streamed or not) adds up to: text, reasoning, tool calls, errors. */
export function reassembleOpenAI(payloads: Json[]): Reply | null {
  const evs = payloads.map(obj).filter((p) => Array.isArray(p.choices) || p.error != null);
  if (!evs.some((p) => Array.isArray(p.choices))) return null;
  let text = '';
  let reasoning = '';
  let stop = '';
  let usage: Json = {};
  const calls: { name: string; args: string }[] = [];
  const errors: Block[] = [];
  const addCalls = (tc: Json | undefined) => {
    if (!Array.isArray(tc)) return;
    tc.map(obj).forEach((t, i) => {
      const f = obj(t.function);
      const c = (calls[typeof t.index === 'number' ? t.index : i] ??= { name: '', args: '' });
      if (typeof f.name === 'string') c.name += f.name;
      if (typeof f.arguments === 'string') c.args += f.arguments;
    });
  };
  for (const p of evs) {
    if (p.error != null) {
      errors.push({ type: 'error', text: clip(pretty(p.error)) });
      stop = 'error';
    }
    if (p.usage) usage = p.usage;
    if (!Array.isArray(p.choices)) continue;
    const c = p.choices.map(obj).find((x) => (typeof x.index === 'number' ? x.index : 0) === 0);
    if (!c) continue;
    const d = obj(c.delta ?? c.message);
    if (typeof d.content === 'string') text += d.content;
    const r = d.reasoning ?? d.reasoning_content;
    if (typeof r === 'string') reasoning += r;
    addCalls(d.tool_calls);
    if (c.finish_reason) stop = String(c.finish_reason);
  }
  const blocks: Block[] = [
    ...(reasoning ? [{ type: 'thinking', text: reasoning }] : []),
    ...(text ? [{ type: 'text', text }] : []),
    ...calls.filter(Boolean).map((c) => ({ type: 'tool_use', text: `${c.name} ${clip(c.args)}` })),
    ...errors,
  ];
  return { blocks, stop, usage };
}

/**
 * A chat request's conversation as turns of blocks, without the system prompt (it is in the LLM settings), plus its other
 * parameters. Takes an Anthropic request or an OpenAI-compatible one (`{ endpoint, body }`).
 */
export function chatOf(request: Json | undefined): { turns: { role: string; blocks: Block[] }[]; params: Obj } | null {
  const outer = obj(request);
  const r = Array.isArray(outer.messages) ? outer : obj(outer.body);
  if (!Array.isArray(r.messages)) return null;
  const turns = r.messages.map(obj).flatMap((m) => {
    const role = String(m.role ?? '');
    if (role === 'system' || role === 'developer') return [];
    const content = typeof m.content === 'string' ? [{ type: 'text', text: m.content } as Obj] : Array.isArray(m.content) ? m.content.map(obj) : [];
    const blocks = content.map((b): Block => {
      if (b.type === 'text') return { type: role === 'tool' ? 'tool_result' : 'text', text: String(b.text ?? '') };
      if (b.type === 'thinking') return { type: 'thinking', text: String(b.thinking ?? '') };
      if (b.type === 'redacted_thinking') return { type: 'thinking', text: '' };
      if (b.type === 'tool_use') return { type: 'tool_use', text: `${String(b.name)} ${clip(JSON.stringify(b.input))}` };
      if (b.type === 'tool_result') return { type: 'tool_result', text: clip(typeof b.content === 'string' ? b.content : JSON.stringify(b.content)) };
      return { type: String(b.type), text: clip(JSON.stringify(b)) };
    });
    // OpenAI-style tool calls and results live outside `content`.
    if (Array.isArray(m.tool_calls))
      for (const t of m.tool_calls.map(obj))
        blocks.push({ type: 'tool_use', text: `${String(obj(t.function).name)} ${clip(String(obj(t.function).arguments ?? ''))}` });
    return [{ role: role === 'tool' ? 'tool' : role, blocks }];
  });
  const params: Obj = {};
  if (r !== outer && outer.endpoint != null) params.endpoint = outer.endpoint;
  for (const [k, v] of Object.entries(r)) if (!['messages', 'system', 'stream'].includes(k)) params[k] = v;
  return { turns, params };
}

/** One short line for a request parameter: tool lists as a count, long values cut. */
function paramText(k: string, v: Json): string {
  if (k === 'tools' && Array.isArray(v)) return `${v.length} tool${v.length === 1 ? '' : 's'}`;
  const t = typeof v === 'object' ? JSON.stringify(v) : String(v);
  return t.length > 80 ? t.slice(0, 80) + '…' : t;
}

/** The conversation as message bubbles, ending with this call's reply. */
function Chat({ turns, reply }: { turns: { role: string; blocks: Block[] }[]; reply: Reply | null }) {
  const [all, setAll] = useState(false);
  const hidden = all ? 0 : Math.max(0, turns.length - RECENT_TURNS);
  return (
    <div className="chat">
      {hidden > 0 && (
        <button className="link chat-more" onClick={() => setAll(true)}>
          Show {hidden} earlier message{hidden === 1 ? '' : 's'}
        </button>
      )}
      {turns.slice(hidden).map((t, i) => (
        <Bubble key={hidden + i} role={t.role} blocks={t.blocks} />
      ))}
      {reply && <Bubble role="assistant" blocks={reply.blocks} current note={`this reply · ${reply.stop || 'not finished'}`} />}
    </div>
  );
}

function Bubble({ role, blocks, current, note }: { role: string; blocks: Block[]; current?: boolean; note?: string }) {
  return (
    <div className={`bubble ${role === 'user' ? 'user' : role === 'tool' ? 'tool' : 'assistant'}${current ? ' current' : ''}`}>
      {blocks.map((b, i) =>
        b.type === 'text' ? (
          <div key={i}>{b.text}</div>
        ) : b.type === 'thinking' && !b.text ? (
          <div key={i} className="bubble-thinking">
            thinking (text not returned by the API)
          </div>
        ) : b.type === 'thinking' ? (
          <details key={i} className="bubble-thinking">
            <summary>thinking</summary>
            {b.text}
          </details>
        ) : (
          <div key={i} className="bubble-other">
            <b>{b.type}</b> {b.text}
          </div>
        ),
      )}
      {note && <div className="bubble-note">{note}</div>}
    </div>
  );
}

const pretty = (j: Json | undefined) => (j === undefined ? '(none)' : JSON.stringify(j, null, 2));
/** Long JSON inside a bubble is cut; the raw request below has it whole. */
const clip = (t: string) => (t.length > 500 ? t.slice(0, 500) + `… (+${(t.length - 500).toLocaleString()} chars)` : t);
const short = (j: Json | undefined) => (j === undefined ? '' : JSON.stringify(j).slice(0, 160));
