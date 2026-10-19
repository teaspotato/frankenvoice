import { useMemo } from 'react';
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
              .map(([k, v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : String(v)}`)
              .join(' · ')}
          </div>
          {reply && <JsonView value={reply.usage} open={0} label="usage" />}
          <details>
            <summary>Raw request</summary>
            <JsonView value={request} open={1} />
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
function reassembleAnthropic(payloads: Json[]): Reply | null {
  const evs = payloads.map(obj);
  if (!evs.some((p) => p.type === 'message_start')) return null;
  const blocks: Block[] = [];
  let stop = '';
  let usage: Obj = {};
  for (const p of evs) {
    if (p.type === 'message_start') usage = { ...obj(obj(p.message).usage) };
    else if (p.type === 'content_block_start') {
      const b = obj(p.content_block);
      blocks[Number(p.index)] = { type: String(b.type), text: typeof b.text === 'string' ? b.text : b.type === 'tool_use' ? `${String(b.name)} ` : '' };
    } else if (p.type === 'content_block_delta') {
      const d = obj(p.delta);
      const b = (blocks[Number(p.index)] ??= { type: 'text', text: '' });
      b.text += String(d.text ?? d.thinking ?? d.partial_json ?? '');
    } else if (p.type === 'message_delta') {
      stop = String(obj(p.delta).stop_reason ?? '');
      usage = { ...usage, ...obj(p.usage) };
    } else if (p.type === 'error') return { blocks: [{ type: 'error', text: pretty(p.error) }], stop: 'error', usage };
  }
  return { blocks: blocks.filter(Boolean), stop, usage };
}

/** The reply an OpenAI-compatible chat completion stream adds up to. */
function reassembleOpenAI(payloads: Json[]): Reply | null {
  const evs = payloads.map(obj).filter((p) => Array.isArray(p.choices));
  if (!evs.length) return null;
  let text = '';
  let reasoning = '';
  let stop = '';
  let usage: Json = {};
  for (const p of evs) {
    const c = obj((p.choices as Json[])[0]);
    const d = obj(c.delta);
    if (typeof d.content === 'string') text += d.content;
    const r = d.reasoning ?? d.reasoning_content;
    if (typeof r === 'string') reasoning += r;
    if (c.finish_reason) stop = String(c.finish_reason);
    if (p.usage) usage = p.usage;
  }
  return { blocks: [...(reasoning ? [{ type: 'thinking', text: reasoning }] : []), { type: 'text', text }], stop, usage };
}

/** A chat request's conversation as turns of blocks, without the system prompt (it is in the LLM settings), plus its other parameters. */
function chatOf(request: Json | undefined): { turns: { role: string; blocks: Block[] }[]; params: Obj } | null {
  const r = obj(request);
  if (!Array.isArray(r.messages)) return null;
  const turns = r.messages.map(obj).flatMap((m) => {
    const role = String(m.role ?? '');
    if (role === 'system' || role === 'developer') return [];
    const content = typeof m.content === 'string' ? [{ type: 'text', text: m.content } as Obj] : Array.isArray(m.content) ? m.content.map(obj) : [];
    const blocks = content.map((b): Block => {
      if (b.type === 'text') return { type: 'text', text: String(b.text ?? '') };
      if (b.type === 'thinking') return { type: 'thinking', text: String(b.thinking ?? '') };
      if (b.type === 'redacted_thinking') return { type: 'thinking', text: '' };
      if (b.type === 'tool_use') return { type: 'tool_use', text: `${String(b.name)} ${JSON.stringify(b.input)}` };
      if (b.type === 'tool_result') return { type: 'tool_result', text: typeof b.content === 'string' ? b.content : JSON.stringify(b.content) };
      return { type: String(b.type), text: JSON.stringify(b) };
    });
    return [{ role, blocks }];
  });
  const params: Obj = {};
  for (const [k, v] of Object.entries(r)) if (!['messages', 'system', 'stream'].includes(k)) params[k] = v;
  return { turns, params };
}

/** The conversation as message bubbles, ending with this call's reply. */
function Chat({ turns, reply }: { turns: { role: string; blocks: Block[] }[]; reply: Reply | null }) {
  return (
    <div className="chat">
      {turns.map((t, i) => (
        <Bubble key={i} role={t.role} blocks={t.blocks} />
      ))}
      {reply && <Bubble role="assistant" blocks={reply.blocks} current note={`this reply · ${reply.stop || 'not finished'}`} />}
    </div>
  );
}

function Bubble({ role, blocks, current, note }: { role: string; blocks: Block[]; current?: boolean; note?: string }) {
  return (
    <div className={`bubble ${role === 'user' ? 'user' : 'assistant'}${current ? ' current' : ''}`}>
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
const short = (j: Json | undefined) => (j === undefined ? '' : JSON.stringify(j).slice(0, 160));
