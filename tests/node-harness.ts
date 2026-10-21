import { resolveParams, type IOHandlers, type NodeContext, type NodeDef } from '../src/core/node';
import type { Json, Msg } from '../src/core/types';

/** Drives one node by hand: a manual clock, timers, fake IO (with closes), and what it emitted. */
export function harness(def: NodeDef, params: Record<string, unknown> = {}, id = def.type) {
  let now = 0;
  const timers: { at: number; fn: () => void; live: boolean }[] = [];
  const out: { port: string; msg: Msg }[] = [];
  const ios: { channel: string; request: Json; h: IOHandlers; closed: boolean; sent: Json[] }[] = [];
  let media = 0;
  const ctx: NodeContext = {
    id,
    params: resolveParams(def, params as never),
    now: () => now,
    cause: undefined,
    emit: (port, msg) => (out.push({ port, msg }), out.length),
    setState() {},
    log() {},
    setTimer(ms, fn) {
      const t = { at: now + ms, fn, live: true };
      timers.push(t);
      return { clear: () => void (t.live = false) };
    },
    io: {
      open(channel, request, h) {
        const x = { channel, request, h, closed: false, sent: [] as Json[] };
        ios.push(x);
        return { xid: String(ios.length), send: (d: Json) => void x.sent.push(d), close: () => void (x.closed = true), get closed() { return x.closed; } };
      },
    },
    media: { append: (track: string, s: Float32Array) => ({ track, start: media, end: (media += s.length) }) },
  };
  const node = def.create(ctx as never);
  node.start?.();
  const advance = (to: number) => {
    for (;;) {
      const t = timers.filter((x) => x.live && x.at <= to).sort((a, b) => a.at - b.at)[0];
      if (!t) break;
      t.live = false;
      now = t.at;
      t.fn();
    }
    now = to;
  };
  const send = (t: number, port: string, msg: Msg) => (advance(t), node.onInput?.(port, msg));
  const of = (type: string) => out.filter((o) => o.msg.type === type).map((o) => o.msg);
  return { send, advance, out, of, ios };
}

