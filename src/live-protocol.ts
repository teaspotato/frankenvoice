// The page's one socket to the server (/api/live). The pipeline runs in the server; the page
// captures the mic, plays the agent and mirrors the run's trajectory for the dashboard.
//
// Text frames are JSON messages. Binary frames carry a JSON header and a payload:
//   [u32 LE header length][header JSON][payload bytes]
//
// Page -> server
//   {type:'ping', id, t}                       clock sync; answered by pong
//   {type:'start', graph, env, autosave, title} run the graph live (title: model naming the session, '' = off)
//   {type:'stop'} {type:'pause'} {type:'resume'}
//   {type:'mic.ready', ...device} {type:'mic.error', error}
//   {type:'speaker.late', id, at, ms}          a playback run started `ms` after its session time `at`
//   {type:'speaker.gate', id, at, pos, stop, silent}  the page gate cut playback at session time `at`, on user speech from mic
//                                              sample `pos`; the sound ended at `stop` (syllable end) and was silent at `silent`
//   bin {k:'mic', frame, ctxTime} + f32 PCM    a capture frame
//   bin {k:'replay', graph?, liveChannels, audible} + session bundle   replay it on the server
// Server -> page
//   {type:'pong', id, t, s}                    s: server clock (performance.now) when answered
//   {type:'session', meta}                     a run started; events follow
//   {type:'events', events, clock:{now, s}}    trajectory events in order, and the session clock at server time s
//   {type:'state', states}                     latest state per node
//   bin {k:'track', name, sampleRate} + f32    samples appended to a media track (before the events using them)
//   {type:'mic.open', request} {type:'mic.close'}
//   bin {k:'speaker', id, cmd, ..., now, s} + f32   a speaker command, with the session clock when sent
//                                              (cmd 'gate', until: the page may stop playback on its own VAD until then; null: not)
//   {type:'speaker.close', id}
//   {type:'ended', id, stoppedAt?}             the run stopped; its full bundle follows
//   bin {k:'bundle'} + session bundle
//   {type:'saved', id, path} {type:'renamed', from, to, path} {type:'notice', text}
//   {type:'users', pages, sessions, max}       pages connected, sessions using the GPU box, the cap; on connect and on change
//   {type:'full', sessions, max, text}         a start (or a replay with live channels) refused: the box runs `max` sessions
//
// Each page has its own run; pages don't take over each other.

import type { Json } from './core/types';

/** Where the server's turn model is (POST /api/turn-model), for the page to show before any session runs. */
export interface TurnModelLoad {
  state: 'loading' | 'ready' | 'failed';
  /** The latest progress or status message. */
  last?: { type: string; [k: string]: Json };
  error?: string;
}

export type Header = { k: string; [key: string]: Json };

export function frame(header: Header, payload: Uint8Array = new Uint8Array()): Uint8Array<ArrayBuffer> {
  const h = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(4 + h.length + payload.byteLength);
  new DataView(out.buffer).setUint32(0, h.length, true);
  out.set(h, 4);
  out.set(payload, 4 + h.length);
  return out;
}

export function unframe(buf: Uint8Array): { header: Header; payload: Uint8Array } {
  const n = new DataView(buf.buffer, buf.byteOffset, buf.byteLength).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(buf.subarray(4, 4 + n))) as Header;
  return { header, payload: buf.subarray(4 + n) };
}

/** Float32 samples from a payload, copied (payloads are rarely 4-byte aligned). */
export const f32Of = (payload: Uint8Array) => new Float32Array(payload.slice().buffer);
