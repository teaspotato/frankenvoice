// Core data model shared by the runtime, nodes, recorder and replay.
//
// Time model: every event carries `t`, its arrival time in milliseconds on the
// session clock (0 = session start). Events that describe audio additionally
// carry `media`, a sample range on a named media track, so the log knows both
// *when* something happened and *which audio* it is about.

export type Json = null | boolean | number | string | Json[] | { [k: string]: Json };
export type JsonObject = { [k: string]: Json };

/** A range of samples on a media track (float32 PCM). `end` is exclusive. */
export interface MediaRef {
  track: string;
  start: number;
  end: number;
}

/** A range of bytes on a byte track (raw IO payloads). */
export interface BinRef {
  track: string;
  offset: number;
  length: number;
}

export type EventKind =
  | 'session.start'
  | 'session.stop'
  | 'node.start'
  | 'node.stop'
  | 'msg' // a node emitted a message on an output port
  | 'state' // a node changed a piece of internal state
  | 'log'
  | 'timer.set'
  | 'timer.fire'
  | 'timer.clear'
  | 'io.open' // a node opened an exchange with the outside world
  | 'io.send' // node -> outside
  | 'io.recv' // outside -> node; the only source of nondeterminism besides timers
  | 'io.close'
  | 'io.diverge'; // replay could not reproduce an exchange exactly

export interface TrajEvent {
  seq: number;
  t: number;
  node: string;
  kind: EventKind;
  port?: string;
  data?: Json;
  media?: MediaRef;
  bin?: BinRef;
  /** Seqs of the events that caused this one. */
  cause?: number[];
}

/** Audio flowing between nodes. Samples live in memory; the log stores only `media`. */
export interface AudioMsg {
  type: 'audio';
  track: string;
  start: number;
  end: number;
  sampleRate: number;
  samples: Float32Array;
  [k: string]: unknown;
}

export interface DataMsg {
  type: string;
  [k: string]: unknown;
}

export type Msg = AudioMsg | DataMsg;

export function isAudioMsg(m: Msg): m is AudioMsg {
  return m.type === 'audio' && (m as AudioMsg).samples instanceof Float32Array;
}
