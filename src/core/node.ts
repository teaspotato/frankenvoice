import type { Json, MediaRef, Msg } from './types';

export interface PortDef {
  name: string;
  /** Free-form stream type; the editor only connects ports of equal type (or 'any'). */
  type: string;
  description?: string;
}

export interface ParamDef {
  key: string;
  label: string;
  type: 'string' | 'number' | 'boolean' | 'select' | 'text';
  default: Json;
  options?: string[];
  description?: string;
  /** Show (and use) this param only when other params have one of these values, e.g. {provider: ['deepgram']}. */
  when?: Record<string, Json[]>;
}

export function paramActive(p: ParamDef, values: Record<string, Json>): boolean {
  return !p.when || Object.entries(p.when).every(([k, allowed]) => allowed.includes(values[k]));
}

export interface ChannelDef {
  name: string;
  description?: string;
  /**
   * Output-only channels (e.g. the speaker) receive nothing that affects the
   * graph, so during replay their live driver can still run as a monitor.
   */
  output?: boolean;
}

export interface IOHandlers {
  onRecv?(data: Json, bin: Uint8Array | undefined): void;
  onClose?(data: Json): void;
}

export interface SendOptions {
  bin?: Uint8Array;
  /** Audio this payload was derived from; logged instead of the bytes when `logBin` is false. */
  media?: MediaRef;
  /** Store `bin` in the trajectory (default true). Set false when `media` already identifies it. */
  logBin?: boolean;
}

export interface IOHandle {
  readonly xid: string;
  send(data: Json, opts?: SendOptions): void;
  close(reason?: Json): void;
  readonly closed: boolean;
}

export interface TimerHandle {
  clear(): void;
}

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

/**
 * Everything a node may touch. Nodes must be deterministic functions of the
 * calls they receive through this context: no Date.now, no setTimeout, no
 * fetch. Anything external goes through `io`, anything delayed through `setTimer`.
 */
export interface NodeContext<P = Record<string, Json>> {
  readonly id: string;
  readonly params: P;
  now(): number;
  /** Seq of the event currently being handled (default cause for anything emitted). */
  readonly cause: number | undefined;
  emit(port: string, msg: Msg, cause?: number[]): number;
  setState(key: string, value: Json): void;
  log(level: LogLevel, message: string, data?: Json): void;
  setTimer(delayMs: number, fn: () => void, label?: string): TimerHandle;
  io: { open(channel: string, request: Json, handlers: IOHandlers): IOHandle };
  media: { append(track: string, samples: Float32Array, sampleRate: number): MediaRef };
}

export interface NodeImpl {
  start?(): void;
  onInput?(port: string, msg: Msg): void;
  stop?(): void;
}

export interface NodeDef<P = any> {
  type: string;
  title: string;
  category: 'source' | 'cleanup' | 'vad' | 'asr' | 'control' | 'llm' | 'tts' | 'sink' | 'mock';
  description?: string;
  inputs: PortDef[];
  outputs: PortDef[];
  params: ParamDef[];
  channels?: ChannelDef[];
  /** Proxy upstreams (API keys) this node needs with these params, e.g. ['anthropic'] or ['openai/openrouter']. */
  requires?(params: Record<string, Json>): string[];
  create(ctx: NodeContext<P>): NodeImpl;
}

export type NodeRegistry = Record<string, NodeDef>;

export function resolveParams(def: NodeDef, params: Record<string, Json>): Record<string, Json> {
  const out: Record<string, Json> = {};
  for (const p of def.params) out[p.key] = params[p.key] !== undefined ? params[p.key] : p.default;
  return out;
}
