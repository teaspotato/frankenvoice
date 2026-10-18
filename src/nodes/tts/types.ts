import type { ParamDef } from '../../core/node';
import type { Json, JsonObject } from '../../core/types';

/** Pure provider adapter: builds the recorded request; the driver for `channel` streams raw PCM bytes back. */
export interface TtsAdapter {
  provider: string;
  channel: string;
  params: ParamDef[];
  buildRequest(text: string, params: JsonObject): Json;
  /** Format of the response bytes. */
  format(params: JsonObject): { encoding: 'f32le' | 's16le'; sampleRate: number };
}
