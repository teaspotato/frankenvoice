import type { ParamDef } from '../../core/node';
import type { Json, JsonObject } from '../../core/types';

/** Times are seconds from the first audio sample sent on the connection. */
export type AsrEvent =
  | { type: 'open' }
  | { type: 'meta'; key: string; value: Json }
  | { type: 'speech_started'; at: number }
  | { type: 'utterance_end'; at: number }
  | {
      type: 'partial' | 'final';
      text: string;
      confidence: number | null;
      speechFinal: boolean;
      /** This final was flushed by a finalize request, not by the provider's own endpointing. */
      fromFinalize?: boolean;
      start: number;
      end: number;
      words: { word: string; start: number; end: number }[];
    }
  | { type: 'unknown' };

/** Pure provider adapter; the network side is the driver for `channel`. */
export interface AsrAdapter {
  provider: string;
  channel: string;
  /** Provider-specific params, shown in the inspector only for this provider. */
  params: ParamDef[];
  buildRequest(params: JsonObject, sampleRate: number): Json;
  encode(samples: Float32Array): Uint8Array;
  /** Sent before closing, to flush final results. */
  closeMessage?(): Json;
  /** Asks for the final transcript of the audio sent so far right away, keeping the stream open. */
  finalizeMessage?(): Json;
  parse(data: Json): AsrEvent[];
}
