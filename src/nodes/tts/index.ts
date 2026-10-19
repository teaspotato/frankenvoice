import type { NodeContext, NodeDef } from '../../core/node';
import type { JsonObject, Msg } from '../../core/types';
import { SentenceSplitter } from '../segment';
import { paramsFor, providerParams } from '../providers';
import { cartesiaAdapter, deepgramTtsAdapter, openaiTtsAdapter } from './adapters';
import type { TtsAdapter } from './types';

export const ttsAdapters: Record<string, TtsAdapter> = {
  [cartesiaAdapter.provider]: cartesiaAdapter,
  [deepgramTtsAdapter.provider]: deepgramTtsAdapter,
  [openaiTtsAdapter.provider]: openaiTtsAdapter,
};

interface Segment {
  rid: string;
  seg: number;
  text: string;
  io?: ReturnType<NodeContext['io']['open']>;
  chunks: Float32Array[];
  remainder: Uint8Array;
  done: boolean;
  started: boolean;
}

/**
 * Streaming TTS, one request per sentence. Requests run concurrently; audio is
 * emitted strictly in sentence order. The provider is a parameter; raw
 * response bytes are recorded, so replay reproduces the exact chunking.
 */
export const ttsNode: NodeDef = {
  type: 'tts',
  title: 'TTS',
  category: 'tts',
  description: 'Sentence-level streaming text-to-speech. Providers: Cartesia, Deepgram Aura, OpenAI-compatible /audio/speech.',
  inputs: [
    { name: 'text', type: 'text' },
    { name: 'control', type: 'control' },
  ],
  outputs: [{ name: 'audio', type: 'audio' }],
  params: [
    { key: 'provider', label: 'Provider', type: 'select', options: Object.keys(ttsAdapters), default: 'cartesia' },
    { key: 'minSegmentChars', label: 'Min segment chars', type: 'number', default: 1 },
    ...providerParams(Object.values(ttsAdapters)),
  ],
  channels: Object.values(ttsAdapters).map((a) => ({ name: a.channel, description: `${a.provider}: streamed raw PCM response` })),
  requires: (p) => (p.provider === 'openai-compatible' ? [`openai/${String(p['openai-compatible.endpoint'])}`] : [String(p.provider)]),
  create(ctx) {
    const p = ctx.params as JsonObject & { minSegmentChars: number };
    const adapter = ttsAdapters[p.provider as string];
    const ap = adapter ? paramsFor(adapter.provider, p) : {};
    const fmt = adapter ? adapter.format(ap) : { encoding: 'f32le' as const, sampleRate: 24000 };
    const bytesPerSample = fmt.encoding === 'f32le' ? 4 : 2;
    const decode = (bytes: Uint8Array): Float32Array => {
      if (fmt.encoding === 'f32le') return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const out = new Float32Array(bytes.byteLength / 2);
      for (let i = 0; i < out.length; i++) out[i] = view.getInt16(i * 2, true) / 0x8000;
      return out;
    };
    if (!adapter) ctx.log('error', `unknown TTS provider "${String(p.provider)}"`);
    ctx.setState('provider', p.provider);
    let splitter = new SentenceSplitter(p.minSegmentChars);
    let queue: Segment[] = []; // segments not yet fully emitted, in order
    let rid = '';
    let segNo = 0;
    let llmDone = false;

    const emitAudio = (s: Segment, samples: Float32Array) => {
      if (!s.started) {
        s.started = true;
        ctx.emit('audio', { type: 'tts.segment_start', rid: s.rid, seg: s.seg, text: s.text });
      }
      if (!samples.length) return;
      const ref = ctx.media.append('audio', samples, fmt.sampleRate);
      ctx.emit('audio', { type: 'audio', ...ref, sampleRate: fmt.sampleRate, samples, rid: s.rid, seg: s.seg });
    };

    /** Emit everything that is ready at the head of the queue. */
    const pump = () => {
      while (queue.length) {
        const head = queue[0];
        for (const c of head.chunks) emitAudio(head, c);
        head.chunks = [];
        if (!head.done) return;
        if (!head.started) emitAudio(head, new Float32Array());
        ctx.emit('audio', { type: 'tts.segment_end', rid: head.rid, seg: head.seg });
        queue.shift();
      }
      if (llmDone && rid) {
        ctx.emit('audio', { type: 'tts.response_end', rid });
        ctx.setState('status', 'idle');
        rid = '';
      }
    };

    const speak = (text: string) => {
      if (!adapter) return;
      const s: Segment = { rid, seg: segNo++, text, chunks: [], remainder: new Uint8Array(), done: false, started: false };
      queue.push(s);
      ctx.setState('status', 'synthesizing');
      s.io = ctx.io.open(
        adapter.channel,
        adapter.buildRequest(text, ap),
        {
          onRecv(_data, bin) {
            if (!bin) return;
            // Chunk boundaries need not align with sample boundaries.
            const all = new Uint8Array(s.remainder.length + bin.length);
            all.set(s.remainder);
            all.set(bin, s.remainder.length);
            const usable = all.length - (all.length % bytesPerSample);
            s.remainder = all.slice(usable);
            s.chunks.push(decode(all.slice(0, usable)));
            if (queue[0] === s) pump();
          },
          onClose(data) {
            s.done = true;
            if (data && (data as { error?: unknown }).error) ctx.log('error', 'TTS request failed', data);
            if (queue[0] === s) pump();
          },
        },
      );
    };

    const cancel = () => {
      for (const s of queue) s.io?.close({ status: 'cancelled' });
      if (rid) ctx.emit('audio', { type: 'tts.cancelled', rid });
      queue = [];
      rid = '';
      splitter = new SentenceSplitter(p.minSegmentChars);
      ctx.setState('status', 'idle');
    };

    return {
      onInput(port, msg: Msg) {
        if (port === 'control') {
          if (msg.type === 'cancel') cancel();
          return;
        }
        switch (msg.type) {
          case 'llm.start':
            if (rid) cancel();
            rid = msg.rid as string;
            segNo = 0;
            llmDone = false;
            splitter = new SentenceSplitter(p.minSegmentChars);
            break;
          case 'opener.say':
          case 'llm.delta':
            if (msg.rid !== rid) return;
            for (const seg of splitter.push(msg.text as string)) speak(seg);
            if (msg.type === 'opener.say') {
              // An opener is said as soon as it arrives, whatever its punctuation.
              const rest = splitter.flush();
              if (rest) speak(rest);
            }
            break;
          case 'llm.end': {
            if (msg.rid !== rid) return;
            const rest = splitter.flush();
            if (rest) speak(rest);
            llmDone = true;
            pump();
            break;
          }
        }
      },
      stop() {
        cancel();
      },
    };
  },
};
