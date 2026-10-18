import type { NodeContext, NodeDef } from '../core/node';
import type { Msg } from '../core/types';
import { SentenceSplitter } from './segment';

interface CartesiaParams {
  modelId: string;
  voiceId: string;
  language: string;
  sampleRate: number;
  minSegmentChars: number;
}

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
 * Streaming TTS via Cartesia's /tts/bytes endpoint, one request per sentence.
 * Requests run concurrently; audio is emitted strictly in sentence order.
 * Raw response bytes are recorded, so replay reproduces the exact chunking.
 */
export const cartesiaTtsNode: NodeDef<CartesiaParams> = {
  type: 'tts.cartesia',
  title: 'Cartesia TTS',
  category: 'tts',
  description: 'Sentence-level streaming text-to-speech (raw float32 PCM).',
  inputs: [
    { name: 'text', type: 'text' },
    { name: 'control', type: 'control' },
  ],
  outputs: [{ name: 'audio', type: 'audio' }],
  params: [
    { key: 'modelId', label: 'Model', type: 'string', default: 'sonic-2' },
    { key: 'voiceId', label: 'Voice ID', type: 'string', default: 'a0e99841-438c-4a64-b679-ae501e7d6091' },
    { key: 'language', label: 'Language', type: 'string', default: 'en' },
    { key: 'sampleRate', label: 'Sample rate', type: 'number', default: 24000 },
    { key: 'minSegmentChars', label: 'Min segment chars', type: 'number', default: 1 },
  ],
  channels: [{ name: 'cartesia', description: 'POST /tts/bytes, streamed response body' }],
  create(ctx) {
    const p = ctx.params;
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
      const ref = ctx.media.append('audio', samples, p.sampleRate);
      ctx.emit('audio', { type: 'audio', ...ref, sampleRate: p.sampleRate, samples, rid: s.rid, seg: s.seg });
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
      const s: Segment = { rid, seg: segNo++, text, chunks: [], remainder: new Uint8Array(), done: false, started: false };
      queue.push(s);
      ctx.setState('status', 'synthesizing');
      s.io = ctx.io.open(
        'cartesia',
        {
          path: '/tts/bytes',
          body: {
            model_id: p.modelId,
            transcript: text,
            voice: { mode: 'id', id: p.voiceId },
            output_format: { container: 'raw', encoding: 'pcm_f32le', sample_rate: p.sampleRate },
            language: p.language,
          },
        },
        {
          onRecv(_data, bin) {
            if (!bin) return;
            // Chunk boundaries need not align with 4-byte samples.
            const all = new Uint8Array(s.remainder.length + bin.length);
            all.set(s.remainder);
            all.set(bin, s.remainder.length);
            const usable = all.length - (all.length % 4);
            s.remainder = all.slice(usable);
            s.chunks.push(new Float32Array(all.buffer.slice(0, usable)));
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
          case 'llm.delta':
            if (msg.rid !== rid) return;
            for (const seg of splitter.push(msg.text as string)) speak(seg);
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
