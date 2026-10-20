import type { NodeDef } from '../core/node';

/** Streams "You said: …" back word by word on timers. No API key needed. */
export const echoLlmNode: NodeDef<{ prefix: string; wordMs: number }> = {
  type: 'llm.echo',
  title: 'Echo LLM',
  category: 'mock',
  description: 'Repeats the user turn back, one word per tick. For testing pipelines without an API key.',
  inputs: [
    { name: 'turn', type: 'turn' },
    { name: 'control', type: 'control' },
  ],
  outputs: [{ name: 'text', type: 'text' }],
  params: [
    { key: 'prefix', label: 'Prefix', type: 'string', default: 'You said:' },
    { key: 'wordMs', label: 'Ms per word', type: 'number', default: 60 },
  ],
  create(ctx) {
    let n = 0;
    let cancel: (() => void) | undefined;
    return {
      onInput(port, msg) {
        if (port === 'control') {
          if (msg.type === 'cancel') cancel?.();
          return;
        }
        if (msg.type !== 'turn.user') return;
        cancel?.();
        const rid = `${ctx.id}-r${n++}`;
        const words = `${ctx.params.prefix} ${msg.text as string}`.trim().split(/\s+/);
        ctx.emit('text', { type: 'llm.start', rid, turnId: (msg.turnId as string) ?? null });
        let i = 0;
        let text = '';
        let timer = ctx.setTimer(ctx.params.wordMs, function tick() {
          const w = (i ? ' ' : '') + words[i++];
          text += w;
          ctx.emit('text', { type: 'llm.delta', rid, text: w });
          if (i < words.length) timer = ctx.setTimer(ctx.params.wordMs, tick);
          else {
            cancel = undefined;
            ctx.emit('text', { type: 'llm.end', rid, status: 'done', stopReason: 'end_turn', text });
          }
        });
        cancel = () => {
          timer.clear();
          cancel = undefined;
          ctx.emit('text', { type: 'llm.end', rid, status: 'cancelled', stopReason: null, text });
        };
      },
    };
  },
};

/** Renders each sentence as a short tone sequence. Deterministic, no IO. */
export const toneTtsNode: NodeDef<{ sampleRate: number; msPerChar: number }> = {
  type: 'tts.tone',
  title: 'Tone TTS',
  category: 'mock',
  description: 'Turns text into beeps (one per word). For testing pipelines without an API key.',
  inputs: [
    { name: 'text', type: 'text' },
    { name: 'control', type: 'control' },
  ],
  outputs: [{ name: 'audio', type: 'audio' }],
  params: [
    { key: 'sampleRate', label: 'Sample rate', type: 'number', default: 24000 },
    { key: 'msPerChar', label: 'Ms per character', type: 'number', default: 40 },
  ],
  create(ctx) {
    const sr = ctx.params.sampleRate;
    let seg = 0;
    return {
      onInput(port, msg) {
        if ((port === 'control' && msg.type === 'cancel') || msg.type === 'llm.start') {
          seg = 0;
          return;
        }
        if (msg.type === 'llm.end') {
          ctx.emit('audio', { type: 'tts.response_end', rid: msg.rid as string });
          return;
        }
        if (msg.type !== 'llm.delta') return;
        const word = (msg.text as string).trim();
        if (!word) return;
        const n = Math.round((word.length * ctx.params.msPerChar * sr) / 1000);
        const samples = new Float32Array(n);
        const f = 300 + (word.charCodeAt(0) % 12) * 40;
        for (let i = 0; i < n; i++) samples[i] = 0.2 * Math.sin((2 * Math.PI * f * i) / sr) * Math.min(1, i / 200, (n - i) / 200);
        const rid = msg.rid as string;
        ctx.emit('audio', { type: 'tts.segment_start', rid, seg, text: word });
        const ref = ctx.media.append('audio', samples, sr);
        ctx.emit('audio', { type: 'audio', ...ref, sampleRate: sr, samples, rid, seg });
        ctx.emit('audio', { type: 'tts.segment_end', rid, seg: seg++ });
      },
    };
  },
};
