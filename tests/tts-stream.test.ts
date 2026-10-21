import { describe, expect, it } from 'vitest';
import { cutAt, kyutaiTtsNode } from '../src/nodes/tts/stream';
import { resolveParams, type IOHandlers, type NodeContext } from '../src/core/node';
import type { Json, Msg } from '../src/core/types';

function harness(params: Record<string, unknown> = {}, def = kyutaiTtsNode) {
  let now = 0;
  const out: Msg[] = [];
  const ios: { request: Json; h: IOHandlers; closed: boolean; sent: Json[] }[] = [];
  let media = 0;
  const ctx: NodeContext = {
    id: 'tts',
    params: resolveParams(def, { alignWords: false, ...params } as never),
    now: () => now,
    cause: undefined,
    emit: (_port, msg) => (out.push(msg), out.length),
    setState() {},
    log() {},
    setTimer: () => ({ clear() {} }),
    io: {
      open(_channel, request, h) {
        const x = { request, h, closed: false, sent: [] as Json[] };
        ios.push(x);
        return { xid: String(ios.length), send: (d) => void x.sent.push(d), close: () => void (x.closed = true), get closed() { return x.closed; } };
      },
    },
    media: {
      append: (track, samples) => ({ track, start: media, end: (media += samples.length) }),
    },
  };
  const node = def.create(ctx as never);
  const send = (t: number, port: string, msg: Msg) => ((now = t), node.onInput?.(port, msg));
  const llm = (t: number, type: string, text = '') => send(t, 'text', { type: `llm.${type}`, rid: 'r0', text });
  const pcm = (n: number) => new Uint8Array(n * 2);
  const types = () => out.map((m) => m.type).filter((t) => t !== 'tts.sent');
  return { send, llm, ios, out, pcm, types };
}

describe('segment cuts', () => {
  it('cuts at sentence ends, and at the first clause of a reply once it is long enough', () => {
    expect(cutAt('Hey, nice', 20)).toBe(-1);
    expect(cutAt('Sure, I can help with that, ', 20)).toBe('Sure, I can help with that, '.length);
    expect(cutAt('Hi there. How', 0)).toBe('Hi there. '.length);
    expect(cutAt('It costs 3.', 0)).toBe(-1);
  });
});

describe('streaming TTS (Kyutai on the GPU box)', () => {
  const kyutai = (params: Record<string, unknown> = {}) => harness(params, kyutaiTtsNode);

  it('sends whole words as they come, flushes at sentence ends, and closes the text when the LLM ends', () => {
    const h = kyutai();
    h.llm(0, 'start');
    expect(h.ios[0].request).toEqual({ path: '/v1/tts', query: {} });
    h.llm(100, 'delta', 'Sure, I can he');
    expect(h.ios[0].sent).toEqual([{ type: 'Speak', text: 'Sure, I can ' }]);
    h.ios[0].h.onRecv!(null, h.pcm(100));
    h.llm(150, 'delta', 'lp. Then');
    expect(h.ios[0].sent.slice(1)).toEqual([{ type: 'Speak', text: 'help. ' }, { type: 'Flush' }]);
    h.llm(300, 'delta', ' bye.');
    h.llm(310, 'end');
    expect(h.ios[0].sent.slice(3)).toEqual([{ type: 'Speak', text: 'Then ' }, { type: 'Speak', text: 'bye.' }, { type: 'Flush' }, { type: 'Close' }]);
    h.ios[0].h.onRecv!(null, h.pcm(50));
    h.ios[0].h.onRecv!({ type: 'Flushed' }, undefined);
    h.ios[0].h.onRecv!(null, h.pcm(30));
    h.ios[0].h.onRecv!({ type: 'Flushed' }, undefined);
    expect(h.ios[0].closed).toBe(false);
    h.ios[0].h.onRecv!({ type: 'Done' }, undefined);
    h.ios[0].h.onClose!(null);
    // The first audio plays at once, before its sentence is all written; the rest of its text follows.
    expect(h.types()).toEqual(['tts.segment_start', 'audio', 'tts.segment_text', 'audio', 'tts.segment_end', 'tts.segment_start', 'audio', 'tts.segment_end', 'tts.response_end']);
    expect(h.out.filter((m) => m.type === 'tts.segment_start').map((m) => m.text)).toEqual(['Sure, I can', 'Then bye.']);
    expect(h.out.find((m) => m.type === 'tts.segment_text')).toMatchObject({ seg: 0, text: 'Sure, I can help.' });
    // Close went out with the end of the text; it is not sent again.
    expect(h.ios[0].sent.filter((m) => (m as { type: string }).type === 'Close')).toHaveLength(1);
  });

  it("sends the reply's voice direction as Instruct before its first word", () => {
    const h = kyutai();
    h.llm(0, 'start');
    h.llm(50, 'style', 'warm, unhurried');
    h.llm(100, 'delta', 'Sure thing. ');
    expect(h.ios[0].sent.slice(0, 2)).toEqual([{ type: 'Instruct', text: 'warm, unhurried' }, { type: 'Speak', text: 'Sure thing. ' }]);
  });

  it('plays the audio after the last Flushed (the model speaks on), and ends the reply on Done', () => {
    const h = kyutai();
    h.llm(0, 'start');
    h.llm(100, 'delta', 'Reggae it is.');
    h.llm(110, 'end');
    h.ios[0].h.onRecv!(null, h.pcm(100));
    // Flushed comes a little early, and the model's last frames follow it.
    h.ios[0].h.onRecv!({ type: 'Flushed' }, undefined);
    expect(h.types()).toEqual(['tts.segment_start', 'audio', 'tts.segment_end']);
    expect(h.ios[0].closed).toBe(false);
    h.ios[0].h.onRecv!(null, h.pcm(40));
    h.ios[0].h.onRecv!(null, h.pcm(40));
    h.ios[0].h.onRecv!({ type: 'Done', stats: {} }, undefined);
    h.ios[0].h.onClose!(null);
    expect(h.types()).toEqual(['tts.segment_start', 'audio', 'tts.segment_end', 'audio', 'audio', 'tts.response_end']);
    expect(h.out.filter((m) => m.type === 'audio').map((m) => [m.seg, (m.samples as Float32Array).length])).toEqual([[0, 100], [0, 40], [0, 40]]);
  });

  it('ends a reply that said nothing at once', () => {
    const h = kyutai();
    h.llm(0, 'start');
    h.llm(10, 'end');
    expect(h.types()).toEqual(['tts.response_end']);
    expect(h.ios[0].closed).toBe(true);
  });

  it('lets a finished reply play out when the next one starts, and queues the new audio behind it', () => {
    const h = kyutai();
    h.send(0, 'text', { type: 'llm.start', rid: 'a' });
    h.send(10, 'text', { type: 'llm.delta', rid: 'a', text: 'Sure, reggae.' });
    h.send(20, 'text', { type: 'llm.end', rid: 'a' });
    h.ios[0].h.onRecv!(null, h.pcm(10));
    // The next reply starts while the first still renders: the first is not cut.
    h.send(30, 'text', { type: 'llm.start', rid: 'b' });
    expect(h.ios[0].closed).toBe(false);
    expect(h.types()).not.toContain('tts.cancelled');
    h.send(40, 'text', { type: 'llm.delta', rid: 'b', text: 'Bob Marley. ' });
    expect(h.ios[1].sent[0]).toEqual({ type: 'Speak', text: 'Bob Marley. ' });
    // The new reply's audio comes first, but plays after the old one's.
    h.ios[1].h.onRecv!(null, h.pcm(7));
    h.ios[0].h.onRecv!(null, h.pcm(20));
    h.ios[0].h.onRecv!({ type: 'Flushed' }, undefined);
    h.ios[0].h.onRecv!(null, h.pcm(30));
    expect(h.out.filter((m) => m.type === 'audio').map((m) => [m.rid, (m.samples as Float32Array).length])).toEqual([['a', 10], ['a', 20], ['a', 30]]);
    h.ios[0].h.onClose!(null);
    h.ios[1].h.onRecv!(null, h.pcm(8));
    const seq = h.out.filter((m) => m.type !== 'tts.sent').map((m) => `${m.rid}:${m.type}${m.type === 'audio' ? (m.samples as Float32Array).length : ''}`);
    expect(seq).toEqual(['a:tts.segment_start', 'a:audio10', 'a:audio20', 'a:tts.segment_end', 'a:audio30', 'a:tts.response_end', 'b:tts.segment_start', 'b:audio7', 'b:audio8']);
  });

  it('cuts a reply still being written when the next one starts', () => {
    const h = kyutai();
    h.send(0, 'text', { type: 'llm.start', rid: 'a' });
    h.send(10, 'text', { type: 'llm.delta', rid: 'a', text: 'Sure, reggae ' });
    h.send(30, 'text', { type: 'llm.start', rid: 'b' });
    expect(h.ios[0].closed).toBe(true);
    expect(h.out.at(-1)).toEqual({ type: 'tts.cancelled', rid: 'a' });
  });

  it('a cancel cuts both the reply playing out and the new one, at once', () => {
    const h = kyutai();
    h.send(0, 'text', { type: 'llm.start', rid: 'a' });
    h.send(10, 'text', { type: 'llm.delta', rid: 'a', text: 'Sure, reggae.' });
    h.send(20, 'text', { type: 'llm.end', rid: 'a' });
    h.ios[0].h.onRecv!(null, h.pcm(10));
    h.send(30, 'text', { type: 'llm.start', rid: 'b' });
    h.send(40, 'text', { type: 'llm.delta', rid: 'b', text: 'Bob Marley. ' });
    h.ios[1].h.onRecv!(null, h.pcm(7));
    h.send(50, 'control', { type: 'cancel' });
    expect(h.ios[0].closed).toBe(true);
    expect(h.ios[1].closed).toBe(true);
    h.ios[0].h.onRecv!(null, h.pcm(10));
    h.ios[1].h.onRecv!(null, h.pcm(10));
    h.ios[0].h.onClose!(null);
    // b's audio never played: it was waiting behind a's.
    const seq = h.out.filter((m) => m.type !== 'tts.sent').map((m) => `${m.rid}:${m.type}`);
    expect(seq).toEqual(['a:tts.segment_start', 'a:audio', 'a:tts.cancelled', 'b:tts.cancelled']);
  });

  it('says an opener not rendered ahead on a socket of its own, closed at once, and the reply plays after it', () => {
    const h = kyutai();
    h.llm(0, 'start');
    h.send(100, 'text', { type: 'opener.say', rid: 'r0', text: 'Hmm, let me see ', line: 'Hmm, let me see' });
    // Closed, so the model says it without waiting for words after it.
    expect(h.ios).toHaveLength(2);
    expect(h.ios[0].sent).toEqual([]);
    expect(h.ios[1].sent).toEqual([{ type: 'Speak', text: 'Hmm, let me see ' }, { type: 'Flush' }, { type: 'Close' }]);
    h.llm(900, 'delta', 'It is 391.');
    h.llm(950, 'end');
    expect(h.ios[0].sent).toEqual([{ type: 'Speak', text: 'It is ' }, { type: 'Speak', text: '391.' }, { type: 'Flush' }, { type: 'Close' }]);
    // The reply's audio comes first, but plays after the opener's.
    h.ios[0].h.onRecv!(null, h.pcm(7));
    h.ios[1].h.onRecv!(null, h.pcm(10));
    h.ios[1].h.onRecv!({ type: 'Flushed' }, undefined);
    h.ios[1].h.onRecv!(null, h.pcm(3));
    h.ios[1].h.onClose!(null);
    h.ios[0].h.onRecv!({ type: 'Flushed' }, undefined);
    h.ios[0].h.onClose!(null);
    const seq = h.out.filter((m) => m.type !== 'tts.sent').map((m) => `${m.rid}:${m.type}${m.type === 'audio' ? (m.samples as Float32Array).length : ''}${'seg' in m ? `/${m.seg}` : ''}`);
    expect(seq).toEqual([
      'r0:tts.segment_start/0',
      'r0:audio10/0',
      'r0:tts.segment_end/0',
      'r0:audio3/0',
      'r0:tts.segment_start/1',
      'r0:audio7/1',
      'r0:tts.segment_end/1',
      'r0:tts.response_end',
    ]);
  });

  it('cuts an opener on its own socket with its reply', () => {
    const h = kyutai();
    h.llm(0, 'start');
    h.send(100, 'text', { type: 'opener.say', rid: 'r0', text: 'Hmm, let me see ', line: 'Hmm, let me see' });
    h.ios[1].h.onRecv!(null, h.pcm(10));
    h.send(200, 'control', { type: 'cancel' });
    expect(h.ios[0].closed).toBe(true);
    expect(h.ios[1].closed).toBe(true);
    expect(h.types()).toEqual(['tts.segment_start', 'audio', 'tts.cancelled']);
  });

  it('places the server’s word times on the media track', () => {
    const h = kyutai({ alignWords: true });
    h.llm(0, 'start');
    h.llm(10, 'delta', 'Hello there. ');
    h.send(20, 'text', { type: 'bc.say', id: 'b', line: 'x', take: 'x' }); // not rendered: logs only
    h.ios[0].h.onRecv!(null, h.pcm(24000));
    h.ios[0].h.onRecv!({ type: 'Words', words: [{ word: 'Hello', start: 0.1, end: 0.5 }] }, undefined);
    expect(h.out.find((m) => m.type === 'tts.words')).toEqual({ type: 'tts.words', rid: 'r0', words: [{ word: 'Hello', start: 2400, end: 12000 }] });
  });

  it("synthesizes the opener's lines ahead and plays one without a request", () => {
    const h = kyutai();
    h.send(0, 'text', { type: 'opener.lines', lines: ['Hmm, let me think.'] });
    const pre = h.ios[0];
    expect(pre.request).toEqual({ text: 'Hmm, let me think.', voice: '', sampleRate: 24000 });
    pre.h.onRecv!({ type: 'Canned', cached: false }, undefined);
    pre.h.onRecv!(null, h.pcm(30));
    pre.h.onRecv!(null, new Uint8Array(3)); // odd bytes carry over
    pre.h.onRecv!(null, new Uint8Array(1));
    pre.h.onClose!(null);
    expect(h.out).toEqual([]);
    h.llm(1000, 'start');
    h.send(1000, 'text', { type: 'opener.say', rid: 'r0', text: 'Hmm, let me think. ', line: 'Hmm, let me think.' });
    expect(h.ios[1].sent).toEqual([]);
    expect(h.types()).toEqual(['tts.segment_start', 'audio', 'audio', 'audio', 'tts.segment_end']);
    expect(h.out.filter((m) => m.type === 'audio').map((m) => (m.samples as Float32Array).length)).toEqual([30, 1, 1]);
    h.llm(1500, 'delta', 'Because of scattering. ');
    expect(h.ios[1].sent).toEqual([{ type: 'Speak', text: 'Because of scattering. ' }, { type: 'Flush' }]);
    h.ios[1].h.onRecv!(null, h.pcm(5));
    expect(h.out.at(-2)).toMatchObject({ type: 'tts.segment_start', seg: 1 });
  });

  it('renders a canned line again once when it fails', () => {
    const h = kyutai();
    h.send(0, 'text', { type: 'bc.lines', lines: ['Mm-hmm.'] });
    h.ios[0].h.onClose!({ error: 'tts closed before the line was rendered' });
    expect(h.ios).toHaveLength(2);
    expect(h.ios[1].request).toMatchObject({ text: 'Mm-hmm.' });
    h.ios[1].h.onClose!({ error: 'websocket closed (1006)' });
    expect(h.ios).toHaveLength(2);
  });

  it('stops on cancel and drops audio still in flight', () => {
    const h = kyutai();
    h.llm(0, 'start');
    h.llm(100, 'delta', 'Hello there. ');
    h.ios[0].h.onRecv!(null, h.pcm(10));
    h.send(200, 'control', { type: 'cancel' });
    h.ios[0].h.onRecv!(null, h.pcm(10));
    expect(h.ios[0].closed).toBe(true);
    expect(h.types()).toEqual(['tts.segment_start', 'audio', 'tts.cancelled']);
  });

  it("renders canned lines one at a time, and doesn't hold them while a reply speaks (the box does)", () => {
    const h = kyutai();
    h.send(0, 'text', { type: 'opener.lines', lines: ['Hmm.', 'Right.', 'Okay.'] });
    expect(h.ios.map((x) => x.request)).toEqual([{ text: 'Hmm.', voice: '', sampleRate: 24000 }]);
    h.ios[0].h.onRecv!(null, h.pcm(10));
    h.ios[0].h.onClose!(null);
    expect(h.ios[1].request).toMatchObject({ text: 'Right.' });
    // A reply starts while a line renders: the next line goes out anyway, as a background job.
    h.llm(10, 'start');
    h.ios[1].h.onRecv!(null, h.pcm(10));
    h.ios[1].h.onClose!(null);
    expect(h.ios[3].request).toMatchObject({ text: 'Okay.' });
    expect(h.ios[2].request).toEqual({ path: '/v1/tts', query: {} });
  });
});
