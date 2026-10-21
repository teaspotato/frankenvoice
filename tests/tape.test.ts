import { describe, expect, it } from 'vitest';
import { EXAMPLE_TAPE, PROMPTS, actionOdds, cutTape, grammar, headroom, line, newCell, parseAction, promptHead, tapePrompt, timed, trimIdle, type Cell, type TapeState } from '../src/nodes/tape';
import { noPeriods, tapeTurnNode } from '../src/nodes/turn-tape';
import { harness } from './node-harness';
import type { Json, Msg } from '../src/core/types';

/** The pilot's barge-in run (sample-transcripts.md §2), as Gemma wrote it, up to the 12.16 cell. */
function bargeIn(): Cell[] {
  const spec: [string[], boolean, TapeState, string, string[]][] = [
    [[], false, 'silent', 'listen', []],
    [["what's"], true, 'silent', 'listen', []],
    [['the', 'fastest'], true, 'silent', 'listen', []],
    [['way', 'to', 'get'], true, 'silent', 'listen', []],
    [['from', 'Paris'], true, 'silent', 'listen', []],
    [['to'], true, 'silent', 'listen', []],
    [['Amsterdam?'], false, 'silent', 'listen', []],
    [[], false, 'silent', 'listen', []],
    [[], false, 'silent', 'listen', []],
    [[], false, 'silent', 'respond', ['The', 'quickest', 'way']],
    [[], false, 'speaking', 'continue', ['is', 'usually', 'by']],
    [[], false, 'speaking', 'continue', ['taking', 'a', 'train']],
    [[], false, 'speaking', 'finish', ['from', 'Paris.']],
    [[], false, 'silent', 'listen', []],
    [[], true, 'cut', 'listen', []],
    [['hold', 'on,'], true, 'cut', 'listen', []],
    [['what', 'about', 'by'], true, 'cut', 'listen', []],
    [['car?'], false, 'cut', 'listen', []],
    [[], false, 'cut', 'listen', []],
  ];
  return spec.map(([u, v, s, a, w], i) => ({ ...newCell(i + 1, Math.round((i + 1) * 640) / 1000, s, u, v), action: a, say: [...w] }));
}

describe('the tape format', () => {
  it('writes lines as the pilot does', () => {
    const c = { ...newCell(19, 12.16, 'cut', ['car?'], false), action: 'respond', say: ['By', 'car,', "it's", 'a'] };
    expect(line(c)).toBe(' 12.16 | user: "car?" (silent) | agent (cut off): respond "By car, it\'s a"');
    expect(line(newCell(1, 0.64, 'silent'))).toBe('  0.64 | user: (silent) | agent (silent): listen');
  });

  it('has a grammar per state, capped at the words the TTS has room for', () => {
    expect(grammar(['listen', 'resume', 'respond'], 4)).toBe(
      [
        'root ::= " listen" | " resume " say | " respond " say',
        'say ::= "\\"" word (" " word){0,3} "\\""',
        "word ::= [A-Za-z0-9$%&'] [A-Za-z0-9$%&'-]{0,18} [,.?!;:]?",
        'bc ::= "\\"mhm\\"" | "\\"yeah\\"" | "\\"right\\"" | "\\"okay\\"" | "\\"uh-huh\\"" | "\\"I see\\""',
      ].join('\n'),
    );
    expect(grammar(['continue', 'finish'], 1).split('\n').slice(0, 2)).toEqual(['root ::= " continue " say | " finish" (" " end)?', 'say ::= "\\"" word (" " word){0,0} "\\""']);
    expect(grammar(['listen', 'respond', 'backchannel'], 2).split('\n')[0]).toBe('root ::= " listen" | " respond " say | " backchannel " bc');
  });

  it("reads Gemma's odds for each action from the first token's alternatives", () => {
    const top = (pairs: [string, number][]) => ({ completion_probabilities: [{ token: pairs[0][0], logprob: Math.log(pairs[0][1]), top_logprobs: pairs.map(([token, p]) => ({ token, logprob: Math.log(p) })) }] });
    // " backchannel" starts with the token " back"; "\n" belongs to no action.
    expect(actionOdds(top([[' listen', 0.7], [' respond', 0.2], [' back', 0.05], ['\n', 0.05]]), ['listen', 'respond', 'backchannel'])).toEqual({ listen: 0.7, respond: 0.2, backchannel: 0.05 });
    // An action outside the top alternatives gets 0.
    expect(actionOdds(top([[' continue', 0.99]]), ['continue', 'finish'])).toEqual({ continue: 0.99, finish: 0 });
    expect(actionOdds({ content: ' listen' } as never, ['listen'])).toBeNull();
  });

  it('reads what Gemma wrote', () => {
    expect(parseAction(' respond "By car, it\'s a"')).toEqual({ action: 'respond', words: ['By', 'car,', "it's", 'a'] });
    expect(parseAction(' finish')).toEqual({ action: 'finish', words: [] });
    expect(parseAction(' backchannel "I see"')).toEqual({ action: 'backchannel', words: ['I', 'see'] });
  });

  it('gives the TTS two cells of headroom at 0.3 s a word', () => {
    expect(headroom(10, 0, 0.4, 0.64, 6)).toBe(4);
    expect(headroom(10, 10.4 + 1.28, 0.4, 0.64, 6)).toBe(0);
    expect(headroom(10, 10.4 + 0.6, 0.4, 0.64, 6)).toBe(2);
    expect(headroom(10, 0, 0.4, 2, 6)).toBe(6);
  });

  it('takes words never heard off the tape and marks the cut, as the pilot', () => {
    const cells = bargeIn();
    const sent = cells.flatMap((c) => (c.action === 'listen' ? [] : c.say.map((word) => ({ n: c.n, word }))));
    // "The quickest way is usually by taking a" was heard.
    expect(cutTape(cells, sent, 8)).toEqual(['train', 'from', 'Paris.']);
    expect(cells.slice(9, 13).map(line)).toEqual([
      '  6.40 | user: (silent) | agent (silent): respond "The quickest way"',
      '  7.04 | user: (silent) | agent (speaking): continue "is usually by"',
      '  7.68 | user: (silent) | agent (speaking): continue "taking a [cut]"',
      '  8.32 | user: (silent) | agent (speaking): finish',
    ]);
    // Nothing heard: the mark goes on the first cell that sent words.
    const again = bargeIn();
    cutTape(again, sent, 0);
    expect(line(again[9])).toBe('  6.40 | user: (silent) | agent (silent): respond "[cut]"');
  });

  it('cuts each finished run of idle cells to its first lines, and keeps the run before the open cell', () => {
    const idle = (n: number) => newCell(n, n * 0.64, 'silent');
    const heard = (n: number, ...user: string[]) => newCell(n, n * 0.64, 'silent', user, true);
    const answer = Object.assign(newCell(6, 3.84, 'silent'), { action: 'respond', say: ['Sure'] });
    const tape = [heard(0, 'hi'), idle(1), idle(2), idle(3), idle(4), heard(5, 'so'), answer, idle(7), idle(8), idle(9), newCell(10, 6.4, 'silent')];
    expect(trimIdle(tape, 2).map((c) => c.n)).toEqual([0, 1, 2, 5, 6, 7, 8, 9, 10]);
    expect(trimIdle(tape, 0)).toBe(tape);
  });

  it('builds the prompt in Gemma 4 chat format, the live tape as the model’s own turn', () => {
    const cells = bargeIn();
    const sent = cells.flatMap((c) => (c.action === 'listen' ? [] : c.say.map((word) => ({ n: c.n, word }))));
    cutTape(cells, sent, 8);
    const prompt = tapePrompt(promptHead(PROMPTS.base, EXAMPLE_TAPE), cells);
    expect(prompt.startsWith('<|turn>system\nYou are the turn-taking layer')).toBe(true);
    expect(prompt).toContain('Talk like a person on the phone: one or two short sentences, no lists.<turn|>\n<|turn>user\nAn example tape from another call:\n\n  0.64 |');
    expect(prompt).toContain('finish "closed on Sundays."\n\nNow the live call. Continue the tape with the agent\'s action on each new line.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>  0.64 |');
    expect(prompt.endsWith(' 11.52 | user: "car?" (silent) | agent (cut off): listen\n 12.16 | user: (silent) | agent (cut off):')).toBe(true);
    expect(promptHead(PROMPTS.base, EXAMPLE_TAPE, 'user: hi')).toContain('Now the live call. What was said before the tape below:\n\nuser: hi\n\nContinue the tape');
  });

  it('stretches the time on the tape: the cell length in the prompt and the example follow the scale', () => {
    expect(timed(PROMPTS.eager, EXAMPLE_TAPE, 0.64)).toEqual({ system: PROMPTS.eager, example: EXAMPLE_TAPE });
    const { system, example } = timed(PROMPTS.eager, EXAMPLE_TAPE, 0.96);
    expect(system).toContain('Time runs in cells of 0.96 seconds');
    expect(system).toContain('Every cell you wait is 0.96 s of silence');
    expect(system).toContain('spoken aloud 0.4 s after');
    expect(example.split('\n')[0]).toBe('  0.96 | user: (silent) | agent (silent): listen');
    expect(example.split('\n').at(-1)).toBe(' 27.84 | user: (silent) | agent (speaking): finish "closed on Sundays."');
  });
});

describe('the tape stage', () => {
  const run = (params: Record<string, unknown> = {}) => {
    const h = harness(tapeTurnNode, { cellMs: 640, greeting: '', specCells: 0, draftGate: false, ...params });
    const answer = (content: string) => {
      const io = h.ios.at(-1)!;
      io.h.onRecv?.({ content, timings: { prompt_n: 20 } }, undefined);
      io.h.onClose?.(null);
    };
    const prompt = () => ((h.ios.at(-1)!.request as { body: { prompt: string } }).body.prompt);
    const grammarOf = () => ((h.ios.at(-1)!.request as { body: { grammar: string } }).body.grammar);
    const cells = () => h.of('tape.cell') as (Msg & { line: string; forced?: string; action: string })[];
    return { ...h, answer, prompt, grammarOf, cells };
  };

  it('asks once per cell, speaks what Gemma writes, and takes unheard words off the tape on a cut', () => {
    const h = run();
    h.send(100, 'vad', { type: 'vad.speech_start', at: 0 });
    h.send(300, 'transcript', { type: 'asr.partial', text: 'how do', start: 0, end: 1 });
    h.send(500, 'transcript', { type: 'asr.partial', text: 'how do I', start: 0, end: 1 });
    h.advance(640);
    // "how do" came twice: stable, so on the tape; "I" may still change.
    expect(h.prompt().endsWith('  0.64 | user: "how do" (talking) | agent (silent):')).toBe(true);
    expect(h.grammarOf().split('\n')[0]).toBe('root ::= " listen" | " respond " say | " backchannel " bc');
    h.answer(' listen');
    h.send(900, 'vad', { type: 'vad.speech_end', at: 0 });
    h.send(1000, 'transcript', { type: 'asr.final', text: 'how do I clean it?', start: 0, end: 1 });
    expect(noPeriods('if.')).toBe('if');
    expect(noPeriods('U.S.')).toBe('US');
    expect(noPeriods('3.5')).toBe('3.5');
    expect(noPeriods('...')).toBe('');
    h.advance(1280);
    expect(h.prompt().endsWith('  0.64 | user: "how do" (talking) | agent (silent): listen\n  1.28 | user: "I clean it?" (talking) | agent (silent):')).toBe(true);
    h.answer(' listen');
    h.advance(1920);
    h.answer(' respond "You soak it"');
    expect(h.of('llm.start')).toHaveLength(1);
    expect(h.of('llm.delta').map((m) => m.text)).toEqual(['You soak it ']);
    h.advance(2560);
    expect(h.prompt()).toContain('  1.92 | user: (silent) | agent (silent): respond "You soak it"\n  2.56 | user: (silent) | agent (speaking):');
    expect(h.grammarOf().split('\n')[0]).toBe('root ::= " continue " say | " finish" (" " end)?');
    h.answer(' continue "in warm water"');
    const rid = h.of('llm.start')[0].rid as string;
    h.send(2700, 'playback', { type: 'playback.start', rid, at: 2700 });
    // The user talks over it: cut after 200 ms of voice.
    h.send(2900, 'vad', { type: 'vad.speech_start', at: 0 });
    h.advance(3050);
    expect(h.of('cancel')).toHaveLength(0);
    h.advance(3100);
    expect(h.of('cancel')).toHaveLength(1);
    expect(h.of('llm.end').at(-1)).toMatchObject({ rid, status: 'cancelled' });
    h.send(3110, 'playback', { type: 'playback.interrupted', rid, at: 3110, heard: 'You soak it in' });
    expect(h.of('tape.cut').at(-1)).toMatchObject({ said: 4, unsaid: ['warm', 'water'] });
    h.advance(3200);
    expect(h.prompt()).toContain('  2.56 | user: (silent) | agent (speaking): continue "in [cut]"\n  3.20 | user: (talking) | agent (cut off):');
    expect(h.grammarOf().split('\n')[0]).toBe('root ::= " listen" | " resume " say | " respond " say');
    // Quiet for three cells after the cut without an answer: the cut-off state is over.
    h.send(3300, 'vad', { type: 'vad.speech_end', at: 0 });
    for (const t of [3840, 4480, 5120, 5760, 6400]) {
      h.answer(' listen');
      h.advance(t);
    }
    expect(h.prompt().endsWith('  6.40 | user: (silent) | agent (silent):')).toBe(true);
    expect(h.prompt()).toContain('  5.76 | user: (silent) | agent (cut off): listen');
  });

  it("asks for the token odds and reports each action's on the cell", () => {
    const h = run();
    h.send(100, 'transcript', { type: 'asr.final', text: 'is it open?', start: 0, end: 1 });
    h.advance(640);
    expect((h.ios.at(-1)!.request as { body: { n_probs: number } }).body.n_probs).toBe(8);
    const io = h.ios.at(-1)!;
    const tok = (token: string, p: number) => ({ token, logprob: Math.log(p) });
    io.h.onRecv?.({ content: ' listen', completion_probabilities: [{ ...tok(' listen', 0.6), top_logprobs: [tok(' listen', 0.6), tok(' respond', 0.4)] }] }, undefined);
    io.h.onClose?.(null);
    expect(h.cells().at(-1)).toMatchObject({ action: 'listen', xid: '1', odds: { listen: 0.6, respond: 0.4, backchannel: 0 } });
    // Off: no n_probs, no odds.
    const off = run({ nProbs: 0 });
    off.advance(640);
    expect((off.ios.at(-1)!.request as { body: Record<string, unknown> }).body).not.toHaveProperty('n_probs');
    off.answer(' listen');
    expect(off.cells().at(-1)).not.toHaveProperty('odds');
  });

  it('biases respond in silent cells, trims finished idle runs, and answers after a long silence', () => {
    const h = run();
    const body = () => (h.ios.at(-1)!.request as { body: Record<string, unknown> }).body;
    h.send(100, 'transcript', { type: 'asr.final', text: 'hello', start: 0, end: 1 });
    h.advance(640);
    expect(body().logit_bias).toEqual([[' respond', 2.2]]);
    h.answer(' respond "Hi"');
    h.advance(1280);
    // Speaking: no bias.
    expect(body()).not.toHaveProperty('logit_bias');
    h.answer(' finish');
    for (let n = 3; n <= 6; n++) {
      h.advance(n * 640);
      // Silent, but its answer is done and nothing new was said: no bias (it made Gemma answer itself).
      expect(body()).not.toHaveProperty('logit_bias');
      h.answer(' listen');
    }
    h.send(6 * 640 + 100, 'transcript', { type: 'asr.final', text: 'tell me more', start: 0, end: 1 });
    h.advance(7 * 640);
    h.answer(' listen');
    // The idle run 1.92 to 3.84 is finished: its first two lines stay.
    h.advance(8 * 640);
    expect(h.prompt().split('<channel|>')[1].split('\n').map((l) => l.slice(0, 6))).toEqual(['  0.64', '  1.28', '  1.92', '  2.56', '  4.48', '  5.12']);
    h.answer(' listen');
    // Words, then silence: after 3 s (5 cells) only respond is allowed.
    for (let n = 9; n <= 11; n++) {
      h.advance(n * 640);
      expect(h.grammarOf().split('\n')[0]).toBe('root ::= " listen" | " respond " say | " backchannel " bc');
      h.answer(' listen');
    }
    h.advance(12 * 640);
    expect(h.grammarOf().split('\n')[0]).toBe('root ::= " respond " say');
    expect(body()).not.toHaveProperty('logit_bias');
    h.answer(' respond "Sure, what about"');
    expect(h.cells().at(-1)).toMatchObject({ action: 'respond', backstop: 'answer after silence' });
    // Off.
    const off = run({ respondBias: 0, idleKeep: 0, answerAfterS: 0, respondToWords: false });
    off.advance(640);
    expect((off.ios.at(-1)!.request as { body: Record<string, unknown> }).body).not.toHaveProperty('logit_bias');
  });

  it('responds only to new words, and finishes on a sentence end', () => {
    const h = run();
    h.advance(640);
    // Nothing said yet: no respond.
    expect(h.grammarOf().split('\n')[0]).toBe('root ::= " listen" | " backchannel " bc');
    h.answer(' listen');
    h.send(700, 'transcript', { type: 'asr.final', text: 'hello', start: 0, end: 1 });
    h.advance(1280);
    expect(h.grammarOf().split('\n')[0]).toBe('root ::= " listen" | " respond " say | " backchannel " bc');

    h.answer(' respond "Hi there."');
    h.advance(1920);
    h.answer(' finish');
    // Its answer is done and nothing new was said: no respond again (it answered itself).
    h.advance(2560);
    expect(h.grammarOf().split('\n')[0]).toBe('root ::= " listen" | " backchannel " bc');
  });

  it('drafts, then gates: listens before the earliest gate, Gemma chooses in between, responds by the latest', () => {
    const h = run({ draftGate: true, specCells: 4 });
    h.send(50, 'vad', { type: 'vad.speech_start', at: 0 });
    h.send(300, 'transcript', { type: 'asr.final', text: 'how are you?', start: 0, end: 1 });
    h.send(400, 'vad', { type: 'vad.speech_end', at: 0 });
    // The cell with the words is too soon: no call for it; the first call is for the next one.
    expect(h.ios).toHaveLength(1);
    expect(h.prompt().endsWith('  0.64 | user: "how are you?" (talking) | agent (silent): listen\n  1.28 | user: (silent) | agent (silent):')).toBe(true);
    expect(h.grammarOf().split('\n')[0]).toBe('root ::= " listen" | " respond " say | " backchannel " bc');
    h.answer(' listen');
    // 1.28 s of silence: only respond, drafted ahead and held.
    expect(h.grammarOf().split('\n')[0]).toBe('root ::= " respond " say');
    h.answer(' respond "Fine, thanks."');
    expect(h.of('llm.start')[0]).toMatchObject({ held: true });
    h.advance(640);
    expect(h.cells().at(-1)).toMatchObject({ action: 'listen', forced: 'gate: too soon', spec: 'adopted' });
    h.advance(1280);
    expect(h.of('llm.release')).toHaveLength(0);
    h.advance(1920);
    expect(h.cells().at(-1)).toMatchObject({ action: 'respond', backstop: 'gate: answer now', spec: 'adopted' });
    expect(h.of('llm.release')).toHaveLength(1);
  });

  it('skips the call while one is out, and while the TTS queue is full', () => {
    const h = run();
    h.send(100, 'transcript', { type: 'asr.final', text: 'hi there. Wait...', start: 0, end: 1 });
    h.advance(640);
    expect(h.prompt().endsWith('  0.64 | user: "hi there Wait" (silent) | agent (silent):')).toBe(true);
    expect(h.ios).toHaveLength(1);
    h.advance(1280);
    expect(h.ios).toHaveLength(1);
    expect(h.cells().at(-1)).toMatchObject({ action: 'listen', forced: 'a call is still out' });
    h.answer(' respond "Hello, how are you"');
    h.advance(1920);
    h.answer(' continue "doing today, my good old friend?"');
    // Ten words queued at 0.3 s each: no room in the next cell.
    h.advance(2560);
    expect(h.cells().at(-1)).toMatchObject({ action: 'continue', forced: 'TTS queue full' });
    expect(h.ios).toHaveLength(2);
  });

  it('writes ahead once the user stops, renders the reply held, and releases it when real silence confirms the cell', () => {
    const h = run({ specCells: 4 });
    h.send(50, 'vad', { type: 'vad.speech_start', at: 0 });
    h.send(300, 'transcript', { type: 'asr.final', text: 'how are you?', start: 0, end: 1 });
    h.send(400, 'vad', { type: 'vad.speech_end', at: 0 });
    // The cell in progress, as it stands.
    expect(h.prompt().endsWith('  0.64 | user: "how are you?" (talking) | agent (silent):')).toBe(true);
    h.answer(' listen');
    expect(h.prompt().endsWith('  0.64 | user: "how are you?" (talking) | agent (silent): listen\n  1.28 | user: (silent) | agent (silent):')).toBe(true);
    h.answer(' respond "I am fine"');
    expect(h.of('llm.start')[0]).toMatchObject({ held: true });
    expect(h.prompt().endsWith('  1.28 | user: (silent) | agent (silent): respond "I am fine"\n  1.92 | user: (silent) | agent (speaking):')).toBe(true);
    h.answer(' finish "thanks."');
    expect(h.of('llm.end')[0]).toMatchObject({ status: 'done', text: 'I am fine thanks.' });
    expect(h.ios).toHaveLength(3);
    // The real cells match: taken from the speculation, no calls.
    h.advance(640);
    expect(h.cells().at(-1)).toMatchObject({ action: 'listen', spec: 'adopted' });
    expect(h.of('llm.release')).toHaveLength(0);
    h.advance(1280);
    expect(h.cells().at(-1)).toMatchObject({ action: 'respond', spec: 'adopted' });
    expect(h.of('llm.release')).toHaveLength(1);
    h.advance(1920);
    expect(h.cells().at(-1)).toMatchObject({ action: 'finish', spec: 'adopted' });
    expect(h.ios).toHaveLength(3);
    expect(h.cells().map((c) => c.line)).toEqual([
      '  0.64 | user: "how are you?" (talking) | agent (silent): listen',
      '  1.28 | user: (silent) | agent (silent): respond "I am fine"',
      '  1.92 | user: (silent) | agent (speaking): finish "thanks."',
    ]);
  });

  it('drops the speculation and its held reply when the user speaks again', () => {
    const h = run({ specCells: 4 });
    h.send(300, 'transcript', { type: 'asr.final', text: 'so I was thinking', start: 0, end: 1 });
    h.answer(' respond "Yes?"');
    const rid = h.of('llm.start')[0].rid;
    h.send(500, 'vad', { type: 'vad.speech_start', at: 0 });
    expect(h.of('llm.drop')).toEqual([{ type: 'llm.drop', rid }]);
    expect(h.of('tape.spec').at(-1)).toMatchObject({ event: 'drop', why: 'user voice' });
    h.send(600, 'transcript', { type: 'asr.final', text: 'about lunch', start: 0, end: 1 });
    h.advance(640);
    // Back to the live tape.
    expect(h.prompt().endsWith('  0.64 | user: "so I was thinking about lunch" (talking) | agent (silent):')).toBe(true);
    expect(h.of('llm.release')).toHaveLength(0);
  });

  it('drops what it wrote ahead when the opening line starts, so no held reply plays over it', () => {
    const h = run({ specCells: 6, greeting: 'Hi there.' });
    h.advance(640);
    h.answer(' listen');
    // Live cell 1 is silent: writing ahead starts.
    expect(h.of('tape.spec')[0]).toMatchObject({ event: 'start' });
    h.answer(' listen');
    h.send(900, 'transcript', { type: 'asr.open' } as Msg);
    expect(h.of('tape.spec').at(-1)).toMatchObject({ event: 'drop', why: 'greeting' });
    expect(h.ios.at(-1)!.closed).toBe(true);
    h.advance(2000);
    expect(h.of('llm.start')).toHaveLength(0);
    expect(h.cells().map((c) => c.line).slice(0, 3)).toEqual([
      '  0.64 | user: (silent) | agent (silent): listen',
      '  1.28 | user: (silent) | agent (silent): respond "Hi there."',
      '  1.92 | user: (silent) | agent (speaking): finish',
    ]);
  });

  it('does not write ahead between the speech end and its ASR final', () => {
    const h = run({ specCells: 4 });
    h.send(50, 'vad', { type: 'vad.speech_start', at: 0 });
    h.send(500, 'vad', { type: 'vad.speech_end', at: 400 });
    // Its words are still to come: nothing written ahead, so no reply held for silence that isn't.
    expect(h.ios).toHaveLength(0);
    h.advance(640);
    expect(h.ios).toHaveLength(1);
    h.send(700, 'transcript', { type: 'asr.final', text: 'no wait', start: 0, end: 450 });
    h.answer(' listen');
    expect(h.of('tape.spec')[0]).toMatchObject({ event: 'start' });
    expect(h.prompt().endsWith('  1.28 | user: "no wait" (silent) | agent (silent):')).toBe(true);
    expect(h.of('llm.release')).toHaveLength(0);
  });

  it('writes ahead anyway when the ASR final doesn’t come', () => {
    const h = run({ specCells: 4 });
    h.send(50, 'vad', { type: 'vad.speech_start', at: 0 });
    h.send(500, 'vad', { type: 'vad.speech_end', at: 400 });
    h.advance(640);
    h.answer(' listen');
    expect(h.of('tape.spec')).toHaveLength(0);
    for (const t of [1280, 1920]) {
      h.advance(t);
      h.answer(' listen');
    }
    h.advance(1999);
    expect(h.of('tape.spec')).toHaveLength(0);
    h.advance(2000);
    expect(h.of('tape.spec')[0]).toMatchObject({ event: 'start' });
  });

  it('decides a real cell that waited for a speculative call when the speculation is dropped', () => {
    const h = run({ specCells: 4 });
    h.send(300, 'transcript', { type: 'asr.final', text: 'tell me a joke', start: 0, end: 1 });
    // The call for cell 1 is still out at its boundary: the real cell waits for it.
    h.advance(640);
    expect(h.cells()).toHaveLength(0);
    h.send(700, 'vad', { type: 'vad.speech_start', at: 0 });
    expect(h.cells().map((c) => c.line)).toEqual(['  0.64 | user: "tell me a joke" (silent) | agent (silent): listen']);
    expect(h.cells()[0].forced).toBe('speculation dropped (user voice)');
    h.advance(1280);
    expect(h.prompt()).toContain('  0.64 | user: "tell me a joke" (silent) | agent (silent): listen\n  1.28 | user: (talking) | agent (silent):');
  });

  it('keeps a finish written ahead when the user speaks after the reply was released', () => {
    const h = run({ specCells: 4 });
    h.send(300, 'transcript', { type: 'asr.final', text: 'hi', start: 0, end: 1 });
    h.answer(' listen');
    h.answer(' respond "Hello there friend"');
    h.answer(' finish');
    h.advance(640);
    h.advance(1280);
    expect(h.of('llm.release')).toHaveLength(1);
    expect(h.ios).toHaveLength(3);
    // Words in the cell the finish was written for: the speculation breaks, its finish stays.
    h.send(1500, 'transcript', { type: 'asr.final', text: 'wait', start: 1, end: 2 });
    h.advance(1920);
    expect(h.cells().at(-1)).toMatchObject({ line: '  1.92 | user: "wait" (silent) | agent (speaking): finish', forced: 'sent ahead' });
    expect(h.ios).toHaveLength(3);
    h.advance(2560);
    expect(h.prompt().endsWith('  2.56 | user: (silent) | agent (silent):')).toBe(true);
  });

  it('counts the words sent ahead in the TTS headroom when a released speculation breaks', () => {
    const h = run({ specCells: 4 });
    h.send(300, 'transcript', { type: 'asr.final', text: 'hi', start: 0, end: 1 });
    h.answer(' respond "One two three"');
    h.advance(640);
    expect(h.of('llm.release')).toHaveLength(1);
    h.answer(' continue "four five six"');
    h.send(1000, 'transcript', { type: 'asr.final', text: 'um', start: 1, end: 2 });
    h.advance(1280);
    expect(h.cells().at(-1)).toMatchObject({ line: '  1.28 | user: "um" (silent) | agent (speaking): continue "four five six"', forced: 'sent ahead' });
    h.advance(1920);
    // Six words queued by 2.84 s: room for three more, not four.
    expect(h.prompt().endsWith('  1.92 | user: (silent) | agent (speaking):')).toBe(true);
    expect(h.grammarOf().split('\n')[1]).toBe('say ::= "\\"" word (" " word){0,2} "\\""');
  });

  it('says the opening line and writes it on the tape', () => {
    const h = harness(tapeTurnNode, { cellMs: 640, specCells: 0 });
    expect(h.of('bc.lines')[0].lines).toContain("Hi! What's on your mind?");
    h.send(10, 'transcript', { type: 'asr.open' } as Msg);
    expect(h.of('bc.say')[0]).toMatchObject({ why: 'greeting' });
    h.advance(1300);
    const lines = (h.of('tape.cell') as unknown as { line: Json }[]).map((c) => c.line);
    expect(lines).toEqual(['  0.64 | user: (silent) | agent (silent): respond "Hi! What\'s on your mind?"', '  1.28 | user: (silent) | agent (speaking): finish']);
    expect(h.ios).toHaveLength(0);
  });
});
