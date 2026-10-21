import { describe, expect, it } from 'vitest';
import { Trajectory, TRAJECTORY_FORMAT } from '../src/core/trajectory';
import type { EventKind, Json } from '../src/core/types';
import { TapeModel, groupRows, lineTime, openLine, seriesOdds, splitPrompt } from '../src/ui/tape-model';

function session() {
  const traj = new Trajectory({
    format: TRAJECTORY_FORMAT,
    formatVersion: 1,
    id: 't',
    createdAt: '',
    mode: 'live',
    graph: { nodes: [{ id: 'turn', type: 'turn.tape', params: { cellMs: 500, timeScale: 2 } }], edges: [] } as never,
    graphHash: '',
  });
  let seq = 0;
  const add = (t: number, kind: EventKind, data: Json, node = 'turn') => traj.append({ seq: seq++, t, node, kind, data, ...(kind === 'msg' ? { port: 'turn' } : {}) });
  const cell = (t: number, n: number, action: string, say: string[] = [], extra: Record<string, Json> = {}) =>
    add(t, 'msg', { type: 'tape.cell', n, t: n * 0.64, at: n * 640, state: 'silent', action, say, user: [], voice: false, line: `${(n * 0.64).toFixed(2).padStart(6)} | user: (silent) | agent (silent): ${action}${say.length ? ` "${say.join(' ')}"` : ''}`, ...extra });
  return { traj, add, cell, model: new TapeModel(traj) };
}

describe('the Tape inspector model', () => {
  it('follows a call from request to answer, and links it to its cell', () => {
    const { add, cell, model } = session();
    const prompt = '<|turn>system\nrules<turn|>\n<|turn>model\n<|channel>thought\n<channel|>  0.64 | user: "is it open?" (silent) | agent (silent):';
    add(640, 'state', { calling: 'live' });
    add(641, 'io.open', { xid: 'turn/llama-completion#0', channel: 'llama-completion', request: { body: { prompt, grammar: 'root ::= " listen"' } } });
    expect(model.update()).toBe(true);
    expect(model.pending?.ahead).toBe(false);
    expect(openLine(model.pending?.body.prompt)).toBe('  0.64 | user: "is it open?" (silent) | agent (silent):');
    expect(splitPrompt(prompt).tape).toBe('  0.64 | user: "is it open?" (silent) | agent (silent):');
    add(880, 'io.recv', { xid: 'turn/llama-completion#0', payload: { content: ' listen', timings: { prompt_n: 17 } } });
    add(880, 'io.close', { xid: 'turn/llama-completion#0', by: 'remote', payload: null });
    cell(881, 1, 'listen', [], { xid: 'turn/llama-completion#0', odds: { listen: 0.7, respond: 0.3, backchannel: 0 }, ms: 240 });
    model.update();
    expect(model.pending).toBeNull();
    expect(model.rows).toHaveLength(1);
    expect(model.rows[0]).toMatchObject({ n: 1, action: 'listen', ms: 240, odds: { respond: 0.3 } });
    expect(model.callOf(model.rows[0])?.response?.content).toBe(' listen');
    // A recording from before cells named their call: matched by the line it asked for.
    add(1280, 'io.open', { xid: 'turn/llama-completion#1', channel: 'llama-completion', request: { body: { prompt: `${prompt} listen\n  1.28 | user: (silent) | agent (silent):` } } });
    add(1500, 'io.recv', { xid: 'turn/llama-completion#1', payload: { content: ' listen' } });
    add(1500, 'io.close', { xid: 'turn/llama-completion#1', by: 'remote', payload: null });
    cell(1501, 2, 'listen', [], { ms: 220 });
    model.update();
    expect(model.rows[1].xid).toBe('turn/llama-completion#1');
    // Only the tape stage's events.
    add(900, 'msg', { type: 'tape.cell', n: 9 }, 'other');
    expect(model.update()).toBe(false);
  });

  it('shows cells written ahead until a real cell takes them, and notes a dropped speculation', () => {
    const { add, cell, model } = session();
    add(100, 'state', { calling: 'ahead' });
    add(101, 'io.open', { xid: 'x#0', channel: 'llama-completion', request: { body: { prompt: 'h<channel|>  1.28 | user: (silent) | agent (silent):' } } });
    model.update();
    expect(model.pending?.ahead).toBe(true);
    add(200, 'msg', { type: 'tape.spec', event: 'start', from: 2 });
    add(300, 'msg', { type: 'tape.spec', event: 'cell', n: 2, action: 'listen', line: '  1.28 | user: (silent) | agent (silent): listen', xid: 'x#0' });
    add(400, 'msg', { type: 'tape.spec', event: 'cell', n: 3, action: 'respond', line: '  1.92 | user: (silent) | agent (silent): respond "Yes"' });
    model.update();
    expect(model.ahead.map((a) => a.n)).toEqual([2, 3]);
    cell(1280, 2, 'listen', [], { spec: 'adopted', xid: 'x#0' });
    model.update();
    expect(model.ahead.map((a) => a.n)).toEqual([3]);
    add(1500, 'msg', { type: 'tape.spec', event: 'drop', why: 'user voice', ahead: 1 });
    model.update();
    expect(model.ahead).toEqual([]);
    expect(model.notes).toMatchObject([{ afterN: 2, why: 'user voice', ahead: 1 }]);
  });

  it('strikes the words a cut took off the tape and marks where the voice stopped', () => {
    const { add, cell, model } = session();
    cell(100, 1, 'respond', ['The', 'quickest', 'way']);
    cell(200, 2, 'continue', ['is', 'usually', 'by']);
    cell(300, 3, 'continue', ['taking', 'a', 'train']);
    add(400, 'msg', { type: 'tape.cut', heard: 'The quickest way is usually', said: 5, unsaid: ['by', 'taking', 'a', 'train'] });
    model.update();
    expect(model.rows.map((r) => [r.unsaid, r.cut])).toEqual([
      [0, false],
      [1, true],
      [3, false],
    ]);
    // Nothing heard: the mark goes on the reply's first row.
    const b = session();
    b.cell(100, 1, 'respond', ['Sure']);
    b.add(200, 'msg', { type: 'tape.cut', heard: '', said: 0, unsaid: ['Sure'] });
    b.model.update();
    expect(b.model.rows[0]).toMatchObject({ unsaid: 1, cut: true });
  });

  it('has nothing to show for a session without a tape stage', () => {
    const traj = new Trajectory({ format: TRAJECTORY_FORMAT, formatVersion: 1, id: 't', createdAt: '', mode: 'live', graph: { nodes: [{ id: 'turn', type: 'turn.duplex', params: {} }], edges: [] } as never, graphHash: '' });
    traj.append({ seq: 0, t: 0, node: 'turn', kind: 'msg', data: { type: 'tape.cell', n: 1 } });
    const m = new TapeModel(traj);
    expect(m.node).toBeNull();
    expect(m.update()).toBe(false);
    expect(m.rows).toEqual([]);
  });

  it('keeps the tape in cell order when a late answer comes after the cells decided without it', () => {
    const { cell, model } = session();
    expect([model.cellMs, model.cellS]).toEqual([500, 1]);
    cell(100, 1, 'listen', [], { ms: 200 });
    cell(900, 3, 'listen', [], { forced: 'a call is still out' });
    cell(950, 2, 'respond', ['Hi'], { ms: 1300, late: true });
    model.update();
    expect(model.rows.map((r) => r.n)).toEqual([1, 2, 3]);
    expect(lineTime(model.rows[1].line)).toBe(1.28);
  });

  it('folds runs of idle cells, and groups the odds by way of acting', () => {
    const { cell, model } = session();
    cell(0, 1, 'listen', [], { ms: 200 });
    for (let n = 2; n <= 5; n++) cell(n * 640, n, 'listen', [], { forced: 'idle' });
    cell(6 * 640, 6, 'listen', [], { forced: 'TTS queue full' });
    cell(7 * 640, 7, 'listen', [], { forced: 'idle' });
    cell(8 * 640, 8, 'listen', [], { forced: 'idle' });
    model.update();
    const items = groupRows(model.rows);
    expect(items.map((i) => (i.kind === 'idle' ? `idle ${i.rows.length}` : i.row.n))).toEqual([1, 'idle 4', 6, 7, 8]);
    expect(seriesOdds({ listen: 0.6, resume: 0.1, respond: 0.3 })).toEqual({ respond: 0.4 });
    expect(seriesOdds({ continue: 0.9, finish: 0.1 })).toEqual({ finish: 0.1 });
  });
});
