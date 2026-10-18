import type { TrajEvent } from '../core/types';
import type { StageId } from '../pipeline';

/** How long an output port counts as streaming after its last message, ms of wall time. */
export const EDGE_HOLD_MS = 350;

export interface Activity {
  /** Output ports (`node.port`) that carried data recently or have a stream open. */
  edges: Set<string>;
  /** Stages whose boolean state is on, with a short label for it. */
  stages: Map<StageId, string>;
}

export const NO_ACTIVITY: Activity = { edges: new Set(), stages: new Map() };

/**
 * Derives what is active right now from the live event stream: which connections
 * are carrying data and which stages are "on" (VAD hears speech, the LLM is
 * generating, the speaker is playing...). Telemetry such as per-frame VAD
 * probabilities doesn't count as data on a connection.
 */
export class ActivityTracker {
  private lastMsg = new Map<string, number>();
  private on = new Map<StageId, string>();
  /** Open streams per output port, e.g. an LLM response that hasn't ended. */
  private open = new Map<string, Set<string>>();
  private floor = 'listening';
  private bargeIn = false;

  consume(e: TrajEvent, wall: number) {
    const d = (e.data ?? {}) as Record<string, unknown>;
    if (e.kind === 'state' && e.node === 'turn' && typeof d.floor === 'string') {
      this.floor = d.floor;
      this.showTurn();
      return;
    }
    if (e.kind !== 'msg' || !e.port) return;
    const type = String(d.type ?? '');
    if (type.endsWith('.frame')) {
      if (type === 'cleanup.frame') this.set('cleanup', d.doubleTalk ? 'double talk' : null);
      return;
    }
    const key = `${e.node}.${e.port}`;
    this.lastMsg.set(key, wall);
    const rid = String(d.rid ?? '');
    switch (type) {
      case 'turn.decision':
        this.bargeIn = d.decision === 'candidate';
        this.showTurn();
        break;
      case 'vad.speech_start':
        this.set('vad', 'speech');
        break;
      case 'vad.speech_end':
        this.set('vad', null);
        break;
      case 'asr.partial':
        this.set('asr', 'hearing');
        break;
      case 'asr.final':
        this.set('asr', null);
        break;
      case 'llm.start':
        this.set('llm', 'generating');
        this.stream(key, rid, true);
        break;
      case 'llm.end':
        this.set('llm', null);
        this.stream(key, rid, false);
        break;
      case 'tts.segment_start':
        this.set('tts', 'synthesizing');
        this.stream(key, rid, true);
        break;
      case 'tts.response_end':
      case 'tts.cancelled':
        this.set('tts', null);
        this.stream(key, rid, false);
        break;
      case 'playback.start':
        // The browser TTS stage plays audio itself; otherwise the speaker does.
        this.set(e.node === 'speaker' ? 'speaker' : 'tts', 'playing');
        this.stream(key, rid, true);
        if (e.node === 'speaker') this.stream('speaker.played', rid, true);
        break;
      case 'playback.end':
      case 'playback.interrupted':
        this.set(e.node === 'speaker' ? 'speaker' : 'tts', null);
        this.open.delete(key);
        if (e.node === 'speaker') this.open.delete('speaker.played');
        break;
    }
  }

  snapshot(wall: number): Activity {
    const edges = new Set<string>();
    for (const [k, t] of this.lastMsg) if (wall - t < EDGE_HOLD_MS) edges.add(k);
    for (const [k, s] of this.open) if (s.size) edges.add(k);
    return { edges, stages: new Map(this.on) };
  }

  private showTurn() {
    this.set('turn', this.bargeIn ? 'user talking over' : this.floor === 'listening' ? null : this.floor);
  }

  private set(stage: StageId, label: string | null) {
    if (label) this.on.set(stage, label);
    else this.on.delete(stage);
  }

  private stream(key: string, rid: string, isOpen: boolean) {
    let s = this.open.get(key);
    if (!s) this.open.set(key, (s = new Set()));
    if (isOpen) s.add(rid);
    else s.delete(rid);
  }
}
