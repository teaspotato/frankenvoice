import { useCallback, useEffect, useRef, useState, type CSSProperties } from 'react';
import { registry } from '../nodes';
import { GRAPH_FORMAT, type GraphSpec } from '../core/graph';
import type { SimScheduler } from '../core/scheduler';
import { prepareReplay } from '../core/replay';
import { decodeBundle, encodeBundle } from '../core/bundle';
import { compareTrajectories, type Comparison } from '../core/compare';
import type { Trajectory } from '../core/trajectory';
import { SpanDetails } from './SpanDetails';
import type { Json } from '../core/types';
import { micMonitor, replaySpeaker, setOutputMuted, setOutputPaused, setOutputSuppressed } from '../client/audio';
import { LiveClient, type LiveHandlers } from '../client/live';
import type { TurnModelLoad } from '../live-protocol';
import { resolveParams } from '../core/node';
import { turnModelRequest } from '../nodes/turn-model';
import {
  CONFIG_FORMAT,
  CONFIG_FORMAT_VERSION,
  buildGraph,
  configFromGraph,
  defaultConfig,
  normalizeConfig,
  requiredUpstreams,
  STAGES,
  stageParams,
  stageType,
  type PipelineConfig,
  type StageId,
} from '../pipeline';
import { SettingsPanel } from './Settings';
import { PipelineView } from './PipelineView';
import { StagePanel } from './StagePanel';
import { Dashboard } from './Dashboard';
import { EventLog } from './EventLog';
import { TapeInspector } from './TapeInspector';
import { Splitter, useLayout } from './Splitter';
import { ActivityTracker, NO_ACTIVITY, type Activity } from './activity';

const CONFIG_KEY = 'frankenvoice.pipeline';
const AUTOSAVE_KEY = 'frankenvoice.autosave';
const MUTE_KEY = 'frankenvoice.mute';
const TITLE_KEY = 'frankenvoice.titleModel';

function loadConfig(): PipelineConfig {
  try {
    const raw = localStorage.getItem(CONFIG_KEY);
    const saved = raw ? (JSON.parse(raw) as { formatVersion?: number; stages?: PipelineConfig }) : null;
    // A config saved by another version starts over from the defaults.
    if (saved?.formatVersion === CONFIG_FORMAT_VERSION) return normalizeConfig(saved.stages);
  } catch {
    /* storage unavailable or corrupt */
  }
  return defaultConfig();
}

function download(name: string, data: BlobPart, type: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([data], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

interface Session {
  traj: Trajectory;
  saved?: string;
  comparison?: Comparison;
  /** Replay stopped by the user before the source ended: compare only what ran. */
  stoppedAt?: number;
}

function describeComparison(s: Session): string {
  const c = s.comparison!;
  if (c.identical) return 'identical to source ✓';
  if (s.stoppedAt !== undefined) {
    // Everything before the stop turn should match; the stop itself is new.
    const stopIdx = s.traj.events.findIndex((e) => e.kind === 'node.stop');
    const sec = (s.stoppedAt / 1000).toFixed(1);
    return c.matched >= stopIdx ? `stopped at ${sec} s · matches source so far ✓` : `stopped at ${sec} s · diverged at event ${c.matched}`;
  }
  return `diverged at event ${c.matched}`;
}

export function App() {
  const [config, setConfig] = useState<PipelineConfig>(loadConfig);
  const [selected, setSelected] = useState<StageId | null>(null);
  /**
   * Who this page is to the server: the host (settings, recordings, any pipeline) or a public demo visitor, who
   * runs the host's pipeline and sees it read-only.
   */
  const [me, setMe] = useState<{ host: boolean; public: boolean }>({ host: true, public: false });
  useEffect(() => {
    fetch('/api/me')
      .then((r) => (r.ok ? (r.json() as Promise<{ host: boolean; public: boolean; graph?: GraphSpec }>) : null))
      .then((m) => {
        if (!m) return;
        setMe({ host: m.host, public: m.public });
        if (!m.host && m.graph) setConfig(configFromGraph(m.graph));
      })
      .catch(() => {});
  }, []);
  const host = me.host;
  // The turn model is big: the server downloads and compiles it as soon as it is configured, not when a session starts.
  const turnModelReq =
    stageType(config, 'turnmodel') === 'turnmodel.ssl'
      ? turnModelRequest(resolveParams(registry['turnmodel.ssl'], stageParams(config, 'turnmodel')) as { modelUrl: string; encoder: string; ep: string })
      : null;
  const turnModelKey = JSON.stringify(turnModelReq);
  const [turnModel, setTurnModel] = useState<TurnModelLoad | undefined>();
  useEffect(() => {
    setTurnModel(undefined);
    if (!turnModelReq || !host) return;
    let done = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = () =>
      fetch('/api/turn-model', { method: 'POST', body: JSON.stringify(turnModelReq) })
        .then((r) => (r.ok ? (r.json() as Promise<TurnModelLoad>) : Promise.reject(new Error(`HTTP ${r.status}`))))
        .then((l) => {
          if (done) return;
          setTurnModel(l);
          if (l.state === 'loading') timer = setTimeout(poll, 1000);
        })
        .catch(() => {});
    void poll();
    return () => {
      done = true;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [turnModelKey, host]);
  const [running, setRunning] = useState<'live' | 'replay' | null>(null);
  /** Where the running run runs: the server (live sessions, replays with live channels) or the page (other replays). */
  const [runWhere, setRunWhere] = useState<'server' | 'page'>('page');
  /** Mute switches on the Mic and Playback lanes: the recorded mic (heard during replay) and the agent's voice. */
  const [muted, setMuted] = useState<{ mic: boolean; agent: boolean }>(() => {
    try {
      return { mic: false, agent: false, ...(JSON.parse(localStorage.getItem(MUTE_KEY) ?? '{}') as object) };
    } catch {
      return { mic: false, agent: false };
    }
  });
  useEffect(() => {
    setOutputMuted('mic', muted.mic);
    setOutputMuted('agent', muted.agent);
    try {
      localStorage.setItem(MUTE_KEY, JSON.stringify(muted));
    } catch {
      /* storage unavailable: muting still works for this tab */
    }
  }, [muted]);
  /** The running replay plays a session back as recorded: the dashboard keeps showing that session. */
  const [playingBack, setPlayingBack] = useState(false);
  const [nodeStates, setNodeStates] = useState<Record<string, Record<string, Json>>>({});
  /** Node type behind each entry of nodeStates (the run they came from may predate a config change). */
  const [stateTypes, setStateTypes] = useState<Record<string, string>>({});
  const [sessions, setSessions] = useState<Session[]>([]);
  const sessionsRef = useRef(sessions);
  sessionsRef.current = sessions;
  const [current, setCurrent] = useState<Session | null>(null);
  const [health, setHealth] = useState<Record<string, boolean> | null>(null);
  const [replayOpts, setReplayOpts] = useState({ audible: true, realtime: true, useCurrentConfig: false, liveChannels: '' });
  const [notice, setNotice] = useState('');
  /** The session whose name is being edited: a Session, or the id of one only on disk. */
  const [renaming, setRenaming] = useState<Session | string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [serverSessions, setServerSessions] = useState<string[]>([]);
  /** Name each finished live session from its content, with this model ('' = off). */
  const [autoTitle, setAutoTitle] = useState(() => {
    try {
      return localStorage.getItem(TITLE_KEY) ?? 'gemma-4-12b-it';
    } catch {
      return 'gemma-4-12b-it';
    }
  });
  const [titleOn, setTitleOn] = useState(() => {
    try {
      return localStorage.getItem(TITLE_KEY + '.on') !== 'off';
    } catch {
      return true;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(TITLE_KEY, autoTitle);
      localStorage.setItem(TITLE_KEY + '.on', titleOn ? 'on' : 'off');
    } catch {
      /* storage unavailable */
    }
  }, [autoTitle, titleOn]);
  const [autosave, setAutosave] = useState(() => {
    try {
      return localStorage.getItem(AUTOSAVE_KEY) !== 'off';
    } catch {
      return true;
    }
  });
  // Read when a run finishes, so toggling mid-run applies to that run.
  const autosaveRef = useRef(autosave);
  const [layout, setLayout] = useLayout();
  const [activity, setActivity] = useState<Activity>(NO_ACTIVITY);
  const [paused, setPaused] = useState(false);
  /** Events of the timeline item last clicked, to show in the event log. */
  const [focus, setFocus] = useState<{ seqs: number[]; n: number } | null>(null);
  /** The timeline item last clicked, shown in full in the side panel. */
  const [picked, setPicked] = useState<number[] | null>(null);
  const stopActivity = useRef<() => void>(() => {});
  /** The session clock of what runs (or ran last). */
  const nowRef = useRef<() => number>(() => 0);
  const simRef = useRef<SimScheduler | null>(null);
  const clientRef = useRef<LiveClient | null>(null);
  /** Who else is on the server (from its live socket, opened at load). */
  const [users, setUsers] = useState<{ pages: number; sessions: number; max: number; queued: number } | null>(null);
  /** This page's place in the queue for the box (0: not waiting). */
  const [queuePos, setQueuePos] = useState(0);
  /** When this page's wait ends at the latest, and when its live session hits the time limit (Date.now() ms). */
  const [queueEta, setQueueEta] = useState<number | null>(null);
  const [endsAt, setEndsAt] = useState<number | null>(null);
  /** Ticks each second while a countdown shows. */
  const [clockNow, setClockNow] = useState(() => Date.now());
  useEffect(() => {
    if (queueEta === null && endsAt === null) return;
    const t = setInterval(() => setClockNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [queueEta, endsAt]);
  const reconnectRef = useRef<() => void>(() => {});
  /** The server run in progress: the session it replays, whether its result is kept, where it stopped. */
  const serverRun = useRef<{ origin?: Session; keep: boolean; stoppedAt?: number } | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem(CONFIG_KEY, JSON.stringify({ formatVersion: CONFIG_FORMAT_VERSION, stages: config }));
    } catch {
      /* storage unavailable */
    }
  }, [config]);

  useEffect(() => {
    autosaveRef.current = autosave;
    try {
      localStorage.setItem(AUTOSAVE_KEY, autosave ? 'on' : 'off');
    } catch {
      /* storage unavailable */
    }
  }, [autosave]);

  const refreshProxy = useCallback(() => {
    fetch('/api/health')
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));
    if (!host) return setServerSessions([]);
    fetch('/api/sessions')
      .then((r) => (r.ok ? r.json() : []))
      .then((list: { id: string }[]) => setServerSessions(list.map((s) => s.id)))
      .catch(() => {});
  }, [host]);
  useEffect(refreshProxy, [refreshProxy]);

  // ---- running ----------------------------------------------------------------

  /** Batches node states into one render per frame. */
  const statesRef = useRef<{ pending: Record<string, Record<string, Json>>; raf: number }>({ pending: {}, raf: 0 });
  const pushState = (node: string, state: Record<string, Json>) => {
    const st = statesRef.current;
    st.pending[node] = state;
    if (!st.raf)
      st.raf = requestAnimationFrame(() => {
        st.raf = 0;
        const p = st.pending;
        st.pending = {};
        setNodeStates((s) => ({ ...s, ...p }));
      });
  };

  const attach = (traj: Trajectory, now: () => number) => {
    nowRef.current = now;
    setNodeStates({});
    setStateTypes(Object.fromEntries(traj.meta.graph.nodes.map((n) => [n.id, n.type])));
    stopActivity.current();
    const tracker = new ActivityTracker();
    const unsub = traj.subscribe((e) => tracker.consume(e, performance.now()));
    const timer = setInterval(() => setActivity(tracker.snapshot(performance.now())), 100);
    stopActivity.current = () => {
      unsub();
      clearInterval(timer);
      setActivity(NO_ACTIVITY);
    };
  };

  /**
   * A live run or a replay with a changed config becomes a new session in the list. Playing a session
   * back as recorded doesn't: the view returns to that session, with a notice only if the replay diverged.
   */
  const finishSession = (traj: Trajectory, origin?: Session, keep = true, stoppedAt?: number) => {
    const s: Session = { traj, stoppedAt };
    if (origin) s.comparison = compareTrajectories(origin.traj, traj);
    if (keep) {
      setSessions((ss) => [s, ...ss]);
      setCurrent(s);
    } else {
      setCurrent(origin!);
      const verdict = describeComparison(s);
      if (!verdict.includes('✓')) setNotice(`Playback of ${origin!.traj.meta.id} ${verdict}: the code changed since it was recorded.`);
    }
    setRunning(null);
    setRunWhere('page');
    stopActivity.current();
    setPaused(false);
  };

  const saveSession = async (s: Session) => {
    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(s.traj.meta.id)}`, { method: 'PUT', body: encodeBundle(s.traj) as BodyInit });
      if (!res.ok) throw new Error(`save failed: ${res.status}`);
      s.saved = ((await res.json()) as { path: string }).path;
      setServerSessions((l) => [s.traj.meta.id, ...l.filter((id) => id !== s.traj.meta.id)]);
      setSessions((ss) => [...ss]);
    } catch (e) {
      setNotice(`Session not saved (${String(e)}). It stays in this tab and can be exported.`);
    }
  };

  // ---- the server ------------------------------------------------------------------

  /** What the server tells about its run, against the latest state of this component. */
  const handlers = useRef<LiveHandlers>(null!);
  handlers.current = {
    session(traj) {
      const run = serverRun.current;
      if (!host && traj.meta.mode === 'live') setConfig(configFromGraph(traj.meta.graph));
      attach(traj, () => clientRef.current?.now() ?? 0);
      if (run?.keep !== false) setCurrent({ traj });
      setPlayingBack(run?.keep === false);
      setRunning(traj.meta.mode);
      setRunWhere('server');
      setPlayFrom(0);
    },
    states(states) {
      for (const [node, state] of Object.entries(states)) pushState(node, state);
    },
    ended(info) {
      setEndsAt(null);
      if (serverRun.current) serverRun.current.stoppedAt = info.stoppedAt;
    },
    bundle(bytes) {
      const run = serverRun.current;
      serverRun.current = null;
      let traj: Trajectory;
      try {
        traj = decodeBundle(bytes);
      } catch (e) {
        setRunning(null);
        setRunWhere('page');
        return setNotice(`The server's recording could not be read: ${String(e)}`);
      }
      finishSession(traj, run?.origin, run?.keep ?? true, run?.stoppedAt);
    },
    saved(id, path) {
      if (!host) return;
      setSessions((ss) => {
        const s = ss.find((x) => x.traj.meta.id === id && x.traj.meta.mode === 'live');
        if (s) s.saved = path;
        return [...ss];
      });
      setServerSessions((l) => [id, ...l.filter((x) => x !== id)]);
    },
    renamed(from, to, path) {
      setSessions((ss) => {
        const s = ss.find((x) => x.traj.meta.id === from && x.traj.meta.mode === 'live');
        if (s) {
          s.traj.meta.id = to;
          s.saved = path;
        }
        return [...ss];
      });
      setServerSessions((l) => l.map((x) => (x === from ? to : x)));
    },
    notice: (text) => setNotice(text),
    users: (u) => setUsers(u),
    limit: (leftMs) => setEndsAt(Date.now() + leftMs),
    queue(position, started, waitMs) {
      setQueuePos(position);
      setQueueEta(position && waitMs !== null ? Date.now() + waitMs : null);
      if (position) setNotice(`The GPU box is busy: you're ${position === 1 ? 'next' : `number ${position}`} in line. Your session starts by itself.`);
      else if (started) setNotice('');
      // Left the queue: nothing will start.
      else serverRun.current = null;
    },
    closed(reason) {
      setUsers(null);
      setQueuePos(0);
      setQueueEta(null);
      setEndsAt(null);
      reconnectRef.current();
      if (!serverRun.current) return;
      serverRun.current = null;
      setRunning(null);
      setRunWhere('page');
      stopActivity.current();
      setPaused(false);
      setNotice(`Lost the server: ${reason}`);
    },
  };
  const client = () => {
    const h = handlers;
    return (clientRef.current ??= new LiveClient({
      session: (t) => h.current.session(t),
      states: (s) => h.current.states(s),
      ended: (i) => h.current.ended(i),
      bundle: (b) => h.current.bundle(b),
      saved: (id, p) => h.current.saved(id, p),
      renamed: (f, t, p) => h.current.renamed(f, t, p),
      notice: (t) => h.current.notice(t),
      users: (u) => h.current.users(u),
      queue: (p, s, w) => h.current.queue(p, s, w),
      limit: (l) => h.current.limit(l),
      closed: (r) => h.current.closed(r),
    }));
  };

  // Connect at load to show who else is on the server; after losing it, try again now and then.
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    let alive = true;
    const tryConnect = () =>
      client()
        .connect()
        .catch(() => {
          if (alive) timer = setTimeout(tryConnect, 5000);
        });
    reconnectRef.current = () => {
      clearTimeout(timer);
      if (alive) timer = setTimeout(tryConnect, 5000);
    };
    tryConnect();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** Start a run on the server; `send` asks for it once connected. */
  const runOnServer = async (run: { origin?: Session; keep: boolean }, send: (c: LiveClient) => void) => {
    setNotice('');
    const c = client();
    try {
      await c.connect();
    } catch (e) {
      return setNotice(String(e));
    }
    serverRun.current = run;
    send(c);
  };

  const runLive = () =>
    runOnServer({ keep: true }, (c) =>
      c.start(buildGraph(config), { env: { userAgent: navigator.userAgent }, autosave: autosaveRef.current, title: titleOn ? autoTitle : '' }),
    );

  const stop = () => {
    if (runWhere === 'server') return clientRef.current?.stop();
    if (running === 'replay') {
      simRef.current?.stop();
      setOutputPaused(false);
      setPaused(false);
    }
  };

  /** Session time ▶ starts from when nothing runs; set by clicking the timeline. */
  const [playFrom, setPlayFrom] = useState(0);
  /** Set when a seek backwards restarts the replay from this time. */
  const restartAt = useRef<number | null>(null);
  // A different session starts from its beginning.
  useEffect(() => setPlayFrom(0), [current?.traj]);
  useEffect(() => setPicked(null), [current?.traj]);

  /** Run everything before `t` at once with outputs silenced, then let the replay carry on from there. */
  const skipTo = (sim: SimScheduler, t: number) => {
    setOutputSuppressed(true);
    try {
      sim.seek(t);
    } finally {
      setOutputSuppressed(false);
    }
  };

  const replay = async (origin: Session, from = 0) => {
    const source = origin.traj;
    const keep = replayOpts.useCurrentConfig;
    setNotice('');
    const liveChannels = replayOpts.liveChannels
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
    if (liveChannels.length) {
      // Live channels are served by the server's drivers, so the replay runs there, in real time.
      return runOnServer({ origin, keep }, (c) =>
        c.replay(encodeBundle(source), { graph: keep ? buildGraph(config) : undefined, liveChannels, audible: replayOpts.audible }),
      );
    }
    const outputChannels = Object.values(registry).flatMap((d) => (d.channels ?? []).filter((c) => c.output).map((c) => c.name));
    const { runtime: rt, sim } = prepareReplay(source, {
      registry,
      graph: keep ? buildGraph(config) : source.meta.graph,
      drivers: { speaker: replaySpeaker },
      monitorChannels: replayOpts.audible && replayOpts.realtime ? outputChannels : [],
    });
    simRef.current = sim;
    attach(rt.trajectory, () => sim.now());
    rt.onState(pushState);
    if (keep) setCurrent({ traj: rt.trajectory });
    setPlayingBack(!keep);
    setRunning('replay');
    setRunWhere('page');
    setPlayFrom(0);
    if (replayOpts.realtime) {
      // Hear the user's side too: the mic frames as the replay reaches them.
      const mic = replayOpts.audible ? micMonitor() : null;
      const unsub = mic
        ? rt.trajectory.subscribe((e) => {
            if (e.kind === 'msg' && e.node === 'mic' && e.media) {
              const d = e.data as { sampleRate?: number };
              mic.play(new Float32Array(rt.trajectory.tracks.readF32Range(e.media.track, e.media.start, e.media.end)), d.sampleRate ?? 16000);
            }
          })
        : () => {};
      const done = sim.runRealtime(1);
      if (from > 0) skipTo(sim, from);
      await done;
      unsub();
      mic?.close();
    } else sim.runFast();
    const stoppedAt = rt.isRunning ? sim.now() : undefined;
    if (rt.isRunning) rt.stop();
    const again = restartAt.current;
    if (again !== null) {
      // Seeking backwards: start over and skip to the new time; this partial run isn't kept.
      restartAt.current = null;
      stopActivity.current();
      return replay(origin, again);
    }
    finishSession(rt.trajectory, origin, keep, stoppedAt);
  };

  /** Seek: a replay jumps there (restarting if it's earlier); otherwise ▶ will start there. */
  const seek = (t: number) => {
    t = Math.max(0, t);
    const sim = simRef.current;
    // A replay on the server can't seek.
    if (running === 'replay' && runWhere === 'server') return;
    if (running === 'replay' && sim && replayOpts.realtime) {
      if (t > sim.now()) skipTo(sim, t);
      else {
        restartAt.current = t;
        sim.stop();
      }
    } else if (!running) setPlayFrom(t);
  };

  /** Pause or resume a realtime replay: the session clock and the audio already scheduled both freeze. */
  const togglePause = () => {
    if (running === 'replay' && runWhere === 'server') {
      if (paused) clientRef.current?.resume();
      else clientRef.current?.pause();
      setOutputPaused(!paused);
      return setPaused(!paused);
    }
    const sim = simRef.current;
    if (running !== 'replay' || !sim) return;
    if (sim.isPaused) sim.resume();
    else sim.pause();
    setOutputPaused(sim.isPaused);
    setPaused(sim.isPaused);
  };

  const transport = (
    <span className="transport">
      {running === 'replay' ? (
        <button onClick={togglePause} disabled={!replayOpts.realtime && runWhere === 'page'} title={paused ? 'Resume (space)' : 'Pause (space)'}>
          {paused ? '▶' : '❚❚'}
        </button>
      ) : (
        <button
          onClick={() => current && replay(current, playFrom)}
          disabled={!!running || !current}
          title={running === 'live' ? "A live session can't be paused" : 'Replay this session (space)'}
        >
          ▶
        </button>
      )}
      <button onClick={stop} disabled={!running} title="Stop">
        ■
      </button>
    </span>
  );

  // Space plays, pauses or resumes, unless typing somewhere.
  const spaceRef = useRef(() => {});
  spaceRef.current = () => (running === 'replay' ? togglePause() : !running && current && replay(current, playFrom));
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || (e.target as HTMLElement).closest('input, textarea, select, button, [contenteditable]')) return;
      e.preventDefault();
      spaceRef.current();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  /** Clear the dashboard and stage states for a fresh start; past sessions stay in the list. */
  const newSession = () => {
    setCurrent(null);
    setNodeStates({});
    setStateTypes({});
    setNotice('');
  };

  /** Remove a session from the list and, for a recorded live session, its file on disk. */
  const deleteSession = async (id: string, s?: Session) => {
    const onDisk = (!s || s.traj.meta.mode === 'live') && serverSessions.includes(id);
    if (!confirm(onDisk ? `Delete session ${id}? Its file on disk is deleted too.` : `Remove ${s?.traj.meta.mode === 'replay' ? 'replay' : 'session'} ${id} from the list?`)) return;
    if (onDisk) {
      const res = await fetch(`/api/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' }).catch(() => null);
      if (!res?.ok) return setNotice(`could not delete ${id} from disk`);
      setServerSessions((l) => l.filter((x) => x !== id));
    }
    if (s) {
      setSessions((ss) => ss.filter((x) => x !== s));
      if (current === s) newSession();
    }
  };

  const loadServerSession = async (id: string) => {
    const res = await fetch(`/api/sessions/${encodeURIComponent(id)}`);
    let traj;
    try {
      traj = decodeBundle(new Uint8Array(await res.arrayBuffer()));
    } catch (e) {
      return setNotice(`${id}: ${String(e)}`);
    }
    // The file name is the session's name; it changes when the session is renamed.
    traj.meta.id = id;
    const s = { traj, saved: id };
    setSessions((ss) => [s, ...ss.filter((x) => x.traj.meta.id !== id || x.traj.meta.mode !== traj.meta.mode)]);
    setCurrent(s);
  };

  /** Rename a session; a saved one has its file renamed too. */
  const renameSession = async (from: string, raw: string, s?: Session) => {
    setRenaming(null);
    const to = raw.trim();
    if (!to || to === from) return;
    if (s?.saved || (!s && serverSessions.includes(from))) {
      const res = await fetch(`/api/sessions/${encodeURIComponent(from)}`, { method: 'PATCH', body: JSON.stringify({ id: to }) }).catch(() => null);
      if (!res?.ok) return setNotice(`could not rename ${from}: ${res ? ((await res.json()) as { error: string }).error : 'proxy not running'}`);
      if (s) s.saved = ((await res.json()) as { path: string }).path;
      setServerSessions((l) => l.map((x) => (x === from ? to : x)));
    } else if (sessions.some((x) => x !== s && x.traj.meta.id === to) || serverSessions.includes(to)) {
      return setNotice(`a session named ${to} already exists`);
    }
    if (s) s.traj.meta.id = to;
    setNotice('');
    setSessions((ss) => [...ss]);
  };

  const nameField = (id: string, s?: Session) =>
    renaming === (s ?? id) ? (
      <input
        className="session-name"
        autoFocus
        defaultValue={id}
        maxLength={100}
        onFocus={(e) => e.target.select()}
        onBlur={(e) => renameSession(id, e.target.value, s)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') e.currentTarget.blur();
          if (e.key === 'Escape') setRenaming(null);
        }}
      />
    ) : (
      <button
        className="link session-name"
        title="Click to rename"
        disabled={!!running && current === s}
        onClick={() => {
          if (s) setCurrent(s);
          setRenaming(s ?? id);
        }}
      >
        {id}
      </button>
    );

  const importFile = async (f: File) => {
    try {
      if (f.name.endsWith('.json')) {
        if (running) throw new Error('Stop the run before loading a config: stage settings are locked while it runs.');
        const j = JSON.parse(await f.text()) as { format?: string; formatVersion?: number };
        if (j.format === CONFIG_FORMAT && j.formatVersion !== CONFIG_FORMAT_VERSION)
          throw new Error(`pipeline config format ${j.formatVersion ?? 1}; this version only loads format ${CONFIG_FORMAT_VERSION} (older configs are not migrated)`);
        if (j.format === CONFIG_FORMAT) setConfig(normalizeConfig((j as { stages: PipelineConfig }).stages));
        else if (j.format === GRAPH_FORMAT) setConfig(configFromGraph(j as GraphSpec));
        else throw new Error('not a frankenvoice pipeline config');
      } else {
        const traj = decodeBundle(new Uint8Array(await f.arrayBuffer()));
        const s = { traj };
        setSessions((ss) => [s, ...ss]);
        setCurrent(s);
      }
    } catch (e) {
      setNotice(String(e));
    }
  };

  // ---- render -------------------------------------------------------------------

  const missingKeys = health ? requiredUpstreams(config).filter((k) => !health[k]) : [];
  /** Selecting a stage shows its settings, so it opens a collapsed side panel. */
  const selectStage = (id: StageId | null, reveal = true) => {
    setSelected(id);
    // The tape stage's settings, and what it is doing in the Tape inspector below.
    const tape = id === 'turn' && stageType(config, 'turn') === 'turn.tape';
    if (reveal && id && (!layout.sideOpen || (tape && (!layout.logOpen || layout.logTab !== 'tape'))))
      setLayout({ ...layout, sideOpen: true, ...(tape ? { logOpen: true, logTab: 'tape' as const } : {}) });
  };
  const logTabs = (
    <span className="log-tabs">
      {(['events', 'tape'] as const).map((tab) => (
        <button
          key={tab}
          className={`link${layout.logTab === tab ? ' on' : ''}`}
          onClick={() => setLayout({ ...layout, logTab: tab, logOpen: true })}
          title={tab === 'tape' ? 'The tape stage (turn.tape): every line Gemma sees, what it wrote and how sure it was, and each call’s prompt' : 'Every event of the session'}
        >
          {tab === 'events' ? 'Events' : 'Tape'}
        </button>
      ))}
    </span>
  );
  const stageOf = (lane: string | null): StageId | null =>
    lane === null ? null : STAGES.some((s) => s.id === lane) ? (lane as StageId) : 'tts';

  const mmss = (until: number) => {
    const s = Math.max(0, Math.round((until - clockNow) / 1000));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  };

  // Waiting for the box: only the queue shows; the session starts by itself when its turn comes.
  if (queuePos > 0 && !running)
    return (
      <div className="queue-page">
        <span className="brand">frankenvoice</span>
        <h1>The GPU box is busy</h1>
        <p>
          You're {queuePos === 1 ? 'next' : `number ${queuePos}`} in line. Your session starts by itself; keep this tab open.
        </p>
        {queueEta !== null && (
          <p className="eta">
            starts in at most <b>{mmss(queueEta)}</b>
          </p>
        )}
        {users && (
          <p className="muted">
            {users.pages} user{users.pages === 1 ? '' : 's'} connected · {users.sessions}/{users.max} live · {users.queued} waiting
          </p>
        )}
        <button onClick={() => clientRef.current?.stop()}>✕ Leave queue</button>
      </div>
    );

  return (
    <div
      className={`app${layout.logOpen ? '' : ' log-collapsed'}${layout.sideOpen ? '' : ' side-collapsed'}`}
      style={{ '--diagram-h': `${layout.diagram}px`, '--log-h': `${layout.log}px`, '--side-w': `${layout.side}px` } as CSSProperties}
    >
      <header className="toolbar">
        <span className="brand">frankenvoice</span>
        {queuePos > 0 && !running ? (
          <button onClick={() => clientRef.current?.stop()} title="Your session starts by itself when the one ahead ends">
            ✕ Leave queue ({queuePos === 1 ? 'next' : `${queuePos - 1} ahead`})
          </button>
        ) : !running ? (
          <button className="primary" onClick={runLive}>
            ● Run live
          </button>
        ) : (
          <button className="danger" onClick={stop}>
            ■ Stop {running}
          </button>
        )}
        {running === 'live' && endsAt !== null && (
          <span className={endsAt - clockNow < 30_000 ? 'warn' : 'muted'} title="Live sessions stop (and are saved) at the time limit, so others in line get their turn">
            {mmss(endsAt)} left
          </span>
        )}
        <button
          onClick={() => download('frankenvoice-pipeline.json', JSON.stringify({ format: CONFIG_FORMAT, formatVersion: CONFIG_FORMAT_VERSION, stages: config }, null, 2), 'application/json')}
        >
          Save config
        </button>
        <label className="button">
          Open…
          <input type="file" accept=".json,.fvs,.zip" hidden onChange={(e) => e.target.files?.[0] && importFile(e.target.files[0])} />
        </label>
        <button disabled={!!running || !current} onClick={newSession} title="Clear the dashboard; past sessions stay in the list">
          New session
        </button>
        {host && (
          <button disabled={!!running} onClick={() => confirm('Reset every stage to its default settings?') && setConfig(defaultConfig())}>
            Reset config
          </button>
        )}
        <span className="spacer" />
        {health === null ? (
          <span className="warn">server not reachable: run `pnpm dev`</span>
        ) : !host ? (
          <span className="muted" title="Conversations are recorded for research">
            demo · conversations are recorded
          </span>
        ) : missingKeys.length ? (
          <button className="link warn" onClick={() => setShowSettings(true)}>
            needs keys: {missingKeys.join(', ')}
          </button>
        ) : (
          <span className="ok">server ok</span>
        )}
        {users && (
          <span
            className={users.sessions >= users.max ? 'warn' : 'muted'}
            title={`${users.pages} page${users.pages === 1 ? '' : 's'} connected; ${users.sessions} of ${users.max} sessions running on the GPU box; ${users.queued} waiting`}
          >
            {users.pages} user{users.pages === 1 ? '' : 's'} · {users.sessions}/{users.max} live{users.queued ? ` · ${users.queued} waiting` : ''}
          </span>
        )}
        {host && <button onClick={() => setShowSettings(true)}>Settings</button>}
        <button
          onClick={() => setLayout({ ...layout, sideOpen: !layout.sideOpen })}
          title={layout.sideOpen ? 'Hide the side panel' : 'Show the side panel (stage settings, sessions)'}
        >
          {layout.sideOpen ? 'Hide panel ⇥' : '⇤ Show panel'}
        </button>
      </header>
      {showSettings && host && <SettingsPanel onClose={() => setShowSettings(false)} onSaved={refreshProxy} />}

      <section className="diagram">
        <Splitter layout={layout} onChange={setLayout} field="diagram" axis="y" sign={1} max={() => window.innerHeight - 260} />
        <PipelineView config={config} turnModel={turnModel} states={nodeStates} stateTypes={stateTypes} activity={activity} running={!!running} selected={selected} onSelect={selectStage} />
      </section>

      <section className="dash">
        <Dashboard
          trajectory={current?.traj}
          running={!!running}
          now={() => nowRef.current()}
          selected={selected}
          onSelect={(lane, item) => selectStage(stageOf(lane), !item)}
          transport={transport}
          onSeek={seek}
          muted={muted}
          onMute={(which) => setMuted((m) => ({ ...m, [which]: !m[which] }))}
          playhead={running === 'replay' && playingBack ? () => nowRef.current() : running ? null : playFrom}
          onFocusEvents={(seqs) => {
            setFocus((f) => ({ seqs, n: (f?.n ?? 0) + 1 }));
            setPicked(seqs);
          }}
        />
      </section>

      <Splitter layout={layout} onChange={setLayout} field="side" axis="x" sign={-1} max={() => window.innerWidth - 400} />
      <aside className="inspector">
        {picked && current && <SpanDetails trajectory={current.traj} seqs={picked} onClose={() => setPicked(null)} />}
        {selected ? (
          <StagePanel config={config} stage={selected} state={stateTypes[selected] === stageType(config, selected) ? nodeStates[selected] : undefined} disabled={!!running || !host} onChange={setConfig} />
        ) : (
          <p className="muted">
            {host
              ? "Click a stage in the diagram or a lane in the dashboard to configure it. The pipeline's wiring is defined in code (src/pipeline.ts)."
              : 'Click a stage in the diagram or a lane in the dashboard to see its settings and what it is doing.'}
          </p>
        )}

        <h3>Sessions</h3>
        {host && (
          <>
            <label className="autosave" title="The server saves each live session to its sessions folder when it stops (set when a session starts)">
              <input type="checkbox" checked={autosave} onChange={(e) => setAutosave(e.target.checked)} /> autosave live sessions
            </label>
            <label className="autosave" title="When a live session stops, the server asks this model on the LLM endpoint for a short title and renames the saved session to it">
              <input type="checkbox" checked={titleOn} onChange={(e) => setTitleOn(e.target.checked)} /> name sessions with{' '}
              <input className="title-model" value={autoTitle} disabled={!titleOn} onChange={(e) => setAutoTitle(e.target.value)} />
            </label>
          </>
        )}
        <div className="replay-opts">
          <label>
            <input type="checkbox" checked={replayOpts.realtime} onChange={(e) => setReplayOpts({ ...replayOpts, realtime: e.target.checked })} /> real time
          </label>
          <label>
            <input type="checkbox" checked={replayOpts.audible} onChange={(e) => setReplayOpts({ ...replayOpts, audible: e.target.checked })} /> audible
          </label>
          <label title="Run the recording through the current stage config instead of the one it was recorded with. The result is kept as a new entry in the list, compared with the original.">
            <input
              type="checkbox"
              checked={replayOpts.useCurrentConfig}
              onChange={(e) => setReplayOpts({ ...replayOpts, useCurrentConfig: e.target.checked })}
            />{' '}
            replay with current config
          </label>
          <label hidden={!host} title="Channels served live by the server's drivers instead of the recording; such a replay runs on the server, in real time">
            live channels{' '}
            <input
              placeholder="e.g. openai,gpubox-tts"
              value={replayOpts.liveChannels}
              onChange={(e) => setReplayOpts({ ...replayOpts, liveChannels: e.target.value })}
            />
          </label>
        </div>
        {notice && <p className="warn">{notice}</p>}
        <ul className="sessions">
          {sessions.map((s, i) => (
            <li
              key={i}
              className={current === s ? 'current' : ''}
              // Clicking the row (not one of its buttons) selects the session; the name renames it.
              onClick={(e) => !(e.target as HTMLElement).closest('button, input') && setCurrent(s)}
            >
              <div className="session-head">
                {s.traj.meta.mode === 'replay' ? '↻' : '●'}
                {nameField(s.traj.meta.id, s)}
              </div>
              <small>
                {(s.traj.duration / 1000).toFixed(1)} s · {s.traj.events.length} events
                {s.comparison && ` · ${describeComparison(s)}`}
                {s.saved && ' · saved'}
              </small>
              <span>
                <button disabled={!!running} onClick={() => replay(s)}>
                  Replay
                </button>
                {host && !s.saved && s.traj.meta.mode === 'live' && (
                  <button onClick={() => saveSession(s)} title="Save to the server's sessions folder">
                    Save
                  </button>
                )}
                <button onClick={() => download(`${s.traj.meta.id}.fvs`, encodeBundle(s.traj) as BlobPart, 'application/zip')}>Export</button>
                <button hidden={!host} disabled={!!running} onClick={() => setConfig(configFromGraph(s.traj.meta.graph))} title="Set every stage to what this session ran with">
                  Use its config
                </button>
                <button className="danger-link" disabled={!!running && current === s} onClick={() => deleteSession(s.traj.meta.id, s)}>
                  Delete
                </button>
              </span>
            </li>
          ))}
          {serverSessions
            .filter((id) => !sessions.some((s) => s.traj.meta.id === id))
            .map((id) => (
              <li key={id}>
                <div className="session-head">⤓{nameField(id)}</div>
                <small>on disk</small>
                <span>
                  <button onClick={() => loadServerSession(id)}>Load</button>
                  <button className="danger-link" onClick={() => deleteSession(id)}>
                    Delete
                  </button>
                </span>
              </li>
            ))}
        </ul>
      </aside>

      <section className="log">
        {layout.logOpen && <Splitter layout={layout} onChange={setLayout} field="log" axis="y" sign={-1} max={() => window.innerHeight - 260} />}
        {layout.logTab === 'tape' ? (
          <TapeInspector
            trajectory={current?.traj}
            live={!!running}
            open={layout.logOpen}
            onToggle={() => setLayout({ ...layout, logOpen: !layout.logOpen })}
            now={() => nowRef.current()}
            tabs={logTabs}
            focus={focus}
          />
        ) : (
          <EventLog trajectory={current?.traj} live={!!running} focus={focus} open={layout.logOpen} onToggle={() => setLayout({ ...layout, logOpen: !layout.logOpen })} tabs={logTabs} />
        )}
      </section>
    </div>
  );
}
