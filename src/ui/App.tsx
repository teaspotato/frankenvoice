import { useCallback, useEffect, useReducer, useRef, useState, type CSSProperties } from 'react';
import { registry } from '../nodes';
import { GRAPH_FORMAT, type GraphSpec } from '../core/graph';
import { Runtime } from '../core/runtime';
import { LiveScheduler, type Scheduler, type SimScheduler } from '../core/scheduler';
import { LiveWorld } from '../core/world';
import { prepareReplay } from '../core/replay';
import { decodeBundle, encodeBundle } from '../core/bundle';
import { compareTrajectories, type Comparison } from '../core/compare';
import type { Trajectory } from '../core/trajectory';
import { SpanDetails } from './SpanDetails';
import type { Json } from '../core/types';
import { browserDrivers, micMonitor, preloadTurnModel, warmUp, sessionTitle, setOutputMuted, setOutputPaused, setOutputSuppressed, turnModelLoad, watchTurnModels } from '../drivers';
import { resolveParams } from '../core/node';
import { turnModelRequest } from '../nodes/turn-model';
import {
  CONFIG_FORMAT,
  activeStages,
  buildGraph,
  configFromGraph,
  defaultConfig,
  normalizeConfig,
  requiredUpstreams,
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
import { Splitter, useLayout } from './Splitter';
import { ActivityTracker, NO_ACTIVITY, type Activity } from './activity';

const CONFIG_KEY = 'frankenvoice.pipeline';
const AUTOSAVE_KEY = 'frankenvoice.autosave';
const MUTE_KEY = 'frankenvoice.mute';
const TITLE_KEY = 'frankenvoice.autoTitle';

/** What was said in a session, for naming it: user turns and the agent's replies, in order. */
function transcriptOf(traj: Trajectory): string {
  const lines: string[] = [];
  for (const e of traj.events) {
    if (e.kind !== 'msg') continue;
    const d = e.data as { type?: string; text?: string };
    if (d.type === 'turn.user' && d.text) lines.push(`User: ${d.text}`);
    else if (d.type === 'llm.end' && d.text) lines.push(`Agent: ${d.text}`);
  }
  return lines.join('\n').slice(0, 6000);
}

/** A title made safe as a session (file) name. */
function titleToName(title: string): string {
  return title
    .replace(/[\/\\:*?"<>|\x00-\x1f]/g, ' ')
    .replace(/^[\s.]+/, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 60)
    .trim();
}

function loadConfig(): PipelineConfig {
  try {
    const raw = localStorage.getItem(CONFIG_KEY);
    if (raw) return normalizeConfig(JSON.parse(raw) as PipelineConfig);
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
  useEffect(() => {
    const t = setTimeout(warmUp, 1000);
    return () => clearTimeout(t);
  }, []);
  // The turn model is big: download and compile it as soon as it is configured, not when a session starts.
  const [, turnModelChanged] = useReducer((n: number) => n + 1, 0);
  useEffect(() => watchTurnModels(turnModelChanged), []);
  const turnModelReq =
    stageType(config, 'turnmodel') === 'turnmodel.ssl'
      ? turnModelRequest(resolveParams(registry['turnmodel.ssl'], stageParams(config, 'turnmodel')) as { modelUrl: string; encoder: string; ep: string })
      : null;
  const turnModelKey = JSON.stringify(turnModelReq);
  useEffect(() => {
    if (turnModelReq) preloadTurnModel(turnModelReq);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [turnModelKey]);
  const turnModel = turnModelReq ? turnModelLoad(turnModelReq) : undefined;
  const [running, setRunning] = useState<'live' | 'replay' | null>(null);
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
      return localStorage.getItem(TITLE_KEY) ?? 'claude-haiku-5-5';
    } catch {
      return 'claude-haiku-5-5';
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
  const titleRef = useRef({ on: titleOn, model: autoTitle });
  titleRef.current = { on: titleOn, model: autoTitle };
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
  const runtimeRef = useRef<Runtime | null>(null);
  const schedRef = useRef<Scheduler | null>(null);
  const simRef = useRef<SimScheduler | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem(CONFIG_KEY, JSON.stringify(config));
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
    fetch('/api/sessions')
      .then((r) => (r.ok ? r.json() : []))
      .then((list: { id: string }[]) => setServerSessions(list.map((s) => s.id)))
      .catch(() => {});
  }, []);
  useEffect(refreshProxy, [refreshProxy]);

  // ---- running ----------------------------------------------------------------

  const attach = (rt: Runtime, scheduler: Scheduler) => {
    runtimeRef.current = rt;
    schedRef.current = scheduler;
    setNodeStates({});
    setStateTypes(Object.fromEntries(rt.trajectory.meta.graph.nodes.map((n) => [n.id, n.type])));
    let pending: Record<string, Record<string, Json>> = {};
    let raf = 0;
    stopActivity.current();
    const tracker = new ActivityTracker();
    const unsub = rt.trajectory.subscribe((e) => tracker.consume(e, performance.now()));
    const timer = setInterval(() => setActivity(tracker.snapshot(performance.now())), 100);
    stopActivity.current = () => {
      unsub();
      clearInterval(timer);
      setActivity(NO_ACTIVITY);
    };
    rt.onState((node, state) => {
      pending[node] = state;
      if (!raf)
        raf = requestAnimationFrame(() => {
          raf = 0;
          const p = pending;
          pending = {};
          setNodeStates((s) => ({ ...s, ...p }));
        });
    });
  };

  /**
   * A live run or a replay with a changed config becomes a new session in the list. Playing a session
   * back as recorded doesn't: the view returns to that session, with a notice only if the replay diverged.
   */
  const finishSession = async (rt: Runtime, origin?: Session, keep = true, stoppedAt?: number) => {
    const s: Session = { traj: rt.trajectory, stoppedAt };
    if (origin) s.comparison = compareTrajectories(origin.traj, rt.trajectory);
    if (keep) {
      setSessions((ss) => [s, ...ss]);
      setCurrent(s);
    } else {
      setCurrent(origin!);
      const verdict = describeComparison(s);
      if (!verdict.includes('✓')) setNotice(`Playback of ${origin!.traj.meta.id} ${verdict}: the code changed since it was recorded.`);
    }
    setRunning(null);
    stopActivity.current();
    setPaused(false);
    if (rt.trajectory.meta.mode === 'live' && autosaveRef.current) await saveSession(s);
    if (rt.trajectory.meta.mode === 'live' && titleRef.current.on && titleRef.current.model) void nameSession(s, titleRef.current.model);
  };

  /** Rename a finished session after what it was about; on any failure it keeps its generated id. */
  const nameSession = async (s: Session, model: string) => {
    const transcript = transcriptOf(s.traj);
    if (!transcript) return;
    let name: string;
    try {
      name = titleToName(await sessionTitle(model, transcript));
    } catch (e) {
      return setNotice(`Could not title the session with ${model}: ${String(e).slice(0, 200)}`);
    }
    if (!name) return;
    const from = s.traj.meta.id;
    for (let n = 1; n <= 9; n++) {
      const to = n === 1 ? name : `${name} ${n}`;
      if (s.saved) {
        const res = await fetch(`/api/sessions/${encodeURIComponent(from)}`, { method: 'PATCH', body: JSON.stringify({ id: to }) }).catch(() => null);
        if (res?.status === 409) continue;
        if (!res?.ok) return;
        s.saved = ((await res.json()) as { path: string }).path;
        setServerSessions((l) => l.map((x) => (x === from ? to : x)));
      } else if (sessionsRef.current.some((x) => x !== s && x.traj.meta.id === to)) continue;
      s.traj.meta.id = to;
      setSessions((ss) => [...ss]);
      return;
    }
  };

  const saveSession = async (s: Session) => {
    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(s.traj.meta.id)}`, { method: 'PUT', body: encodeBundle(s.traj) as BodyInit });
      if (!res.ok) throw new Error(`save failed: ${res.status}`);
      s.saved = ((await res.json()) as { path: string }).path;
      setServerSessions((l) => [s.traj.meta.id, ...l.filter((id) => id !== s.traj.meta.id)]);
      setSessions((ss) => [...ss]);
    } catch (e) {
      /* proxy not running: the session stays in memory and can be exported */
      setNotice(`Session not saved (${String(e)}). It stays in this tab and can be exported.`);
    }
  };

  const runLive = () => {
    const scheduler = new LiveScheduler();
    const rt = new Runtime({
      graph: buildGraph(config),
      registry,
      scheduler,
      world: new LiveWorld(browserDrivers, scheduler),
      mode: 'live',
      env: { userAgent: navigator.userAgent },
    });
    attach(rt, scheduler);
    setCurrent({ traj: rt.trajectory });
    setRunning('live');
    setNotice('');
    rt.start();
  };

  const stop = () => {
    if (running === 'replay') {
      simRef.current?.stop();
      setOutputPaused(false);
      setPaused(false);
      return;
    }
    const rt = runtimeRef.current;
    if (!rt) return;
    rt.stop();
    void finishSession(rt);
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
    const outputChannels = Object.values(registry).flatMap((d) => (d.channels ?? []).filter((c) => c.output).map((c) => c.name));
    const liveChannels = replayOpts.liveChannels
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
    const { runtime: rt, sim } = prepareReplay(source, {
      registry,
      graph: replayOpts.useCurrentConfig ? buildGraph(config) : source.meta.graph,
      drivers: browserDrivers,
      liveChannels,
      monitorChannels: replayOpts.audible && replayOpts.realtime ? outputChannels : [],
    });
    simRef.current = sim;
    attach(rt, sim);
    if (keep) setCurrent({ traj: rt.trajectory });
    setPlayingBack(!keep);
    setRunning('replay');
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
    await finishSession(rt, origin, keep, stoppedAt);
  };

  /** Seek: a replay jumps there (restarting if it's earlier); otherwise ▶ will start there. */
  const seek = (t: number) => {
    t = Math.max(0, t);
    const sim = simRef.current;
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
        <button onClick={togglePause} disabled={!replayOpts.realtime} title={paused ? 'Resume (space)' : 'Pause (space)'}>
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
        const j = JSON.parse(await f.text()) as { format?: string };
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
  // The Playback lane belongs to the TTS stage when the browser speaks for itself.
  /** Selecting a stage shows its settings, so it opens a collapsed side panel. */
  const selectStage = (id: StageId | null, reveal = true) => {
    setSelected(id);
    if (reveal && id && !layout.sideOpen) setLayout({ ...layout, sideOpen: true });
  };
  const stageOf = (lane: string | null): StageId | null =>
    lane === null ? null : activeStages(config).some((s) => s.id === lane) ? (lane as StageId) : 'tts';

  return (
    <div
      className={`app${layout.logOpen ? '' : ' log-collapsed'}${layout.sideOpen ? '' : ' side-collapsed'}`}
      style={{ '--diagram-h': `${layout.diagram}px`, '--log-h': `${layout.log}px`, '--side-w': `${layout.side}px` } as CSSProperties}
    >
      <header className="toolbar">
        <span className="brand">frankenvoice</span>
        {!running ? (
          <button className="primary" onClick={runLive}>
            ● Run live
          </button>
        ) : (
          <button className="danger" onClick={stop}>
            ■ Stop {running}
          </button>
        )}
        <button
          onClick={() => download('frankenvoice-pipeline.json', JSON.stringify({ format: CONFIG_FORMAT, stages: config }, null, 2), 'application/json')}
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
        <button disabled={!!running} onClick={() => confirm('Reset every stage to its default settings?') && setConfig(defaultConfig())}>Reset config</button>
        <span className="spacer" />
        {health === null ? (
          <span className="warn">proxy not reachable: run `pnpm dev`</span>
        ) : missingKeys.length ? (
          <button className="link warn" onClick={() => setShowSettings(true)}>
            needs keys: {missingKeys.join(', ')}
          </button>
        ) : (
          <span className="ok">proxy ok</span>
        )}
        <button onClick={() => setShowSettings(true)}>Settings</button>
        <button
          onClick={() => setLayout({ ...layout, sideOpen: !layout.sideOpen })}
          title={layout.sideOpen ? 'Hide the side panel' : 'Show the side panel (stage settings, sessions)'}
        >
          {layout.sideOpen ? 'Hide panel ⇥' : '⇤ Show panel'}
        </button>
      </header>
      {showSettings && <SettingsPanel onClose={() => setShowSettings(false)} onSaved={refreshProxy} />}

      <section className="diagram">
        <Splitter layout={layout} onChange={setLayout} field="diagram" axis="y" sign={1} max={() => window.innerHeight - 260} />
        <PipelineView config={config} turnModel={turnModel} states={nodeStates} stateTypes={stateTypes} activity={activity} running={!!running} selected={selected} onSelect={selectStage} />
      </section>

      <section className="dash">
        <Dashboard
          trajectory={current?.traj}
          running={!!running}
          now={() => schedRef.current?.now() ?? 0}
          selected={selected}
          onSelect={(lane, item) => selectStage(stageOf(lane), !item)}
          transport={transport}
          onSeek={seek}
          muted={muted}
          onMute={(which) => setMuted((m) => ({ ...m, [which]: !m[which] }))}
          playhead={running === 'replay' && playingBack ? () => simRef.current?.now() ?? 0 : running ? null : playFrom}
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
          <StagePanel config={config} stage={selected} state={stateTypes[selected] === stageType(config, selected) ? nodeStates[selected] : undefined} disabled={!!running} onChange={setConfig} />
        ) : (
          <p className="muted">Click a stage in the diagram or a lane in the dashboard to configure it. The pipeline's wiring is defined in code (src/pipeline.ts).</p>
        )}

        <h3>Sessions</h3>
        <label className="autosave" title="Save each live session to the proxy's sessions folder when it stops">
          <input type="checkbox" checked={autosave} onChange={(e) => setAutosave(e.target.checked)} /> autosave live sessions
        </label>
        <label className="autosave" title="When a live session stops, ask this model for a short title and rename the session to it (uses the Anthropic key)">
          <input type="checkbox" checked={titleOn} onChange={(e) => setTitleOn(e.target.checked)} /> name sessions with{' '}
          <input className="title-model" value={autoTitle} disabled={!titleOn} onChange={(e) => setAutoTitle(e.target.value)} />
        </label>
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
          <label>
            live channels{' '}
            <input
              placeholder="e.g. anthropic,cartesia"
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
                {!s.saved && s.traj.meta.mode === 'live' && (
                  <button onClick={() => saveSession(s)} title="Save to the proxy's sessions folder">
                    Save
                  </button>
                )}
                <button onClick={() => download(`${s.traj.meta.id}.fvs`, encodeBundle(s.traj) as BlobPart, 'application/zip')}>Export</button>
                <button disabled={!!running} onClick={() => setConfig(configFromGraph(s.traj.meta.graph))} title="Set every stage to what this session ran with">
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
        <EventLog trajectory={current?.traj} live={!!running} focus={focus} open={layout.logOpen} onToggle={() => setLayout({ ...layout, logOpen: !layout.logOpen })} />
      </section>
    </div>
  );
}
