import { useCallback, useEffect, useRef, useState } from 'react';
import { registry } from '../nodes';
import { GRAPH_FORMAT, type GraphSpec } from '../core/graph';
import { Runtime } from '../core/runtime';
import { LiveScheduler, type Scheduler, type SimScheduler } from '../core/scheduler';
import { LiveWorld } from '../core/world';
import { prepareReplay } from '../core/replay';
import { decodeBundle, encodeBundle } from '../core/bundle';
import { compareTrajectories, type Comparison } from '../core/compare';
import type { Trajectory } from '../core/trajectory';
import type { Json } from '../core/types';
import { browserDrivers } from '../drivers';
import {
  CONFIG_FORMAT,
  activeStages,
  buildGraph,
  configFromGraph,
  defaultConfig,
  normalizeConfig,
  requiredUpstreams,
  stageType,
  type PipelineConfig,
  type StageId,
} from '../pipeline';
import { SettingsPanel } from './Settings';
import { PipelineView } from './PipelineView';
import { StagePanel } from './StagePanel';
import { Dashboard } from './Dashboard';
import { EventLog } from './EventLog';

const CONFIG_KEY = 'frankenvoice.pipeline';

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
  const [running, setRunning] = useState<'live' | 'replay' | null>(null);
  const [nodeStates, setNodeStates] = useState<Record<string, Record<string, Json>>>({});
  /** Node type behind each entry of nodeStates (the run they came from may predate a config change). */
  const [stateTypes, setStateTypes] = useState<Record<string, string>>({});
  const [sessions, setSessions] = useState<Session[]>([]);
  const [current, setCurrent] = useState<Session | null>(null);
  const [health, setHealth] = useState<Record<string, boolean> | null>(null);
  const [replayOpts, setReplayOpts] = useState({ audible: true, realtime: true, useCurrentConfig: false, liveChannels: '' });
  const [notice, setNotice] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [serverSessions, setServerSessions] = useState<string[]>([]);
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

  const finishSession = async (rt: Runtime, source?: Trajectory, stoppedAt?: number) => {
    const s: Session = { traj: rt.trajectory, stoppedAt };
    if (source) s.comparison = compareTrajectories(source, rt.trajectory);
    setSessions((ss) => [s, ...ss]);
    setCurrent(s);
    setRunning(null);
    if (rt.trajectory.meta.mode === 'live') {
      try {
        const res = await fetch(`/api/sessions/${rt.trajectory.meta.id}`, { method: 'PUT', body: encodeBundle(rt.trajectory) as BodyInit });
        if (res.ok) {
          s.saved = ((await res.json()) as { path: string }).path;
          setServerSessions((l) => [rt.trajectory.meta.id, ...l]);
          setSessions((ss) => [...ss]);
        }
      } catch {
        /* proxy not running: the session stays in memory and can be exported */
      }
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
      return;
    }
    const rt = runtimeRef.current;
    if (!rt) return;
    rt.stop();
    void finishSession(rt);
  };

  const replay = async (source: Trajectory) => {
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
    setCurrent({ traj: rt.trajectory });
    setRunning('replay');
    if (replayOpts.realtime) await sim.runRealtime(1);
    else sim.runFast();
    const stoppedAt = rt.isRunning ? sim.now() : undefined;
    if (rt.isRunning) rt.stop();
    await finishSession(rt, source, stoppedAt);
  };

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
    const res = await fetch(`/api/sessions/${id}`);
    const traj = decodeBundle(new Uint8Array(await res.arrayBuffer()));
    const s = { traj, saved: id };
    setSessions((ss) => [s, ...ss.filter((x) => x.traj.meta.id !== id || x.traj.meta.mode !== traj.meta.mode)]);
    setCurrent(s);
  };

  const importFile = async (f: File) => {
    try {
      if (f.name.endsWith('.json')) {
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
  const stageOf = (lane: string | null): StageId | null =>
    lane === null ? null : activeStages(config).some((s) => s.id === lane) ? (lane as StageId) : 'tts';

  return (
    <div className="app">
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
        <button onClick={() => confirm('Reset every stage to its default settings?') && setConfig(defaultConfig())}>Reset config</button>
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
      </header>
      {showSettings && <SettingsPanel onClose={() => setShowSettings(false)} onSaved={refreshProxy} />}

      <section className="diagram">
        <PipelineView config={config} states={nodeStates} stateTypes={stateTypes} running={!!running} selected={selected} onSelect={setSelected} />
      </section>

      <section className="dash">
        <Dashboard
          trajectory={current?.traj}
          running={!!running}
          now={() => schedRef.current?.now() ?? 0}
          selected={selected}
          onSelect={(lane) => setSelected(stageOf(lane))}
        />
      </section>

      <aside className="inspector">
        {selected ? (
          <StagePanel config={config} stage={selected} state={stateTypes[selected] === stageType(config, selected) ? nodeStates[selected] : undefined} disabled={!!running} onChange={setConfig} />
        ) : (
          <p className="muted">Click a stage in the diagram or a lane in the dashboard to configure it. The pipeline's wiring is defined in code (src/pipeline.ts).</p>
        )}

        <h3>Sessions</h3>
        <div className="replay-opts">
          <label>
            <input type="checkbox" checked={replayOpts.realtime} onChange={(e) => setReplayOpts({ ...replayOpts, realtime: e.target.checked })} /> real time
          </label>
          <label>
            <input type="checkbox" checked={replayOpts.audible} onChange={(e) => setReplayOpts({ ...replayOpts, audible: e.target.checked })} /> audible
          </label>
          <label>
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
            <li key={`${s.traj.meta.id}-${i}`} className={current === s ? 'current' : ''}>
              <button className="link" onClick={() => setCurrent(s)}>
                {s.traj.meta.mode === 'replay' ? '↻ ' : '● '}
                {s.traj.meta.id}
              </button>
              <small>
                {(s.traj.duration / 1000).toFixed(1)} s · {s.traj.events.length} events
                {s.comparison && ` · ${describeComparison(s)}`}
                {s.saved && ' · saved'}
              </small>
              <span>
                <button disabled={!!running} onClick={() => replay(s.traj)}>
                  Replay
                </button>
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
                <button className="link" onClick={() => loadServerSession(id)}>
                  ⤓ {id}
                </button>
                <small>on disk</small>
                <span>
                  <button className="danger-link" onClick={() => deleteSession(id)}>
                    Delete
                  </button>
                </span>
              </li>
            ))}
        </ul>
      </aside>

      <section className="log">
        <EventLog trajectory={current?.traj} live={!!running} />
      </section>
    </div>
  );
}
