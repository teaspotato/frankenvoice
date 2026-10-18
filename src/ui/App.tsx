import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Background,
  Controls,
  ReactFlow,
  type Connection,
  type Edge,
  type EdgeChange,
  type NodeChange,
  applyNodeChanges,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { registry, defaultGraph } from '../nodes';
import type { GraphSpec } from '../core/graph';
import { GRAPH_FORMAT } from '../core/graph';
import { Runtime } from '../core/runtime';
import { LiveScheduler, type SimScheduler } from '../core/scheduler';
import { LiveWorld } from '../core/world';
import { prepareReplay } from '../core/replay';
import { decodeBundle, encodeBundle } from '../core/bundle';
import { compareTrajectories, type Comparison } from '../core/compare';
import type { Trajectory } from '../core/trajectory';
import type { Json } from '../core/types';
import { resolveParams } from '../core/node';
import { browserDrivers } from '../drivers';
import { FvNode, type FvFlowNode } from './FvNode';
import { EventLog } from './EventLog';

const GRAPH_KEY = 'frankenvoice.graph';
const nodeTypes = { fv: FvNode };

function loadGraph(): GraphSpec {
  try {
    const raw = localStorage.getItem(GRAPH_KEY);
    if (raw) {
      const g = JSON.parse(raw) as GraphSpec;
      if (g.format === GRAPH_FORMAT) return g;
    }
  } catch {
    /* storage unavailable */
  }
  return defaultGraph();
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
}

export function App() {
  const [graph, setGraph] = useState<GraphSpec>(loadGraph);
  const [selected, setSelected] = useState<string | null>(null);
  const [runtime, setRuntime] = useState<Runtime | null>(null);
  const [running, setRunning] = useState<'live' | 'replay' | null>(null);
  const [nodeStates, setNodeStates] = useState<Record<string, Record<string, Json>>>({});
  const [sessions, setSessions] = useState<Session[]>([]);
  const [current, setCurrent] = useState<Session | null>(null);
  const [health, setHealth] = useState<Record<string, boolean> | null>(null);
  const [replayOpts, setReplayOpts] = useState({ audible: true, realtime: true, useCurrentGraph: false, liveChannels: '' });
  const [notice, setNotice] = useState('');
  const simRef = useRef<SimScheduler | null>(null);
  const [flowNodes, setFlowNodes] = useState<FvFlowNode[]>([]);
  const [serverSessions, setServerSessions] = useState<string[]>([]);

  useEffect(() => {
    try {
      localStorage.setItem(GRAPH_KEY, JSON.stringify(graph));
    } catch {
      /* storage unavailable */
    }
  }, [graph]);

  useEffect(() => {
    fetch('/api/health')
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));
    fetch('/api/sessions')
      .then((r) => (r.ok ? r.json() : []))
      .then((list: { id: string }[]) => setServerSessions(list.map((s) => s.id)))
      .catch(() => {});
  }, []);

  // ---- graph <-> React Flow -------------------------------------------------

  useEffect(() => {
    setFlowNodes((prev) => {
      const prevById = new Map(prev.map((n) => [n.id, n]));
      return graph.nodes.map((n, i) => ({
        id: n.id,
        type: 'fv' as const,
        position: n.ui ?? prevById.get(n.id)?.position ?? { x: 80 + i * 40, y: 80 + i * 40 },
        selected: n.id === selected,
        data: { def: registry[n.type], label: n.id, state: nodeStates[n.id], active: !!running },
      }));
    });
  }, [graph, nodeStates, selected, running]);

  const flowEdges: Edge[] = useMemo(
    () =>
      graph.edges.map((e) => ({
        id: `${e.from.node}.${e.from.port}->${e.to.node}.${e.to.port}`,
        source: e.from.node,
        sourceHandle: e.from.port,
        target: e.to.node,
        targetHandle: e.to.port,
        animated: !!running,
      })),
    [graph.edges, running],
  );

  const onNodesChange = useCallback((changes: NodeChange<FvFlowNode>[]) => {
    setFlowNodes((ns) => applyNodeChanges(changes, ns));
    const moved = changes.flatMap((c) => (c.type === 'position' && c.position && !c.dragging ? [{ id: c.id, position: c.position }] : []));
    const removed = changes.flatMap((c) => (c.type === 'remove' ? [c.id] : []));
    if (moved.length || removed.length) {
      setGraph((g) => ({
        ...g,
        nodes: g.nodes
          .filter((n) => !removed.includes(n.id))
          .map((n) => {
            const m = moved.find((c) => c.id === n.id);
            return m ? { ...n, ui: { x: Math.round(m.position.x), y: Math.round(m.position.y) } } : n;
          }),
        edges: g.edges.filter((e) => !removed.includes(e.from.node) && !removed.includes(e.to.node)),
      }));
    }
  }, []);

  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    const removed = changes.filter((c) => c.type === 'remove').map((c) => c.id);
    if (!removed.length) return;
    setGraph((g) => ({ ...g, edges: g.edges.filter((e) => !removed.includes(`${e.from.node}.${e.from.port}->${e.to.node}.${e.to.port}`)) }));
  }, []);

  const portType = (node: string | null, port: string | null | undefined, dir: 'in' | 'out') => {
    const spec = graph.nodes.find((n) => n.id === node);
    const def = spec && registry[spec.type];
    return (dir === 'in' ? def?.inputs : def?.outputs)?.find((p) => p.name === port)?.type;
  };

  const isValidConnection = (c: Connection | Edge) => {
    const a = portType(c.source, c.sourceHandle, 'out');
    const b = portType(c.target, c.targetHandle, 'in');
    return !!a && !!b && (a === b || a === 'any' || b === 'any');
  };

  const onConnect = (c: Connection) => {
    if (!isValidConnection(c) || !c.sourceHandle || !c.targetHandle) return;
    setGraph((g) => ({ ...g, edges: [...g.edges, { from: { node: c.source, port: c.sourceHandle! }, to: { node: c.target, port: c.targetHandle! } }] }));
  };

  const addNode = (type: string) => {
    const base = type.split('.').pop()!;
    let id = base;
    for (let i = 2; graph.nodes.some((n) => n.id === id); i++) id = `${base}${i}`;
    setGraph((g) => ({ ...g, nodes: [...g.nodes, { id, type, params: {}, ui: { x: 200, y: 360 } }] }));
    setSelected(id);
  };

  const setParam = (id: string, key: string, value: Json) =>
    setGraph((g) => ({ ...g, nodes: g.nodes.map((n) => (n.id === id ? { ...n, params: { ...n.params, [key]: value } } : n)) }));

  // ---- running ----------------------------------------------------------------

  const attach = (rt: Runtime) => {
    setNodeStates({});
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

  const finishSession = async (rt: Runtime, source?: Trajectory) => {
    const s: Session = { traj: rt.trajectory };
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
      graph,
      registry,
      scheduler,
      world: new LiveWorld(browserDrivers, scheduler),
      mode: 'live',
      env: { userAgent: navigator.userAgent },
    });
    attach(rt);
    setRuntime(rt);
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
    if (!runtime) return;
    runtime.stop();
    void finishSession(runtime);
  };

  const replay = async (source: Trajectory) => {
    const liveChannels = replayOpts.liveChannels
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
    const { runtime: rt, sim } = prepareReplay(source, {
      registry,
      graph: replayOpts.useCurrentGraph ? graph : undefined,
      drivers: browserDrivers,
      liveChannels,
      monitorChannels: replayOpts.audible && replayOpts.realtime ? ['speaker'] : [],
    });
    simRef.current = sim;
    attach(rt);
    setRuntime(rt);
    setCurrent({ traj: rt.trajectory });
    setRunning('replay');
    if (replayOpts.realtime) await sim.runRealtime(1);
    else sim.runFast();
    if (rt.isRunning) rt.stop();
    await finishSession(rt, source);
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
        const g = JSON.parse(await f.text()) as GraphSpec;
        if (g.format !== GRAPH_FORMAT) throw new Error('not a frankenvoice graph');
        setGraph(g);
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

  const sel = graph.nodes.find((n) => n.id === selected);
  const selDef = sel && registry[sel.type];
  const usedChannels = new Set(graph.nodes.flatMap((n) => registry[n.type]?.channels?.map((c) => c.name) ?? []));
  const missingKeys = health ? (['anthropic', 'deepgram', 'cartesia'] as const).filter((k) => usedChannels.has(k) && !health[k]) : [];

  return (
    <div className="app">
      <header className="toolbar">
        <span className="brand">frankenvoice</span>
        <input className="graph-name" value={graph.name} onChange={(e) => setGraph({ ...graph, name: e.target.value })} />
        {!running ? (
          <button className="primary" onClick={runLive}>
            ● Run live
          </button>
        ) : (
          <button className="danger" onClick={stop}>
            ■ Stop {running}
          </button>
        )}
        <button onClick={() => download(`${graph.name || 'graph'}.json`, JSON.stringify(graph, null, 2), 'application/json')}>Save graph</button>
        <label className="button">
          Open…
          <input type="file" accept=".json,.fvs,.zip" hidden onChange={(e) => e.target.files?.[0] && importFile(e.target.files[0])} />
        </label>
        <button onClick={() => confirm('Replace the current graph with the default pipeline?') && setGraph(defaultGraph())}>Reset graph</button>
        <span className="spacer" />
        {health === null ? (
          <span className="warn">proxy not reachable: run `npm run dev`</span>
        ) : missingKeys.length ? (
          <span className="warn">missing API keys: {missingKeys.join(', ')}</span>
        ) : (
          <span className="ok">proxy ok</span>
        )}
      </header>

      <aside className="palette">
        <h3>Nodes</h3>
        {Object.values(registry).map((d) => (
          <button key={d.type} className={`palette-item cat-${d.category}`} onClick={() => addNode(d.type)} title={d.description}>
            {d.title}
            <small>{d.type}</small>
          </button>
        ))}
      </aside>

      <main className="canvas">
        <ReactFlow
          nodes={flowNodes}
          edges={flowEdges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          isValidConnection={isValidConnection}
          onNodeClick={(_e, n) => setSelected(n.id)}
          onPaneClick={() => setSelected(null)}
          deleteKeyCode={running ? null : ['Backspace', 'Delete']}
          nodesConnectable={!running}
          fitView
        >
          <Background />
          <Controls />
        </ReactFlow>
      </main>

      <aside className="inspector">
        {sel && selDef ? (
          <>
            <h3>
              {sel.id} <small>{selDef.type}</small>
            </h3>
            <p className="muted">{selDef.description}</p>
            {selDef.params.map((p) => {
              const v = resolveParams(selDef, sel.params)[p.key];
              const set = (x: Json) => setParam(sel.id, p.key, x);
              return (
                <label key={p.key} className="param">
                  <span>{p.label}</span>
                  {p.type === 'boolean' ? (
                    <input type="checkbox" checked={!!v} onChange={(e) => set(e.target.checked)} />
                  ) : p.type === 'select' ? (
                    <select value={String(v)} onChange={(e) => set(e.target.value)}>
                      {p.options!.map((o) => (
                        <option key={o}>{o}</option>
                      ))}
                    </select>
                  ) : p.type === 'text' ? (
                    <textarea rows={5} value={String(v)} onChange={(e) => set(e.target.value)} />
                  ) : (
                    <input
                      type={p.type === 'number' ? 'number' : 'text'}
                      value={String(v)}
                      onChange={(e) => set(p.type === 'number' ? Number(e.target.value) : e.target.value)}
                    />
                  )}
                </label>
              );
            })}
            {nodeStates[sel.id] && (
              <>
                <h4>State</h4>
                <pre className="state">{JSON.stringify(nodeStates[sel.id], null, 2)}</pre>
              </>
            )}
          </>
        ) : (
          <p className="muted">Select a node to edit its parameters. Drag from an output to an input of the same type to connect.</p>
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
              checked={replayOpts.useCurrentGraph}
              onChange={(e) => setReplayOpts({ ...replayOpts, useCurrentGraph: e.target.checked })}
            />{' '}
            replay with current graph
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
                {s.comparison && (s.comparison.identical ? ' · identical to source ✓' : ` · diverged at event ${s.comparison.matched}`)}
                {s.saved && ' · saved'}
              </small>
              <span>
                <button disabled={!!running} onClick={() => replay(s.traj)}>
                  Replay
                </button>
                <button onClick={() => download(`${s.traj.meta.id}.fvs`, encodeBundle(s.traj) as BlobPart, 'application/zip')}>Export</button>
                <button onClick={() => setGraph(s.traj.meta.graph)}>Load graph</button>
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

