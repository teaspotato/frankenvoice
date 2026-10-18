import { useMemo } from 'react';
import { Background, ReactFlow, type Edge } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { registry } from '../nodes';
import type { Json } from '../core/types';
import { buildGraph, STAGES, type PipelineConfig, type StageId } from '../pipeline';
import { FvNode, type FvFlowNode } from './FvNode';

const nodeTypes = { fv: FvNode };

/** Read-only diagram of the fixed pipeline with each stage's live state. */
export function PipelineView({
  config,
  states,
  stateTypes,
  running,
  selected,
  onSelect,
}: {
  config: PipelineConfig;
  /** Latest state per stage; only shown for stages that still run the same node type. */
  states: Record<string, Record<string, Json>>;
  stateTypes: Record<string, string>;
  running: boolean;
  selected: StageId | null;
  onSelect(id: StageId | null): void;
}) {
  const graph = useMemo(() => buildGraph(config), [config]);
  const nodes: FvFlowNode[] = graph.nodes.map((n) => {
    const stage = STAGES.find((s) => s.id === n.id)!;
    const variant = config[stage.id].variant;
    const provider = typeof n.params.provider === 'string' ? n.params.provider : undefined;
    return {
      id: n.id,
      type: 'fv' as const,
      position: n.ui ?? { x: 0, y: 0 },
      selected: n.id === selected,
      draggable: false,
      connectable: false,
      data: {
        def: registry[n.type],
        label: stage.label,
        subtitle: provider ?? (Object.keys(stage.variants).length > 1 ? variant : undefined),
        state: stateTypes[n.id] === n.type ? states[n.id] : undefined,
        active: running,
      },
    };
  });
  const edges: Edge[] = graph.edges.map((e) => ({
    id: `${e.from.node}.${e.from.port}->${e.to.node}.${e.to.port}`,
    source: e.from.node,
    sourceHandle: e.from.port,
    target: e.to.node,
    targetHandle: e.to.port,
    animated: running,
    focusable: false,
  }));
  return (
    <ReactFlow
      key={graph.nodes.map((n) => n.type).join()}
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      nodesDraggable={false}
      nodesConnectable={false}
      edgesFocusable={false}
      elementsSelectable={false}
      deleteKeyCode={null}
      onNodeClick={(_e, n) => onSelect(n.id as StageId)}
      onPaneClick={() => onSelect(null)}
      fitView
      onInit={(rf) => setTimeout(() => rf.fitView({ padding: 0.08 }), 50)}
      fitViewOptions={{ padding: 0.08 }}
      minZoom={0.3}
    >
      <Background />
    </ReactFlow>
  );
}
