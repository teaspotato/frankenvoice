import { Handle, Position, type NodeProps, type Node } from '@xyflow/react';
import type { NodeDef } from '../core/node';
import type { Json } from '../core/types';

export type FvNodeData = {
  def: NodeDef;
  label: string;
  state?: Record<string, Json>;
  active: boolean;
};

export type FvFlowNode = Node<FvNodeData, 'fv'>;

const ROW = 20;
const HEADER = 30;

function summarize(v: Json): string {
  if (v === null) return '–';
  if (typeof v === 'object') return Array.isArray(v) ? `[${v.length}]` : '{…}';
  const s = String(v);
  return s.length > 28 ? `${s.slice(0, 27)}…` : s;
}

export function FvNode({ data, selected }: NodeProps<FvFlowNode>) {
  const { def, label, state } = data;
  const rows = Math.max(def.inputs.length, def.outputs.length);
  const entries = Object.entries(state ?? {}).filter(([, v]) => typeof v !== 'object' || v === null);
  return (
    <div className={`fv-node cat-${def.category}${selected ? ' selected' : ''}${data.active ? ' active' : ''}`}>
      <div className="fv-node-header">
        <span className="fv-node-title">{label}</span>
        <span className="fv-node-type">{def.type}</span>
      </div>
      <div className="fv-node-ports" style={{ height: rows * ROW }}>
        {def.inputs.map((p, i) => (
          <div key={`in-${p.name}`} className="fv-port in" style={{ top: i * ROW }} title={p.type}>
            {p.name}
          </div>
        ))}
        {def.outputs.map((p, i) => (
          <div key={`out-${p.name}`} className="fv-port out" style={{ top: i * ROW }} title={p.type}>
            {p.name}
          </div>
        ))}
      </div>
      {def.inputs.map((p, i) => (
        <Handle key={`in-${p.name}`} type="target" position={Position.Left} id={p.name} style={{ top: HEADER + i * ROW + ROW / 2 }} className={`t-${p.type}`} />
      ))}
      {def.outputs.map((p, i) => (
        <Handle key={`out-${p.name}`} type="source" position={Position.Right} id={p.name} style={{ top: HEADER + i * ROW + ROW / 2 }} className={`t-${p.type}`} />
      ))}
      {entries.length > 0 && (
        <div className="fv-node-state">
          {entries.map(([k, v]) => (
            <div key={k}>
              <span className="k">{k}</span> <span className="v">{summarize(v)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
