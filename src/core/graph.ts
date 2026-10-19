import type { Json } from './types';

export const GRAPH_FORMAT = 'frankenvoice-graph';
export const GRAPH_FORMAT_VERSION = 5;

export interface NodeSpec {
  id: string;
  type: string;
  params: Record<string, Json>;
  /** Editor position; not used by the runtime. */
  ui?: { x: number; y: number };
}

export interface EdgeSpec {
  from: { node: string; port: string };
  to: { node: string; port: string };
}

/** A pipeline as saved to disk and recorded into every trajectory. */
export interface GraphSpec {
  format: typeof GRAPH_FORMAT;
  formatVersion: number;
  name: string;
  nodes: NodeSpec[];
  edges: EdgeSpec[];
}

/** The graph minus editor-only fields: what determines runtime behavior, and what graphHash covers. */
export function runtimeView(g: GraphSpec): Omit<GraphSpec, 'name'> {
  return {
    format: g.format,
    formatVersion: g.formatVersion,
    nodes: g.nodes.map(({ id, type, params }) => ({ id, type, params })),
    edges: g.edges,
  };
}
