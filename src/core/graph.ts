import type { Json } from './types';

export const GRAPH_FORMAT = 'frankenvoice-graph';
export const GRAPH_FORMAT_VERSION = 15;

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

/**
 * Throws unless `g` is a graph of this build's format. Older sessions and configs are not
 * migrated: the pipeline changes too fast to keep them running as they did.
 */
export function checkGraph(g: GraphSpec): GraphSpec {
  if (g?.format !== GRAPH_FORMAT) throw new Error('not a frankenvoice pipeline');
  if (g.formatVersion !== GRAPH_FORMAT_VERSION)
    throw new Error(`recorded with pipeline format ${g.formatVersion}; this version only loads format ${GRAPH_FORMAT_VERSION} (older sessions are not migrated)`);
  return g;
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
