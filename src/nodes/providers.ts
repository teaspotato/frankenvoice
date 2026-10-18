import type { ParamDef } from '../core/node';
import type { JsonObject } from '../core/types';

/**
 * Provider-specific params are stored on the node as "<provider>.<key>" so two
 * providers can both have e.g. a "model" without clashing, and switching
 * provider keeps each provider's settings.
 */
export function providerParams(adapters: { provider: string; params: ParamDef[] }[]): ParamDef[] {
  return adapters.flatMap((a) => a.params.map((p) => ({ ...p, key: `${a.provider}.${p.key}`, when: { provider: [a.provider] } })));
}

/** The params one provider sees, with its prefix stripped. */
export function paramsFor(provider: string, all: JsonObject): JsonObject {
  const out: JsonObject = {};
  const prefix = `${provider}.`;
  for (const [k, v] of Object.entries(all)) if (k.startsWith(prefix)) out[k.slice(prefix.length)] = v;
  return out;
}
