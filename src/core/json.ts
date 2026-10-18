import type { Json } from './types';

/** JSON.stringify with sorted object keys, so equal values always serialize identically. */
export function stableStringify(v: unknown): string {
  return JSON.stringify(sortKeys(v));
}

function sortKeys(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(sortKeys);
  if (v && typeof v === 'object' && !ArrayBuffer.isView(v)) {
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(v as object).sort()) {
      const x = (v as Record<string, unknown>)[k];
      if (x !== undefined) out[k] = sortKeys(x);
    }
    return out;
  }
  return v;
}

/** cyrb53: small, fast, stable 53-bit string hash, returned as hex. */
export function hashString(str: string, seed = 0): string {
  let h1 = 0xdeadbeef ^ seed;
  let h2 = 0x41c6ce57 ^ seed;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(16).padStart(14, '0');
}

export function hashJson(v: unknown): string {
  return hashString(stableStringify(v));
}

export function jsonEqual(a: unknown, b: unknown): boolean {
  return stableStringify(a) === stableStringify(b);
}

/** Convert an arbitrary value into plain JSON (drops functions/undefined, typed arrays become a marker). */
export function toJson(v: unknown): Json {
  if (v === undefined) return null;
  return JSON.parse(
    JSON.stringify(v, (_k, x) => {
      if (ArrayBuffer.isView(x)) return { $bytes: (x as ArrayBufferView).byteLength };
      if (typeof x === 'bigint') return x.toString();
      return x;
    }),
  ) as Json;
}
