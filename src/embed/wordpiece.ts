/**
 * BERT uncased WordPiece tokenizer (what all-MiniLM-L6-v2 and similar sentence
 * encoders use): lowercase, strip accents, split on whitespace and
 * punctuation, then greedy longest-match subwords from the vocab.
 */
export class WordPiece {
  private ids = new Map<string, number>();
  readonly cls: number;
  readonly sep: number;
  private unk: number;

  constructor(vocab: string) {
    vocab.split('\n').forEach((w, i) => w && this.ids.set(w.replace(/\r$/, ''), i));
    this.cls = this.ids.get('[CLS]')!;
    this.sep = this.ids.get('[SEP]')!;
    this.unk = this.ids.get('[UNK]')!;
  }

  /** Token ids with [CLS] and [SEP], at most `max` in all. */
  encode(text: string, max = 128): number[] {
    const out = [this.cls];
    const clean = text
      .toLowerCase()
      .normalize('NFD')
      .replace(/\p{Mn}/gu, '');
    const words = clean.match(/[\p{L}\p{N}]+|[^\s\p{L}\p{N}]/gu) ?? [];
    for (const w of words) {
      out.push(...this.piece(w));
      if (out.length >= max - 1) break;
    }
    out.length = Math.min(out.length, max - 1);
    out.push(this.sep);
    return out;
  }

  private piece(word: string): number[] {
    if (word.length > 100) return [this.unk];
    const out: number[] = [];
    let start = 0;
    while (start < word.length) {
      let end = word.length;
      let id: number | undefined;
      while (start < end) {
        id = this.ids.get((start ? '##' : '') + word.slice(start, end));
        if (id !== undefined) break;
        end--;
      }
      if (id === undefined) return [this.unk];
      out.push(id);
      start = end;
    }
    return out;
  }
}

/** Mean of the token vectors (attention mask all ones), L2-normalized. */
export function meanPool(hidden: Float32Array, tokens: number, dim: number): number[] {
  const v = new Array<number>(dim).fill(0);
  for (let t = 0; t < tokens; t++) for (let d = 0; d < dim; d++) v[d] += hidden[t * dim + d];
  const norm = Math.hypot(...v) || 1;
  return v.map((x) => x / norm);
}
