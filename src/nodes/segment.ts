/**
 * Incremental sentence splitter for streaming LLM text: feed deltas, get back
 * complete segments as soon as a sentence boundary is seen.
 */
export class SentenceSplitter {
  private buf = '';
  constructor(private minChars = 1) {}

  push(text: string): string[] {
    this.buf += text;
    const out: string[] = [];
    const re = /[.!?。！？…]+["')\]]*(\s+|$)/g;
    let last = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(this.buf))) {
      // A boundary at the very end of the buffer might still be "3." of "3.5"; wait for more text.
      if (m.index + m[0].length === this.buf.length && !m[1]) break;
      const end = m.index + m[0].length;
      const seg = this.buf.slice(last, end).trim();
      if (seg.length >= this.minChars) {
        out.push(seg);
        last = end;
      }
    }
    this.buf = this.buf.slice(last);
    return out;
  }

  flush(): string | undefined {
    const seg = this.buf.trim();
    this.buf = '';
    return seg || undefined;
  }
}
