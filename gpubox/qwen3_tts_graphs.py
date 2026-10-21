"""CUDA-graph step for the Qwen3-TTS talker (used by qwen3_tts_worker.py).

Through HF, one 80 ms frame cost ~186 ms on the L4: the code predictor runs as a 15-token
`generate()` per frame, and every layer call is Python and kernel-launch bound. Here both models
run their decoder layers directly over a preallocated KV cache, and each frame is two graph
replays:
  1. code predictor: [talker hidden, codebook 0] -> codebooks 1..15, sampled in the graph
     (top-k, temperature, Gumbel-max), plus the summed codec embedding of the frame;
  2. talker: (that sum + the next text embedding) at position `pos` -> codebook-0 logits and hidden.
Codebook 0 is sampled outside (repetition penalty over the reply, eos rules), as generate() does.
"""
import torch


class KV:
    """Preallocated KV cache with the `update()` the HF attention modules call."""

    def __init__(self, n_layers, n_kv, max_len, head_dim, dtype, device):
        self.k = [torch.zeros(1, n_kv, max_len, head_dim, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.zeros(1, n_kv, max_len, head_dim, dtype=dtype, device=device) for _ in range(n_layers)]
        self.max_len = max_len

    def update(self, k, v, layer_idx, cache_kwargs=None):
        cp = cache_kwargs["cache_position"]
        self.k[layer_idx].index_copy_(2, cp, k)
        self.v[layer_idx].index_copy_(2, cp, v)
        return self.k[layer_idx], self.v[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return 0


def _mask(cp, max_len, dtype):
    cols = torch.arange(max_len, device=cp.device)
    m = torch.zeros(1, 1, cp.shape[0], max_len, dtype=dtype, device=cp.device)
    return m.masked_fill(cols[None, None, None, :] > cp[None, None, :, None], torch.finfo(dtype).min)


def _gumbel_topk(logits, k, temp):
    v, i = torch.topk(logits.float(), k, dim=-1)
    g = -torch.log(-torch.log(torch.rand_like(v).clamp_(1e-10, 1.0)))
    j = torch.argmax(v / temp + g, dim=-1, keepdim=True)
    return torch.gather(i, -1, j)


class FastTalker:
    def __init__(self, talker, max_len: int = 2048, sub_temp: float = 0.9, sub_top_k: int = 50):
        self.t = talker
        self.cp = talker.code_predictor
        p = next(talker.parameters())
        self.dev, self.dtype = p.device, p.dtype
        tc, cc = talker.config, self.cp.config
        hd = getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads)
        chd = getattr(cc, "head_dim", cc.hidden_size // cc.num_attention_heads)
        self.max_len = max_len
        self.kv = KV(tc.num_hidden_layers, tc.num_key_value_heads, max_len, hd, self.dtype, self.dev)
        self.groups = tc.num_code_groups
        self.ckv = KV(cc.num_hidden_layers, cc.num_key_value_heads, self.groups, chd, self.dtype, self.dev)
        self.sub_temp, self.sub_top_k = sub_temp, sub_top_k
        H = tc.hidden_size
        # graph inputs / outputs
        self.g_hidden = torch.zeros(1, 1, H, dtype=self.dtype, device=self.dev)    # talker hidden (past_hidden)
        self.g_c0 = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
        self.g_codes = torch.zeros(1, self.groups, dtype=torch.long, device=self.dev)
        self.g_sum = torch.zeros(1, 1, H, dtype=self.dtype, device=self.dev)
        self.g_emb = torch.zeros(1, 1, H, dtype=self.dtype, device=self.dev)       # talker input
        self.g_pos = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.g_logits = torch.zeros(1, tc.vocab_size, dtype=torch.float32, device=self.dev)
        self.g_out_hidden = torch.zeros(1, 1, H, dtype=self.dtype, device=self.dev)
        self.cp_graph = self.talker_graph = None
        self.pos = 0

    # ---- eager building blocks (also what the graphs capture) ----------------------------------------
    def _talker_layers(self, x, cp):
        m = self.t.model
        pos3 = cp.view(1, 1, -1).expand(3, 1, -1)
        pe = m.rotary_emb(x, pos3)
        mask = _mask(cp, self.max_len, x.dtype)
        h = x
        for layer in m.layers:
            h = layer(h, attention_mask=mask, position_ids=cp.view(1, -1), past_key_values=self.kv, use_cache=True,
                      cache_position=cp, position_embeddings=pe)[0]
        h = m.norm(h)
        return self.t.codec_head(h[:, -1:]), h[:, -1:]

    def _cp_layers(self, x, cp):
        m = self.cp.model
        pe = m.rotary_emb(x, cp.view(1, -1))
        mask = _mask(cp, self.groups, x.dtype)
        h = x
        for layer in m.layers:
            h = layer(h, attention_mask=mask, position_ids=cp.view(1, -1), past_key_values=self.ckv, use_cache=True,
                      cache_position=cp, position_embeddings=pe)[0]
        return m.norm(h)

    def _cp_run(self):
        """past hidden + codebook 0 -> codebooks 1..15 and the frame's summed codec embedding."""
        cp, E = self.cp, self.cp.model.codec_embedding
        first = self.t.model.codec_embedding(self.g_c0)                       # 1 x 1 x H
        x = cp.small_to_mtp_projection(torch.cat([self.g_hidden, first], dim=1))
        pos = torch.arange(2, device=self.dev)
        h = self._cp_layers(x, pos)
        codes = [self.g_c0[:, 0]]
        total = first
        tok = _gumbel_topk(cp.lm_head[0](h[:, -1]), self.sub_top_k, self.sub_temp)   # 1 x 1
        for i in range(1, self.groups - 1):
            codes.append(tok[:, 0])
            e = E[i - 1](tok)
            total = total + e
            h = self._cp_layers(cp.small_to_mtp_projection(e), pos[-1:] + i)
            tok = _gumbel_topk(cp.lm_head[i](h[:, -1]), self.sub_top_k, self.sub_temp)
        codes.append(tok[:, 0])
        total = total + E[self.groups - 2](tok)
        self.g_codes.copy_(torch.stack(codes, dim=1))
        self.g_sum.copy_(total)

    def _talker_run(self):
        logits, h = self._talker_layers(self.g_emb, self.g_pos)
        self.g_logits.copy_(logits[:, 0].float())
        self.g_out_hidden.copy_(h)

    def capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        self.g_pos.fill_(self.max_len - 1)  # a harmless cache slot for the warm-up runs
        with torch.cuda.stream(s):
            for _ in range(3):
                self._cp_run()
                self._talker_run()
        torch.cuda.current_stream().wait_stream(s)
        self.cp_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cp_graph):
            self._cp_run()
        self.talker_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.talker_graph):
            self._talker_run()

    # ---- the API the worker uses ------------------------------------------------------------------
    def prefill(self, x):
        """Run the prompt; returns (codebook-0 logits, hidden) of its last position."""
        L = x.shape[1]
        if L >= self.max_len:
            raise ValueError("prompt too long")
        logits, h = self._talker_layers(x, torch.arange(L, device=self.dev))
        self.pos = L
        self.g_hidden.copy_(h)
        return logits[0, 0].float(), h

    def frame(self, c0: int, text_embed):
        """One frame: codebooks for c0 (with the hidden from the last call), then the talker reads the frame
        plus `text_embed`. Returns (16 codes, next codebook-0 logits)."""
        if self.pos >= self.max_len - 1:
            raise ValueError("reply too long")
        self.g_c0.fill_(c0)
        self.cp_graph.replay()
        codes = self.g_codes[0].clone()
        self.g_emb.copy_(self.g_sum + text_embed)
        self.g_pos.fill_(self.pos)
        self.talker_graph.replay()
        self.pos += 1
        self.g_hidden.copy_(self.g_out_hidden)
        return codes, self.g_logits[0]
