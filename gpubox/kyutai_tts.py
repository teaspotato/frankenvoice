"""Streaming Kyutai TTS worker (kyutai/tts-1.6b-en_fr via the `moshi` package).

Runs in its own venv (moshi's deps conflict with qwen-asr) and listens on localhost only.
The gateway (server.py) authenticates clients and relays `/v1/tts` WebSockets here.
Protocol: see PROTOCOL.md ("TTS: WS /v1/tts").

Continuous batching: one worker thread steps every active reply as one batch of up to
KYUTAI_MAX_BATCH rows (default 1, for one user at a time; 4 measured fine, see README), one model
step (80 ms of audio) per row per batch step.
Replies join and leave at step boundaries: a new one takes a free row (its rows of the model,
codec and conditioning state are reset), a finished or cancelled one frees it. The CUDA graphs
are captured once at the full batch size, so a step costs the same with 1 or 4 rows busy; rows
that are idle or waiting for text are masked out (moshi's exec_mask), so their state stands still.
Background jobs (`?priority=background`, canned lines rendered ahead) take a row only when one is
free and no reply is waiting for it; a reply that finds no free row preempts one at the next step,
and the line is rendered again from the start afterwards. A background job's output is held until
it is done, so a restart is invisible.

Adapted from the streaming loop in delayed-streams-modeling/scripts/tts_pytorch_streaming.py
(via open-tts/scripts/kyutai.py): words are fed only when the model's lookahead needs them,
and a row is not stepped while it could ask for a word that has not arrived yet.
"""
import argparse
import asyncio
import collections
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from moshi.models.loaders import CheckpointInfo
from moshi.models.lm import LMGen
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, Entry, TTSModel
from moshi.modules.transformer import StreamingMultiheadAttention

log = logging.getLogger("kyutai")

DEFAULT_VOICE = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
SR = 24000
FRAME = 1920  # samples per Mimi frame (12.5 Hz)
# Audio time of a word = (its text step + delay_steps + max_delay + 1 - step of first pcm frame) * 0.08 s
# + WORD_OFFSET_S. Calibrated with Qwen3-ForcedAligner-0.6B on 6 renders (124 words): aligner start - our start
# had median -0.08 s, mean -0.04 s (the aligner's resolution is 80 ms), so we shift by -0.04 s.
WORD_OFFSET_S = float(os.environ.get("KYUTAI_WORD_OFFSET_S", "-0.04"))
IDLE_CLOSE_S = float(os.environ.get("KYUTAI_IDLE_CLOSE_S", "30"))
# 1 by default: the box serves one user at a time (the app queues the others), and a step at batch 1 is ~10% faster;
# a reply still preempts a background (canned) render
MAX_BATCH = int(os.environ.get("KYUTAI_MAX_BATCH", "1"))


@dataclass
class Job:
    voice: str
    cfg: float
    loop: asyncio.AbstractEventLoop
    out: asyncio.Queue
    events: "queue.Queue" = field(default_factory=queue.Queue)
    cancelled: bool = False
    t_first_speak: float | None = None
    background: bool = False
    history: list = field(default_factory=list)   # events consumed, to run a preempted job again
    replay: list = field(default_factory=list)
    held: list = field(default_factory=list)      # a background job's output, sent when it is done
    preempted: int = 0
    finished: bool = False


def _send(job: Job, item):
    if job.background:
        job.held.append(item)
    elif not job.cancelled and job.out is not None:
        job.loop.call_soon_threadsafe(job.out.put_nowait, item)


class Run:
    """One job's generation state: text taken so far, its row's step count, what has been sent."""

    def __init__(self, eng: "Engine", job: Job):
        self.eng = eng
        self.job = job
        self.frame_s = 1.0 / eng.m.mimi.frame_rate
        self.reset()

    def reset(self):
        self.buf = ""            # partial word not yet terminated by whitespace
        self.pending: list[Entry] = []
        self.words: list[str] = []   # every word queued so far (== transcript order)
        self.marks: list[int] = []   # word count at each Flush
        self.n_flushed = 0
        self.closed = False
        self.frames_sent = 0
        self.first_pcm_offset = None
        self.words_sent = 0
        self.reported = (0, 0)
        self.t_gen = 0.0
        self.steps: list[float] = []        # seconds per batch step this row took part in
        self.audio_steps: list[float] = []  # the steps that produced audio
        self.batch_sizes: list[int] = []    # rows stepped together, per step
        self.t_first_audio = None
        self.state = None        # the TTS state machine (moshi.models.tts.State) once it has a row
        self.offset = 0          # this row's model step
        self.end_at = None
        self.last_ev = time.perf_counter()

    # ---- text in -------------------------------------------------------------------------
    def take_words(self, text: str, final: bool):
        buf = self.buf + text
        buf = buf.replace("’", "'").replace(":", " ").replace("(", "").replace(")", "")
        if final:
            parts, self.buf = buf.split(), ""
        else:
            cut = max(buf.rfind(" "), buf.rfind("\n"), buf.rfind("\t"))
            if cut < 0:
                self.buf = buf
                return
            parts, self.buf = buf[:cut].split(), buf[cut + 1:]
        ents = self.eng.make_entries(parts, first=not self.words)
        self.pending.extend(ents)
        self.words.extend(e.text for e in ents)

    def handle(self, ev):
        job = self.job
        job.history.append(ev)
        self.last_ev = time.perf_counter()
        kind, val = ev
        if kind == "text":
            if job.t_first_speak is None:
                job.t_first_speak = time.perf_counter()
            self.take_words(val, False)
        elif kind == "flush":
            self.take_words("", True)
            self.marks.append(len(self.words))
        elif kind == "close":
            self.take_words("", True)
            self.closed = True

    def pull(self):
        """Take every event that has arrived; a reply that stops sending text without Close
        would hold its row forever, so IDLE_CLOSE_S of silence counts as Close."""
        job = self.job
        while True:
            if job.replay:
                ev = job.replay.pop(0)
            else:
                try:
                    ev = job.events.get_nowait()
                except queue.Empty:
                    break
            self.handle(ev)
        if not self.closed and time.perf_counter() - self.last_ev > IDLE_CLOSE_S:
            log.warning("no text for %.0f s: treating as Close", IDLE_CLOSE_S)
            self.take_words("", True)
            self.closed = True

    def ready(self) -> bool:
        """Feed entries up to the lookahead; can this row take a model step now?"""
        st, ssa = self.state, self.eng.ssa
        while self.pending and len(st.entries) <= ssa:
            st.entries.append(self.pending.pop(0))
        if not self.closed:
            # Could the model ask for a word at this step? Then it must see the full lookahead.
            can_ask = not st.queued and st.forced_padding <= 0
            if can_ask and len(st.entries) <= ssa:
                return False
        return True

    def done(self) -> bool:
        st, m = self.state, self.eng.m
        if st.end_step is not None:
            if self.end_at is None:
                self.end_at = st.end_step + m.delay_steps + self.eng.max_delay + 8
            if self.offset >= self.end_at:
                return True
        return self.offset > 30000

    # ---- out -----------------------------------------------------------------------------
    def word_frame(self, step: int) -> float:
        # pcm index (fractional, in frames) at which text consumed at `step` is heard
        return step + self.eng.m.delay_steps + self.eng.max_delay + 1 - self.first_pcm_offset + WORD_OFFSET_S / self.frame_s

    def word_end(self, i: int):
        tr = self.state.transcript
        if i + 1 < len(tr):
            return self.word_frame(tr[i + 1][1])
        if self.state.end_step is not None and i < len(tr):
            return self.word_frame(self.state.end_step)
        return None

    def emit_progress(self, final: bool):
        if self.first_pcm_offset is None and not final:
            return
        frame_s = self.frame_s
        tr = self.state.transcript if self.state else []
        batch = []
        while self.words_sent < len(tr):
            w, s = tr[self.words_sent]
            st = self.word_frame(s) if self.first_pcm_offset is not None else 0.0
            en = self.word_end(self.words_sent)
            if en is None and final:
                en = self.frames_sent
            if en is None or (self.frames_sent < st and not final):
                break
            en = min(en, self.frames_sent) if final else en
            batch.append({"word": w, "start": round(max(0.0, st) * frame_s, 3), "end": round(max(st, en) * frame_s, 3)})
            self.words_sent += 1
        if batch:
            _send(self.job, {"type": "Words", "words": batch})
        while self.n_flushed < len(self.marks):
            n = self.marks[self.n_flushed]
            if not final and n > 0:
                en = self.word_end(n - 1) if self.first_pcm_offset is not None else None
                if en is None or self.frames_sent < en:
                    break
            _send(self.job, {"type": "Flushed"})
            self.n_flushed += 1

    def after_step(self, dt: float, pcm, n_rows: int):
        self.offset += 1
        self.steps.append(dt)
        self.batch_sizes.append(n_rows)
        self.t_gen += dt
        if pcm is not None:
            self.audio_steps.append(dt)
            if self.first_pcm_offset is None:
                self.first_pcm_offset = self.offset
                self.t_first_audio = time.perf_counter()
            _send(self.job, (pcm * 32767).astype("<i2").tobytes())
            self.frames_sent += 1
        self.emit_progress(False)
        q = (len(self.state.transcript), len(self.words) - len(self.state.transcript))
        if q != self.reported:
            self.reported = q
            _send(self.job, {"type": "Consumed", "words": q[0], "queued": q[1]})

    def finish(self):
        job, frame_s = self.job, self.frame_s
        self.emit_progress(True)
        audio_s = self.frames_sent * frame_s
        head = self.audio_steps[:int(3 / frame_s)]  # the first 3 s of audio, while the reply's text is still being written
        srt = sorted(self.steps)
        stats = {"audio_s": round(audio_s, 3), "gen_s": round(self.t_gen, 3), "rtf": round(self.t_gen / audio_s, 3) if audio_s else None,
                 "rtf_3s": round(sum(head) / (len(head) * frame_s), 3) if head else None,
                 "step_ms_p50": round(srt[len(srt) // 2] * 1000, 1) if srt else None,
                 "step_ms_p90": round(srt[int(0.9 * (len(srt) - 1))] * 1000, 1) if srt else None,
                 "step_ms_max": round(srt[-1] * 1000, 1) if srt else None,
                 "batch_mean": round(sum(self.batch_sizes) / len(self.batch_sizes), 2) if self.batch_sizes else None,
                 "batch_max": max(self.batch_sizes) if self.batch_sizes else None,
                 "ttfa_ms": round((self.t_first_audio - job.t_first_speak) * 1000) if self.t_first_audio and job.t_first_speak else None,
                 "words": len(self.words)}
        if job.background:
            stats["background"] = True
            stats["preempted"] = job.preempted
        log.info("tts done %s", stats)
        _send(job, {"type": "Done", "stats": stats})
        _send(job, None)
        self.release()

    def fail(self, msg: str):
        _send(self.job, {"type": "Error", "message": msg})
        _send(self.job, None)
        self.release()

    def release(self):
        job = self.job
        job.finished = True
        if job.background:
            held, job.background = job.held, False
            for item in held:
                _send(job, item)


class Engine:
    def __init__(self, n_q: int, temp: float, voice: str, cfg: float, max_batch: int = MAX_BATCH, device: str = "cuda"):
        ci = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
        self.m = TTSModel.from_checkpoint_info(ci, n_q=n_q, temp=temp, device=device)
        m = self.m
        m.lm.dep_q = m.n_q
        self.B = B = max(1, max_batch)
        self.ssa = m.machine.second_stream_ahead
        self.max_delay = max(m.lm.delays)
        self.device = device
        log.info("loaded: n_q=%d delay_steps=%d max_delay=%d second_stream_ahead=%d valid_cfg=%s max_batch=%d",
                 m.n_q, m.delay_steps, self.max_delay, self.ssa, sorted(m.valid_cfg_conditionings), B)
        self.audio_delays = [m.lm.delays[q + m.lm.audio_offset] + m.delay_steps for q in range(m.lm.dep_q)]
        self.zero = m.machine.token_ids.zero
        self.missing = torch.full((B, m.lm.n_q - m.lm.dep_q, 1), self.zero, dtype=torch.long, device=device)
        self.no_depformer = torch.full((B, m.lm.dep_q, 1), self.zero, dtype=torch.long, device=device)
        self.conds: dict = {}
        # voices are keyed by their local file (the WS handler resolves a name before submitting)
        voice = str(m.get_voice_path(voice))
        ct = m.lm.condition_provider(m.lm.condition_provider.prepare([self._cond_attrs(voice, cfg)] * B))
        # support_out_of_sync: rows start at different times, so the per-row offsets decide which
        # outputs are not generated yet (-1), not the global step count
        self.lm_gen = LMGen(m.lm, temp=m.temp, temp_text=m.temp, cfg_coef=m.cfg_coef, condition_tensors=ct,
                            on_text_hook=self._on_text, on_audio_hook=self._on_audio,
                            cfg_is_masked_until=None, cfg_is_no_text=True, support_out_of_sync=True)
        self.lm_gen.streaming_forever(B)
        m.mimi.streaming_forever(B)
        self.row_cond: list[tuple[str, float] | None] = [(voice, cfg)] * B
        self.exec_mask: list[bool] | None = None
        self.mimi_mask: list[bool] | None = None
        self.slots: list[Run | None] = [None] * B
        self.stepping: list[int] = []
        self.fg: collections.deque = collections.deque()
        self.bg: collections.deque = collections.deque()
        self.lock = threading.Lock()
        self.wake = threading.Event()

    # ---- conditioning per row --------------------------------------------------------------
    def _cond_attrs(self, voice: str, cfg: float):
        return self.m.make_condition_attributes([self.m.get_voice_path(voice)], cfg_coef=cfg)

    def _set_row_cond(self, i: int, voice: str, cfg: float):
        key = (voice, cfg)
        if self.row_cond[i] == key:
            return
        m = self.m
        if key not in self.conds:
            ct = m.lm.condition_provider(m.lm.condition_provider.prepare([self._cond_attrs(voice, cfg)]))
            self.conds[key] = (m.lm.fuser.get_sum(ct), m.lm.fuser.get_cross(ct))
        s, c = self.conds[key]
        st = self.lm_gen._streaming_state
        if s is not None:
            st.condition_sum[i:i + 1].copy_(s.to(st.condition_sum.dtype))
        if c is not None:
            st.condition_cross[i:i + 1].copy_(c.to(st.condition_cross.dtype))  # raises if the voice's length differs
            # cross-attention keys/values are cached per streaming state (and baked into the CUDA graph's inputs)
            for mod in m.lm.modules():
                if isinstance(mod, StreamingMultiheadAttention) and mod.cross_attention and mod._streaming_state is not None \
                        and mod._streaming_state.k_cross is not None:
                    mod.update_streaming_cross_attention_src(st.condition_cross)
        self.row_cond[i] = key

    # ---- model plumbing -------------------------------------------------------------------
    def _on_text(self, t):
        toks = t.tolist()
        for i in self.stepping:
            r = self.slots[i]
            toks[i] = self.m.machine.process(r.offset, r.state, toks[i])[0]
        t[:] = torch.tensor(toks, dtype=torch.long, device=t.device)

    def _on_audio(self, a):
        for i in self.stepping:
            off = self.slots[i].offset
            qs = [q for q, d in enumerate(self.audio_delays) if off < d]
            if qs:
                a[i, qs] = self.zero

    def _step(self, rows: list[int]):
        """One batch step: 80 ms of audio for each row in `rows`. Returns {row: float32 pcm}."""
        B = self.B
        mask = [False] * B
        for i in rows:
            mask[i] = True
        if mask != self.exec_mask:
            self.lm_gen.set_exec_mask(torch.tensor(mask, device=self.device))
            self.exec_mask = mask
        self.stepping = rows
        rep = self.no_depformer if all(self.slots[i].offset < self.m.delay_steps for i in rows) else None
        try:
            f = self.lm_gen.step(self.missing, depformer_replace_tokens=rep)
        finally:
            self.stepping = []
        # -2: not generated yet (the row's first max_delay steps); -1: zeroed audio (the first delay_steps)
        valid = (f >= 0).all(dim=2).all(dim=1).tolist()
        out_rows = [i for i in rows if valid[i]]
        if not out_rows:
            return {}
        first = [i for i in out_rows if self.slots[i].first_pcm_offset is None]
        if first:
            reset = torch.zeros(B, dtype=torch.bool)
            reset[first] = True
            self.m.mimi.reset_streaming(reset)
            self.mimi_mask = None  # reset sets the rows' exec mask
        # only rows with a real frame advance the codec: one waiting for words must not decode junk
        mmask = [i in out_rows for i in range(B)]
        if mmask != self.mimi_mask:
            self.m.mimi.set_exec_mask(torch.tensor(mmask, device=self.device))
            self.mimi_mask = mmask
        pcm = self.m.mimi.decode(f[:, 1:, :].clamp(min=0)).float().cpu().numpy()
        return {i: np.clip(pcm[i, 0], -1, 1) for i in out_rows}

    # ---- text -> entries -----------------------------------------------------------------
    def make_entries(self, words: list[str], first: bool) -> list[Entry]:
        m = self.m
        out = []
        for w in words:
            toks = m.tokenizer.encode(w)
            if not toks:
                continue
            if first and m.multi_speaker:
                toks = [m.machine.token_ids.main] + toks
            first = False
            out.append(Entry(tokens=toks, text=w, padding=max(0, 1 + len(toks) - 1)))  # padding_between=1
        return out

    # ---- scheduling ---------------------------------------------------------------------------
    def submit(self, job: Job):
        r = Run(self, job)
        with self.lock:
            (self.bg if job.background else self.fg).append(r)
        self.wake.set()

    def active(self) -> int:
        return sum(1 for r in self.slots if r is not None)

    def _admit(self, i: int, r: Run):
        self._set_row_cond(i, r.job.voice, r.job.cfg)
        reset = torch.zeros(self.B, dtype=torch.bool)
        reset[i] = True
        self.lm_gen.reset_streaming(reset)
        self.exec_mask = None  # reset sets the row's exec mask
        r.state = self.m.machine.new_state([])
        r.offset = 0
        self.slots[i] = r

    def _try_admit(self, i: int, r: Run) -> bool:
        """Admit `r` to row `i`; on failure (a bad voice) fail its job instead of leaving it in no queue."""
        try:
            self._admit(i, r)
            return True
        except Exception as e:  # noqa: BLE001
            log.exception("tts: could not admit a job (voice %s)", r.job.voice)
            self.slots[i] = None
            self.row_cond[i] = None  # its conditioning may be half written: set it again on the next admit
            r.fail(f"bad voice: {e}")
            return False

    def _preempt(self, i: int):
        """Give a background job's row back; it starts over later. The caller holds the lock."""
        r = self.slots[i]
        self.slots[i] = None
        job = r.job
        job.preempted += 1
        job.replay = job.history + job.replay
        job.history = []
        job.held = []
        r.reset()
        log.info("background job preempted by a reply (%d)", job.preempted)
        self.bg.appendleft(r)

    def _schedule(self):
        """Drop cancelled jobs, take new events, give free rows to waiting jobs (replies first)."""
        for i, r in enumerate(self.slots):
            if r is not None and r.job.cancelled:
                log.info("tts cancelled after %d frames", r.frames_sent)
                self.slots[i] = None
        with self.lock:
            for q in (self.fg, self.bg):
                live = [r for r in q if not r.job.cancelled]
                q.clear()
                q.extend(live)
            fg, bg = list(self.fg), list(self.bg)
        for r in fg + bg:
            r.pull()
        # a job needs its first word (or Close) before it takes a row
        for r in fg + bg:
            if r.closed and not r.pending:  # closed with no words: nothing to say
                with self.lock:
                    (self.bg if r.job.background else self.fg).remove(r)
                r.finish()
        free = [i for i, r in enumerate(self.slots) if r is None]
        with self.lock:
            for r in [r for r in self.fg if r.pending]:
                if not free:
                    bgs = [i for i, s in enumerate(self.slots) if s is not None and s.job.background]
                    if not bgs:
                        break
                    self._preempt(bgs[-1])
                    free.append(bgs[-1])
                self.fg.remove(r)
                i = free.pop(0)
                if not self._try_admit(i, r):
                    free.insert(0, i)
            # a background job only takes a row no reply is waiting for
            spare = len(free) - len(self.fg)
            for r in [r for r in self.bg if r.pending]:
                if spare <= 0:
                    break
                self.bg.remove(r)
                i = free.pop(0)
                if not self._try_admit(i, r):
                    free.insert(0, i)
                    continue
                spare -= 1

    def _tick(self) -> bool:
        """One scheduling round and at most one batch step. False when nothing could step."""
        self.wake.clear()
        self._schedule()
        rows = []
        for i, r in enumerate(self.slots):
            if r is None:
                continue
            r.pull()
            if not r.ready():
                continue
            if r.done():
                self.slots[i] = None
                r.finish()
                continue
            rows.append(i)
        if not rows:
            return False
        t0 = time.perf_counter()
        try:
            pcm = self._step(rows)
        except Exception as e:  # noqa: BLE001
            log.exception("tts batch step failed")
            for i in rows:
                r, self.slots[i] = self.slots[i], None
                r.fail(f"tts failed: {e}")
            return True
        dt = time.perf_counter() - t0
        for i in rows:
            self.slots[i].after_step(dt, pcm.get(i), len(rows))
        return True

    def run_forever(self):
        while True:
            try:
                with torch.no_grad():
                    stepped = self._tick()
            except Exception:  # noqa: BLE001
                log.exception("tts scheduler failed")
                stepped = False
            if not stepped:
                self.wake.wait(0.05)

    def warmup(self, n: int):
        jobs = [Job(voice=self.row_cond[0][0], cfg=self.row_cond[0][1], loop=_NullLoop(), out=None) for _ in range(n)]
        for job in jobs:
            for ev in [("text", "Hello there, this is a short warm up sentence to capture the graphs."), ("close", None)]:
                job.events.put(ev)
            self.submit(job)
        t = time.perf_counter()
        with torch.no_grad():
            while not all(j.finished for j in jobs):
                self._tick()
        log.info("warmup x%d %.2fs", n, time.perf_counter() - t)


class _NullLoop:
    def call_soon_threadsafe(self, *a):
        pass


def build_app(engine: Engine, default_voice: str, default_cfg: float) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True, "tts": DEFAULT_DSM_TTS_REPO, "n_q": engine.m.n_q, "max_batch": engine.B, "active": engine.active()}

    @app.get("/config")
    def config():
        # everything besides the text that decides what a render sounds like (canned-line cache keys)
        return {"model": DEFAULT_DSM_TTS_REPO, "n_q": engine.m.n_q, "temp": engine.m.temp, "voice": default_voice,
                "cfg": default_cfg, "sample_rate": SR}

    @app.websocket("/v1/tts")
    async def tts(ws: WebSocket):
        await ws.accept()
        voice = ws.query_params.get("voice") or default_voice
        try:
            cfg = float(ws.query_params.get("cfg") or default_cfg)
        except ValueError:
            cfg = default_cfg
        valid = engine.m.valid_cfg_conditionings
        if valid and cfg not in valid:
            await ws.send_text(json.dumps({"type": "Error", "message": f"cfg must be one of {sorted(valid)}"}))
            await ws.close()
            return
        # resolve (maybe download) the voice here, not on the step thread, where it would stall every stream
        try:
            voice = str(await asyncio.to_thread(engine.m.get_voice_path, voice))
        except Exception as e:  # noqa: BLE001
            log.warning("tts: bad voice %r: %s", voice, e)
            await ws.send_text(json.dumps({"type": "Error", "message": f"bad voice: {voice}"}))
            await ws.close()
            return
        job = Job(voice=voice, cfg=cfg, loop=asyncio.get_running_loop(), out=asyncio.Queue(),
                  background=ws.query_params.get("priority") == "background")
        engine.submit(job)
        await ws.send_text(json.dumps({"type": "Ready", "sample_rate": SR}))

        async def pump_out():
            while True:
                item = await job.out.get()
                if item is None:
                    return
                if isinstance(item, bytes):
                    await ws.send_bytes(item)
                else:
                    await ws.send_text(json.dumps(item))

        async def pump_in():
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect()
                txt = msg.get("text")
                if txt is None:
                    continue
                try:
                    d = json.loads(txt)
                except ValueError:
                    continue
                t = d.get("type")
                if t == "Speak":
                    job.events.put(("text", str(d.get("text", ""))))
                elif t == "Flush":
                    job.events.put(("flush", None))
                elif t == "Close":
                    job.events.put(("close", None))
                else:
                    continue
                engine.wake.set()

        out_task = asyncio.create_task(pump_out())
        in_task = asyncio.create_task(pump_in())
        try:
            done, _ = await asyncio.wait({out_task, in_task}, return_when=asyncio.FIRST_COMPLETED)
            if out_task in done:
                await ws.close()
        except Exception:  # noqa: BLE001
            pass
        finally:
            if not out_task.done():
                job.cancelled = True
                engine.wake.set()
            for t in (out_task, in_task):
                t.cancel()

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8102)
    ap.add_argument("--n-q", type=int, default=int(os.environ.get("KYUTAI_NQ", "32")))
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--max-batch", type=int, default=MAX_BATCH)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    engine = Engine(n_q=a.n_q, temp=a.temp, voice=a.voice, cfg=a.cfg, max_batch=a.max_batch)
    engine.warmup(engine.B)
    engine.warmup(1)
    engine.warmup(engine.B)
    threading.Thread(target=engine.run_forever, daemon=True).start()
    uvicorn.run(build_app(engine, a.voice, a.cfg), host=a.host, port=a.port, log_level="warning", ws_max_size=2**24)


if __name__ == "__main__":
    main()
