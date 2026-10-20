"""Streaming Kyutai TTS worker (kyutai/tts-1.6b-en_fr via the `moshi` package).

Runs in its own venv (moshi's deps conflict with qwen-asr) and listens on localhost only.
The gateway (server.py) authenticates clients and relays `/v1/tts` WebSockets here.
Protocol: see PROTOCOL.md ("TTS: WS /v1/tts").

Generation runs on one worker thread, one reply at a time. A cancelled reply (socket closed)
stops at its next model step, so the next connection never waits behind it. Background jobs
(`?priority=background`, canned lines rendered ahead) run only when no reply is queued: a reply
that arrives preempts one at its next model step, and the line is rendered again from the start
afterwards. A background job's output is held until it is done, so a restart is invisible.

Adapted from the streaming loop in delayed-streams-modeling/scripts/tts_pytorch_streaming.py
(via open-tts/scripts/kyutai.py): words are fed only when the model's lookahead needs them,
and the model is not stepped while it could ask for a word that has not arrived yet.
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

log = logging.getLogger("kyutai")

DEFAULT_VOICE = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
SR = 24000
FRAME = 1920  # samples per Mimi frame (12.5 Hz)
# Audio time of a word = (its text step + delay_steps + max_delay + 1 - step of first pcm frame) * 0.08 s
# + WORD_OFFSET_S. Calibrated with Qwen3-ForcedAligner-0.6B on 6 renders (124 words): aligner start - our start
# had median -0.08 s, mean -0.04 s (the aligner's resolution is 80 ms), so we shift by -0.04 s.
WORD_OFFSET_S = float(os.environ.get("KYUTAI_WORD_OFFSET_S", "-0.04"))
IDLE_CLOSE_S = float(os.environ.get("KYUTAI_IDLE_CLOSE_S", "30"))


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


class Preempted(Exception):
    pass


class Engine:
    def __init__(self, n_q: int, temp: float, device: str = "cuda"):
        ci = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
        self.m = TTSModel.from_checkpoint_info(ci, n_q=n_q, temp=temp, device=device)
        m = self.m
        m.lm.dep_q = m.n_q
        self.ssa = m.machine.second_stream_ahead
        self.max_delay = max(m.lm.delays)
        self.device = device
        log.info("loaded: n_q=%d delay_steps=%d max_delay=%d second_stream_ahead=%d valid_cfg=%s",
                 m.n_q, m.delay_steps, self.max_delay, self.ssa, sorted(m.valid_cfg_conditionings))
        m.mimi.streaming_forever(1)
        self.gen_key = None
        self.lm_gen = None
        self.gen_stack = None
        # per-reply state, read by the hooks
        self.offset = 0
        self.state = None
        self.fg: collections.deque = collections.deque()
        self.bg: collections.deque = collections.deque()
        self.cv = threading.Condition()
        self.missing = torch.full((1, m.lm.n_q - m.lm.dep_q, 1), m.machine.token_ids.zero, dtype=torch.long, device=device)
        self.no_depformer = torch.full((1, m.lm.dep_q, 1), m.machine.token_ids.zero, dtype=torch.long, device=device)

    # ---- model plumbing -------------------------------------------------------------------
    def _on_text(self, t):
        out = [self.m.machine.process(self.offset, self.state, tok)[0] for tok in t.tolist()]
        t[:] = torch.tensor(out, dtype=torch.long, device=t.device)

    def _on_audio(self, a):
        m = self.m
        for q in range(a.shape[1]):
            d = m.lm.delays[q + m.lm.audio_offset]
            if self.offset < d + m.delay_steps:
                a[:, q] = m.machine.token_ids.zero

    def _gen_for(self, voice: str, cfg: float):
        """One LMGen per (voice, cfg), kept streaming forever so its CUDA graphs are reused."""
        key = (voice, cfg)
        if self.gen_key != key:
            m = self.m
            if self.gen_stack is not None:
                self.gen_stack.close()
            path = m.get_voice_path(voice)
            cond = m.make_condition_attributes([path], cfg_coef=cfg)
            ct = m.lm.condition_provider(m.lm.condition_provider.prepare([cond]))
            self.lm_gen = LMGen(m.lm, temp=m.temp, temp_text=m.temp, cfg_coef=m.cfg_coef, condition_tensors=ct,
                                on_text_hook=self._on_text, on_audio_hook=self._on_audio,
                                cfg_is_masked_until=None, cfg_is_no_text=True)
            self.gen_stack = self.lm_gen.streaming(1)
            self.gen_key = key
        else:
            self.lm_gen.reset_streaming()
        self.m.mimi.reset_streaming()
        return self.lm_gen

    def _step(self, g):
        """One model step (80 ms of audio). Returns float32 pcm or None."""
        m = self.m
        rep = self.no_depformer if self.offset < m.delay_steps else None
        f = g.step(self.missing, depformer_replace_tokens=rep)
        self.offset += 1
        if f is not None and (f != -1).all():
            return np.clip(m.mimi.decode(f[:, 1:, :]).float().cpu().numpy()[0, 0], -1, 1)
        return None

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

    # ---- worker ------------------------------------------------------------------------------
    def submit(self, job: Job):
        with self.cv:
            (self.bg if job.background else self.fg).append(job)
            self.cv.notify()

    def _next(self) -> Job:
        with self.cv:
            while True:
                for q in (self.fg, self.bg):
                    while q:
                        job = q.popleft()
                        if not job.cancelled:
                            return job
                self.cv.wait()

    def _check_preempt(self, job: Job):
        if job.background:
            with self.cv:
                if any(not j.cancelled for j in self.fg):
                    raise Preempted()

    def run_forever(self):
        while True:
            job = self._next()
            try:
                with torch.no_grad():
                    self._run(job)
            except Preempted:
                job.preempted += 1
                job.replay = job.history + job.replay
                job.history = []
                job.held = []
                log.info("background job preempted by a reply (%d)", job.preempted)
                with self.cv:
                    self.bg.appendleft(job)
                continue
            except Exception as e:  # noqa: BLE001
                log.exception("tts job failed")
                self._send(job, {"type": "Error", "message": f"tts failed: {e}"})
                self._send(job, None)
            if job.background:
                held, job.background = job.held, False
                for item in held:
                    self._send(job, item)

    def _send(self, job: Job, item):
        if job.background:
            job.held.append(item)
        elif not job.cancelled and job.out is not None:
            job.loop.call_soon_threadsafe(job.out.put_nowait, item)

    @staticmethod
    def _event(job: Job, timeout: float | None = None):
        if job.replay:
            return job.replay.pop(0)
        return job.events.get_nowait() if timeout is None else job.events.get(timeout=timeout)

    def _run(self, job: Job):
        m = self.m
        frame_s = 1.0 / m.mimi.frame_rate
        buf = ""            # partial word not yet terminated by whitespace
        pending: list[Entry] = []
        words: list[str] = []   # every word queued so far (== transcript order)
        marks: list[int] = []   # word count at each Flush
        n_flushed = 0
        closed = False
        frames_sent = 0
        first_pcm_offset = None
        words_sent = 0
        g = None
        t_gen = 0.0
        t_first_audio = None
        self.state = None

        def take_words(text: str, final: bool):
            nonlocal buf
            buf += text
            buf = buf.replace("’", "'").replace(":", " ").replace("(", "").replace(")", "")
            if final:
                parts, buf = buf.split(), ""
            else:
                cut = max(buf.rfind(" "), buf.rfind("\n"), buf.rfind("\t"))
                if cut < 0:
                    return
                parts, buf = buf[:cut].split(), buf[cut + 1:]
            ents = self.make_entries(parts, first=not words)
            pending.extend(ents)
            words.extend(e.text for e in ents)

        last_ev = [time.perf_counter()]

        def idle_check():
            # a reply that stops sending text without Close would hold the worker forever
            nonlocal closed
            if not closed and time.perf_counter() - last_ev[0] > IDLE_CLOSE_S:
                log.warning("no text for %.0f s: treating as Close", IDLE_CLOSE_S)
                take_words("", True)
                closed = True

        def handle(ev):
            nonlocal closed
            job.history.append(ev)
            last_ev[0] = time.perf_counter()
            kind, val = ev
            if kind == "text":
                if job.t_first_speak is None:
                    job.t_first_speak = time.perf_counter()
                take_words(val, False)
            elif kind == "flush":
                take_words("", True)
                marks.append(len(words))
            elif kind == "close":
                take_words("", True)
                closed = True

        def word_frame(step: int) -> float:
            # pcm index (fractional, in frames) at which text consumed at `step` is heard
            return step + m.delay_steps + self.max_delay + 1 - first_pcm_offset + WORD_OFFSET_S / frame_s

        def word_end(i: int):
            tr = self.state.transcript
            if i + 1 < len(tr):
                return word_frame(tr[i + 1][1])
            if self.state.end_step is not None and i < len(tr):
                return word_frame(self.state.end_step)
            return None

        def emit_progress(final: bool):
            nonlocal words_sent, n_flushed
            if first_pcm_offset is None and not final:
                return
            tr = self.state.transcript if self.state else []
            batch = []
            while words_sent < len(tr):
                w, s = tr[words_sent]
                st = word_frame(s) if first_pcm_offset is not None else 0.0
                en = word_end(words_sent)
                if en is None and final:
                    en = frames_sent
                if en is None or (frames_sent < st and not final):
                    break
                en = min(en, frames_sent) if final else en
                batch.append({"word": w, "start": round(max(0.0, st) * frame_s, 3), "end": round(max(st, en) * frame_s, 3)})
                words_sent += 1
            if batch:
                self._send(job, {"type": "Words", "words": batch})
            while n_flushed < len(marks):
                n = marks[n_flushed]
                if not final and n > 0:
                    en = word_end(n - 1) if first_pcm_offset is not None else None
                    if en is None or frames_sent < en:
                        break
                self._send(job, {"type": "Flushed"})
                n_flushed += 1

        # wait for the first word (or Close) before touching the model
        while not job.cancelled and not pending and not closed:
            self._check_preempt(job)
            try:
                handle(self._event(job, 0.1))
            except queue.Empty:
                idle_check()
        if job.cancelled:
            return
        if pending:
            g = self._gen_for(job.voice, job.cfg)
            self.offset = 0
            self.state = m.machine.new_state([])
            end_at = None
            while not job.cancelled:
                self._check_preempt(job)
                while True:
                    try:
                        handle(self._event(job))
                    except queue.Empty:
                        break
                st = self.state
                while pending and len(st.entries) <= self.ssa:
                    st.entries.append(pending.pop(0))
                if not closed:
                    # Could the model ask for a word at this step? Then it must see the full lookahead.
                    can_ask = not st.queued and st.forced_padding <= 0
                    if can_ask and len(st.entries) <= self.ssa:
                        try:
                            handle(self._event(job, 0.05))
                        except queue.Empty:
                            idle_check()
                        continue
                if st.end_step is not None:
                    if end_at is None:
                        end_at = st.end_step + m.delay_steps + self.max_delay + 8
                    if self.offset >= end_at:
                        break
                if self.offset > 30000:
                    break
                t0 = time.perf_counter()
                pcm = self._step(g)
                t_gen += time.perf_counter() - t0
                if pcm is not None:
                    if first_pcm_offset is None:
                        first_pcm_offset = self.offset
                        t_first_audio = time.perf_counter()
                    pcm16 = (pcm * 32767).astype("<i2").tobytes()
                    self._send(job, pcm16)
                    frames_sent += 1
                emit_progress(False)
        if job.cancelled:
            log.info("tts cancelled after %d frames", frames_sent)
            return
        emit_progress(True)
        audio_s = frames_sent * frame_s
        stats = {"audio_s": round(audio_s, 3), "gen_s": round(t_gen, 3), "rtf": round(t_gen / audio_s, 3) if audio_s else None,
                 "ttfa_ms": round((t_first_audio - job.t_first_speak) * 1000) if t_first_audio and job.t_first_speak else None,
                 "words": len(words)}
        if job.background:
            stats["background"] = True
            stats["preempted"] = job.preempted
        log.info("tts done %s", stats)
        self._send(job, {"type": "Done", "stats": stats})
        self._send(job, None)

    def warmup(self):
        job = Job(voice=DEFAULT_VOICE, cfg=2.0, loop=_NullLoop(), out=None)
        for ev in [("text", "Hello there, this is a short warm up sentence to capture the graphs."), ("close", None)]:
            job.events.put(ev)
        t = time.perf_counter()
        with torch.no_grad():
            self._run(job)
        log.info("warmup %.2fs", time.perf_counter() - t)


class _NullLoop:
    def call_soon_threadsafe(self, *a):
        pass


def build_app(engine: Engine, default_voice: str, default_cfg: float) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True, "tts": DEFAULT_DSM_TTS_REPO, "n_q": engine.m.n_q}

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
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    engine = Engine(n_q=a.n_q, temp=a.temp)
    engine.warmup()
    engine.warmup()
    threading.Thread(target=engine.run_forever, daemon=True).start()
    uvicorn.run(build_app(engine, a.voice, a.cfg), host=a.host, port=a.port, log_level="warning", ws_max_size=2**24)


if __name__ == "__main__":
    main()
