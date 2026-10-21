"""Streaming Qwen3-TTS worker (Qwen3-TTS-12Hz-*-CustomVoice via the `qwen-tts` package), a drop-in for kyutai_tts.py.

Same port and protocol as the Kyutai worker (PROTOCOL.md, "TTS: WS /v1/tts"); setup.sh runs one or the
other (TTS_ENGINE=kyutai|qwen), since both do not fit on the L4 next to the ASR and the LLM.

How the model reads text (issue #27): the talker takes exactly one text token per 80 ms frame, added to
that frame's codec embedding, then tts_eos, then tts_pad until it ends the audio with codec_eos. A pad
in the middle of the text means "the text is over" and the model winds down, so a token that has not
arrived yet must stall the step, never be replaced by a pad. qwen-tts 0.1.1 only simulates streaming
(it needs the whole text up front), so this is our own step loop over its modules:
prefill = the generate() prompt with the first text token, then one talker forward per frame (the
code predictor fills the other 15 codebooks inside it), and the 12 Hz codec decodes frames in small
chunks with 25 frames of left context (it is causal, so chunks join without seams).
Stalling costs nothing in the audio: it only depends on the token sequence. Real time needs the text
to come at >= 12.5 tokens/s on average once the audio has started.

The model gives no word timings, so the gateway's Qwen3-ForcedAligner (POST /v1/align) times the
words on the audio as it is made: words that end well before the end of the audio so far are
settled and sent as `Words`; the rest are aligned again once more audio is out, and all of them at
the end.

One reply at a time (the box serves one user); background jobs (canned lines) run only when no
reply waits, and a reply preempts one, which then starts over (its output is held until it is done).
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

import httpx
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

log = logging.getLogger("qwen-tts")

DEFAULT_MODEL = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
DEFAULT_VOICE = os.environ.get("QWEN_TTS_VOICE", "ryan")
SR = 24000
FRAME_S = 0.08
IDLE_CLOSE_S = float(os.environ.get("QWEN_TTS_IDLE_CLOSE_S", "30"))
FIRST_CHUNK = int(os.environ.get("QWEN_TTS_FIRST_CHUNK", "2"))   # frames decoded for the first audio
CHUNK = int(os.environ.get("QWEN_TTS_CHUNK", "4"))               # frames per decode after that
LEFT_CTX = 25                                                    # codec frames of left context per decode
ALIGN_URL = os.environ.get("ALIGN_URL", "http://127.0.0.1:8000/v1/align")
ALIGN_EVERY_S = float(os.environ.get("QWEN_TTS_ALIGN_EVERY_S", "0.8"))
SETTLE_MARGIN_S = 1.2    # a word is settled once it ends this far before the end of the audio aligned: the model
                         # reads text well ahead of speaking it, and the aligner squeezes words not spoken yet into the end
MAX_FRAMES = 1900   # ~150 s; the graphed KV cache holds 2048 positions


@dataclass
class Job:
    voice: str
    loop: asyncio.AbstractEventLoop
    out: asyncio.Queue | None
    instruct: str = ""
    events: "queue.Queue" = field(default_factory=queue.Queue)
    cancelled: bool = False
    background: bool = False
    t_first_speak: float | None = None
    history: list = field(default_factory=list)
    held: list = field(default_factory=list)
    preempted: int = 0
    finished: bool = False


def _send(job: Job, item):
    if job.background:
        job.held.append(item)
    elif not job.cancelled and job.out is not None:
        job.loop.call_soon_threadsafe(job.out.put_nowait, item)


class Engine:
    def __init__(self, model: str, voice: str, temp: float, sub_temp: float, device: str = "cuda"):
        from qwen_tts import Qwen3TTSModel
        self.device = device
        attn = os.environ.get("QWEN_TTS_ATTN", "sdpa")
        self.w = Qwen3TTSModel.from_pretrained(model, device_map=device, dtype=torch.bfloat16, attn_implementation=attn)
        self.m = self.w.model
        self.talker = self.m.talker
        self.cfg = self.m.config
        self.tcfg = self.cfg.talker_config
        self.tok = self.w.processor.tokenizer if hasattr(self.w.processor, "tokenizer") else self.w.processor
        self.codec = self.m.speech_tokenizer.model.decoder
        self.upsample = int(self.codec.total_upsample)
        self.model_name = model
        self.voice = voice
        self.temp, self.sub_temp = temp, sub_temp
        self.speakers = sorted(self.tcfg.spk_id.keys())
        self.use_instruct = getattr(self.m, "tts_model_size", "") not in ("0b6", "") if hasattr(self.m, "tts_model_size") else "0.6B" not in model
        V = self.tcfg.vocab_size
        self.suppress = torch.tensor([i for i in range(V - 1024, V) if i != self.tcfg.codec_eos_token_id], device=device)
        emb = self.talker.get_text_embeddings()
        with torch.no_grad():
            ids = torch.tensor([[self.cfg.tts_bos_token_id, self.cfg.tts_eos_token_id, self.cfg.tts_pad_token_id]], device=device)
            self.tts_bos, self.tts_eos, self.tts_pad = self.talker.text_projection(emb(ids)).chunk(3, dim=1)
            role = self.tok("<|im_start|>assistant\n", return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            self.role_embed = self.talker.text_projection(emb(role))
        self.fg: collections.deque = collections.deque()
        self.bg: collections.deque = collections.deque()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.http = httpx.Client(timeout=10)
        self.token = os.environ.get("GPUBOX_TOKEN", "")
        self.fast = None
        if os.environ.get("QWEN_TTS_GRAPHS", "1") == "1":
            from qwen3_tts_graphs import FastTalker
            with torch.no_grad():
                self.fast = FastTalker(self.talker, sub_temp=sub_temp)
                self.fast.capture()
        log.info("loaded %s speakers=%s upsample=%d graphs=%s", model, self.speakers, self.upsample, self.fast is not None)

    # ---- model ---------------------------------------------------------------------------------
    def text_embed(self, ids: list[int]) -> torch.Tensor:
        t = torch.tensor([ids], device=self.device)
        return self.talker.text_projection(self.talker.get_text_embeddings()(t))

    def prefill(self, voice: str, first_tok: int, instruct: str = ""):
        """The prompt generate() builds for a custom voice in streaming mode, up to the first text token,
        after the instruct (a voice direction; the 0.6B models ignore it, as qwen-tts does)."""
        c = self.tcfg
        E = self.talker.get_input_embeddings()
        lang = c.codec_language_id["english"]
        dev = self.device
        c0 = E(torch.tensor([[c.codec_think_id, c.codec_think_bos_id, lang, c.codec_think_eos_id]], device=dev))
        spk = E(torch.tensor(c.spk_id[voice], device=dev)).view(1, 1, -1)
        c1 = E(torch.tensor([[c.codec_pad_id, c.codec_bos_id]], device=dev))
        codec = torch.cat([c0, spk, c1], dim=1)
        x = torch.cat((self.tts_pad.expand(-1, codec.shape[1] - 2, -1), self.tts_bos), dim=1) + codec[:, :-1]
        x = torch.cat([self.role_embed, x, self.text_embed([first_tok]) + codec[:, -1:]], dim=1)
        if instruct and self.use_instruct:
            ids = self.tok(f"<|im_start|>user\n{instruct}<|im_end|>\n", add_special_tokens=False)["input_ids"]
            x = torch.cat([self.text_embed(ids), x], dim=1)
        return x

    def sample(self, logits: torch.Tensor, prev: list[int], n: int) -> int:
        lg = logits.float().clone()
        if prev:
            idx = torch.tensor(sorted(set(prev)), device=lg.device)
            s = lg[idx]
            lg[idx] = torch.where(s > 0, s / 1.05, s * 1.05)
        lg[self.suppress] = -float("inf")
        if n < 2:
            lg[self.tcfg.codec_eos_token_id] = -float("inf")
        lg = lg / self.temp
        v, i = torch.topk(lg, 50)
        p = torch.softmax(v, dim=-1)
        return int(i[torch.multinomial(p, 1)])

    def decode(self, frames: list[torch.Tensor], start: int, end: int) -> np.ndarray:
        ctx = min(LEFT_CTX, start)
        codes = torch.stack(frames[start - ctx:end], dim=0).T.unsqueeze(0)  # 1 x 16 x T
        wav = self.codec(codes)[0, 0, ctx * self.upsample:]
        return wav.float().cpu().numpy()

    # ---- scheduling --------------------------------------------------------------------------
    def submit(self, job: Job):
        with self.lock:
            (self.bg if job.background else self.fg).append(job)
        self.wake.set()

    def next_job(self) -> Job | None:
        with self.lock:
            for q in (self.fg, self.bg):
                while q:
                    j = q.popleft()
                    if not j.cancelled:
                        return j
        return None

    def reply_waiting(self) -> bool:
        with self.lock:
            return any(not j.cancelled for j in self.fg)

    def run_forever(self):
        while True:
            job = self.next_job()
            if job is None:
                self.wake.wait(0.05)
                self.wake.clear()
                continue
            r = Run(self, job)
            try:
                with torch.no_grad():
                    r.run()
            except Preempted:
                job.preempted += 1
                job.held = []
                replay = job.history
                job.history = []
                fresh = queue.Queue()
                for ev in replay:
                    fresh.put(ev)
                while True:
                    try:
                        fresh.put(job.events.get_nowait())
                    except queue.Empty:
                        break
                job.events = fresh
                log.info("background job preempted by a reply (%d)", job.preempted)
                with self.lock:
                    self.bg.appendleft(job)
            except Exception as e:  # noqa: BLE001
                log.exception("tts failed")
                r.fail(f"tts failed: {e}")

    def align(self, pcm24: np.ndarray, words: list[str]) -> list[tuple[float, float]] | None:
        if not words or len(pcm24) < SR // 10:
            return None
        n16 = int(len(pcm24) * 16000 / SR)
        pcm16 = np.interp(np.linspace(0, len(pcm24) - 1, n16), np.arange(len(pcm24)), pcm24)
        body = (np.clip(pcm16, -1, 1) * 32767).astype("<i2").tobytes()
        try:
            r = self.http.post(ALIGN_URL, content=body, params={"text": " ".join(words), "language": "English"},
                               headers={"Authorization": f"Bearer {self.token}"})
            items = r.json()["items"]
        except Exception as e:  # noqa: BLE001
            log.warning("align failed: %s", e)
            return None
        if len(items) != len(words):
            log.warning("align: %d items for %d words", len(items), len(words))
            return None
        return [(float(it["start"]), float(it["end"])) for it in items]

    def warmup(self):
        job = Job(voice=self.voice, loop=_NullLoop(), out=None)
        for ev in [("text", "Hello there, this is a short warm up sentence."), ("close", None)]:
            job.events.put(ev)
        t = time.perf_counter()
        with torch.no_grad():
            Run(self, job, align=False).run()
        log.info("warmup %.2fs", time.perf_counter() - t)


class Preempted(Exception):
    pass


class Run:
    def __init__(self, eng: Engine, job: Job, align: bool = True):
        self.eng, self.job, self.use_align = eng, job, align
        self.buf = ""               # text not yet committed (no whitespace after it yet)
        self.committed = ""         # text up to the last whitespace
        self.toks: list[int] = []   # tokens of `committed`
        self.fed = 0                # tokens given to the model
        self.words: list[str] = []
        self.word_tok_end: list[int] = []   # token count at the end of each word
        self.marks: list[int] = []
        self.closed = False
        self.instruct = job.instruct
        self.last_ev = time.perf_counter()

    # ---- text in -------------------------------------------------------------------------------
    def commit(self, final: bool):
        if final:
            text, self.buf = self.committed + " " + self.buf, ""
        else:
            cut = max(self.buf.rfind(" "), self.buf.rfind("\n"), self.buf.rfind("\t"))
            if cut < 0:
                return
            text, self.buf = self.committed + " " + self.buf[:cut], self.buf[cut + 1:]
        text = " ".join(text.split())
        if not text or text == self.committed:
            return
        # words carry their leading space into their tokens, so the tokens of the text so far are a stable prefix
        toks = self.eng.tok(text, add_special_tokens=False)["input_ids"]
        if toks[:len(self.toks)] != self.toks:
            log.warning("token prefix changed; keeping what was fed")
            toks = self.toks + toks[len(self.toks):]
        new_words = text[len(self.committed):].split()
        self.committed, self.toks = text, toks
        # word i ends at the token count of the text up to it (approximate at worst by a token)
        for w in new_words:
            self.words.append(w)
            upto = " ".join(self.words)
            self.word_tok_end.append(len(self.eng.tok(upto, add_special_tokens=False)["input_ids"]))

    def handle(self, ev):
        job = self.job
        job.history.append(ev)
        self.last_ev = time.perf_counter()
        kind, val = ev
        if kind == "text":
            if job.t_first_speak is None:
                job.t_first_speak = time.perf_counter()
            self.buf += val.replace("’", "'")
            self.commit(False)
        elif kind == "flush":
            self.commit(True)
            self.marks.append(len(self.words))
        elif kind == "instruct":
            if not self.toks:   # only before the first word: it goes in the prompt
                self.instruct = str(val or "").strip()[:300]
        elif kind == "close":
            self.commit(True)
            self.closed = True

    def pull(self, block: bool):
        job = self.job
        while True:
            try:
                ev = job.events.get(timeout=0.02) if block else job.events.get_nowait()
            except queue.Empty:
                break
            block = False
            self.handle(ev)
        if not self.closed and time.perf_counter() - self.last_ev > IDLE_CLOSE_S:
            log.warning("no text for %.0f s: treating as Close", IDLE_CLOSE_S)
            self.commit(True)
            self.closed = True

    def check(self):
        if self.job.cancelled:
            raise Cancelled()
        if self.job.background and self.eng.reply_waiting():
            raise Preempted()

    def words_read(self) -> int:
        n = 0
        while n < len(self.words) and self.word_tok_end[n] <= self.fed:
            n += 1
        return n

    # ---- the loop ---------------------------------------------------------------------------------
    def run(self):
        eng, job = self.eng, self.job
        try:
            # first token (or nothing to say)
            while not self.toks and not self.closed:
                self.check()
                self.pull(True)
            if not self.toks:
                self.finish([], np.zeros(0, np.float32), 0.0, [])
                return
            t_gen = 0.0
            steps: list[float] = []
            t0 = time.perf_counter()
            x = eng.prefill(job.voice, self.toks[0], self.instruct)
            if self.instruct:
                log.info("instruct %r (%s)", self.instruct, "used" if eng.use_instruct else "ignored by this model")
            self.fed = 1
            if eng.fast is not None:
                logits0, _ = eng.fast.prefill(x)
            else:
                L = x.shape[1]
                from transformers import DynamicCache
                cache = DynamicCache()
                out = eng.talker(inputs_embeds=x, attention_mask=torch.ones(1, L, dtype=torch.long, device=eng.device),
                                 past_key_values=cache, use_cache=True, cache_position=torch.arange(L, device=eng.device))
                logits0, past_hidden, pos = out.logits[0, -1], out.past_hidden, L
            t_gen += time.perf_counter() - t0
            prev: list[int] = []
            c0 = eng.sample(logits0, prev, 0)
            frames: list[torch.Tensor] = []
            decoded = 0
            pcm_parts: list[np.ndarray] = []
            n_samples = 0
            t_first_audio = None
            eos_fed = False
            words_sent = 0
            settled_t = 0.0
            last_align_at = 0.0
            reported = (-1, -1)
            flushed = 0
            word_times: list[tuple[float, float]] = []
            while True:
                self.check()
                self.pull(False)
                if c0 == eng.tcfg.codec_eos_token_id or len(frames) >= MAX_FRAMES:
                    break
                # the text for the next input: a token, else eos once closed, else pad after eos; else wait
                if self.fed < len(self.toks):
                    nxt = eng.text_embed([self.toks[self.fed]])
                    self.fed += 1
                elif self.closed:
                    nxt = eng.tts_pad if eos_fed else eng.tts_eos
                    eos_fed = True
                else:
                    self.pull(True)
                    continue
                ts = time.perf_counter()
                if eng.fast is not None:
                    codes, logits = eng.fast.frame(c0, nxt)
                else:
                    out = eng.talker(input_ids=torch.tensor([[c0]], device=eng.device), past_key_values=cache, use_cache=True,
                                     attention_mask=torch.ones(1, pos + 1, dtype=torch.long, device=eng.device),
                                     cache_position=torch.tensor([pos], device=eng.device), past_hidden=past_hidden,
                                     trailing_text_hidden=nxt, tts_pad_embed=eng.tts_pad, generation_step=0,
                                     subtalker_dosample=True, subtalker_top_k=50, subtalker_top_p=1.0,
                                     subtalker_temperature=eng.sub_temp)
                    pos += 1
                    codes, logits, past_hidden = out.hidden_states[1][0], out.logits[0, -1], out.past_hidden
                frames.append(codes)
                prev.append(c0)
                c0 = eng.sample(logits, prev, len(frames))
                # decode a chunk of frames
                want = FIRST_CHUNK if decoded == 0 else CHUNK
                if len(frames) - decoded >= want:
                    pcm = eng.decode(frames, decoded, len(frames))
                    decoded = len(frames)
                    self.emit_audio(pcm)
                    pcm_parts.append(pcm)
                    n_samples += len(pcm)
                    if t_first_audio is None:
                        t_first_audio = time.perf_counter()
                dt = time.perf_counter() - ts
                steps.append(dt)
                t_gen += dt
                # progress: words read, words aligned
                q = (self.words_read(), len(self.words) - self.words_read())
                if q != reported:
                    reported = q
                    _send(job, {"type": "Consumed", "words": q[0], "queued": q[1]})
                audio_s = n_samples / SR
                if self.use_align and audio_s - last_align_at >= ALIGN_EVERY_S and words_sent < self.words_read():
                    last_align_at = audio_s
                    words_sent, settled_t, flushed = self.settle(pcm_parts, n_samples, words_sent, settled_t,
                                                                 word_times, flushed, final=False)
            if decoded < len(frames):
                pcm = eng.decode(frames, decoded, len(frames))
                self.emit_audio(pcm)
                pcm_parts.append(pcm)
                n_samples += len(pcm)
                if t_first_audio is None:
                    t_first_audio = time.perf_counter()
            self.settle(pcm_parts, n_samples, words_sent, settled_t, word_times, flushed, final=True)
            audio_s = n_samples / SR
            srt = sorted(steps)
            head = steps[:int(3 / FRAME_S)]
            stats = {"audio_s": round(audio_s, 3), "gen_s": round(t_gen, 3),
                     "rtf": round(t_gen / audio_s, 3) if audio_s else None,
                     "rtf_3s": round(sum(head) / (len(head) * FRAME_S), 3) if head else None,
                     "step_ms_p50": round(srt[len(srt) // 2] * 1000, 1) if srt else None,
                     "step_ms_p90": round(srt[int(0.9 * (len(srt) - 1))] * 1000, 1) if srt else None,
                     "step_ms_max": round(srt[-1] * 1000, 1) if srt else None,
                     "ttfa_ms": round((t_first_audio - job.t_first_speak) * 1000) if t_first_audio and job.t_first_speak else None,
                     "frames": len(frames), "tokens": len(self.toks), "words": len(self.words), "engine": "qwen3-tts"}
            if job.background:
                stats["background"] = True
                stats["preempted"] = job.preempted
            self.finish_stats(stats)
        except Cancelled:
            log.info("tts cancelled")
            job.finished = True

    def emit_audio(self, pcm: np.ndarray):
        _send(self.job, (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())

    def settle(self, parts, n_samples, words_sent, settled_t, word_times, flushed, final):
        """Align the words read so far that are not settled yet on the audio after the last settled word."""
        upto = len(self.words) if final else self.words_read()
        words = self.words[words_sent:upto]
        audio_s = n_samples / SR
        batch = []
        if words:
            pcm = np.concatenate(parts) if parts else np.zeros(0, np.float32)
            s0 = int(settled_t * SR)
            times = self.eng.align(pcm[s0:], words) if self.use_align else None
            if times is None and final:
                # no aligner: spread the rest over the rest of the audio
                d = max(audio_s - settled_t, 0.0) / len(words)
                times = [(i * d, (i + 1) * d) for i in range(len(words))]
            if times is not None:
                for k, (s, e) in enumerate(times):
                    s, e = settled_t + s, settled_t + e
                    if not final and (e > audio_s - SETTLE_MARGIN_S or e - s < 0.04 or k >= len(words) - 1):
                        break
                    batch.append({"word": words[k], "start": round(s, 3), "end": round(max(s, e), 3)})
        if batch:
            _send(self.job, {"type": "Words", "words": batch})
            words_sent += len(batch)
            word_times.extend((b["start"], b["end"]) for b in batch)
            # the next alignment starts between the last settled word and the next one
            settled_t = batch[-1]["end"]
        while flushed < len(self.marks) and (final or self.marks[flushed] <= words_sent):
            _send(self.job, {"type": "Flushed"})
            flushed += 1
        return words_sent, settled_t, flushed

    def finish_stats(self, stats):
        log.info("tts done %s", stats)
        _send(self.job, {"type": "Done", "stats": stats})
        _send(self.job, None)
        self.release()

    def finish(self, *_):
        self.finish_stats({"audio_s": 0, "words": 0, "engine": "qwen3-tts"})

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


class Cancelled(Exception):
    pass


class _NullLoop:
    def call_soon_threadsafe(self, *a):
        pass


def build_app(engine: Engine) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True, "tts": engine.model_name, "engine": "qwen3-tts", "speakers": engine.speakers,
                "instruct": engine.use_instruct}

    @app.get("/config")
    def config():
        return {"model": engine.model_name, "n_q": 16, "temp": engine.temp, "voice": engine.voice, "cfg": None,
                "sample_rate": SR, "engine": "qwen3-tts", "speakers": engine.speakers}

    @app.websocket("/v1/tts")
    async def tts(ws: WebSocket):
        await ws.accept()
        voice = (ws.query_params.get("voice") or engine.voice).lower()
        if voice not in engine.speakers:
            # a Kyutai voice path from a graph made for Kyutai: use the default speaker
            voice = engine.voice
        job = Job(voice=voice, loop=asyncio.get_running_loop(), out=asyncio.Queue(), instruct=ws.query_params.get("instruct") or "",
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
                elif t == "Instruct":
                    job.events.put(("instruct", str(d.get("text", ""))))
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
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--sub-temp", type=float, default=0.9)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    engine = Engine(a.model, a.voice.lower(), a.temp, a.sub_temp)
    engine.warmup()
    engine.warmup()
    threading.Thread(target=engine.run_forever, daemon=True).start()
    uvicorn.run(build_app(engine), host=a.host, port=a.port, log_level="warning", ws_max_size=2**24)


if __name__ == "__main__":
    main()
