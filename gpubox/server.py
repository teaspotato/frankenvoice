"""Frankenvoice GPU box gateway. Implements PROTOCOL.md.

- `WS /v1/asr`: streaming ASR on top of a local sglang-omni server (Qwen3-ASR-1.7B, FP8) that only
  does whole-utterance transcriptions. This process keeps the per-connection segment buffer, the
  energy gate, partial re-decodes, and times final words with Qwen3-ForcedAligner-0.6B (loaded here).
- `WS /v1/tts`, `GET /v1/tts/config`: relayed to the Kyutai TTS worker (kyutai_tts.py, its own venv) on localhost.
- `POST /v1/chat/completions`, `GET /v1/models`: passed through (streaming) to the local OpenAI-compatible
  LLM server (llama-server with Gemma 4 12B on LLM_URL). Closing the client connection stops generation.
- `GET /health`.
Every route needs `Authorization: Bearer $GPUBOX_TOKEN` (or `?token=`).

Env: GPUBOX_TOKEN (required), ASR_URL (http://127.0.0.1:8101), TTS_URL (ws://127.0.0.1:8102),
LLM_URL (http://127.0.0.1:8103), ASR_MODEL, ALIGN_MODEL, TTS_MODEL, LLM_MODEL,
NO_ALIGNER=1 (skip the aligner: final words spread evenly).
"""
import argparse
import asyncio
import hmac
import io
import json
import logging
import os
import re
import threading
import time
import wave
from collections import deque
from urllib.parse import urlencode

import httpx
import numpy as np
import uvicorn
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse

log = logging.getLogger("gpubox")

TOKEN = os.environ.get("GPUBOX_TOKEN", "")
ASR_URL = os.environ.get("ASR_URL", "http://127.0.0.1:8101")
TTS_URL = os.environ.get("TTS_URL", "ws://127.0.0.1:8102")
ASR_MODEL = os.environ.get("ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
ALIGN_MODEL = os.environ.get("ALIGN_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")
TTS_MODEL = os.environ.get("TTS_MODEL", "kyutai/tts-1.6b-en_fr")
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8103")
LLM_MODEL = os.environ.get("LLM_MODEL", "")

SR = 16000
FRAME = 320                 # 20 ms energy-gate frames
FLOOR_FRAMES = 250          # noise floor over the last ~5 s
VOICE_ABOVE_FLOOR_DB = 12.0
VOICE_MIN_DBFS = -55.0
LEAD_SILENCE_S = 1.0        # no voiced frames for longer than this -> trim
LEAD_KEEP_S = 0.5           # ... keeping the last 0.5 s
MAX_SEGMENT_S = 25.0        # force a final
UTTERANCE_END_S = 1.0

LANG_CODES = {"english": "en", "chinese": "zh", "french": "fr", "german": "de", "spanish": "es", "japanese": "ja",
              "korean": "ko", "italian": "it", "portuguese": "pt", "russian": "ru"}

app = FastAPI()
http: httpx.AsyncClient | None = None
aligner = None
align_lock = threading.Lock()
align_stats: deque = deque(maxlen=200)


# ---------------------------------------------------------------- auth
def _authorized(headers, query) -> bool:
    if not TOKEN:
        return False
    auth = headers.get("authorization", "")
    got = auth[7:].strip() if auth.lower().startswith("bearer ") else query.get("token", "")
    return hmac.compare_digest(got.encode(), TOKEN.encode())


@app.middleware("http")
async def auth_mw(request: Request, call_next):
    if not _authorized(request.headers, request.query_params):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


@app.get("/health")
async def health():
    return {"ok": True, "asr": ASR_MODEL, "tts": TTS_MODEL, "llm": LLM_MODEL or None, "aligner": ALIGN_MODEL if aligner else None,
            "align_ms_p50": _p50(align_stats)}


def _p50(xs):
    xs = sorted(xs)
    return round(xs[len(xs) // 2], 1) if xs else None


# ---------------------------------------------------------------- ASR helpers
def wav_bytes(pcm16: np.ndarray) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm16.astype("<i2").tobytes())
    return b.getvalue()


def clean_text(s: str) -> str:
    s = re.sub(r"<[^>]*>", " ", s)
    s = re.sub(r"^\s*language\s+[A-Za-z]+\s*", " ", s)
    return re.sub(r"\s+", " ", s).strip()


async def transcribe(pcm16: np.ndarray, lang_code: str, context: str | None) -> tuple[str, float]:
    data = {"model": ASR_MODEL, "language": lang_code, "temperature": "0"}
    if context:
        data["prompt"] = context
    t = time.perf_counter()
    r = await http.post(f"{ASR_URL}/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes(pcm16), "audio/wav")}, data=data)
    dt = (time.perf_counter() - t) * 1000
    r.raise_for_status()
    return clean_text(r.json().get("text", "")), dt


def _align_sync(pcm16: np.ndarray, text: str, language: str):
    with align_lock:
        t = time.perf_counter()
        res = aligner.align(audio=(pcm16.astype(np.float32) / 32768.0, SR), text=text, language=language)
        dt = (time.perf_counter() - t) * 1000
    return [(it.text, float(it.start_time), float(it.end_time)) for it in res[0].items], dt


def load_aligner():
    global aligner
    if os.environ.get("NO_ALIGNER"):
        return
    import torch
    from qwen_asr import Qwen3ForcedAligner
    aligner = Qwen3ForcedAligner.from_pretrained(ALIGN_MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    rs = np.random.RandomState(0)
    for sec in (1, 2, 3, 5, 8, 12, 2, 4):
        _align_sync((rs.randn(SR * sec) * 300).astype(np.int16), " ".join(["hello there how are you"] * sec), "English")
    log.info("aligner ready")


def spread(words: list[str], t0: float, t1: float) -> list[dict]:
    if not words:
        return []
    d = max(t1 - t0, 0.0) / len(words)
    return [{"word": w, "start": round(t0 + i * d, 3), "end": round(t0 + (i + 1) * d, 3)} for i, w in enumerate(words)]


class AsrSession:
    def __init__(self, ws: WebSocket, language: str, in_sr: int, context: str | None, partial_ms: int):
        self.ws = ws
        self.language = language
        self.lang_code = LANG_CODES.get(language.lower(), language.lower())
        self.in_sr = in_sr
        self.context = context or None
        self.partial_samples = max(1, int(partial_ms * SR / 1000))
        self.total = 0                  # samples received (at 16 kHz)
        self.seg = np.zeros(0, np.int16)
        self.seg_start = 0              # sample index of seg[0]
        self.frame_tail = np.zeros(0, np.int16)
        self.frames: deque = deque()    # (start_sample, voiced) for frames inside the current segment
        self.db_hist: deque = deque(maxlen=FLOOR_FRAMES)
        self.last_voiced = -1           # start sample of the last voiced frame
        self.speech_started = False
        self.new_since_partial = 0
        self.voiced_since_partial = False
        self.inflight: asyncio.Task | None = None
        self.seg_gen = 0                # bumps at every final; stale partials are dropped
        self.ue_at: float | None = None  # pending UtteranceEnd time (stream seconds)
        self.ue_word_end = 0.0
        self.send_lock = asyncio.Lock()
        self.carry = np.zeros(0, np.float32)

    async def send(self, obj):
        async with self.send_lock:
            await self.ws.send_text(json.dumps(obj))

    def t(self, sample: int) -> float:
        return round(sample / SR, 3)

    # ---- audio in
    def _resample(self, x: np.ndarray) -> np.ndarray:
        if self.in_sr == SR:
            return x
        n_out = int(len(x) * SR / self.in_sr)
        if n_out <= 0:
            return np.zeros(0, np.int16)
        xi = np.linspace(0, len(x) - 1, n_out)
        return np.interp(xi, np.arange(len(x)), x.astype(np.float32)).astype(np.int16)

    async def on_audio(self, b: bytes):
        if len(b) % 2:
            b = b[:-1]
        x = self._resample(np.frombuffer(b, "<i2"))
        if not len(x):
            return
        start = self.total
        self.total += len(x)
        self.seg = np.concatenate([self.seg, x])
        self.new_since_partial += len(x)
        # energy gate on 20 ms frames
        buf = np.concatenate([self.frame_tail, x])
        n = len(buf) // FRAME
        first_frame_start = start - len(self.frame_tail)
        for i in range(n):
            fr = buf[i * FRAME:(i + 1) * FRAME].astype(np.float32)
            db = 10 * np.log10(np.mean(fr * fr) / (32768.0 ** 2) + 1e-12)
            self.db_hist.append(db)
            floor = float(np.percentile(self.db_hist, 10))
            if len(self.db_hist) < 50:      # not enough history yet: don't let speech set the floor
                floor = min(floor, -70.0)
            voiced = db > floor + VOICE_ABOVE_FLOOR_DB and db > VOICE_MIN_DBFS
            fs = first_frame_start + i * FRAME
            self.frames.append((fs, voiced))
            if voiced:
                self.last_voiced = fs
                self.voiced_since_partial = True
                if not self.speech_started and fs >= self.seg_start:
                    self.speech_started = True
                    self.ue_at = None
                    await self.send({"type": "SpeechStarted", "at": self.t(max(fs, self.seg_start))})
        self.frame_tail = buf[n * FRAME:]

        # trim leading silence while nothing voiced in the segment
        if not self.speech_started and len(self.seg) > LEAD_SILENCE_S * SR:
            keep = int(LEAD_KEEP_S * SR)
            self.seg_start += len(self.seg) - keep
            self.seg = self.seg[-keep:]
        while self.frames and self.frames[0][0] < self.seg_start - FRAME:
            self.frames.popleft()

        # UtteranceEnd
        if self.ue_at is not None and self.total / SR >= self.ue_at:
            self.ue_at = None
            await self.send({"type": "UtteranceEnd", "at": round(self.ue_word_end, 3)})

        if len(self.seg) > MAX_SEGMENT_S * SR:
            await self.final(from_finalize=False, speech_final=False)
            return
        if (self.new_since_partial >= self.partial_samples and self.speech_started and self.voiced_since_partial
                and (self.inflight is None or self.inflight.done())):
            self.new_since_partial = 0
            self.voiced_since_partial = False
            self.inflight = asyncio.create_task(self.partial(self.seg.copy(), self.seg_start, self.seg_gen))

    def _voiced_span(self, seg_start: int, seg_end: int):
        v = [fs for fs, ok in self.frames if ok and seg_start <= fs < seg_end]
        if not v:
            return seg_start / SR, seg_end / SR
        return max(v[0], seg_start) / SR, min(v[-1] + FRAME, seg_end) / SR

    async def partial(self, pcm: np.ndarray, seg_start: int, gen: int):
        try:
            text, dt = await transcribe(pcm, self.lang_code, self.context)
        except Exception as e:  # noqa: BLE001
            log.warning("partial decode failed: %s", e)
            return
        if gen != self.seg_gen or not text:
            return
        seg_end = seg_start + len(pcm)
        v0, v1 = self._voiced_span(seg_start, seg_end)
        await self.send({"type": "Results", "is_final": False, "speech_final": False, "from_finalize": False,
                         "start": self.t(seg_start), "end": self.t(seg_end), "text": text,
                         "words": spread(text.split(), v0, v1), "decode_ms": round(dt)})

    async def final(self, from_finalize: bool, speech_final: bool):
        t_req = time.perf_counter()
        gen = self.seg_gen
        self.seg_gen += 1             # any in-flight partial is now stale
        # Don't wait for an in-flight partial (it cost up to ~250 ms): its result is dropped (stale gen). Don't cancel it
        # either: closing its request aborts it in sglang-omni, which can crash the server (release_kv_cache assertion,
        # with several streams running). It finishes alongside the final's decode in the same batch.
        self.inflight = None
        pcm, seg_start = self.seg, self.seg_start
        seg_end = seg_start + len(pcm)
        had_speech = self.speech_started
        # new segment starts here
        self.seg = np.zeros(0, np.int16)
        self.seg_start = seg_end
        self.speech_started = False
        self.new_since_partial = 0
        self.voiced_since_partial = False
        text, dt, words, align_ms = "", 0.0, [], None
        if had_speech and len(pcm) > FRAME:
            try:
                text, dt = await transcribe(pcm, self.lang_code, self.context)
            except Exception as e:  # noqa: BLE001
                await self.send({"type": "Error", "message": f"asr decode failed: {e}"})
            if text:
                if aligner is not None:
                    try:
                        items, align_ms = await asyncio.to_thread(_align_sync, pcm, text, self.language)
                        align_stats.append(align_ms)
                        words = [{"word": w, "start": round(seg_start / SR + s, 3), "end": round(seg_start / SR + e, 3)}
                                 for w, s, e in items]
                    except Exception as e:  # noqa: BLE001
                        log.warning("align failed: %s", e)
                if not words:
                    v0, v1 = self._voiced_span(seg_start, seg_end)
                    words = spread(text.split(), v0, v1)
        msg = {"type": "Results", "is_final": True, "speech_final": speech_final, "from_finalize": from_finalize,
               "start": self.t(seg_start), "end": self.t(seg_end), "text": text, "words": words, "decode_ms": round(dt)}
        if align_ms is not None:
            msg["align_ms"] = round(align_ms)
        await self.send(msg)
        total_ms = (time.perf_counter() - t_req) * 1000
        log.info("final seg=%d %.2fs dur=%.2fs decode=%.0fms align=%sms total=%.0fms text=%r", gen, seg_start / SR,
                 len(pcm) / SR, dt, None if align_ms is None else round(align_ms), total_ms, text[:80])
        if words:
            self.ue_word_end = words[-1]["end"]
            self.ue_at = self.ue_word_end + UTTERANCE_END_S
            if self.total / SR >= self.ue_at and self.last_voiced < self.ue_word_end * SR:
                self.ue_at = None
                await self.send({"type": "UtteranceEnd", "at": round(self.ue_word_end, 3)})


@app.websocket("/v1/asr")
async def asr_ws(ws: WebSocket):
    if not _authorized(ws.headers, ws.query_params):
        await ws.close(code=4401)
        return
    q = ws.query_params
    try:
        in_sr = int(q.get("sample_rate", SR))
        partial_ms = int(q.get("partial_ms", 250))
    except ValueError:
        await ws.accept()
        await ws.send_text(json.dumps({"type": "Error", "message": "bad sample_rate/partial_ms"}))
        await ws.close()
        return
    await ws.accept()
    s = AsrSession(ws, q.get("language", "English") or "English", in_sr, q.get("context"), partial_ms)
    await s.send({"type": "Ready", "model": ASR_MODEL})
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                await s.on_audio(msg["bytes"])
                continue
            txt = msg.get("text")
            if txt is None:
                continue
            try:
                d = json.loads(txt)
            except ValueError:
                await s.send({"type": "Error", "message": "bad json"})
                continue
            typ = d.get("type")
            if typ == "Finalize":
                await s.final(from_finalize=True, speech_final=True)
            elif typ == "CloseStream":
                if s.speech_started:
                    await s.final(from_finalize=False, speech_final=True)
                await ws.close()
                break
            elif typ == "KeepAlive":
                pass
            else:
                await s.send({"type": "Error", "message": f"unknown message type {typ!r}"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        s.seg_gen += 1  # an in-flight partial finishes and is dropped (aborting it can crash sglang-omni, see final())


# ---------------------------------------------------------------- TTS relay
@app.websocket("/v1/tts")
async def tts_ws(ws: WebSocket):
    if not _authorized(ws.headers, ws.query_params):
        await ws.close(code=4401)
        return
    params = {k: v for k, v in ws.query_params.items() if k != "token"}
    url = f"{TTS_URL}/v1/tts" + (f"?{urlencode(params)}" if params else "")
    await ws.accept()
    try:
        up = await websockets.connect(url, max_size=None, ping_interval=None, compression=None, open_timeout=10)
    except Exception as e:  # noqa: BLE001
        await ws.send_text(json.dumps({"type": "Error", "message": f"tts backend unavailable: {e}"}))
        await ws.close()
        return

    async def down():
        async for m in up:
            if isinstance(m, bytes):
                await ws.send_bytes(m)
            else:
                await ws.send_text(m)

    async def upstream():
        while True:
            m = await ws.receive()
            if m["type"] == "websocket.disconnect":
                return
            if m.get("text") is not None:
                await up.send(m["text"])
            elif m.get("bytes") is not None:
                await up.send(m["bytes"])

    t_down, t_up = asyncio.create_task(down()), asyncio.create_task(upstream())
    try:
        await asyncio.wait({t_down, t_up}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in (t_down, t_up):
            t.cancel()
        await up.close()
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


@app.post("/v1/align")
async def align(request: Request):
    """Word timings for known text on 16 kHz 16-bit mono PCM (the body); query: text, language.
    The Qwen3-TTS worker times its own output with it, as that model gives no word timestamps."""
    if aligner is None:
        return JSONResponse({"error": "no aligner"}, status_code=503)
    text = request.query_params.get("text", "")
    pcm = np.frombuffer(await request.body(), dtype="<i2")
    if not text.strip() or len(pcm) == 0:
        return {"items": [], "align_ms": 0}
    items, ms = await asyncio.to_thread(_align_sync, pcm, text, request.query_params.get("language", "English"))
    return {"items": [{"text": w, "start": s, "end": e} for w, s, e in items], "align_ms": round(ms)}


@app.get("/v1/tts/config")
async def tts_config():
    try:
        r = await http.get(re.sub(r"^ws", "http", TTS_URL) + "/config", timeout=5)
        return JSONResponse(r.json(), status_code=r.status_code)
    except (httpx.HTTPError, ValueError) as e:
        return JSONResponse({"error": f"tts backend unavailable: {e}"}, status_code=502)


# ---------------------------------------------------------------- TTS engine swap
# Kyutai and Qwen3-TTS don't fit on the L4 together with ASR and the LLM: the page picks one, and the box swaps the
# worker on port 8102 (setup.sh tts kyutai|qwen; the rest stays up). The page polls GET while it runs.
GPUBOX_ROOT = os.environ.get("GPUBOX_ROOT") or ("/content" if os.path.isdir("/content") else os.path.expanduser("~/gpubox"))
SETUP_SH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup.sh")
ENGINES = ("kyutai", "qwen")
swap = {"engine": None, "state": "ready", "t0": 0.0, "error": None, "proc": None}
swap_s = {"kyutai": 40.0, "qwen": 90.0}   # how long a swap took last time: the page's estimate


def _engine_now():
    try:
        with open(os.path.join(GPUBOX_ROOT, "tts.engine")) as f:
            e = f.read().strip()
        return e if e in ENGINES else "kyutai"
    except OSError:
        return "kyutai"


def _swap_status():
    p = swap["proc"]
    if p is not None and p.returncode is not None:
        swap["proc"] = None
        if p.returncode == 0:
            swap["state"] = "ready"
            swap_s[swap["engine"]] = time.time() - swap["t0"]
        else:
            swap["state"], swap["error"] = "failed", f"setup.sh tts {swap['engine']} exited {p.returncode} (see /content/logs)"
    switching = swap["state"] == "switching"
    return {"engine": swap["engine"] if switching else _engine_now(), "state": swap["state"], "error": swap["error"],
            "elapsed_s": round(time.time() - swap["t0"], 1) if switching else None,
            "estimate_s": round(swap_s.get(swap["engine"] or "", 60.0)) if switching else None}


@app.get("/v1/tts/engine")
async def tts_engine():
    return _swap_status()


@app.post("/v1/tts/engine")
async def tts_engine_set(request: Request):
    want = request.query_params.get("engine", "")
    if want not in ENGINES:
        return JSONResponse({"error": f"engine must be one of {list(ENGINES)}"}, status_code=400)
    st = _swap_status()
    if st["state"] == "switching" or (want == st["engine"] and st["state"] == "ready"):
        return st
    swap.update(engine=want, state="switching", t0=time.time(), error=None)
    log_path = os.path.join(GPUBOX_ROOT, "logs", "tts-swap.log")
    out = open(log_path, "ab") if os.path.isdir(os.path.dirname(log_path)) else asyncio.subprocess.DEVNULL
    # Its own session: the worker it starts outlives this request (and a gateway restart).
    swap["proc"] = await asyncio.create_subprocess_exec("bash", SETUP_SH, "tts", want, cwd=os.path.dirname(SETUP_SH),
                                                         stdout=out, stderr=out, start_new_session=True)
    asyncio.get_running_loop().create_task(swap["proc"].wait())
    log.info("tts engine -> %s", want)
    return _swap_status()


# ---------------------------------------------------------------- LLM pass-through
LLM_TIMEOUT = httpx.Timeout(600, connect=5)
_HOP = {"content-length", "transfer-encoding", "connection", "keep-alive", "content-encoding"}


async def _llm_proxy(request: Request, path: str):
    body = await request.body() if request.method == "POST" else None
    headers = {"content-type": request.headers.get("content-type", "application/json")}
    try:
        up = await http.send(http.build_request(request.method, LLM_URL + path, content=body, headers=headers,
                                                timeout=LLM_TIMEOUT), stream=True)
    except httpx.HTTPError as e:
        return JSONResponse({"error": f"llm unavailable: {e}"}, status_code=502)
    out_headers = {k: v for k, v in up.headers.items() if k.lower() not in _HOP}
    if "text/event-stream" not in up.headers.get("content-type", ""):
        data = await up.aread()
        await up.aclose()
        return Response(data, status_code=up.status_code, headers=out_headers)

    async def relay():
        try:  # cancelled when the client goes away -> closing upstream makes llama-server stop generating
            async for chunk in up.aiter_raw():
                yield chunk
        finally:
            await up.aclose()
    out_headers["cache-control"] = "no-cache"
    out_headers["x-accel-buffering"] = "no"
    return StreamingResponse(relay(), status_code=up.status_code, headers=out_headers)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _llm_proxy(request, "/v1/chat/completions")


@app.get("/v1/models")
async def models(request: Request):
    return await _llm_proxy(request, "/v1/models")


@app.on_event("startup")
async def _startup():
    global http
    http = httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_keepalive_connections=32, max_connections=64))
    await asyncio.to_thread(load_aligner)
    # warm the ASR path
    try:
        await transcribe((np.random.RandomState(0).randn(SR) * 300).astype(np.int16), "en", None)
    except Exception as e:  # noqa: BLE001
        log.warning("asr warmup failed: %s", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not TOKEN:
        raise SystemExit("GPUBOX_TOKEN is not set")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", ws_max_size=2**24, ws_ping_interval=20, ws_ping_timeout=60)


if __name__ == "__main__":
    main()
