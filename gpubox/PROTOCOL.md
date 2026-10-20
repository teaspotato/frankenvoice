# GPU box protocol

One HTTP/WebSocket server on the GPU box (`gpubox/server.py`). The Frankenvoice server connects
to it, adding `Authorization: Bearer <GPUBOX_TOKEN>`. The server
rejects any request without that token (`?token=` also accepted, for testing). All JSON
messages are text frames; audio is binary frames.

## `GET /health`

Returns `{"ok": true, "asr": "<model>", "tts": "<model>"}`.

## ASR: `WS /v1/asr`

Query: `language` (`English`, default), `sample_rate` (16000), `context` (optional
hotwords / topic text passed to the model as context), `partial_ms` (250).

Client to server:
- binary: 16-bit little-endian mono PCM at `sample_rate`.
- `{"type": "Finalize"}`: transcribe everything since the last final now and send it as a
  final (`from_finalize: true`, `speech_final: true`). The stream stays open.
- `{"type": "CloseStream"}`: send a last final if there is speech pending, then close.

Server to client (times are seconds from the first sample received on this connection):
- `{"type": "Ready", "model": "..."}` once the connection is accepted.
- `{"type": "SpeechStarted", "at": t}` when the server's energy gate first hears speech
  in a segment.
- `{"type": "Results", "is_final": bool, "speech_final": bool, "from_finalize": bool,
  "start": t, "end": t, "text": "...", "words": [{"word", "start", "end"}],
  "decode_ms": n}`. Partials (`is_final: false`) come while speech continues, at most
  every `partial_ms` of new audio, each one a re-decode of the whole segment so far.
  A final's words are timed by the forced aligner; a partial's are spread evenly over
  the voiced part of the segment. Finals also carry `align_ms`. A Finalize on a segment
  with no voiced audio returns an empty final at once.
- `{"type": "UtteranceEnd", "at": t}` 1 s after the last final's last word when no
  speech followed.
- `{"type": "Error", "message": "..."}`.

## TTS: `WS /v1/tts`

One connection per reply. Query: `voice` (Kyutai voice path, optional), `cfg` (optional).

Client to server:
- `{"type": "Speak", "text": "..."}`: more text of the reply, as it streams.
- `{"type": "Flush"}`: marks a segment boundary after the text sent so far. The server
  answers each Flush with one `Flushed` once that segment's audio has all been sent.
  It does not end the utterance: the model keeps speaking across segments.
- `{"type": "Close"}`: end of text. The server speaks what is left, sends the remaining
  `Flushed`, then `{"type": "Done"}`, and closes.

Server to client:
- `{"type": "Ready", "sample_rate": 24000}`.
- binary: 16-bit little-endian mono PCM at 24 kHz, in order.
- `{"type": "Words", "words": [{"word", "start", "end"}]}`: word timings in seconds
  from this connection's first audio sample, sent as words are spoken.
- `{"type": "Flushed"}`, `{"type": "Done", "stats": {...}}`, `{"type": "Error", "message": "..."}`.

The model needs the current word plus 2 words of lookahead, so it waits until 3 words are
queued (or Close has come): a pause in the text pauses the audio, and the last words of a
Flushed segment only come out once more text or Close arrives. A word's `end` is the next
word's start. Closing the socket early cancels generation; 30 s with no text and no Close
counts as Close.
