"""apps/openai_speech.py with a stand-in model: request limits, voice enrolment,
stream slots, health, auth and the listening address. ``Engine.__init__`` runs
for real."""
import asyncio
import gc
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps import openai_speech as api


class FakeTTS:
    backend = "onnx"

    def __init__(self):
        self._preset_voices = {"Hải Đăng": {}, "Mai Anh": {}}
        self._voice_aliases = {"Minh Quân": "Hải Đăng"}
        self.stream_calls = []
        self.enrolled = []
        self.stream_len = 1        # chunks per stream
        self.streams_closed = 0    # streams whose generator ran its finally

    def resolve_voice_name(self, name):
        if name in self._preset_voices:
            return name
        target = self._voice_aliases.get(name)
        return target if target in self._preset_voices else None

    def infer_stream(self, text, **kw):
        self.stream_calls.append(kw)
        try:
            for _ in range(self.stream_len):
                yield np.zeros(4800, dtype=np.float32)
        finally:
            self.streams_closed += 1

    def add_voice(self, name, path, **kw):
        self.enrolled.append(name)
        self._preset_voices[name] = {}


@pytest.fixture
def eng(monkeypatch):
    for var in ("VIENEU_API_KEY", "VIENEU_MAX_STREAMS", "VIENEU_QUEUE",
                "VIENEU_STREAM_CHUNK_FRAMES", "VIENEU_STREAM_BABBLE_RETRIES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(api, "Vieneu", lambda **kw: FakeTTS())
    e = api.Engine()
    monkeypatch.setattr(api, "ENGINE", e)
    return e


@pytest.fixture
def client(eng):
    with TestClient(api.app) as c:
        yield c


def _speech(client, **body):
    return client.post("/v1/audio/speech", json={"input": "Xin chào.", **body})


def test_valid_request_streams_wav(client, eng):
    r = _speech(client, voice="Mai Anh", temperature=0.5, top_k=50)
    assert r.status_code == 200
    assert r.content[:4] == b"RIFF" and len(r.content) == 44 + 4800 * 2
    assert eng.tts.stream_calls[-1]["temperature"] == 0.5
    assert eng.tts.stream_calls[-1]["top_k"] == 50
    gc.collect()                               # the drop-time release must be a no-op now
    assert eng.active == 0                     # the stream slot was given back, once


def _serve_until_disconnect(resp, chunks_before_leaving):
    """Run ``resp`` as an ASGI app for a client that leaves after receiving
    ``chunks_before_leaving`` body chunks (0: before the body starts)."""
    async def run():
        left = asyncio.Event()
        sent = 0

        async def receive():
            await left.wait()
            return {"type": "http.disconnect"}

        async def send(msg):
            nonlocal sent
            if msg["type"] == "http.response.body" and msg.get("body"):
                sent += 1
            if sent >= chunks_before_leaving:
                left.set()

        if chunks_before_leaving == 0:
            left.set()
        await resp({"type": "http", "asgi": {"version": "3.0"}, "method": "POST"}, receive, send)
    asyncio.run(run())


@pytest.mark.parametrize("chunks_before_leaving", [0, 2])   # queued / mid-stream
def test_client_gone_frees_slot_without_gc(eng, chunks_before_leaving):
    # Starlette drops a disconnected sync body without closing it, and the
    # abandoned generators sit in a reference cycle: on an idle server their
    # finally could wait minutes for the collector, with every slot taken (429s).
    eng.tts.stream_len = 1000
    gc.disable()
    try:
        resp = api.speech(api.SpeechRequest(input="Xin chào."))
        assert eng.active == 1
        _serve_until_disconnect(resp, chunks_before_leaving)
        assert eng.active == 0
        assert eng._gate.acquire(blocking=False)
        eng._gate.release()
        if chunks_before_leaving:
            # The model's stream was closed too (on GPU: its scheduler row cancelled).
            assert eng.tts.streams_closed == 2      # warm-up + this one
    finally:
        gc.enable()


@pytest.mark.parametrize("field, value", [
    ("top_k", 10 ** 30),          # overflowed the GPU scheduler's int64 slot and killed it
    ("top_k", 0),
    ("temperature", 5.0),
    ("top_p", 0.0),
    ("repetition_penalty", 0.0),
    ("max_chars", 0),             # ZeroDivisionError in the chunker, after the 200 was sent
])
def test_out_of_range_sampling_is_rejected_up_front(client, eng, field, value):
    r = _speech(client, **{field: value})
    assert r.status_code == 400
    assert r.json()["error"]["message"].startswith(f"{field}:")   # OpenAI's error shape
    assert eng.tts.stream_calls == [{"apply_watermark": False}]   # only Engine's warm-up ran


def test_nan_temperature_is_rejected(client, eng):
    # json.loads accepts NaN; it used to reach torch.multinomial. Echoing it back
    # in the error body is what turned FastAPI's default 422 into a 500.
    r = client.post("/v1/audio/speech", content=b'{"input": "x", "temperature": NaN}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and "temperature" in r.json()["error"]["message"]


@pytest.mark.parametrize("name", ["Hải Đăng", "Minh Quân"])   # the default voice, an alias of it
def test_builtin_voice_cannot_be_replaced(client, eng, name):
    r = client.post("/v1/voices", data={"name": name}, files={"file": ("a.wav", b"RIFF", "audio/wav")})
    assert r.status_code == 409
    assert eng.tts.enrolled == []


def test_new_voice_is_enrolled(client, eng):
    r = client.post("/v1/voices", data={"name": "Giọng của tôi"},
                    files={"file": ("me.wav", b"RIFF", "audio/wav")})
    assert r.status_code == 200
    assert eng.tts.enrolled == ["Giọng của tôi"]


def test_dead_scheduler_fails_health_and_new_requests(client, eng):
    assert client.get("/health").status_code == 200
    eng.sched = SimpleNamespace(error=RuntimeError("worker died"))
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "error"
    assert _speech(client).status_code == 503
    assert eng.active == 0 and eng.waiting == 0


def test_api_key_is_enforced_when_set(client, monkeypatch):
    monkeypatch.setenv("VIENEU_API_KEY", "s3cret")
    assert client.get("/v1/voices").status_code == 401
    assert client.get("/v1/voices", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/v1/voices", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_listens_on_loopback_by_default(monkeypatch):
    seen = {}
    monkeypatch.setattr(api.uvicorn, "run", lambda app, **kw: seen.update(kw))
    monkeypatch.delenv("HOST", raising=False)
    api.main()
    assert seen["host"] == "127.0.0.1"


def test_open_public_bind_is_logged(monkeypatch, caplog):
    monkeypatch.setattr(api.uvicorn, "run", lambda app, **kw: None)
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.delenv("VIENEU_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING, logger="vieneu.api"):
        api.main()
    assert "without VIENEU_API_KEY" in caplog.text


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_disconnect_sets_cancel_event_and_releases_slot_across_asgi_versions(eng, spec_version):
    import threading
    import time

    started = threading.Event()
    saw_cancel = threading.Event()
    receive_calls = 0

    def blocking_stream(text, **kw):
        cancel_event = kw.get("cancel_event")
        assert cancel_event is not None
        started.set()
        try:
            # Block until disconnect sets cancel_event (simulates long ONNX prefill/frame).
            if cancel_event.wait(timeout=5.0):
                saw_cancel.set()
                return
            yield np.zeros(4800, dtype=np.float32)
        finally:
            eng.tts.streams_closed += 1

    eng.tts.infer_stream = blocking_stream
    resp = api.speech(api.SpeechRequest(input="Xin chào.", response_format="pcm"))
    assert eng.active == 1

    async def run():
        nonlocal receive_calls
        disconnect_ready = asyncio.Event()

        async def receive():
            nonlocal receive_calls
            receive_calls += 1
            await disconnect_ready.wait()
            return {"type": "http.disconnect"}

        async def send(_msg):
            pass

        async def trigger_disconnect():
            while not started.is_set():
                await asyncio.sleep(0.01)
            disconnect_ready.set()

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": spec_version}, "method": "POST"}
        t0 = time.perf_counter()
        await asyncio.gather(resp(scope, receive, send), trigger_disconnect())
        return time.perf_counter() - t0

    elapsed = asyncio.run(run())
    assert saw_cancel.is_set(), f"ASGI {spec_version}: cancel_event was not set promptly on disconnect"
    assert elapsed < 2.0, f"ASGI {spec_version}: disconnect took too long ({elapsed:.2f}s)"
    assert eng.active == 0
    assert eng._gate.acquire(blocking=False)
    eng._gate.release()
    assert eng.tts.streams_closed == 2   # warm-up + cancelled stream
    assert receive_calls == 1, f"ASGI {spec_version}: expected single receive() consumer, got {receive_calls}"


def test_asgi_24_normal_completion_cancels_disconnect_watcher(eng):
    resp = api.speech(api.SpeechRequest(input="Xin chào.", response_format="pcm"))
    assert eng.active == 1

    async def run():
        never = asyncio.Event()
        bodies = []

        async def receive():
            await never.wait()
            return {"type": "http.disconnect"}

        async def send(msg):
            if msg["type"] == "http.response.body":
                bodies.append(msg.get("body", b""))

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "method": "POST"}
        await asyncio.wait_for(resp(scope, receive, send), timeout=3.0)
        return b"".join(bodies)

    payload = asyncio.run(run())
    assert len(payload) == 4800 * 2
    assert eng.active == 0
    assert eng.tts.streams_closed == 2


def test_asgi_24_send_oserror_propagates_client_disconnect_and_releases_slot(eng):
    from starlette.requests import ClientDisconnect

    eng.tts.stream_len = 5
    resp = api.speech(api.SpeechRequest(input="Xin chào.", response_format="pcm"))
    assert eng.active == 1

    async def run():
        never = asyncio.Event()

        async def receive():
            await never.wait()
            return {"type": "http.disconnect"}

        async def send(msg):
            if msg["type"] == "http.response.body":
                raise OSError("broken pipe")

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "method": "POST"}
        with pytest.raises(ClientDisconnect):
            await resp(scope, receive, send)

    asyncio.run(run())
    assert eng.active == 0
    assert eng._gate.acquire(blocking=False)
    eng._gate.release()


def test_onnx_stream_env_vars_clamped_and_only_forwarded_on_onnx(monkeypatch):
    monkeypatch.setenv("VIENEU_STREAM_CHUNK_FRAMES", "8")
    monkeypatch.setenv("VIENEU_STREAM_BABBLE_RETRIES", "1")
    monkeypatch.setattr(api, "Vieneu", lambda **kw: FakeTTS())
    e_onnx = api.Engine()
    monkeypatch.setattr(api, "ENGINE", e_onnx)
    with TestClient(api.app) as c:
        r = c.post("/v1/audio/speech", json={"input": "Xin chào.", "response_format": "pcm"})
        assert r.status_code == 200
    call = e_onnx.tts.stream_calls[-1]
    assert call["chunk_frames"] == 8
    assert call["stream_babble_retries"] == 1
    assert "cancel_event" in call

    # Out-of-bounds env values clamp safely to 4..25 and 0..2.
    monkeypatch.setenv("VIENEU_STREAM_CHUNK_FRAMES", "999")
    monkeypatch.setenv("VIENEU_STREAM_BABBLE_RETRIES", "99")
    e_hi = api.Engine()
    assert e_hi.stream_chunk_frames == 25
    assert e_hi.stream_babble_retries == 2
    monkeypatch.setenv("VIENEU_STREAM_CHUNK_FRAMES", "1")
    monkeypatch.setenv("VIENEU_STREAM_BABBLE_RETRIES", "-5")
    e_lo = api.Engine()
    assert e_lo.stream_chunk_frames == 4
    assert e_lo.stream_babble_retries == 0

    # PyTorch backend must NOT receive ONNX-only chunk_frames or stream_babble_retries.
    class FakePyTorchTTS(FakeTTS):
        backend = "pytorch"
        def _get_stream_scheduler(self):
            return None

    monkeypatch.setenv("VIENEU_STREAM_CHUNK_FRAMES", "8")
    monkeypatch.setenv("VIENEU_STREAM_BABBLE_RETRIES", "1")
    monkeypatch.setattr(api, "Vieneu", lambda **kw: FakePyTorchTTS())
    e_pt = api.Engine()
    monkeypatch.setattr(api, "ENGINE", e_pt)
    with TestClient(api.app) as c:
        r = c.post("/v1/audio/speech", json={"input": "Xin chào.", "response_format": "pcm"})
        assert r.status_code == 200
    pt_call = e_pt.tts.stream_calls[-1]
    assert "chunk_frames" not in pt_call
    assert "stream_babble_retries" not in pt_call
