"""Tests for ONNX streaming cancellation, chunk cap, cancellable short-cue babble guard,
backend gating in V3TurboVieNeuTTS, and thread-safe lazy denoiser loading."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vieneu._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine
from vieneu.v3turbo import V3TurboVieNeuTTS
from vieneu_utils.core_utils import CODEC_SAMPLES_PER_FRAME


def _make_burst_wav(n_bursts: int, sr: int = 48_000) -> np.ndarray:
    """Create a synthetic 48 kHz mono waveform with ``n_bursts`` distinct speech bursts."""
    seg_on = np.full(int(sr * 0.08), 0.5, dtype=np.float32)
    seg_off = np.zeros(int(sr * 0.12), dtype=np.float32)
    parts = [seg_off]
    for _ in range(n_bursts):
        parts.append(seg_on)
        parts.append(seg_off)
    return np.concatenate(parts)


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return SimpleNamespace(ids=[1, 2])


class _FakeSession:
    def __init__(self, handler):
        self.handler = handler
        self.calls = 0

    def run(self, _out, feed):
        self.calls += 1
        return self.handler(self.calls, feed)


def _make_fake_engine(
    *,
    eos_on_frame: int = 6,
    on_pre=None,
    on_ac=None,
    on_dec=None,
    on_stream_decode=None,
    on_full_decode=None,
):
    """Build an ``OnnxV3LiteEngine`` instance with deterministic fake ONNX sessions."""
    eng = OnnxV3LiteEngine.__new__(OnnxV3LiteEngine)
    eng._lock = threading.RLock()
    eng.n_vq = 2
    eng.hidden = 4
    eng.L = 1
    eng.L_loc = 1
    eng.nH_loc = 1
    eng.hd_loc = 4
    eng.audio_pad = 0
    eng.tps = 1
    eng.tpe = 2
    eng.sgs = 3
    eng.eos_speech = 4
    eng.ref_slot = 5
    eng.text_vocab = 8
    eng.default_style_id = 6
    eng.use_speaker_embedding = False
    eng.xvec_w = None
    eng.tokenizer = _FakeTokenizer()
    eng.text_emb = np.eye(8, 4, dtype=np.float32)
    # Make slot0 dot text_emb peak at index 4 (eos_speech) when slot0 == [0,0,0,1]
    eng.text_emb[4] = np.array([0.0, 0.0, 0.0, 5.0], dtype=np.float32)
    eng.text_emb[3] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    eng.audio_emb = np.ones((2, 4, 4), dtype=np.float32)
    eng.speaker_encoder = None
    eng._denoiser = None
    eng._denoiser_loaded = False
    eng._denoiser_lock = threading.Lock()

    frame_state = {"attempt": 0, "frame_in_attempt": 0}

    def pre_handler(call_n, feed):
        frame_state["attempt"] += 1
        frame_state["frame_in_attempt"] = 0
        if on_pre:
            on_pre(call_n, frame_state["attempt"], feed)
        h = np.zeros((1, 3, 4), dtype=np.float32)
        pk = np.zeros((1, 1, 3, 4), dtype=np.float32)
        pv = np.zeros((1, 1, 3, 4), dtype=np.float32)
        return [h, pk, pv]

    def ac_handler(call_n, feed):
        # First sub-call of each frame has token_emb shape (1, 2, H).
        is_first_sub = feed["token_emb"].shape[1] == 2
        if is_first_sub:
            frame_state["frame_in_attempt"] += 1
        if on_ac:
            on_ac(call_n, frame_state["attempt"], frame_state["frame_in_attempt"], is_first_sub)
        eos_now = frame_state["frame_in_attempt"] >= eos_on_frame
        if is_first_sub:
            hidden = np.zeros((1, 2, 4), dtype=np.float32)
            hidden[0, 0] = np.array([0.0, 0.0, 0.0, 1.0] if eos_now else [1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            hidden = np.zeros((1, 1, 4), dtype=np.float32)
        pk = np.zeros((1, 1, 1, 4), dtype=np.float32)
        pv = np.zeros((1, 1, 1, 4), dtype=np.float32)
        return [hidden, pk, pv]

    def dec_handler(call_n, feed):
        if on_dec:
            on_dec(call_n, frame_state["attempt"], frame_state["frame_in_attempt"])
        h = np.zeros((1, 1, 4), dtype=np.float32)
        pk = np.zeros((1, 1, 1, 4), dtype=np.float32)
        pv = np.zeros((1, 1, 1, 4), dtype=np.float32)
        return [h, pk, pv]

    decoded_chunk_sizes: list[int] = []

    def step_handler(call_n, feed):
        k = int(feed["audio_code_lengths"][0])
        decoded_chunk_sizes.append(k)
        if on_stream_decode:
            on_stream_decode(call_n, k)
        n_samp = k * CODEC_SAMPLES_PER_FRAME
        audio = np.full((1, 1, n_samp), 0.1, dtype=np.float32)
        return [audio, np.array([n_samp], dtype=np.int32)]

    def full_decode_handler(call_n, feed):
        k = int(feed["audio_code_lengths"][0])
        if on_full_decode:
            wav = on_full_decode(call_n, frame_state["attempt"], k)
            if wav is not None:
                return [wav[None, None, :]]
        n_samp = k * CODEC_SAMPLES_PER_FRAME
        return [np.full((1, 1, n_samp), 0.1, dtype=np.float32)]

    eng.sess_pre = _FakeSession(pre_handler)
    eng.sess_ac = _FakeSession(ac_handler)
    eng.sess_dec = _FakeSession(dec_handler)
    eng.sess_codec_step = _FakeSession(step_handler)
    eng.sess_codec_dec = _FakeSession(full_decode_handler)
    eng._codec_stream_spec = {"transformer_offsets": [], "attention_caches": []}
    eng._codec_step_out_names = ["audio", "audio_lengths"]
    eng._decoded_chunk_sizes = decoded_chunk_sizes
    return eng


def test_cancel_before_prefill_runs_zero_onnx_calls():
    eng = _make_fake_engine(eos_on_frame=6)
    ev = threading.Event()
    ev.set()

    out = list(eng.infer_stream(phonemes="a b c d e", cancel_event=ev, chunk_frames=8))
    assert out == []
    assert eng.sess_pre.calls == 0
    assert eng.sess_ac.calls == 0
    assert eng.sess_dec.calls == 0
    assert eng.sess_codec_step.calls == 0


def test_cancel_during_prefill_aborts_before_first_frame():
    ev = threading.Event()
    eng = _make_fake_engine(eos_on_frame=10, on_pre=lambda *_args: ev.set())

    out = list(eng.infer_stream(phonemes="a b c d e", cancel_event=ev, chunk_frames=8))
    assert out == []
    assert eng.sess_pre.calls == 1
    assert eng.sess_ac.calls == 0
    assert eng.sess_dec.calls == 0
    assert eng.sess_codec_step.calls == 0


def test_cancel_during_short_cue_attempt1_yields_zero_output():
    ev = threading.Event()

    def on_ac(_call_n, attempt, frame_n, is_first_sub):
        if attempt == 1 and frame_n == 2 and is_first_sub:
            ev.set()

    eng = _make_fake_engine(eos_on_frame=6, on_ac=on_ac)
    # Single-syllable phoneme qualifies for short-cue babble guard when stream_babble_retries=1.
    out = list(eng.infer_stream(
        phonemes="vˈəŋ",
        cancel_event=ev,
        stream_babble_retries=1,
        frame_cap=True,
    ))
    assert out == []
    assert eng.sess_pre.calls == 1
    assert eng.sess_dec.calls == 1           # stopped on frame 2 before frame 2's sess_dec
    assert eng.sess_codec_dec.calls == 0
    assert eng.sess_codec_step.calls == 0


def test_cancel_during_short_cue_babble_retry_aborts_without_yielding():
    ev = threading.Event()

    def on_full_decode(call_n, attempt, _k):
        # Attempt 1 produces 2 distinct bursts for a 1-syllable cue -> babble suspect triggers retry!
        if attempt == 1:
            return _make_burst_wav(2)
        return _make_burst_wav(1)

    def on_ac(_call_n, attempt, frame_n, is_first_sub):
        if attempt == 2 and frame_n == 1 and is_first_sub:
            ev.set()

    eng = _make_fake_engine(eos_on_frame=5, on_ac=on_ac, on_full_decode=on_full_decode)
    out = list(eng.infer_stream(
        phonemes="vˈəŋ",
        cancel_event=ev,
        stream_babble_retries=1,
        frame_cap=True,
    ))
    assert out == []
    assert eng.sess_pre.calls == 2           # attempt 1 + cancelled attempt 2
    assert eng.sess_codec_dec.calls == 1     # only attempt 1 decoded; attempt 2 cancelled before decode


def test_short_cue_babble_guard_retries_and_returns_clean_attempt():
    wav_bad = _make_burst_wav(2)
    wav_good = _make_burst_wav(1)

    def on_full_decode(_call_n, attempt, _k):
        return wav_bad if attempt == 1 else wav_good

    eng = _make_fake_engine(eos_on_frame=5, on_full_decode=on_full_decode)
    chunks = list(eng.infer_stream(
        phonemes="vˈəŋ",
        stream_babble_retries=1,
        frame_cap=True,
    ))
    assert len(chunks) == 1
    assert np.array_equal(chunks[0], wav_good)
    assert eng.sess_pre.calls == 2
    assert eng.sess_codec_dec.calls == 2
    assert eng.sess_codec_step.calls == 0

    # Cue-only (<|emotion_0|>) also uses the short-cue guard.
    eng_cue = _make_fake_engine(eos_on_frame=13, on_full_decode=lambda _c, a, _k: wav_good)
    cue_chunks = list(eng_cue.infer_stream(
        phonemes="<|emotion_0|>",
        stream_babble_retries=1,
        frame_cap=True,
    ))
    assert len(cue_chunks) == 1
    assert eng_cue.sess_codec_dec.calls >= 1
    assert eng_cue.sess_codec_step.calls == 0

    # Longer (>3 syllable) chunk and emotion+text chunk bypass short-cue guard and stream via decode_step.
    eng_long = _make_fake_engine(eos_on_frame=6)
    long_chunks = list(eng_long.infer_stream(
        phonemes="sˈin tɕˈaːw mˈɔj ŋˈɨəj",
        stream_babble_retries=1,
        frame_cap=True,
        chunk_frames=8,
    ))
    assert len(long_chunks) >= 1
    assert eng_long.sess_codec_dec.calls == 0
    assert eng_long.sess_codec_step.calls >= 1

    eng_emo = _make_fake_engine(eos_on_frame=6)
    emo_chunks = list(eng_emo.infer_stream(
        phonemes="<|emotion_0|> vˈəŋ",
        stream_babble_retries=1,
        frame_cap=True,
        chunk_frames=8,
    ))
    assert len(emo_chunks) >= 1
    assert eng_emo.sess_codec_dec.calls == 0
    assert eng_emo.sess_codec_step.calls >= 1


def test_cancel_after_first_pcm_and_chunk_frames_cap():
    ev = threading.Event()
    chunks_seen = []

    def on_ac(_call_n, _attempt, frame_n, is_first_sub):
        if frame_n == 6 and is_first_sub:
            ev.set()

    eng = _make_fake_engine(eos_on_frame=30, on_ac=on_ac)
    for chunk in eng.infer_stream(phonemes="a b c d e f g", cancel_event=ev, chunk_frames=8):
        chunks_seen.append(chunk)

    assert len(chunks_seen) == 1             # lead-in 4 frames decoded and yielded, cancelled at frame 6
    assert eng._decoded_chunk_sizes == [4]
    assert eng.sess_codec_step.calls == 1

    # Without cancellation, chunk_frames=8 caps every decode_step call at <= 8 frames.
    eng_cap = _make_fake_engine(eos_on_frame=40)
    all_chunks = list(eng_cap.infer_stream(
        phonemes="a b c d e f g h i j k l m n o p q r s t",
        chunk_frames=8,
        frame_cap=False,
    ))
    assert len(all_chunks) >= 5
    assert max(eng_cap._decoded_chunk_sizes) <= 8
    assert sum(eng_cap._decoded_chunk_sizes) == 40


def test_v3turbo_forwards_onnx_stream_args_only_on_onnx_backend(monkeypatch):
    import vieneu.v3turbo as v3mod

    monkeypatch.setattr(v3mod, "normalize_to_chunks_v3_with_gaps", lambda text, max_chars=256: ([text], []))
    monkeypatch.setattr(v3mod, "phonemize_text_with_emotions", lambda chunk: "a b c d")

    for backend in ("onnx", "pytorch"):
        seen_kwargs = []

        class StubEngine:
            def infer_stream(self, **kw):
                seen_kwargs.append(kw)
                yield np.zeros(480, dtype=np.float32)

        tts = V3TurboVieNeuTTS.__new__(V3TurboVieNeuTTS)
        tts.backend = backend
        tts.engine = StubEngine()
        tts.sample_rate = 48_000
        tts._preset_voices = {"v1": {"speaker_emb": np.ones(4, dtype=np.float32), "codes": None}}
        tts._voice_aliases = {}
        tts._default_voice = "v1"
        tts._get_stream_scheduler = lambda: None

        ev = threading.Event()
        out = list(tts.infer_stream(
            "Xin chào",
            apply_watermark=False,
            chunk_frames=8,
            cancel_event=ev,
            stream_babble_retries=1,
        ))
        assert len(out) == 1
        assert len(seen_kwargs) == 1
        if backend == "onnx":
            assert seen_kwargs[0]["chunk_frames"] == 8
            assert seen_kwargs[0]["cancel_event"] is ev
            assert seen_kwargs[0]["stream_babble_retries"] == 1
        else:
            assert "chunk_frames" not in seen_kwargs[0]
            assert "cancel_event" not in seen_kwargs[0]
            assert "stream_babble_retries" not in seen_kwargs[0]

    # GPU scheduler path stays completely untouched.
    sched_calls = []

    class FakeHandle:
        def __init__(self):
            self.gen_done = threading.Event()
            self.gen_done.set()
            self.closed = False

        def __iter__(self):
            yield np.zeros(480, dtype=np.float32)

        def close(self):
            self.closed = True

    class FakeSched:
        def submit(self, **kw):
            sched_calls.append(kw)
            return FakeHandle()

    tts_gpu = V3TurboVieNeuTTS.__new__(V3TurboVieNeuTTS)
    tts_gpu.backend = "pytorch"
    tts_gpu.engine = SimpleNamespace()
    tts_gpu.sample_rate = 48_000
    tts_gpu._preset_voices = {"v1": {"speaker_emb": np.ones(4, dtype=np.float32), "codes": None}}
    tts_gpu._voice_aliases = {}
    tts_gpu._default_voice = "v1"
    tts_gpu._get_stream_scheduler = lambda: FakeSched()

    gpu_out = list(tts_gpu.infer_stream(
        "Xin chào",
        apply_watermark=False,
        chunk_frames=8,
        cancel_event=threading.Event(),
        stream_babble_retries=1,
    ))
    assert len(gpu_out) == 1
    assert len(sched_calls) == 1
    assert "chunk_frames" not in sched_calls[0]
    assert "cancel_event" not in sched_calls[0]
    assert "stream_babble_retries" not in sched_calls[0]


def test_lazy_denoiser_startup_denoise_false_concurrency_and_missing_asset(tmp_path):
    eng = _make_fake_engine()
    load_calls = 0
    sentinel = SimpleNamespace(denoise=lambda wav, sr: wav)

    def counting_load():
        nonlocal load_calls
        load_calls += 1
        time.sleep(0.03)
        return sentinel

    eng._load_denoiser = counting_load
    eng._encode_ref_wav = lambda wav, sr: np.zeros((2, 2), dtype=np.int64)

    # 1. Startup / initial state does not load denoiser.
    assert eng._denoiser_loaded is False
    assert load_calls == 0

    # 2. prepare_reference(..., denoise=False) does not touch denoiser.
    emb, codes = eng.prepare_reference(np.zeros(4800, dtype=np.float32), sr=48_000, denoise=False)
    assert emb is None and codes is not None
    assert eng._denoiser_loaded is False
    assert load_calls == 0

    # 3. Concurrent first accesses load denoiser exactly once.
    n_threads = 8
    barrier = threading.Barrier(n_threads)
    results = [None] * n_threads

    def worker(idx: int):
        barrier.wait()
        results[idx] = eng.denoiser

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert load_calls == 1
    assert all(r is sentinel for r in results)

    # 4. Missing asset returns None and V3TurboVieNeuTTS.denoise() still raises RuntimeError.
    eng_missing = _make_fake_engine()
    eng_missing.checkpoint_path = str(tmp_path)
    eng_missing._so = None
    assert eng_missing.denoiser is None
    assert eng_missing._denoiser_loaded is True

    tts = V3TurboVieNeuTTS.__new__(V3TurboVieNeuTTS)
    tts.engine = eng_missing
    with pytest.raises(RuntimeError, match="Denoiser"):
        tts.denoise(str(tmp_path / "unused.wav"))


def test_lazy_denoiser_loads_local_verified_asset_offline(monkeypatch):
    hf_home = Path(os.environ.get("HF_HOME", "/models"))
    candidates = list(hf_home.glob("hub/models--pnnbao-ump--VieNeu-TTS-v3-Turbo/snapshots/*/denoiser.onnx"))
    if not candidates:
        pytest.skip("Local denoiser.onnx model cache not mounted")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    # Real OnnxV3LiteEngine startup must NOT load denoiser.onnx eagerly, and must load it
    # lazily from the offline HF cache via repo ID on first access.
    eng = OnnxV3LiteEngine(checkpoint_path="pnnbao-ump/VieNeu-TTS-v3-Turbo", threads=1)
    assert eng._denoiser_loaded is False
    assert eng._denoiser is None

    den = eng.denoiser
    assert den is not None
    assert eng._denoiser_loaded is True
    assert eng.denoiser is den
