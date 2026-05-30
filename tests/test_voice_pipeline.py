from __future__ import annotations

from pathlib import Path

from codex_voice_steer.config import load_config
from codex_voice_steer.segment import AudioFrame
from codex_voice_steer.stt import SttResult
from codex_voice_steer.voice_pipeline import EndpointCollector, VoicePipeline


def frame(samples: int = 1280, value: bytes = b"\1\0") -> AudioFrame:
    return AudioFrame(pcm16=value * samples, sample_rate=16000, channels=1)


class FakeWake:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, _pcm16_frame: bytes) -> bool:
        self.calls += 1
        return self.calls == 2


class FakeVad:
    def speech_timestamps(self, pcm16: bytes) -> list[dict[str, int]]:
        samples = len(pcm16) // 2
        if samples < 3200:
            return []
        return [{"start": 0, "end": 3200}]


class EarlySpeechVad:
    def speech_timestamps(self, pcm16: bytes) -> list[dict[str, int]]:
        samples = len(pcm16) // 2
        return [{"start": 0, "end": min(samples, 3200)}]


class FakeStt:
    def transcribe(self, wav_path: Path, timeout_sec: int = 120) -> SttResult:
        return SttResult(text="check status now", command=["fake", str(wav_path)])


class FragmentStt:
    def transcribe(self, wav_path: Path, timeout_sec: int = 120) -> SttResult:
        return SttResult(text="check logs and", command=["fake", str(wav_path)])


def test_endpoint_finalizes_after_final_silence(tmp_path) -> None:
    config = load_config(path=tmp_path / "missing.toml")
    endpoint = EndpointCollector(config, FakeVad())
    assert endpoint.add(frame()) is False
    for _ in range(13):
        done = endpoint.add(frame(value=b"\0\0"))
    assert done is True


def test_endpoint_uses_min_silence_as_floor(tmp_path) -> None:
    config = load_config(
        overrides={
            "audio": {"post_wake_grace_ms": 0},
            "vad": {"min_speech_ms": 80, "min_silence_ms": 450, "final_silence_ms": 80},
        },
        path=tmp_path / "missing.toml",
    )
    endpoint = EndpointCollector(config, FakeVad())
    assert endpoint.add(frame()) is False
    assert endpoint.add(frame()) is False
    assert endpoint.add(frame()) is False
    for _ in range(5):
        assert endpoint.add(frame(value=b"\0\0")) is False
    assert endpoint.add(frame(value=b"\0\0")) is True


def test_voice_pipeline_wake_vad_stt_delivery(tmp_path) -> None:
    config = load_config(path=tmp_path / "missing.toml")
    delivered: list[str] = []
    events: list[tuple[str, dict[str, object]]] = []
    pipeline = VoicePipeline(
        config,
        FakeWake(),
        FakeVad(),
        FakeStt(),
        delivered.append,
        event_sink=lambda event, fields: events.append((event, fields)),
        temp_dir=tmp_path,
    )
    frames = [frame(), frame(), *[frame(value=b"\0\0") for _ in range(14)]]
    result = pipeline.run_once(frames)
    assert result.status == "delivered"
    assert result.wav_path is not None and result.wav_path.exists()
    assert delivered == ["check status now"]
    assert [event for event, _fields in events] == ["wake_detected", "vad_final", "stt_final"]


def test_voice_pipeline_can_disable_fragment_prompt(tmp_path) -> None:
    config = load_config(overrides={"endpointing": {"ask_if_fragment": False}}, path=tmp_path / "missing.toml")
    delivered: list[str] = []
    pipeline = VoicePipeline(config, FakeWake(), FakeVad(), FragmentStt(), delivered.append, temp_dir=tmp_path)

    result = pipeline.run_once([frame(), frame(), *[frame(value=b"\0\0") for _ in range(14)]])

    assert result.status == "delivered"
    assert delivered == ["check logs and"]


def test_endpoint_honors_post_wake_grace_before_finalizing(tmp_path) -> None:
    config = load_config(
        overrides={
            "audio": {"post_wake_grace_ms": 250},
            "vad": {"min_speech_ms": 80, "min_silence_ms": 80, "final_silence_ms": 80},
        },
        path=tmp_path / "missing.toml",
    )
    endpoint = EndpointCollector(config, EarlySpeechVad())
    assert endpoint.add(frame()) is False
    assert endpoint.add(frame(value=b"\0\0")) is False
    assert endpoint.add(frame(value=b"\0\0")) is False
    assert endpoint.add(frame(value=b"\0\0")) is True
