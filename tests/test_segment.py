from __future__ import annotations

import wave

from codex_voice_steer.config import load_config
from codex_voice_steer.runtime import SegmentWriter
from codex_voice_steer.segment import AudioFrame, PreRollBuffer, looks_fragmentary, write_wav


def frame(samples: int = 160) -> AudioFrame:
    return AudioFrame(pcm16=b"\0\0" * samples, sample_rate=16000, channels=1)


def test_preroll_keeps_recent_audio_only() -> None:
    buffer = PreRollBuffer(sample_rate=16000, channels=1, max_ms=10)
    buffer.add(frame(100))
    buffer.add(frame(100))
    assert len(buffer.drain()) == 1


def test_write_wav(tmp_path) -> None:
    path = write_wav(tmp_path / "segment.wav", [frame(160), frame(160)], 16000, 1)
    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 320


def test_segment_writer_includes_preroll(tmp_path) -> None:
    writer = SegmentWriter(load_config(path=tmp_path / "missing.toml"))
    writer.add_preroll(frame(160))
    result = writer.write_segment(tmp_path / "turn.wav", [frame(160)])
    assert result.frame_count == 2
    assert result.wav_path.exists()


def test_fragment_heuristic() -> None:
    assert looks_fragmentary("and", ["and"], min_chars=8)
    assert looks_fragmentary("check the logs and", ["and"], min_chars=8)
    assert not looks_fragmentary("check the logs now", ["and"], min_chars=8)
