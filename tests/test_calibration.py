from __future__ import annotations

from pathlib import Path

from codex_voice_steer import calibration
from codex_voice_steer.audio import AudioRecordResult
from codex_voice_steer.calibration import WakeCalibrationResult, calibrate_wake
from codex_voice_steer.config import load_config
from codex_voice_steer.wake import WakeAudioTest


def test_calibration_requires_wake_hit_and_sufficient_input_level() -> None:
    result = WakeCalibrationResult(
        recording=_recording(rms=1200, peak=6000),
        wake=_wake(hit=True),
        min_rms=1000,
        min_peak=4000,
    )
    assert result.ok is True
    assert result.level_ok is True
    assert result.to_dict()["verdict"] == "wake detected with sufficient input level"


def test_calibration_rejects_weak_input_even_when_score_hits() -> None:
    result = WakeCalibrationResult(
        recording=_recording(rms=200, peak=800),
        wake=_wake(hit=True),
        min_rms=1000,
        min_peak=4000,
    )
    assert result.ok is False
    assert result.level_ok is False
    assert "too low" in result.verdict


def test_calibration_records_then_scores_wake_audio(tmp_path, monkeypatch) -> None:
    wav_path = tmp_path / "live.wav"
    seen = {}

    def fake_record(config, path, seconds):
        seen["record"] = (config, path, seconds)
        return _recording(wav_path=path, rms=1200, peak=6000)

    def fake_score(config, path, threshold=None):
        seen["score"] = (config, path, threshold)
        return _wake(wav_path=path, hit=True)

    monkeypatch.setattr(calibration, "record_input_wav", fake_record)
    monkeypatch.setattr(calibration, "score_wake_audio", fake_score)
    config = load_config(path=tmp_path / "missing.toml")

    result = calibrate_wake(config, wav_path, seconds=2.5, threshold=0.4)

    assert result.ok is True
    assert seen["record"] == (config, wav_path, 2.5)
    assert seen["score"] == (config, wav_path, 0.4)


def _recording(wav_path: Path | None = None, rms: float = 0.0, peak: int = 0) -> AudioRecordResult:
    return AudioRecordResult(
        wav_path=wav_path or Path("/tmp/live.wav"),
        sample_rate=16000,
        channels=1,
        samples=16000,
        seconds=1.0,
        device="default",
        rms=rms,
        peak=peak,
        gain_db=0.0,
        clipped_samples=0,
        clipped_ratio=0.0,
    )


def _wake(wav_path: Path | None = None, hit: bool = False) -> WakeAudioTest:
    return WakeAudioTest(
        wav_path=wav_path or Path("/tmp/live.wav"),
        hit=hit,
        max_score=0.6 if hit else 0.1,
        threshold=0.5,
        frame_count=12,
        sample_rate=16000,
        channels=1,
        max_score_time_sec=0.8,
        rms=1200,
        peak=6000,
    )
