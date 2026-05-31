from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .audio import AudioRecordResult, record_input_wav
from .config import Config
from .wake import WakeAudioTest, score_wake_audio


@dataclass(frozen=True)
class WakeCalibrationResult:
    recording: AudioRecordResult
    wake: WakeAudioTest
    min_rms: float
    min_peak: int

    @property
    def level_ok(self) -> bool:
        return self.recording.rms >= self.min_rms and self.recording.peak >= self.min_peak

    @property
    def ok(self) -> bool:
        return self.level_ok and self.wake.hit

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "level_ok": self.level_ok,
            "recording": self.recording.to_dict(),
            "wake": self.wake.to_dict(),
            "min_rms": self.min_rms,
            "min_peak": self.min_peak,
            "verdict": self.verdict,
        }

    @property
    def verdict(self) -> str:
        if self.ok:
            return "wake detected with sufficient input level"
        if not self.level_ok:
            return "input level is too low for a reliable live wake proof"
        if self.wake.max_score < self.wake.threshold * 0.2:
            return "input level is sufficient, but wake score is far below threshold; retrain or replace the wake model instead of lowering sensitivity"
        return "input level is sufficient, but wake score did not reach threshold"


def calibrate_wake(
    config: Config,
    wav_path: Path,
    seconds: float,
    threshold: float | None = None,
    min_rms: float = 1000.0,
    min_peak: int = 4000,
) -> WakeCalibrationResult:
    recording = record_input_wav(config, wav_path, seconds=seconds)
    wake = score_wake_audio(config, wav_path, threshold=threshold)
    return WakeCalibrationResult(recording=recording, wake=wake, min_rms=min_rms, min_peak=min_peak)
