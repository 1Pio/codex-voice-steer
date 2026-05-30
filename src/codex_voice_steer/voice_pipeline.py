from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from .config import Config
from .segment import AudioFrame, looks_fragmentary
from .runtime import SegmentWriter
from .stt import SttResult


class WakeDetector(Protocol):
    def predict(self, pcm16_frame: bytes) -> bool: ...


class VadDetector(Protocol):
    def speech_timestamps(self, pcm16: bytes) -> list[dict[str, int]]: ...


class SttBackend(Protocol):
    def transcribe(self, wav_path: Path, timeout_sec: int = 120) -> SttResult: ...


@dataclass(frozen=True)
class VoiceTurnResult:
    status: str
    wav_path: Path | None = None
    transcript: str = ""
    delivered: bool = False
    reason: str = ""


class EndpointCollector:
    def __init__(self, config: Config, vad: VadDetector) -> None:
        self.config = config
        self.vad = vad
        self.frames: list[AudioFrame] = []
        self.sample_rate = int(config.get("audio.sample_rate", 16000))
        self.min_speech_samples = int(self.sample_rate * int(config.get("vad.min_speech_ms", 180)) / 1000)
        self.final_silence_samples = int(self.sample_rate * int(config.get("vad.final_silence_ms", 900)) / 1000)
        self.force_final_samples = int(self.sample_rate * int(config.get("vad.force_final_silence_ms", 3000)) / 1000)
        self.max_samples = int(self.sample_rate * int(config.get("vad.max_utterance_sec", 45)))
        self.post_wake_grace_samples = int(self.sample_rate * int(config.get("audio.post_wake_grace_ms", 250)) / 1000)

    def add(self, frame: AudioFrame) -> bool:
        self.frames.append(frame)
        total_samples = sum(item.samples for item in self.frames)
        pcm = b"".join(item.pcm16 for item in self.frames)
        speech = self.vad.speech_timestamps(pcm)
        if total_samples < self.post_wake_grace_samples:
            return False
        if not speech:
            return total_samples >= self.force_final_samples
        speech_samples = sum(int(item["end"]) - int(item["start"]) for item in speech)
        last_end = int(speech[-1]["end"])
        trailing_silence = total_samples - last_end
        if total_samples >= self.max_samples:
            return True
        return speech_samples >= self.min_speech_samples and trailing_silence >= self.final_silence_samples


class VoicePipeline:
    def __init__(
        self,
        config: Config,
        wake: WakeDetector,
        vad: VadDetector,
        stt: SttBackend,
        deliver_text: Callable[[str], object],
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
        temp_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.wake = wake
        self.vad = vad
        self.stt = stt
        self.deliver_text = deliver_text
        self.event_sink = event_sink or (lambda _event, _fields: None)
        self.temp_dir = temp_dir or Path(tempfile.gettempdir())

    def run_once(self, frames: Iterable[AudioFrame]) -> VoiceTurnResult:
        segment_writer = SegmentWriter(self.config)
        endpoint: EndpointCollector | None = None
        for frame in frames:
            if endpoint is None:
                segment_writer.add_preroll(frame)
                if self.wake.predict(frame.pcm16):
                    self.event_sink("wake_detected", {})
                    endpoint = EndpointCollector(self.config, self.vad)
                continue
            if endpoint.add(frame):
                return self._finalize(segment_writer, endpoint.frames)
        return VoiceTurnResult(status="no_input", reason="capture ended before a complete utterance")

    def _finalize(self, segment_writer: SegmentWriter, speech_frames: list[AudioFrame]) -> VoiceTurnResult:
        wav_path = Path(tempfile.NamedTemporaryFile(prefix="cxv-", suffix=".wav", dir=self.temp_dir, delete=False).name)
        segment_writer.write_segment(wav_path, speech_frames)
        self.event_sink("vad_final", {"wav_path": str(wav_path)})
        transcript = self.stt.transcribe(wav_path).text.strip()
        if not transcript:
            self.event_sink("stt_final", {"transcript": ""})
            return VoiceTurnResult(status="empty_transcript", wav_path=wav_path, reason="STT returned no text")
        trailing_words = list(self.config.get("endpointing.trailing_fragment_words", []))
        if looks_fragmentary(transcript, trailing_words, int(self.config.get("endpointing.min_chars_to_send", 8))):
            transcript = "[cxv: transcript may be fragmentary]\n" + transcript
        self.event_sink("stt_final", {"transcript": transcript})
        self.deliver_text(transcript)
        return VoiceTurnResult(status="delivered", wav_path=wav_path, transcript=transcript, delivered=True)
