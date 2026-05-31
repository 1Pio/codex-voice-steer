from __future__ import annotations

import json
import math
import re
import contextlib
import select
import sys
import termios
import time
import tty
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, TextIO

from .audio import MicCapture, pcm16_level_stats
from .config import Config
from .wake import OpenWakeWordDetector, score_wake_audio

LABELS = ("positive", "negative", "noise")
SCARLETT_PROMPTS = [
    "scarlett",
    "scarlett",
    "hey scarlett",
    "ok scarlett",
    "scarlett please",
    "scarlett now",
    "scarlett",
]
HARD_NEGATIVE_PROMPTS = [
    "starlet",
    "Charlotte",
    "scarlet fever",
    "let",
    "start it",
    "scale it",
]
ENVIRONMENTAL_NEGATIVE_PROMPTS = [
    "keyboard",
    "fan",
    "room noise",
    "silence",
    "handling noise",
]


@dataclass(frozen=True)
class SampleTake:
    path: Path
    label: str
    prompt: str
    tag: str
    sample_rate: int
    channels: int
    samples: int
    seconds: float
    rms: float
    peak: int
    clipped_samples: int
    clipped_ratio: float
    gain_db: float
    device: str
    created_at: str
    command: str
    mode: str
    weak: bool

    def to_metadata(self, dataset_dir: Path) -> dict[str, object]:
        try:
            path = self.path.relative_to(dataset_dir)
        except ValueError:
            path = self.path
        return {
            "path": str(path),
            "label": self.label,
            "prompt": self.prompt,
            "tag": self.tag,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "seconds": self.seconds,
            "rms": self.rms,
            "peak": self.peak,
            "clipped_samples": self.clipped_samples,
            "clipped_ratio": self.clipped_ratio,
            "gain_db": self.gain_db,
            "created_at": self.created_at,
            "command": self.command,
            "mode": self.mode,
            "weak": self.weak,
        }


@dataclass(frozen=True)
class CaptureResult:
    action: str
    take: SampleTake | None = None
    reason: str = ""

    @property
    def saved(self) -> bool:
        return self.take is not None and self.action.startswith("saved")


@dataclass(frozen=True)
class DatasetSummary:
    dataset_dir: Path
    labels: dict[str, dict[str, object]]
    total_count: int
    total_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "labels": self.labels,
            "total_count": self.total_count,
            "total_seconds": self.total_seconds,
        }


@dataclass(frozen=True)
class ScoreSummary:
    dataset_dir: Path
    threshold: float
    model: str
    scored_count: int
    positive_count: int
    positive_hits: int
    negative_count: int
    negative_hits: int
    noise_count: int
    noise_hits: int
    scores_path: Path

    @property
    def positive_recall(self) -> float:
        return self.positive_hits / self.positive_count if self.positive_count else 0.0

    @property
    def negative_false_hit_rate(self) -> float:
        return self.negative_hits / self.negative_count if self.negative_count else 0.0

    @property
    def noise_false_hit_rate(self) -> float:
        return self.noise_hits / self.noise_count if self.noise_count else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "threshold": self.threshold,
            "model": self.model,
            "scored_count": self.scored_count,
            "positive_count": self.positive_count,
            "positive_hits": self.positive_hits,
            "positive_recall": self.positive_recall,
            "negative_count": self.negative_count,
            "negative_hits": self.negative_hits,
            "negative_false_hit_rate": self.negative_false_hit_rate,
            "noise_count": self.noise_count,
            "noise_hits": self.noise_hits,
            "noise_false_hit_rate": self.noise_false_hit_rate,
            "scores_path": str(self.scores_path),
        }


class TerminalKeyReader:
    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdin
        self.fd = self.stream.fileno()
        self.is_tty = self.stream.isatty()
        self.original_attrs = None

    def __enter__(self) -> "TerminalKeyReader":
        if self.is_tty:
            self.original_attrs = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *_exc) -> None:
        if self.original_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original_attrs)

    def read_key(self) -> str | None:
        ready, _, _ = select.select([self.stream], [], [], 0)
        if not ready:
            return None
        char = self.stream.read(1)
        return normalize_key(char)


def normalize_key(char: str) -> str | None:
    if char in {"\r", "\n"}:
        return "enter"
    if char == " ":
        return "space"
    if char in {"q", "Q"}:
        return "q"
    if char == "\x03":
        return "ctrl-c"
    return char or None


def init_dataset(dataset_dir: Path) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for label in LABELS:
        (dataset_dir / label).mkdir(exist_ok=True)
    metadata = dataset_dir / "metadata.jsonl"
    metadata.touch(exist_ok=True)
    return dataset_dir


def capture_take(
    config: Config,
    dataset_dir: Path,
    label: str,
    prompt: str = "",
    tag: str = "",
    mode: str = "record",
    command: str = "wake samples record",
    min_rms: float = 100.0,
    min_peak: int = 500,
    keep_weak: bool = False,
    stop_keys: tuple[str, ...] = ("enter",),
    discard_keys: tuple[str, ...] = ("q",),
    key_reader: object | None = None,
    status: Callable[[dict[str, object]], None] | None = None,
) -> CaptureResult:
    if label not in LABELS:
        raise ValueError(f"label must be one of: {', '.join(LABELS)}")
    dataset_dir = init_dataset(dataset_dir)
    target_path = next_sample_path(dataset_dir, label, prompt=prompt, tag=tag)
    tmp_path = target_path.with_name("." + target_path.name + ".tmp")
    sample_rate = int(config.get("audio.sample_rate", 16000))
    channels = int(config.get("audio.channels", 1))
    gain_db = float(config.get("audio.input_gain_db", 0.0))
    device = str(config.get("audio.device", "default"))
    captured_samples = 0
    level_samples = 0
    sum_squares = 0.0
    peak = 0
    clipped_samples = 0
    started = time.monotonic()
    last_status = 0.0
    action = "saved_done"

    reader = key_reader or TerminalKeyReader()
    reader_context = reader if hasattr(reader, "__enter__") and hasattr(reader, "__exit__") else contextlib.nullcontext(reader)
    with reader_context as active_reader:
        try:
            with wave.open(str(tmp_path), "wb") as wav:
                wav.setnchannels(channels)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                for frame in MicCapture(config).frames():
                    chunk = frame.pcm16
                    stats = pcm16_level_stats(chunk)
                    level_samples += int(stats["samples"])
                    sum_squares += float(stats["sum_squares"])
                    peak = max(peak, int(stats["peak"]))
                    clipped_samples += int(stats["clipped_samples"])
                    captured_samples += frame.samples
                    wav.writeframes(chunk)
                    key = active_reader.read_key()
                    if key == "ctrl-c":
                        raise KeyboardInterrupt
                    if key in discard_keys:
                        action = "discard_quit"
                        break
                    if key in stop_keys:
                        action = "saved_next" if key == "space" else "saved_done"
                        break
                    elapsed = time.monotonic() - started
                    if status is not None and elapsed - last_status >= 0.2:
                        rms = math.sqrt(sum_squares / level_samples) if level_samples else 0.0
                        status(
                            {
                                "label": label,
                                "prompt": prompt,
                                "tag": tag,
                                "seconds": captured_samples / sample_rate,
                                "rms": rms,
                                "peak": peak,
                                "samples": captured_samples,
                            }
                        )
                        last_status = elapsed
        except KeyboardInterrupt:
            tmp_path.unlink(missing_ok=True)
            raise

    if action == "discard_quit":
        tmp_path.unlink(missing_ok=True)
        return CaptureResult(action=action, reason="discarded")

    rms = math.sqrt(sum_squares / level_samples) if level_samples else 0.0
    clipped_ratio = clipped_samples / level_samples if level_samples else 0.0
    weak = rms < min_rms or peak < min_peak
    if weak and not keep_weak:
        tmp_path.unlink(missing_ok=True)
        return CaptureResult(action="weak_discarded", reason=f"weak sample not saved: rms={rms:.2f} peak={peak}")

    tmp_path.replace(target_path)
    take = SampleTake(
        path=target_path,
        label=label,
        prompt=prompt,
        tag=tag,
        sample_rate=sample_rate,
        channels=channels,
        samples=captured_samples,
        seconds=captured_samples / sample_rate,
        rms=rms,
        peak=peak,
        clipped_samples=clipped_samples,
        clipped_ratio=clipped_ratio,
        gain_db=gain_db,
        device=device,
        created_at=datetime.now().isoformat(timespec="seconds"),
        command=command,
        mode=mode,
        weak=weak,
    )
    append_metadata(dataset_dir, take)
    return CaptureResult(action=action, take=take)


def append_metadata(dataset_dir: Path, take: SampleTake) -> None:
    with (dataset_dir / "metadata.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(take.to_metadata(dataset_dir), sort_keys=True) + "\n")


def next_sample_path(dataset_dir: Path, label: str, prompt: str = "", tag: str = "") -> Path:
    index = next_take_index(dataset_dir, label)
    slug = safe_slug(tag or prompt or label)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    label_dir = dataset_dir / label
    while True:
        path = label_dir / f"{label}_{timestamp}_{index:04d}_{slug}.wav"
        if not path.exists():
            return path
        index += 1


def next_take_index(dataset_dir: Path, label: str) -> int:
    pattern = re.compile(rf"^{re.escape(label)}_\d{{8}}T\d{{6}}_(\d{{4}})_.+\.wav$")
    highest = 0
    for path in (dataset_dir / label).glob(f"{label}_*.wav"):
        match = pattern.match(path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return (slug or "sample")[:48]


def prompts_for_args(prompt: str = "", prompts_file: Path | None = None, preset: str = "") -> list[str]:
    sources = [bool(prompt), bool(prompts_file), bool(preset)]
    if sum(sources) > 1:
        raise ValueError("choose only one of --prompt, --prompts, or --preset")
    if prompt:
        return [prompt]
    if prompts_file:
        prompts = [line.strip() for line in prompts_file.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if not prompts:
            raise ValueError(f"prompts file is empty: {prompts_file}")
        return prompts
    if preset:
        if preset != "scarlett":
            raise ValueError("only --preset scarlett is supported")
        return SCARLETT_PROMPTS
    return [""]


def cycle_prompts(prompts: list[str]) -> Iterator[str]:
    while True:
        for prompt in prompts:
            yield prompt


def iter_sample_wavs(dataset_dir: Path, label: str | None = None) -> Iterator[Path]:
    labels = [label] if label else list(LABELS)
    for current in labels:
        if current not in LABELS:
            raise ValueError(f"label must be one of: {', '.join(LABELS)}")
        yield from sorted((dataset_dir / current).glob("*.wav"))


def read_metadata(dataset_dir: Path) -> dict[str, dict[str, object]]:
    metadata_path = dataset_dir / "metadata.jsonl"
    records: dict[str, dict[str, object]] = {}
    if not metadata_path.exists():
        return records
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[str(record.get("path", ""))] = record
    return records


def summarize_dataset(dataset_dir: Path) -> DatasetSummary:
    metadata = read_metadata(dataset_dir)
    labels: dict[str, dict[str, object]] = {}
    total_count = 0
    total_seconds = 0.0
    for label in LABELS:
        entries = []
        for wav_path in iter_sample_wavs(dataset_dir, label):
            rel = str(wav_path.relative_to(dataset_dir))
            entries.append(_sample_stats(wav_path, metadata.get(rel, {})))
        count = len(entries)
        seconds = sum(float(item["seconds"]) for item in entries)
        total_count += count
        total_seconds += seconds
        rms_values = [float(item["rms"]) for item in entries]
        peak_values = [int(item["peak"]) for item in entries]
        labels[label] = {
            "count": count,
            "seconds": seconds,
            "rms_min": min(rms_values) if rms_values else 0.0,
            "rms_max": max(rms_values) if rms_values else 0.0,
            "peak_min": min(peak_values) if peak_values else 0,
            "peak_max": max(peak_values) if peak_values else 0,
            "weak_count": sum(1 for item in entries if bool(item.get("weak", False))),
            "clipped_count": sum(1 for item in entries if int(item.get("clipped_samples", 0)) > 0),
            "tags": sorted({str(item.get("tag", "")) for item in entries if item.get("tag")}),
            "prompts": sorted({str(item.get("prompt", "")) for item in entries if item.get("prompt")}),
        }
    return DatasetSummary(dataset_dir=dataset_dir, labels=labels, total_count=total_count, total_seconds=total_seconds)


def _sample_stats(wav_path: Path, metadata: dict[str, object]) -> dict[str, object]:
    if metadata:
        return metadata
    with wave.open(str(wav_path), "rb") as wav:
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
        stats = pcm16_level_stats(frames)
        return {
            "path": str(wav_path),
            "seconds": wav.getnframes() / sample_rate if sample_rate else 0.0,
            "rms": stats["rms"],
            "peak": stats["peak"],
            "clipped_samples": stats["clipped_samples"],
            "weak": False,
        }


def score_dataset(
    config: Config,
    dataset_dir: Path,
    model: str | None = None,
    threshold: float | None = None,
    label: str | None = None,
) -> ScoreSummary:
    cfg = config.with_overrides({"wake": {"model_path": model}}) if model else config
    scores_path = dataset_dir / "scores.jsonl"
    metadata = read_metadata(dataset_dir)
    counts = {"positive": 0, "negative": 0, "noise": 0}
    hits = {"positive": 0, "negative": 0, "noise": 0}
    scored_count = 0
    wav_paths = list(iter_sample_wavs(dataset_dir, label))
    detector = OpenWakeWordDetector(cfg) if wav_paths else None
    with scores_path.open("a", encoding="utf-8") as handle:
        for wav_path in wav_paths:
            sample_label = wav_path.parent.name
            result = score_wake_audio(cfg, wav_path, threshold=threshold, detector=detector)
            rel = str(wav_path.relative_to(dataset_dir))
            record = {
                **result.to_dict(),
                "path": rel,
                "label": sample_label,
                "prompt": metadata.get(rel, {}).get("prompt", ""),
                "tag": metadata.get(rel, {}).get("tag", ""),
                "model": model or str(config.get("wake.model_path", "models/wake/scarlett.onnx")),
                "scored_at": datetime.now().isoformat(timespec="seconds"),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            scored_count += 1
            counts[sample_label] += 1
            if result.hit:
                hits[sample_label] += 1
    effective_threshold = float(threshold if threshold is not None else config.get("wake.sensitivity", 0.5))
    return ScoreSummary(
        dataset_dir=dataset_dir,
        threshold=effective_threshold,
        model=model or str(config.get("wake.model_path", "models/wake/scarlett.onnx")),
        scored_count=scored_count,
        positive_count=counts["positive"],
        positive_hits=hits["positive"],
        negative_count=counts["negative"],
        negative_hits=hits["negative"],
        noise_count=counts["noise"],
        noise_hits=hits["noise"],
        scores_path=scores_path,
    )


def render_dataset_summary(summary: DatasetSummary) -> str:
    lines = [f"wake samples: {summary.dataset_dir}", f"total: {summary.total_count} take(s), {summary.total_seconds:.2f}s"]
    for label in LABELS:
        item = summary.labels[label]
        lines.append(
            f"{label}: {item['count']} take(s), {float(item['seconds']):.2f}s, "
            f"rms {float(item['rms_min']):.1f}-{float(item['rms_max']):.1f}, "
            f"peak {int(item['peak_min'])}-{int(item['peak_max'])}, "
            f"weak {item['weak_count']}, clipped {item['clipped_count']}"
        )
        if item["tags"]:
            lines.append("  tags: " + ", ".join(str(tag) for tag in item["tags"]))
        if item["prompts"]:
            lines.append("  prompts: " + ", ".join(str(prompt) for prompt in item["prompts"]))
    return "\n".join(lines)


def render_score_summary(summary: ScoreSummary) -> str:
    return "\n".join(
        [
            f"scored {summary.scored_count} wake sample(s) at threshold {summary.threshold:g}",
            f"positive recall: {summary.positive_hits}/{summary.positive_count} ({summary.positive_recall:.3f})",
            f"negative false hits: {summary.negative_hits}/{summary.negative_count} ({summary.negative_false_hit_rate:.3f})",
            f"noise false hits: {summary.noise_hits}/{summary.noise_count} ({summary.noise_false_hit_rate:.3f})",
            f"receipts: {summary.scores_path}",
        ]
    )
