#!/usr/bin/env python3
from __future__ import annotations

import argparse
import array
import json
import math
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path


FALSE_HIT_TAG = "mined-false-hit"
POSITIVE_MISS_TAG = "mined-positive-miss"


@dataclass(frozen=True)
class SourceAudio:
    channels: int
    sample_width: int
    sample_rate: int
    frames: bytes

    @property
    def frame_count(self) -> int:
        return len(self.frames) // (self.channels * self.sample_width)

    @property
    def seconds(self) -> float:
        return self.frame_count / self.sample_rate


def read_source_audio(path: Path) -> SourceAudio:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if channels != 1 or sample_width != 2 or sample_rate != 16000:
        raise ValueError(
            f"{path} must be 16 kHz mono PCM16 WAV "
            f"(got {sample_rate} Hz, {channels} channel(s), {sample_width * 8}-bit)"
        )
    return SourceAudio(channels=channels, sample_width=sample_width, sample_rate=sample_rate, frames=frames)


def write_crop(source: SourceAudio, target: Path, start_frame: int, end_frame: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    byte_start = start_frame * source.channels * source.sample_width
    byte_end = end_frame * source.channels * source.sample_width
    with wave.open(str(target), "wb") as wav:
        wav.setnchannels(source.channels)
        wav.setsampwidth(source.sample_width)
        wav.setframerate(source.sample_rate)
        wav.writeframes(source.frames[byte_start:byte_end])


def audio_stats(source: SourceAudio, start_frame: int, end_frame: int) -> tuple[int, float]:
    byte_start = start_frame * source.channels * source.sample_width
    byte_end = end_frame * source.channels * source.sample_width
    samples = array.array("h")
    samples.frombytes(source.frames[byte_start:byte_end])
    if not samples:
        return 0, 0.0
    peak = max(abs(sample) for sample in samples)
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    return peak, rms


def crop_bounds(source: SourceAudio, center_sec: float, window_sec: float, offset_sec: float) -> tuple[int, int]:
    duration = source.seconds
    window = min(window_sec, duration)
    start = center_sec - (window / 2.0) + offset_sec
    start = max(0.0, min(start, duration - window))
    end = start + window
    return round(start * source.sample_rate), round(end * source.sample_rate)


def should_mine_record(record: dict[str, object], failure_kind: str = "all") -> tuple[str, str] | None:
    label = str(record.get("label", ""))
    hit = bool(record.get("hit", False))
    if label == "positive" and not hit and failure_kind in {"all", "positive-miss"}:
        return label, POSITIVE_MISS_TAG
    if label in {"negative", "noise"} and hit and failure_kind in {"all", "false-hit"}:
        return label, FALSE_HIT_TAG
    return None


def mine_receipts(
    dataset_dir: Path,
    receipt_paths: list[Path],
    output_dir: Path,
    *,
    window_seconds: list[float],
    offset_seconds: list[float],
    failure_kind: str = "all",
    clear: bool = False,
    dry_run: bool = False,
) -> list[dict[str, object]]:
    if clear and output_dir.exists() and not dry_run:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.rglob("*.wav")) and not dry_run:
        raise RuntimeError(f"{output_dir} already contains WAV files. Use --clear to replace it.")

    records: list[dict[str, object]] = []
    seen: set[tuple[str, str, int, int]] = set()
    next_index = 1
    for receipt_path in receipt_paths:
        for line_number, line in enumerate(receipt_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            score_record = json.loads(line)
            mined = should_mine_record(score_record, failure_kind=failure_kind)
            if mined is None:
                continue
            label, tag = mined
            rel_path = str(score_record.get("path", ""))
            if not rel_path:
                raise ValueError(f"{receipt_path}:{line_number} is missing path")
            source_path = dataset_dir / rel_path
            source = read_source_audio(source_path)
            center = float(score_record.get("max_score_time_sec", source.seconds / 2.0))
            for window in window_seconds:
                for offset in offset_seconds:
                    start_frame, end_frame = crop_bounds(source, center, window, offset)
                    if end_frame <= start_frame:
                        continue
                    key = (rel_path, label, start_frame, end_frame)
                    if key in seen:
                        continue
                    seen.add(key)
                    target_rel = f"{label}/{label}_mined_{next_index:04d}.wav"
                    target_path = output_dir / target_rel
                    peak, rms = audio_stats(source, start_frame, end_frame)
                    seconds = (end_frame - start_frame) / source.sample_rate
                    record = {
                        "channels": source.channels,
                        "clipped_ratio": 0.0,
                        "clipped_samples": 0,
                        "command": "tools/livekit-wakeword/mine_score_receipt_clips.py",
                        "label": label,
                        "mode": "score-receipt-mine",
                        "path": target_rel,
                        "peak": peak,
                        "prompt": f"{tag} from {rel_path}",
                        "rms": rms,
                        "sample_rate": source.sample_rate,
                        "seconds": seconds,
                        "source_max_score": score_record.get("max_score"),
                        "source_max_score_time_sec": center,
                        "source_path": rel_path,
                        "source_receipt": str(receipt_path),
                        "source_threshold": score_record.get("threshold"),
                        "start_sec": start_frame / source.sample_rate,
                        "tag": tag,
                        "weak": False,
                    }
                    records.append(record)
                    if not dry_run:
                        write_crop(source, target_path, start_frame, end_frame)
                    next_index += 1

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    return records


def csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create cxv-style hard-mined clips from wake score receipts.")
    parser.add_argument("--dataset", required=True, help="Source cxv wake sample dataset directory.")
    parser.add_argument("--receipt", action="append", required=True, help="cxv wake samples score JSONL receipt. Repeatable.")
    parser.add_argument("--output-dir", required=True, help="Output cxv-style dataset directory for mined clips.")
    parser.add_argument("--window-seconds", default="1.2,1.6,2.0", help="Comma-separated crop window durations.")
    parser.add_argument("--offset-seconds", default="-0.24,0,0.24", help="Comma-separated offsets around max score time.")
    parser.add_argument(
        "--failure-kind",
        choices=("all", "false-hit", "positive-miss"),
        default="all",
        help="Which score failures to mine.",
    )
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = mine_receipts(
        Path(args.dataset).expanduser(),
        [Path(item).expanduser() for item in args.receipt],
        Path(args.output_dir).expanduser(),
        window_seconds=csv_floats(args.window_seconds),
        offset_seconds=csv_floats(args.offset_seconds),
        failure_kind=args.failure_kind,
        clear=args.clear,
        dry_run=args.dry_run,
    )
    counts: dict[str, int] = {}
    for record in records:
        counts[str(record["label"])] = counts.get(str(record["label"]), 0) + 1
    for label in sorted(counts):
        print(f"{label}: {counts[label]}")
    print(f"total: {len(records)}")
    if args.dry_run:
        print("dry-run: no files written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
