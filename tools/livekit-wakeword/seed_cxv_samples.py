#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


LABEL_TO_SPLITS = {
    "positive": ("positive_train", "positive_test"),
    "negative": ("negative_train", "negative_test"),
    "noise": ("background_train", "background_test"),
}


@dataclass(frozen=True)
class SourceSample:
    dataset_dir: Path
    path: Path
    rel_path: str
    label: str
    prompt: str
    tag: str
    seconds: float


def read_metadata(dataset_dir: Path) -> dict[str, dict[str, object]]:
    metadata_path = dataset_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return {}
    records: dict[str, dict[str, object]] = {}
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        path = str(record.get("path", ""))
        if path:
            records[path] = record
    return records


def inspect_wav(path: Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
    if channels != 1 or sample_width != 2 or sample_rate != 16000:
        raise ValueError(
            f"{path} must be 16 kHz mono PCM16 WAV "
            f"(got {sample_rate} Hz, {channels} channel(s), {sample_width * 8}-bit)"
        )
    return frame_count, frame_count / 16000


def iter_source_samples(dataset_dir: Path) -> list[SourceSample]:
    metadata = read_metadata(dataset_dir)
    samples: list[SourceSample] = []
    for label in LABEL_TO_SPLITS:
        for wav_path in sorted((dataset_dir / label).glob("*.wav")):
            rel_path = str(wav_path.relative_to(dataset_dir))
            _, seconds = inspect_wav(wav_path)
            record = metadata.get(rel_path, {})
            samples.append(
                SourceSample(
                    dataset_dir=dataset_dir,
                    path=wav_path,
                    rel_path=rel_path,
                    label=label,
                    prompt=str(record.get("prompt", "")),
                    tag=str(record.get("tag", "")),
                    seconds=seconds,
                )
            )
    return samples


def stable_order_key(sample: SourceSample) -> str:
    value = f"{sample.dataset_dir.resolve()}::{sample.rel_path}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def split_samples(
    samples: list[SourceSample],
    train_ratio: float,
    *,
    train_on_all: bool = False,
    train_repeats: dict[str, int] | None = None,
    sample_train_repeats: dict[str, int] | None = None,
    tag_train_repeats: dict[str, int] | None = None,
) -> dict[str, list[SourceSample]]:
    by_label: dict[str, list[SourceSample]] = defaultdict(list)
    for sample in samples:
        by_label[sample.label].append(sample)

    repeats = train_repeats or {}
    sample_repeats = sample_train_repeats or {}
    tag_repeats = tag_train_repeats or {}
    splits: dict[str, list[SourceSample]] = {name: [] for names in LABEL_TO_SPLITS.values() for name in names}
    for label, label_samples in by_label.items():
        train_split, test_split = LABEL_TO_SPLITS[label]
        ordered = sorted(label_samples, key=stable_order_key)
        test_count = round(len(ordered) * (1.0 - train_ratio))
        if len(ordered) > 1:
            test_count = max(1, test_count)
        test_keys = {sample.path for sample in ordered[:test_count]}
        for sample in sorted(label_samples, key=lambda item: (item.rel_path, stable_order_key(item))):
            splits[test_split if sample.path in test_keys else train_split].append(sample)
        if train_on_all:
            splits[train_split] = sorted(label_samples, key=lambda item: (item.rel_path, stable_order_key(item)))
        label_repeat = repeats.get(label, 1)
        if label_repeat < 1:
            raise ValueError("train repeat values must be positive")
        repeated_train_samples: list[SourceSample] = []
        for sample in splits[train_split]:
            sample_repeat = max(
                label_repeat,
                sample_repeats.get(sample.rel_path, label_repeat),
                tag_repeats.get(sample.tag, label_repeat),
            )
            if sample_repeat < 1:
                raise ValueError("sample train repeat values must be positive")
            repeated_train_samples.extend(sample for _ in range(sample_repeat))
        splits[train_split] = repeated_train_samples
    return splits


def parse_tag_train_repeats(values: list[str]) -> dict[str, int]:
    repeats: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--tag-train-repeat must use tag=count")
        tag, repeat_text = value.split("=", 1)
        tag = tag.strip()
        if not tag:
            raise ValueError("--tag-train-repeat tag cannot be empty")
        repeat = int(repeat_text)
        if repeat < 1:
            raise ValueError("--tag-train-repeat count must be positive")
        repeats[tag] = repeat
    return repeats


def read_score_receipt_repeats(
    paths: list[Path],
    *,
    positive_miss_repeat: int,
    negative_false_hit_repeat: int,
    noise_false_hit_repeat: int,
) -> dict[str, int]:
    repeats: dict[str, int] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            rel_path = str(record.get("path", ""))
            label = str(record.get("label", ""))
            hit = bool(record.get("hit", False))
            if not rel_path:
                raise ValueError(f"{path}:{line_number} is missing path")
            repeat = 0
            if label == "positive" and not hit:
                repeat = positive_miss_repeat
            elif label == "negative" and hit:
                repeat = negative_false_hit_repeat
            elif label == "noise" and hit:
                repeat = noise_false_hit_repeat
            if repeat > 1:
                repeats[rel_path] = max(repeats.get(rel_path, 1), repeat)
    return repeats


def clear_existing(model_dir: Path) -> None:
    for split in LABEL_TO_SPLITS.values():
        for split_name in split:
            split_dir = model_dir / split_name
            if not split_dir.exists():
                continue
            for path in split_dir.glob("clip_*.wav"):
                path.unlink()


def ensure_no_existing(model_dir: Path) -> None:
    existing: list[Path] = []
    for split in LABEL_TO_SPLITS.values():
        for split_name in split:
            existing.extend((model_dir / split_name).glob("clip_*.wav"))
    if existing:
        preview = "\n".join(str(path) for path in sorted(existing)[:10])
        raise RuntimeError(
            f"LiveKit split clips already exist under {model_dir}. "
            f"Use --clear to replace only clip_*.wav in known split dirs.\n{preview}"
        )


def seed_samples(
    datasets: list[Path],
    output_dir: Path,
    model_name: str,
    train_ratio: float,
    *,
    train_on_all: bool = False,
    train_repeats: dict[str, int] | None = None,
    sample_train_repeats: dict[str, int] | None = None,
    tag_train_repeats: dict[str, int] | None = None,
    clear: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    if not 0.5 <= train_ratio <= 0.95:
        raise ValueError("--train-ratio must be between 0.5 and 0.95")

    all_samples: list[SourceSample] = []
    for dataset in datasets:
        if not dataset.exists():
            raise FileNotFoundError(dataset)
        all_samples.extend(iter_source_samples(dataset))

    model_dir = output_dir / model_name
    if clear and not dry_run:
        clear_existing(model_dir)
    if not dry_run:
        ensure_no_existing(model_dir)

    splits = split_samples(
        all_samples,
        train_ratio,
        train_on_all=train_on_all,
        train_repeats=train_repeats,
        sample_train_repeats=sample_train_repeats,
        tag_train_repeats=tag_train_repeats,
    )
    manifest_records: list[dict[str, object]] = []
    for split_name, split_samples_for_name in splits.items():
        split_dir = model_dir / split_name
        if not dry_run:
            split_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(split_samples_for_name):
            target = split_dir / f"clip_{index:06d}.wav"
            record = {
                "split": split_name,
                "index": index,
                "target": str(target.relative_to(model_dir)),
                "source_dataset": str(sample.dataset_dir),
                "source_path": sample.rel_path,
                "label": sample.label,
                "prompt": sample.prompt,
                "tag": sample.tag,
                "seconds": sample.seconds,
            }
            manifest_records.append(record)
            if not dry_run:
                shutil.copy2(sample.path, target)

    if not dry_run:
        manifest_path = model_dir / "seed_manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for record in manifest_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    return {split_name: len(items) for split_name, items in splits.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed LiveKit wakeword split dirs from cxv wake sample datasets.")
    parser.add_argument("--dataset", action="append", required=True, help="cxv wake sample dataset directory. Repeatable.")
    parser.add_argument("--output-dir", required=True, help="LiveKit output directory from scarlett.yaml.")
    parser.add_argument("--model-name", default="scarlett")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--train-on-all",
        action="store_true",
        help="Also copy held-out samples into the training split while keeping test split copies.",
    )
    parser.add_argument("--positive-train-repeat", type=int, default=1, help="Training copies per positive source sample.")
    parser.add_argument("--negative-train-repeat", type=int, default=1, help="Training copies per negative source sample.")
    parser.add_argument("--noise-train-repeat", type=int, default=1, help="Training copies per noise source sample.")
    parser.add_argument(
        "--tag-train-repeat",
        action="append",
        default=[],
        help="Override training copies for samples with a metadata tag, as tag=count. Repeatable.",
    )
    parser.add_argument(
        "--score-receipt",
        action="append",
        default=[],
        help="cxv wake samples score JSONL receipt to mine for missed positives and false-hit negatives/noise.",
    )
    parser.add_argument(
        "--positive-miss-repeat",
        type=int,
        default=1,
        help="Training copies for positive samples missed in --score-receipt.",
    )
    parser.add_argument(
        "--negative-false-hit-repeat",
        type=int,
        default=1,
        help="Training copies for negative samples hit in --score-receipt.",
    )
    parser.add_argument(
        "--noise-false-hit-repeat",
        type=int,
        default=1,
        help="Training copies for noise samples hit in --score-receipt.",
    )
    parser.add_argument("--clear", action="store_true", help="Replace existing clip_*.wav files in known LiveKit split dirs.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    score_receipt_repeats = read_score_receipt_repeats(
        [Path(item).expanduser() for item in args.score_receipt],
        positive_miss_repeat=args.positive_miss_repeat,
        negative_false_hit_repeat=args.negative_false_hit_repeat,
        noise_false_hit_repeat=args.noise_false_hit_repeat,
    )
    counts = seed_samples(
        datasets=[Path(item).expanduser() for item in args.dataset],
        output_dir=Path(args.output_dir).expanduser(),
        model_name=args.model_name,
        train_ratio=args.train_ratio,
        train_on_all=args.train_on_all,
        train_repeats={
            "positive": args.positive_train_repeat,
            "negative": args.negative_train_repeat,
            "noise": args.noise_train_repeat,
        },
        sample_train_repeats=score_receipt_repeats,
        tag_train_repeats=parse_tag_train_repeats(args.tag_train_repeat),
        clear=args.clear,
        dry_run=args.dry_run,
    )
    for split_name in sorted(counts):
        print(f"{split_name}: {counts[split_name]}")
    if score_receipt_repeats:
        print(f"score_receipt_repeat_overrides: {len(score_receipt_repeats)}")
    if args.dry_run:
        print("dry-run: no files written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
