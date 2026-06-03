from __future__ import annotations

import importlib.util
import json
import sys
import wave
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SEEDER_PATH = ROOT / "tools" / "livekit-wakeword" / "seed_cxv_samples.py"


def load_seeder():
    spec = importlib.util.spec_from_file_location("seed_cxv_samples", SEEDER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_wav(path: Path, frames: int = 1600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x01" * frames)


def append_metadata(dataset: Path, rel_path: str, label: str, prompt: str, tag: str) -> None:
    with (dataset / "metadata.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "path": rel_path,
                    "label": label,
                    "prompt": prompt,
                    "tag": tag,
                },
                sort_keys=True,
            )
            + "\n"
        )


def make_sample(dataset: Path, label: str, name: str, prompt: str, tag: str) -> None:
    rel_path = f"{label}/{name}.wav"
    write_wav(dataset / rel_path)
    append_metadata(dataset, rel_path, label, prompt, tag)


def test_seed_cxv_samples_copies_splits_and_manifest(tmp_path: Path) -> None:
    seeder = load_seeder()
    dataset = tmp_path / "samples"
    make_sample(dataset, "positive", "positive_1", "scarlett", "voice")
    make_sample(dataset, "positive", "positive_2", "hey scarlett", "voice")
    make_sample(dataset, "positive", "positive_3", "scarlett now", "voice")
    make_sample(dataset, "negative", "negative_1", "starlet", "hard-negative")
    make_sample(dataset, "negative", "negative_2", "open the browser", "normal-speech")
    make_sample(dataset, "noise", "noise_1", "silence", "silence")
    make_sample(dataset, "noise", "noise_2", "room noise", "room-noise")

    output_dir = tmp_path / "livekit-output"
    counts = seeder.seed_samples([dataset], output_dir, "scarlett", 0.8)

    assert counts == {
        "background_test": 1,
        "background_train": 1,
        "negative_test": 1,
        "negative_train": 1,
        "positive_test": 1,
        "positive_train": 2,
    }
    model_dir = output_dir / "scarlett"
    assert (model_dir / "positive_train" / "clip_000000.wav").exists()
    assert (model_dir / "negative_test" / "clip_000000.wav").exists()
    assert (model_dir / "background_train" / "clip_000000.wav").exists()

    manifest = [
        json.loads(line)
        for line in (model_dir / "seed_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(manifest) == 7
    assert {record["label"] for record in manifest} == {"positive", "negative", "noise"}
    assert any(record["prompt"] == "hey scarlett" for record in manifest)
    assert all(str(record["target"]).startswith(("positive_", "negative_", "background_")) for record in manifest)


def test_seed_cxv_samples_refuses_existing_clips_without_clear(tmp_path: Path) -> None:
    seeder = load_seeder()
    dataset = tmp_path / "samples"
    make_sample(dataset, "positive", "positive_1", "scarlett", "voice")
    output_dir = tmp_path / "livekit-output"

    seeder.seed_samples([dataset], output_dir, "scarlett", 0.8)
    with pytest.raises(RuntimeError, match="already exist"):
        seeder.seed_samples([dataset], output_dir, "scarlett", 0.8)


def test_seed_cxv_samples_can_train_on_all_with_repeats(tmp_path: Path) -> None:
    seeder = load_seeder()
    dataset = tmp_path / "samples"
    make_sample(dataset, "positive", "positive_1", "scarlett", "voice")
    make_sample(dataset, "positive", "positive_2", "hey scarlett", "voice")
    make_sample(dataset, "positive", "positive_3", "scarlett now", "voice")
    make_sample(dataset, "negative", "negative_1", "starlet", "hard-negative")
    make_sample(dataset, "negative", "negative_2", "charlotte", "hard-negative")

    output_dir = tmp_path / "livekit-output"
    counts = seeder.seed_samples(
        [dataset],
        output_dir,
        "scarlett",
        0.8,
        train_on_all=True,
        train_repeats={"positive": 2, "negative": 3},
    )

    assert counts["positive_test"] == 1
    assert counts["positive_train"] == 6
    assert counts["negative_test"] == 1
    assert counts["negative_train"] == 6
    manifest = [
        json.loads(line)
        for line in (output_dir / "scarlett" / "seed_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sum(1 for record in manifest if record["split"] == "positive_train") == 6
    train_records = [record for record in manifest if str(record["split"]).endswith("_train")]
    assert sum(1 for record in train_records if record["source_path"] == "negative/negative_1.wav") == 3


def test_seed_cxv_samples_can_mine_score_receipts_for_repeat_overrides(tmp_path: Path) -> None:
    seeder = load_seeder()
    dataset = tmp_path / "samples"
    make_sample(dataset, "positive", "positive_1", "scarlett", "voice")
    make_sample(dataset, "positive", "positive_2", "scarlett now", "voice")
    make_sample(dataset, "negative", "negative_1", "starlet", "hard-negative")
    make_sample(dataset, "negative", "negative_2", "open browser", "normal-speech")
    make_sample(dataset, "noise", "noise_1", "silence", "silence")

    receipt = tmp_path / "score.jsonl"
    with receipt.open("w", encoding="utf-8") as handle:
        for record in (
            {"path": "positive/positive_1.wav", "label": "positive", "hit": False},
            {"path": "positive/positive_2.wav", "label": "positive", "hit": True},
            {"path": "negative/negative_1.wav", "label": "negative", "hit": True},
            {"path": "noise/noise_1.wav", "label": "noise", "hit": True},
        ):
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    mined_repeats = seeder.read_score_receipt_repeats(
        [receipt],
        positive_miss_repeat=5,
        negative_false_hit_repeat=7,
        noise_false_hit_repeat=6,
    )
    output_dir = tmp_path / "livekit-output"
    counts = seeder.seed_samples(
        [dataset],
        output_dir,
        "scarlett",
        0.8,
        train_on_all=True,
        train_repeats={"positive": 2, "negative": 2, "noise": 2},
        sample_train_repeats=mined_repeats,
    )

    assert mined_repeats == {
        "negative/negative_1.wav": 7,
        "noise/noise_1.wav": 6,
        "positive/positive_1.wav": 5,
    }
    assert counts["positive_train"] == 7
    assert counts["negative_train"] == 9
    assert counts["background_train"] == 6
    manifest = [
        json.loads(line)
        for line in (output_dir / "scarlett" / "seed_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_records = [record for record in manifest if str(record["split"]).endswith("_train")]
    assert sum(1 for record in train_records if record["source_path"] == "positive/positive_1.wav") == 5
    assert sum(1 for record in train_records if record["source_path"] == "positive/positive_2.wav") == 2
    assert sum(1 for record in train_records if record["source_path"] == "negative/negative_1.wav") == 7
    assert sum(1 for record in train_records if record["source_path"] == "noise/noise_1.wav") == 6


def test_seed_cxv_samples_can_repeat_by_metadata_tag(tmp_path: Path) -> None:
    seeder = load_seeder()
    dataset = tmp_path / "samples"
    make_sample(dataset, "negative", "negative_1", "starlet", "hard-negative")
    make_sample(dataset, "negative", "negative_2", "mined starlet", "mined-false-hit")
    make_sample(dataset, "positive", "positive_1", "scarlett", "voice")

    output_dir = tmp_path / "livekit-output"
    counts = seeder.seed_samples(
        [dataset],
        output_dir,
        "scarlett",
        0.8,
        train_on_all=True,
        train_repeats={"positive": 2, "negative": 2},
        tag_train_repeats=seeder.parse_tag_train_repeats(["mined-false-hit=8"]),
    )

    assert counts["negative_train"] == 10
    manifest = [
        json.loads(line)
        for line in (output_dir / "scarlett" / "seed_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_records = [record for record in manifest if str(record["split"]).endswith("_train")]
    assert sum(1 for record in train_records if record["source_path"] == "negative/negative_1.wav") == 2
    assert sum(1 for record in train_records if record["source_path"] == "negative/negative_2.wav") == 8
