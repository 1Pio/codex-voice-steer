from __future__ import annotations

import importlib.util
import json
import sys
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MINER_PATH = ROOT / "tools" / "livekit-wakeword" / "mine_score_receipt_clips.py"


def load_miner():
    spec = importlib.util.spec_from_file_location("mine_score_receipt_clips", MINER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_wav(path: Path, frames: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x01\x00" * frames)


def test_mine_score_receipt_clips_can_filter_false_hits(tmp_path: Path) -> None:
    miner = load_miner()
    dataset = tmp_path / "samples"
    write_wav(dataset / "positive" / "positive_1.wav")
    write_wav(dataset / "negative" / "negative_1.wav")
    write_wav(dataset / "noise" / "noise_1.wav")
    receipt = tmp_path / "score.jsonl"
    with receipt.open("w", encoding="utf-8") as handle:
        for record in (
            {
                "path": "positive/positive_1.wav",
                "label": "positive",
                "hit": False,
                "max_score_time_sec": 0.5,
            },
            {
                "path": "negative/negative_1.wav",
                "label": "negative",
                "hit": True,
                "max_score_time_sec": 0.5,
            },
            {"path": "noise/noise_1.wav", "label": "noise", "hit": True, "max_score_time_sec": 0.5},
        ):
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    records = miner.mine_receipts(
        dataset,
        [receipt],
        tmp_path / "mined",
        window_seconds=[0.5],
        offset_seconds=[0.0],
        failure_kind="false-hit",
    )

    assert [record["label"] for record in records] == ["negative", "noise"]
    assert {record["tag"] for record in records} == {miner.FALSE_HIT_TAG}
    assert not (tmp_path / "mined" / "positive").exists()
