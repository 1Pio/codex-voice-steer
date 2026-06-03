from __future__ import annotations

import json
import wave
from types import SimpleNamespace

import pytest

from codex_voice_steer import wake_samples
from codex_voice_steer.config import load_config
from codex_voice_steer.segment import AudioFrame
from codex_voice_steer.wake_samples import (
    capture_take,
    generate_synthetic_msd_samples,
    init_dataset,
    normalize_key,
    prompts_for_args,
    render_dataset_summary,
    render_score_summary,
    safe_slug,
    score_dataset,
    summarize_dataset,
    validate_synthetic_negative_prompts,
)


class FakeKeyReader:
    def __init__(self, keys):
        self.keys = list(keys)
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *_exc):
        self.exited = True

    def read_key(self):
        if not self.keys:
            return "enter"
        return self.keys.pop(0)


def test_init_dataset_preserves_append_only_metadata(tmp_path) -> None:
    dataset = tmp_path / "samples"
    init_dataset(dataset)
    metadata = dataset / "metadata.jsonl"
    metadata.write_text('{"existing":true}\n', encoding="utf-8")

    init_dataset(dataset)

    assert (dataset / "positive").is_dir()
    assert (dataset / "negative").is_dir()
    assert (dataset / "noise").is_dir()
    assert metadata.read_text(encoding="utf-8") == '{"existing":true}\n'


def test_capture_take_writes_sortable_wav_and_metadata(tmp_path, monkeypatch) -> None:
    class FakeMicCapture:
        def __init__(self, config):
            self.config = config

        def frames(self):
            yield AudioFrame(pcm16=(1000).to_bytes(2, "little", signed=True) * 1280, sample_rate=16000, channels=1)
            yield AudioFrame(pcm16=(2000).to_bytes(2, "little", signed=True) * 1280, sample_rate=16000, channels=1)

    monkeypatch.setattr(wake_samples, "MicCapture", FakeMicCapture)
    cfg = load_config(overrides={"audio": {"device": "Mic", "input_gain_db": 3}}, path=tmp_path / "missing.toml")
    reader = FakeKeyReader([None, "enter"])

    result = capture_take(
        cfg,
        tmp_path / "samples",
        label="positive",
        prompt="scarlett",
        tag="near mic",
        min_rms=10,
        min_peak=10,
        key_reader=reader,
    )

    assert result.action == "saved_done"
    assert result.take is not None
    assert reader.entered is True
    assert reader.exited is True
    assert result.take.path.name.startswith("positive_")
    assert result.take.path.name.endswith("_near_mic.wav")
    with wave.open(str(result.take.path), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 2560
    records = [json.loads(line) for line in (tmp_path / "samples" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[-1]["path"].startswith("positive/")
    assert records[-1]["label"] == "positive"
    assert records[-1]["prompt"] == "scarlett"
    assert records[-1]["tag"] == "near mic"
    assert records[-1]["device"] == "Mic"
    assert records[-1]["sample_rate"] == 16000
    assert records[-1]["channels"] == 1
    assert records[-1]["gain_db"] == 3
    assert records[-1]["command"] == "wake samples record"


def test_capture_take_discards_weak_sample_unless_keep_weak(tmp_path, monkeypatch) -> None:
    class FakeMicCapture:
        def __init__(self, config):
            self.config = config

        def frames(self):
            yield AudioFrame(pcm16=b"\0\0" * 1280, sample_rate=16000, channels=1)

    monkeypatch.setattr(wake_samples, "MicCapture", FakeMicCapture)
    cfg = load_config(path=tmp_path / "missing.toml")

    discarded = capture_take(cfg, tmp_path / "samples", label="noise", min_rms=1, min_peak=1, key_reader=FakeKeyReader(["enter"]))
    kept = capture_take(
        cfg,
        tmp_path / "samples",
        label="noise",
        min_rms=1,
        min_peak=1,
        keep_weak=True,
        key_reader=FakeKeyReader(["enter"]),
    )

    assert discarded.action == "weak_discarded"
    assert discarded.take is None
    assert kept.take is not None
    assert kept.take.weak is True
    assert len(list((tmp_path / "samples" / "noise").glob("*.wav"))) == 1


def test_session_controls_space_enter_q_and_ctrl_c(tmp_path, monkeypatch) -> None:
    class FakeMicCapture:
        def __init__(self, config):
            self.config = config

        def frames(self):
            while True:
                yield AudioFrame(pcm16=(1000).to_bytes(2, "little", signed=True) * 1280, sample_rate=16000, channels=1)

    monkeypatch.setattr(wake_samples, "MicCapture", FakeMicCapture)
    cfg = load_config(path=tmp_path / "missing.toml")
    first = capture_take(cfg, tmp_path / "samples", label="positive", stop_keys=("space", "enter"), key_reader=FakeKeyReader(["space"]), min_rms=1, min_peak=1)
    done = capture_take(cfg, tmp_path / "samples", label="positive", stop_keys=("space", "enter"), key_reader=FakeKeyReader(["enter"]), min_rms=1, min_peak=1)
    discarded = capture_take(cfg, tmp_path / "samples", label="positive", stop_keys=("space", "enter"), key_reader=FakeKeyReader(["q"]), min_rms=1, min_peak=1)

    assert first.action == "saved_next"
    assert done.action == "saved_done"
    assert discarded.action == "discard_quit"
    with pytest.raises(KeyboardInterrupt):
        capture_take(cfg, tmp_path / "samples", label="positive", stop_keys=("space", "enter"), key_reader=FakeKeyReader(["ctrl-c"]), min_rms=1, min_peak=1)
    assert len(list((tmp_path / "samples" / "positive").glob("*.wav"))) == 2


def test_list_summary_uses_metadata_ranges_tags_and_prompts(tmp_path, monkeypatch) -> None:
    class FakeMicCapture:
        def __init__(self, config):
            self.config = config

        def frames(self):
            yield AudioFrame(pcm16=(1000).to_bytes(2, "little", signed=True) * 1280, sample_rate=16000, channels=1)

    monkeypatch.setattr(wake_samples, "MicCapture", FakeMicCapture)
    cfg = load_config(path=tmp_path / "missing.toml")
    dataset = tmp_path / "samples"
    capture_take(cfg, dataset, label="positive", prompt="scarlett", tag="desk", min_rms=1, min_peak=1, key_reader=FakeKeyReader(["enter"]))
    capture_take(cfg, dataset, label="negative", prompt="starlet", tag="hard", min_rms=1, min_peak=1, key_reader=FakeKeyReader(["enter"]))

    summary = summarize_dataset(dataset)
    rendered = render_dataset_summary(summary)

    assert summary.total_count == 2
    assert summary.labels["positive"]["count"] == 1
    assert summary.labels["negative"]["prompts"] == ["starlet"]
    assert "positive: 1 take(s)" in rendered
    assert "tags: desk" in rendered


def test_score_dataset_appends_score_receipts_and_summarizes_hits(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "samples"
    init_dataset(dataset)
    for label in ("positive", "negative", "noise"):
        path = dataset / label / f"{label}_20260531T171522_0001_test.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes((1000).to_bytes(2, "little", signed=True) * 1280)

    def fake_score(_config, wav_path, threshold=None, detector=None):
        assert detector is not None
        hit = wav_path.parent.name in {"positive", "noise"}
        return SimpleNamespace(
            hit=hit,
            to_dict=lambda: {
                "wav_path": str(wav_path),
                "hit": hit,
                "max_score": 0.9 if hit else 0.1,
                "threshold": threshold if threshold is not None else 0.5,
            },
        )

    monkeypatch.setattr(wake_samples, "score_wake_audio", fake_score)

    summary = score_dataset(load_config(path=tmp_path / "missing.toml"), dataset, model="models/wake/scarlett.onnx", threshold=0.5)

    assert summary.positive_recall == 1.0
    assert summary.negative_false_hit_rate == 0.0
    assert summary.noise_false_hit_rate == 1.0
    assert "positive recall: 1/1" in render_score_summary(summary)
    receipts = [json.loads(line) for line in (dataset / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(receipts) == 3
    assert receipts[0]["path"].startswith("positive/")
    assert receipts[0]["model"] == "models/wake/scarlett.onnx"


def test_score_dataset_can_skip_receipts(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "samples"
    init_dataset(dataset)
    path = dataset / "negative" / "negative_20260531T171522_0001_test.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes((1000).to_bytes(2, "little", signed=True) * 1280)

    monkeypatch.setattr(
        wake_samples,
        "score_wake_audio",
        lambda *_args, **_kwargs: SimpleNamespace(
            hit=False,
            to_dict=lambda: {
                "wav_path": str(path),
                "hit": False,
                "max_score": 0.1,
                "threshold": 0.5,
            },
        ),
    )

    summary = score_dataset(load_config(path=tmp_path / "missing.toml"), dataset, threshold=0.5, write_receipts=False)

    assert summary.receipt_written is False
    assert not (dataset / "scores.jsonl").exists()
    assert "receipts: disabled" in render_score_summary(summary)


def test_synthetic_msd_generation_writes_negative_wavs_and_metadata(tmp_path) -> None:
    def fake_runner(command, **kwargs):
        assert command[:2] == ["msd", "render"]
        assert kwargs["check"] is True
        output = command[command.index("--output") + 1]
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes((1200).to_bytes(2, "little", signed=True) * 1600)
        return SimpleNamespace(returncode=0)

    summary = generate_synthetic_msd_samples(
        tmp_path / "synthetic",
        prompts=["starlet", "hey Charlotte"],
        count=3,
        tag="synthetic-hard-negative",
        voices=["Aiden", "Ryan"],
        languages=["English"],
        instructs=["neutral", "fast"],
        speed=1.1,
        runner=fake_runner,
    )

    assert summary.generated_count == 3
    wavs = sorted((tmp_path / "synthetic" / "negative").glob("*.wav"))
    assert len(wavs) == 3
    with wave.open(str(wavs[0]), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
    records = [json.loads(line) for line in (tmp_path / "synthetic" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[0]["label"] == "negative"
    assert records[0]["tag"] == "synthetic-hard-negative"
    assert records[0]["synthetic"] is True
    assert records[0]["generator"] == "msd"
    assert records[0]["voice"] == "Aiden"
    assert records[0]["language"] == "English"
    assert records[0]["instruct"] == "neutral"
    assert records[0]["speed"] == 1.1
    assert records[0]["command"] == "wake samples synthetic-msd"
    assert records[1]["voice"] == "Ryan"


def test_synthetic_negative_prompts_reject_exact_wake_words() -> None:
    with pytest.raises(ValueError, match="blocked wake word"):
        validate_synthetic_negative_prompts(["that color is scarlet"])
    with pytest.raises(ValueError, match="blocked wake word"):
        validate_synthetic_negative_prompts(["hey Scarlett"])


def test_prompt_and_name_helpers(tmp_path) -> None:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("# comment\nstarlet\nCharlotte\n\n", encoding="utf-8")
    assert safe_slug("Scarlet fever!") == "scarlet_fever"
    assert prompts_for_args(prompt="scarlett") == ["scarlett"]
    assert prompts_for_args(prompts_file=prompts) == ["starlet", "Charlotte"]
    assert "hey scarlett" in prompts_for_args(preset="scarlett")
    with pytest.raises(ValueError):
        prompts_for_args(prompt="scarlett", preset="scarlett")


def test_normalize_key_maps_session_controls() -> None:
    assert normalize_key("\n") == "enter"
    assert normalize_key("\r") == "enter"
    assert normalize_key(" ") == "space"
    assert normalize_key("q") == "q"
    assert normalize_key("\x03") == "ctrl-c"
