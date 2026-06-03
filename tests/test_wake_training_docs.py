from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_livekit_recipe_is_kept_outside_runtime_source() -> None:
    doc = ROOT / "docs" / "wakeword-livekit-training.md"
    config = ROOT / "tools" / "livekit-wakeword" / "scarlett.yaml"
    seeder = ROOT / "tools" / "livekit-wakeword" / "seed_cxv_samples.py"
    verifier = ROOT / "tools" / "verify_wake_model.py"
    assert doc.exists()
    assert config.exists()
    assert seeder.exists()
    assert verifier.exists()
    assert "models/wake/scarlett.onnx" in doc.read_text()
    assert "cxv wake test-audio" in doc.read_text()
    assert 'target_phrases:\n  - "scarlett"' in config.read_text()
    assert "/Users/main/Documents/cxv-wake-samples/livekit-wakeword-cache-v1/data/backgrounds" in config.read_text()
    assert "seed_cxv_samples.py" in doc.read_text()
    assert "--train-on-all" in doc.read_text()
    assert "model_size: small" in config.read_text()


def test_runtime_dependencies_do_not_include_livekit() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text().lower()
    src = "\n".join(path.read_text().lower() for path in (ROOT / "src").rglob("*.py"))
    assert "livekit" not in pyproject
    assert "livekit" not in src
