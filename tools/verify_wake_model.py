#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codex_voice_steer.config import load_config
from codex_voice_steer.wake import OpenWakeWordDetector, wake_readiness


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a Scarlett ONNX wake model through cxv's OpenWakeWord adapter.")
    parser.add_argument("model", nargs="?", default="models/wake/scarlett.onnx")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    config = load_config({"wake": {"model_path": str(model_path)}})
    readiness = wake_readiness(config)
    if not readiness.ok:
        print(f"blocked: {readiness.reason}")
        return 1
    OpenWakeWordDetector(config)
    print(f"ok: loaded {readiness.model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
