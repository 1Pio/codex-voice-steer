# Scarlett Wake Model

V1 requires a real custom openWakeWord model at:

```text
models/wake/scarlett.onnx
```

Current receipt:

- `models/wake/scarlett.onnx` is candidate15 from the 2026-06-03 LiveKit training run, exported outside the V1 runtime environment.
- V1 runtime still uses OpenWakeWord. LiveKit is external training/evaluation tooling only.
- The model is also packaged under `codex_voice_steer/resources/wake/scarlett.onnx` so an installed PATH `cxv` can load it outside the repo.
- `cxv wake test-audio` verifies controlled 16 kHz mono PCM16 WAV fixtures through the same OpenWakeWord adapter.
- SHA-256: `c07cf631660e73ef355c6fc3e941dbd472f8ba80e7c5dfb899aaf69f52c4191e`
- Fresh deterministic reset-safe score receipts with `--model models/wake/scarlett.onnx` and threshold `0.5`:
  - `/Users/main/Documents/cxv-wake-samples/livekit-wakeword-cache-v1/output/final-repo-candidate15-real-threshold-05-deterministic.jsonl`
  - `/Users/main/Documents/cxv-wake-samples/scarlett-real-v1`: `216/219` positives, `10/82` negative false hits, `0/60` noise false hits.
  - `/Users/main/Documents/cxv-wake-samples/livekit-wakeword-cache-v1/output/final-repo-candidate15-synthetic-threshold-05-deterministic.jsonl`
  - `/Users/main/Documents/cxv-wake-samples/scarlett-synthetic-negatives-v1`: `0/500` negative false hits.
- This is not a perfect model. Candidate12, 15, and 17-21 were preserved under `/Users/main/Documents/cxv-wake-samples/livekit-wakeword-cache-v1/output/scarlett-candidate*`; candidate15 remains the best high-recall anchor.

Earlier training skipped standalone background/RIR augmentation because LiveKit defaulted those paths relative to `./data`. `tools/livekit-wakeword/scarlett.yaml` now pins durable cache paths under `/Users/main/Documents/cxv-wake-samples/livekit-wakeword-cache-v1/data`; the current model was exported after that corrected rerun.

Known sample blockers:

- `positive_20260602T140227_0154_voice.wav` scores near zero across candidates and transcribes like "That's it", so treating it as a true wake positive forces a bad model tradeoff.
- `positive_20260602T140016_0112_voice.wav` remains low even after targeted positive-miss repeat training.
- Hard negatives such as `negative_20260602T143731_0035_hard_negative.wav` contain repeated near-wake words and remain realistic collisions.

Historical local openWakeWord training blocker:

- `openwakeword.train` was not self-contained on this machine. Direct import receipt:

```text
ModuleNotFoundError: No module named 'torchinfo'
```

- Inspecting the installed training CLI shows it also expects a `piper_sample_generator_path`, RIR paths, background audio paths, positive/negative sample generation, and feature/training stages.
- Direct package spike:
  - `uv pip install torchinfo torchmetrics pyyaml piper-sample-generator` advanced the import chain but `openwakeword.train` next failed on `pronouncing`.
  - `uv pip install pronouncing` advanced it again and next failed on `torch_audiomentations`.
  - `uv pip install "piper-tts[train]" torch-audiomentations` advanced it again and next failed on `speechbrain`.
  - `piper-sample-generator==3.2.0` from PyPI failed because `piper_train` was missing.
  - `piper-tts-plus` still did not provide `piper_train` and changed the `piper` import surface enough that `piper_sample_generator` failed on `SynthesisConfig`.
  - The project venv was restored to declared dependencies with `uv sync --extra test --extra audio --extra wake`.
- Repro command:

```bash
cxv wake training-status
```

Do not treat speaker playback into the microphone as reliable wake evidence. Use `cxv wake test-audio` for controlled file scoring, or route system audio into the configured input with an explicit loopback device before treating playback as a wake receipt.
