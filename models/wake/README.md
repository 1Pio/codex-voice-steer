# Scarlett Wake Model

V1 requires a real custom openWakeWord model at:

```text
models/wake/scarlett.onnx
```

Current receipt:

- `models/wake/scarlett.onnx` is a real LiveKit-trained ONNX classifier exported outside the V1 runtime environment.
- V1 runtime still uses OpenWakeWord. LiveKit is external training/evaluation tooling only.
- The model is also packaged under `codex_voice_steer/resources/wake/scarlett.onnx` so an installed PATH `cxv` can load it outside the repo.
- `cxv wake test-audio` verifies controlled 16 kHz mono PCM16 WAV fixtures through the same OpenWakeWord adapter.
- Current direct fixture smoke: positive generated Scarlett clip hit true at max score 0.644560 with threshold 0.55; negative generated clip hit false at max score 0.020178.
- `cxv doctor` passes with the packaged Scarlett model in the installed tool environment.

The first successful LiveKit run is useful but caveated: it skipped standalone background/RIR augmentation because LiveKit defaulted those paths relative to `./data`. `tools/livekit-wakeword/scarlett.yaml` now pins `/private/tmp/cxv-livekit-wakeword-data/backgrounds` and `/private/tmp/cxv-livekit-wakeword-data/rirs`; rerun train/eval with those paths before treating wake reliability as fully accepted.

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
