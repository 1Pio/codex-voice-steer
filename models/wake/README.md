# Scarlett Wake Model

V1 requires a real custom openWakeWord model at:

```text
models/wake/scarlett.onnx
```

Current blocker receipt:

- `openwakeword==0.6.0` installs and imports in the project venv.
- The package runtime is available after installing the optional wake extra.
- The repository does not yet contain a reliable trained `scarlett` model.
- Official openWakeWord guidance says models process 16-bit 16 kHz PCM in 80 ms frames, custom model training uses synthetic wake-word clips plus negative data, and reliable evaluation needs false-reject and false-accept testing in realistic audio.
- `openwakeword.train` is not self-contained on this machine. Current direct import receipt:

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

- A direct `cxv listen` smoke now refuses to enable listening and reports the missing model instead of pretending wake detection is active.
- A synthetic or placeholder model must not be shipped as the V1 wake model.

Do not treat this README as the model. `cxv doctor` remains blocked until the ONNX file exists and can be loaded.
