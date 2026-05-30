# Scarlett Wake Model

V1 requires a real custom openWakeWord model at:

```text
models/wake/scarlett.onnx
```

Current blocker receipt:

- `openwakeword==0.6.0` installs and imports in the project venv.
- The package runtime is available after installing the optional wake extra.
- The repository does not yet contain a reliable trained `scarlett` model.
- `openwakeword.train` is not self-contained on this machine. Importing it initially failed on missing training-only dependencies, and the training CLI also expects a `piper_sample_generator_path` plus RIR/background datasets.
- A synthetic or placeholder model must not be shipped as the V1 wake model.

Do not treat this README as the model. `cxv doctor` remains blocked until the ONNX file exists and can be loaded.
