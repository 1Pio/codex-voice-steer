# Scarlett Wake Model Training

V1 runtime stays OpenWakeWord. LiveKit wakeword is only an external training and evaluation path for producing the required classifier:

```text
models/wake/scarlett.onnx
```

LiveKit's current wakeword documentation says custom models export as standard ONNX files compatible with OpenWakeWord. Its training pipeline has six stages:

```text
setup -> generate -> augment -> train -> export -> eval
```

Do not add LiveKit imports or dependencies to `src/`. The only artifact this repo should consume at runtime is the exported ONNX classifier loaded by `openwakeword.model.Model`.

## Training Recipe

From a separate training environment, not the `cxv` runtime environment:

```bash
brew install espeak-ng ffmpeg sox portaudio
python -m venv /private/tmp/cxv-livekit-wakeword
/private/tmp/cxv-livekit-wakeword/bin/pip install "livekit-wakeword[train,eval,export]"

/private/tmp/cxv-livekit-wakeword/bin/livekit-wakeword setup --config tools/livekit-wakeword/scarlett.yaml
/private/tmp/cxv-livekit-wakeword/bin/livekit-wakeword run tools/livekit-wakeword/scarlett.yaml
/private/tmp/cxv-livekit-wakeword/bin/livekit-wakeword eval tools/livekit-wakeword/scarlett.yaml
```

Expected exported model:

```text
/private/tmp/cxv-livekit-wakeword-output/scarlett/scarlett.onnx
```

Copy only the final model into the repo if it is small enough and legally safe to commit:

```bash
cp /private/tmp/cxv-livekit-wakeword-output/scarlett/scarlett.onnx models/wake/scarlett.onnx
```

## Acceptance Checks

The exported model is not accepted just because the file exists. It must pass:

```bash
python tools/verify_wake_model.py models/wake/scarlett.onnx
cxv wake test-audio /private/tmp/cxv-livekit-wakeword-output/scarlett/positive_test/clip_000000_r0.wav
cxv doctor
cxv listen
```

Then run real live checks:

```text
1. Say "scarlett" from normal working distance.
2. Confirm foreground `cxv` logs wake detection, VAD final, STT final, and Codex send.
3. Speak non-wake distractors for several minutes and confirm no obvious false trigger.
4. Tune `wake.sensitivity` and `vad.speech_threshold` only from observed receipts.
```

Do not treat speaker playback into the microphone as a reliable live wake test. Use `cxv wake test-audio` for controlled file-based wake scoring, or route system audio into the configured input device with an explicit loopback device before treating playback as evidence.

## Sources

- LiveKit wakeword docs: https://docs.livekit.io/agents/multimodality/audio/wakeword/
- LiveKit wakeword repository: https://github.com/livekit/livekit-wakeword
- LiveKit wakeword launch post: https://livekit.com/blog/livekit-wakeword
- OpenWakeWord runtime: https://github.com/dscripka/openWakeWord
