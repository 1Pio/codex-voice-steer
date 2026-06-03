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

From a separate training environment, not the `cxv` runtime environment. Keep the training venv, downloaded ACAV/features, RIRs, background audio, generated clips, and split manifest in the durable wake-sample cache so another run does not throw away the expensive setup:

```bash
brew install espeak-ng ffmpeg sox portaudio
python -m venv "$HOME/Documents/cxv-wake-samples/livekit-wakeword-cache-v1/venv"
"$HOME/Documents/cxv-wake-samples/livekit-wakeword-cache-v1/venv/bin/pip" install "livekit-wakeword[train,eval,export]"

"$HOME/Documents/cxv-wake-samples/livekit-wakeword-cache-v1/venv/bin/livekit-wakeword" setup --config tools/livekit-wakeword/scarlett.yaml
cxv wake training-status
```

Seed the LiveKit splits from real cxv samples, synthetic negatives, and mined false-hit crops before generation so manual positives and hard negatives are present in the training and validation distribution. Candidate15 is the current bundled high-recall model:

```bash
python tools/livekit-wakeword/seed_cxv_samples.py \
  --dataset "$HOME/Documents/cxv-wake-samples/scarlett-real-v1" \
  --dataset "$HOME/Documents/cxv-wake-samples/scarlett-synthetic-negatives-v1" \
  --dataset "$HOME/Documents/cxv-wake-samples/scarlett-real-false-hit-mined-candidates5-8-10-11-v1" \
  --dataset "$HOME/Documents/cxv-wake-samples/scarlett-real-false-hit-mined-candidates12-13-14-v1" \
  --dataset "$HOME/Documents/cxv-wake-samples/scarlett-real-false-hit-mined-candidate14-v1" \
  --dataset "$HOME/Documents/cxv-wake-samples/scarlett-synthetic-false-hit-mined-candidates5-8-10-11-v1" \
  --dataset "$HOME/Documents/cxv-wake-samples/scarlett-synthetic-false-hit-mined-candidates12-13-14-v1" \
  --output-dir "$HOME/Documents/cxv-wake-samples/livekit-wakeword-cache-v1/output" \
  --model-name scarlett \
  --train-on-all \
  --positive-train-repeat 25 \
  --negative-train-repeat 2 \
  --noise-train-repeat 2 \
  --tag-train-repeat hard-negative=20 \
  --tag-train-repeat synthetic-hard-negative=4 \
  --tag-train-repeat synthetic-multilingual=3 \
  --tag-train-repeat mined-false-hit=18 \
  --clear
```

`--train-on-all` keeps the deterministic test split but also places every real/manual, synthetic, and mined sample into training. The repeat flags intentionally weight manual positives and known hard negatives above generated filler clips. Later candidates tried heavier false-hit mining, ACAV sampling, and targeted positive-miss boosting; none beat candidate15's practical high-recall result.

```bash
"$HOME/Documents/cxv-wake-samples/livekit-wakeword-cache-v1/venv/bin/livekit-wakeword" generate tools/livekit-wakeword/scarlett.yaml
"$HOME/Documents/cxv-wake-samples/livekit-wakeword-cache-v1/venv/bin/livekit-wakeword" augment tools/livekit-wakeword/scarlett.yaml
"$HOME/Documents/cxv-wake-samples/livekit-wakeword-cache-v1/venv/bin/livekit-wakeword" train tools/livekit-wakeword/scarlett.yaml
"$HOME/Documents/cxv-wake-samples/livekit-wakeword-cache-v1/venv/bin/livekit-wakeword" export tools/livekit-wakeword/scarlett.yaml
"$HOME/Documents/cxv-wake-samples/livekit-wakeword-cache-v1/venv/bin/livekit-wakeword" eval tools/livekit-wakeword/scarlett.yaml
```

Expected exported model:

```text
/Users/main/Documents/cxv-wake-samples/livekit-wakeword-cache-v1/output/scarlett/scarlett.onnx
```

Copy only the final model into the repo if it is small enough and legally safe to commit:

```bash
cp "$HOME/Documents/cxv-wake-samples/livekit-wakeword-cache-v1/output/scarlett/scarlett.onnx" models/wake/scarlett.onnx
```

## Acceptance Checks

The exported model is not accepted just because the file exists. It must pass:

```bash
python tools/verify_wake_model.py models/wake/scarlett.onnx
cxv wake test-audio "$HOME/Documents/cxv-wake-samples/livekit-wakeword-cache-v1/output/scarlett/positive_test/clip_000000_r0.wav"
! cxv wake test-audio "$HOME/Documents/cxv-wake-samples/livekit-wakeword-cache-v1/output/scarlett/negative_test/clip_000000_r0.wav"
cxv doctor
cxv listen
```

For the current bundled candidate15 model, the deterministic reset-safe direct receipts are:

```text
real manual samples @0.5:      216/219 positives, 10/82 negative false hits, 0/60 noise false hits
synthetic negatives @0.5:      0/500 negative false hits
repo model before replacement: 0/219 positives at 0.5
```

The model therefore improves the old checked-in artifact materially, but it does not satisfy a perfect acceptance bar. The main blockers are real sample label/audio conflicts: `positive_20260602T140227_0154_voice.wav` transcribes like "That's it", `positive_20260602T140016_0112_voice.wav` remains low even after positive-miss boosting, and hard negatives such as `negative_20260602T143731_0035_hard_negative.wav` contain repeated near-wake words.

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
