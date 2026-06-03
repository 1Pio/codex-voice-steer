# codex-voice-steer

`codex-voice-steer` provides the `cxv` command: a local, manually launched voice-to-Codex bridge.

V1 scope is intentionally local and user-owned:

- `cxv` starts the foreground mini TUI/listener.
- `cxv serve` is the daemon process. There is no separate daemon binary.
- `cxv text` and `cxv ttt` send typed input through the same Codex delivery route as finalized speech.
- Wake word is `scarlett` by default.
- STT defaults to finalized MacParakeet segments, not fake streaming.
- No launchd, cron, autostart service, containers, VMs, Codex patches, built-in TTS, Hermes mode, or Hermes routing.

The current implementation includes the CLI, config, daemon lifecycle, foreground listener, typed Codex route, app-server client, doctor, model catalog, bundled Codex agent templates, OpenWakeWord Scarlett model, Silero VAD, and MacParakeet finalized STT.

Controlled audio receipts should use file or explicit loopback input, not speaker playback into the microphone:

```bash
cxv audio devices
cxv config set audio.device "Loopback Input"
cxv wake test-audio /path/to/scarlett.wav
cxv voice test-audio /path/to/full-turn.wav
cxv voice test-audio /path/to/full-turn.wav --send
```

Wake sample collection for retraining/evaluation uses real microphone takes and keeps LiveKit out of runtime source:

```bash
cxv wake samples init ./scarlett-samples
cxv wake samples session ./scarlett-samples --label positive --preset scarlett
cxv wake samples session ./scarlett-samples --label negative --prompt starlet --tag hard-negative
cxv wake samples session ./scarlett-samples --label noise --prompt keyboard --tag keyboard
cxv wake samples list ./scarlett-samples
cxv wake samples score ./scarlett-samples --model models/wake/scarlett.onnx --threshold 0.5
```

Useful hard negatives: `starlet`, `Charlotte`, `star lit`, `let`, `start it`, `scale it`, normal speech. Useful environmental negatives: keyboard, fan, room noise, silence, and handling noise. Use `--keep-weak` for intentional silence or very quiet noise takes.

Synthetic negative augmentation stays separate from real microphone samples and is optional. It uses `msd render` when `msd` is installed, normalizes each result to 16 kHz mono PCM16 WAV, and writes metadata that `cxv wake samples list` can read:

```bash
cxv wake samples synthetic-msd "$HOME/Documents/cxv-wake-samples/scarlett-synthetic-negatives-v1" \
  --prompts tools/prompts/wake-negative-hard.txt \
  --prompts tools/prompts/wake-negative-normal.txt \
  --tag synthetic-negative \
  --voices Aiden,Ryan,Vivian,Serena,Dylan,Eric,Uncle_Fu \
  --languages English,German,French,Spanish \
  --instructs "neutral, clear, natural" "fast, casual" "quiet, close microphone" \
  --count 500
```

Synthetic negatives reject exact `scarlett` and `scarlet` prompt tokens. To evaluate a manual dataset without modifying it, use:

```bash
cxv wake samples score "$HOME/Documents/cxv-wake-samples/scarlett-real-v1" --model models/wake/scarlett.onnx --threshold 0.5 --no-receipt
```

`cxv wake samples score` uses configured `wake.model_path` unless `--model` is passed. The bundled Scarlett model is the best high-recall candidate from the current local training run, not a perfect model. Fresh deterministic reset-safe scoring at `wake.sensitivity = 0.5` on `/Users/main/Documents/cxv-wake-samples/scarlett-real-v1` hits `216/219` manual positives, with `10/82` manual negative false hits and `0/60` manual noise false hits. The 500-sample synthetic negative set scores `0/500` false hits. The remaining misses and hard-negative collisions are documented in `models/wake/README.md`.
