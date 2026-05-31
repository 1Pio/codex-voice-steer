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
cxv wake samples score ./scarlett-samples --threshold 0.5
```

Useful hard negatives: `starlet`, `Charlotte`, `scarlet fever`, `let`, `start it`, `scale it`, normal speech. Useful environmental negatives: keyboard, fan, room noise, silence, and handling noise. Use `--keep-weak` for intentional silence or very quiet noise takes.
