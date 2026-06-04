# codex-voice-steer

`codex-voice-steer` provides `cxv`, a local voice bridge for steering Codex from a microphone, wake word, typed text, or test audio.

It is built for a simple premise: keep the bridge local, inspectable, and user-owned. `cxv` listens, turns finalized speech into Codex input, shows a compact foreground status stream, and leaves the actual Codex work to Codex.

## What It Is

- A manually launched local CLI for voice-to-Codex sessions.
- A foreground interactive UI plus a small background daemon, both controlled by `cxv`.
- A wake-word, VAD, and STT pipeline using OpenWakeWord, Silero VAD, and MacParakeet by default.
- A typed route with `cxv text` and `cxv ttt` that uses the same delivery path as finalized speech.
- A project that is intentionally compatible with [`msd`](https://github.com/1Pio/mlx-speechd) for spoken Codex responses.

What it is not: an autostart service, launchd setup, container, VM, built-in TTS daemon, Hermes router, Codex patch, or hosted voice product.

## Current Shape

`cxv` has three everyday surfaces:

```bash
cxv                  # start the foreground listener/TUI
cxv text "status?"   # send typed input through the voice route
cxv --no-start status
```

The daemon is just `cxv serve` running in the background. Use `cxv up` and `cxv down` when you want to manage it explicitly.

```bash
cxv up
cxv status
cxv down
```

For low-latency voice turns, the biggest practical lever is usually the Codex turn settings, not the audio pipeline:

```bash
cxv --fast --effort minimal
cxv --fast --effort low text "check the current repo status"
```

## Setup

Start by cloning the repo and checking that the project runs from the checkout:

```bash
git clone https://github.com/1Pio/codex-voice-steer.git
cd codex-voice-steer
uv run --extra audio --extra wake cxv --help
```

For regular use, install the `cxv` command from the checkout so it is available on `PATH`:

```bash
uv tool install --editable ".[audio,wake]"
cxv --help
```

If `cxv` is not found after installation, let `uv` update your shell path and open a new terminal:

```bash
uv tool update-shell
```

Then create the default config:

```bash
cxv config init
cxv config edit
```

Optional, but recommended for spoken Codex responses: install [`msd`](https://github.com/1Pio/mlx-speechd) from its own repository and confirm that `msd say` works before enabling the MSD-aware `cxv` agent.

Then check local readiness:

```bash
cxv doctor
cxv audio devices
cxv wake test-audio /path/to/scarlett.wav
```

`cxv doctor` treats MSD as optional unless you explicitly configure it as required.

## Foreground UI

The foreground UI is designed to show the useful parts of a voice turn without flooding the terminal:

```text
15:37:33  wake detected
15:38:10  user: Scarlet?
15:38:10  sent: turn/start
15:38:16  codex action: command: /bin/zsh -lc "sed -n '1,220p' README.md"
15:38:19  codex msd: --text 'I will look at the screen now.' --instruct 'brief, warm, fast'
15:38:30  codex: Done.
15:38:30  turn completed: 019e...
```

By default, user-facing labels such as `user:`, `codex msd:`, and `codex:` are bold. Operational labels such as `sent:` and `turn completed:` stay plain.

## Session Management

CXV saves the Codex thread/session returned by `codex app-server` and resumes it for future voice and text turns.

```bash
cxv session status
cxv session new
cxv session new --force
```

`cxv session status` shows the saved thread, saved session id, effective resume target, relevant Codex config, and the current delivery behavior. `cxv session new` starts a fresh Codex thread immediately and saves it for future turns. If a turn is active, it refuses by default; pass `--force` to interrupt the active turn and replace the saved session.

If `codex.thread_id` or `codex.resume_thread_id` is set in config, CXV refuses `session new` because that pinned config would override the saved session. Unset the pinned key first:

```bash
cxv config unset codex.thread_id
cxv config unset codex.resume_thread_id
```

Useful UI settings:

```toml
[ui]
timestamp_opacity = 0.45
bold_labels = true
show_codex_tool_traces = true
show_codex_msd_traces = true
show_codex_final_answers = true
max_codex_action_lines = 1
max_codex_msd_lines = 40
max_codex_answer_lines = 200
visible_events = ["wake_detected", "stt_final", "user_final", "sent", "codex_tool_started", "codex_msd_started", "codex_final_answer", "turn_completed", "voice_error"]
hidden_events = []
```

`visible_events` is an allow-list when non-empty. `hidden_events` suppresses specific events. The same surface can be adjusted per invocation:

```bash
cxv --timestamp-opacity 0.45 --show-events wake_detected,user_final,sent,codex_msd_started,codex_final_answer,turn_completed
cxv --plain-labels
```

## MSD Compatibility 🎧

[`msd`](https://github.com/1Pio/mlx-speechd) is a recommended companion for `cxv`, especially if you want Codex to speak acknowledgements, important status updates, and final results. MSD is a local MLX speech daemon for fast text-to-speech on macOS: it can speak short messages with `msd say`, render audio files with `msd render`, keep a warm local model available, and let Codex use voice without turning `cxv` itself into a TTS server. That matters when you want a voice-first workflow where important Codex output is heard immediately, not only printed in a terminal.

It is not required by default. `cxv` should work as a voice-to-text-to-Codex bridge without MSD installed, and upstream defaults keep it optional:

```toml
[instructions.msd]
enabled = false
require_msd_on_path = false
```

When you do want spoken Codex responses, first install and configure [`msd`](https://github.com/1Pio/mlx-speechd) from its own repository, confirm that `msd say` works in your shell, then opt `cxv` into the MSD-aware Codex agent:

```bash
cxv agents install msd
cxv config set codex.agent cxv-voice-msd
cxv config set instructions.msd.enabled true
```

Only set `instructions.msd.require_msd_on_path = true` when you intentionally want `cxv doctor` to fail if `msd` is missing.

When Codex uses `msd say`, the interactive UI renders that action as `codex msd:` and shows only the arguments after the exact `msd say` command. This keeps spoken-response text readable while hiding shell wrappers such as `/bin/zsh -lc`.

## Controlled Audio Tests

For repeatable checks, use files or explicit loopback input. Speaker playback into the microphone is not a reliable receipt.

```bash
cxv audio devices
cxv config set audio.device "Loopback Input"
cxv wake test-audio /path/to/scarlett.wav
cxv voice test-audio /path/to/full-turn.wav
cxv voice test-audio /path/to/full-turn.wav --send
```

`cxv voice test-audio --send` runs the full route and sends the finalized transcript to Codex.

## Optional Wake Word Model Work

Normal `cxv` usage does not require collecting wake samples, generating synthetic negatives, or training a new wake model. The shipped Scarlett model is already bundled for the default wake word. This section is only for users who want to inspect, evaluate, or train their own compatible OpenWakeWord model, similar to how the bundled Scarlett model was produced.

### Real Wake Samples

Wake sample collection uses real microphone takes and keeps runtime source separate from training/evaluation tooling:

```bash
cxv wake samples init ./scarlett-samples
cxv wake samples session ./scarlett-samples --label positive --preset scarlett
cxv wake samples session ./scarlett-samples --label negative --prompt starlet --tag hard-negative
cxv wake samples session ./scarlett-samples --label noise --prompt keyboard --tag keyboard
cxv wake samples list ./scarlett-samples
cxv wake samples score ./scarlett-samples --model models/wake/scarlett.onnx --threshold 0.5
```

Good hard negatives include `starlet`, `Charlotte`, `star lit`, `let`, `start it`, ordinary speech, and natural near misses. Good noise samples include keyboard, fan, room noise, silence, and handling noise. Use `--keep-weak` for intentional silence or very quiet noise takes.

### Synthetic Negatives

Synthetic negative augmentation is optional and stays separate from real microphone samples. It uses `msd render` when MSD is installed, normalizes output to 16 kHz mono PCM16 WAV, and writes metadata that `cxv wake samples list` can read.

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

Synthetic prompts reject exact `scarlett` and `scarlet` tokens. The goal is to add realistic non-wake pressure, not to silently mix generated wake-word clips into the real dataset.

### Models And Training

Runtime remains OpenWakeWord-based. LiveKit tooling is used only for external training/evaluation support, not as a runtime dependency.

The bundled Scarlett model lives at:

```text
models/wake/scarlett.onnx
src/codex_voice_steer/resources/wake/scarlett.onnx
```

See `models/wake/README.md` and `docs/wakeword-livekit-training.md` for model receipts, training notes, and current limitations.

## Useful Commands

```bash
cxv --help
cxv config show
cxv agents list
cxv agents install slim
cxv agents install msd
cxv models
cxv doctor
```

Use `cxv down` to stop the daemon cleanly. Use `cxv --no-start status` when you only want to inspect state without starting anything.
