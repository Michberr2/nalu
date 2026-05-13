# Nalu

> Fully local, open-source vision agent for macOS. Sees, hears, speaks, acts. No cloud, no APIs, no recurring cost.

Nalu is a standalone AI assistant that runs entirely on your Mac. It captures the screen, reasons over what it sees with a local vision-language model, and executes mouse and keyboard actions to complete tasks. It can listen and speak via local speech models. Nothing leaves your machine.

## Status

Phase 4 shipped. Full closed-loop pipeline — capture, reason, act, learn, merge, swap. Includes a menu-bar shell, "Hey Jarvis" wake-word, and a guided first-run wizard. See [`BUILD_PLAN.md`](./BUILD_PLAN.md) for the full roadmap.

## Requirements

- Apple Silicon Mac (M-series). M3/M4/M5 Max with 32 GB+ unified memory recommended.
- macOS 14 or newer.
- ~10 GB free disk for models on first run.

## Install

```bash
git clone <this repo> nalu
cd nalu
uv sync
uv run nalu onboard
```

`nalu onboard` walks you through the required permissions, downloads the Piper voice and faster-whisper STT models, warms the vision model, and confirms a real screenshot decodes round-trip. It is idempotent — re-run any time.

Permissions Nalu will ask for:

- **Screen Recording** — so Nalu can see what you see (required).
- **Accessibility** — so Nalu can move the mouse and type (required).
- **Input Monitoring** — so the global pause hotkey works.
- **Microphone** — only if you use voice input or the wake-word.

## Run

```bash
uv run nalu serve              # start the daemon (model stays warm)
uv run nalu ask "..."          # one-shot text query
uv run nalu dashboard          # Streamlit dashboard
uv run nalu menubar            # macOS status-bar shell
```

Global hotkey **⌃⌥⌘.** instantly pauses all input dispatch. Press again to resume. **⌃⌥⌘+Space** is push-to-talk.

Set `NALU_WAKEWORD=1` to enable always-listening "Hey Jarvis" wake-word activation.

## CLI

```text
nalu doctor / setup            # check / open the right macOS Settings panes
nalu serve / stop / status     # daemon lifecycle
nalu ask "<task>"              # one-shot or via daemon
nalu speak / listen            # voice primitives
nalu dashboard                 # Streamlit UI

nalu train collect             # real session runs -> JSONL dataset (with eval split)
nalu train run <dataset>       # QLoRA fine-tune via mlx_vlm.lora
nalu train eval <dataset>      # action accuracy + click hit-rate report
nalu train activate <run>      # apply adapter; hot-swap a running daemon
nalu train deactivate          # revert to base
nalu train report              # dataset inventory + retraining recommendations

nalu model list / active       # base-model registry
nalu model register <id> <p>   # add a base model
nalu model use <id>            # set active + hot-swap running daemon
nalu model merge <repo>...     # mergekit pipeline + MLX quantize + register
nalu model merges              # past merge runs

nalu menubar                   # NSStatusBar shell (needs `nalu serve` running)
nalu wake [--keyword]          # standalone wake-word tester
nalu onboard [--yes]           # first-run wizard
```

## Architecture

Stand-alone agents that coordinate over a local Unix domain socket bus. No agent depends on the network.

```
agents/
  vision/    UI-TARS-1.5-7B via MLX-VLM, JSON-backed model registry,
             LoRA hot-swap, action parser hardened to UI-TARS / JSON /
             Python-dict / natural-language outputs.
  voice/     Piper TTS (en_GB-alan-medium) + faster-whisper STT,
             push-to-talk, openwakeword "Hey Jarvis" wake-word.
  planner/   perceive -> reason -> act loop with step caps, timeouts,
             and conversation history injected into the vision prompt.
  trainer/   QLoRA fine-tune runner, eval harness, side-by-side eval
             comparison, mergekit + MLX quantize pipeline.
bus/         UDS pub/sub (asyncio).
capture/     PyObjC continuous screen capture.
actuator/    Quartz CGEvent dispatch + global pause hotkey.
menubar/     NSStatusBar shell over a pure-Python state machine.
dashboard/   Streamlit, real data only.
onboarding   First-run wizard (permissions -> models -> round-trip).
```

State lives under `~/Library/Application Support/Nalu/` — see [`BUILD_PLAN.md`](./BUILD_PLAN.md) for the layout.

## Develop

```bash
uv sync
uv run pytest           # 130+ tests, all offline
uv run nalu serve       # in one terminal
uv run nalu dashboard   # in another
```

Contributions welcome — see [`CONTRIBUTING.md`](./CONTRIBUTING.md).

## Voice & ethics

Nalu uses a Jarvis-*inspired* persona — dry, formal, British, addresses the user as "sir." The voice itself is **original synthesis** from the open-source Piper TTS model `en_GB-alan-medium`. It is **not** a clone of Paul Bettany or any other actor. Distributing such a clone would violate copyright and likeness rights.

## License

Apache 2.0. See [`LICENSE`](./LICENSE).
