# Nalu

> Fully local, open-source vision agent for macOS. Sees, hears, speaks, acts. No cloud, no APIs, no recurring cost.

Nalu is a standalone AI assistant that runs entirely on your Mac. It captures the screen, reasons over what it sees with a local vision-language model, and executes mouse and keyboard actions to complete tasks. It can listen and speak via local speech models. Nothing leaves your machine.

## Status

Phase 2 shipped. Full closed-loop training pipeline: collect a dataset from real sessions with a leak-free run-level train/eval split, fine-tune a LoRA adapter, evaluate base vs adapter and compare side-by-side, then hot-swap the running daemon to the new adapter without restart. See [`BUILD_PLAN.md`](./BUILD_PLAN.md) for the full roadmap.

## Requirements

- Apple Silicon Mac (M-series). M3/M4/M5 Max with 32 GB+ unified memory recommended.
- macOS 14 or newer.
- ~10 GB free disk for models on first run.

## Install

```bash
git clone <this repo> ai_n4lu
cd ai_n4lu
uv sync
```

On first run, macOS will prompt for these permissions — all required:

- **Screen Recording** — so Nalu can see what you see.
- **Accessibility** — so Nalu can move the mouse and type.
- **Input Monitoring** — so Nalu's global pause hotkey works.
- **Microphone** — only if you use voice input.

## Run

```bash
uv run nalu start         # launches all agents + dashboard
uv run nalu dashboard     # dashboard only
uv run nalu ask "..."     # one-shot text query
```

Global hotkey **⌃⌥⌘.** instantly pauses all input dispatch. Press again to resume.

## Architecture

Stand-alone agents that coordinate over a local Unix domain socket bus. No agent depends on the network.

```
agents/
  vision/   UI-TARS-1.5-7B via MLX-VLM — screenshot + intent -> action
  voice/    Piper TTS (en_GB-alan) + faster-whisper STT
  planner/  perceive -> reason -> act loop, step caps, timeouts
  trainer/  eval harness + retrain recommendations
bus/        UDS pub/sub
capture/    PyObjC screen capture
actuator/   Quartz CGEvent dispatch + global pause hotkey
dashboard/  Streamlit, real data only
```

## Voice & ethics

Nalu uses a Jarvis-*inspired* persona — dry, formal, British, addresses the user as "sir." The voice itself is **original synthesis** from the open-source Piper TTS model `en_GB-alan-medium`. It is **not** a clone of Paul Bettany or any other actor. Distributing such a clone would violate copyright and likeness rights.

## License

Apache 2.0. See `LICENSE`.
