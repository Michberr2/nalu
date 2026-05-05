# Nalu — Build Plan & Status

Single source of truth for what's built, what's in flight, and what's next.
Update this file whenever a phase item ships.

Last updated: 2026-05-05

---

## Vision

A fully local, open-source vision agent for macOS. Sees, hears, speaks, acts.
No cloud APIs. No per-request cost. Apache 2.0.

**Hardware target:** M-series Mac with ≥32 GB unified memory (developed on M5 Max / 48 GB).
**Voice:** Jarvis-inspired persona via open-source British male TTS (Piper `en_GB-alan-medium`).
**License:** Apache 2.0.

---

## Phase 0 — Scaffold ✅ shipped

- [x] Project layout, `uv` env, Typer CLI
- [x] UDS message bus (pub/sub over Unix domain socket)
- [x] PyObjC continuous screen capture
- [x] Quartz CGEvent actuator
- [x] MLX-VLM vision agent loading UI-TARS-1.5-7B-4bit
- [x] Planner: perceive → reason → act loop with step caps + timeouts
- [x] Streamlit dashboard scaffold (Overview / Runs / Model)
- [x] Global pause hotkey ⌃⌥⌘.

## Phase 1 — End-to-end hardening ✅ shipped

- [x] **Vision parser hardened** — handles UI-TARS native, JSON, Python-dict, and natural-language outputs. 17 unit tests.
- [x] **Persistent daemon** — `nalu serve` keeps the model warm between asks (~16s cold → ~5s warm).
- [x] **Voice loop** — Piper TTS @ length_scale=0.85, faster-whisper STT, push-to-talk on ⌃⌥⌘+Space.
- [x] **TTS reads back** task_completed answers in background thread.
- [x] **macOS permissions check** — `nalu doctor` / `nalu setup` open the right Settings panes.
- [x] **Chat tab** — multi-turn UI in the dashboard, model stays loaded.
- [x] **Live tab** — tails the events log with optional 2 s auto-refresh.
- [x] **Dataset collection** — `nalu train collect` walks runs and emits a JSONL dataset of (screenshot, goal, action) triples.

## Phase 2 — Training pipeline 🚧 in progress

- [x] **QLoRA fine-tune runner** — `nalu train run <dataset>` consumes JSONL, calls `mlx_vlm.lora`, writes `adapters.safetensors` + `adapter_config.json` per run, streams metrics to `metrics.jsonl`.
- [x] **Adapter activation** — `nalu train activate <run>` writes a pointer file; `VisionAgent.load` applies it via `apply_lora_layers` on next daemon start. `nalu train deactivate` reverts to base.
- [x] **Eval harness** — `nalu train eval <dataset>` runs the active model over the dataset and reports action-kind accuracy, click hit-rate @ 64 px, click MAE, text accuracy. Run with adapter on/off to compare.
- [ ] **Adapter hot-swap mid-daemon** — currently requires daemon restart. Need to undo previous LoRA layers cleanly.
- [ ] **Eval comparison view** — dashboard panel showing base vs adapter side-by-side from `training/evals/`.
- [ ] **Train/eval split** — `collect` should optionally hold out N% for eval to prevent leakage.

## Phase 3 — Weight merging & multi-model 📋 planned

- [ ] **Mergekit pipeline** — merge UI-TARS-1.5-7B with OS-Atlas-7B, quantize, register as `nalu-merged-v1`.
- [ ] **Per-model registry** — `nalu model list` / `nalu model use <id>` to swap base models.
- [ ] **History-aware planner** — feed last N actions into the prompt (currently sends `(no prior steps)`).

## Phase 4 — UX polish 📋 planned

- [ ] **Menu-bar app** — always-available Jarvis surface (no Streamlit).
- [ ] **Voice activation phrase** — "Hey Nalu" via local wake-word model.
- [ ] **Onboarding flow** — first-run wizard: permissions → voice download → test screenshot → ready.
- [ ] **Public Apache 2.0 release** — README, contribution guide, signed binary distribution.

---

## Working set today

**You are here:** end of Phase 2, three items complete (runner / activation / eval), three items remaining.

Next pull: hot-swap mid-daemon **or** dashboard eval comparison view. Hot-swap is the higher-value one because every adapter change currently costs a `nalu stop && nalu serve` cycle (~10 s of model reload).

## Where things live on disk

```
~/Library/Application Support/Nalu/
├── runs/<ts>/                     # each ask's screenshots + actions.jsonl + meta.json
├── training/
│   ├── datasets/<ts>/             # dataset.jsonl + summary.json
│   ├── runs/<ts>/                 # adapters.safetensors + adapter_config.json + metrics.jsonl
│   ├── evals/<ts>/                # results.jsonl + summary.json
│   └── active_adapter             # text file → path of currently active run
├── models/                         # mlx_vlm cache
├── logs/events.jsonl              # tee of every bus event for the Live tab
└── run/{bus.sock, daemon.pid}
```

## CLI surface

```
nalu doctor / setup        # macOS permissions
nalu serve / stop / status # daemon
nalu ask "<task>"          # one-shot or via daemon
nalu speak / listen        # voice primitives
nalu dashboard             # Streamlit UI

nalu train collect                      # runs → dataset
nalu train run <dataset> [--lr/--rank]  # QLoRA fine-tune
nalu train eval <dataset> [--limit]     # accuracy report
nalu train activate <run>               # apply adapter on next daemon start
nalu train deactivate                   # revert to base
nalu train report                       # rec engine + dataset inventory
```
