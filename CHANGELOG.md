# Changelog

All notable changes to Nalu. Phases mirror [`BUILD_PLAN.md`](./BUILD_PLAN.md).

## Unreleased

### Added
- `nalu chat` — one-command voice + text chat UI. Starts the persistent daemon if it isn't already running, polls the pid file until ready (20s timeout), then opens the dashboard's Chat tab in the browser. The chat tab gains a "Record voice prompt" button (configurable 2–15s) that synchronously records via sounddevice, transcribes through faster-whisper, and auto-submits the transcript as if the user typed it; a "Speak the answer back" checkbox routes completed-task answers to Piper TTS in a fire-and-forget daemon thread so the UI never blocks on audio playback. Voice failures are caught and silently swallowed — chat keeps working without a microphone.
- `nalu onboard` polish — new disk-space pre-flight (`disk` step) catches "only 4 GB free" before mid-download failures with a fix-hint pointing at `NALU_HOME` for relocating the model cache to a larger volume; per-step download-size hints in the wizard headers (~30 MB / ~150 MB / ~5 GB) so users know what the long bar is doing on first run; screenshot smoke step now uses `VisionAgent.judge()` (raw text) instead of `decide()` (which parsed `Action.kind == "error"` from a free-form prompt and made a green path look broken). `OnboardingReport.is_ready` now explicitly excludes the optional steps (`tts`, `stt`, `wake`) so a user without a microphone can still ship.
- `nalu train fetch seeclick` — ingest a local SeeClick snapshot into our `(image, goal, action, args)` JSONL schema. Fully offline, pure-Python: takes an annotation file + images directory the user already has on disk and walks them. Normalizer accepts either `bbox` (uses center) or `point`, auto-detects normalized [0,1] vs absolute-pixel coords (PIL multiplies up when needed), clamps to image bounds, and maps `task_type` → `{click, type, scroll}` with conservative skips for unknown actions / missing image / missing target / missing goal. Writes `dataset.jsonl` + `summary.json` per run; both shapes (`*.jsonl` and top-level JSON list) supported. 16 unit tests, no network required at runtime or in the CLI.
- Planner stuck/loop detection (`LoopDetector`) — flags repeats and A-B-A-B alternations on a coarse-grained action signature; first signal injects a hint into the action history and skips dispatch (`stuck_detected` event), second consecutive signal emits `task_failed{reason: "stuck:..."}`. Prevents the agent from burning the step cap on the same wrong move.
- Planner action verification (`screen_change.perceptual_diff` + `evaluate_action_effect`) — compares the new screenshot against the prior shot and, when the previous effect-bearing action produced no observable change (≤0.005 mean pixel diff at 64×64 grayscale), appends a "your last action had no effect" hint to the action history and publishes `action_no_effect`. Lets the model self-correct from a missed click on the next turn instead of building on a false narrative.
- Planner completion verification (`verifier.verify_completion` + `VisionAgent.judge`) — when the model emits `done`, re-asks the same model with the current screenshot and the original goal, parses YES/NO + a one-sentence reason. On YES, publishes `task_completed`; on NO/ambiguous/error, publishes `completion_denied`, appends the verifier's reasoning to the action history, and continues the loop. Defaults to "not confirmed" for ambiguous output so a hallucinated completion can't slip through.
- Action-marker annotations (`annotate.draw_action_marker`) — saved screenshots now have a paired `step_NNN_decided.jpg` with a red ring at the click point (orange + inner dot for double-click), a yellow drag line, or a color-coded top banner for type / key / scroll / wait. Dashboard's Runs tab gains a "Show action markers" toggle and falls back to the raw frame for older runs.
- Run outcomes written back to `runs/<ts>/meta.json` (`status`, `reason`, `steps`, `answer`, `ended_ts`) at every terminal path of the planner loop.
- `dashboard.analytics.summarize_runs` + `coarse_failure_kind` + Overview "Failure modes" panel — completion rate, top failure mode, and a bar chart over the coarse breakdown (stuck / timeout / max_steps_exceeded / dispatch / vision / parse).
- Per-run timeline view (`dashboard.timeline.build_run_timeline`) joins `actions.jsonl` with the slice of `events.jsonl` inside the run's wall-clock window, dedups doubled `action_decided` entries, and renders a severity-coloured vertical timeline above the per-step screenshots in the Runs tab.
- Self-recovery on stuck/dispatch failure — planner now retries once (`MAX_RECOVERIES_PER_TASK = 1`) before publishing `task_failed` for `stuck:*` or dispatch exceptions: emits `task_recovering`, appends a `RECOVERY` line to history, resets the loop detector, and continues. `recoveries_used` is stamped into every terminal `meta.json` outcome.
- Click-jitter retry on no-effect (`agents.planner.jitter.jitter_click_args`) — when a click produces no observable change, the planner synthesizes a ±8 px jittered re-click (clamped to image bounds, forced to move at least 1 px) and dispatches it directly without spending a vision turn. Synthetic actions carry `synthetic: "jitter"` in `actions.jsonl`; the new `action_jittered` bus event carries from→to coords for the timeline.
- Per-run latency profile (`dashboard.latency.build_run_latency`) — per-step durations from `actions.jsonl` + `meta.json` plus aggregates (median, nearest-rank p95, total wall-clock, longest step), with sane fallbacks when meta is partial or corrupt.
- `double_click` and `drag` dispatch — the vision agent's `Action` normalizer already produced these kinds (UI-TARS native names: `left_double_click`, `drag`), but the planner had no `_dispatch` cases, so any double-click or drag emitted by the model failed the task with `dispatch: unknown action: ...`. `Actuator.drag(x1, y1, x2, y2, steps=20)` posts a real Quartz mouse-down → drag → mouse-up sequence; `dispatch_action` is now a free function so the routing is unit-testable with a fake actuator (15 new tests covering every kind).
- Bounds-aware action validation (`agents.planner.validate.validate_action`) — clicks / double-clicks / drags with coordinates outside the captured frame, missing/non-numeric `x`/`y`, or non-dict args are refused before reaching the actuator. Planner publishes `action_refused`, appends a `"step N: REFUSED -- ..."` self-correction line to history, and continues. History compaction's marker set extended with `REFUSED` so refusals fold into "Earlier:" summaries.
- Dataset quality filtering — `nalu train collect` now reads `meta.json` and gates run eligibility on the planner-stamped `status` field (only `completed` survives). Verifier-denied / stuck / timed-out runs that happened to emit a `done` action are now correctly excluded from training data. Legacy runs without a status field fall back to the prior "has any done action" heuristic.
- Action history compaction (`agents.planner.history.compact_history`) — once history exceeds 20 entries, older steps are folded into a single "Earlier: N prior steps (3× click, …) and K self-correction events." summary while the last 8 entries stay verbatim. The planner only compacts at the `vision.decide` boundary (raw `actions.jsonl` is unchanged), so the screenshot keeps the model's attention budget on long tasks.

## Phase 4 — UX polish — 2026-05-05

### Added
- `nalu menubar` — `NSStatusItem` shell over a pure-Python `MenubarState` machine subscribed to the bus. Ask, Pause/Resume, Model submenu (hot-swaps a running daemon), Recent conversation, dashboard launcher.
- `nalu wake` + `WakeWordRunner` driving openwakeword (default keyword `hey_jarvis`). On detection, reuses the push-to-talk capture path. Opt-in via `NALU_WAKEWORD=1`.
- `nalu onboard` — first-run wizard over discrete `OnboardingStep`s: permissions, TTS, STT, wake-word, vision warm-up, screenshot decode round-trip. Idempotent.

## Phase 3 — Weight merging & multi-model — 2026-05-05

### Added
- `nalu model merge <repo[@weight[:density]]>...` — drives `mergekit-yaml` then `mlx_vlm.convert -q`, writes `merge.yaml` + `summary.json` per run, supports linear / slerp / ties / dare_ties / dare_linear / task_arithmetic.
- `nalu model {list,active,register,unregister,use}` — JSON-backed registry at `models/registry.json`. Seeds with UI-TARS-1.5-7B-4bit. `nalu model use <id>` hot-swaps a running daemon via `vision_swap_model`.
- History-aware planner — daemon's conversation deque is filtered by timestamp and injected into the vision prompt as a `## Conversation` block. Stored in `meta.json` for replay/training.

## Phase 2 — Training pipeline — earlier

### Added
- `nalu train run` — QLoRA fine-tune via `mlx_vlm.lora`. Writes `adapters.safetensors` + `adapter_config.json` + `metrics.jsonl` per run.
- `nalu train activate / deactivate` — pointer-file activation. Mid-daemon hot-swap via `vision_swap_adapter`.
- `nalu train eval` — action-kind accuracy, click hit-rate @ 64 px, click MAE, text accuracy.
- `nalu train collect --eval-ratio` — deterministic *run-level* (not example-level) train/eval split.
- Eval comparison view — joins two `results.jsonl` files by `(run, step)` for head-to-head deltas.

## Phase 1 — End-to-end hardening — earlier

### Added
- Vision parser hardened — handles UI-TARS native, JSON, Python-dict, and natural-language outputs.
- `nalu serve` — persistent daemon keeps the model warm (~16s cold → ~5s warm).
- Voice loop — Piper TTS @ length_scale=0.85, faster-whisper STT, push-to-talk on ⌃⌥⌘+Space.
- TTS reads back `task_completed` answers in a background thread.
- `nalu doctor` / `nalu setup` — macOS permissions check.
- Chat tab in the dashboard.
- Live tab tails the events log.
- `nalu train collect` — runs → JSONL dataset.

## Phase 0 — Scaffold — earlier

### Added
- Project layout, `uv` env, Typer CLI.
- UDS message bus (pub/sub over Unix domain socket).
- PyObjC continuous screen capture.
- Quartz CGEvent actuator.
- MLX-VLM vision agent loading UI-TARS-1.5-7B-4bit.
- Planner: perceive → reason → act loop with step caps + timeouts.
- Streamlit dashboard scaffold.
- Global pause hotkey ⌃⌥⌘.
