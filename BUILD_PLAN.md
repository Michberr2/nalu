# Nalu — Build Plan & Status

Single source of truth for what's built, what's in flight, and what's next.
Update this file whenever a phase item ships.

Last updated: 2026-05-15 (Phase 7 interaction-model upgrades shipped)

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

## Phase 2 — Training pipeline ✅ shipped

- [x] **QLoRA fine-tune runner** — `nalu train run <dataset>` consumes JSONL, calls `mlx_vlm.lora`, writes `adapters.safetensors` + `adapter_config.json` per run, streams metrics to `metrics.jsonl`.
- [x] **Adapter activation** — `nalu train activate <run>` writes a pointer file; `VisionAgent.load` applies it via `apply_lora_layers` on next daemon start. `nalu train deactivate` reverts to base.
- [x] **Eval harness** — `nalu train eval <dataset>` runs the active model over the dataset and reports action-kind accuracy, click hit-rate @ 64 px, click MAE, text accuracy. Run with adapter on/off to compare.
- [x] **Adapter hot-swap mid-daemon** — `nalu train activate` publishes `vision_swap_adapter`; daemon reloads base + applies new LoRA in a thread, gated by a `VisionAgent` lock so swaps and asks queue safely. Dashboard buttons swap without restart.
- [x] **Train/eval split** — `nalu train collect --eval-ratio 0.2` deterministically partitions *runs* (not examples) into `train.jsonl` + `eval.jsonl` so frames from the same task can't leak across the boundary. Dashboard slider exposes the same.
- [x] **Eval comparison view** — dashboard panel joins two `results.jsonl` files by `(run, step)` and reports metric deltas, gain/regression tally, and per-action flip breakdown. Lets you confirm a fine-tune is helping where you expected, not just on average.

## Phase 3 — Weight merging & multi-model ✅ shipped

- [x] **Mergekit pipeline** — `nalu model merge <repo[@weight[:density]]>...` drives `mergekit-yaml` then `mlx_vlm.convert -q`, writes a deterministic `merge.yaml` + `summary.json` per run, and (with `--register-as`) auto-adds the result to the registry as `kind="merged"`. Supports linear / slerp / ties / dare_ties / dare_linear / task_arithmetic. 14 unit tests with subprocess injection.
- [x] **Per-model registry** — `nalu model {list,active,register,unregister,use}` backed by `models/registry.json`. Seeds with UI-TARS-1.5-7B-4bit on first read; `nalu model use <id>` updates the pointer and hot-swaps a running daemon via `vision_swap_model` → `VisionAgent.swap_model`. 14 unit tests.
- [x] **History-aware planner** — daemon's conversation deque is wired into Planner; `conversation_snapshot` filters turns strictly older than the current intent (max 6) and `format_conversation` injects them into the vision prompt as a `## Conversation` block. Stored alongside the goal in `meta.json` for replay/training.

## Phase 4 — UX polish ✅ shipped

- [x] **Menu-bar app** — `nalu menubar` runs an `NSStatusItem` shell that observes a pure-Python `MenubarState` machine subscribed to the bus. Provides Ask (osascript dialog → `user_intent`), Pause/Resume (`pause_request` ↔ daemon `PauseController`), Model submenu (`vision_swap_model`), Recent conversation, and dashboard launch. Cocoa view is a thin wrapper; state transitions are covered by 17 unit tests.
- [x] **Voice activation phrase** — `WakeWordRunner` driving `openwakeword` (default keyword `hey_jarvis`, fits the persona) with configurable threshold + cooldown. On detection it reuses the PTT capture path, so the same record→STT→`user_intent` flow runs whether the trigger was the hotkey or "Hey Nalu". Opt-in via `NALU_WAKEWORD=1`; `nalu wake` standalone tester verifies setup. 9 unit tests with stub spotter.
- [x] **Onboarding flow** — `nalu onboard` runs `OnboardingWizard` over discrete steps (permissions → TTS → STT → wake-word → vision → screenshot decode round-trip). Each step is independently testable via injected callables; required failures gate forward progress, optional ones (wake-word, mic) don't. Idempotent — re-running skips work that already passes. 10 unit tests cover step ordering, callback abort, exception trapping, readiness logic.
- [x] **Public Apache 2.0 release prep** — README scrubbed against shipped surface (every CLI command + opt-in env var documented), `CONTRIBUTING.md` and `CHANGELOG.md` added, `uv.lock` un-ignored so installs are reproducible, empty `scripts/` dir removed. Distribution path is `git clone && uv sync && nalu onboard` for now; signed binaries deferred until we hear from real users about packaging needs.

## Phase 5 — Smarter agent ✅ shipped

- [x] **Stuck/loop detection** — `LoopDetector` (pure-Python, 14 unit tests) tracks a coarse signature of recent actions (clicks bucketed to a 32 px grid) and flags two patterns: same action ≥3× in a row, or A-B-A-B... for ≥3 cycles. Planner consults it after every parsed action; first signal injects a hint into the action history and skips dispatch (publishes `stuck_detected`), second consecutive signal for the same signature emits `task_failed{reason: "stuck:<repeat|alternation>"}`. Stops the agent from burning the step cap reproducing the same wrong move; gives the model one chance to self-correct from the hint before bailing.
- [x] **Action verification** — `screen_change.perceptual_diff` downsamples both frames to 64×64 grayscale and returns mean pixel diff in [0, 1]; `evaluate_action_effect` flags effect-bearing kinds (click/double_click/type/key/scroll/drag) that produced no observable change. Planner compares each new screenshot against the prior shot and, when the previous action had no effect, appends a hint to the action history and publishes `action_no_effect`. Catches the complementary failure mode to stuck-detection: vision missed its target by a few pixels and the click landed nowhere. 13 unit tests with synthetic identical/inverted/half-split frames.
- [x] **Completion verification** — `verifier.build_verify_prompt` + `parse_verify_response` + `verify_completion` re-ask the model whether a `done` action is real, parsing a YES/NO + short reason from the first line. New `VisionAgent.judge(image, prompt)` returns raw text (no Action parsing) under the same lock as `decide`. Planner runs the verifier in a background thread when `done` fires; on YES it publishes `task_completed`, on NO/ambiguous/error it publishes `completion_denied`, appends the verifier's reasoning to the action history, and continues the loop. Defaults to "not confirmed" for empty/ambiguous output — false-confirms mislead the user, false-denies just keep the agent working. 13 unit tests cover prompt building, lead-token parsing across casing/punctuation/multi-line, and the judge-callable plumbing including exception handling.
- [x] **Action-marker annotations** — `annotate.draw_action_marker` overlays a kind-specific marker on the captured screenshot: red ring + dot for clicks, larger orange ring for double-clicks, yellow line for drags, color-coded top banner for type / key (with modifiers) / scroll / wait. Planner saves both `step_NNN.jpg` (clean, used as training input) and `step_NNN_decided.jpg` (annotated, surfaced in the dashboard). Runs tab gets a "Show action markers" toggle that transparently falls back to the raw frame for older runs. Closes the gap where you couldn't tell at a glance whether a click hit its target. 12 unit tests verify the marker is localized to the click region, drag lines pass through their midpoint, banners render only in the top BANNER_HEIGHT rows, and out-of-image coordinates don't crash.
- [x] **Run outcomes + failure analytics** — planner now stamps every terminal path back into `runs/<ts>/meta.json` with `status` (completed/failed/unknown), `reason` (timeout / max_steps_exceeded / stuck:repeat / stuck:alternation / parse: / dispatch: / vision:), `steps`, optional `answer`, and `ended_ts`. `dashboard.analytics.summarize_runs` aggregates these into a `RunsSummary` (totals, completion rate, failure-breakdown by raw reason, top failure, avg steps split by outcome) with `since_ts` filtering for "last N hours" panels. `coarse_failure_kind` collapses `stuck:repeat` and `stuck:alternation` into one bucket for charting. Dashboard Overview gains "Failure modes" — three KPI tiles (tasks finished, completion rate, top failure) plus a bar chart over the coarse breakdown. Triage finally has a one-glance answer to "what's actually going wrong across runs?". 16 unit tests cover meta filtering, corruption tolerance, since_ts cutoff, breakdown aggregation, and the coarse-bucket logic.
- [x] **Per-run timeline view** — `dashboard.timeline.build_run_timeline` joins `actions.jsonl` with the slice of `events.jsonl` that falls inside `[started_ts, ended_ts]` from `meta.json`, dedups `action_decided` events that already came from the actions log, and sorts chronologically. Returns `TimelineEntry`s with severity classification (success / warning / failure / info) for color-coding. Dashboard's Runs tab now renders a vertical timeline above the per-step screenshots — left-edge bar coloured by severity, monospace step+topic+summary so a glance tells you "agent got stuck at step 3, hint applied, then completion denied at step 8, finally completed at step 11." Bridges the gap between "I have screenshots" and "I understand what happened." 12 unit tests cover window filtering (events outside `[started_ts, ended_ts]` excluded), corrupted events log tolerance, action dedup, sort stability, severity classification, and graceful empty/malformed meta handling.
- [x] **Self-recovery on stuck/dispatch failure** — terminal `task_failed` paths for `stuck:repeat` / `stuck:alternation` and dispatch exceptions now consult a `recoveries_used` counter (capped at `MAX_RECOVERIES_PER_TASK = 1`) before bailing. On the first hit, the planner publishes `task_recovering`, appends `"step N: RECOVERY -- previous attempt got stuck (...): <hint>. Try a different approach."` to history, resets the loop detector, and continues; on the second hit (or any non-recoverable kind: timeout / max_steps / vision / parse) it fails as before. `recoveries_used` is stamped into every terminal-outcome dict in `meta.json` so analytics can split "completed straight through" vs "completed after one recovery." Recovery markers are folded into history compaction's self-correction count alongside SKIPPED / NO EFFECT / VERIFICATION DENIED / REFUSED / JITTERED RETRY.
- [x] **Click-jitter retry on no-effect** (`agents.planner.jitter.jitter_click_args`) — when the no-effect detector fires for a click / double-click, the planner synthesizes a jittered re-click (±8 px uniform offset, clamped to image bounds, forced to move at least one pixel even on a 0,0 RNG roll) and dispatches it directly without spending a vision turn. Tracked per-target via `jittered_for_step`, so we never jitter the same miss twice — if the jittered click also produces nothing, the model takes over on the following turn. Synthetic jitter actions are written to `actions.jsonl` with `synthetic: "jitter"` so the dataset filter and timeline can distinguish them from model-emitted actions, and a separate `action_jittered` bus event carries the from→to coords for the timeline. 12 unit tests cover non-mutation, max-offset bound, custom max-offset, edge clamping at (0,0) and (W-1,H-1), zero-roll forced-movement, missing-x/y pass-through, deterministic seeding, and float-coord coercion.
- [x] **Per-run latency profile** (`dashboard.latency.build_run_latency`) — joins `actions.jsonl` with `meta.json` and computes per-step durations (gap to next `action_decided.ts`, with the last step using `meta.ended_ts` when available), plus aggregates: median, nearest-rank p95, total wall-clock, and the longest step. `total_wall_ms` prefers `meta.ended_ts - meta.started_ts` but falls back to summed step durations when meta is corrupt or one of the timestamps is missing — same defensive shape as the analytics module. Skips records without a `ts` field and tolerates malformed JSONL lines without crashing. 12 unit tests cover empty runs, single-step, multi-step gaps, last-step duration with/without `ended_ts`, missing-meta fallback, corrupted meta/jsonl, longest-step selection, p95 with few samples, even-count median averaging, and clock-skewed (ended < started) meta clamping.
- [x] **`double_click` and `drag` dispatch** — closes a real bug: vision normalizes UI-TARS's `left_double_click` → `double_click` and `drag` → `drag`, but the planner's old `_dispatch` had no cases for either, so both fell through to `raise ValueError(f"unknown action: {kind}")` and the task ended in `dispatch:...` failure. `Actuator.drag(x1, y1, x2, y2, steps=20)` posts `kCGEventLeftMouseDown` → N intermediate `kCGEventLeftMouseDragged` events → `kCGEventLeftMouseUp` so the OS sees a real drag, not a teleport-and-release. `dispatch_action` extracted from the Planner class as a free function — same routing logic, but now testable with a fake actuator without spinning up bus / vision / capture. Both new kinds scale endpoint coords through `shot.scale_x` / `shot.scale_y` like clicks, so a `(100, 100)` drag-end on a 1.5× display lands at the right pixel. 15 new dispatch unit tests cover the routing for every kind (click / double_click / drag / type / key / scroll / wait / error / unknown), coord scaling, default arg handling, the wait-clamp-to-5s rule, and that `wait` never touches the actuator.
- [x] **Bounds-aware action validation** — `agents.planner.validate.validate_action(kind, args, w, h)` returns a `RefusalSignal` for clicks / double-clicks / drags whose coordinates fall outside the captured frame, are missing/non-numeric, or arrive on a non-dict args. Planner runs the check immediately after `vision.decide` returns and, on refusal, publishes `action_refused`, appends `"step N: REFUSED -- <hint>"` to history (with the actual offending coords + the screenshot's `WxH` so the model has enough to course-correct), and `continue`s without dispatching. The actuator never sees out-of-bounds coords, so we no longer rely on the no-effect detector to clean up after a hallucinated `(4096, 50)` click. Bool coords are explicitly rejected (Python's `bool ⊂ int` would otherwise let `True` slip through as `x=1`); float coords are accepted and truncated. `history._SELF_CORRECTION_MARKERS` extended with `REFUSED` so the new entries fold into compaction summaries the same as the other self-correction events. 16 unit tests cover edge-of-frame inclusivity, off-by-one out-of-bounds, far-out hallucinations, missing/non-numeric/bool coords, drag start vs end refusals, and pass-through for non-coord kinds and unknown actions.
- [x] **Dataset quality filtering by run outcome** — `nalu train collect`'s eligibility check now reads `meta.json` and prefers the planner-stamped `status` field over the old "has any done action" heuristic. A `done` that was verifier-denied → planner kept looping → run timed out → `status="failed"` is now correctly excluded from the JSONL dataset (the old heuristic would have kept it and trained the model on a hallucinated completion). Legacy runs without a `status` field fall back to the done-action heuristic so pre-Phase-5 data still flows. `_read_meta` reads the file once per run instead of twice (was once for eligibility, once for goal). 4 new unit tests cover: meta.status="failed" with done emitted is dropped; meta.status="completed" without done is kept; legacy no-status runs use the heuristic; `only_completed=False` overrides everything for raw inspection.
- [x] **History compaction for long tasks** — `agents.planner.history.compact_history` keeps the last 8 entries verbatim and folds older entries into a one-line "Earlier: N prior steps (3× click, 2× type) and 1 self-correction event." summary. `summarize_head` parses the `step N: kind …` / `step N: SKIPPED|NO EFFECT|VERIFICATION DENIED …` shapes the planner already emits, so no schema change. Planner swaps its raw `history` list for `compact_history(history)` only at the `vision.decide` call site — `actions.jsonl` still records every step verbatim, and the prompt stops dragging hundreds of stale tokens past the screenshot once tasks run 30+ steps. Defaults `COMPACT_AFTER=20`, `KEEP_TAIL=8` (overridable per-call). 12 unit tests cover threshold boundaries, kind counting, singular-vs-plural self-correction phrasing, only-self-correction omits the kind block, custom thresholds, `keep_tail=0`, unparseable entries, and input non-mutation.
- [x] **First-run wizard polish** — `nalu onboard` gains a disk-space pre-flight (`_step_disk`, `DISK_REQUIRED_GB = 8`) that fails before TTS/STT/vision pulls would die mid-stream and points at `NALU_HOME` as the relocate-to-bigger-volume escape hatch. Step headers now disclose first-run download sizes (~30 MB / ~150 MB / ~5 GB) so users know which bar is the long one. Screenshot smoke step calls `VisionAgent.judge()` (raw text round-trip) instead of `decide()` — the old free-form prompt produced `Action.kind == "error"` from the parser, which printed "vision returned error" on a green path and looked like a real failure. `OnboardingReport.is_ready` was tightened to explicitly exclude `tts`/`stt`/`wake` so a user on a machine without a microphone can still ship; the prior implementation only skipped `wake`. 12 onboarding tests including the two new disk-step branches.
- [x] **Public-corpus ingestion (`nalu train fetch seeclick`)** — pure-Python, fully-offline adapter that bridges a local SeeClick snapshot into our `(image, goal, action, args)` JSONL schema so the QLoRA pipeline isn't gated on the user's tiny session log. The user brings the data however they like (out-of-band download, USB drive, NAS sync) — Nalu itself never reaches out to a network. `agents.trainer.external.normalize_seeclick_record` accepts SeeClick's native shape (either `bbox` → take center, or `point`) and auto-detects whether coords are normalized [0,1] floats or absolute pixels (any value > 1.0 → pixels), multiplying by image dims via lazy PIL when needed and clamping to image bounds before emitting. `_action_kind_from_seeclick` maps `task_type` → `{click, type, scroll}` and returns "" for unknown kinds so the caller can skip cleanly. `iter_seeclick_records` handles both `*.jsonl` and top-level JSON arrays — both shapes appear in the wild for SeeClick subsets — and silently skips malformed lines. `fetch_seeclick(annotation_path, images_root, *, out_dir, limit)` walks the corpus, writes `dataset.jsonl` + `summary.json` with examples_in/out + per-skip-reason counters + per-action histogram. 16 unit tests run fully offline with PIL-generated tiny PNG fixtures — cover normalized/absolute coord detection, bbox center, image-bounds clamping, missing-image / missing-target / missing-goal / unknown-action skips, type / scroll / click action mapping, both annotation file shapes, and the limit cap.

## Phase 6 — Hierarchical planning ✅ shipped

- [x] **LLMDecomposer (text-only MLX-LM)** — `agents.planner_llm` wraps a local mlx-lm chat model (default `mlx-community/Qwen2.5-7B-Instruct-4bit`, Apache 2.0, configurable via `NALU_PLANNER_LLM`). Lazy load, threading lock so concurrent decompose/replan calls from the daemon queue safely, deterministic temp=0 decoding, tight `max_tokens=512`. `_generate(system, user)` is the single monkeypatch seam tests use to avoid pulling weights. Both `decompose(goal, conversation="")` and `replan(*, original_goal, completed, failed_subgoal, failure_reason, screen_summary="")` always return a non-empty `Plan` — model errors degrade to a one-subgoal fallback rather than dropping the task.
- [x] **Tolerant Plan / Subgoal parser** — `Plan(subgoals, raw, fallback)` + frozen `Subgoal(goal, success_criteria="")`. `parse_plan_response` extracts the first plausible JSON value (fenced ```json block → balanced `[...]` → balanced `{...}` with string-aware brace tracking), unwraps dict-wrapped arrays under `plan`/`subgoals`/`steps`/`tasks` keys, coerces strings-as-subgoals, accepts field aliases (`goal`/`task`/`step`/`description`, `success_criteria`/`done_when`/`success`), and falls back to numbered/bulleted prose parsing before giving up. Last resort returns `[Subgoal(goal=fallback_goal)]` with `fallback=True` so the orchestrator can log degradation without failing the task. Never fatal.
- [x] **Constrained decomposer prompts** — `DECOMPOSE_SYSTEM` enforces 1–6 subgoals, each ≤4 UI steps (matches UI-TARS's planning horizon — longer subgoals reliably loop), visible-name only ("don't invent UI affordances"), JSON-array-only output, and a retrieval-task convention ("the LAST subgoal should be: 'Report the value … as the final answer.'"). `REPLAN_SYSTEM` forces the first subgoal to address the failure (close modal / switch apps / scroll into view) and forbids repeating completed subgoals. Prompts isolated from the decomposer so they're unit-testable without MLX.
- [x] **Planner integration (subgoal queue + replan-on-failure)** — `Planner.__init__` accepts an optional `decomposer`; `_make_plan` calls it (gated by `config.USE_LLM_PLANNER`) and falls back to single-subgoal mode on exception or empty result. `SubgoalOutcome` tracks per-subgoal status; loop dequeues `(Subgoal, source)` pairs where source ∈ `{plan, replan}`, runs the existing vision loop against `_compose_effective_goal`, and on failure consults `_replan_after_failure(original_goal, completed, failed_subgoal, reason)` to extend the queue with a corrected continuation. Plan publishes `plan_decomposed` event with subgoal list for the timeline.
- [x] **Daemon opt-in load + onboarding step** — `daemon.py` instantiates `LLMDecomposer` only when `USE_LLM_PLANNER` is set, awaits `load()` in a thread (publishes `planner_llm_loaded` / `planner_llm_load_failed`), and degrades to `decomposer=None` on failure so the vision-only path still works. `onboarding._step_planner_llm` is an *optional* step (~5 GB download disclosed in header), excluded from `is_ready` so users without the planner can still ship. CLI gains `nalu planner` Typer app with `decompose <goal>` for offline plan inspection and `model` for showing the active planner model id.
- [x] **Test coverage** — 174 lines for the parser (fenced/embedded/dict-wrapped/string/bullet/empty/None/non-string-raw inputs), 160 lines for the decomposer (monkeypatched `_generate`, empty goal short-circuit, replan plumbing, exception → fallback, log line shapes). MLX is never imported during tests.

## Phase 7 — Interaction model ✅ shipped

Inspired by Thinking Machines' "interaction models" framing: the agent's *felt* latency depends on the perceive↔act tick rate and on having something to say while it works, not just raw model speed.

- [x] **Active screen-stabilization wait** — `agents/planner/settle.py` replaces post-action `asyncio.sleep(0.4)` with a poll loop that watches consecutive captured frames and exits when the screen has visibly stopped changing (perceptual_diff ≤ 0.005 across two consecutive 80ms polls) or hits a 1.5s cap. Heavy menus / new tabs get the time they actually need; trivial clicks don't burn 400ms doing nothing. Publishes `screen_settled` events with elapsed_ms + stable flag so the dashboard can chart per-step latency. Falls back to fixed sleep when continuous capture isn't wired (unit tests / headless). 6 unit tests with an injected sleeper.
- [x] **Dual-model responder** — `agents/responder.py` adds a `Responder` that subscribes to `user_query` and produces short conversational replies using the same Qwen LLM the planner loaded — separate request lock so a slow decompose doesn't block a "what time is it?". Daemon classifies voice/PTT transcripts via a heuristic router (`classify_user_text`): question prefixes and short turns → `user_query`; action verbs and long instructions → `user_intent`. Replies are spoken via the existing TTS path and appended to conversation history. 13 responder + 17 router tests.
- [x] **Proactive Jarvis speech** — `agents/voice/proactive.py` adds an opt-in `ProactiveSpeaker` that listens on the bus and emits short Jarvis-style status quips while the agent works ("Working on it.", "That didn't land. Trying again."). Per-event and global cooldowns prevent chatter on bursty events. Gated behind `NALU_PROACTIVE_VOICE` so quiet operation stays default. Phrase pool config-driven; speak_fn + clock injected so tests run without Piper / wall clock. 10 unit tests covering enable gate, cooldowns, RNG phrase choice, post-run toggle.
- [x] **Always-on STT with VAD** — `agents/voice/always_on.py` adds an opt-in `AlwaysOnRunner` that listens continuously, gates audio through silero-vad, and transcribes each detected utterance the moment silence resumes. Pure-Python state machine — audio source, VAD, and transcriber are injected callables, so tests run without sounddevice, torch, or faster-whisper. Barge-in is a drop-policy: `is_muted()` True discards audio and resets any partial utterance. silero-vad load failure degrades gracefully ("everything is speech" passthrough) so a missing model never blocks daemon startup. Gated behind `NALU_ALWAYS_ON_STT`. 9 unit tests.
- [x] **Interaction-quality evals** — `dashboard/interaction.py` adds a metrics module that complements the existing per-step latency profile with how Nalu *feels* to use: median TTFA (user_intent → first action_decided), median TTFR (user_query → responder_reply), median + p95 screen_settle elapsed_ms, settle stability rate, median + p95 inter-step gap (chain resets at each user_intent so idle time between tasks isn't miscounted), and proactive-utterance count. Pure-Python so the metrics can be reused in CLI eval scripts; rendered as a new Interaction section on the Overview tab. 11 unit tests.

---

## Working set today

**You are here:** Phases 0–7 all shipped. Phase 5 closed out the self-correcting executor. Phase 6 added the hierarchical planning layer (Qwen2.5-7B-Instruct-4bit decomposes the goal, vision executes each subgoal, replan on failure). Phase 7 reshaped the felt-latency profile: active screen-settle replaces a fixed sleep, a dual-model responder splits conversational turns from screen actions on the same Qwen LLM, proactive Jarvis quips fill the gap between task_started and task_completed, an always-on VAD-gated STT path complements push-to-talk, and the dashboard now reports TTFA/TTFR/settle/inter-step-gap so regressions are visible. 410 unit tests, all passing. Public repo at https://github.com/Michberr2/nalu.

**Next pull:** real-user feedback loop. Watch what people hit on first run, what tasks they actually try, and what the planner+executor get wrong. Likely Phase 8 candidates: signed-binary distribution if packaging is real friction, custom "hey nalu" wake-word model (currently `hey_jarvis` placeholder), curated dataset shipped with releases so the LoRA pipeline is useful day-one, and a subgoal-level eval harness that mirrors the existing per-step `nalu train eval` but scores plans end-to-end.

## Where things live on disk

```
~/Library/Application Support/Nalu/
├── runs/<ts>/                     # each ask's screenshots + actions.jsonl + meta.json
├── training/
│   ├── datasets/<ts>/             # dataset.jsonl + summary.json
│   ├── runs/<ts>/                 # adapters.safetensors + adapter_config.json + metrics.jsonl
│   ├── evals/<ts>/                # results.jsonl + summary.json
│   ├── merges/<ts>/               # merge.yaml + merged/ + mlx/ + summary.json
│   └── active_adapter             # text file → path of currently active run
├── models/
│   ├── registry.json               # base-model registry + active pointer
│   └── (mlx_vlm cache)             # downloaded weights
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

nalu model list / active                # registry inventory
nalu model register <id> <path>         # add a base model to the registry
nalu model use <id>                     # set active + hot-swap running daemon
nalu model unregister <id>              # drop a non-active entry
nalu model merge <repo>... [--method]   # mergekit + MLX quantize + register
nalu model merges                       # past merge runs

nalu menubar                            # macOS NSStatusBar shell (needs running daemon)
nalu wake [--keyword] [--threshold]     # standalone wake-word tester
nalu onboard [--yes]                    # first-run wizard

nalu planner model                      # show active planner LLM id
nalu planner decompose "<goal>"         # offline plan inspection (no daemon needed)
```
