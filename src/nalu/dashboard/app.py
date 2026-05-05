from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from .. import config, daemon
from ..agents.trainer import TrainerAgent
from ..bus import BusClient

st.set_page_config(page_title="Nalu", layout="wide")

st.title("Nalu — training & inspection")
st.caption("Fully local. Real data only. If a panel is empty, no data exists yet.")

config.ensure_dirs()

tab_chat, tab_live, tab_overview, tab_runs, tab_train, tab_model = st.tabs(
    ["Chat", "Live", "Overview", "Runs", "Training", "Model"]
)


async def _send_intent(text: str, timeout: float) -> tuple[bool, str]:
    pub = BusClient(source="dashboard")
    sub = BusClient(source="dashboard-listener")
    await pub.connect()
    await sub.connect()
    done = asyncio.Event()
    result = {"ok": False, "answer": ""}

    async def on_terminal(ev):
        if ev.topic in ("task_completed", "task_failed"):
            result["ok"] = ev.topic == "task_completed"
            result["answer"] = ev.payload.get("answer", "") or ev.payload.get("reason", "")
            done.set()

    await sub.subscribe("task_completed", on_terminal)
    await sub.subscribe("task_failed", on_terminal)
    await pub.publish("user_intent", {"text": text, "via": "dashboard"})
    try:
        await asyncio.wait_for(done.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        result["answer"] = "(timeout waiting for daemon)"
    finally:
        await pub.close()
        await sub.close()
    return result["ok"], result["answer"]


def _recent_conversation(max_items: int = 20) -> list[dict]:
    runs = sorted([p for p in config.RUNS_DIR.glob("*") if p.is_dir()], reverse=True)
    out: list[dict] = []
    for run in runs:
        ap = run / "actions.jsonl"
        if not ap.exists():
            continue
        records = [json.loads(line) for line in ap.read_text().splitlines() if line.strip()]
        if not records:
            continue
        terminal = next(
            (r for r in reversed(records) if r["action"] in ("done", "error") or "answer" in r.get("args", {})),
            records[-1],
        )
        steps = len(records)
        answer = terminal.get("args", {}).get("answer") or terminal.get("reason", "")
        out.append({"run": run.name, "steps": steps, "answer": answer})
        if len(out) >= max_items:
            break
    return out


def _read_recent_events(n: int) -> list[dict]:
    if not config.EVENTS_LOG.exists():
        return []
    lines = config.EVENTS_LOG.read_text().splitlines()
    out = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


with tab_live:
    pid = daemon.daemon_pid()
    if pid is None:
        st.warning("Daemon is not running. Start with `nalu serve` to stream live events.")
    else:
        st.caption(f"Tailing {config.EVENTS_LOG}")

    col_a, col_b, col_c = st.columns([1, 1, 4])
    n = col_a.number_input("Show last", min_value=10, max_value=2000, value=200, step=50)
    auto = col_b.checkbox("Auto-refresh (2s)", value=False)

    events = _read_recent_events(n)
    if not events:
        st.info("No events yet — run `nalu ask <task>` to generate some.")
    else:
        df = pd.DataFrame(
            [
                {
                    "ts": pd.to_datetime(e["ts"], unit="s").strftime("%H:%M:%S"),
                    "topic": e["topic"],
                    "source": e.get("source", ""),
                    "payload": json.dumps(e.get("payload", {}), default=str)[:120],
                }
                for e in events
            ]
        )
        st.dataframe(df, use_container_width=True, height=560)

    if auto and pid is not None:
        import time as _time

        _time.sleep(2.0)
        st.rerun()


with tab_chat:
    pid = daemon.daemon_pid()
    if pid is None:
        st.warning("Daemon is not running. Start it with `nalu serve` in a terminal to chat.")
    else:
        st.caption(f"Daemon: pid {pid} — model stays loaded between turns.")

    for entry in reversed(_recent_conversation()):
        with st.chat_message("user"):
            st.write(f"_(run {entry['run']} — {entry['steps']} step{'s' if entry['steps'] != 1 else ''})_")
        with st.chat_message("assistant"):
            st.write(entry["answer"] or "(no terminal answer)")

    prompt = st.chat_input("Tell Nalu what to do…", disabled=pid is None)
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Nalu is working…"):
                ok, answer = asyncio.run(_send_intent(prompt, config.PLANNER_TASK_TIMEOUT_S))
            st.write(answer if answer else ("done" if ok else "failed"))
        st.rerun()

with tab_overview:
    trainer = TrainerAgent()
    metrics = trainer.collect_metrics()
    rec = trainer.recommend()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Logged runs", metrics.get("runs", 0))
    c2.metric("Completed", metrics.get("completed", 0))
    c3.metric("Success rate", f"{metrics.get('success_rate', 0.0):.0%}" if metrics.get("runs") else "—")
    c4.metric("Avg steps / task", f"{metrics.get('avg_steps', 0):.1f}" if metrics.get("runs") else "—")

    st.subheader("Should I retrain?")
    if rec.should_retrain:
        st.error("Yes — retraining recommended.")
    elif metrics.get("runs", 0) == 0:
        st.info("No runs yet. Use Nalu to generate real session data, then come back.")
    else:
        st.success("Not yet. Performance is within bounds.")

    if rec.reasons:
        st.markdown("**Reasons:**")
        for r in rec.reasons:
            st.write(f"- {r}")
    if rec.suggested_data:
        st.markdown("**Data to collect:**")
        for s in rec.suggested_data:
            st.write(f"- {s}")

    if metrics.get("action_counts"):
        st.subheader("Action distribution")
        df = pd.DataFrame({"action": list(metrics["action_counts"].keys()), "count": list(metrics["action_counts"].values())})
        st.bar_chart(df.set_index("action"))

with tab_runs:
    runs = sorted([p for p in config.RUNS_DIR.glob("*") if p.is_dir()], reverse=True)
    if not runs:
        st.info("No runs yet.")
    else:
        choice = st.selectbox("Run", [r.name for r in runs])
        run = config.RUNS_DIR / choice
        actions_path = run / "actions.jsonl"
        if actions_path.exists():
            records = [json.loads(line) for line in actions_path.read_text().splitlines() if line.strip()]
            st.dataframe(pd.DataFrame(records))
            for rec in records:
                shot = run / f"step_{rec['step']:03d}.jpg"
                if shot.exists():
                    with st.expander(f"Step {rec['step']}: {rec['action']} — {rec.get('reason','')}"):
                        st.image(str(shot), use_container_width=True)
                        st.code(json.dumps(rec, indent=2), language="json")
        else:
            st.warning("No actions.jsonl in this run.")

with tab_train:
    from ..agents.trainer import collect as collect_dataset, list_datasets

    st.subheader("Datasets")
    datasets = list_datasets()
    col_a, col_b = st.columns([1, 4])
    if col_a.button("Collect from runs"):
        with st.spinner("Walking runs…"):
            summary = collect_dataset()
        st.success(f"wrote {summary.examples} examples to {summary.out_path}")
        datasets = list_datasets()

    if not datasets:
        st.info("No datasets yet. Click **Collect from runs** once you have a few completed sessions.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "dataset": d["name"],
                        "examples": d["examples"],
                        "runs scanned": d["runs_total"],
                        "runs with done": d["runs_with_done"],
                        "actions": ", ".join(f"{k}={v}" for k, v in d.get("actions", {}).items()),
                    }
                    for d in datasets
                ]
            ),
            use_container_width=True,
        )

    st.subheader("Training runs")
    from ..agents.trainer import list_runs

    runs = list_runs()
    if not runs:
        st.info(
            "No fine-tune runs yet. Run `nalu train run <dataset.jsonl>` to start one — "
            "metrics and the saved adapter will appear here."
        )
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "run": r["name"],
                        "examples": r.get("examples", "—"),
                        "iters": r.get("iters", "—"),
                        "steps logged": r.get("steps_logged", 0),
                        "last loss": r.get("last_loss", "—"),
                        "adapter": "✓" if r.get("has_adapter") else "—",
                    }
                    for r in runs
                ]
            ),
            use_container_width=True,
        )
        for r in runs:
            metrics_path = Path(r["path"]) / "metrics.jsonl"
            if not metrics_path.exists():
                continue
            df = pd.DataFrame(
                [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
            )
            if df.empty:
                continue
            with st.expander(f"{r['name']} — {df.shape[0]} steps"):
                st.line_chart(df.set_index("step")[["train_loss"]])
                st.caption(
                    f"Peak mem: {df['peak_mem_gb'].max():.2f} GB  •  "
                    f"Tokens/sec: {df['tokens_per_sec'].mean():.1f}  •  "
                    f"Final loss: {df['train_loss'].iloc[-1]:.4f}"
                )

with tab_model:
    st.subheader("Active model")
    st.code(config.VISION_MODEL)
    model_dir = config.MODELS_DIR
    if model_dir.exists():
        rows = []
        for p in sorted(model_dir.rglob("*")):
            if p.is_file():
                rows.append({"path": str(p.relative_to(model_dir)), "size_mb": round(p.stat().st_size / 1e6, 2)})
        if rows:
            st.dataframe(pd.DataFrame(rows))
        else:
            st.info("No model files cached yet. Run a task to trigger model download.")
    else:
        st.info("Model cache directory not initialized.")
