from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from .. import config
from ..agents.trainer import TrainerAgent

st.set_page_config(page_title="Nalu", layout="wide")

st.title("Nalu — training & inspection")
st.caption("Fully local. Real data only. If a panel is empty, no data exists yet.")

config.ensure_dirs()

tab_overview, tab_runs, tab_train, tab_model = st.tabs(["Overview", "Runs", "Training", "Model"])

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
    st.subheader("Training runs")
    train_dir = config.ROOT / "training"
    if not train_dir.exists() or not any(train_dir.iterdir()):
        st.info("No training runs yet. The trainer pipeline lands in the next phase.")
    else:
        for run in sorted(train_dir.iterdir(), reverse=True):
            metrics_path = run / "metrics.jsonl"
            if metrics_path.exists():
                df = pd.DataFrame([json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()])
                with st.expander(run.name):
                    st.line_chart(df.set_index("step")[[c for c in df.columns if c != "step"]])

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
