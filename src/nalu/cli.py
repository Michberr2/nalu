from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import structlog
import typer
from rich.console import Console

from . import config, daemon
from .actuator import Actuator, PauseController
from .agents.planner import Planner
from .agents.vision import VisionAgent
from .bus import BusClient, BusServer
from .capture import ContinuousCapture

app = typer.Typer(help="Nalu — local vision agent.", no_args_is_help=True)
train_app = typer.Typer(help="Build training datasets from real session logs.", no_args_is_help=True)
fetch_app = typer.Typer(help="Ingest public GUI-agent datasets into our schema.", no_args_is_help=True)
model_app = typer.Typer(help="Manage registered base vision models.", no_args_is_help=True)
planner_app = typer.Typer(help="Multi-step LLM planner that decomposes goals into vision subgoals.", no_args_is_help=True)
app.add_typer(train_app, name="train")
train_app.add_typer(fetch_app, name="fetch")
app.add_typer(model_app, name="model")
app.add_typer(planner_app, name="planner")
console = Console()
log = structlog.get_logger("cli")


@model_app.command("list")
def model_list() -> None:
    """List registered base models. The active one is marked with *."""
    from .agents.vision import active_model_id, list_models

    active = active_model_id()
    models = list_models()
    if not models:
        console.print("[yellow]no models registered.[/yellow]")
        return
    for m in models:
        marker = "[green]*[/green]" if m.id == active else " "
        label = f" — {m.label}" if m.label else ""
        console.print(f"  {marker} [bold]{m.id}[/bold]  {m.path}{label}")


@model_app.command("active")
def model_active() -> None:
    """Show the currently active base model."""
    from .agents.vision import active_model

    a = active_model()
    label = f" — {a.label}" if a.label else ""
    console.print(f"[bold]{a.id}[/bold]  {a.path}{label}")


@model_app.command("register")
def model_register(
    model_id: str = typer.Argument(..., help="Short id (lowercase, [a-z0-9._-])."),
    path: str = typer.Argument(..., help="MLX-VLM model path or HF repo id."),
    label: str = typer.Option("", help="Human-readable label."),
) -> None:
    """Add (or replace) a base model in the registry."""
    from .agents.vision import register_model

    try:
        entry = register_model(model_id, path, label=label)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]registered:[/green] {entry.id}  {entry.path}")


@model_app.command("unregister")
def model_unregister(model_id: str = typer.Argument(...)) -> None:
    """Drop a base model from the registry (cannot remove the active one)."""
    from .agents.vision import unregister_model

    try:
        ok = unregister_model(model_id)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    if ok:
        console.print(f"[green]unregistered:[/green] {model_id}")
    else:
        console.print(f"[yellow]no such model:[/yellow] {model_id}")


@model_app.command("merge")
def model_merge(
    sources: list[str] = typer.Argument(
        ...,
        help="Source models as `repo[@weight[:density]]`. Need ≥2.",
    ),
    method: str = typer.Option("linear", help="linear | slerp | ties | task_arithmetic | dare_ties | dare_linear"),
    base: str = typer.Option(None, help="Base model (required for ties/dare_ties/task_arithmetic)."),
    dtype: str = typer.Option("bfloat16", help="Output dtype passed to mergekit."),
    out: Path = typer.Option(None, help="Override output directory."),
    no_quantize: bool = typer.Option(False, "--no-quantize", help="Skip MLX 4-bit conversion (keep HF weights only)."),
    bits: int = typer.Option(4, help="Quantization bits (only used when not --no-quantize)."),
    register_as: str = typer.Option(None, "--register-as", help="If set, register the merged model under this id."),
    label: str = typer.Option("", help="Human-readable label for the registry entry."),
) -> None:
    """Merge two or more compatible HF checkpoints, quantize to MLX, and register."""
    from .agents.trainer import MergeConfig, MergeRunner, parse_sources

    parsed = parse_sources(sources)
    cfg = MergeConfig(sources=parsed, merge_method=method, dtype=dtype, base_model=base)
    try:
        cfg.validate()
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    runner = MergeRunner(
        cfg,
        out_dir=out,
        quantize=not no_quantize,
        quant_bits=bits,
        register_as=register_as,
        register_label=label,
    )
    with console.status(f"merging {len(parsed)} models with {method}…"):
        try:
            summary = runner.run()
        except RuntimeError as e:
            console.print(f"[red]merge failed:[/red] {e}")
            raise typer.Exit(1)

    console.print(f"[green]merged:[/green] {summary.merged_dir}")
    if summary.mlx_dir is not None:
        console.print(f"[green]quantized:[/green] {summary.mlx_dir}")
    if summary.registered_id is not None:
        console.print(f"[green]registered as:[/green] {summary.registered_id}")
    console.print(f"  elapsed: {summary.elapsed_s:.1f}s")


@model_app.command("merges")
def model_merges() -> None:
    """List past merge runs."""
    from .agents.trainer import list_merges

    rows = list_merges()
    if not rows:
        console.print("[yellow]no merges yet — try `nalu model merge`.[/yellow]")
        return
    for r in rows[:20]:
        sources = ", ".join(r.get("sources", []))
        rid = r.get("registered_id") or "(unregistered)"
        console.print(f"  {r['out_dir']}  method={r['method']}  -> {rid}")
        console.print(f"    sources: {sources}")


@model_app.command("use")
def model_use(
    model_id: str = typer.Argument(...),
    no_hot: bool = typer.Option(False, "--no-hot", help="Skip hot-swap; only update the registry."),
) -> None:
    """Set the active base model (and hot-swap a running daemon)."""
    from .agents.vision import set_active

    try:
        entry = set_active(model_id)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]active:[/green] {entry.id}  {entry.path}")

    if daemon.is_running() and not no_hot:
        from .hotswap import hot_swap_model

        with console.status(f"hot-swapping daemon to {entry.id}…"):
            ok, msg = asyncio.run(hot_swap_model(model_id))
        if ok:
            console.print(f"[green]daemon swapped to:[/green] {msg}")
        else:
            console.print(f"[red]hot-swap failed:[/red] {msg}")


@train_app.command("collect")
def train_collect(
    out: Path = typer.Option(None, help="Override output directory."),
    include_failed: bool = typer.Option(False, "--include-failed", help="Include runs that never reached a done action."),
    eval_ratio: float = typer.Option(
        0.0,
        "--eval-ratio",
        help="Fraction of runs to hold out as eval (0 = no split, all in train).",
    ),
    seed: int = typer.Option(1337, "--seed", help="Deterministic shuffle seed for split."),
) -> None:
    """Walk past runs and write a JSONL dataset of (screenshot, action) examples."""
    from .agents.trainer import collect

    if not 0.0 <= eval_ratio < 1.0:
        console.print("[red]--eval-ratio must be in [0.0, 1.0).[/red]")
        raise typer.Exit(1)

    summary = collect(
        out_dir=out,
        only_completed=not include_failed,
        eval_ratio=eval_ratio,
        seed=seed,
    )
    console.print(f"[green]wrote[/green] {summary.out_path}")
    console.print(
        f"  runs scanned: {summary.runs_total}  "
        f"runs with done: {summary.runs_with_done}  "
        f"examples: {summary.examples}"
    )
    if summary.train_path and summary.eval_path:
        console.print(
            f"  split: train={summary.train_examples} ({len(summary.train_runs)} runs)  "
            f"eval={summary.eval_examples} ({len(summary.eval_runs)} runs)"
        )
        console.print(f"  train: {summary.train_path}")
        console.print(f"  eval:  {summary.eval_path}")
    if summary.actions:
        console.print("  by action: " + ", ".join(f"{k}={v}" for k, v in summary.actions.items()))


@fetch_app.command("seeclick")
def train_fetch_seeclick(
    annotation_path: Path = typer.Argument(
        ...,
        help="Local SeeClick annotation file (.jsonl or .json).",
    ),
    images_root: Path = typer.Argument(
        ...,
        help="Local directory under which `img_filename` paths in the annotation resolve.",
    ),
    out: Path = typer.Option(None, "--out", help="Output directory (default: training/datasets/external-<ts>/)."),
    limit: int = typer.Option(None, "--limit", help="Cap on number of output examples (smoke runs)."),
) -> None:
    """Ingest a local SeeClick snapshot into our training schema as JSONL.

    Fully offline, pure-Python. Bring the data however you like (download
    out-of-band, copy from a USB drive, sync from a NAS) — point this command
    at a local annotation file + images directory and it walks the records,
    normalizes coords, and writes `dataset.jsonl` + `summary.json`.
    """
    from .agents.trainer import fetch_seeclick

    if not annotation_path.exists():
        console.print(f"[red]annotation file not found:[/red] {annotation_path}")
        raise typer.Exit(1)
    if not images_root.exists():
        console.print(f"[red]images root not found:[/red] {images_root}")
        raise typer.Exit(1)

    summary = fetch_seeclick(annotation_path, images_root, out_dir=out, limit=limit)
    console.print(f"[green]wrote[/green] {summary.out_path}")
    console.print(
        f"  in: {summary.examples_in}  out: {summary.examples_out}  "
        f"skipped(no_image)={summary.skipped_no_image}  "
        f"skipped(no_target)={summary.skipped_no_target}  "
        f"skipped(unknown_action)={summary.skipped_unknown_action}"
    )
    if summary.actions:
        console.print("  by action: " + ", ".join(f"{k}={v}" for k, v in summary.actions.items()))


@train_app.command("run")
def train_run(
    dataset: Path = typer.Argument(..., help="Path to dataset.jsonl from `nalu train collect`."),
    rank: int = typer.Option(8, help="LoRA rank."),
    alpha: float = typer.Option(16.0, help="LoRA alpha."),
    dropout: float = typer.Option(0.0, help="LoRA dropout."),
    learning_rate: float = typer.Option(2e-5, "--lr", help="Adam learning rate."),
    batch_size: int = typer.Option(1, help="Minibatch size."),
    epochs: int = typer.Option(1, help="Passes over the dataset."),
    iters: int = typer.Option(None, help="Override iteration count (else epochs * dataset)."),
    out: Path = typer.Option(None, help="Override output directory."),
) -> None:
    """Fine-tune the vision model on a collected dataset (LoRA on quantized base)."""
    from .agents.trainer import QLoRARunner

    if not dataset.exists():
        console.print(f"[red]dataset not found:[/red] {dataset}")
        raise typer.Exit(1)

    runner = QLoRARunner(
        dataset_path=dataset,
        out_dir=out,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        iters=iters,
    )
    summary = runner.run()
    console.print(f"[green]adapter saved:[/green] {summary.adapter_dir}")
    console.print(
        f"  examples={summary.examples}  iters={summary.iters}  "
        f"final_loss={summary.final_loss if summary.final_loss is not None else '—'}"
    )


@train_app.command("eval")
def train_eval(
    dataset: Path = typer.Argument(..., help="Path to dataset.jsonl from `nalu train collect`."),
    limit: int = typer.Option(25, help="Max examples to evaluate (None for all)."),
    out: Path = typer.Option(None, help="Override output directory."),
) -> None:
    """Run the active VisionAgent over a dataset and report accuracy.

    The active LoRA adapter (if any) is applied automatically — run twice with
    `nalu train deactivate` / `nalu train activate <run>` between runs to
    compare base vs fine-tuned.
    """
    from .agents.trainer import evaluate

    if not dataset.exists():
        console.print(f"[red]dataset not found:[/red] {dataset}")
        raise typer.Exit(1)

    summary = evaluate(dataset_path=dataset, out_dir=out, limit=limit if limit > 0 else None)
    console.print(f"[green]eval saved:[/green] {summary.out_dir}")
    if summary.adapter_dir:
        console.print(f"  adapter: {summary.adapter_dir.name}")
    else:
        console.print("  adapter: (base model)")
    console.print(
        f"  total={summary.total}  "
        f"kind_acc={summary.action_correct / summary.total:.0%}  "
        f"click_hit@64={summary.click_hit_64}/{summary.click_examples}  "
        f"click_mae={summary.click_mae:.1f}px  "
        f"text_acc={(summary.text_correct / summary.text_examples) if summary.text_examples else 0:.0%}  "
        f"elapsed={summary.elapsed_s:.1f}s"
    )


@train_app.command("activate")
def train_activate(
    run: Path = typer.Argument(..., help="Path to a training run directory containing adapters.safetensors."),
    no_hot: bool = typer.Option(False, "--no-hot", help="Skip hot-swap; only update the pointer."),
) -> None:
    """Mark a fine-tuned adapter as the active one (and hot-swap a running daemon)."""
    from .agents.trainer import activate_adapter

    try:
        target = activate_adapter(run)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]active adapter:[/green] {target}")

    if daemon.is_running() and not no_hot:
        from .hotswap import hot_swap

        with console.status("hot-swapping daemon model…"):
            ok, msg = asyncio.run(hot_swap(str(target)))
        if ok:
            console.print(f"[green]daemon swapped to:[/green] {msg}")
        else:
            console.print(f"[red]hot-swap failed:[/red] {msg}")


@train_app.command("deactivate")
def train_deactivate(
    no_hot: bool = typer.Option(False, "--no-hot", help="Skip hot-swap; only clear the pointer."),
) -> None:
    """Clear the active adapter (and hot-swap a running daemon back to base)."""
    from .agents.trainer import deactivate_adapter

    if deactivate_adapter():
        console.print("[green]active adapter cleared.[/green]")
    else:
        console.print("[yellow]no active adapter set.[/yellow]")

    if daemon.is_running() and not no_hot:
        from .hotswap import hot_swap

        with console.status("hot-swapping daemon back to base model…"):
            ok, msg = asyncio.run(hot_swap(None))
        if ok:
            console.print(f"[green]daemon now running:[/green] {msg}")
        else:
            console.print(f"[red]hot-swap failed:[/red] {msg}")


@train_app.command("report")
def train_report() -> None:
    """Show training recommendations and dataset inventory."""
    from .agents.trainer import TrainerAgent, list_datasets

    trainer = TrainerAgent()
    metrics = trainer.collect_metrics()
    rec = trainer.recommend()

    console.print("[bold]Run metrics[/bold]")
    if metrics.get("runs", 0) == 0:
        console.print("  no runs yet — generate some with `nalu ask`.")
    else:
        console.print(f"  runs: {metrics['runs']}  completed: {metrics['completed']}  failed: {metrics['failed']}")
        console.print(f"  success rate: {metrics['success_rate']:.0%}  avg steps: {metrics['avg_steps']:.1f}")

    console.print("\n[bold]Should I retrain?[/bold]")
    if rec.should_retrain:
        console.print("  [red]yes[/red]")
    elif metrics.get("runs", 0) == 0:
        console.print("  [yellow]not enough data[/yellow]")
    else:
        console.print("  [green]not yet[/green]")
    for r in rec.reasons:
        console.print(f"   • {r}")
    for s in rec.suggested_data:
        console.print(f"   ▸ {s}")

    console.print("\n[bold]Datasets[/bold]")
    datasets = list_datasets()
    if not datasets:
        console.print("  none yet — run `nalu train collect`.")
    else:
        for d in datasets[:10]:
            console.print(f"  {d['name']}  examples={d['examples']}  runs={d['runs_total']}")


@app.command()
def doctor() -> None:
    """Check environment, deps, and macOS permissions."""
    from . import permissions

    console.print("[bold]Nalu environment check[/bold]")
    console.print(f"Python: {sys.version.split()[0]}")
    console.print(f"NALU_HOME: {config.ROOT}")
    config.ensure_dirs()

    try:
        import mlx
        console.print(f"mlx: {mlx.__version__ if hasattr(mlx, '__version__') else 'installed'}")
    except Exception as e:
        console.print(f"[red]mlx missing: {e}[/red]")

    console.print("\n[bold]macOS permissions[/bold]")
    failed = []
    for status in permissions.check_all():
        if status.granted:
            console.print(f"  [green]✓[/green] {status.name} — {status.detail}")
        else:
            console.print(f"  [red]✗[/red] {status.name} — {status.detail}")
            failed.append(status)

    if failed:
        console.print("\n[yellow]Some permissions are missing.[/yellow]")
        console.print("Run [bold]nalu setup[/bold] to open the right Settings panes.")


@app.command()
def setup() -> None:
    """First-run helper: open System Settings panes for any missing permissions."""
    from . import permissions

    console.print("[bold]Nalu setup — granting macOS permissions[/bold]\n")
    statuses = permissions.check_all()
    needed = [s for s in statuses if not s.granted]

    if not needed:
        console.print("[green]All permissions are already granted. You're ready to run `nalu serve`.[/green]")
        return

    for s in needed:
        console.print(f"[red]✗[/red] {s.name} — {s.detail}")
        console.print(f"  Opening Settings → {s.name}…")
        permissions.open_settings(s.fix_url)
        typer.prompt(
            f"  After enabling Nalu / your terminal under {s.name}, press Enter to re-check",
            default="",
            show_default=False,
        )

    console.print("\n[bold]Re-checking…[/bold]")
    for s in permissions.check_all():
        marker = "[green]✓[/green]" if s.granted else "[red]✗[/red]"
        console.print(f"  {marker} {s.name} — {s.detail}")


@app.command()
def ask(text: str) -> None:
    """Send a task to the running daemon, or run one-shot in-process if no daemon."""
    if daemon.is_running():
        asyncio.run(_ask_daemon(text))
    else:
        console.print("[yellow]No daemon running — starting in-process (model will reload).[/yellow]")
        console.print("[dim]Tip: run `nalu serve` in another terminal to keep the model warm.[/dim]")
        asyncio.run(_run_one_shot(text))


@app.command()
def serve() -> None:
    """Run the persistent Nalu daemon (keeps the vision model loaded)."""
    asyncio.run(daemon.serve())


@app.command()
def menubar() -> None:
    """Run the macOS menu-bar app (connects to a running daemon over the bus)."""
    from .menubar import app as menubar_app

    raise typer.Exit(menubar_app.run())


@app.command()
def onboard(yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts.")) -> None:
    """First-run wizard: permissions → voice → wake-word → vision → screenshot decode."""
    from .onboarding import OnboardingWizard, StepStatus

    config.ensure_dirs()

    def before(step) -> None:
        console.print(f"\n[bold cyan]→ {step.summary}[/bold cyan]  ({step.name})")

    def after(step, result) -> bool:
        marker = {
            StepStatus.PASS: "[green]✓[/green]",
            StepStatus.FAIL: "[red]✗[/red]",
            StepStatus.SKIP: "[yellow]·[/yellow]",
        }[result.status]
        console.print(f"  {marker} {result.detail or result.status.value}  [dim]({result.elapsed_s:.1f}s)[/dim]")
        if result.status == StepStatus.FAIL and result.fix_hint:
            console.print(f"  [dim]hint:[/dim] {result.fix_hint}")
        if result.status == StepStatus.FAIL and step.required and not yes:
            return typer.confirm("  continue anyway?", default=False)
        return True

    console.print("[bold]Nalu onboarding[/bold] — verifying everything Nalu needs is in place.\n")
    wizard = OnboardingWizard(before_step=before, after_step=after)
    report = wizard.run()

    console.print("\n[bold]Summary[/bold]")
    for r in report.results:
        marker = {
            StepStatus.PASS: "[green]✓[/green]",
            StepStatus.FAIL: "[red]✗[/red]",
            StepStatus.SKIP: "[yellow]·[/yellow]",
        }[r.status]
        console.print(f"  {marker} {r.name}: {r.detail}")

    if report.is_ready:
        console.print("\n[bold green]Nalu is ready.[/bold green] Try [bold]nalu serve[/bold] or [bold]nalu menubar[/bold].")
    else:
        console.print("\n[bold red]Onboarding incomplete.[/bold red] Resolve the failures above and re-run [bold]nalu onboard[/bold].")
        raise typer.Exit(1)


@app.command()
def stop() -> None:
    """Stop the running Nalu daemon."""
    if daemon.stop():
        console.print("[green]daemon stop signal sent.[/green]")
    else:
        console.print("[yellow]no daemon running.[/yellow]")


@app.command()
def status() -> None:
    """Show daemon status."""
    pid = daemon.daemon_pid()
    if pid is None:
        console.print("[yellow]daemon: not running[/yellow]")
    else:
        console.print(f"[green]daemon: running (pid {pid})[/green]")
        console.print(f"  bus socket: {config.BUS_SOCKET}")


async def _ask_daemon(goal: str) -> None:
    pub = BusClient(source="cli")
    await pub.connect()

    done_evt = asyncio.Event()
    completed = {"ok": False, "answer": ""}

    async def on_terminal(ev):
        if ev.topic in ("task_completed", "task_failed"):
            completed["ok"] = ev.topic == "task_completed"
            completed["answer"] = ev.payload.get("answer", "") or ev.payload.get("reason", "")
            done_evt.set()

    sub = BusClient(source="cli-listener")
    await sub.connect()
    await sub.subscribe("task_completed", on_terminal)
    await sub.subscribe("task_failed", on_terminal)

    await pub.publish("user_intent", {"text": goal})
    console.print(f"[cyan]Nalu is working on:[/cyan] {goal}")
    console.print("Press [bold]⌃⌥⌘.[/bold] to pause/resume.")

    try:
        await asyncio.wait_for(done_evt.wait(), timeout=config.PLANNER_TASK_TIMEOUT_S + 30)
    except asyncio.TimeoutError:
        console.print("[red]CLI timed out waiting for completion.[/red]")
    finally:
        await pub.close()
        await sub.close()

    if completed["ok"]:
        console.print(f"[green]done:[/green] {completed['answer']}")
    else:
        console.print(f"[red]failed:[/red] {completed['answer']}")


async def _run_one_shot(goal: str) -> None:
    config.ensure_dirs()
    server = BusServer()
    bus_server = await server.start()

    pause = PauseController()
    pause.start()
    actuator = Actuator(pause)
    vision = VisionAgent()

    capture = ContinuousCapture()
    capture.start()

    client = BusClient(source="planner")
    await client.connect()
    planner = Planner(client, actuator, vision, pause, capture=capture)
    await planner.run()

    pub = BusClient(source="cli")
    await pub.connect()

    done_evt = asyncio.Event()
    completed = {"ok": False, "answer": ""}

    async def on_terminal(ev):
        if ev.topic in ("task_completed", "task_failed"):
            completed["ok"] = ev.topic == "task_completed"
            completed["answer"] = ev.payload.get("answer", "") or ev.payload.get("reason", "")
            done_evt.set()

    sub = BusClient(source="cli-listener")
    await sub.connect()
    await sub.subscribe("task_completed", on_terminal)
    await sub.subscribe("task_failed", on_terminal)

    await pub.publish("user_intent", {"text": goal})

    console.print(f"[cyan]Nalu is working on:[/cyan] {goal}")
    console.print("Press [bold]⌃⌥⌘.[/bold] to pause/resume.")

    try:
        await asyncio.wait_for(done_evt.wait(), timeout=config.PLANNER_TASK_TIMEOUT_S + 30)
    except asyncio.TimeoutError:
        console.print("[red]CLI timed out waiting for completion.[/red]")
    finally:
        capture.stop()
        pause.stop()
        bus_server.close()
        await bus_server.wait_closed()

    if completed["ok"]:
        console.print(f"[green]done:[/green] {completed['answer']}")
    else:
        console.print(f"[red]failed:[/red] {completed['answer']}")


@app.command()
def dashboard() -> None:
    """Launch the Streamlit training/inspection dashboard."""
    here = Path(__file__).parent / "dashboard" / "app.py"
    os.execvp("streamlit", ["streamlit", "run", str(here), "--server.headless=true"])


@app.command()
def chat() -> None:
    """One-command chat UI: start the daemon if needed, then open the dashboard's Chat tab.

    Equivalent to running `nalu serve &` in one terminal and `nalu dashboard` in
    another, but as a single command. Voice (mic record → STT) and text inputs
    both supported in the browser; answers are spoken back via Piper TTS.
    """
    import subprocess

    pid = daemon.daemon_pid()
    started_here = False
    if pid is None:
        console.print("[cyan]starting daemon[/cyan] (model warm-up takes ~16s on first load)…")
        log_path = config.LOG_DIR / f"daemon-{int(time.time())}.log"
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as log_f:
            subprocess.Popen(
                ["uv", "run", "nalu", "serve"],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        started_here = True
        # Poll until the daemon writes its pid file. The warm-up itself is async,
        # so the daemon will respond to the first prompt with a longer cold-load.
        for _ in range(40):
            time.sleep(0.5)
            if daemon.daemon_pid() is not None:
                break
        else:
            console.print(f"[red]daemon did not start within 20s. tail {log_path} for details.[/red]")
            raise typer.Exit(1)
        console.print(f"[green]daemon up[/green] (pid {daemon.daemon_pid()}, log {log_path})")

    if started_here:
        console.print("[dim]daemon will keep running when you close the browser; stop it with `nalu stop`.[/dim]")
    here = Path(__file__).parent / "dashboard" / "app.py"
    os.execvp("streamlit", ["streamlit", "run", str(here), "--server.headless=false"])


@app.command()
def speak(text: str) -> None:
    """Speak text aloud via Piper TTS (downloads voice on first use)."""
    from .agents.voice import TTS

    tts = TTS()
    tts.speak(text)


@app.command()
def listen(seconds: float = 4.0) -> None:
    """Record from mic and transcribe with faster-whisper."""
    from .agents.voice import STT
    from .agents.voice.stt import record

    console.print(f"[cyan]listening for {seconds}s…[/cyan]")
    samples, sr = record(seconds)
    stt = STT()
    text = stt.transcribe_array(samples, sr)
    console.print(f"[green]heard:[/green] {text}")


@app.command()
def wake(
    seconds: float = typer.Option(15.0, help="Listen window before stopping."),
    keyword: str = typer.Option(None, help=f"Override (default: {config.WAKEWORD_KEYWORD})."),
    threshold: float = typer.Option(None, help=f"Override (default: {config.WAKEWORD_THRESHOLD})."),
) -> None:
    """Standalone wake-word test — prints each detection. No daemon required."""
    from .agents.voice import OpenWakeWordSpotter, WakeWordRunner

    kw = keyword or config.WAKEWORD_KEYWORD
    thr = threshold if threshold is not None else config.WAKEWORD_THRESHOLD

    detections: list[tuple[str, float]] = []

    def on_wake(k: str, score: float) -> None:
        detections.append((k, score))
        console.print(f"[green]wake:[/green] {k}  score={score:.2f}")

    try:
        spotter = OpenWakeWordSpotter(models=[kw])
    except ImportError:
        console.print("[red]openwakeword not installed.[/red] `uv pip install openwakeword`.")
        raise typer.Exit(1)

    runner = WakeWordRunner(on_wake=on_wake, spotter=spotter, threshold=thr)
    with console.status(f"loading {kw}…"):
        runner.warm()
    console.print(f"[cyan]listening {seconds:.0f}s for {kw!r} @ threshold {thr}…[/cyan]")
    runner.start()
    try:
        time.sleep(seconds)
    finally:
        runner.stop()
    console.print(f"[bold]{len(detections)} detection(s).[/bold]")


@planner_app.command("status")
def planner_status() -> None:
    """Show the LLM planner's enabled state + selected model."""
    cfg = config.read_planner_config()
    enabled = config.USE_LLM_PLANNER
    env_override = "NALU_USE_LLM_PLANNER" in os.environ
    console.print(f"[bold]Enabled:[/bold] {enabled}{' [dim](env override)[/dim]' if env_override else ''}")
    console.print(f"[bold]Model:[/bold]   {config.PLANNER_LLM_MODEL}")
    console.print(f"[bold]Config:[/bold]  {config.PLANNER_CONFIG_FILE}")
    if cfg:
        console.print(f"[dim]on-disk state:[/dim] {cfg}")
    else:
        console.print("[dim]no on-disk state — defaults / env in effect[/dim]")


@planner_app.command("enable")
def planner_enable(
    model_id: str = typer.Option(None, help="Override planner LLM (defaults to current selection)."),
) -> None:
    """Turn on the LLM planner. Persists to MODELS_DIR/planner.json so it survives restarts."""
    new = config.write_planner_config(enabled=True, model_id=model_id)
    console.print(f"[green]planner enabled.[/green] state: {new}")
    if daemon.daemon_pid() is not None:
        console.print("[yellow]daemon is running — restart it to pick up the change.[/yellow]")


@planner_app.command("disable")
def planner_disable() -> None:
    """Turn off the LLM planner. Vision-only single-shot behavior restored."""
    new = config.write_planner_config(enabled=False)
    console.print(f"[green]planner disabled.[/green] state: {new}")
    if daemon.daemon_pid() is not None:
        console.print("[yellow]daemon is running — restart it to pick up the change.[/yellow]")


@planner_app.command("test")
def planner_test(goal: str) -> None:
    """Decompose `goal` (no execution) and print the resulting plan. Verifies the LLM is wired."""
    from .agents.planner_llm import LLMDecomposer
    from .agents.planner_llm.prompts import format_plan_for_log

    console.print(f"[cyan]decomposing[/cyan]: {goal}")
    decomposer = LLMDecomposer()
    console.print(f"[dim]loading {decomposer.model_id}…[/dim]")
    plan = decomposer.decompose(goal)
    if plan.fallback:
        console.print(f"[yellow]fallback plan[/yellow] (decomposer output was unusable):")
    else:
        console.print(f"[green]{len(plan)} subgoal(s):[/green]")
    console.print(format_plan_for_log(plan))


@app.command()
def start() -> None:
    """Start the dashboard. Inference runs are launched per-ask via 'nalu ask'."""
    typer.echo("Phase 0 entry point. Use 'nalu doctor', 'nalu ask <task>', or 'nalu dashboard'.")
    typer.echo("Full multi-process daemon mode lands in Phase 1.")


if __name__ == "__main__":
    app()
