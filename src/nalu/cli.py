from __future__ import annotations

import asyncio
import os
import sys
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
app.add_typer(train_app, name="train")
console = Console()
log = structlog.get_logger("cli")


@train_app.command("collect")
def train_collect(
    out: Path = typer.Option(None, help="Override output directory."),
    include_failed: bool = typer.Option(False, "--include-failed", help="Include runs that never reached a done action."),
) -> None:
    """Walk past runs and write a JSONL dataset of (screenshot, action) examples."""
    from .agents.trainer import collect

    summary = collect(out_dir=out, only_completed=not include_failed)
    console.print(f"[green]wrote[/green] {summary.out_path}")
    console.print(
        f"  runs scanned: {summary.runs_total}  "
        f"runs with done: {summary.runs_with_done}  "
        f"examples: {summary.examples}"
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
    console.print(f"[green]adapter saved:[/green] {summary.adapter_path}")
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
def start() -> None:
    """Start the dashboard. Inference runs are launched per-ask via 'nalu ask'."""
    typer.echo("Phase 0 entry point. Use 'nalu doctor', 'nalu ask <task>', or 'nalu dashboard'.")
    typer.echo("Full multi-process daemon mode lands in Phase 1.")


if __name__ == "__main__":
    app()
