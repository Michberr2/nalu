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
