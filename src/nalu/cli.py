from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path

import structlog
import typer
from rich.console import Console

from . import config
from .actuator import Actuator, PauseController
from .agents.planner import Planner
from .agents.vision import VisionAgent
from .bus import BusClient, BusServer

app = typer.Typer(help="Nalu — local vision agent.", no_args_is_help=True)
console = Console()
log = structlog.get_logger("cli")


@app.command()
def doctor() -> None:
    """Check environment, deps, and macOS permissions."""
    console.print("[bold]Nalu environment check[/bold]")
    console.print(f"Python: {sys.version.split()[0]}")
    console.print(f"NALU_HOME: {config.ROOT}")
    config.ensure_dirs()

    try:
        import mlx
        console.print(f"mlx: {mlx.__version__ if hasattr(mlx, '__version__') else 'installed'}")
    except Exception as e:
        console.print(f"[red]mlx missing: {e}[/red]")

    try:
        from .capture import capture_main_display
        s = capture_main_display()
        console.print(f"[green]screen capture OK[/green] — {s.captured_width}x{s.captured_height}")
    except Exception as e:
        console.print(f"[red]screen capture failed: {e}[/red]")
        console.print("→ open System Settings → Privacy & Security → Screen Recording and grant your terminal/Python.")

    try:
        from pynput import keyboard  # noqa: F401
        console.print("[green]pynput OK[/green]")
    except Exception as e:
        console.print(f"[red]pynput failed: {e}[/red]")


@app.command()
def ask(text: str) -> None:
    """One-shot: capture screen, ask the model, dispatch actions until done."""
    asyncio.run(_run_one_shot(text))


async def _run_one_shot(goal: str) -> None:
    config.ensure_dirs()
    server = BusServer()
    bus_server = await server.start()

    pause = PauseController()
    pause.start()
    actuator = Actuator(pause)
    vision = VisionAgent()

    client = BusClient(source="planner")
    await client.connect()
    planner = Planner(client, actuator, vision, pause)
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
