from __future__ import annotations

import asyncio
import os
import signal
import sys

import structlog

from . import config
from .actuator import Actuator, PauseController
from .agents.planner import Planner
from .agents.vision import VisionAgent
from .bus import BusClient, BusServer
from .capture import ContinuousCapture

log = structlog.get_logger("daemon")


def is_running() -> bool:
    if not config.DAEMON_PID.exists():
        return False
    try:
        pid = int(config.DAEMON_PID.read_text().strip())
    except (ValueError, OSError):
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        try:
            config.DAEMON_PID.unlink()
        except OSError:
            pass
        return False
    return config.BUS_SOCKET.exists()


def daemon_pid() -> int | None:
    if not is_running():
        return None
    try:
        return int(config.DAEMON_PID.read_text().strip())
    except (ValueError, OSError):
        return None


def stop() -> bool:
    pid = daemon_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except ProcessLookupError:
        try:
            config.DAEMON_PID.unlink()
        except OSError:
            pass
        return False


async def serve() -> None:
    config.ensure_dirs()
    if is_running():
        print(f"daemon already running (pid {daemon_pid()})", file=sys.stderr)
        sys.exit(1)

    config.DAEMON_PID.write_text(str(os.getpid()))
    log.info("daemon_starting", pid=os.getpid())

    server = BusServer()
    bus_server = await server.start()

    pause = PauseController()
    pause.start()
    actuator = Actuator(pause)

    capture = ContinuousCapture()
    capture.start()

    vision = VisionAgent()
    log.info("loading_vision_model", model=vision.model_id)
    await asyncio.to_thread(vision.load)
    log.info("vision_model_loaded")

    planner_bus = BusClient(source="planner")
    await planner_bus.connect()
    planner = Planner(planner_bus, actuator, vision, pause, capture=capture)
    await planner.run()

    stop_evt = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_evt.set)

    log.info("daemon_ready")
    print(f"Nalu daemon ready (pid {os.getpid()}). Send tasks with `nalu ask <text>`. Stop with `nalu stop`.")

    try:
        await stop_evt.wait()
    finally:
        log.info("daemon_stopping")
        capture.stop()
        pause.stop()
        bus_server.close()
        await bus_server.wait_closed()
        try:
            config.DAEMON_PID.unlink()
        except OSError:
            pass
        log.info("daemon_stopped")
