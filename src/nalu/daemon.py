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
from .agents.voice import PushToTalk
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

    pause: PauseController | None = None
    capture: ContinuousCapture | None = None
    ptt: PushToTalk | None = None
    bus_server = None
    try:
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

        voice_bus = BusClient(source="voice")
        await voice_bus.connect()
        loop = asyncio.get_running_loop()

        def _on_transcript(text: str) -> None:
            log.info("ptt_transcript", text=text)
            asyncio.run_coroutine_threadsafe(
                voice_bus.publish("user_intent", {"text": text, "via": "voice"}), loop
            )

        def _on_listening(listening: bool) -> None:
            asyncio.run_coroutine_threadsafe(
                voice_bus.publish("voice_listening", {"listening": listening}), loop
            )

        ptt = PushToTalk(on_transcript=_on_transcript, on_listening=_on_listening)
        log.info("loading_stt_model", model=config.STT_MODEL)
        await asyncio.to_thread(ptt.warm)
        log.info("stt_model_loaded")
        ptt.start()

        stop_evt = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_evt.set)

        log.info("daemon_ready")
        print(
            f"Nalu daemon ready (pid {os.getpid()}). "
            f"Tasks: `nalu ask <text>`. "
            f"Voice: tap {config.PUSH_TO_TALK_HOTKEY}. "
            f"Stop: `nalu stop`."
        )
        await stop_evt.wait()
    finally:
        log.info("daemon_stopping")
        if ptt is not None:
            ptt.stop()
        if capture is not None:
            capture.stop()
        if pause is not None:
            pause.stop()
        if bus_server is not None:
            bus_server.close()
            await bus_server.wait_closed()
        try:
            config.DAEMON_PID.unlink()
        except OSError:
            pass
        log.info("daemon_stopped")
