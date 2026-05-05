from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import threading
from collections import deque
from dataclasses import asdict

import structlog

from . import config
from .actuator import Actuator, PauseController
from .agents.planner import Planner
from .agents.vision import VisionAgent
from .agents.voice import PushToTalk, TTS
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

    from . import permissions as perms

    blockers = [s for s in perms.check_all() if not s.granted and s.name in ("Screen Recording", "Accessibility")]
    if blockers:
        print("Nalu cannot start — required permissions are missing:", file=sys.stderr)
        for s in blockers:
            print(f"  ✗ {s.name}: {s.detail}", file=sys.stderr)
        print("\nRun `nalu setup` to grant them, then try again.", file=sys.stderr)
        sys.exit(2)

    config.DAEMON_PID.write_text(str(os.getpid()))
    log.info("daemon_starting", pid=os.getpid())

    pause: PauseController | None = None
    capture: ContinuousCapture | None = None
    ptt: PushToTalk | None = None
    bus_server = None
    events_log = None
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

        events_log = config.EVENTS_LOG.open("a", buffering=1)

        async def _log_event(ev) -> None:
            events_log.write(json.dumps(asdict(ev)) + "\n")

        log_bus = BusClient(source="event-log")
        await log_bus.connect()
        await log_bus.subscribe("*", _log_event)

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

        tts = TTS()
        log.info("loading_tts_voice", voice=config.TTS_VOICE)
        await asyncio.to_thread(tts.load)
        log.info("tts_voice_loaded")

        history: deque[dict] = deque(maxlen=20)

        def _speak_async(text: str) -> None:
            if not text:
                return
            threading.Thread(target=lambda: tts.speak(text), daemon=True).start()

        chat_bus = BusClient(source="daemon")
        await chat_bus.connect()

        async def _on_intent(ev) -> None:
            history.append({"role": "user", "text": ev.payload.get("text", ""), "ts": ev.ts})

        async def _on_completed(ev) -> None:
            answer = ev.payload.get("answer", "") or ""
            history.append({"role": "assistant", "text": answer, "ts": ev.ts})
            _speak_async(answer)

        async def _on_failed(ev) -> None:
            reason = ev.payload.get("reason", "") or ""
            history.append({"role": "assistant", "text": f"failed: {reason}", "ts": ev.ts})

        async def _on_history_request(ev) -> None:
            await chat_bus.publish("conversation_history", {"history": list(history)})

        await chat_bus.subscribe("user_intent", _on_intent)
        await chat_bus.subscribe("task_completed", _on_completed)
        await chat_bus.subscribe("task_failed", _on_failed)
        await chat_bus.subscribe("history_request", _on_history_request)

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
        if events_log is not None:
            try:
                events_log.close()
            except Exception:
                pass
        try:
            config.DAEMON_PID.unlink()
        except OSError:
            pass
        log.info("daemon_stopped")
