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
from .agents.responder import Responder, make_default_generate_fn
from .agents.voice.always_on import (
    AlwaysOnRunner,
    chunked_sounddevice_source,
    make_silero_vad_fn,
)
from .agents.voice.proactive import ProactiveSpeaker, is_proactive_enabled
from .agents.vision import VisionAgent
from .agents.voice import PushToTalk, TTS, WakeWordRunner
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


CONVERSATIONAL_PREFIXES = (
    "what ", "what's ", "whats ", "who ", "who's ", "whos ",
    "when ", "where ", "why ", "how ",
    "is ", "are ", "do ", "does ", "did ", "can ", "could ", "would ", "will ",
    "tell me ", "explain ", "say ", "thanks", "thank you", "hi ", "hello",
    "hey ", "good morning", "good afternoon", "good evening",
)
ACTION_HINTS = (
    "open ", "close ", "click ", "type ", "search ", "go to ", "navigate ",
    "find ", "save ", "delete ", "scroll ", "send ", "copy ", "paste ",
    "create ", "make ", "write ", "edit ", "switch ", "select ", "drag ",
    "drop ", "minimize ", "maximize ", "quit ", "launch ",
)
CONVERSATIONAL_MAX_WORDS = 14


def classify_user_text(text: str) -> str:
    """Return 'query' for short conversational turns, 'intent' for actionable goals.

    Heuristic-only — no model call. A query word at the front and no action verb
    means we hand it to the Responder; otherwise the Planner gets it. Anything
    above ~14 words is assumed to be a goal, regardless of phrasing.
    """
    s = (text or "").strip().lower()
    if not s:
        return "intent"
    word_count = len(s.split())
    if any(s.startswith(h) for h in ACTION_HINTS):
        return "intent"
    if word_count <= CONVERSATIONAL_MAX_WORDS and any(s.startswith(p) for p in CONVERSATIONAL_PREFIXES):
        return "query"
    return "intent"


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
    wake: WakeWordRunner | None = None
    bus_server = None
    events_log = None
    try:
        server = BusServer()
        bus_server = await server.start()

        pause_pub = BusClient(source="pause")
        await pause_pub.connect()
        loop_for_pause = asyncio.get_running_loop()

        def _pause_changed(paused: bool) -> None:
            asyncio.run_coroutine_threadsafe(
                pause_pub.publish("pause_state", {"paused": paused}), loop_for_pause,
            )

        pause = PauseController(on_change=_pause_changed)
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

        history: deque[dict] = deque(maxlen=20)

        planner_bus = BusClient(source="planner")
        await planner_bus.connect()
        decomposer = None
        if config.USE_LLM_PLANNER:
            from .agents.planner_llm import LLMDecomposer

            decomposer = LLMDecomposer()
            log.info("loading_planner_llm", model=decomposer.model_id)
            try:
                await asyncio.to_thread(decomposer.load)
                log.info("planner_llm_loaded")
            except Exception:
                log.exception("planner_llm_load_failed")
                decomposer = None
        planner = Planner(
            planner_bus,
            actuator,
            vision,
            pause,
            capture=capture,
            conversation=history,
            decomposer=decomposer,
        )
        await planner.run()

        responder = None
        if decomposer is not None:
            responder_bus = BusClient(source="responder")
            await responder_bus.connect()
            responder = Responder(
                responder_bus,
                generate_fn=make_default_generate_fn(decomposer),
            )
            await responder.run()
            log.info("responder_ready")

        voice_bus = BusClient(source="voice")
        await voice_bus.connect()
        loop = asyncio.get_running_loop()

        def _on_transcript(text: str) -> None:
            log.info("ptt_transcript", text=text)
            route = "user_query" if responder is not None and classify_user_text(text) == "query" else "user_intent"
            asyncio.run_coroutine_threadsafe(
                voice_bus.publish(route, {"text": text, "via": "voice"}), loop
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

        if config.WAKEWORD_ENABLED:
            from .agents.voice import OpenWakeWordSpotter

            def _on_wake(keyword: str, score: float) -> None:
                log.info("wakeword_detected", keyword=keyword, score=score)
                asyncio.run_coroutine_threadsafe(
                    voice_bus.publish("wakeword_detected", {"keyword": keyword, "score": score}), loop,
                )
                ptt.trigger()  # reuse PTT capture path

            try:
                spotter = OpenWakeWordSpotter(models=[config.WAKEWORD_KEYWORD])
                wake = WakeWordRunner(
                    on_wake=_on_wake,
                    spotter=spotter,
                    threshold=config.WAKEWORD_THRESHOLD,
                    cooldown_s=config.WAKEWORD_COOLDOWN_S,
                )
                log.info("loading_wakeword_model", keyword=config.WAKEWORD_KEYWORD)
                await asyncio.to_thread(wake.warm)
                wake.start()
                log.info("wakeword_running", keyword=config.WAKEWORD_KEYWORD, threshold=config.WAKEWORD_THRESHOLD)
            except Exception as e:
                log.warning("wakeword_disabled", reason=str(e))
                wake = None

        tts = TTS()
        log.info("loading_tts_voice", voice=config.TTS_VOICE)
        await asyncio.to_thread(tts.load)
        log.info("tts_voice_loaded")

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

        async def _on_responder_reply(ev) -> None:
            reply = ev.payload.get("reply", "") or ""
            history.append({"role": "assistant", "text": reply, "ts": ev.ts})
            _speak_async(reply)

        async def _on_user_query(ev) -> None:
            history.append({"role": "user", "text": ev.payload.get("text", ""), "ts": ev.ts})

        async def _on_history_request(ev) -> None:
            await chat_bus.publish("conversation_history", {"history": list(history)})

        async def _on_swap_adapter(ev) -> None:
            target = ev.payload.get("path")
            target_path = None if target in (None, "", False) else target
            log.info("vision_swap_starting", target=target_path)
            try:
                applied = await asyncio.to_thread(vision.swap_adapter, target_path)
                log.info("vision_swap_completed", adapter=str(applied) if applied else None)
                await chat_bus.publish(
                    "vision_swap_completed",
                    {"adapter": str(applied) if applied else None},
                )
            except Exception as e:
                log.exception("vision_swap_failed")
                await chat_bus.publish("vision_swap_failed", {"reason": str(e)})

        async def _on_swap_model(ev) -> None:
            model_id = ev.payload.get("model_id")
            if not model_id:
                await chat_bus.publish("vision_model_swap_failed", {"reason": "missing model_id"})
                return
            log.info("vision_model_swap_starting", model_id=model_id)
            try:
                applied = await asyncio.to_thread(vision.swap_model, model_id)
                log.info("vision_model_swap_completed", model=applied)
                await chat_bus.publish("vision_model_swap_completed", {"model": applied})
            except Exception as e:
                log.exception("vision_model_swap_failed")
                await chat_bus.publish("vision_model_swap_failed", {"reason": str(e)})

        async def _on_pause_request(ev) -> None:
            wanted = bool(ev.payload.get("paused"))
            pause.set(wanted)

        await chat_bus.subscribe("user_intent", _on_intent)
        await chat_bus.subscribe("user_query", _on_user_query)
        await chat_bus.subscribe("task_completed", _on_completed)
        await chat_bus.subscribe("task_failed", _on_failed)
        await chat_bus.subscribe("responder_reply", _on_responder_reply)

        if os.environ.get("NALU_ALWAYS_ON_STT", "0") not in ("0", "", "false", "False"):
            try:
                always_on_stt = ptt._stt  # reuse the warm STT model

                def _on_always_on_transcript(text: str) -> None:
                    log.info("always_on_transcript", text=text)
                    route = "user_query" if responder is not None and classify_user_text(text) == "query" else "user_intent"
                    asyncio.run_coroutine_threadsafe(
                        voice_bus.publish(route, {"text": text, "via": "always_on"}), loop
                    )

                src_iter = chunked_sounddevice_source()

                def _source():
                    try:
                        return next(src_iter)
                    except StopIteration:
                        return None

                always_on = AlwaysOnRunner(
                    audio_source=_source,
                    vad_fn=make_silero_vad_fn(),
                    transcribe_fn=lambda s, sr: always_on_stt.transcribe_array(s, sr),
                    on_transcript=_on_always_on_transcript,
                )
                always_on.start()
                log.info("always_on_stt_started")
            except Exception:
                log.exception("always_on_stt_start_failed")

        if is_proactive_enabled():
            proactive_bus = BusClient(source="proactive")
            await proactive_bus.connect()
            proactive = ProactiveSpeaker(proactive_bus, speak_fn=_speak_async, enabled=True)
            await proactive.run()
            log.info("proactive_voice_enabled")
        await chat_bus.subscribe("history_request", _on_history_request)
        await chat_bus.subscribe("vision_swap_adapter", _on_swap_adapter)
        await chat_bus.subscribe("vision_swap_model", _on_swap_model)
        await chat_bus.subscribe("pause_request", _on_pause_request)

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
        if wake is not None:
            wake.stop()
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
