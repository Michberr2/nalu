from __future__ import annotations

import asyncio

from .bus import BusClient


async def hot_swap(adapter_path: str | None, timeout: float = 120.0) -> tuple[bool, str]:
    """Tell the running daemon to swap its vision adapter.

    `adapter_path=None` reverts to the base model. Returns `(ok, message)`.
    """
    pub = BusClient(source="hotswap")
    sub = BusClient(source="hotswap-listener")
    await pub.connect()
    await sub.connect()
    done = asyncio.Event()
    result = {"ok": False, "msg": ""}

    async def on_terminal(ev):
        if ev.topic == "vision_swap_completed":
            result["ok"] = True
            result["msg"] = ev.payload.get("adapter") or "(base model)"
            done.set()
        elif ev.topic == "vision_swap_failed":
            result["ok"] = False
            result["msg"] = ev.payload.get("reason", "unknown")
            done.set()

    await sub.subscribe("vision_swap_completed", on_terminal)
    await sub.subscribe("vision_swap_failed", on_terminal)
    await pub.publish("vision_swap_adapter", {"path": adapter_path})
    try:
        await asyncio.wait_for(done.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        result["msg"] = "(timeout)"
    finally:
        await pub.close()
        await sub.close()
    return result["ok"], result["msg"]


async def hot_swap_model(model_id: str, timeout: float = 240.0) -> tuple[bool, str]:
    """Tell the running daemon to switch base models. Re-applies the active adapter."""
    pub = BusClient(source="hotswap-model")
    sub = BusClient(source="hotswap-model-listener")
    await pub.connect()
    await sub.connect()
    done = asyncio.Event()
    result = {"ok": False, "msg": ""}

    async def on_terminal(ev):
        if ev.topic == "vision_model_swap_completed":
            result["ok"] = True
            result["msg"] = ev.payload.get("model") or "(unknown)"
            done.set()
        elif ev.topic == "vision_model_swap_failed":
            result["ok"] = False
            result["msg"] = ev.payload.get("reason", "unknown")
            done.set()

    await sub.subscribe("vision_model_swap_completed", on_terminal)
    await sub.subscribe("vision_model_swap_failed", on_terminal)
    await pub.publish("vision_swap_model", {"model_id": model_id})
    try:
        await asyncio.wait_for(done.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        result["msg"] = "(timeout)"
    finally:
        await pub.close()
        await sub.close()
    return result["ok"], result["msg"]
