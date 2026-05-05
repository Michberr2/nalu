from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from .. import config


@dataclass
class Event:
    topic: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    source: str = ""

    def to_line(self) -> bytes:
        return (json.dumps(asdict(self)) + "\n").encode()

    @classmethod
    def from_line(cls, line: bytes) -> "Event":
        d = json.loads(line.decode())
        return cls(**d)


class BusServer:
    def __init__(self, sock_path: Path = config.BUS_SOCKET):
        self.sock_path = sock_path
        self._subs: dict[str, set[asyncio.StreamWriter]] = {}
        self._all: set[asyncio.StreamWriter] = set()

    async def start(self) -> asyncio.AbstractServer:
        self.sock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.sock_path.exists():
            self.sock_path.unlink()
        server = await asyncio.start_unix_server(self._handle, path=str(self.sock_path))
        os.chmod(self.sock_path, 0o600)
        return server

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break
                msg = json.loads(line.decode())
                kind = msg.get("kind")
                if kind == "sub":
                    topic = msg["topic"]
                    if topic == "*":
                        self._all.add(writer)
                    else:
                        self._subs.setdefault(topic, set()).add(writer)
                elif kind == "pub":
                    ev = Event(topic=msg["topic"], payload=msg.get("payload", {}), source=msg.get("source", ""))
                    await self._fanout(ev)
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            for s in self._subs.values():
                s.discard(writer)
            self._all.discard(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _fanout(self, ev: Event) -> None:
        targets = set(self._all) | self._subs.get(ev.topic, set())
        line = ev.to_line()
        for w in list(targets):
            try:
                w.write(line)
                await w.drain()
            except (ConnectionResetError, BrokenPipeError):
                self._all.discard(w)
                for s in self._subs.values():
                    s.discard(w)


class BusClient:
    def __init__(self, source: str, sock_path: Path = config.BUS_SOCKET):
        self.source = source
        self.sock_path = sock_path
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._handlers: list[tuple[str, Callable[[Event], Awaitable[None]]]] = []
        self._task: asyncio.Task | None = None

    async def connect(self) -> None:
        self._reader, self._writer = await asyncio.open_unix_connection(str(self.sock_path))

    async def publish(self, topic: str, payload: dict[str, Any] | None = None) -> None:
        assert self._writer is not None, "call connect() first"
        msg = {"kind": "pub", "topic": topic, "payload": payload or {}, "source": self.source}
        self._writer.write((json.dumps(msg) + "\n").encode())
        await self._writer.drain()

    async def subscribe(self, topic: str, handler: Callable[[Event], Awaitable[None]]) -> None:
        assert self._writer is not None, "call connect() first"
        self._handlers.append((topic, handler))
        self._writer.write((json.dumps({"kind": "sub", "topic": topic}) + "\n").encode())
        await self._writer.drain()
        if self._task is None:
            self._task = asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        assert self._reader is not None
        while not self._reader.at_eof():
            line = await self._reader.readline()
            if not line:
                break
            ev = Event.from_line(line)
            for topic, handler in self._handlers:
                if topic == "*" or topic == ev.topic:
                    await handler(ev)

    async def close(self) -> None:
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
