from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional, Tuple, List


class LogStreamManager:
    """Manages async subscribers for streaming application logs."""

    def __init__(self) -> None:
        self._subscribers: List[Tuple[asyncio.Queue, Optional[str]]] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def subscribe(self, contains: Optional[str] = None) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        keyword = contains.lower() if contains else None
        self._subscribers.append((queue, keyword))
        return queue

    async def unsubscribe(self, target: asyncio.Queue) -> None:
        self._subscribers = [sub for sub in self._subscribers if sub[0] != target]

    async def _publish(self, entry: dict) -> None:
        for queue, keyword in list(self._subscribers):
            if keyword and keyword not in entry["message"].lower():
                continue
            try:
                queue.put_nowait(entry)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(entry)
                except asyncio.QueueFull:
                    pass

    def publish(self, entry: dict) -> None:
        if not self._loop or not self._subscribers:
            return
        asyncio.run_coroutine_threadsafe(self._publish(entry), self._loop)


class LogStreamingHandler(logging.Handler):
    """Logging handler that forwards log records to the stream manager."""

    def __init__(self, manager: LogStreamManager) -> None:
        super().__init__()
        self.manager = manager

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:  # pragma: no cover - fallback if formatting fails
            message = record.getMessage()

        entry = {
            "message": message,
            "level": record.levelname,
            "logger": record.name,
            "timestamp": record.created,
            "module": record.module,
        }

        # Include extra metadata when available (e.g., server_id)
        for key in ("server_id", "model_name", "backend_type"):
            if hasattr(record, key):
                entry[key] = getattr(record, key)

        self.manager.publish(entry)


log_stream_manager = LogStreamManager()
log_stream_handler = LogStreamingHandler(log_stream_manager)
log_stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
