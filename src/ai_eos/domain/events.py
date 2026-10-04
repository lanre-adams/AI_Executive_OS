"""Domain events and an in-process event bus.

The bus decouples orchestration from observers (dashboard live feed, audit, metrics).
Subscribers receive every event; the dashboard filters by user.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from ai_eos.domain.models import new_id, utcnow


class Event(BaseModel):
    id: str = Field(default_factory=new_id)
    type: str
    user_id: str | None = None
    task_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    at: str = Field(default_factory=lambda: utcnow().isoformat())


# Event type names used across the system
TASK_CREATED = "task.created"
TASK_PLANNED = "task.planned"
TASK_COMPLETED = "task.completed"
TASK_FAILED = "task.failed"
TASK_NEEDS_INPUT = "task.needs_input"
SUBTASK_STARTED = "subtask.started"
SUBTASK_COMPLETED = "subtask.completed"
SUBTASK_REVIEWED = "subtask.reviewed"
TOOL_CALLED = "tool.called"
APPROVAL_REQUESTED = "approval.requested"
APPROVAL_DECIDED = "approval.decided"
SECURITY_FLAG = "security.flag"


class EventBus:
    def __init__(self, history: int = 500) -> None:
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self._recent: deque[Event] = deque(maxlen=history)

    async def publish(self, event: Event) -> None:
        self._recent.append(event)
        for queue in list(self._subscribers):
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(event)

    def recent(self, user_id: str | None = None, limit: int = 100) -> list[Event]:
        items = [e for e in self._recent if user_id is None or e.user_id in (None, user_id)]
        return items[-limit:]

    @contextlib.asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[Event]]:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=1000)
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
