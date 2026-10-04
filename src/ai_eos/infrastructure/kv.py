"""Key-value store used for short-term memory and rate limiting.

RedisKV is used when redis.url is configured; InMemoryKV otherwise (single replica only).
"""

from __future__ import annotations

import json
import time
from typing import Any, Protocol


class KeyValueStore(Protocol):
    async def push(self, key: str, value: dict[str, Any], max_len: int, ttl: int) -> None: ...
    async def range(self, key: str, count: int) -> list[dict[str, Any]]: ...
    async def incr_window(self, key: str, window_seconds: int) -> int: ...
    async def set(self, key: str, value: str, ttl: int) -> None: ...
    async def get(self, key: str) -> str | None: ...
    async def delete(self, key: str) -> None: ...
    async def ping(self) -> bool: ...


class InMemoryKV:
    def __init__(self) -> None:
        self._lists: dict[str, tuple[list[dict[str, Any]], float]] = {}
        self._values: dict[str, tuple[str, float]] = {}
        self._counters: dict[str, tuple[int, float]] = {}

    @staticmethod
    def _alive(expiry: float) -> bool:
        return expiry == 0 or expiry > time.monotonic()

    async def push(self, key: str, value: dict[str, Any], max_len: int, ttl: int) -> None:
        items, expiry = self._lists.get(key, ([], 0.0))
        if not self._alive(expiry):
            items = []
        items = (items + [value])[-max_len:]
        self._lists[key] = (items, time.monotonic() + ttl if ttl else 0.0)

    async def range(self, key: str, count: int) -> list[dict[str, Any]]:
        items, expiry = self._lists.get(key, ([], 0.0))
        return list(items[-count:]) if self._alive(expiry) else []

    async def incr_window(self, key: str, window_seconds: int) -> int:
        count, expiry = self._counters.get(key, (0, 0.0))
        now = time.monotonic()
        if expiry <= now:
            count, expiry = 0, now + window_seconds
        count += 1
        self._counters[key] = (count, expiry)
        return count

    async def set(self, key: str, value: str, ttl: int) -> None:
        self._values[key] = (value, time.monotonic() + ttl if ttl else 0.0)

    async def get(self, key: str) -> str | None:
        value, expiry = self._values.get(key, (None, 0.0))
        return value if value is not None and self._alive(expiry) else None

    async def delete(self, key: str) -> None:
        self._values.pop(key, None)
        self._lists.pop(key, None)
        self._counters.pop(key, None)

    async def ping(self) -> bool:
        return True


class RedisKV:  # pragma: no cover - exercised in Docker integration environments
    def __init__(self, url: str) -> None:
        import redis.asyncio as redis

        self._r = redis.from_url(url, decode_responses=True)

    async def push(self, key: str, value: dict[str, Any], max_len: int, ttl: int) -> None:
        pipe = self._r.pipeline()
        pipe.rpush(key, json.dumps(value))
        pipe.ltrim(key, -max_len, -1)
        if ttl:
            pipe.expire(key, ttl)
        await pipe.execute()

    async def range(self, key: str, count: int) -> list[dict[str, Any]]:
        return [json.loads(v) for v in await self._r.lrange(key, -count, -1)]

    async def incr_window(self, key: str, window_seconds: int) -> int:
        pipe = self._r.pipeline()
        pipe.incr(key)
        pipe.expire(key, window_seconds, nx=True)
        count, _ = await pipe.execute()
        return int(count)

    async def set(self, key: str, value: str, ttl: int) -> None:
        await self._r.set(key, value, ex=ttl or None)

    async def get(self, key: str) -> str | None:
        return await self._r.get(key)

    async def delete(self, key: str) -> None:
        await self._r.delete(key)

    async def ping(self) -> bool:
        try:
            return bool(await self._r.ping())
        except Exception:  # noqa: BLE001
            return False


def make_kv(url: str) -> KeyValueStore:
    return RedisKV(url) if url else InMemoryKV()
