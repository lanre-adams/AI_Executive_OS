"""Shared fixtures. Every test runs offline: SQLite file DB, in-memory Qdrant and KV, scripted LLMs."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient

from ai_eos.config import Settings, load_settings
from ai_eos.container import Container, build_container
from ai_eos.infrastructure.vector import HashingEmbedder, VectorStore
from ai_eos.llm.base import LLMRequest, LLMResponse
from ai_eos.security.auth import hash_password

ROOT = Path(__file__).resolve().parents[1]
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "Adm1nPassword!"


class ScriptedLLM:
    """Test double: returns responses chosen by request purpose.

    `script` maps purpose -> list of responses (consumed in order; last one repeats) or a callable(request)->str.
    Every request is recorded in `calls`.
    """

    def __init__(self, script: dict[str, Any] | None = None, name: str = "scripted") -> None:
        self.name, self.model = name, "scripted-1"
        self.script = script or {}
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        entry = self.script.get(request.purpose)
        if callable(entry):
            text = entry(request)
        elif isinstance(entry, list) and entry:
            text = entry.pop(0) if len(entry) > 1 else entry[0]
        elif entry is not None:
            text = entry
        else:
            text = DEFAULTS[request.purpose](request) if request.purpose in DEFAULTS else "ok"
        if isinstance(text, dict):
            text = json.dumps(text)
        return LLMResponse(text=text, provider=self.name, model=self.model, usage={"input": 3, "output": 5})

    def purposes(self) -> list[str]:
        return [c.purpose for c in self.calls]


DEFAULTS: dict[str, Callable[[LLMRequest], str]] = {
    "plan": lambda r: json.dumps(
        {"intent": "test", "subtasks": [{"id": "t1", "agent": "research_analytics", "instruction": "Research it"}]}
    ),
    "agent": lambda r: json.dumps({"action": "final", "content": "A thorough answer with specifics."}),
    "review": lambda r: json.dumps({"score": 0.9, "passed": True, "feedback": ""}),
    "synthesize": lambda r: "Combined answer.",
    "reflect": lambda r: json.dumps({"facts": [], "preferences": [], "lessons": [], "summary": "did a test"}),
}


def make_settings(tmp_path: Path, **overrides: str) -> Settings:
    env = {
        "EOS__DATABASE__URL": f"sqlite+aiosqlite:///{tmp_path / 'eos.db'}",
        "EOS__VECTOR__PERSIST_LOCAL": "false",
        "EOS__TOOLS__SANDBOX_ROOT": str(tmp_path / "workspace"),
        "EOS__TOOLS__AUTO_APPROVE": "[]",
        "EOS__APP__LOG_JSON": "false",
        "EOS__SECURITY__RATE_LIMIT_PER_MINUTE": "1000",
        "EOS_JWT_SECRET": "test-secret-" + "x" * 40,
        "EOS_ENCRYPTION_KEY": "",
        **overrides,
    }
    # Run the same suite against real PostgreSQL / Redis by exporting these (CI and local verification).
    if os.environ.get("EOS_TEST_DATABASE_URL"):
        env["EOS__DATABASE__URL"] = os.environ["EOS_TEST_DATABASE_URL"]
    if os.environ.get("EOS_TEST_REDIS_URL"):
        env["EOS__REDIS__URL"] = os.environ["EOS_TEST_REDIS_URL"]
    return load_settings(ROOT / "config" / "settings.yaml", ROOT / "config" / "agents.yaml", environ=env)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def vectors() -> VectorStore:
    return VectorStore(QdrantClient(location=":memory:"), HashingEmbedder(384))


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
async def container(settings: Settings, vectors: VectorStore, llm: ScriptedLLM) -> Container:
    c = build_container(settings, vectors=vectors)
    c.llm.register("offline", llm)
    await reset_external_state(c)
    await c.startup()
    yield c
    await c.shutdown()


async def reset_external_state(c: Container) -> None:
    """Shared PostgreSQL/Redis outlive a test; wipe them so every test starts clean."""
    if not c.settings.database.url.startswith("sqlite"):
        from ai_eos.infrastructure.db import Base

        async with c.db.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    if c.settings.redis.url:
        import redis.asyncio as redis

        r = redis.from_url(c.settings.redis.url)
        await r.flushdb()
        await r.aclose()


@pytest.fixture
async def admin(container: Container) -> dict[str, Any]:
    return await container.users.create(ADMIN_EMAIL, hash_password(ADMIN_PASSWORD), "admin")
