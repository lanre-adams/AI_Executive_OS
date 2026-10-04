"""Tool plug-in framework: specs, registry, credential resolution and the approval-aware executor."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from prometheus_client import Counter

from ai_eos.config import Settings
from ai_eos.domain import events as ev
from ai_eos.domain.events import Event, EventBus
from ai_eos.domain.models import ApprovalStatus, ToolCallRecord
from ai_eos.infrastructure.kv import KeyValueStore
from ai_eos.infrastructure.repositories import ApprovalRepo, AuditRepo, SecretRepo
from ai_eos.security.crypto import Cipher
from ai_eos.security.guard import wrap_untrusted

log = logging.getLogger(__name__)
TOOL_CALLS = Counter("eos_tool_calls_total", "Tool invocations", ["tool", "outcome"])


class ToolError(Exception):
    """Raised by tools for expected failures; the message is shown to the agent."""


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    requires_approval: bool = False
    untrusted_output: bool = True  # wrap output as external data
    timeout: float = 60

    def as_prompt(self) -> str:
        props = self.parameters.get("properties", {})
        required = set(self.parameters.get("required", []))
        args = ", ".join(f"{k}{'' if k in required else '?'}: {v.get('type', 'string')}" for k, v in props.items())
        flag = " [needs user approval]" if self.requires_approval else ""
        return f"- {self.name}({args}){flag}: {self.description}"


class SecretResolver:
    """Looks up credentials: environment variable first, then the encrypted secrets table."""

    def __init__(self, repo: SecretRepo | None, cipher: Cipher, environ: dict[str, str] | None = None) -> None:
        self.repo, self.cipher = repo, cipher
        self.environ = environ if environ is not None else os.environ

    async def get(self, name: str) -> str:
        if not name:
            return ""
        if self.environ.get(name):
            return self.environ[name]
        if self.repo:
            token = await self.repo.get(name)
            if token:
                return self.cipher.decrypt(token)
        return ""


@dataclass
class ToolContext:
    user_id: str
    agent: str
    task_id: str | None
    settings: Settings
    secrets: SecretResolver
    kv: KeyValueStore
    services: dict[str, Any] = field(default_factory=dict)  # e.g. {"memory": MemoryManager}
    transport: httpx.AsyncBaseTransport | None = None  # injected in tests

    def http(self, timeout: float | None = None, **kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=timeout or self.settings.tools.http_timeout_seconds,
            transport=self.transport,
            follow_redirects=kwargs.pop("follow_redirects", False),
            **kwargs,
        )


class Tool(ABC):
    spec: ToolSpec

    @abstractmethod
    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str: ...

    def summarize(self, args: dict[str, Any]) -> str:
        """Human-readable one-liner shown on approval cards."""
        text = json.dumps(args, ensure_ascii=False)
        return f"{self.spec.name} {text[:300]}"


class FunctionTool(Tool):
    """Wrap an async function as a tool - handy for plug-ins."""

    def __init__(self, spec: ToolSpec, fn: Callable[[dict[str, Any], ToolContext], Awaitable[str]]) -> None:
        self.spec, self._fn = spec, fn

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return await self._fn(args, ctx)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, names: list[str] | None = None) -> list[ToolSpec]:
        selected = names if names is not None else self.names()
        return [self._tools[n].spec for n in selected if n in self._tools]

    def load_plugins(self, modules: list[str], settings: Settings) -> None:
        for mod in modules:
            importlib.import_module(mod).register(self, settings)


def validate_args(spec: ToolSpec, args: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ToolError("arguments must be a JSON object")
    props = spec.parameters.get("properties", {})
    missing = [k for k in spec.parameters.get("required", []) if k not in args or args[k] in (None, "")]
    if missing:
        raise ToolError(f"missing required argument(s): {', '.join(missing)}")
    unknown = [k for k in args if k not in props]
    if unknown:
        raise ToolError(f"unknown argument(s): {', '.join(unknown)}")
    type_map = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
    for k, v in args.items():
        expected = type_map.get(props[k].get("type", "string"))
        if expected and not isinstance(v, expected):
            raise ToolError(f"argument '{k}' must be of type {props[k].get('type')}")
    return args


class ToolExecutor:
    """Runs tool calls for agents, enforcing allow-lists, approvals, timeouts and auditing."""

    def __init__(
        self,
        registry: ToolRegistry,
        approvals: ApprovalRepo,
        audit: AuditRepo,
        bus: EventBus,
        cipher: Cipher,
        auto_approve: list[str] | None = None,
    ) -> None:
        self.registry, self.approvals, self.audit, self.bus, self.cipher = registry, approvals, audit, bus, cipher
        self.auto_approve = set(auto_approve or [])

    def needs_approval(self, spec: ToolSpec) -> bool:
        return spec.requires_approval and spec.name not in self.auto_approve

    async def execute(self, name: str, args: Any, ctx: ToolContext, allowed: list[str]) -> ToolCallRecord:
        record = ToolCallRecord(tool=name, args=args if isinstance(args, dict) else {"_raw": str(args)})
        tool = self.registry.get(name)
        if tool is None or name not in allowed:
            record.ok, record.result = False, f"Tool '{name}' is not available to you. Allowed: {', '.join(allowed)}"
            TOOL_CALLS.labels(name, "denied").inc()
            return record
        try:
            validate_args(tool.spec, args)
        except ToolError as exc:
            record.ok, record.result = False, f"Invalid arguments: {exc}"
            TOOL_CALLS.labels(name, "invalid").inc()
            return record

        if self.needs_approval(tool.spec):
            approval_id = await self.approvals.create(
                ctx.user_id, ctx.task_id, ctx.agent, name, self.cipher.encrypt(json.dumps(args)), tool.summarize(args)
            )
            record.approval_id = approval_id
            record.result = (
                f"Queued for the user's approval (id {approval_id}). It has NOT run yet. "
                "Continue with the rest of your work and mention that this action awaits approval."
            )
            await self.bus.publish(
                Event(
                    type=ev.APPROVAL_REQUESTED,
                    user_id=ctx.user_id,
                    task_id=ctx.task_id,
                    data={
                        "approval_id": approval_id,
                        "tool": name,
                        "agent": ctx.agent,
                        "summary": tool.summarize(args),
                    },
                )
            )
            await self.audit.log(
                "tool.approval_requested", ctx.user_id, name, {"approval_id": approval_id, "agent": ctx.agent}
            )
            TOOL_CALLS.labels(name, "pending_approval").inc()
            return record

        ok, output = await self._run(tool, args, ctx)
        record.ok, record.result = ok, output
        await self.audit.log("tool.call", ctx.user_id, name, {"agent": ctx.agent, "ok": ok, "task_id": ctx.task_id})
        await self.bus.publish(
            Event(
                type=ev.TOOL_CALLED,
                user_id=ctx.user_id,
                task_id=ctx.task_id,
                data={"tool": name, "agent": ctx.agent, "ok": ok},
            )
        )
        return record

    async def _run(self, tool: Tool, args: dict[str, Any], ctx: ToolContext) -> tuple[bool, str]:
        try:
            raw = await asyncio.wait_for(tool.run(args, ctx), timeout=tool.spec.timeout)
            TOOL_CALLS.labels(tool.spec.name, "ok").inc()
            return True, wrap_untrusted(tool.spec.name, raw) if tool.spec.untrusted_output else raw
        except TimeoutError:
            TOOL_CALLS.labels(tool.spec.name, "timeout").inc()
            return False, f"Tool '{tool.spec.name}' timed out after {tool.spec.timeout}s"
        except ToolError as exc:
            TOOL_CALLS.labels(tool.spec.name, "error").inc()
            return False, f"Tool error: {exc}"
        except httpx.HTTPStatusError as exc:
            TOOL_CALLS.labels(tool.spec.name, "error").inc()
            return False, f"Tool error: HTTP {exc.response.status_code} from {exc.request.url.host}"
        except Exception as exc:  # noqa: BLE001 - a broken tool must not crash the agent
            log.exception("tool %s failed", tool.spec.name)
            TOOL_CALLS.labels(tool.spec.name, "error").inc()
            return False, f"Tool error: {type(exc).__name__}: {exc}"

    async def decide(self, approval_id: str, user_id: str, approve: bool, ctx: ToolContext) -> dict[str, Any]:
        row = await self.approvals.get_raw(approval_id, user_id)
        if row is None:
            raise KeyError(approval_id)
        if row.status != ApprovalStatus.PENDING.value:
            raise ValueError(f"approval already {row.status}")
        if not approve:
            await self.approvals.decide(approval_id, ApprovalStatus.REJECTED, user_id)
            await self.audit.log("tool.approval_rejected", user_id, row.tool, {"approval_id": approval_id})
            await self.bus.publish(
                Event(
                    type=ev.APPROVAL_DECIDED,
                    user_id=user_id,
                    task_id=row.task_id,
                    data={"approval_id": approval_id, "status": "rejected"},
                )
            )
            return {"id": approval_id, "status": "rejected"}
        tool = self.registry.get(row.tool)
        if tool is None:
            raise ValueError(f"tool {row.tool} no longer registered")
        args = json.loads(self.cipher.decrypt(row.args_encrypted))
        ctx.agent, ctx.task_id = row.agent, row.task_id
        ok, output = await self._run(tool, args, ctx)
        status = ApprovalStatus.EXECUTED if ok else ApprovalStatus.FAILED
        await self.approvals.decide(approval_id, status, user_id, output[:4000])
        await self.audit.log("tool.approval_executed", user_id, row.tool, {"approval_id": approval_id, "ok": ok})
        await self.bus.publish(
            Event(
                type=ev.APPROVAL_DECIDED,
                user_id=user_id,
                task_id=row.task_id,
                data={"approval_id": approval_id, "status": status.value},
            )
        )
        return {"id": approval_id, "status": status.value, "result": output}
