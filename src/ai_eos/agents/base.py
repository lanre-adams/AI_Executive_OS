"""Specialist agents and the prompt library.

A specialist is configuration (agents.yaml) + a prompt (prompts/<name>.md) + an allow-list of
tools. The class below implements the shared reasoning loop: think -> call a tool -> observe ->
repeat -> final answer, using a JSON protocol that works with every supported model.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from ai_eos.config import AgentDef, Settings
from ai_eos.domain.models import AgentResult, ChatMessage, Subtask, ToolCallRecord
from ai_eos.infrastructure.repositories import PromptRepo
from ai_eos.llm.base import LLMRequest, parse_json
from ai_eos.llm.router import LLMRouter
from ai_eos.tools.base import ToolContext, ToolExecutor, ToolRegistry


class PromptLibrary:
    """Loads prompts from package files; admin edits in the dashboard override them (stored in DB)."""

    def __init__(self, repo: PromptRepo | None = None) -> None:
        self.repo = repo
        self._overrides: dict[str, str] = {}

    async def refresh(self) -> None:
        if self.repo:
            self._overrides = await self.repo.get_all()

    @staticmethod
    def default(name: str) -> str:
        path = resources.files("ai_eos.agents.prompts").joinpath(f"{name}.md")
        if not path.is_file():
            raise KeyError(f"prompt '{name}' not found")
        return path.read_text(encoding="utf-8")

    def get(self, name: str) -> str:
        return self._overrides.get(name) or self.default(name)

    def names(self) -> list[str]:
        files = resources.files("ai_eos.agents.prompts").iterdir()
        return sorted(f.name[:-3] for f in files if f.name.endswith(".md"))

    def is_overridden(self, name: str) -> bool:
        return name in self._overrides

    async def set(self, name: str, content: str, user_id: str) -> None:
        self.default(name)  # raises if unknown
        assert self.repo is not None
        await self.repo.set(name, content, user_id)
        self._overrides[name] = content

    async def reset(self, name: str) -> None:
        assert self.repo is not None
        await self.repo.reset(name)
        self._overrides.pop(name, None)


class SpecialistAgent:
    def __init__(
        self,
        key: str,
        definition: AgentDef,
        settings: Settings,
        llm: LLMRouter,
        prompts: PromptLibrary,
        tools: ToolRegistry,
        executor: ToolExecutor,
    ) -> None:
        self.key, self.definition, self.settings = key, definition, settings
        self.llm, self.prompts, self.tools, self.executor = llm, prompts, tools, executor

    @property
    def allowed_tools(self) -> list[str]:
        return [t for t in self.definition.tools if self.tools.get(t)]

    def system_prompt(self) -> str:
        tool_lines = "\n".join(s.as_prompt() for s in self.tools.specs(self.allowed_tools)) or "(none)"
        return "\n\n".join(
            [
                self.prompts.get(self.definition.prompt),
                self.prompts.get("_shared_rules"),
                f"## Your tools\n{tool_lines}",
                self.prompts.get("_tool_protocol"),
                f"Step budget: {self.settings.orchestration.max_agent_steps} tool calls.",
            ]
        )

    def _assignment(self, subtask: Subtask, context: str, feedback: str, previous: str) -> str:
        parts = [f"YOUR ASSIGNMENT:\n{subtask.instruction}"]
        if subtask.expected_output:
            parts.append(f"EXPECTED OUTPUT:\n{subtask.expected_output}")
        if context:
            parts.append(f"CONTEXT FROM THE CHIEF OF STAFF:\n{context}")
        if feedback:
            parts.append(f"REVIEWER FEEDBACK ON YOUR PREVIOUS ATTEMPT:\n{feedback}\n\nPREVIOUS OUTPUT:\n{previous}")
        return "\n\n".join(parts)

    async def _ask(self, messages: list[ChatMessage], provider: str | None = None) -> str:
        req = LLMRequest(
            system=self.system_prompt(),
            messages=messages,
            temperature=self.settings.llm.temperature,
            max_tokens=self.settings.llm.max_tokens,
            json_mode=True,
            purpose="agent",
        )
        resp = await self.llm.complete(req, self.definition.provider or provider, self.definition.model)
        return resp.text

    async def run(
        self,
        subtask: Subtask,
        ctx: ToolContext,
        context: str = "",
        feedback: str = "",
        previous: str = "",
        provider: str | None = None,
    ) -> AgentResult:
        messages = [ChatMessage(role="user", content=self._assignment(subtask, context, feedback, previous))]
        calls: list[ToolCallRecord] = []
        budget = self.settings.orchestration.max_agent_steps
        try:
            for step in range(budget + 1):
                raw = await self._ask(messages, provider)
                try:
                    decision = parse_json(raw)
                except ValueError:
                    return self._result(subtask, raw.strip(), calls)  # model answered in prose: accept it
                action = decision.get("action", "final")
                if action != "tool" or step == budget:
                    content = decision.get("content")
                    if action == "tool":  # out of budget: force a final answer
                        messages += [
                            ChatMessage(role="assistant", content=raw),
                            ChatMessage(
                                role="user",
                                content="Step budget exhausted. Reply now with "
                                '{"action": "final", "content": ...} using what you have.',
                            ),
                        ]
                        raw = await self._ask(messages, provider)
                        try:
                            content = parse_json(raw).get("content")
                        except ValueError:
                            content = raw
                    if not isinstance(content, str):
                        content = json.dumps(content, indent=2) if content is not None else raw
                    return self._result(subtask, content, calls)

                name, args = str(decision.get("tool", "")), decision.get("args") or {}
                record = await self.executor.execute(name, args, ctx, self.allowed_tools)
                calls.append(record)
                messages += [
                    ChatMessage(role="assistant", content=raw),
                    ChatMessage(role="user", content=f"TOOL RESULT ({name}, ok={record.ok}):\n{record.result}"),
                ]
        except Exception as exc:  # noqa: BLE001 - surface failure to the Chief of Staff instead of crashing
            return AgentResult(
                subtask_id=subtask.id,
                agent=self.key,
                output="",
                tool_calls=calls,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        raise AssertionError("unreachable")  # pragma: no cover

    def _result(self, subtask: Subtask, content: str, calls: list[ToolCallRecord]) -> AgentResult:
        pending = [c.approval_id for c in calls if c.approval_id]
        return AgentResult(
            subtask_id=subtask.id, agent=self.key, output=content, tool_calls=calls, pending_approvals=pending
        )

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.definition.title,
            "description": self.definition.description,
            "tools": self.allowed_tools,
            "provider": self.definition.provider or self.settings.llm.default_provider,
            "model": self.definition.model,
        }
