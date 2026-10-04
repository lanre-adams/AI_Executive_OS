"""Provider-neutral request/response types and helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ai_eos.domain.models import ChatMessage


class LLMRequest(BaseModel):
    system: str = ""
    messages: list[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = 2048
    json_mode: bool = False
    # A hint describing what the call is for (plan, agent, review, synthesize, reflect).
    # Real providers ignore it; the offline provider uses it to produce sensible output.
    purpose: str = "chat"


class LLMResponse(BaseModel):
    text: str
    provider: str
    model: str
    usage: dict[str, int] = Field(default_factory=dict)


class LLMError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str
    model: str

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output (tolerates code fences and prose)."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        start = cand.find("{")
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(cand)):
                ch = cand[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            data = json.loads(cand[start : i + 1])
                            if isinstance(data, dict):
                                return data
                        except json.JSONDecodeError:
                            pass
                        break
            start = cand.find("{", start + 1)
    raise ValueError("no JSON object found in model output")
