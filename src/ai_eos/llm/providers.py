"""Concrete LLM providers.

- OpenAICompatibleProvider: OpenAI, DeepSeek, Mistral, Ollama (Llama), vLLM, LM Studio, Azure-style gateways
- AnthropicProvider: Claude (Messages API)
- GeminiProvider: Google Gemini (generateContent)
- OfflineProvider: deterministic, no network; makes the whole system runnable without keys
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx

from ai_eos.llm.base import LLMError, LLMRequest, LLMResponse

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class _HTTPProvider:
    name: str
    model: str

    def __init__(
        self,
        name: str,
        model: str,
        base_url: str,
        api_key: str,
        timeout: float = 120,
        retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name, self.model, self.base_url, self.api_key = name, model, base_url.rstrip("/"), api_key
        self.timeout, self.retries, self._transport = timeout, retries, transport

    async def _post(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        delay = 1.0
        last: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
            for attempt in range(self.retries + 1):
                try:
                    r = await client.post(url, headers=headers, json=body)
                except httpx.HTTPError as exc:
                    last = exc
                else:
                    if r.status_code < 400:
                        return r.json()
                    last = LLMError(f"{self.name} HTTP {r.status_code}: {r.text[:300]}")
                    if r.status_code not in RETRY_STATUS:
                        raise last
                if attempt < self.retries:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 20)
        raise LLMError(f"{self.name} request failed after retries: {last}")


class OpenAICompatibleProvider(_HTTPProvider):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages += [m.model_dump() for m in request.messages]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = await self._post(f"{self.base_url}/chat/completions", headers, body)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected response shape from {self.name}") from exc
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            usage={"input": usage.get("prompt_tokens", 0), "output": usage.get("completion_tokens", 0)},
        )


class AnthropicProvider(_HTTPProvider):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        system = request.system
        if request.json_mode:
            system += "\n\nRespond with a single JSON object and nothing else."
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [m.model_dump() for m in request.messages if m.role != "system"],
        }
        if system:
            body["system"] = system
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data = await self._post(f"{self.base_url}/v1/messages", headers, body)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            usage={"input": usage.get("input_tokens", 0), "output": usage.get("output_tokens", 0)},
        )


class GeminiProvider(_HTTPProvider):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in request.messages
            if m.role != "system"
        ]
        gen: dict[str, Any] = {"temperature": request.temperature, "maxOutputTokens": request.max_tokens}
        if request.json_mode:
            gen["responseMimeType"] = "application/json"
        body: dict[str, Any] = {"contents": contents, "generationConfig": gen}
        if request.system:
            body["systemInstruction"] = {"parts": [{"text": request.system}]}
        headers = {"x-goog-api-key": self.api_key, "content-type": "application/json"}
        data = await self._post(f"{self.base_url}/models/{self.model}:generateContent", headers, body)
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError) as exc:
            raise LLMError("Gemini returned no candidates (possibly blocked by safety filters)") from exc
        usage = data.get("usageMetadata") or {}
        return LLMResponse(
            text="".join(p.get("text", "") for p in parts),
            provider=self.name,
            model=self.model,
            usage={"input": usage.get("promptTokenCount", 0), "output": usage.get("candidatesTokenCount", 0)},
        )


# --------------------------------------------------------------------------------------
# Offline provider
# --------------------------------------------------------------------------------------

_ROUTES: list[tuple[str, tuple[str, ...]]] = [
    (
        "executive_assistant",
        (
            "email",
            "inbox",
            "calendar",
            "meeting",
            "schedule",
            "remind",
            "travel",
            "agenda",
            "follow up",
            "follow-up",
            "draft a",
            "minutes",
        ),
    ),
    (
        "research_analytics",
        (
            "research",
            "analy",
            "market",
            "competit",
            "trend",
            "data",
            "report",
            "statistic",
            "compare",
            "study",
            "benchmark",
            "forecast",
        ),
    ),
    (
        "operations",
        (
            "project",
            "sop",
            "process",
            "risk",
            "kpi",
            "workflow",
            "compliance",
            "plan",
            "roadmap",
            "work item",
            "backlog",
            "operational",
        ),
    ),
    (
        "software_engineer",
        (
            "code",
            "api",
            "bug",
            "function",
            "database",
            "schema",
            "backend",
            "frontend",
            "python",
            "review",
            "refactor",
            "test",
        ),
    ),
    (
        "devops_engineer",
        (
            "deploy",
            "pipeline",
            "ci/cd",
            "docker",
            "kubernetes",
            "terraform",
            "infrastructure",
            "monitor",
            "cloud",
            "azure",
            "aws",
            "helm",
        ),
    ),
]
_GREETING = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you)\b[\s!.,]*$", re.I)
_REMEMBER = re.compile(r"\b(?:remember that|note that|my preference is|i prefer)\s+(.+?)(?:[.!]|$)", re.I)


def _section(text: str, header: str) -> str:
    """Return the text after `HEADER:` up to the next blank-line-separated ALLCAPS header."""
    m = re.search(rf"{re.escape(header)}:\s*\n?(.*?)(?:\n\n[A-Z][A-Z _]+:|\Z)", text, re.S)
    return m.group(1).strip() if m else ""


class OfflineProvider:
    """A rule-based stand-in for an LLM.

    It keeps the full pipeline (planning, delegation, review, synthesis, memory) runnable
    with no API keys so you can verify an installation. Its answers are templated, not
    intelligent; switch llm.default_provider to a real model for actual work.
    """

    def __init__(self, name: str = "offline", model: str = "offline-1") -> None:
        self.name, self.model = name, model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        prompt = "\n\n".join(m.content for m in request.messages if m.role == "user")
        handler = getattr(self, f"_{request.purpose}", self._chat)
        text = handler(prompt)
        return LLMResponse(text=text, provider=self.name, model=self.model, usage={"input": 0, "output": 0})

    def _plan(self, prompt: str) -> str:
        req = _section(prompt, "USER REQUEST") or prompt
        if _GREETING.match(req):
            return json.dumps(
                {"intent": "greeting", "direct_answer": "Hello. What would you like the team to work on?"}
            )
        if len(req.split()) < 3:
            return json.dumps(
                {
                    "intent": "unclear",
                    "needs_clarification": True,
                    "clarifying_question": f"Could you say a little more about what you need regarding '{req}'?",
                }
            )
        lowered = req.lower()
        agents = [agent for agent, words in _ROUTES if any(w in lowered for w in words)] or ["research_analytics"]
        subtasks = [
            {
                "id": f"t{i + 1}",
                "agent": a,
                "instruction": f"Handle the {a.replace('_', ' ')} aspects of: {req}",
                "depends_on": [],
                "expected_output": "A concise, actionable result",
            }
            for i, a in enumerate(agents[:4])
        ]
        urgent = any(w in lowered for w in ("urgent", "asap", "immediately", "today"))
        return json.dumps({"intent": req[:160], "priority": "high" if urgent else "normal", "subtasks": subtasks})

    def _agent(self, prompt: str) -> str:
        instruction = _section(prompt, "YOUR ASSIGNMENT") or prompt[:400]
        content = (
            f"**Offline draft** for: {instruction}\n\n"
            "1. Clarify the objective and success criteria.\n"
            "2. Gather the relevant facts and constraints.\n"
            "3. Produce the deliverable and list next actions.\n\n"
            "_Generated by the offline model. Configure a real LLM provider for substantive output._"
        )
        return json.dumps({"action": "final", "content": content})

    def _review(self, prompt: str) -> str:
        output = _section(prompt, "AGENT OUTPUT")
        score = 0.8 if len(output) > 40 else 0.3
        return json.dumps(
            {
                "score": score,
                "passed": score >= 0.6,
                "feedback": "" if score >= 0.6 else "Too thin; expand with specifics.",
            }
        )

    def _synthesize(self, prompt: str) -> str:
        req = _section(prompt, "USER REQUEST")
        outputs = _section(prompt, "TEAM OUTPUTS")
        return f"Here is the team's consolidated response to: *{req}*\n\n{outputs}"

    def _reflect(self, prompt: str) -> str:
        req = _section(prompt, "USER REQUEST")
        facts = [m.group(1).strip() for m in _REMEMBER.finditer(req)]
        return json.dumps({"facts": facts, "preferences": [], "lessons": [], "summary": req[:200]})

    def _chat(self, prompt: str) -> str:
        return f"(offline) {prompt[-500:]}"
