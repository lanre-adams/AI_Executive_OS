import json

import httpx
import pytest

from ai_eos.domain.models import ChatMessage
from ai_eos.llm.base import LLMError, LLMRequest, parse_json
from ai_eos.llm.providers import AnthropicProvider, GeminiProvider, OfflineProvider, OpenAICompatibleProvider
from ai_eos.llm.router import LLMRouter, build_provider


def req(**kw) -> LLMRequest:
    return LLMRequest(system="sys", messages=[ChatMessage(role="user", content="hi")], **kw)


# ---------------------------------------------------------------- parse_json
@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('Sure!\n```json\n{"a": 2}\n```', {"a": 2}),
        ('prefix {"a": {"b": "}"}} suffix', {"a": {"b": "}"}}),
        ('{"bad": } then {"a": "x\\"y"}', {"a": 'x"y'}),
    ],
)
def test_parse_json(text: str, expected: dict) -> None:
    assert parse_json(text) == expected


def test_parse_json_failure() -> None:
    with pytest.raises(ValueError):
        parse_json("no json here [1, 2]")


# ---------------------------------------------------------------- providers over HTTP
def transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


async def test_openai_compatible_request_and_parse() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"], seen["body"] = (
            str(request.url),
            request.headers["authorization"],
            json.loads(request.content),
        )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )

    p = OpenAICompatibleProvider("openai", "gpt-x", "https://api.test/v1/", "k", transport=transport(handler))
    r = await p.complete(req(json_mode=True))
    assert r.text == "hello" and r.usage == {"input": 4, "output": 2}
    assert seen["url"] == "https://api.test/v1/chat/completions" and seen["auth"] == "Bearer k"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["body"]["response_format"] == {"type": "json_object"}


async def test_openai_no_key_and_bad_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"weird": True})

    p = OpenAICompatibleProvider("ollama", "llama", "http://x/v1", "", transport=transport(handler))
    with pytest.raises(LLMError, match="unexpected"):
        await p.complete(req())


async def test_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    import ai_eos.llm.providers as prov

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(prov.asyncio, "sleep", no_sleep)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        if calls["n"] == 2:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    p = OpenAICompatibleProvider("openai", "m", "http://x", "k", transport=transport(handler))
    assert (await p.complete(req())).text == "ok"
    assert calls["n"] == 3


async def test_non_retryable_error_and_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    import ai_eos.llm.providers as prov

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(prov.asyncio, "sleep", no_sleep)
    p = OpenAICompatibleProvider(
        "openai", "m", "http://x", "k", transport=transport(lambda r: httpx.Response(401, text="bad key"))
    )
    with pytest.raises(LLMError, match="401"):
        await p.complete(req())
    p = OpenAICompatibleProvider(
        "openai", "m", "http://x", "k", retries=1, transport=transport(lambda r: httpx.Response(503, text="down"))
    )
    with pytest.raises(LLMError, match="after retries"):
        await p.complete(req())


async def test_anthropic_provider() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.headers["x-api-key"] == "k" and request.headers["anthropic-version"]
        assert "JSON" in body["system"] and body["messages"] == [{"role": "user", "content": "hi"}]
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "a"}, {"type": "tool_use"}, {"type": "text", "text": "b"}],
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        )

    p = AnthropicProvider("anthropic", "claude", "https://api.anthropic.com", "k", transport=transport(handler))
    r = await p.complete(req(json_mode=True))
    assert r.text == "ab" and r.usage["output"] == 2


async def test_gemini_provider() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path.endswith("/models/gem:generateContent")
        assert body["systemInstruction"]["parts"][0]["text"] == "sys"
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert body["contents"][1]["role"] == "model"
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "g"}]}}], "usageMetadata": {"promptTokenCount": 7}},
        )

    p = GeminiProvider("gemini", "gem", "https://g/v1beta", "k", transport=transport(handler))
    r = await p.complete(
        LLMRequest(
            system="sys",
            json_mode=True,
            messages=[
                ChatMessage(role="user", content="q"),
                ChatMessage(role="assistant", content="a"),
                ChatMessage(role="user", content="q2"),
            ],
        )
    )
    assert r.text == "g" and r.usage["input"] == 7

    blocked = GeminiProvider(
        "gemini", "gem", "https://g", "k", transport=transport(lambda r: httpx.Response(200, json={"candidates": []}))
    )
    with pytest.raises(LLMError, match="no candidates"):
        await blocked.complete(req())


# ---------------------------------------------------------------- offline provider
async def test_offline_purposes() -> None:
    o = OfflineProvider()

    async def run(purpose: str, content: str) -> str:
        return (
            await o.complete(LLMRequest(messages=[ChatMessage(role="user", content=content)], purpose=purpose))
        ).text

    assert "Hello" in json.loads(await run("plan", "USER REQUEST:\nhello"))["direct_answer"]
    assert json.loads(await run("plan", "USER REQUEST:\nbudget"))["needs_clarification"] is True
    plan = json.loads(await run("plan", "USER REQUEST:\nUrgent: deploy the API to kubernetes today"))
    assert plan["priority"] == "high" and {s["agent"] for s in plan["subtasks"]} >= {"devops_engineer"}
    assert (
        json.loads(await run("plan", "USER REQUEST:\nwhat about the weather"))["subtasks"][0]["agent"]
        == "research_analytics"
    )
    assert json.loads(await run("agent", "YOUR ASSIGNMENT:\ndo x"))["action"] == "final"
    assert json.loads(await run("review", "AGENT OUTPUT:\nshort"))["passed"] is False
    assert json.loads(await run("review", "AGENT OUTPUT:\n" + "long " * 20))["passed"] is True
    assert "TEAM" not in await run("synthesize", "USER REQUEST:\nq\n\nTEAM OUTPUTS:\nstuff")
    assert json.loads(await run("reflect", "USER REQUEST:\nRemember that my team meets on Mondays."))["facts"] == [
        "my team meets on Mondays"
    ]
    assert (await run("other", "x")).startswith("(offline)")


# ---------------------------------------------------------------- router
def test_build_provider_errors(settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        build_provider("openai", settings.llm.providers["openai"], 10)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert isinstance(build_provider("openai", settings.llm.providers["openai"], 10), OpenAICompatibleProvider)
    bad = settings.llm.providers["openai"].model_copy(update={"kind": "mystery"})
    with pytest.raises(LLMError, match="unknown provider kind"):
        build_provider("x", bad, 10)
    assert isinstance(build_provider("ollama", settings.llm.providers["ollama"], 10), OpenAICompatibleProvider)


async def test_router(settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    r = LLMRouter(settings)
    names = {p["name"]: p["configured"] for p in r.available()}
    assert names["offline"] and names["anthropic"] and names["ollama"]
    assert r.get() is r.get("offline")
    assert r.get("anthropic", "claude-other").model == "claude-other"
    with pytest.raises(LLMError):
        r.get("nope")
    resp = await r.complete(req(purpose="plan"))
    assert resp.provider == "offline"

    class Boom:
        name, model = "boom", "b"

        async def complete(self, request):  # noqa: ANN001, ANN201
            raise RuntimeError("x")

    r.register("boom", Boom())
    with pytest.raises(RuntimeError):
        await r.complete(req(), "boom", "any-model")
