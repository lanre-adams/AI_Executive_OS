import base64
import json
from typing import Any

import httpx
import pytest

from ai_eos.tools.base import FunctionTool, SecretResolver, ToolContext, ToolError, ToolSpec, validate_args
from ai_eos.tools.core import WebSearchTool, html_to_text
from ai_eos.tools.google import _plain_body


def ctx_for(container, admin, handler=None, env: dict[str, str] | None = None, public: bool = True) -> ToolContext:
    ctx = container.tool_context(admin["id"], "research_analytics", "task1")
    if handler:
        ctx.transport = httpx.MockTransport(handler)
    ctx.secrets = SecretResolver(container.secret_repo, container.cipher, environ=env or {})
    ctx.services["resolve"] = (lambda host: ["93.184.216.34"]) if public else (lambda host: ["10.0.0.5"])
    return ctx


async def run(container, ctx, name: str, args: dict[str, Any]) -> str:
    rec = await container.executor.execute(name, args, ctx, [name])
    return rec.result


# ---------------------------------------------------------------- framework
def test_validate_args() -> None:
    spec = ToolSpec(
        "t",
        "d",
        {"type": "object", "properties": {"a": {"type": "string"}, "n": {"type": "integer"}}, "required": ["a"]},
    )
    assert validate_args(spec, {"a": "x", "n": 2}) == {"a": "x", "n": 2}
    for bad, msg in [([], "JSON object"), ({}, "missing"), ({"a": "x", "z": 1}, "unknown"), ({"a": 1}, "type")]:
        with pytest.raises(ToolError, match=msg):
            validate_args(spec, bad)
    assert "n?: integer" in spec.as_prompt()


async def test_executor_paths(container, admin) -> None:
    ctx = ctx_for(container, admin)
    rec = await container.executor.execute("web_search", {"query": "x"}, ctx, ["read_file"])
    assert not rec.ok and "not available" in rec.result
    rec = await container.executor.execute("nope", {}, ctx, ["nope"])
    assert not rec.ok
    rec = await container.executor.execute("read_file", {}, ctx, ["read_file"])
    assert not rec.ok and "Invalid arguments" in rec.result

    async def slow(args, ctx):  # noqa: ANN001, ANN202
        import asyncio

        await asyncio.sleep(5)
        return "late"

    async def broken(args, ctx):  # noqa: ANN001, ANN202
        raise RuntimeError("kaboom")

    async def http_fail(args, ctx):  # noqa: ANN001, ANN202
        req = httpx.Request("GET", "https://api.example.com/x")
        raise httpx.HTTPStatusError("x", request=req, response=httpx.Response(404, request=req))

    obj = {"type": "object", "properties": {}, "required": []}
    container.tools.register(FunctionTool(ToolSpec("slow", "s", obj, timeout=0.05), slow))
    container.tools.register(FunctionTool(ToolSpec("broken", "b", obj), broken))
    container.tools.register(FunctionTool(ToolSpec("httpfail", "h", obj), http_fail))
    assert "timed out" in (await container.executor.execute("slow", {}, ctx, ["slow"])).result
    assert "kaboom" in (await container.executor.execute("broken", {}, ctx, ["broken"])).result
    assert "HTTP 404" in (await container.executor.execute("httpfail", {}, ctx, ["httpfail"])).result


async def test_approval_flow(container, admin) -> None:
    ctx = ctx_for(container, admin)
    rec = await container.executor.execute("write_file", {"path": "a/b.txt", "content": "hello"}, ctx, ["write_file"])
    assert rec.approval_id and "NOT run" in rec.result
    pending = await container.approvals.list(admin["id"], "pending")
    assert pending[0]["tool"] == "write_file"
    with pytest.raises(KeyError):
        await container.executor.decide("missing", admin["id"], True, ctx)
    out = await container.executor.decide(rec.approval_id, admin["id"], True, ctx)
    assert out["status"] == "executed" and "Wrote 5" in out["result"]
    with pytest.raises(ValueError, match="already"):
        await container.executor.decide(rec.approval_id, admin["id"], True, ctx)
    rec2 = await container.executor.execute("write_file", {"path": "x.txt", "content": "no"}, ctx, ["write_file"])
    assert (await container.executor.decide(rec2.approval_id, admin["id"], False, ctx))["status"] == "rejected"
    # another user cannot see or decide it
    with pytest.raises(KeyError):
        await container.executor.decide(rec2.approval_id, "someone-else", True, ctx)
    container.executor.auto_approve.add("write_file")
    rec3 = await container.executor.execute("write_file", {"path": "y.txt", "content": "auto"}, ctx, ["write_file"])
    assert rec3.ok and rec3.approval_id is None


async def test_secret_resolver(container) -> None:
    await container.secret_repo.set("MY_TOKEN", container.cipher.encrypt("from-db"))
    r = SecretResolver(container.secret_repo, container.cipher, environ={"ENV_TOKEN": "from-env"})
    assert await r.get("ENV_TOKEN") == "from-env"
    assert await r.get("MY_TOKEN") == "from-db"
    assert await r.get("NONE") == "" and await r.get("") == ""
    assert await SecretResolver(None, container.cipher, environ={}).get("MY_TOKEN") == ""


def test_plugin_loading(container, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    mod = types.ModuleType("fake_plugin")

    def register(registry, settings):  # noqa: ANN001, ANN202
        async def hello(args, ctx):  # noqa: ANN001, ANN202
            return "hi"

        registry.register(FunctionTool(ToolSpec("hello", "h", {"type": "object", "properties": {}}), hello))

    mod.register = register
    monkeypatch.setitem(sys.modules, "fake_plugin", mod)
    container.tools.load_plugins(["fake_plugin"], container.settings)
    assert "hello" in container.tools.names()


# ---------------------------------------------------------------- core tools
async def test_file_tools_and_sandbox(container, admin) -> None:
    container.executor.auto_approve.add("write_file")
    ctx = ctx_for(container, admin)
    allowed = ["write_file", "read_file", "list_files"]
    assert (await container.executor.execute("write_file", {"path": "r/x.md", "content": "# X"}, ctx, allowed)).ok
    assert "# X" in (await container.executor.execute("read_file", {"path": "r/x.md"}, ctx, allowed)).result
    assert "r/x.md" in (await container.executor.execute("list_files", {}, ctx, allowed)).result
    assert (
        "escapes" in (await container.executor.execute("read_file", {"path": "../../etc/passwd"}, ctx, allowed)).result
    )
    assert "no such file" in (await container.executor.execute("read_file", {"path": "nope"}, ctx, allowed)).result
    assert "not a folder" in (await container.executor.execute("list_files", {"path": "r/x.md"}, ctx, allowed)).result


async def test_python_exec(container, admin) -> None:
    container.executor.auto_approve.add("python_exec")
    ctx = ctx_for(container, admin)
    out = await run(
        container, ctx, "python_exec", {"code": "import os\nprint(6*7)\nprint(os.environ.get('EOS_JWT_SECRET'))"}
    )
    assert "exit code 0" in out and "42" in out and "None" in out  # secrets not inherited
    err = await run(container, ctx, "python_exec", {"code": "raise SystemExit('bad')"})
    assert "exit code 1" in err and "[stderr]" in err
    container.settings.tools.python_timeout_seconds = 1
    assert "exceeded" in await run(container, ctx, "python_exec", {"code": "import time\ntime.sleep(5)"})


async def test_memory_and_knowledge_tools(container, admin) -> None:
    ctx = ctx_for(container, admin)
    assert "No memories" in await run(container, ctx, "memory_search", {"query": "x"})
    assert "No matching" in await run(container, ctx, "knowledge_search", {"query": "x"})
    from ai_eos.domain.models import MemoryKind

    await container.memory.remember(admin["id"], MemoryKind.FACT, "Board meets quarterly in Lagos")
    await container.memory.ingest(admin["id"], "Handbook", "Leave requests need two weeks notice.")
    assert "Lagos" in await run(container, ctx, "memory_search", {"query": "board meeting"})
    assert "Handbook" in await run(container, ctx, "knowledge_search", {"query": "leave notice"})


DDG = (
    '<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&rut=1">'
    'Example &amp; Co</a><a class="result__snippet" href="#">The <b>best</b> result</a>'
    '<a class="result__a" href="https://direct.example/b">Second</a>'
)


async def test_web_search_providers(container, admin) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "html.duckduckgo.com":
            return httpx.Response(200, text=DDG)
        assert json.loads(request.content)["api_key"] == "tv"
        return httpx.Response(200, json={"results": [{"title": "T", "url": "https://t", "content": "c"}]})

    ctx = ctx_for(container, admin, handler, env={"TAVILY_API_KEY": "tv"})
    out = await run(container, ctx, "web_search", {"query": "x"})
    assert "Example & Co" in out and "https://example.com/a" in out and "best result" in out and "Second" in out
    container.settings.tools.web_search.provider = "tavily"
    assert "https://t" in await run(container, ctx, "web_search", {"query": "x"})
    ctx.secrets.environ = {}
    assert "TAVILY" in await run(container, ctx, "web_search", {"query": "x"})
    container.settings.tools.web_search.provider = "none"
    assert "disabled" in await run(container, ctx, "web_search", {"query": "x"})
    assert WebSearchTool._parse_ddg("nothing") == []
    container.settings.tools.web_search.provider = "duckduckgo"
    empty = ctx_for(container, admin, lambda r: httpx.Response(200, text="none"))
    assert "No results" in await run(container, empty, "web_search", {"query": "x"})


async def test_fetch_url(container, admin) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/page"})
        if request.url.path == "/page":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html><head><title>x</title><script>evil()</script></head><body><p>Hello</p>"
                "<p>World &amp; more</p></body></html>",
            )
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="plain text")

    ctx = ctx_for(container, admin, handler)
    out = await run(container, ctx, "fetch_url", {"url": "https://site.example/start"})
    assert "Hello" in out and "World & more" in out and "evil" not in out and "URL: https://site.example/page" in out
    assert "plain text" in await run(container, ctx, "fetch_url", {"url": "https://site.example/t.txt"})
    internal = ctx_for(container, admin, handler, public=False)
    assert "internal network" in await run(container, internal, "fetch_url", {"url": "http://intranet/"})
    assert "absolute http" in await run(container, ctx, "fetch_url", {"url": "file:///etc/passwd"})

    def unresolvable(host):  # noqa: ANN001, ANN202
        raise OSError("dns")

    ctx.services["resolve"] = unresolvable
    assert "cannot resolve" in await run(container, ctx, "fetch_url", {"url": "https://nowhere.example"})


def test_html_to_text() -> None:
    assert html_to_text("<style>x{}</style><h1>T</h1>line<br>two") == "T\nline\ntwo"


# ---------------------------------------------------------------- GitHub & Azure DevOps
async def test_github_tools(container, admin) -> None:
    container.executor.auto_approve.add("github_create_issue")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer gh"
        p = request.url.path
        if p == "/repos/o/r/issues" and request.method == "GET":
            assert request.url.params["labels"] == "bug"
            return httpx.Response(
                200,
                json=[
                    {"number": 1, "title": "Bug", "labels": [{"name": "bug"}], "html_url": "u1"},
                    {"number": 2, "title": "PR", "labels": [], "html_url": "u2", "pull_request": {}},
                ],
            )
        if p == "/repos/o/r/issues":
            return httpx.Response(201, json={"number": 3, "html_url": "u3"})
        if p == "/search/code":
            return httpx.Response(200, json={"items": [{"repository": {"full_name": "o/r"}, "path": "a.py"}]})
        if p == "/repos/o/r/contents/src/a.py":
            return httpx.Response(200, json={"content": base64.b64encode(b"print(1)").decode()})
        if p == "/repos/o/r/contents/src":
            return httpx.Response(200, json=[{"type": "file", "path": "src/a.py"}])
        if p == "/repos/o/r/actions/runs":
            return httpx.Response(
                200,
                json={
                    "workflow_runs": [
                        {
                            "name": "CI",
                            "run_number": 9,
                            "head_branch": "main",
                            "status": "completed",
                            "conclusion": "failure",
                            "created_at": "t",
                            "html_url": "u",
                        }
                    ]
                },
            )
        return httpx.Response(404, json={})

    ctx = ctx_for(container, admin, handler, env={"GITHUB_TOKEN": "gh"})
    out = await run(container, ctx, "github_list_issues", {"repo": "o/r", "labels": "bug"})
    assert "#1 [issue] Bug (bug)" in out and "#2 [PR]" in out
    assert "#3" in await run(container, ctx, "github_create_issue", {"repo": "o/r", "title": "New"})
    assert "o/r:a.py" in await run(container, ctx, "github_search_code", {"query": "x"})
    assert "print(1)" in await run(
        container, ctx, "github_get_file", {"repo": "o/r", "path": "src/a.py", "ref": "main"}
    )
    assert "Directory listing" in await run(container, ctx, "github_get_file", {"repo": "o/r", "path": "src"})
    assert "failure" in await run(container, ctx, "github_list_workflow_runs", {"repo": "o/r", "branch": "main"})
    assert "HTTP 404" in await run(container, ctx, "github_list_issues", {"repo": "o/missing"})
    noauth = ctx_for(container, admin, handler)
    assert "not connected" in await run(container, noauth, "github_list_issues", {"repo": "o/r"})
    tool = container.tools.get("github_create_issue")
    assert "Create issue in o/r" in tool.summarize({"repo": "o/r", "title": "t"})


async def test_azure_devops_tools(container, admin) -> None:
    container.executor.auto_approve.add("azdo_create_work_item")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"].startswith("Basic ")
        p = request.url.path
        if p.endswith("/_apis/wit/wiql"):
            q = json.loads(request.content)["query"]
            assert "O''Brien" in q and "[System.WorkItemType] = 'Bug'" in q
            return httpx.Response(200, json={"workItems": [{"id": 7}]})
        if p.endswith("/_apis/wit/workitems"):
            return httpx.Response(
                200,
                json={
                    "value": [
                        {
                            "id": 7,
                            "fields": {
                                "System.Title": "Fix login",
                                "System.State": "Active",
                                "System.WorkItemType": "Bug",
                                "System.AssignedTo": {"displayName": "Ada"},
                            },
                        }
                    ]
                },
            )
        if "/_apis/wit/workitems/$" in p:
            assert request.headers["content-type"] == "application/json-patch+json"
            return httpx.Response(200, json={"id": 8, "_links": {"html": {"href": "https://dev.azure.com/x/8"}}})
        if p.endswith("/_apis/pipelines"):
            return httpx.Response(200, json={"value": [{"id": 1, "name": "deploy", "folder": "\\"}]})
        if p.endswith("/_apis/build/builds"):
            return httpx.Response(
                200,
                json={
                    "value": [
                        {
                            "buildNumber": "20261004.1",
                            "definition": {"name": "ci"},
                            "status": "completed",
                            "result": "succeeded",
                        }
                    ]
                },
            )
        return httpx.Response(404)

    ctx = ctx_for(container, admin, handler, env={"AZDO_ORG": "acme", "AZDO_PAT": "pat"})
    out = await run(container, ctx, "azdo_list_work_items", {"project": "O'Brien", "work_item_type": "Bug"})
    assert "#7 [Bug] Fix login - Active (Ada)" in out
    assert "#8" in await run(
        container,
        ctx,
        "azdo_create_work_item",
        {"project": "P", "work_item_type": "Task", "title": "T", "description": "d"},
    )
    assert "deploy" in await run(container, ctx, "azdo_list_pipelines", {"project": "P"})
    assert "succeeded" in await run(container, ctx, "azdo_list_builds", {"project": "P"})
    empty = ctx_for(
        container,
        admin,
        lambda r: httpx.Response(200, json={"workItems": [], "value": []}),
        env={"AZDO_ORG": "a", "AZDO_PAT": "p"},
    )
    assert "No active" in await run(container, empty, "azdo_list_work_items", {"project": "P"})
    assert "No pipelines" in await run(container, empty, "azdo_list_pipelines", {"project": "P"})
    assert "No builds" in await run(container, empty, "azdo_list_builds", {"project": "P"})
    assert "not connected" in await run(
        container, ctx_for(container, admin, handler), "azdo_list_builds", {"project": "P"}
    )
    assert "Create Task in P" in container.tools.get("azdo_create_work_item").summarize(
        {"project": "P", "work_item_type": "Task", "title": "T"}
    )


# ---------------------------------------------------------------- Google Workspace
def b64(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")


async def test_google_tools(container, admin) -> None:
    container.executor.auto_approve.update({"gmail_send", "calendar_create_event"})
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        p, host = request.url.path, request.url.host
        if host == "oauth2.googleapis.com":
            seen["refresh"] = request.content.decode()
            return httpx.Response(200, json={"access_token": "fresh", "expires_in": 3600})
        assert request.headers["authorization"] == "Bearer fresh"
        if p.endswith("/messages") and request.method == "GET":
            return httpx.Response(200, json={"messages": [{"id": "m1"}]})
        if p.endswith("/messages/m1") and request.url.params.get("format") == "metadata":
            return httpx.Response(
                200,
                json={
                    "threadId": "t1",
                    "labelIds": ["UNREAD"],
                    "snippet": "Hi there",
                    "payload": {
                        "headers": [{"name": "From", "value": "ceo@x.com"}, {"name": "Subject", "value": "Q3"}]
                    },
                },
            )
        if p.endswith("/messages/m1"):
            return httpx.Response(
                200,
                json={
                    "payload": {
                        "headers": [{"name": "Subject", "value": "Q3"}],
                        "mimeType": "multipart/alternative",
                        "parts": [{"mimeType": "text/plain", "body": {"data": b64("Body text")}}],
                    }
                },
            )
        if p.endswith("/drafts"):
            seen["draft"] = json.loads(request.content)
            return httpx.Response(200, json={"id": "d1"})
        if p.endswith("/messages/send"):
            seen["send"] = json.loads(request.content)
            return httpx.Response(200, json={"id": "s1"})
        if p.endswith("/primary/events") and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "summary": "Standup",
                            "start": {"dateTime": "2026-10-05T09:00"},
                            "end": {"dateTime": "2026-10-05T09:15"},
                            "attendees": [{"email": "a@x.com"}],
                        },
                        {"start": {"date": "2026-10-06"}, "end": {"date": "2026-10-07"}},
                    ]
                },
            )
        if p.endswith("/primary/events"):
            seen["event"] = json.loads(request.content)
            assert request.url.params["sendUpdates"] == "all"
            return httpx.Response(200, json={"htmlLink": "https://cal/e1"})
        if p == "/drive/v3/files":
            assert "fullText contains 'O\\'Neil'" in request.url.params["q"]
            return httpx.Response(200, json={"files": [{"name": "Plan.docx", "mimeType": "doc", "webViewLink": "l"}]})
        return httpx.Response(404)

    env = {"GOOGLE_OAUTH_REFRESH_TOKEN": "r", "GOOGLE_OAUTH_CLIENT_ID": "id", "GOOGLE_OAUTH_CLIENT_SECRET": "sec"}
    ctx = ctx_for(container, admin, handler, env=env)
    out = await run(container, ctx, "gmail_search", {"query": "is:unread"})
    assert "UNREAD" in out and "ceo@x.com" in out and "grant_type=refresh_token" in seen["refresh"]
    assert "Body text" in await run(container, ctx, "gmail_read", {"message_id": "m1"})
    assert "Draft saved" in await run(
        container,
        ctx,
        "gmail_draft",
        {"to": "a@x.com", "subject": "S", "body": "B", "cc": "c@x.com", "thread_id": "t1"},
    )
    assert seen["draft"]["message"]["threadId"] == "t1"
    raw = base64.urlsafe_b64decode(seen["draft"]["message"]["raw"]).decode()
    assert "To: a@x.com" in raw and "Cc: c@x.com" in raw
    assert "Email sent" in await run(
        container, ctx, "gmail_send", {"to": "a@x.com", "subject": "S", "body": "B", "thread_id": "t1"}
    )
    cal = await run(container, ctx, "calendar_list_events", {})
    assert "Standup" in cal and "(no title)" in cal
    assert "https://cal/e1" in await run(
        container,
        ctx,
        "calendar_create_event",
        {"summary": "Review", "start": "2026-10-06T10:00:00", "end": "2026-10-06T10:30:00", "attendees": ["b@x.com"]},
    )
    assert seen["event"]["start"]["timeZone"] == "Africa/Lagos"
    assert "Plan.docx" in await run(container, ctx, "drive_search", {"query": "O'Neil"})
    # cached access token is reused (no second refresh)
    seen.pop("refresh")
    await run(container, ctx, "gmail_search", {"query": "x"})
    assert "refresh" not in seen

    await container.kv.delete("google:access_token")
    static = ctx_for(container, admin, handler, env={"GOOGLE_OAUTH_ACCESS_TOKEN": "fresh"})
    assert "Plan.docx" in await run(container, static, "drive_search", {"query": "O'Neil"})
    none = ctx_for(container, admin, handler)
    assert "not connected" in await run(container, none, "drive_search", {"query": "x"})
    bad = ctx_for(container, admin, lambda r: httpx.Response(400, text="invalid_grant"), env=env)
    assert "refresh failed" in await run(container, bad, "drive_search", {"query": "x"})
    send = container.tools.get("gmail_send").summarize({"to": "a", "subject": "s", "body": "b"})
    ev = container.tools.get("calendar_create_event").summarize({"summary": "s", "start": "1", "end": "2"})
    assert "Send email to a" in send and "no attendees" in ev


def test_plain_body_html_fallback() -> None:
    payload = {"mimeType": "text/html", "body": {"data": b64("<p>Hi <b>there</b></p>")}}
    assert _plain_body(payload) == "Hi there"
    assert _plain_body({"mimeType": "image/png"}) == ""
