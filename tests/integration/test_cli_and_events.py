"""CLI commands against a live in-process API, plus the server-sent events stream."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import ai_eos.cli as cli
from ai_eos.domain.events import Event
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD
from tests.integration.test_api import client, llm, login  # noqa: F401 - fixtures

runner = CliRunner()


@pytest.fixture
def cli_env(client, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(cli, "CRED_FILE", tmp_path / "creds.json")
    monkeypatch.delenv("EOS_API_KEY", raising=False)

    def fake_client():  # noqa: ANN202
        creds = cli._load_creds()
        client.headers.clear()
        if creds.get("token"):
            client.headers["Authorization"] = f"Bearer {creds['token']}"
        return _NoClose(client)

    monkeypatch.setattr(cli, "_client", fake_client)
    return client


class _NoClose:
    """Wrap TestClient so `with _client() as c` doesn't shut the app down."""

    def __init__(self, inner) -> None:  # noqa: ANN001
        self.inner, self.base_url = inner, inner.base_url

    def __enter__(self):  # noqa: ANN204
        return self.inner

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        return None


def invoke(*args: str, input: str | None = None):  # noqa: A002, ANN201
    result = runner.invoke(cli.app, list(args), input=input)
    return result


def test_cli_end_to_end(cli_env, llm, tmp_path: Path) -> None:  # noqa: F811
    assert invoke("tasks").exit_code == 1  # not logged in
    r = invoke("login", "--email", ADMIN_EMAIL, "--password", ADMIN_PASSWORD)
    assert r.exit_code == 0 and "Signed in" in r.output
    assert (tmp_path / "creds.json").stat().st_mode & 0o777 == 0o600

    r = invoke("chat", "Research solar inverter suppliers in Lagos")
    assert r.exit_code == 0 and "thorough answer" in r.output and "completed" in r.output
    r = invoke("chat", input="Analyse our Q3 numbers please\nnew\n\nexit\n")
    assert r.exit_code == 0 and r.output.count("thorough answer") == 1

    r = invoke("tasks")
    assert "Research solar" in r.output
    task_id = cli_env.get("/api/tasks").json()[0]["id"]
    assert "Final answer" in invoke("task", task_id[:8]).output
    assert invoke("task", "zzzz").exit_code == 1

    assert "Nothing waiting" in invoke("approvals").output
    llm.script["plan"] = json.dumps({"subtasks": [{"id": "t", "agent": "software_engineer", "instruction": "x"}]})
    llm.script["agent"] = [
        json.dumps({"action": "tool", "tool": "write_file", "args": {"path": "a.txt", "content": "A"}}),
        json.dumps({"action": "final", "content": "queued"}),
        json.dumps({"action": "tool", "tool": "write_file", "args": {"path": "b.txt", "content": "B"}}),
        json.dumps({"action": "final", "content": "queued"}),
    ]
    invoke("chat", "save file a")
    invoke("chat", "save file b")
    pending = cli_env.get("/api/approvals?status=pending").json()
    assert "write_file" in invoke("approvals").output
    assert "executed" in invoke("approve", pending[0]["id"]).output
    assert "rejected" in invoke("reject", pending[1]["id"]).output
    assert "executed" in invoke("approvals", "--all").output

    llm.script.pop("agent")
    assert "operations" in invoke("agents").output
    assert "thorough answer" in invoke("run-agent", "operations", "List KPIs").output

    doc = tmp_path / "handbook.md"
    doc.write_text("# Handbook\n\nWorking hours are 8am to 5pm.")
    assert "Indexed" in invoke("ingest", str(doc), "--title", "Handbook").output

    assert "Saved" in invoke("memory", "add", "CFO is Ngozi", "--kind", "fact").output
    assert "CFO is Ngozi" in invoke("memory", "list").output
    assert "CFO" in invoke("memory", "search", "who is the CFO").output
    mid = cli_env.get("/api/memory").json()[0]["id"]
    assert "Deleted" in invoke("memory", "forget", mid).output

    r = invoke("health")
    assert r.exit_code == 0 and '"database": true' in r.output
    assert "Error 404" in invoke("run-agent", "ghost", "x").output


def test_cli_offline_commands(monkeypatch, tmp_path: Path) -> None:
    r = invoke("gen-secrets")
    assert "EOS_JWT_SECRET=" in r.output and "EOS_ENCRYPTION_KEY=" in r.output

    from tests.conftest import make_settings

    settings = make_settings(tmp_path)
    monkeypatch.setattr("ai_eos.config.get_settings", lambda: settings)
    assert "needs" in invoke("create-user", "--email", "o@x.com", "--password", "weak", "--role", "operator").output
    r = invoke("create-user", "--email", "o@x.com", "--password", "OperatorPass1", "--role", "operator")
    assert r.exit_code == 0 and "Created operator" in r.output
    r = invoke("create-user", "--email", "o@x.com", "--password", "OperatorPass1")
    assert r.exit_code == 1


def test_cli_unreachable_server(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "CRED_FILE", tmp_path / "c.json")
    monkeypatch.setenv("EOS_SERVER", "http://127.0.0.1:9")
    r = invoke("tasks")
    assert r.exit_code == 2 and "Cannot reach" in r.output


async def test_sse_stream_filters_by_user(container, admin) -> None:
    from ai_eos.api.routes import events

    await container.bus.publish(Event(type="task.created", user_id=admin["id"], data={"x": 1}))
    await container.bus.publish(Event(type="task.created", user_id="other", data={"x": 2}))

    class FakeRequest:
        def __init__(self) -> None:
            self.calls = 0

        async def is_disconnected(self) -> bool:
            self.calls += 1
            if self.calls == 1:
                await container.bus.publish(Event(type="approval.requested", user_id=admin["id"], data={"y": 1}))
                await container.bus.publish(Event(type="approval.requested", user_id="other", data={"y": 2}))
            return self.calls > 3

    import ai_eos.api.routes as routes

    original_wait = routes.asyncio.wait_for

    async def quick_wait(coro, timeout):  # noqa: ANN001, ANN202
        return await original_wait(coro, 0.05)

    routes.asyncio.wait_for = quick_wait
    try:
        resp = await events(FakeRequest(), admin, container)
        chunks = [c async for c in resp.body_iterator]
    finally:
        routes.asyncio.wait_for = original_wait
    text = "".join(chunks)
    assert '"x": 1' in text.replace('"x":1', '"x": 1') and '"x":2' not in text and '"x": 2' not in text
    assert "approval.requested" in text and '"y": 2' not in text
    assert ": keep-alive" in text
