"""End-to-end tests through the HTTP API (FastAPI TestClient, real DB, real orchestration, scripted LLM)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from ai_eos.api.app import create_app
from ai_eos.container import build_container
from ai_eos.infrastructure.vector import HashingEmbedder, VectorStore
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, ScriptedLLM, make_settings


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def client(tmp_path, llm):
    settings = make_settings(
        tmp_path, EOS_BOOTSTRAP_ADMIN_EMAIL=ADMIN_EMAIL, EOS_BOOTSTRAP_ADMIN_PASSWORD=ADMIN_PASSWORD
    )
    c = build_container(settings, vectors=VectorStore(QdrantClient(location=":memory:"), HashingEmbedder(384)))
    c.llm.register("offline", llm)
    import asyncio

    from tests.conftest import reset_external_state

    asyncio.run(reset_external_state(c))
    asyncio.run(c.db.engine.dispose())  # don't carry pooled connections across event loops
    app = create_app(container=c)
    with TestClient(app) as tc:
        tc.container = c  # type: ignore[attr-defined]
        yield tc


def login(client, email=ADMIN_EMAIL, password=ADMIN_PASSWORD) -> dict[str, str]:
    r = client.post("/api/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_health_dashboard_metrics(client) -> None:
    assert client.get("/health/live").json()["status"] == "ok"
    assert client.get("/health/ready").json()["checks"] == {"database": True, "cache": True, "vector": True}
    page = client.get("/")
    assert page.status_code == 200 and "Chief of Staff" in page.text and "Content-Security-Policy" in page.headers
    assert "eos_http_requests_total" in client.get("/metrics").text
    r = client.get("/health/live", headers={"X-Request-ID": "abc"})
    assert r.headers["X-Request-ID"] == "abc" and r.headers["X-Frame-Options"] == "DENY"
    assert client.get("/docs").status_code == 200


def test_auth_flows(client) -> None:
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer junk"}).status_code == 401
    assert client.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": "wrong"}).status_code == 401
    h = login(client)
    me = client.get("/api/auth/me", headers=h).json()
    assert me["email"] == ADMIN_EMAIL and me["role"] == "admin"

    key = client.post("/api/auth/api-keys", json={"name": "cli"}, headers=h).json()
    assert key["key"].startswith("eos_")
    assert client.get("/api/auth/me", headers={"X-API-Key": key["key"]}).status_code == 200
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {key['key']}"}).status_code == 200
    assert client.get("/api/auth/me", headers={"X-API-Key": key["key"] + "x"}).status_code == 401
    keys = client.get("/api/auth/api-keys", headers=h).json()
    assert keys[0]["last_used_at"]
    assert client.delete(f"/api/auth/api-keys/{key['id']}", headers=h).status_code == 204
    assert client.delete(f"/api/auth/api-keys/{key['id']}x", headers=h).status_code == 404
    assert client.get("/api/auth/me", headers={"X-API-Key": key["key"]}).status_code == 401


def test_login_rate_limit(client) -> None:
    client.container.rate_limiter.per_minute = 3
    codes = [client.post("/api/auth/login", json={"email": "x@y.com", "password": "p"}).status_code for _ in range(4)]
    assert codes == [401, 401, 401, 429]


def test_global_rate_limit(client) -> None:
    h = login(client)
    client.container.rate_limiter.per_minute = 2
    codes = [client.get("/api/tasks", headers=h).status_code for _ in range(3)]
    assert codes[-1] == 429


def test_user_management_and_rbac(client) -> None:
    h = login(client)
    assert (
        client.post(
            "/api/users", json={"email": "v@x.com", "password": "weak", "role": "viewer"}, headers=h
        ).status_code
        == 422
    )
    v = client.post("/api/users", json={"email": "v@x.com", "password": "ViewerPass123", "role": "viewer"}, headers=h)
    assert v.status_code == 201
    assert (
        client.post("/api/users", json={"email": "v@x.com", "password": "ViewerPass123"}, headers=h).status_code == 409
    )
    vh = login(client, "v@x.com", "ViewerPass123")
    assert client.post("/api/chat", json={"message": "hi there team"}, headers=vh).status_code == 403
    assert client.get("/api/users", headers=vh).status_code == 403
    assert client.get("/api/tasks", headers=vh).status_code == 200
    uid = v.json()["id"]
    assert (
        client.patch(f"/api/users/{uid}", json={"role": "operator", "password": "weak"}, headers=h).status_code == 422
    )
    r = client.patch(f"/api/users/{uid}", json={"role": "operator", "password": "NewViewerPass1"}, headers=h)
    assert r.json()["role"] == "operator"
    assert client.patch(f"/api/users/{uid}", json={"is_active": False}, headers=h).json()["is_active"] is False
    assert client.get("/api/auth/me", headers=vh).status_code == 401  # disabled
    assert client.patch("/api/users/nope", json={"is_active": True}, headers=h).status_code == 404
    assert len(client.get("/api/users", headers=h).json()) == 2


def test_chat_tasks_conversations(client, llm) -> None:
    h = login(client)
    r = client.post("/api/chat", json={"message": "Research the Lagos real estate market"}, headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed" and body["answer"] == "A thorough answer with specifics."
    conv = body["conversation_id"]
    r2 = client.post("/api/chat", json={"message": "And Abuja?", "conversation_id": conv}, headers=h).json()
    assert r2["conversation_id"] == conv
    assert client.post("/api/chat", json={"message": "x", "provider": "nope"}, headers=h).status_code == 422
    assert len(client.get(f"/api/conversations/{conv}", headers=h).json()) == 4
    assert client.get("/api/conversations/missing", headers=h).status_code == 404
    assert client.get("/api/conversations", headers=h).json()[0]["id"] == conv
    tasks = client.get("/api/tasks", headers=h).json()
    assert len(tasks) == 2
    assert client.get("/api/tasks?status=completed", headers=h).json()
    detail = client.get(f"/api/tasks/{body['task_id']}", headers=h).json()
    assert detail["subtasks"][0]["agent"] == "research_analytics"
    assert client.get("/api/tasks/nope", headers=h).status_code == 404
    summary = client.get("/api/metrics/summary", headers=h).json()
    assert summary["tasks_by_status"]["completed"] == 2


def test_approvals_api(client, llm) -> None:
    h = login(client)
    llm.script["plan"] = json.dumps({"subtasks": [{"id": "t", "agent": "software_engineer", "instruction": "save"}]})
    llm.script["agent"] = [
        json.dumps({"action": "tool", "tool": "write_file", "args": {"path": "notes.md", "content": "# Notes"}}),
        json.dumps({"action": "final", "content": "Saved notes (pending approval)."}),
    ]
    r = client.post("/api/chat", json={"message": "Save my notes to a file"}, headers=h).json()
    assert r["status"] == "awaiting_approval"
    pending = client.get("/api/approvals?status=pending", headers=h).json()
    aid = pending[0]["id"]
    assert pending[0]["tool"] == "write_file" and "notes.md" in pending[0]["summary"]
    ok = client.post(f"/api/approvals/{aid}", json={"approve": True}, headers=h).json()
    assert ok["status"] == "executed"
    assert client.post(f"/api/approvals/{aid}", json={"approve": True}, headers=h).status_code == 409
    assert client.post("/api/approvals/nope", json={"approve": False}, headers=h).status_code == 404
    assert client.get("/api/approvals", headers=h).json()[0]["status"] == "executed"


def test_agents_and_tools(client) -> None:
    h = login(client)
    agents = client.get("/api/agents", headers=h).json()
    assert agents[0]["key"] == "chief_of_staff" and len(agents) == 6
    r = client.post("/api/agents/operations/run", json={"instruction": "Draft a KPI list"}, headers=h).json()
    assert r["ok"] and r["agent"] == "operations"
    assert client.post("/api/agents/ghost/run", json={"instruction": "x"}, headers=h).status_code == 404
    tools = {t["name"]: t for t in client.get("/api/tools", headers=h).json()}
    assert tools["gmail_send"]["requires_approval"] and not tools["web_search"]["requires_approval"]


def test_memory_and_knowledge(client) -> None:
    h = login(client)
    m = client.post("/api/memory", json={"kind": "preference", "content": "Weekly report on Fridays"}, headers=h).json()
    assert client.get("/api/memory", headers=h).json()[0]["id"] == m["id"]
    assert client.get("/api/memory?kind=fact", headers=h).json() == []
    assert client.get("/api/memory/search?q=weekly report", headers=h).json()[0]["content"].startswith("Weekly")
    assert client.delete(f"/api/memory/{m['id']}", headers=h).status_code == 204
    assert client.delete(f"/api/memory/{m['id']}", headers=h).status_code == 404

    files = {"file": ("policy.md", b"# Expense policy\n\nReceipts are required above 50,000 naira.", "text/markdown")}
    doc = client.post("/api/knowledge", files=files, data={"title": "Expenses"}, headers=h).json()
    assert doc["title"] == "Expenses" and doc["chunks"] == 1
    bad = client.post("/api/knowledge", files={"file": ("x.exe", b"MZ", "application/octet-stream")}, headers=h)
    assert bad.status_code == 422
    t = client.post("/api/knowledge/text", json={"title": "Note", "text": "Office closes at 5pm"}, headers=h)
    assert t.status_code == 201
    assert client.get("/api/knowledge/search?q=receipts expense", headers=h).json()[0]["title"] == "Expenses"
    assert len(client.get("/api/knowledge", headers=h).json()) == 2
    assert client.delete(f"/api/knowledge/{doc['id']}", headers=h).status_code == 204
    assert client.delete(f"/api/knowledge/{doc['id']}", headers=h).status_code == 404


def test_upload_size_limit(client, monkeypatch) -> None:
    import ai_eos.api.routes as routes

    monkeypatch.setattr(routes, "MAX_UPLOAD", 10)
    h = login(client)
    r = client.post("/api/knowledge", files={"file": ("big.txt", b"x" * 50, "text/plain")}, headers=h)
    assert r.status_code == 413


def test_prompts_settings_secrets_audit(client) -> None:
    h = login(client)
    names = {p["name"] for p in client.get("/api/prompts", headers=h).json()}
    assert {"chief_of_staff", "operations"} <= names
    assert client.get("/api/prompts/nope", headers=h).status_code == 404
    assert (
        client.put("/api/prompts/operations", json={"content": "# New operations prompt"}, headers=h).status_code == 200
    )
    p = client.get("/api/prompts/operations", headers=h).json()
    assert p["overridden"] and p["content"].startswith("# New")
    assert client.put("/api/prompts/nope", json={"content": "0123456789x"}, headers=h).status_code == 404
    assert client.delete("/api/prompts/operations", headers=h).status_code == 204
    assert not client.get("/api/prompts/operations", headers=h).json()["overridden"]

    models = client.get("/api/settings/models", headers=h).json()
    assert models["default"] == "offline" and any(m["name"] == "anthropic" for m in models["providers"])
    assert client.put("/api/settings/me", json={"provider": "nope"}, headers=h).status_code == 422
    s = client.put("/api/settings/me", json={"provider": "offline", "notifications": True}, headers=h).json()
    assert s == {"provider": "offline", "notifications": True}
    assert client.put("/api/settings/me", json={"provider": ""}, headers=h).json()["provider"] is None

    assert client.put("/api/secrets/bad-name", json={"value": "x"}, headers=h).status_code == 422
    assert client.put("/api/secrets/GITHUB_TOKEN", json={"value": "ghp_x"}, headers=h).status_code == 204
    assert client.get("/api/secrets", headers=h).json()[0]["name"] == "GITHUB_TOKEN"
    stored = client.container  # value is encrypted at rest; read it on the app's own event loop
    cipher_text = client.portal.call(stored.secret_repo.get, "GITHUB_TOKEN")
    assert "ghp_x" not in cipher_text and stored.cipher.decrypt(cipher_text) == "ghp_x"
    assert client.delete("/api/secrets/GITHUB_TOKEN", headers=h).status_code == 204
    assert client.delete("/api/secrets/GITHUB_TOKEN", headers=h).status_code == 404

    actions = [a["action"] for a in client.get("/api/audit", headers=h).json()]
    assert "secret.set" in actions and "prompt.updated" in actions and "auth.login" in actions
    assert all(a["action"].startswith("auth") for a in client.get("/api/audit?action=auth", headers=h).json())


def test_events_require_auth(client) -> None:
    assert client.get("/api/events").status_code == 401
    assert client.get("/api/events?token=bad").status_code == 401
