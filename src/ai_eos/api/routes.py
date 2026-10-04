"""REST API routes (all under /api). OpenAPI docs are served at /docs."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr, Field

from ai_eos.api.deps import User, client_ip, current_user, get_container, require
from ai_eos.container import Container
from ai_eos.domain.models import MemoryKind, Role
from ai_eos.memory.manager import extract_text
from ai_eos.security.auth import generate_api_key, hash_password, password_problems, verify_password

router = APIRouter(prefix="/api")
MAX_UPLOAD = 25 * 1024 * 1024


# ---------------------------------------------------------------- schemas
class LoginIn(BaseModel):
    email: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105
    expires_in: int
    user: dict[str, Any]


class UserIn(BaseModel):
    email: EmailStr
    password: str
    role: Role = Role.OPERATOR


class UserPatch(BaseModel):
    role: Role | None = None
    is_active: bool | None = None
    password: str | None = None


class ChatIn(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None
    provider: str | None = None


class ApprovalDecision(BaseModel):
    approve: bool


class AgentRunIn(BaseModel):
    instruction: str = Field(min_length=1)


class MemoryIn(BaseModel):
    kind: Literal["fact", "preference", "reflection"] = "fact"
    content: str = Field(min_length=1, max_length=2000)


class KnowledgeTextIn(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1)


class PromptIn(BaseModel):
    content: str = Field(min_length=10)


class SecretIn(BaseModel):
    value: str = Field(min_length=1)


class UserSettingsIn(BaseModel):
    provider: str | None = None
    notifications: bool | None = None


class ApiKeyIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)


def _validate_password(pw: str) -> None:
    problems = password_problems(pw)
    if problems:
        raise HTTPException(422, f"password needs {', '.join(problems)}")


# ---------------------------------------------------------------- auth
@router.post("/auth/login", response_model=TokenOut, tags=["auth"])
async def login(body: LoginIn, request: Request, c: Container = Depends(get_container)) -> TokenOut:
    ok, _ = await c.rate_limiter.allow(f"login:{client_ip(request)}:{body.email.lower()}")
    if not ok:
        raise HTTPException(429, "too many login attempts; wait a minute")
    found = await c.users.get_by_email(body.email)
    if not found or not verify_password(body.password, found[1]) or not found[0]["is_active"]:
        await c.audit.log("auth.login_failed", None, body.email.lower(), ip=client_ip(request))
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")
    user = found[0]
    token, ttl = c.tokens.issue(user["id"], user["role"])
    await c.audit.log("auth.login", user["id"], user["email"], ip=client_ip(request))
    return TokenOut(access_token=token, expires_in=ttl, user=user)


@router.get("/auth/me", tags=["auth"])
async def me(user: User = Depends(current_user)) -> User:
    return user


@router.post("/auth/api-keys", tags=["auth"], status_code=201)
async def create_api_key(
    body: ApiKeyIn, user: User = Depends(require("chat")), c: Container = Depends(get_container)
) -> dict[str, Any]:
    full, prefix, digest = generate_api_key()
    info = await c.users.add_api_key(user["id"], body.name, prefix, digest)
    await c.audit.log("auth.api_key_created", user["id"], info["id"])
    return {**info, "key": full, "note": "Store this key now; it cannot be shown again."}


@router.get("/auth/api-keys", tags=["auth"])
async def list_api_keys(user: User = Depends(current_user), c: Container = Depends(get_container)) -> list[dict]:
    return await c.users.list_api_keys(user["id"])


@router.delete("/auth/api-keys/{key_id}", tags=["auth"], status_code=204)
async def revoke_api_key(
    key_id: str, user: User = Depends(current_user), c: Container = Depends(get_container)
) -> None:
    if not await c.users.revoke_api_key(user["id"], key_id):
        raise HTTPException(404, "key not found")
    await c.audit.log("auth.api_key_revoked", user["id"], key_id)


# ---------------------------------------------------------------- users (admin)
@router.get("/users", tags=["users"])
async def list_users(_: User = Depends(require("users:manage")), c: Container = Depends(get_container)) -> list[dict]:
    return await c.users.list()


@router.post("/users", tags=["users"], status_code=201)
async def create_user(
    body: UserIn, admin: User = Depends(require("users:manage")), c: Container = Depends(get_container)
) -> User:
    _validate_password(body.password)
    if await c.users.get_by_email(body.email):
        raise HTTPException(409, "email already registered")
    user = await c.users.create(body.email, hash_password(body.password), body.role)
    await c.audit.log("user.created", admin["id"], user["id"], {"role": body.role})
    return user


@router.patch("/users/{user_id}", tags=["users"])
async def update_user(
    user_id: str, body: UserPatch, admin: User = Depends(require("users:manage")), c: Container = Depends(get_container)
) -> User:
    fields: dict[str, Any] = {}
    if body.role is not None:
        fields["role"] = body.role.value
    if body.is_active is not None:
        fields["is_active"] = body.is_active
    if body.password:
        _validate_password(body.password)
        fields["password_hash"] = hash_password(body.password)
    user = await c.users.update(user_id, **fields)
    if not user:
        raise HTTPException(404, "user not found")
    await c.audit.log("user.updated", admin["id"], user_id, {k: v for k, v in fields.items() if k != "password_hash"})
    return user


# ---------------------------------------------------------------- chat & tasks
@router.post("/chat", tags=["chat"])
async def chat(body: ChatIn, user: User = Depends(require("chat")), c: Container = Depends(get_container)) -> dict:
    provider = body.provider or user.get("settings", {}).get("provider")
    if provider and provider not in c.settings.llm.providers:
        raise HTTPException(422, f"unknown provider '{provider}'")
    assert c.chief is not None
    outcome, conv_id = await c.chief.handle(user["id"], body.message, body.conversation_id, provider)
    return {"conversation_id": conv_id, **outcome.model_dump(mode="json")}


@router.get("/conversations", tags=["chat"])
async def conversations(user: User = Depends(require("tasks:read")), c: Container = Depends(get_container)) -> list:
    return await c.conversations.list(user["id"])


@router.get("/conversations/{conversation_id}", tags=["chat"])
async def conversation(
    conversation_id: str, user: User = Depends(require("tasks:read")), c: Container = Depends(get_container)
) -> list:
    msgs = await c.conversations.messages(user["id"], conversation_id)
    if msgs is None:
        raise HTTPException(404, "conversation not found")
    return msgs


@router.get("/tasks", tags=["tasks"])
async def list_tasks(
    status_: str | None = Query(None, alias="status"),
    user: User = Depends(require("tasks:read")),
    c: Container = Depends(get_container),
) -> list:
    return await c.tasks.list(user["id"], status_)


@router.get("/tasks/{task_id}", tags=["tasks"])
async def get_task(
    task_id: str, user: User = Depends(require("tasks:read")), c: Container = Depends(get_container)
) -> dict:
    task = await c.tasks.get(task_id, user["id"])
    if not task:
        raise HTTPException(404, "task not found")
    return task


# ---------------------------------------------------------------- approvals
@router.get("/approvals", tags=["approvals"])
async def list_approvals(
    status_: str | None = Query(None, alias="status"),
    user: User = Depends(require("tasks:read")),
    c: Container = Depends(get_container),
) -> list:
    return await c.approvals.list(user["id"], status_)


@router.post("/approvals/{approval_id}", tags=["approvals"])
async def decide_approval(
    approval_id: str,
    body: ApprovalDecision,
    user: User = Depends(require("approvals:decide")),
    c: Container = Depends(get_container),
) -> dict:
    try:
        return await c.executor.decide(approval_id, user["id"], body.approve, c.tool_context(user["id"], "", None))
    except KeyError:
        raise HTTPException(404, "approval not found") from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None


# ---------------------------------------------------------------- agents
@router.get("/agents", tags=["agents"])
async def list_agents(_: User = Depends(require("tasks:read")), c: Container = Depends(get_container)) -> list:
    cos = c.settings.agents.chief_of_staff
    return [
        {
            "key": "chief_of_staff",
            "title": cos.title,
            "description": "Orchestrates the team",
            "tools": [],
            "provider": cos.provider or c.settings.llm.default_provider,
            "model": cos.model,
        }
    ] + [a.describe() for a in c.agents.values()]


@router.post("/agents/{agent_key}/run", tags=["agents"])
async def run_agent(
    agent_key: str,
    body: AgentRunIn,
    user: User = Depends(require("agents:direct")),
    c: Container = Depends(get_container),
) -> dict:
    if agent_key not in c.agents:
        raise HTTPException(404, "unknown agent")
    assert c.chief is not None
    result = await c.chief.run_agent(user["id"], agent_key, body.instruction, user.get("settings", {}).get("provider"))
    return result.model_dump(mode="json")


@router.get("/tools", tags=["agents"])
async def list_tools(_: User = Depends(require("tasks:read")), c: Container = Depends(get_container)) -> list:
    return [
        {
            "name": s.name,
            "description": s.description,
            "requires_approval": c.executor.needs_approval(s),
            "parameters": s.parameters,
        }
        for s in c.tools.specs()
    ]


# ---------------------------------------------------------------- memory & knowledge
@router.get("/memory", tags=["memory"])
async def list_memory(
    kind: MemoryKind | None = None, user: User = Depends(require("memory:read")), c: Container = Depends(get_container)
) -> list:
    return [m.model_dump(mode="json") for m in await c.memory.list(user["id"], kind)]


@router.get("/memory/search", tags=["memory"])
async def search_memory(
    q: str, user: User = Depends(require("memory:read")), c: Container = Depends(get_container)
) -> list:
    return [m.model_dump(mode="json") for m in await c.memory.recall(user["id"], q)]


@router.post("/memory", tags=["memory"], status_code=201)
async def add_memory(
    body: MemoryIn, user: User = Depends(require("memory:write")), c: Container = Depends(get_container)
) -> dict:
    item = await c.memory.remember(user["id"], MemoryKind(body.kind), body.content, {"source": "manual"})
    return item.model_dump(mode="json")


@router.delete("/memory/{memory_id}", tags=["memory"], status_code=204)
async def delete_memory(
    memory_id: str, user: User = Depends(require("memory:write")), c: Container = Depends(get_container)
) -> None:
    if not await c.memory.forget(user["id"], memory_id):
        raise HTTPException(404, "memory not found")
    await c.audit.log("memory.deleted", user["id"], memory_id)


@router.get("/knowledge", tags=["knowledge"])
async def list_docs(user: User = Depends(require("memory:read")), c: Container = Depends(get_container)) -> list:
    return await c.memory.list_docs(user["id"])


@router.get("/knowledge/search", tags=["knowledge"])
async def search_docs(
    q: str, user: User = Depends(require("memory:read")), c: Container = Depends(get_container)
) -> list:
    return await c.memory.search_knowledge(user["id"], q)


@router.post("/knowledge", tags=["knowledge"], status_code=201)
async def upload_doc(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    user: User = Depends(require("knowledge:write")),
    c: Container = Depends(get_container),
) -> dict:
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "file larger than 25 MB")
    try:
        text = extract_text(file.filename or "upload.txt", data)
        doc = await c.memory.ingest(user["id"], title or file.filename or "document", text, file.filename or "")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    await c.audit.log("knowledge.ingested", user["id"], doc["id"], {"title": doc["title"], "chunks": doc["chunks"]})
    return doc


@router.post("/knowledge/text", tags=["knowledge"], status_code=201)
async def add_text_doc(
    body: KnowledgeTextIn, user: User = Depends(require("knowledge:write")), c: Container = Depends(get_container)
) -> dict:
    doc = await c.memory.ingest(user["id"], body.title, body.text, "text")
    await c.audit.log("knowledge.ingested", user["id"], doc["id"], {"title": doc["title"]})
    return doc


@router.delete("/knowledge/{doc_id}", tags=["knowledge"], status_code=204)
async def delete_doc(
    doc_id: str, user: User = Depends(require("knowledge:write")), c: Container = Depends(get_container)
) -> None:
    if not await c.memory.delete_doc(user["id"], doc_id):
        raise HTTPException(404, "document not found")


# ---------------------------------------------------------------- prompts (admin)
@router.get("/prompts", tags=["prompts"])
async def list_prompts(_: User = Depends(require("tasks:read")), c: Container = Depends(get_container)) -> list:
    return [{"name": n, "overridden": c.prompts.is_overridden(n)} for n in c.prompts.names()]


@router.get("/prompts/{name}", tags=["prompts"])
async def get_prompt(
    name: str, _: User = Depends(require("tasks:read")), c: Container = Depends(get_container)
) -> dict:
    try:
        return {
            "name": name,
            "content": c.prompts.get(name),
            "default": c.prompts.default(name),
            "overridden": c.prompts.is_overridden(name),
        }
    except KeyError:
        raise HTTPException(404, "prompt not found") from None


@router.put("/prompts/{name}", tags=["prompts"])
async def set_prompt(
    name: str, body: PromptIn, user: User = Depends(require("prompts:write")), c: Container = Depends(get_container)
) -> dict:
    try:
        await c.prompts.set(name, body.content, user["id"])
    except KeyError:
        raise HTTPException(404, "prompt not found") from None
    await c.audit.log("prompt.updated", user["id"], name)
    return {"name": name, "overridden": True}


@router.delete("/prompts/{name}", tags=["prompts"], status_code=204)
async def reset_prompt(
    name: str, user: User = Depends(require("prompts:write")), c: Container = Depends(get_container)
) -> None:
    await c.prompts.reset(name)
    await c.audit.log("prompt.reset", user["id"], name)


# ---------------------------------------------------------------- settings, secrets, audit, metrics
@router.get("/settings/models", tags=["settings"])
async def models(user: User = Depends(current_user), c: Container = Depends(get_container)) -> dict:
    return {
        "default": c.settings.llm.default_provider,
        "selected": user.get("settings", {}).get("provider"),
        "providers": c.llm.available(),
    }


@router.put("/settings/me", tags=["settings"])
async def update_my_settings(
    body: UserSettingsIn, user: User = Depends(current_user), c: Container = Depends(get_container)
) -> dict:
    settings = dict(user.get("settings") or {})
    if body.provider is not None:
        if body.provider and body.provider not in c.settings.llm.providers:
            raise HTTPException(422, "unknown provider")
        settings["provider"] = body.provider or None
    if body.notifications is not None:
        settings["notifications"] = body.notifications
    updated = await c.users.update(user["id"], settings=settings)
    return (updated or {}).get("settings", {})


@router.get("/secrets", tags=["settings"])
async def list_secrets(_: User = Depends(require("secrets:write")), c: Container = Depends(get_container)) -> list:
    return await c.secret_repo.names()


@router.put("/secrets/{name}", tags=["settings"], status_code=204)
async def put_secret(
    name: str, body: SecretIn, user: User = Depends(require("secrets:write")), c: Container = Depends(get_container)
) -> None:
    if not name.replace("_", "").isalnum() or not name.isupper():
        raise HTTPException(422, "secret names must be UPPER_SNAKE_CASE, e.g. GITHUB_TOKEN")
    await c.secret_repo.set(name, c.cipher.encrypt(body.value))
    await c.kv.delete("google:access_token")
    await c.audit.log("secret.set", user["id"], name)


@router.delete("/secrets/{name}", tags=["settings"], status_code=204)
async def delete_secret(
    name: str, user: User = Depends(require("secrets:write")), c: Container = Depends(get_container)
) -> None:
    if not await c.secret_repo.delete(name):
        raise HTTPException(404, "secret not found")
    await c.audit.log("secret.deleted", user["id"], name)


@router.get("/audit", tags=["admin"])
async def audit_log(
    action: str | None = None, _: User = Depends(require("audit:read")), c: Container = Depends(get_container)
) -> list:
    return await c.audit.list(action=action)


@router.get("/metrics/summary", tags=["admin"])
async def metrics_summary(_: User = Depends(require("metrics:read")), c: Container = Depends(get_container)) -> dict:
    data = await c.tasks.metrics()
    data["live_subscribers"] = c.bus.subscriber_count
    return data


@router.get("/events", tags=["events"])
async def events(
    request: Request, user: User = Depends(current_user), c: Container = Depends(get_container)
) -> StreamingResponse:
    """Server-sent events: live task progress and approval notifications for this user."""

    async def stream():  # noqa: ANN202
        for e in c.bus.recent(user["id"], 20):
            yield f"event: {e.type}\ndata: {e.model_dump_json()}\n\n"
        async with c.bus.subscribe() as queue:
            while not await request.is_disconnected():
                try:
                    e = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if e.user_id in (None, user["id"]):
                    yield f"event: {e.type}\ndata: {json.dumps(e.model_dump(mode='json'))}\n\n"

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
