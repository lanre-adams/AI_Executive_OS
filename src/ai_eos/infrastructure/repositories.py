"""Repositories: the only place SQL is written. Each returns plain dicts or domain objects."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import delete, desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ai_eos.domain.models import (
    AgentResult,
    ApprovalStatus,
    MemoryItem,
    MemoryKind,
    Plan,
    TaskStatus,
    new_id,
    utcnow,
)
from ai_eos.infrastructure.db import (
    ApiKeyRow,
    ApprovalRow,
    AuditRow,
    ConversationRow,
    KnowledgeDocRow,
    MemoryRow,
    MessageRow,
    PromptOverrideRow,
    SecretRow,
    SubtaskRow,
    TaskRow,
    UserRow,
)

Sessions = async_sessionmaker[AsyncSession]


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


class UserRepo:
    def __init__(self, sessions: Sessions) -> None:
        self.sessions = sessions

    @staticmethod
    def _dict(row: UserRow) -> dict[str, Any]:
        return {
            "id": row.id,
            "email": row.email,
            "role": row.role,
            "is_active": row.is_active,
            "settings": row.settings or {},
            "created_at": _iso(row.created_at),
        }

    async def count(self) -> int:
        async with self.sessions() as s:
            return int(await s.scalar(select(func.count()).select_from(UserRow)) or 0)

    async def create(self, email: str, password_hash: str, role: str) -> dict[str, Any]:
        async with self.sessions() as s:
            row = UserRow(email=email.lower().strip(), password_hash=password_hash, role=role)
            s.add(row)
            await s.commit()
            return self._dict(row)

    async def get_by_email(self, email: str) -> tuple[dict[str, Any], str] | None:
        async with self.sessions() as s:
            row = await s.scalar(select(UserRow).where(UserRow.email == email.lower().strip()))
            return (self._dict(row), row.password_hash) if row else None

    async def get(self, user_id: str) -> dict[str, Any] | None:
        async with self.sessions() as s:
            row = await s.get(UserRow, user_id)
            return self._dict(row) if row else None

    async def list(self) -> list[dict[str, Any]]:
        async with self.sessions() as s:
            rows = (await s.scalars(select(UserRow).order_by(UserRow.created_at))).all()
            return [self._dict(r) for r in rows]

    async def update(self, user_id: str, **fields: Any) -> dict[str, Any] | None:
        allowed = {k: v for k, v in fields.items() if k in {"role", "is_active", "password_hash", "settings"}}
        async with self.sessions() as s:
            row = await s.get(UserRow, user_id)
            if not row:
                return None
            for k, v in allowed.items():
                setattr(row, k, v)
            await s.commit()
            return self._dict(row)

    # API keys -----------------------------------------------------------------
    async def add_api_key(self, user_id: str, name: str, prefix: str, key_hash: str) -> dict[str, Any]:
        async with self.sessions() as s:
            row = ApiKeyRow(user_id=user_id, name=name, prefix=prefix, key_hash=key_hash)
            s.add(row)
            await s.commit()
            return {"id": row.id, "name": name, "prefix": prefix, "created_at": _iso(row.created_at)}

    async def find_api_keys(self, prefix: str) -> list[tuple[str, str, str]]:
        """Return (key_id, user_id, key_hash) for active keys with this prefix."""
        async with self.sessions() as s:
            rows = (
                await s.scalars(select(ApiKeyRow).where(ApiKeyRow.prefix == prefix, ApiKeyRow.revoked.is_(False)))
            ).all()
            return [(r.id, r.user_id, r.key_hash) for r in rows]

    async def touch_api_key(self, key_id: str) -> None:
        async with self.sessions() as s:
            await s.execute(update(ApiKeyRow).where(ApiKeyRow.id == key_id).values(last_used_at=utcnow()))
            await s.commit()

    async def list_api_keys(self, user_id: str) -> list[dict[str, Any]]:
        async with self.sessions() as s:
            rows = (await s.scalars(select(ApiKeyRow).where(ApiKeyRow.user_id == user_id))).all()
            return [
                {
                    "id": r.id,
                    "name": r.name,
                    "prefix": r.prefix,
                    "revoked": r.revoked,
                    "created_at": _iso(r.created_at),
                    "last_used_at": _iso(r.last_used_at),
                }
                for r in rows
            ]

    async def revoke_api_key(self, user_id: str, key_id: str) -> bool:
        async with self.sessions() as s:
            res = await s.execute(
                update(ApiKeyRow).where(ApiKeyRow.id == key_id, ApiKeyRow.user_id == user_id).values(revoked=True)
            )
            await s.commit()
            return bool(res.rowcount)


class ConversationRepo:
    def __init__(self, sessions: Sessions) -> None:
        self.sessions = sessions

    async def ensure(self, user_id: str, conversation_id: str | None, title: str) -> str:
        async with self.sessions() as s:
            if conversation_id:
                row = await s.get(ConversationRow, conversation_id)
                if row and row.user_id == user_id:
                    row.updated_at = utcnow()
                    await s.commit()
                    return row.id
            row = ConversationRow(user_id=user_id, title=title[:200] or "New conversation")
            s.add(row)
            await s.commit()
            return row.id

    async def add_message(self, conversation_id: str, role: str, content: str, task_id: str | None = None) -> None:
        async with self.sessions() as s:
            s.add(MessageRow(conversation_id=conversation_id, role=role, content=content, task_id=task_id))
            await s.commit()

    async def history(self, conversation_id: str, limit: int) -> list[dict[str, str]]:
        async with self.sessions() as s:
            rows = (
                await s.scalars(
                    select(MessageRow)
                    .where(MessageRow.conversation_id == conversation_id)
                    .order_by(desc(MessageRow.created_at))
                    .limit(limit)
                )
            ).all()
            return [{"role": r.role, "content": r.content} for r in reversed(rows)]

    async def list(self, user_id: str) -> list[dict[str, Any]]:
        async with self.sessions() as s:
            rows = (
                await s.scalars(
                    select(ConversationRow)
                    .where(ConversationRow.user_id == user_id)
                    .order_by(desc(ConversationRow.updated_at))
                    .limit(100)
                )
            ).all()
            return [{"id": r.id, "title": r.title, "updated_at": _iso(r.updated_at)} for r in rows]

    async def messages(self, user_id: str, conversation_id: str) -> list[dict[str, Any]] | None:
        async with self.sessions() as s:
            conv = await s.get(ConversationRow, conversation_id)
            if not conv or conv.user_id != user_id:
                return None
            rows = (
                await s.scalars(
                    select(MessageRow)
                    .where(MessageRow.conversation_id == conversation_id)
                    .order_by(MessageRow.created_at)
                )
            ).all()
            return [
                {"role": r.role, "content": r.content, "task_id": r.task_id, "at": _iso(r.created_at)} for r in rows
            ]


class TaskRepo:
    def __init__(self, sessions: Sessions) -> None:
        self.sessions = sessions

    async def create(self, user_id: str, request: str, conversation_id: str | None) -> str:
        async with self.sessions() as s:
            row = TaskRow(user_id=user_id, request=request, conversation_id=conversation_id, status="planning")
            s.add(row)
            await s.commit()
            return row.id

    async def save_plan(self, task_id: str, plan: Plan) -> None:
        async with self.sessions() as s:
            await s.execute(
                update(TaskRow)
                .where(TaskRow.id == task_id)
                .values(
                    plan=plan.model_dump(mode="json"),
                    intent=plan.intent,
                    priority=plan.priority.value,
                    status=TaskStatus.RUNNING.value,
                    updated_at=utcnow(),
                )
            )
            await s.commit()

    async def finish(
        self,
        task_id: str,
        status: TaskStatus,
        answer: str,
        results: list[AgentResult],
        flags: list[str],
        duration_ms: int,
    ) -> None:
        async with self.sessions() as s:
            await s.execute(delete(SubtaskRow).where(SubtaskRow.task_id == task_id))
            for r in results:
                s.add(
                    SubtaskRow(
                        task_id=task_id,
                        key=r.subtask_id,
                        agent=r.agent,
                        instruction="",
                        status="completed" if r.ok else "failed",
                        output=r.output,
                        score=r.score,
                        feedback=r.feedback,
                        revisions=r.revisions,
                        tool_calls=[t.model_dump(mode="json") for t in r.tool_calls],
                    )
                )
            await s.execute(
                update(TaskRow)
                .where(TaskRow.id == task_id)
                .values(status=status.value, answer=answer, flags=flags, duration_ms=duration_ms, updated_at=utcnow())
            )
            await s.commit()

    async def set_status(self, task_id: str, status: TaskStatus) -> None:
        async with self.sessions() as s:
            await s.execute(
                update(TaskRow).where(TaskRow.id == task_id).values(status=status.value, updated_at=utcnow())
            )
            await s.commit()

    @staticmethod
    def _dict(row: TaskRow) -> dict[str, Any]:
        return {
            "id": row.id,
            "request": row.request,
            "status": row.status,
            "priority": row.priority,
            "intent": row.intent,
            "plan": row.plan,
            "answer": row.answer,
            "flags": row.flags,
            "duration_ms": row.duration_ms,
            "conversation_id": row.conversation_id,
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
        }

    async def list(self, user_id: str | None, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        async with self.sessions() as s:
            q = select(TaskRow).order_by(desc(TaskRow.created_at)).limit(limit)
            if user_id:
                q = q.where(TaskRow.user_id == user_id)
            if status:
                q = q.where(TaskRow.status == status)
            return [self._dict(r) for r in (await s.scalars(q)).all()]

    async def get(self, task_id: str, user_id: str | None) -> dict[str, Any] | None:
        async with self.sessions() as s:
            row = await s.get(TaskRow, task_id)
            if not row or (user_id and row.user_id != user_id):
                return None
            data = self._dict(row)
            subs = (await s.scalars(select(SubtaskRow).where(SubtaskRow.task_id == task_id))).all()
            data["subtasks"] = [
                {
                    "key": r.key,
                    "agent": r.agent,
                    "status": r.status,
                    "output": r.output,
                    "score": r.score,
                    "feedback": r.feedback,
                    "revisions": r.revisions,
                    "tool_calls": r.tool_calls,
                }
                for r in subs
            ]
            return data

    async def metrics(self) -> dict[str, Any]:
        async with self.sessions() as s:
            by_status = dict((await s.execute(select(TaskRow.status, func.count()).group_by(TaskRow.status))).all())
            avg_ms = await s.scalar(select(func.avg(TaskRow.duration_ms)).where(TaskRow.duration_ms > 0))
            by_agent_rows = (
                await s.execute(
                    select(SubtaskRow.agent, func.count(), func.avg(SubtaskRow.score)).group_by(SubtaskRow.agent)
                )
            ).all()
            return {
                "tasks_by_status": by_status,
                "avg_task_ms": round(float(avg_ms or 0)),
                "agents": [
                    {"agent": a, "subtasks": int(c), "avg_score": round(float(sc), 2) if sc is not None else None}
                    for a, c, sc in by_agent_rows
                ],
            }


class MemoryRepo:
    def __init__(self, sessions: Sessions) -> None:
        self.sessions = sessions

    async def add(self, item: MemoryItem) -> None:
        async with self.sessions() as s:
            s.add(
                MemoryRow(
                    id=item.id,
                    user_id=item.user_id,
                    kind=item.kind.value,
                    content=item.content,
                    meta=item.metadata,
                    created_at=item.created_at,
                )
            )
            await s.commit()

    async def list(self, user_id: str, kind: MemoryKind | None = None, limit: int = 200) -> list[MemoryItem]:
        async with self.sessions() as s:
            q = select(MemoryRow).where(MemoryRow.user_id == user_id).order_by(desc(MemoryRow.created_at)).limit(limit)
            if kind:
                q = q.where(MemoryRow.kind == kind.value)
            return [
                MemoryItem(
                    id=r.id,
                    user_id=r.user_id,
                    kind=MemoryKind(r.kind),
                    content=r.content,
                    metadata=r.meta or {},
                    created_at=r.created_at,
                )
                for r in (await s.scalars(q)).all()
            ]

    async def delete(self, user_id: str, memory_id: str) -> bool:
        async with self.sessions() as s:
            res = await s.execute(delete(MemoryRow).where(MemoryRow.id == memory_id, MemoryRow.user_id == user_id))
            await s.commit()
            return bool(res.rowcount)

    # knowledge documents
    async def add_doc(self, user_id: str, title: str, source: str, chunks: int) -> str:
        async with self.sessions() as s:
            row = KnowledgeDocRow(user_id=user_id, title=title, source=source, chunks=chunks)
            s.add(row)
            await s.commit()
            return row.id

    async def list_docs(self, user_id: str) -> list[dict[str, Any]]:
        async with self.sessions() as s:
            rows = (
                await s.scalars(
                    select(KnowledgeDocRow)
                    .where(KnowledgeDocRow.user_id == user_id)
                    .order_by(desc(KnowledgeDocRow.created_at))
                )
            ).all()
            return [
                {"id": r.id, "title": r.title, "source": r.source, "chunks": r.chunks, "created_at": _iso(r.created_at)}
                for r in rows
            ]

    async def delete_doc(self, user_id: str, doc_id: str) -> bool:
        async with self.sessions() as s:
            res = await s.execute(
                delete(KnowledgeDocRow).where(KnowledgeDocRow.id == doc_id, KnowledgeDocRow.user_id == user_id)
            )
            await s.commit()
            return bool(res.rowcount)


class ApprovalRepo:
    def __init__(self, sessions: Sessions) -> None:
        self.sessions = sessions

    @staticmethod
    def _dict(r: ApprovalRow) -> dict[str, Any]:
        return {
            "id": r.id,
            "task_id": r.task_id,
            "agent": r.agent,
            "tool": r.tool,
            "summary": r.summary,
            "status": r.status,
            "result": r.result,
            "created_at": _iso(r.created_at),
            "decided_at": _iso(r.decided_at),
        }

    async def create(
        self, user_id: str, task_id: str | None, agent: str, tool: str, args_encrypted: str, summary: str
    ) -> str:
        async with self.sessions() as s:
            row = ApprovalRow(
                id=new_id(),
                user_id=user_id,
                task_id=task_id,
                agent=agent,
                tool=tool,
                args_encrypted=args_encrypted,
                summary=summary,
            )
            s.add(row)
            await s.commit()
            return row.id

    async def get_raw(self, approval_id: str, user_id: str) -> ApprovalRow | None:
        async with self.sessions() as s:
            row = await s.get(ApprovalRow, approval_id)
            return row if row and row.user_id == user_id else None

    async def decide(self, approval_id: str, status: ApprovalStatus, decided_by: str, result: str = "") -> None:
        async with self.sessions() as s:
            await s.execute(
                update(ApprovalRow)
                .where(ApprovalRow.id == approval_id)
                .values(status=status.value, decided_by=decided_by, decided_at=utcnow(), result=result)
            )
            await s.commit()

    async def list(self, user_id: str, status: str | None = None) -> list[dict[str, Any]]:
        async with self.sessions() as s:
            q = select(ApprovalRow).where(ApprovalRow.user_id == user_id).order_by(desc(ApprovalRow.created_at))
            if status:
                q = q.where(ApprovalRow.status == status)
            return [self._dict(r) for r in (await s.scalars(q.limit(200))).all()]


class AuditRepo:
    def __init__(self, sessions: Sessions) -> None:
        self.sessions = sessions

    async def log(
        self,
        action: str,
        user_id: str | None = None,
        resource: str = "",
        detail: dict[str, Any] | None = None,
        ip: str = "",
    ) -> None:
        async with self.sessions() as s:
            s.add(AuditRow(action=action, user_id=user_id, resource=resource, detail=detail or {}, ip=ip))
            await s.commit()

    async def list(
        self, user_id: str | None = None, action: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        async with self.sessions() as s:
            q = select(AuditRow).order_by(desc(AuditRow.at)).limit(limit)
            if user_id:
                q = q.where(AuditRow.user_id == user_id)
            if action:
                q = q.where(AuditRow.action.like(f"{action}%"))
            return [
                {
                    "id": r.id,
                    "at": _iso(r.at),
                    "user_id": r.user_id,
                    "action": r.action,
                    "resource": r.resource,
                    "detail": r.detail,
                    "ip": r.ip,
                }
                for r in (await s.scalars(q)).all()
            ]


class PromptRepo:
    def __init__(self, sessions: Sessions) -> None:
        self.sessions = sessions

    async def get_all(self) -> dict[str, str]:
        async with self.sessions() as s:
            return {r.name: r.content for r in (await s.scalars(select(PromptOverrideRow))).all()}

    async def set(self, name: str, content: str, user_id: str) -> None:
        async with self.sessions() as s:
            row = await s.get(PromptOverrideRow, name)
            if row:
                row.content, row.updated_by, row.updated_at = content, user_id, utcnow()
            else:
                s.add(PromptOverrideRow(name=name, content=content, updated_by=user_id))
            await s.commit()

    async def reset(self, name: str) -> None:
        async with self.sessions() as s:
            await s.execute(delete(PromptOverrideRow).where(PromptOverrideRow.name == name))
            await s.commit()


class SecretRepo:
    """Stores integration credentials encrypted at rest (ciphertext produced by security.crypto)."""

    def __init__(self, sessions: Sessions) -> None:
        self.sessions = sessions

    async def set(self, name: str, ciphertext: str) -> None:
        async with self.sessions() as s:
            row = await s.get(SecretRow, name)
            if row:
                row.ciphertext, row.updated_at = ciphertext, utcnow()
            else:
                s.add(SecretRow(name=name, ciphertext=ciphertext))
            await s.commit()

    async def get(self, name: str) -> str | None:
        async with self.sessions() as s:
            row = await s.get(SecretRow, name)
            return row.ciphertext if row else None

    async def names(self) -> list[dict[str, Any]]:
        async with self.sessions() as s:
            rows = (await s.scalars(select(SecretRow))).all()
            return [{"name": r.name, "updated_at": _iso(r.updated_at)} for r in rows]

    async def delete(self, name: str) -> bool:
        async with self.sessions() as s:
            res = await s.execute(delete(SecretRow).where(SecretRow.name == name))
            await s.commit()
            return bool(res.rowcount)
