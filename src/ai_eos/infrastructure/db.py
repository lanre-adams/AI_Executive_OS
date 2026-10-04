"""SQLAlchemy async engine and ORM tables (PostgreSQL in production, SQLite locally)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ai_eos.domain.models import new_id, utcnow


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


def _id() -> Mapped[str]:
    return mapped_column(String(32), primary_key=True, default=new_id)


def _ts() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), default=utcnow)


class UserRow(Base):
    __tablename__ = "users"
    id: Mapped[str] = _id()
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(20), default="operator")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = _ts()


class ApiKeyRow(Base):
    __tablename__ = "api_keys"
    id: Mapped[str] = _id()
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    prefix: Mapped[str] = mapped_column(String(16), index=True)
    key_hash: Mapped[str] = mapped_column(String(128))
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = _ts()
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ConversationRow(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = _id()
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(200), default="New conversation")
    created_at: Mapped[datetime] = _ts()
    updated_at: Mapped[datetime] = _ts()


class MessageRow(Base):
    __tablename__ = "messages"
    id: Mapped[str] = _id()
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    task_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = _ts()
    __table_args__ = (Index("ix_messages_conv_created", "conversation_id", "created_at"),)


class TaskRow(Base):
    __tablename__ = "tasks"
    id: Mapped[str] = _id()
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    request: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    priority: Mapped[str] = mapped_column(String(10), default="normal")
    intent: Mapped[str] = mapped_column(Text, default="")
    plan: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    answer: Mapped[str] = mapped_column(Text, default="")
    flags: Mapped[list[Any]] = mapped_column(JSON, default=list)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = _ts()
    updated_at: Mapped[datetime] = _ts()


class SubtaskRow(Base):
    __tablename__ = "subtasks"
    id: Mapped[str] = _id()
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    key: Mapped[str] = mapped_column(String(40))
    agent: Mapped[str] = mapped_column(String(60))
    instruction: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), default="completed")
    output: Mapped[str] = mapped_column(Text, default="")
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    feedback: Mapped[str] = mapped_column(Text, default="")
    revisions: Mapped[int] = mapped_column(Integer, default=0)
    tool_calls: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = _ts()


class MemoryRow(Base):
    __tablename__ = "memories"
    id: Mapped[str] = _id()
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20), index=True)
    content: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = _ts()


class KnowledgeDocRow(Base):
    __tablename__ = "knowledge_docs"
    id: Mapped[str] = _id()
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(300))
    source: Mapped[str] = mapped_column(String(500), default="")
    chunks: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = _ts()


class ApprovalRow(Base):
    __tablename__ = "approvals"
    id: Mapped[str] = _id()
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    task_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    agent: Mapped[str] = mapped_column(String(60))
    tool: Mapped[str] = mapped_column(String(80))
    args_encrypted: Mapped[str] = mapped_column(Text)  # tool arguments can hold email bodies etc.
    summary: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    result: Mapped[str] = mapped_column(Text, default="")
    decided_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = _ts()
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditRow(Base):
    __tablename__ = "audit_logs"
    id: Mapped[str] = _id()
    at: Mapped[datetime] = _ts()
    user_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(80), index=True)
    resource: Mapped[str] = mapped_column(String(200), default="")
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    ip: Mapped[str] = mapped_column(String(64), default="")


class PromptOverrideRow(Base):
    __tablename__ = "prompt_overrides"
    name: Mapped[str] = mapped_column(String(80), primary_key=True)
    content: Mapped[str] = mapped_column(Text)
    updated_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    updated_at: Mapped[datetime] = _ts()


class SecretRow(Base):
    __tablename__ = "secrets"
    name: Mapped[str] = mapped_column(String(80), primary_key=True)
    ciphertext: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = _ts()


class Database:
    def __init__(self, url: str, echo: bool = False) -> None:
        if url.startswith("sqlite") and ":///" in url and ":memory:" not in url:
            Path(url.split(":///", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
        self.engine: AsyncEngine = create_async_engine(url, echo=echo, pool_pre_ping=True)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def create_all(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def ping(self) -> bool:
        from sqlalchemy import text

        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001 - health check must never raise
            return False

    async def dispose(self) -> None:
        await self.engine.dispose()
