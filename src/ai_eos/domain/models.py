"""Domain entities and value objects for the executive operating system."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(UTC)


class Role(StrEnum):
    ADMIN = "admin"  # everything, including users, settings and prompts
    OPERATOR = "operator"  # use the system, approve tool actions
    VIEWER = "viewer"  # read-only dashboards


class TaskStatus(StrEnum):
    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    NEEDS_INPUT = "needs_input"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


class Priority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MemoryKind(StrEnum):
    FACT = "fact"  # durable facts about the user and their organisation
    PREFERENCE = "preference"  # how the user wants work done
    EPISODE = "episode"  # summary of a completed request
    REFLECTION = "reflection"  # lessons the Chief of Staff drew from its own performance
    KNOWLEDGE = "knowledge"  # chunks of ingested documents


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"


class ChatMessage(BaseModel):
    role: str  # system | user | assistant
    content: str


class Subtask(BaseModel):
    id: str
    agent: str
    instruction: str
    depends_on: list[str] = Field(default_factory=list)
    expected_output: str = ""


class Plan(BaseModel):
    intent: str = ""
    priority: Priority = Priority.NORMAL
    needs_clarification: bool = False
    clarifying_question: str = ""
    direct_answer: str = ""  # trivial requests the Chief of Staff answers itself
    subtasks: list[Subtask] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    ok: bool = True
    approval_id: str | None = None


class AgentResult(BaseModel):
    subtask_id: str
    agent: str
    output: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    ok: bool = True
    error: str = ""
    score: float | None = None
    feedback: str = ""
    revisions: int = 0
    pending_approvals: list[str] = Field(default_factory=list)


class Review(BaseModel):
    subtask_id: str
    score: float
    passed: bool
    feedback: str = ""


class MemoryItem(BaseModel):
    id: str = Field(default_factory=new_id)
    user_id: str
    kind: MemoryKind
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    created_at: datetime = Field(default_factory=utcnow)


class RunOutcome(BaseModel):
    """What the Chief of Staff returns for one user request."""

    task_id: str
    status: TaskStatus
    answer: str
    plan: Plan
    results: list[AgentResult] = Field(default_factory=list)
    pending_approvals: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
