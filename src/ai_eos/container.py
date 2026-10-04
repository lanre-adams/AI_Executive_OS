"""Composition root: builds every service from Settings. The only place wiring happens."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ai_eos.agents.base import PromptLibrary, SpecialistAgent
from ai_eos.config import Settings
from ai_eos.domain.events import EventBus
from ai_eos.domain.models import Role
from ai_eos.infrastructure.db import Database
from ai_eos.infrastructure.kv import KeyValueStore, make_kv
from ai_eos.infrastructure.repositories import (
    ApprovalRepo,
    AuditRepo,
    ConversationRepo,
    MemoryRepo,
    PromptRepo,
    SecretRepo,
    TaskRepo,
    UserRepo,
)
from ai_eos.infrastructure.vector import VectorStore
from ai_eos.llm.router import LLMRouter
from ai_eos.memory.manager import MemoryManager
from ai_eos.orchestration.chief_of_staff import ChiefOfStaff
from ai_eos.security.auth import TokenService, hash_password
from ai_eos.security.crypto import Cipher
from ai_eos.security.guard import RateLimiter
from ai_eos.tools import build_registry
from ai_eos.tools.base import SecretResolver, ToolContext, ToolExecutor, ToolRegistry

log = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    db: Database
    kv: KeyValueStore
    vectors: VectorStore
    bus: EventBus
    cipher: Cipher
    tokens: TokenService
    rate_limiter: RateLimiter
    users: UserRepo
    conversations: ConversationRepo
    tasks: TaskRepo
    approvals: ApprovalRepo
    audit: AuditRepo
    prompt_repo: PromptRepo
    secret_repo: SecretRepo
    memory: MemoryManager
    llm: LLMRouter
    prompts: PromptLibrary
    tools: ToolRegistry
    executor: ToolExecutor
    secrets: SecretResolver
    agents: dict[str, SpecialistAgent] = field(default_factory=dict)
    chief: ChiefOfStaff | None = None
    tool_transport: object | None = None  # tests inject an httpx MockTransport

    def tool_context(self, user_id: str, agent: str, task_id: str | None) -> ToolContext:
        return ToolContext(
            user_id=user_id,
            agent=agent,
            task_id=task_id,
            settings=self.settings,
            secrets=self.secrets,
            kv=self.kv,
            services={"memory": self.memory},
            transport=self.tool_transport,
        )  # type: ignore[arg-type]

    async def startup(self) -> None:
        await self.db.create_all()
        await self.prompts.refresh()
        await self.bootstrap_admin()

    async def bootstrap_admin(self) -> None:
        s = self.settings.secrets
        if await self.users.count() == 0 and s.bootstrap_admin_email and s.bootstrap_admin_password:
            await self.users.create(s.bootstrap_admin_email, hash_password(s.bootstrap_admin_password), Role.ADMIN)
            await self.audit.log("user.bootstrap_admin", None, s.bootstrap_admin_email)
            log.info("created bootstrap administrator %s", s.bootstrap_admin_email)

    async def shutdown(self) -> None:
        await self.db.dispose()


def build_container(settings: Settings, vectors: VectorStore | None = None) -> Container:
    db = Database(settings.database.url, settings.database.echo)
    sessions = db.sessions
    kv = make_kv(settings.redis.url)
    vectors = vectors or VectorStore.from_settings(settings)
    bus = EventBus()
    cipher = Cipher(settings.secrets.encryption_key, fallback_seed=settings.secrets.jwt_secret)
    secret_repo = SecretRepo(sessions)
    approvals, audit, prompt_repo = ApprovalRepo(sessions), AuditRepo(sessions), PromptRepo(sessions)
    memory = MemoryManager(MemoryRepo(sessions), vectors, kv, settings.memory)
    registry = build_registry(settings)
    executor = ToolExecutor(registry, approvals, audit, bus, cipher, settings.tools.auto_approve)
    llm = LLMRouter(settings)
    prompts = PromptLibrary(prompt_repo)

    c = Container(
        settings=settings,
        db=db,
        kv=kv,
        vectors=vectors,
        bus=bus,
        cipher=cipher,
        tokens=TokenService(
            settings.secrets.jwt_secret, settings.security.jwt_algorithm, settings.security.access_token_minutes
        ),
        rate_limiter=RateLimiter(kv, settings.security.rate_limit_per_minute),
        users=UserRepo(sessions),
        conversations=ConversationRepo(sessions),
        tasks=TaskRepo(sessions),
        approvals=approvals,
        audit=audit,
        prompt_repo=prompt_repo,
        secret_repo=secret_repo,
        memory=memory,
        llm=llm,
        prompts=prompts,
        tools=registry,
        executor=executor,
        secrets=SecretResolver(secret_repo, cipher),
    )
    c.agents = {
        key: SpecialistAgent(key, d, settings, llm, prompts, registry, executor)
        for key, d in settings.agents.specialists.items()
    }
    c.chief = ChiefOfStaff(
        settings, llm, prompts, c.agents, memory, executor, c.tasks, c.conversations, audit, bus, c.tool_context
    )
    return c
