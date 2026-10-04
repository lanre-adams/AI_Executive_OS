"""The Chief of Staff: a LangGraph state machine that plans, delegates, reviews, synthesises and learns.

analyze ──► clarify? ──► finish
   │        direct?  ──► finish
   ▼
execute ──► review ──► revise? ──► execute (bounded by max_revisions)
               │
               ▼
          synthesize ──► reflect ──► END
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from prometheus_client import Counter, Histogram

from ai_eos.agents.base import PromptLibrary, SpecialistAgent
from ai_eos.config import Settings
from ai_eos.domain import events as ev
from ai_eos.domain.events import Event, EventBus
from ai_eos.domain.models import (
    AgentResult,
    ChatMessage,
    MemoryKind,
    Plan,
    Review,
    RunOutcome,
    Subtask,
    TaskStatus,
)
from ai_eos.infrastructure.repositories import AuditRepo, ConversationRepo, TaskRepo
from ai_eos.llm.base import LLMRequest, parse_json
from ai_eos.llm.router import LLMRouter
from ai_eos.memory.manager import MemoryManager
from ai_eos.orchestration.planning import normalise_plan, waves
from ai_eos.security.guard import screen
from ai_eos.tools.base import ToolContext, ToolExecutor

log = logging.getLogger(__name__)
TASKS = Counter("eos_tasks_total", "Requests handled by the Chief of Staff", ["status"])
TASK_SECONDS = Histogram("eos_task_seconds", "End-to-end request time", buckets=(1, 5, 15, 30, 60, 120, 300, 600))


class RunState(TypedDict, total=False):
    user_id: str
    task_id: str
    request: str
    history: list[dict[str, str]]
    memory_context: str
    provider: str | None
    plan: Plan
    results: dict[str, AgentResult]
    to_run: list[str]
    feedback: dict[str, str]
    revision_round: int
    answer: str
    status: TaskStatus
    flags: list[str]


class ChiefOfStaff:
    def __init__(
        self,
        settings: Settings,
        llm: LLMRouter,
        prompts: PromptLibrary,
        agents: dict[str, SpecialistAgent],
        memory: MemoryManager,
        executor: ToolExecutor,
        tasks: TaskRepo,
        conversations: ConversationRepo,
        audit: AuditRepo,
        bus: EventBus,
        context_factory: Any,
    ) -> None:
        self.settings, self.llm, self.prompts, self.agents = settings, llm, prompts, agents
        self.memory, self.executor, self.tasks, self.conversations = memory, executor, tasks, conversations
        self.audit, self.bus = audit, bus
        self._ctx_factory = context_factory  # (user_id, agent, task_id) -> ToolContext
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ public API
    async def handle(
        self,
        user_id: str,
        request: str,
        conversation_id: str | None = None,
        provider: str | None = None,
    ) -> tuple[RunOutcome, str]:
        """Process one request end-to-end. Returns (outcome, conversation_id)."""
        started = time.perf_counter()
        sec = self.settings.security
        request = request[: sec.max_input_chars]
        check = screen(request)
        flags = [f"input:{f}" for f in check.flags] if sec.prompt_guard else []
        if flags:
            await self.audit.log("security.prompt_flag", user_id, "chat", {"flags": flags})
            await self.bus.publish(Event(type=ev.SECURITY_FLAG, user_id=user_id, data={"flags": flags}))
        conv_id = await self.conversations.ensure(user_id, conversation_id, check.text[:80])
        history = await self.conversations.history(conv_id, self.settings.orchestration.history_turns)
        await self.conversations.add_message(conv_id, "user", check.text)
        await self.memory.push_turn(conv_id, "user", check.text)
        task_id = await self.tasks.create(user_id, check.text, conv_id)
        await self.bus.publish(
            Event(type=ev.TASK_CREATED, user_id=user_id, task_id=task_id, data={"request": check.text[:200]})
        )

        if flags and sec.block_on_injection:
            answer = (
                "I didn't act on that message because it looks like it contains instructions meant to "
                "override my safeguards. Please rephrase if this was a genuine request."
            )
            plan = Plan(intent="blocked by prompt guard")
            outcome = RunOutcome(task_id=task_id, status=TaskStatus.FAILED, answer=answer, plan=plan, flags=flags)
            await self._persist(user_id, conv_id, outcome, started)
            return outcome, conv_id

        state: RunState = {
            "user_id": user_id,
            "task_id": task_id,
            "request": check.text,
            "history": history,
            "provider": provider,
            "results": {},
            "feedback": {},
            "revision_round": 0,
            "flags": flags,
        }
        try:
            final: RunState = await self.graph.ainvoke(state)
            results = list(final.get("results", {}).values())
            pending = [a for r in results for a in r.pending_approvals]
            outcome = RunOutcome(
                task_id=task_id,
                status=final["status"],
                answer=final.get("answer", ""),
                plan=final.get("plan", Plan()),
                results=results,
                pending_approvals=pending,
                flags=final.get("flags", flags),
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("task %s failed", task_id)
            outcome = RunOutcome(
                task_id=task_id,
                status=TaskStatus.FAILED,
                plan=Plan(),
                flags=flags,
                answer=f"I couldn't complete this request: {type(exc).__name__}: {exc}",
            )
        await self._persist(user_id, conv_id, outcome, started)
        return outcome, conv_id

    async def run_agent(
        self, user_id: str, agent_key: str, instruction: str, provider: str | None = None
    ) -> AgentResult:
        """Direct access to one specialist, used only when the principal explicitly asks for it."""
        agent = self.agents[agent_key]
        ctx = self._ctx_factory(user_id, agent_key, None)
        await self.audit.log("agent.direct", user_id, agent_key, {"instruction": instruction[:200]})
        return await agent.run(Subtask(id="direct", agent=agent_key, instruction=instruction), ctx, provider=provider)

    # ------------------------------------------------------------------ graph
    def _build_graph(self) -> Any:
        g = StateGraph(RunState)
        g.add_node("analyze", self._analyze)
        g.add_node("finish_early", self._finish_early)
        g.add_node("execute", self._execute)
        g.add_node("review", self._review)
        g.add_node("synthesize", self._synthesize)
        g.add_node("reflect", self._reflect)
        g.add_edge(START, "analyze")
        g.add_conditional_edges("analyze", self._after_analyze, {"early": "finish_early", "execute": "execute"})
        g.add_edge("finish_early", "reflect")
        g.add_edge("execute", "review")
        g.add_conditional_edges("review", self._after_review, {"revise": "execute", "done": "synthesize"})
        g.add_edge("synthesize", "reflect")
        g.add_edge("reflect", END)
        return g.compile()

    async def _complete(
        self,
        prompt_name: str,
        user: str,
        purpose: str,
        provider: str | None,
        json_mode: bool = True,
        system: str | None = None,
    ) -> str:
        cos = self.settings.agents.chief_of_staff
        system = system or self.prompts.get(prompt_name)
        req = LLMRequest(
            system=system,
            messages=[ChatMessage(role="user", content=user)],
            temperature=self.settings.llm.temperature,
            max_tokens=self.settings.llm.max_tokens,
            json_mode=json_mode,
            purpose=purpose,
        )
        resp = await self.llm.complete(req, cos.provider or provider, cos.model)
        return resp.text

    def _team(self) -> str:
        return "\n".join(
            f"- {k}: {a.definition.title} - {a.definition.description} (tools: {', '.join(a.allowed_tools) or 'none'})"
            for k, a in self.agents.items()
        )

    async def _analyze(self, state: RunState) -> RunState:
        mem = await self.memory.context_for(state["user_id"], state["request"])
        history = "\n".join(f"{m['role']}: {m['content'][:600]}" for m in state.get("history", []))
        system = (
            self.prompts.get(self.settings.agents.chief_of_staff.prompt)
            .replace("{team}", self._team())
            .replace("{max_subtasks}", str(self.settings.orchestration.max_subtasks))
            .replace("{agent_keys}", ", ".join(self.agents))
            + "\n\n"
            + self.prompts.get("_shared_rules")
        )
        user = "\n\n".join(
            p
            for p in [
                f"WHAT YOU KNOW ABOUT THE PRINCIPAL:\n{mem}" if mem else "",
                f"RECENT CONVERSATION:\n{history}" if history else "",
                f"USER REQUEST:\n{state['request']}",
            ]
            if p
        )
        raw = await self._complete("chief_of_staff", user, "plan", state.get("provider"), system=system)
        try:
            data = parse_json(raw)
        except ValueError:
            data = {}
        plan = normalise_plan(data, list(self.agents), self.settings.orchestration.max_subtasks, state["request"])
        await self.tasks.save_plan(state["task_id"], plan)
        await self.bus.publish(
            Event(
                type=ev.TASK_PLANNED,
                user_id=state["user_id"],
                task_id=state["task_id"],
                data={
                    "intent": plan.intent,
                    "priority": plan.priority.value,
                    "subtasks": [s.model_dump() for s in plan.subtasks],
                },
            )
        )
        return {"plan": plan, "memory_context": mem, "to_run": [s.id for s in plan.subtasks]}

    @staticmethod
    def _after_analyze(state: RunState) -> str:
        plan = state["plan"]
        return "early" if plan.needs_clarification or (plan.direct_answer and not plan.subtasks) else "execute"

    async def _finish_early(self, state: RunState) -> RunState:
        plan = state["plan"]
        if plan.needs_clarification:
            await self.bus.publish(
                Event(
                    type=ev.TASK_NEEDS_INPUT,
                    user_id=state["user_id"],
                    task_id=state["task_id"],
                    data={"question": plan.clarifying_question},
                )
            )
            return {"answer": plan.clarifying_question, "status": TaskStatus.NEEDS_INPUT}
        return {"answer": plan.direct_answer, "status": TaskStatus.COMPLETED}

    def _context_for(self, state: RunState, subtask: Subtask) -> str:
        parts = [f"Original request from the principal:\n{state['request']}"]
        if state.get("memory_context"):
            parts.append(f"Relevant memory:\n{state['memory_context']}")
        for dep in subtask.depends_on:
            res = state["results"].get(dep)
            if res and res.ok:
                parts.append(f"Output from {res.agent} ({dep}):\n{res.output[:6000]}")
        return "\n\n".join(parts)

    async def _run_one(self, state: RunState, subtask: Subtask, sem: asyncio.Semaphore) -> AgentResult:
        async with sem:
            agent = self.agents[subtask.agent]
            await self.bus.publish(
                Event(
                    type=ev.SUBTASK_STARTED,
                    user_id=state["user_id"],
                    task_id=state["task_id"],
                    data={"subtask": subtask.id, "agent": subtask.agent},
                )
            )
            prev = state["results"].get(subtask.id)
            ctx: ToolContext = self._ctx_factory(state["user_id"], subtask.agent, state["task_id"])
            result = await agent.run(
                subtask,
                ctx,
                self._context_for(state, subtask),
                feedback=state["feedback"].get(subtask.id, ""),
                previous=prev.output if prev else "",
                provider=state.get("provider"),
            )
            result.revisions = (prev.revisions + 1) if prev else 0
            await self.bus.publish(
                Event(
                    type=ev.SUBTASK_COMPLETED,
                    user_id=state["user_id"],
                    task_id=state["task_id"],
                    data={
                        "subtask": subtask.id,
                        "agent": subtask.agent,
                        "ok": result.ok,
                        "tools": [c.tool for c in result.tool_calls],
                    },
                )
            )
            return result

    async def _execute(self, state: RunState) -> RunState:
        plan = state["plan"]
        targets = set(state.get("to_run", []))
        sem = asyncio.Semaphore(self.settings.orchestration.max_parallel_agents)
        results = dict(state["results"])
        for wave in waves(plan.subtasks):
            batch = [s for s in wave if s.id in targets]
            if not batch:
                continue
            state_view: RunState = {**state, "results": results}
            outs = await asyncio.gather(*(self._run_one(state_view, s, sem) for s in batch))
            for res in outs:
                results[res.subtask_id] = res
        return {"results": results}

    async def _review_one(self, state: RunState, subtask: Subtask, result: AgentResult) -> Review:
        if not result.ok:
            return Review(subtask_id=subtask.id, score=0.0, passed=False, feedback=f"Agent error: {result.error}")
        user = (
            f"INSTRUCTION:\n{subtask.instruction}\n\nEXPECTED OUTPUT:\n{subtask.expected_output or 'n/a'}"
            f"\n\nAGENT OUTPUT:\n{result.output[:12000]}"
        )
        try:
            data = parse_json(await self._complete("chief_of_staff_review", user, "review", state.get("provider")))
            score = max(0.0, min(1.0, float(data.get("score", 0.7))))
        except (ValueError, TypeError):
            return Review(subtask_id=subtask.id, score=0.7, passed=True, feedback="review unavailable")
        passed = score >= self.settings.orchestration.validation_threshold
        return Review(subtask_id=subtask.id, score=score, passed=passed, feedback=str(data.get("feedback", "")))

    async def _review(self, state: RunState) -> RunState:
        plan, results = state["plan"], dict(state["results"])
        just_ran = [s for s in plan.subtasks if s.id in set(state.get("to_run", [])) and s.id in results]
        reviews = await asyncio.gather(*(self._review_one(state, s, results[s.id]) for s in just_ran))
        feedback, redo = dict(state["feedback"]), []
        can_revise = state["revision_round"] < self.settings.orchestration.max_revisions
        for rv in reviews:
            res = results[rv.subtask_id]
            res.score, res.feedback = rv.score, rv.feedback
            await self.bus.publish(
                Event(
                    type=ev.SUBTASK_REVIEWED,
                    user_id=state["user_id"],
                    task_id=state["task_id"],
                    data={"subtask": rv.subtask_id, "score": rv.score, "passed": rv.passed},
                )
            )
            # Don't redo work that is only waiting on approvals - re-running would queue duplicates.
            if not rv.passed and can_revise and not res.pending_approvals:
                feedback[rv.subtask_id] = rv.feedback
                redo.append(rv.subtask_id)
        return {
            "results": results,
            "feedback": feedback,
            "to_run": redo,
            "revision_round": state["revision_round"] + (1 if redo else 0),
        }

    @staticmethod
    def _after_review(state: RunState) -> str:
        return "revise" if state.get("to_run") else "done"

    async def _synthesize(self, state: RunState) -> RunState:
        plan, results = state["plan"], state["results"]
        ordered = [results[s.id] for s in plan.subtasks if s.id in results]
        ok = [r for r in ordered if r.ok and r.output]
        pending = [a for r in ordered for a in r.pending_approvals]
        if not ok:
            errors = "; ".join(f"{r.agent}: {r.error or 'no output'}" for r in ordered)
            return {
                "answer": f"The team could not complete this request. Details: {errors}",
                "status": TaskStatus.FAILED,
            }
        if len(ok) == 1 and len(ordered) == 1:
            answer = ok[0].output
        else:
            outputs = "\n\n".join(
                f"### {self.agents[r.agent].definition.title} (quality {r.score})\n{r.output}" for r in ok
            )
            failed = [r for r in ordered if not r.ok]
            if failed:
                outputs += "\n\n### Could not complete\n" + "\n".join(f"- {r.agent}: {r.error}" for r in failed)
            user = f"USER REQUEST:\n{state['request']}\n\nTEAM OUTPUTS:\n{outputs}"
            if pending:
                user += f"\n\nACTIONS AWAITING APPROVAL:\n{len(pending)} action(s) queued"
            answer = await self._complete(
                "chief_of_staff_synthesize", user, "synthesize", state.get("provider"), json_mode=False
            )
        if pending:
            answer += (
                f"\n\n---\n**{len(pending)} action(s) are waiting for your approval** "
                "in the Approvals tab (or `eos approvals`)."
            )
        status = TaskStatus.AWAITING_APPROVAL if pending else TaskStatus.COMPLETED
        return {"answer": answer, "status": status}

    async def _reflect(self, state: RunState) -> RunState:
        """Store durable facts, preferences, lessons and an episode summary. Never fails the run."""
        if state.get("status") in (TaskStatus.FAILED, TaskStatus.NEEDS_INPUT) or not state["plan"].subtasks:
            return {}
        try:
            scores = [r.score for r in state.get("results", {}).values() if r.score is not None]
            user = (
                f"USER REQUEST:\n{state['request']}\n\nPLAN INTENT:\n{state['plan'].intent}\n\n"
                f"QUALITY SCORES:\n{scores}\n\nFINAL ANSWER (excerpt):\n{state.get('answer', '')[:3000]}"
            )
            data = parse_json(await self._complete("chief_of_staff_reflect", user, "reflect", state.get("provider")))
            uid, meta = state["user_id"], {"task_id": state["task_id"]}
            existing = {m.content.lower() for m in await self.memory.recall(uid, state["request"], top_k=20)}
            for kind, key in (
                (MemoryKind.FACT, "facts"),
                (MemoryKind.PREFERENCE, "preferences"),
                (MemoryKind.REFLECTION, "lessons"),
            ):
                for text in (data.get(key) or [])[:5]:
                    text = str(text).strip()
                    if text and text.lower() not in existing and not screen(text).suspicious:
                        await self.memory.remember(uid, kind, text, meta)
            summary = str(data.get("summary") or state["plan"].intent or state["request"][:200])
            await self.memory.remember(uid, MemoryKind.EPISODE, summary, meta)
        except Exception:  # noqa: BLE001
            log.warning("reflection failed for task %s", state.get("task_id"), exc_info=True)
        return {}

    # ------------------------------------------------------------------ persistence
    async def _persist(self, user_id: str, conv_id: str, outcome: RunOutcome, started: float) -> None:
        elapsed = time.perf_counter() - started
        await self.tasks.finish(
            outcome.task_id, outcome.status, outcome.answer, outcome.results, outcome.flags, int(elapsed * 1000)
        )
        await self.conversations.add_message(conv_id, "assistant", outcome.answer, outcome.task_id)
        await self.memory.push_turn(conv_id, "assistant", outcome.answer)
        TASKS.labels(outcome.status.value).inc()
        TASK_SECONDS.observe(elapsed)
        event_type = ev.TASK_FAILED if outcome.status == TaskStatus.FAILED else ev.TASK_COMPLETED
        await self.bus.publish(
            Event(
                type=event_type,
                user_id=user_id,
                task_id=outcome.task_id,
                data={"status": outcome.status.value, "ms": int(elapsed * 1000)},
            )
        )
        await self.audit.log("task.finished", user_id, outcome.task_id, {"status": outcome.status.value})
