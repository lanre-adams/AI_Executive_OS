"""Pure planning helpers: plan normalisation and dependency scheduling (no I/O)."""

from __future__ import annotations

from typing import Any

from ai_eos.domain.models import Plan, Priority, Subtask


def normalise_plan(raw: dict[str, Any], agents: list[str], max_subtasks: int, request: str) -> Plan:
    """Coerce model output into a valid Plan: known agents, unique ids, valid deps, no cycles."""
    try:
        priority = Priority(str(raw.get("priority", "normal")).lower())
    except ValueError:
        priority = Priority.NORMAL

    subtasks: list[Subtask] = []
    seen: set[str] = set()
    for i, item in enumerate(raw.get("subtasks") or []):
        if not isinstance(item, dict):
            continue
        agent = str(item.get("agent", "")).strip()
        instruction = str(item.get("instruction", "")).strip()
        if agent not in agents or not instruction:
            continue
        sid = str(item.get("id") or f"t{i + 1}")
        while sid in seen:
            sid += "_"
        seen.add(sid)
        deps = item.get("depends_on") or []
        subtasks.append(
            Subtask(
                id=sid,
                agent=agent,
                instruction=instruction,
                depends_on=[str(d) for d in deps if isinstance(d, str | int)],
                expected_output=str(item.get("expected_output", "")),
            )
        )
        if len(subtasks) >= max_subtasks:
            break
    ids = {s.id for s in subtasks}
    for s in subtasks:
        s.depends_on = [d for d in s.depends_on if d in ids and d != s.id]
    _break_cycles(subtasks)

    plan = Plan(
        intent=str(raw.get("intent", ""))[:500],
        priority=priority,
        needs_clarification=bool(raw.get("needs_clarification")) and bool(raw.get("clarifying_question")),
        clarifying_question=str(raw.get("clarifying_question", "")),
        direct_answer=str(raw.get("direct_answer", "") or ""),
        subtasks=subtasks,
    )
    if not plan.needs_clarification and not plan.direct_answer and not plan.subtasks:
        # The planner produced nothing usable: fall back to a single generalist assignment.
        fallback = "research_analytics" if "research_analytics" in agents else agents[0]
        plan.subtasks = [Subtask(id="t1", agent=fallback, instruction=request)]
    return plan


def _break_cycles(subtasks: list[Subtask]) -> None:
    by_id = {s.id: s for s in subtasks}
    state: dict[str, int] = {}

    def visit(sid: str) -> None:
        state[sid] = 1
        node = by_id[sid]
        for dep in list(node.depends_on):
            if state.get(dep) == 1:
                node.depends_on.remove(dep)  # back-edge: drop it
            elif state.get(dep) is None:
                visit(dep)
        state[sid] = 2

    for s in subtasks:
        if s.id not in state:
            visit(s.id)


def waves(subtasks: list[Subtask]) -> list[list[Subtask]]:
    """Group subtasks into batches that can run in parallel, respecting depends_on."""
    remaining = {s.id: s for s in subtasks}
    done: set[str] = set()
    out: list[list[Subtask]] = []
    while remaining:
        ready = [s for s in remaining.values() if all(d not in remaining for d in s.depends_on)]
        if not ready:  # defensive; cycles are removed by normalise_plan
            ready = list(remaining.values())
        out.append(ready)
        for s in ready:
            done.add(s.id)
            remaining.pop(s.id)
    return out
