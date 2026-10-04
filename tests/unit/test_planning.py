from ai_eos.domain.models import Priority, Subtask
from ai_eos.orchestration.planning import normalise_plan, waves

AGENTS = ["research_analytics", "operations", "executive_assistant"]


def test_normalise_filters_and_fixes() -> None:
    raw = {
        "intent": "x",
        "priority": "URGENT",
        "subtasks": [
            {"id": "a", "agent": "research_analytics", "instruction": "r", "depends_on": ["b", "ghost", "a"]},
            {"id": "b", "agent": "operations", "instruction": "o", "depends_on": ["a"]},  # cycle a<->b
            {"id": "a", "agent": "executive_assistant", "instruction": "dup id"},
            {"id": "c", "agent": "hacker", "instruction": "nope"},
            {"id": "d", "agent": "operations", "instruction": ""},
            "garbage",
        ],
    }
    plan = normalise_plan(raw, AGENTS, 5, "req")
    assert plan.priority == Priority.URGENT
    ids = [s.id for s in plan.subtasks]
    assert ids == ["a", "b", "a_"]
    a, b = plan.subtasks[0], plan.subtasks[1]
    assert "ghost" not in a.depends_on and "a" not in a.depends_on
    assert not ("b" in a.depends_on and "a" in b.depends_on)  # cycle broken


def test_normalise_limits_and_fallbacks() -> None:
    many = {"subtasks": [{"agent": "operations", "instruction": str(i)} for i in range(10)], "priority": "weird"}
    plan = normalise_plan(many, AGENTS, 3, "req")
    assert len(plan.subtasks) == 3 and plan.priority == Priority.NORMAL
    empty = normalise_plan({}, AGENTS, 3, "do the thing")
    assert empty.subtasks[0].agent == "research_analytics" and empty.subtasks[0].instruction == "do the thing"
    assert normalise_plan({}, ["operations"], 3, "x").subtasks[0].agent == "operations"
    clar = normalise_plan({"needs_clarification": True, "clarifying_question": "Which quarter?"}, AGENTS, 3, "x")
    assert clar.needs_clarification and not clar.subtasks
    no_q = normalise_plan({"needs_clarification": True}, AGENTS, 3, "x")
    assert not no_q.needs_clarification and no_q.subtasks
    direct = normalise_plan({"direct_answer": "42"}, AGENTS, 3, "x")
    assert direct.direct_answer == "42" and not direct.subtasks


def test_waves() -> None:
    subs = [
        Subtask(id="a", agent="x", instruction="."),
        Subtask(id="b", agent="x", instruction=".", depends_on=["a"]),
        Subtask(id="c", agent="x", instruction="."),
        Subtask(id="d", agent="x", instruction=".", depends_on=["b", "c"]),
    ]
    assert [[s.id for s in w] for w in waves(subs)] == [["a", "c"], ["b"], ["d"]]
    cyclic = [
        Subtask(id="a", agent="x", instruction=".", depends_on=["b"]),
        Subtask(id="b", agent="x", instruction=".", depends_on=["a"]),
    ]
    assert len(waves(cyclic)) == 1  # defensive path
