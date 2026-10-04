import json

import pytest

from ai_eos.agents.base import PromptLibrary
from ai_eos.domain.models import Subtask


def sub(agent: str = "research_analytics") -> Subtask:
    return Subtask(id="t1", agent=agent, instruction="Find the facts", expected_output="bullets")


async def test_tool_loop_then_final(container, admin, llm) -> None:
    await container.memory.remember(
        admin["id"],
        __import__("ai_eos.domain.models", fromlist=["MemoryKind"]).MemoryKind.FACT,
        "Revenue target is 2 billion naira",
    )
    llm.script["agent"] = [
        json.dumps({"action": "tool", "tool": "memory_search", "args": {"query": "revenue target"}}),
        "Here you go: ```json\n" + json.dumps({"action": "final", "content": "Target: 2bn naira"}) + "\n```",
    ]
    agent = container.agents["research_analytics"]
    res = await agent.run(
        sub(), container.tool_context(admin["id"], agent.key, None), "ctx", feedback="be specific", previous="old"
    )
    assert res.ok and res.output == "Target: 2bn naira"
    assert res.tool_calls[0].tool == "memory_search" and res.tool_calls[0].ok
    first_prompt = llm.calls[0].messages[0].content
    assert (
        "YOUR ASSIGNMENT" in first_prompt and "REVIEWER FEEDBACK" in first_prompt and "EXPECTED OUTPUT" in first_prompt
    )
    assert "TOOL RESULT (memory_search, ok=True)" in llm.calls[1].messages[-1].content
    sys = llm.calls[0].system
    assert "Research & Analytics" in sys and "memory_search(" in sys and "untrusted" in sys


async def test_prose_answer_accepted(container, admin, llm) -> None:
    llm.script["agent"] = "Plain prose answer without JSON."
    agent = container.agents["operations"]
    res = await agent.run(sub("operations"), container.tool_context(admin["id"], agent.key, None))
    assert res.output == "Plain prose answer without JSON."


async def test_budget_exhaustion_forces_final(container, admin, llm) -> None:
    container.settings.orchestration.max_agent_steps = 1
    call = json.dumps({"action": "tool", "tool": "list_files", "args": {}})
    llm.script["agent"] = [call, call, json.dumps({"action": "final", "content": {"summary": "structured"}})]
    agent = container.agents["software_engineer"]
    res = await agent.run(sub("software_engineer"), container.tool_context(admin["id"], agent.key, None))
    assert json.loads(res.output) == {"summary": "structured"}
    assert len(res.tool_calls) == 1
    llm.script["agent"] = [call, call, "fine, prose final"]
    res = await agent.run(sub("software_engineer"), container.tool_context(admin["id"], agent.key, None))
    assert res.output == "fine, prose final"


async def test_disallowed_tool_and_approval_tracking(container, admin, llm) -> None:
    llm.script["agent"] = [
        json.dumps({"action": "tool", "tool": "gmail_send", "args": {"to": "x"}}),  # not allowed for research
        json.dumps({"action": "tool", "tool": "python_exec", "args": {"code": "print(1)"}}),  # needs approval
        json.dumps({"action": "final", "content": "done", "extra": 1}),
    ]
    agent = container.agents["research_analytics"]
    res = await agent.run(sub(), container.tool_context(admin["id"], agent.key, "task"))
    assert not res.tool_calls[0].ok and res.tool_calls[1].approval_id
    assert res.pending_approvals == [res.tool_calls[1].approval_id]


async def test_agent_error_is_captured(container, admin, llm) -> None:
    def boom(request):  # noqa: ANN001, ANN202
        raise RuntimeError("model down")

    llm.script["agent"] = boom
    agent = container.agents["devops_engineer"]
    res = await agent.run(sub("devops_engineer"), container.tool_context(admin["id"], agent.key, None))
    assert not res.ok and "model down" in res.error
    assert agent.describe()["provider"] == "offline"


async def test_prompt_library(container, admin) -> None:
    lib: PromptLibrary = container.prompts
    assert "chief_of_staff" in lib.names() and "_tool_protocol" in lib.names()
    with pytest.raises(KeyError):
        lib.default("nope")
    await lib.set("operations", "# Custom ops prompt", admin["id"])
    assert lib.get("operations") == "# Custom ops prompt" and lib.is_overridden("operations")
    fresh = PromptLibrary(container.prompt_repo)
    await fresh.refresh()
    assert fresh.get("operations") == "# Custom ops prompt"
    await lib.reset("operations")
    assert not lib.is_overridden("operations") and "Role: Operations" in lib.get("operations")
    with pytest.raises(KeyError):
        await lib.set("nope", "x", admin["id"])
