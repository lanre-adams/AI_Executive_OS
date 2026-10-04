import asyncio

import pytest

from ai_eos.domain.events import Event, EventBus
from ai_eos.domain.models import MemoryKind
from ai_eos.infrastructure.kv import InMemoryKV, make_kv
from ai_eos.memory.manager import chunk_text, extract_text


def test_chunk_text() -> None:
    assert chunk_text("   ") == []
    assert chunk_text("one\n\ntwo") == ["one\n\ntwo"]
    long_para = "a" * 2500
    chunks = chunk_text(f"intro\n\n{long_para}\n\noutro", size=1000, overlap=100)
    assert all(len(c) <= 1000 for c in chunks)
    assert chunks[0] == "intro" and chunks[-1].endswith("outro")
    paras = "\n\n".join(f"para {i} " + "w" * 300 for i in range(6))
    c2 = chunk_text(paras, size=700, overlap=50)
    assert len(c2) > 2 and all(len(c) <= 760 for c in c2)


def test_extract_text() -> None:
    assert extract_text("a.md", b"# hi") == "# hi"
    with pytest.raises(ValueError):
        extract_text("a.exe", b"x")
    from io import BytesIO

    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(100, 100)
    buf = BytesIO()
    w.write(buf)
    assert extract_text("blank.pdf", buf.getvalue()).strip() == ""


async def test_in_memory_kv(monkeypatch: pytest.MonkeyPatch) -> None:
    kv = make_kv("")
    assert isinstance(kv, InMemoryKV) and await kv.ping()
    for i in range(5):
        await kv.push("l", {"i": i}, max_len=3, ttl=60)
    assert [x["i"] for x in await kv.range("l", 10)] == [2, 3, 4]
    await kv.set("k", "v", 60)
    assert await kv.get("k") == "v"
    assert await kv.incr_window("c", 60) == 1 and await kv.incr_window("c", 60) == 2
    await kv.delete("k")
    assert await kv.get("k") is None
    # expiry
    import ai_eos.infrastructure.kv as kvmod

    now = kvmod.time.monotonic()
    monkeypatch.setattr(kvmod.time, "monotonic", lambda: now + 3600)
    assert await kv.range("l", 10) == []
    assert await kv.incr_window("c", 60) == 1
    await kv.push("l", {"i": 9}, max_len=3, ttl=0)
    assert await kv.range("l", 10) == [{"i": 9}]


async def test_memory_manager(container, admin) -> None:
    mm, uid = container.memory, admin["id"]
    await mm.push_turn("conv", "user", "hi")
    assert await mm.recent_turns("conv", 5) == [{"role": "user", "content": "hi"}]

    fact = await mm.remember(uid, MemoryKind.FACT, "The finance team closes the books on the 5th")
    await mm.remember(uid, MemoryKind.PREFERENCE, "Prefers briefings as tables")
    hits = await mm.recall(uid, "when does finance close the books")
    assert hits and hits[0].id == fact.id
    prefs = await mm.recall(uid, "briefings tables", kinds=[MemoryKind.PREFERENCE, MemoryKind.FACT])
    assert prefs[0].kind == MemoryKind.PREFERENCE
    assert "finance" in await mm.context_for(uid, "finance books close")
    assert await mm.context_for(uid, "zzzz qqqq") == ""
    assert len(await mm.list(uid)) == 2 and len(await mm.list(uid, MemoryKind.FACT)) == 1
    assert await mm.forget(uid, fact.id) and not await mm.forget(uid, fact.id)
    assert all(h.id != fact.id for h in await mm.recall(uid, "finance books"))

    doc = await mm.ingest(uid, "Travel policy", "Economy class for flights under 6 hours.\n\nBusiness class above.")
    assert doc["chunks"] == 1
    assert (await mm.search_knowledge(uid, "flight class policy"))[0]["title"] == "Travel policy"
    assert (await mm.search_knowledge("someone-else", "flight class policy")) == []
    assert len(await mm.list_docs(uid)) == 1
    assert await mm.delete_doc(uid, doc["id"]) and not await mm.delete_doc(uid, doc["id"])
    assert await mm.search_knowledge(uid, "flight class policy") == []
    with pytest.raises(ValueError):
        await mm.ingest(uid, "empty", "  ")


async def test_event_bus() -> None:
    bus = EventBus(history=3)
    async with bus.subscribe() as q:
        assert bus.subscriber_count == 1
        await bus.publish(Event(type="a", user_id="u1"))
        await bus.publish(Event(type="b", user_id="u2"))
        await bus.publish(Event(type="c"))
        assert (await asyncio.wait_for(q.get(), 1)).type == "a"
    assert bus.subscriber_count == 0
    assert [e.type for e in bus.recent("u1")] == ["a", "c"]
    for i in range(5):
        await bus.publish(Event(type=str(i)))
    assert len(bus.recent()) == 3
