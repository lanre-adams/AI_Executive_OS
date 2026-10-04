"""Unified memory service.

| Layer               | Store                       | Lifetime            |
|---------------------|-----------------------------|---------------------|
| Short-term/working  | Redis list per conversation | TTL (default 24h)   |
| Conversation history| PostgreSQL `messages`       | Permanent           |
| Task memory         | PostgreSQL `tasks/subtasks` | Permanent           |
| Long-term facts     | PostgreSQL `memories` + Qdrant `memories` | Permanent, user-deletable |
| Reflection memory   | same, kind=reflection       | Permanent           |
| Semantic recall     | Qdrant vector search        | Mirrors the above   |
| Knowledge base      | Qdrant `knowledge` + `knowledge_docs` | Until document deleted |
"""

from __future__ import annotations

import io
from typing import Any

from ai_eos.config import MemoryCfg
from ai_eos.domain.models import MemoryItem, MemoryKind, new_id
from ai_eos.infrastructure.kv import KeyValueStore
from ai_eos.infrastructure.repositories import MemoryRepo
from ai_eos.infrastructure.vector import VectorStore


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Split on paragraph boundaries, falling back to hard splits for very long paragraphs."""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        while len(para) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:size])
            para = para[size - overlap :]
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if current and overlap else ""
            current = f"{tail}\n\n{para}" if tail else para
    if current:
        chunks.append(current)
    return chunks


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if name.endswith((".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml", ".py", ".log", ".html")):
        return data.decode("utf-8", errors="replace")
    raise ValueError("supported types: .pdf .txt .md .csv .json .yaml .py .log .html")


class MemoryManager:
    def __init__(self, repo: MemoryRepo, vectors: VectorStore, kv: KeyValueStore, cfg: MemoryCfg) -> None:
        self.repo, self.vectors, self.kv, self.cfg = repo, vectors, kv, cfg

    # --- short-term -------------------------------------------------------------
    async def push_turn(self, conversation_id: str, role: str, content: str) -> None:
        await self.kv.push(
            f"stm:{conversation_id}",
            {"role": role, "content": content[:4000]},
            max_len=40,
            ttl=self.cfg.short_term_ttl_seconds,
        )

    async def recent_turns(self, conversation_id: str, count: int) -> list[dict[str, Any]]:
        return await self.kv.range(f"stm:{conversation_id}", count)

    # --- long-term --------------------------------------------------------------
    async def remember(
        self, user_id: str, kind: MemoryKind, content: str, metadata: dict[str, Any] | None = None
    ) -> MemoryItem:
        item = MemoryItem(user_id=user_id, kind=kind, content=content.strip(), metadata=metadata or {})
        await self.repo.add(item)
        await self.vectors.upsert(
            VectorStore.MEMORY, [(item.id, item.content, {"user_id": user_id, "kind": kind.value})]
        )
        return item

    async def recall(
        self, user_id: str, query: str, kinds: list[MemoryKind] | None = None, top_k: int | None = None
    ) -> list[MemoryItem]:
        k = top_k or self.cfg.semantic_top_k
        hits: list[dict[str, Any]] = []
        if kinds:
            for kind in kinds:
                hits += await self.vectors.search(VectorStore.MEMORY, query, user_id, k, {"kind": kind.value})
            hits.sort(key=lambda h: h["score"], reverse=True)
            hits = hits[:k]
        else:
            hits = await self.vectors.search(VectorStore.MEMORY, query, user_id, k)
        return [
            MemoryItem(id=h["ref"], user_id=user_id, kind=MemoryKind(h["kind"]), content=h["text"], score=h["score"])
            for h in hits
        ]

    async def list(self, user_id: str, kind: MemoryKind | None = None) -> list[MemoryItem]:
        return await self.repo.list(user_id, kind)

    async def forget(self, user_id: str, memory_id: str) -> bool:
        ok = await self.repo.delete(user_id, memory_id)
        if ok:
            await self.vectors.delete(VectorStore.MEMORY, [memory_id])
        return ok

    async def context_for(self, user_id: str, query: str) -> str:
        """Memory block injected into the Chief of Staff's planning prompt."""
        items = await self.recall(user_id, query)
        relevant = [i for i in items if (i.score or 0) > 0.15]
        if not relevant:
            return ""
        return "\n".join(f"- [{i.kind.value}] {i.content}" for i in relevant)

    # --- knowledge base ---------------------------------------------------------
    async def ingest(self, user_id: str, title: str, text: str, source: str = "") -> dict[str, Any]:
        chunks = chunk_text(text, self.cfg.chunk_size, self.cfg.chunk_overlap)
        if not chunks:
            raise ValueError("document has no extractable text")
        doc_id = await self.repo.add_doc(user_id, title, source, len(chunks))
        await self.vectors.upsert(
            VectorStore.KNOWLEDGE,
            [
                (new_id(), c, {"user_id": user_id, "doc_id": doc_id, "title": title, "chunk": n})
                for n, c in enumerate(chunks)
            ],
        )
        return {"id": doc_id, "title": title, "chunks": len(chunks)}

    async def search_knowledge(self, user_id: str, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        hits = await self.vectors.search(VectorStore.KNOWLEDGE, query, user_id, top_k or self.cfg.knowledge_top_k)
        return [
            {"title": h.get("title"), "chunk": h.get("chunk"), "score": h["score"], "text": h["text"]} for h in hits
        ]

    async def delete_doc(self, user_id: str, doc_id: str) -> bool:
        ok = await self.repo.delete_doc(user_id, doc_id)
        if ok:
            await self.vectors.delete_where(VectorStore.KNOWLEDGE, "doc_id", doc_id)
        return ok

    async def list_docs(self, user_id: str) -> list[dict[str, Any]]:
        return await self.repo.list_docs(user_id)
