"""Embeddings and the Qdrant-backed vector store.

Two embedders are provided:
- HashingEmbedder: offline, deterministic, no API key. It captures word overlap (lexical
  similarity), not meaning. Good for getting started and for tests.
- OpenAIEmbedder: semantic embeddings from any OpenAI-compatible /embeddings endpoint.

Switching embedders changes vector size; the collections are recreated with a new suffix.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import uuid
from typing import Any, Protocol

import httpx
from qdrant_client import QdrantClient, models

from ai_eos.config import Settings

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have i in is it its me my of on or our so that the this to was "
    "we what when where which who will with you your do does did how can".split()
)


class Embedder(Protocol):
    dim: int
    name: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    @staticmethod
    def _stem(token: str) -> str:
        for suffix in ("ing", "ed", "es", "s"):
            if len(token) > len(suffix) + 3 and token.endswith(suffix):
                return token[: -len(suffix)]
        return token

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = [self._stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        for feat in features:
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class OpenAIEmbedder:  # pragma: no cover - requires a live API
    name = "openai"

    def __init__(self, model: str, base_url: str, api_key: str, dim: int) -> None:
        self.model, self.base_url, self.api_key, self.dim = model, base_url.rstrip("/"), api_key, dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts, "dimensions": self.dim},
            )
            r.raise_for_status()
            return [d["embedding"] for d in r.json()["data"]]


def make_embedder(settings: Settings) -> Embedder:
    v = settings.vector
    if v.embedding_provider == "openai":  # pragma: no cover
        p = settings.llm.providers.get("openai")
        return OpenAIEmbedder(
            v.embedding_model,
            p.base_url if p else "https://api.openai.com/v1",
            os.environ.get(p.api_key_env if p else "OPENAI_API_KEY", ""),
            v.embedding_dim,
        )
    return HashingEmbedder(v.embedding_dim)


def _point_id(raw: str) -> str:
    try:
        return str(uuid.UUID(raw))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


class VectorStore:
    MEMORY = "memories"
    KNOWLEDGE = "knowledge"

    def __init__(self, client: QdrantClient, embedder: Embedder) -> None:
        self.client = client
        self.embedder = embedder
        self._ready: set[str] = set()

    @classmethod
    def from_settings(cls, settings: Settings) -> VectorStore:
        v = settings.vector
        if v.qdrant_url:  # pragma: no cover - needs a Qdrant server
            client = QdrantClient(url=v.qdrant_url, timeout=30)
        elif v.persist_local:
            os.makedirs(v.qdrant_path, exist_ok=True)
            client = QdrantClient(path=v.qdrant_path)
        else:
            client = QdrantClient(location=":memory:")
        return cls(client, make_embedder(settings))

    def _name(self, base: str) -> str:
        return f"{base}_{self.embedder.name}_{self.embedder.dim}"

    def _ensure(self, base: str) -> str:
        name = self._name(base)
        if name not in self._ready:
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    name, vectors_config=models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE)
                )
            self._ready.add(name)
        return name

    async def upsert(self, base: str, items: list[tuple[str, str, dict[str, Any]]]) -> None:
        """items: (id, text, payload). Payload must include user_id."""
        if not items:
            return
        vectors = await self.embedder.embed([t for _, t, _ in items])
        name = self._ensure(base)
        points = [
            models.PointStruct(id=_point_id(i), vector=vec, payload={**payload, "text": text, "ref": i})
            for (i, text, payload), vec in zip(items, vectors, strict=True)
        ]
        await asyncio.to_thread(self.client.upsert, name, points=points)

    async def search(
        self, base: str, query: str, user_id: str, top_k: int, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        name = self._ensure(base)
        [vector] = await self.embedder.embed([query])
        must = [models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
        for k, v in (filters or {}).items():
            must.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
        res = await asyncio.to_thread(
            self.client.query_points, name, query=vector, limit=top_k, query_filter=models.Filter(must=must)
        )
        return [{**(p.payload or {}), "score": round(float(p.score), 4)} for p in res.points]

    async def delete(self, base: str, ids: list[str]) -> None:
        name = self._ensure(base)
        await asyncio.to_thread(
            self.client.delete, name, points_selector=models.PointIdsList(points=[_point_id(i) for i in ids])
        )

    async def delete_where(self, base: str, key: str, value: str) -> None:
        name = self._ensure(base)
        flt = models.Filter(must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))])
        await asyncio.to_thread(self.client.delete, name, points_selector=models.FilterSelector(filter=flt))

    def ping(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001  # pragma: no cover
            return False
