# Changelog

## 0.1.0 — 2026-10-04

First release: local Docker deployment.

- Chief of Staff orchestration on LangGraph (analyse, clarify, delegate with dependencies, parallel
  execution, quality review with bounded revision, synthesis, reflection).
- Five specialists: Executive Assistant, Research & Analytics, Operations, Software Engineer, DevOps Engineer.
- Model providers: Anthropic, OpenAI, Gemini, DeepSeek, Mistral, Ollama/OpenAI-compatible, offline.
- Memory: short-term (Redis), conversations and tasks (PostgreSQL), long-term facts, preferences,
  episodes and reflections plus knowledge base (Qdrant).
- 24 tools incl. Google Workspace, GitHub, Azure DevOps, web, sandboxed Python and files.
- Human approval workflow, RBAC, JWT and API keys, encrypted secrets, audit log, rate limiting,
  prompt-injection defences, SSRF protection.
- Web dashboard, `eos` CLI, REST API with OpenAPI docs.
- Docker Compose with optional Prometheus/Grafana; backup and restore scripts; GitHub Actions CI.
- 99 tests, 98% coverage; verified on SQLite and on PostgreSQL 16 + Redis 7.
