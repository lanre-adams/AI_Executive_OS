# AI-EOS — AI Executive Operating System

You talk to one agent, the **Chief of Staff**. It understands what you want, breaks the work down,
briefs a team of five specialist agents, checks their work, sends weak work back for revision,
merges the results into one answer, asks your approval before anything touches the outside world,
and remembers what it learns about you.

```
                          You
                           │   web dashboard · CLI · REST API
                    Chief of Staff  ── plans · delegates · reviews · synthesises · remembers
   ┌──────────────┬────────┴──────┬──────────────┬─────────────────┐
Executive      Research &      Operations     Software          DevOps
Assistant      Analytics                      Engineer          Engineer
Gmail·Calendar Web·Python·RAG  Azure DevOps   GitHub·Python     GitHub Actions
Drive          Knowledge base  GitHub issues  Code search       Azure Pipelines
```

## Quick start (Docker, about 10 minutes)

```bash
git clone <your-repo-url> ai-eos && cd ai-eos
cp .env.example .env
make secrets            # paste the two lines it prints into .env
# set EOS_BOOTSTRAP_ADMIN_PASSWORD in .env (12+ chars, upper+lower case, a digit)
docker compose up -d --build
open http://localhost:8000          # sign in with EOS_BOOTSTRAP_ADMIN_EMAIL / PASSWORD
```

It runs immediately with the built-in **offline** model so you can check every screen without an
API key. Its answers are templated; set a real provider (e.g. `EOS__LLM__DEFAULT_PROVIDER=anthropic`
plus `ANTHROPIC_API_KEY`) for real work. The full beginner guide is
[docs/guides/installation.md](docs/guides/installation.md).

## What's in the box

| Area | Implementation |
|---|---|
| Orchestration | LangGraph state machine: analyse → clarify/answer/delegate → parallel execution by dependency waves → review → bounded revision → synthesis → reflection |
| Agents | Chief of Staff + 5 specialists, defined in `config/agents.yaml` with prompts in `src/ai_eos/agents/prompts/` (editable live in the dashboard) |
| Models | GPT, Claude, Gemini, DeepSeek, Mistral, Llama (Ollama) and any OpenAI-compatible endpoint; per-user and per-agent selection |
| Memory | Short-term (Redis), conversation history and task memory (PostgreSQL), long-term facts/preferences/reflections and knowledge base (Qdrant vectors) |
| Tools | 24 plug-in tools: web search, URL fetch, sandboxed Python, sandboxed files, memory and knowledge search, GitHub, Azure DevOps, Gmail, Google Calendar, Google Drive |
| Safety | Human approval for outward actions, prompt-injection screening, untrusted-content fencing, SSRF protection, sandboxing, RBAC, JWT + API keys, encrypted secrets, audit log, rate limiting |
| Interfaces | Web dashboard (chat, live progress, task board, approvals, agents, memory explorer, knowledge base, history, metrics, prompt editor, logs, settings, dark mode, notifications), `eos` CLI, REST API with OpenAPI docs at `/docs` |
| Operations | Docker Compose (app, PostgreSQL, Redis, Qdrant, optional Prometheus + Grafana), health checks, Prometheus metrics, JSON logs with request IDs, backup/restore scripts, GitHub Actions CI |
| Quality | 99 automated tests (unit + integration), 98% line coverage, ruff-clean |

## Documentation

| Guide | |
|---|---|
| [Installation](docs/guides/installation.md) | Beginner, step by step: Windows, macOS, Linux |
| [User guide](docs/guides/user-guide.md) | Working with your Chief of Staff, example workflows and prompts |
| [Integrations](docs/guides/integrations.md) | Model providers, Google Workspace, GitHub, Azure DevOps |
| [Administrator guide](docs/guides/admin-guide.md) | Users, roles, prompts, secrets, configuration reference |
| [Security guide](docs/guides/security.md) | Threat model and controls |
| [Operations guide](docs/guides/operations.md) | Monitoring, backup and recovery, upgrades, scaling, performance |
| [Troubleshooting](docs/guides/troubleshooting.md) | Common problems and fixes |
| [Developer guide](docs/guides/developer-guide.md) | Architecture, adding agents and tools, testing |
| [API reference](docs/api.md) | Endpoints (live, interactive docs at `/docs`) |
| [Architecture](docs/architecture.md) · [Word document](docs/AI-EOS_Architecture_and_Delivery.docx) | Diagrams, framework comparison, delivery report |
| [Architecture decisions](docs/adr/) | Why things are the way they are |

## Scope of this release (v0.1)

Built and tested: everything in the table above, for **local Docker deployment**, with **Google
Workspace, GitHub and Azure DevOps** integrations. Not yet built (planned phases): Microsoft 365 /
Outlook / Teams, Slack, Notion, Jira and other PM tools; Kubernetes manifests, Terraform and cloud
deployment; OpenTelemetry tracing; database migrations tooling. The plug-in architecture is designed
so these slot in without changing the core.
