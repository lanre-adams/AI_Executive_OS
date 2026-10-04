"""Configuration loading.

Precedence (highest first):
  1. Environment variables  EOS__SECTION__KEY=value  (nested with double underscores)
  2. YAML file at $EOS_CONFIG (default ./config/settings.yaml)
  3. Defaults declared on the models below

Secrets are read only from the environment (or .env) and never from YAML.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class AppCfg(BaseModel):
    name: str = "AI-EOS"
    environment: str = "development"
    log_level: str = "INFO"
    log_json: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8000"])


class DatabaseCfg(BaseModel):
    url: str = "sqlite+aiosqlite:///./data/eos.db"
    echo: bool = False


class RedisCfg(BaseModel):
    url: str = ""


class VectorCfg(BaseModel):
    backend: str = "qdrant"
    qdrant_url: str = ""
    qdrant_path: str = "./data/qdrant"
    persist_local: bool = True
    embedding_provider: str = "hashing"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 384


class ProviderCfg(BaseModel):
    kind: str
    model: str
    base_url: str = ""
    api_key_env: str = ""


class LLMCfg(BaseModel):
    default_provider: str = "offline"
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_seconds: int = 120
    providers: dict[str, ProviderCfg] = Field(
        default_factory=lambda: {"offline": ProviderCfg(kind="offline", model="offline-1")}
    )


class OrchestrationCfg(BaseModel):
    max_subtasks: int = 6
    max_parallel_agents: int = 3
    max_agent_steps: int = 6
    max_revisions: int = 1
    validation_threshold: float = 0.6
    history_turns: int = 10


class MemoryCfg(BaseModel):
    short_term_ttl_seconds: int = 86400
    semantic_top_k: int = 5
    knowledge_top_k: int = 5
    chunk_size: int = 900
    chunk_overlap: int = 150


class SecurityCfg(BaseModel):
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60
    rate_limit_per_minute: int = 60
    max_input_chars: int = 20000
    prompt_guard: bool = True
    block_on_injection: bool = False


class WebSearchCfg(BaseModel):
    provider: str = "duckduckgo"
    api_key_env: str = "TAVILY_API_KEY"


class GitHubCfg(BaseModel):
    api_url: str = "https://api.github.com"
    token_env: str = "GITHUB_TOKEN"


class AzureDevOpsCfg(BaseModel):
    organization_env: str = "AZDO_ORG"
    token_env: str = "AZDO_PAT"


class GoogleCfg(BaseModel):
    access_token_env: str = "GOOGLE_OAUTH_ACCESS_TOKEN"
    refresh_token_env: str = "GOOGLE_OAUTH_REFRESH_TOKEN"
    client_id_env: str = "GOOGLE_OAUTH_CLIENT_ID"
    client_secret_env: str = "GOOGLE_OAUTH_CLIENT_SECRET"


class ToolsCfg(BaseModel):
    sandbox_root: str = "./data/workspace"
    python_timeout_seconds: int = 20
    http_timeout_seconds: int = 30
    web_search: WebSearchCfg = Field(default_factory=WebSearchCfg)
    github: GitHubCfg = Field(default_factory=GitHubCfg)
    azure_devops: AzureDevOpsCfg = Field(default_factory=AzureDevOpsCfg)
    google: GoogleCfg = Field(default_factory=GoogleCfg)
    # Tools that normally need human approval but which you choose to auto-approve.
    auto_approve: list[str] = Field(default_factory=list)
    # Extra tool modules to load; each must define register(registry, settings).
    plugins: list[str] = Field(default_factory=list)


class AgentDef(BaseModel):
    title: str
    prompt: str
    description: str = ""
    tools: list[str] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None


class AgentsCfg(BaseModel):
    chief_of_staff: AgentDef = Field(default_factory=lambda: AgentDef(title="Chief of Staff", prompt="chief_of_staff"))
    specialists: dict[str, AgentDef] = Field(default_factory=dict)


class Secrets(BaseModel):
    jwt_secret: str = ""
    encryption_key: str = ""
    bootstrap_admin_email: str = ""
    bootstrap_admin_password: str = ""


class Settings(BaseModel):
    app: AppCfg = Field(default_factory=AppCfg)
    database: DatabaseCfg = Field(default_factory=DatabaseCfg)
    redis: RedisCfg = Field(default_factory=RedisCfg)
    vector: VectorCfg = Field(default_factory=VectorCfg)
    llm: LLMCfg = Field(default_factory=LLMCfg)
    orchestration: OrchestrationCfg = Field(default_factory=OrchestrationCfg)
    memory: MemoryCfg = Field(default_factory=MemoryCfg)
    security: SecurityCfg = Field(default_factory=SecurityCfg)
    tools: ToolsCfg = Field(default_factory=ToolsCfg)
    agents: AgentsCfg = Field(default_factory=AgentsCfg)
    secrets: Secrets = Field(default_factory=Secrets)

    @property
    def is_production(self) -> bool:
        return self.app.environment.lower() == "production"


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing environment variables win."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _coerce(value: str) -> Any:
    """Turn env strings into JSON types where possible ("true", "5", "[...]")."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _apply_env_overrides(data: dict[str, Any], environ: dict[str, str]) -> dict[str, Any]:
    for key, value in environ.items():
        if not key.startswith("EOS__"):
            continue
        parts = [p.lower() for p in key[5:].split("__") if p]
        if not parts:
            continue
        node = data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                break
        else:
            node[parts[-1]] = _coerce(value)
    return data


def load_settings(
    config_path: str | Path | None = None,
    agents_path: str | Path | None = None,
    environ: dict[str, str] | None = None,
) -> Settings:
    env = dict(os.environ if environ is None else environ)
    cfg_file = Path(config_path or env.get("EOS_CONFIG", "config/settings.yaml"))
    agents_file = Path(agents_path or env.get("EOS_AGENTS_CONFIG", "config/agents.yaml"))

    data: dict[str, Any] = {}
    if cfg_file.is_file():
        data = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    if agents_file.is_file():
        data["agents"] = yaml.safe_load(agents_file.read_text(encoding="utf-8")) or {}

    data = _apply_env_overrides(data, env)
    data["secrets"] = {
        "jwt_secret": env.get("EOS_JWT_SECRET", ""),
        "encryption_key": env.get("EOS_ENCRYPTION_KEY", ""),
        "bootstrap_admin_email": env.get("EOS_BOOTSTRAP_ADMIN_EMAIL", ""),
        "bootstrap_admin_password": env.get("EOS_BOOTSTRAP_ADMIN_PASSWORD", ""),
    }
    settings = Settings.model_validate(data)
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    """Fail fast on configurations that are unsafe in production."""
    if settings.is_production:
        if len(settings.secrets.jwt_secret) < 32 or "change-me" in settings.secrets.jwt_secret:
            raise ValueError("EOS_JWT_SECRET must be a random string of at least 32 characters in production")
        if not settings.secrets.encryption_key:
            raise ValueError("EOS_ENCRYPTION_KEY is required in production")
    if settings.llm.default_provider not in settings.llm.providers:
        raise ValueError(f"llm.default_provider '{settings.llm.default_provider}' is not defined in llm.providers")


@lru_cache(maxsize=1)
def get_settings() -> Settings:  # pragma: no cover - thin cached wrapper
    load_dotenv()
    return load_settings()
