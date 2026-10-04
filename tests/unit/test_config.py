from pathlib import Path

import pytest

from ai_eos.config import load_dotenv, load_settings
from tests.conftest import ROOT


def test_yaml_and_agents_load() -> None:
    s = load_settings(ROOT / "config/settings.yaml", ROOT / "config/agents.yaml", environ={})
    assert s.llm.default_provider == "offline"
    assert set(s.agents.specialists) == {
        "executive_assistant",
        "research_analytics",
        "operations",
        "software_engineer",
        "devops_engineer",
    }
    assert "anthropic" in s.llm.providers and s.llm.providers["anthropic"].kind == "anthropic"


def test_env_overrides_are_nested_and_typed() -> None:
    s = load_settings(
        ROOT / "config/settings.yaml",
        ROOT / "config/agents.yaml",
        environ={
            "EOS__ORCHESTRATION__MAX_SUBTASKS": "3",
            "EOS__SECURITY__BLOCK_ON_INJECTION": "true",
            "EOS__APP__CORS_ORIGINS": '["https://x.example"]',
            "EOS__REDIS__URL": "redis://r:6379/0",
            "EOS__": "ignored",
            "OTHER": "ignored",
        },
    )
    assert s.orchestration.max_subtasks == 3
    assert s.security.block_on_injection is True
    assert s.app.cors_origins == ["https://x.example"]
    assert s.redis.url == "redis://r:6379/0"


def test_missing_files_use_defaults(tmp_path: Path) -> None:
    s = load_settings(tmp_path / "nope.yaml", tmp_path / "nope2.yaml", environ={})
    assert s.database.url.startswith("sqlite")
    assert s.agents.specialists == {}


def test_production_requires_strong_secrets() -> None:
    env = {"EOS__APP__ENVIRONMENT": "production", "EOS_JWT_SECRET": "short"}
    with pytest.raises(ValueError, match="JWT"):
        load_settings(ROOT / "config/settings.yaml", ROOT / "config/agents.yaml", environ=env)
    env["EOS_JWT_SECRET"] = "a" * 40
    with pytest.raises(ValueError, match="ENCRYPTION"):
        load_settings(ROOT / "config/settings.yaml", ROOT / "config/agents.yaml", environ=env)
    env["EOS_ENCRYPTION_KEY"] = "k"
    s = load_settings(ROOT / "config/settings.yaml", ROOT / "config/agents.yaml", environ=env)
    assert s.is_production


def test_unknown_default_provider_rejected() -> None:
    with pytest.raises(ValueError, match="default_provider"):
        load_settings(
            ROOT / "config/settings.yaml",
            ROOT / "config/agents.yaml",
            environ={"EOS__LLM__DEFAULT_PROVIDER": "nonexistent"},
        )


def test_dotenv_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("# comment\nEOS_TEST_A='one'\nEOS_TEST_B=two\nbadline\n\nEOS_TEST_C=\"three\"\n")
    monkeypatch.setenv("EOS_TEST_B", "preset")
    monkeypatch.delenv("EOS_TEST_A", raising=False)
    monkeypatch.delenv("EOS_TEST_C", raising=False)
    load_dotenv(env)
    import os

    assert os.environ["EOS_TEST_A"] == "one"
    assert os.environ["EOS_TEST_B"] == "preset"
    assert os.environ["EOS_TEST_C"] == "three"
    load_dotenv(tmp_path / "missing")  # no error
