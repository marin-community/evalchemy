import argparse

import pytest

from eval.judges.config import JudgeConfig, JudgeConfigurationError


def test_gpt55_preset_resolves_openai_responses():
    config = JudgeConfig.from_model("gpt-5.5")

    assert config.provider == "openai"
    assert config.model == "gpt-5.5"
    assert config.api_surface == "responses"
    assert config.api_key_env == "OPENAI_API_KEY"
    assert config.reasoning_effort == "medium"


def test_deepseek_preset_resolves_openai_compatible_chat():
    config = JudgeConfig.from_model("deepseek-v4-pro")

    assert config.provider == "deepseek"
    assert config.model == "deepseek-v4-pro"
    assert config.api_surface == "chat_completions"
    assert config.api_key_env == "DEEPSEEK_API_KEY"
    assert config.base_url == "https://api.deepseek.com"
    assert config.reasoning_effort == "high"


def test_auto_annotator_without_overrides_preserves_current_behavior():
    args = argparse.Namespace(annotator_model="auto")

    assert JudgeConfig.from_args(args) is None


def test_legacy_flat_annotator_model_does_not_create_shared_judge_config():
    args = argparse.Namespace(annotator_model="gpt-4o-mini-2024-07-18")

    assert JudgeConfig.from_args(args) is None


def test_provider_override_can_select_deepseek_without_model_flag():
    args = argparse.Namespace(
        annotator_model="auto",
        annotator_provider="deepseek",
        annotator_api_surface=None,
        annotator_api_key_env=None,
        annotator_base_url=None,
        annotator_reasoning_effort=None,
        annotator_max_output_tokens=None,
        annotator_num_workers=None,
        annotator_cache_path=None,
    )

    config = JudgeConfig.from_args(args)

    assert config.model == "deepseek-v4-pro"
    assert config.provider == "deepseek"


def test_explicit_overrides_beat_presets():
    config = JudgeConfig.from_model(
        "gpt-5.5",
        reasoning_effort="high",
        max_output_tokens=8192,
        num_workers=4,
    )

    assert config.reasoning_effort == "high"
    assert config.max_output_tokens == 8192
    assert config.num_workers == 4


def test_validate_env_checks_only_configured_env_var(monkeypatch):
    config = JudgeConfig.from_model("deepseek-v4-pro")

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    with pytest.raises(JudgeConfigurationError, match="DEEPSEEK_API_KEY"):
        config.validate_env()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    assert config.validate_env() == "sk-deepseek"


def test_redacted_dict_and_hash_do_not_include_secret_value(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    config = JudgeConfig.from_model("gpt-5.5")

    redacted = config.redacted_dict()

    assert "sk-secret-value" not in repr(redacted)
    assert redacted["api_key_env"] == "OPENAI_API_KEY"
    assert config.config_hash().startswith("sha256:")


def test_nested_yaml_supports_non_cli_judge_knobs():
    raw = {
        "annotator": {
            "model": "gpt-5.5",
            "temperature": 0.0,
            "timeout": 120,
            "max_retries": 3,
        }
    }

    config = JudgeConfig.from_yaml(raw)

    assert config.model == "gpt-5.5"
    assert config.temperature == 0.0
    assert config.timeout == 120
    assert config.max_retries == 3
