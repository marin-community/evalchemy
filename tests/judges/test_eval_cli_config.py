import argparse

from eval.judges.config import JudgeConfig, add_judge_config_args, apply_annotator_config_overrides


def test_parser_accepts_new_annotator_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator_model", default="auto")
    add_judge_config_args(parser)

    args = parser.parse_args(
        [
            "--annotator_model",
            "deepseek-v4-pro",
            "--annotator_provider",
            "deepseek",
            "--annotator_api_surface",
            "chat_completions",
            "--annotator_api_key_env",
            "DEEPSEEK_API_KEY",
            "--annotator_base_url",
            "https://api.deepseek.com",
            "--annotator_reasoning_effort",
            "high",
            "--annotator_max_output_tokens",
            "4096",
        ]
    )

    config = JudgeConfig.from_args(args)

    assert config.model == "deepseek-v4-pro"
    assert config.provider == "deepseek"
    assert config.api_surface == "chat_completions"
    assert config.base_url == "https://api.deepseek.com"
    assert config.max_output_tokens == 4096


def test_nested_yaml_annotator_config_loads_judge_settings():
    args = argparse.Namespace(
        annotator_model="auto",
        annotator_provider=None,
        annotator_api_surface=None,
        annotator_api_key_env=None,
        annotator_base_url=None,
        annotator_reasoning_effort=None,
        annotator_max_output_tokens=None,
        annotator_num_workers=None,
        annotator_cache_path=None,
        max_tokens=None,
    )
    raw = {
        "annotator": {
            "provider": "openai",
            "model": "gpt-5.5",
            "api_surface": "responses",
            "api_key_env": "OPENAI_API_KEY",
            "reasoning_effort": "high",
            "max_output_tokens": 8192,
        },
        "tasks": [{"task_name": "MATH500", "batch_size": "auto"}],
    }

    apply_annotator_config_overrides(args, raw)
    config = JudgeConfig.from_yaml(raw, args)

    assert args.max_tokens is None
    assert config.model == "gpt-5.5"
    assert config.reasoning_effort == "high"
    assert config.max_output_tokens == 8192


def test_legacy_flat_yaml_still_works():
    args = argparse.Namespace(
        annotator_model="auto",
        max_tokens=None,
    )
    raw = {
        "annotator_model": "gpt-4o-mini-2024-07-18",
        "tasks": [{"task_name": "alpaca_eval", "batch_size": 1}],
    }

    apply_annotator_config_overrides(args, raw)

    assert args.annotator_model == "gpt-4o-mini-2024-07-18"
    assert args.max_tokens is None
    assert JudgeConfig.from_yaml(raw, args) is None
