"""Public contract tests for the portable Evalchemy evaluation configuration."""

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval.serve_eval.config import RunConfig
from eval.serve_eval.providers import ServedModel
from eval.serve_eval.run import LOCAL_CHAT_COMPLETIONS, build_eval_argv
from evalchemy_config import (
    EvaluationConfig,
    TaskOptions,
    apply_evaluation_patch,
    canonical_json,
    fingerprint,
    load_evaluation_config,
    materialize_eval_args,
)

_ROOT = Path(__file__).parents[2]


def test_shipped_flat_yaml_becomes_portable_evaluation_intent():
    cfg = RunConfig.load(_ROOT / "eval" / "serve_eval" / "configs" / "qwen-tiny.yaml")

    assert cfg.evaluation.tasks == ["gsm8k"]
    assert cfg.evaluation.max_tokens == 1024
    assert cfg.evaluation.max_length is None
    assert cfg.tpu == "v5litepod-8"


def test_runner_argv_preserves_shipped_evaluation_behavior():
    cfg = RunConfig.load(_ROOT / "eval" / "serve_eval" / "configs" / "qwen-tiny.yaml")
    served = ServedModel(base_url="http://endpoint/v1", model="Qwen/Qwen3-0.6B", tokenizer=cfg.tokenizer)

    argv = build_eval_argv(served, cfg, "/results", limit=cfg.limit, extra_args=[], python="python")

    assert argv == [
        "python",
        "-m",
        "eval.eval",
        "--model",
        LOCAL_CHAT_COMPLETIONS,
        "--model_args",
        "model=Qwen/Qwen3-0.6B,base_url=http://endpoint/v1/chat/completions,tokenizer_backend=huggingface,tokenized_requests=False,tokenizer=Qwen/Qwen3-0.6B,num_concurrent=1,timeout=120,max_retries=3",
        "--tasks",
        "gsm8k",
        "--apply_chat_template",
        "--limit",
        "200",
        "--num_fewshot",
        "5",
        "--batch_size",
        "1",
        "--seed",
        "1234",
        "--gen_kwargs",
        "temperature=0,do_sample=false,max_gen_toks=1024",
        "--max_tokens",
        "1024",
        "--output_path",
        "/results",
        "--log_samples",
    ]


def test_evaluation_config_rejects_conflicting_legacy_output_alias():
    with pytest.raises(ValidationError, match="conflicting max_tokens"):
        EvaluationConfig.model_validate({"max_tokens": 512, "gen_kwargs": "max_gen_toks=1024"})


def test_evaluation_config_patch_revalidates_canonical_limits():
    config = load_evaluation_config({"tasks": ["gsm8k"], "max_tokens": 512})
    updated = apply_evaluation_patch(config, {"max_tokens": 768})

    assert updated.max_tokens == 768
    assert "max_gen_toks=768" in (updated.gen_kwargs or "")
    assert materialize_eval_args(updated)[-2:] == ["--max_tokens", "768"]


def test_evaluation_config_carries_task_level_client_routing():
    config = EvaluationConfig.model_validate(
        {
            "tasks": ["humaneval", "gsm8k"],
            "task_options": {
                "humaneval": {
                    "num_fewshot": 0,
                    "task_alias": "humaneval_0shot",
                    "generation": True,
                    "unsafe_code": True,
                    "completion_only": True,
                }
            },
        }
    )

    assert config.task_options["humaneval"] == TaskOptions(
        num_fewshot=0,
        task_alias="humaneval_0shot",
        generation=True,
        unsafe_code=True,
        completion_only=True,
    )
    with pytest.raises(ValidationError, match="task_options refer"):
        EvaluationConfig.model_validate({"tasks": ["gsm8k"], "task_options": {"aime24": {}}})


def test_canonical_json_and_schema_fingerprint_are_stable():
    config = EvaluationConfig.model_validate({"tasks": ["gsm8k", "math500"], "max_tokens": 512})
    same_config = EvaluationConfig.model_validate({"max_tokens": 512, "tasks": ["gsm8k", "math500"]})

    assert canonical_json(config) == canonical_json(same_config)
    assert len(fingerprint()) == 64


def test_config_wheel_imports_without_runner_or_inference_dependencies(tmp_path):
    dist = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist), "packages/evalchemy-config"],
        cwd=_ROOT,
        check=True,
    )
    wheel = next(dist.glob("evalchemy_config-*.whl"))
    probe = (
        "import json, sys; import evalchemy_config; "
        "from evalchemy_config import EvaluationConfig; "
        "EvaluationConfig.model_validate({'tasks':['gsm8k']}); "
        "print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in {'torch','vllm','marin','eval'})))"
    )
    result = subprocess.run(
        ["uv", "run", "--no-project", "--with", str(wheel), "python", "-c", probe],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []


def test_full_wheel_serve_eval_extra_contains_the_config_resolver(tmp_path):
    dist = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist), str(_ROOT)],
        cwd=tmp_path,
        check=True,
    )
    wheel = next(dist.glob("evalchemy-*.whl"))
    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            "3.12",
            "--with",
            f"{wheel}[serve-eval]",
            "python",
            "-c",
            "import evalchemy_config; import eval.serve_eval.run; print(evalchemy_config.fingerprint())",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == fingerprint()


def test_release_manifest_binds_wheel_digest_to_schema_fingerprint(tmp_path):
    dist = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist), "packages/evalchemy-config"],
        cwd=_ROOT,
        check=True,
    )
    wheel = next(dist.glob("evalchemy_config-*.whl"))
    manifest = dist / "evalchemy-config-manifest.json"
    subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            "./packages/evalchemy-config",
            "python",
            "scripts/ci/build_evalchemy_config_manifest.py",
            "--wheel",
            str(wheel),
            "--revision",
            "abc123",
            "--repository",
            "marin-community/evalchemy",
            "--output",
            str(manifest),
        ],
        cwd=_ROOT,
        check=True,
    )

    published = json.loads(manifest.read_text())
    assert published["evalchemy_revision"] == "abc123"
    assert published["schema_fingerprint"] == fingerprint()
    assert published["release_tag"] == f"evalchemy-config-{fingerprint()}"
    assert published["wheel"]["sha256"]
    assert published["wheel"]["url"].endswith(wheel.name)
