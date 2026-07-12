# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the e2e eval-args builder. No model / Marin / lm-eval needed."""

import pytest

from eval.e2e.eval_args import (
    LOCAL_CHAT_COMPLETIONS,
    LOCAL_COMPLETIONS,
    EvalInvocation,
    ServedModel,
    adapter_for,
    build_eval_argv,
    build_model_args,
    endpoint_url,
)


def test_adapter_selection():
    assert adapter_for(True) == LOCAL_CHAT_COMPLETIONS
    assert adapter_for(False) == LOCAL_COMPLETIONS


def test_endpoint_url_appends_once_from_v1_root():
    assert endpoint_url("http://h/v1", LOCAL_CHAT_COMPLETIONS) == "http://h/v1/chat/completions"
    assert endpoint_url("http://h/v1/", LOCAL_COMPLETIONS) == "http://h/v1/completions"


def test_endpoint_url_never_doubles_the_path():
    # A caller who redundantly included the adapter path must not get it twice.
    assert endpoint_url("http://h/v1/chat/completions", LOCAL_CHAT_COMPLETIONS) == "http://h/v1/chat/completions"
    assert endpoint_url("http://h/v1/completions", LOCAL_COMPLETIONS) == "http://h/v1/completions"


def test_endpoint_url_rejects_unknown_adapter():
    with pytest.raises(ValueError):
        endpoint_url("http://h/v1", "nope")


def test_build_model_args_shape_matches_marin_recipe():
    served = ServedModel(base_url="http://127.0.0.1:8000/v1", model="served-model", api_key="k", tokenizer="tok")
    args = build_model_args(served, LOCAL_CHAT_COMPLETIONS, extra={"num_concurrent": 2, "timeout": 30})
    # Mirrors marin.evaluation.lm_eval.build_lm_eval_model_args field ordering/semantics.
    assert args == (
        "model=served-model,"
        "base_url=http://127.0.0.1:8000/v1/chat/completions,"
        "tokenizer_backend=huggingface,"
        "tokenized_requests=False,"
        "api_key=k,"
        "tokenizer=tok,"
        "num_concurrent=2,"
        "timeout=30"
    )


def test_build_model_args_omits_optional_fields():
    served = ServedModel(base_url="http://h/v1", model="m")
    args = build_model_args(served, LOCAL_COMPLETIONS)
    assert "api_key" not in args
    assert args.startswith("model=m,base_url=http://h/v1/completions,")


def test_build_model_args_rejects_comma_in_value():
    served = ServedModel(base_url="http://h/v1", model="m")
    with pytest.raises(ValueError):
        build_model_args(served, LOCAL_COMPLETIONS, extra={"bad": "a,b"})


def test_argv_uses_bare_apply_chat_template_flag():
    # The parser option is nargs="?", const=True: passing a value would be read as
    # a template *name*. The flag must appear bare, with no following value.
    inv = EvalInvocation(
        served=ServedModel(base_url="http://h/v1", model="m"),
        tasks=["gsm8k"],
        output_path="/out",
        apply_chat_template=True,
        limit=20,
        seed=1234,
    )
    argv = build_eval_argv(inv, python="py")
    assert "--apply_chat_template" in argv
    idx = argv.index("--apply_chat_template")
    # Nothing (or a new flag) follows -- never the literal "True".
    following = argv[idx + 1] if idx + 1 < len(argv) else None
    assert following is None or following.startswith("--")
    assert "True" not in argv


def test_argv_selects_chat_adapter_and_core_flags():
    inv = EvalInvocation(
        served=ServedModel(base_url="http://h/v1", model="m"),
        tasks=["gsm8k", "arc_easy"],
        output_path="/out",
        apply_chat_template=True,
        limit=5,
    )
    argv = build_eval_argv(inv)
    assert argv[:3] == [argv[0], "-m", "eval.eval"]
    assert "--model" in argv and argv[argv.index("--model") + 1] == LOCAL_CHAT_COMPLETIONS
    assert argv[argv.index("--tasks") + 1] == "gsm8k,arc_easy"
    assert argv[argv.index("--limit") + 1] == "5"
    assert argv[argv.index("--output_path") + 1] == "/out"
    assert "--log_samples" in argv


def test_argv_scoring_task_has_no_chat_template_flag():
    inv = EvalInvocation(
        served=ServedModel(base_url="http://h/v1", model="m"),
        tasks=["arc_easy"],
        output_path="/out",
        apply_chat_template=False,
    )
    argv = build_eval_argv(inv)
    assert argv[argv.index("--model") + 1] == LOCAL_COMPLETIONS
    assert "--apply_chat_template" not in argv


def test_argv_requires_tasks():
    inv = EvalInvocation(served=ServedModel(base_url="http://h/v1", model="m"), tasks=[], output_path="/out")
    with pytest.raises(ValueError):
        build_eval_argv(inv)


def test_argv_appends_extra_args_verbatim():
    # pass@k etc. reach eval.eval via the passthrough, appended last so lm-eval's
    # argparse lets them override our defaults.
    inv = EvalInvocation(
        served=ServedModel(base_url="http://h/v1", model="m"),
        tasks=["MATH500"],
        output_path="/out",
        extra_args=["--num_samples", "8", "--pass_at_k", "1,8"],
    )
    argv = build_eval_argv(inv)
    assert argv[-4:] == ["--num_samples", "8", "--pass_at_k", "1,8"]
