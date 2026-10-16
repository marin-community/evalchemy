"""Construct one streaming FineStore checkpoint namespace per source task."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import distributions
from pathlib import Path
from finestore.store import DataStore

from eval.contracts.finestore_output import DEFAULT_SOURCE_PREFIX
from eval.contracts.finestore_resume import FineStoreResumeManager, OPERATION_PROTOCOL
from eval.execution.run_spec import RunSpec
from eval.limits import parse_key_value_args
from eval.resume.fingerprint import RunFingerprint, digest_files, resolve_model_revision
from eval.runtime.checkpoints import input_digest

MODEL_IDENTITY_FIELDS = {
    "pretrained", "model", "revision", "tokenizer", "tokenizer_backend", "tokenizer_revision",
    "chat_template", "chat_template_args", "chat_template_kwargs", "extra_body", "completion_content_policy",
    "truncate", "max_length", "max_model_len", "dtype",
    "lora_local_path", "peft", "add_bos_token", "base_url", "model_url_suffix",
}


def build_resume_wiring(args, lm, run: RunSpec, *, store: DataStore, judge_configs=None):
    model_args = parse_key_value_args(args.model_args or "")
    identity = {name: value for name, value in model_args.items() if name in MODEL_IDENTITY_FIELDS}
    repository = identity.get("pretrained", identity.get("model"))
    identity["resolved_revision"] = resolve_model_revision(repository, identity.get("revision"), allow_network=False)
    identity["adapter"] = args.model
    identity["template"] = lm.chat_template(run.apply_chat_template)
    root = Path(__file__).resolve().parents[1]
    code = digest_files([*root.rglob("*.py"), *root.rglob("*.yaml")])
    # Wheels have no checkout lockfile. Record the versions actually executing
    # the run in both installed and source environments.
    dependencies = {distribution.metadata["Name"]: distribution.version for distribution in distributions()}

    def factory(task):
        judge = (judge_configs or {}).get(task)
        fingerprint = RunFingerprint(inputs={
            "protocol": OPERATION_PROTOCOL, "task": task, "model": identity,
            "run": asdict(run), "implementation": code,
            "dependencies": dependencies,
            "judge": {"model": judge.model, "endpoint": input_digest(judge.base_url)} if judge else None,
            "include_path": args.include_path,
        })
        return FineStoreResumeManager(
            store=store,
            source_prefix=getattr(args, "finestore_output_prefix", DEFAULT_SOURCE_PREFIX),
            task_name=task, fingerprint=fingerprint, mode=args.resume_mode,
        )

    return factory
