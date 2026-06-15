"""Stage-2 CPU unit tests — `RunFingerprint.from_run_inputs` + the full decision matrix.

Gate (from `notes/evalchemy/stage2_fingerprint_decision_scope.md`):
  - one row per stage-0 partition field:
      * change each MATERIAL field -> decide() == "refuse" (auto), field named in the error;
      * change each IGNORED/cosmetic field -> decide() == "resume" (same fingerprint);
  - alias normalization: max_gen_toks == max_new_tokens == max_tokens -> identical fingerprint;
  - content-hash sensitivity: editing the template / grader / data FILE changes the fingerprint;
  - TP / rank-layout mismatch -> refuse;
  - force-fresh + off behave per spec.

Pure stdlib + tmp files -> runs on any Python with pytest (no GPU env).
"""

import hashlib
from pathlib import Path

import pytest

from eval.resume import ResumeManager, ResumeRefused, RunFingerprint
from eval.resume.fingerprint import (
    IGNORED_FIELDS,
    MATERIAL_FIELDS,
    digest_file,
    digest_files,
    normalize_decoding,
    resolve_model_revision,
)


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _base_kwargs(tmp_path: Path):
    """A fully-resolved, realistic from_run_inputs kwargs set (MATH500-like)."""
    data = tmp_path / "math500.jsonl"
    data.write_text('{"problem": "1+1", "answer": "2"}\n')
    grader = tmp_path / "grader.py"
    grader.write_text("def is_equiv(a, b): return a == b\n")
    return dict(
        model_repo="org/model",
        model_revision="a" * 40,
        task_name="MATH500",
        task_data_path=data,
        grader_source_path=grader,
        grader_version="hendrycks_math.v1",
        rendered_config={"task": "MATH500", "limit": None},
        apply_chat_template=True,
        chat_template="{{messages}}",
        decoding={"temperature": 0.7, "top_p": 1.0, "do_sample": False,
                  "max_new_tokens": 32768, "n": 1, "num_samples": 1},
        seed_set=[0, 1234, 1234, 1234],
        max_model_len=40960,
        num_fewshot=0,
        passk_batch_size=64,
    )


def _fp(tmp_path, **overrides):
    kw = _base_kwargs(tmp_path)
    kw.update(overrides)
    return RunFingerprint.from_run_inputs(**kw)


# --------------------------------------------------------------------------------------------
# from_run_inputs resolves the curated MATERIAL set
# --------------------------------------------------------------------------------------------
def test_from_run_inputs_resolves_material_set(tmp_path: Path):
    fp = _fp(tmp_path)
    payload = fp._canonical_payload()
    # every resolved key is in the MATERIAL partition (or a *_digest derived from it)
    for k in payload:
        assert k in MATERIAL_FIELDS, f"{k} leaked outside MATERIAL partition"
    # files were content-hashed into digests
    assert payload["task_data_digest"].startswith("sha256:")
    assert payload["grader_source_digest"].startswith("sha256:")
    assert payload["template_digest"].startswith("sha256:")
    # alias normalized: max_new_tokens collapsed to max_tokens
    assert "max_tokens" in payload
    assert "max_new_tokens" not in payload
    assert payload["max_tokens"] == 32768
    assert fp.value().startswith("sha256:")


def test_from_run_inputs_stable_across_key_order(tmp_path: Path):
    # decoding dict order / kwarg order must not change the hash
    fp1 = _fp(tmp_path, decoding={"n": 1, "temperature": 0.7, "top_p": 1.0,
                                  "do_sample": False, "max_new_tokens": 32768, "num_samples": 1})
    fp2 = _fp(tmp_path, decoding={"max_new_tokens": 32768, "num_samples": 1, "do_sample": False,
                                  "top_p": 1.0, "temperature": 0.7, "n": 1})
    assert fp1.value() == fp2.value()


# --------------------------------------------------------------------------------------------
# alias normalization
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("alias", ["max_tokens", "max_new_tokens", "max_gen_toks"])
def test_alias_normalization_same_hash(tmp_path: Path, alias):
    ref = _fp(tmp_path, decoding={"temperature": 0.7, "max_tokens": 4096})
    other = _fp(tmp_path, decoding={"temperature": 0.7, alias: 4096})
    assert ref.value() == other.value(), f"{alias} should normalize to max_tokens"


def test_alias_normalization_unit():
    assert normalize_decoding({"max_gen_toks": 100}) == {"max_tokens": 100}
    assert normalize_decoding({"max_new_tokens": 100}) == {"max_tokens": 100}
    assert normalize_decoding({"max_tokens": 100}) == {"max_tokens": 100}
    # None values dropped; non-alias passes through
    assert normalize_decoding({"temperature": 0.7, "top_p": None}) == {"temperature": 0.7}


def test_alias_conflict_raises():
    with pytest.raises(ValueError):
        normalize_decoding({"max_gen_toks": 64, "max_tokens": 128})


def test_alias_agreeing_values_ok():
    # same value via two aliases is not a conflict
    assert normalize_decoding({"max_gen_toks": 64, "max_tokens": 64}) == {"max_tokens": 64}


# --------------------------------------------------------------------------------------------
# content-hash sensitivity: editing a controlling FILE changes the fingerprint
# --------------------------------------------------------------------------------------------
def test_data_file_content_change_changes_fingerprint(tmp_path: Path):
    # build run-1 kwargs ONCE (writes the files), fingerprint, then mutate the file in place and
    # re-fingerprint from the SAME kwargs (do NOT go through _fp, which rewrites the files).
    kw = _base_kwargs(tmp_path)
    fp1 = RunFingerprint.from_run_inputs(**kw)
    # mutate the data file content (e.g. a debug [:2] slice -> different rows)
    Path(kw["task_data_path"]).write_text('{"problem": "2+2", "answer": "4"}\n')
    fp2 = RunFingerprint.from_run_inputs(**kw)
    assert fp1.value() != fp2.value()
    assert "task_data_digest" in fp1.diff_material(fp2)


def test_grader_source_change_changes_fingerprint(tmp_path: Path):
    kw = _base_kwargs(tmp_path)
    fp1 = RunFingerprint.from_run_inputs(**kw)
    Path(kw["grader_source_path"]).write_text("def is_equiv(a, b): return str(a) == str(b)\n")
    fp2 = RunFingerprint.from_run_inputs(**kw)
    assert fp1.value() != fp2.value()
    assert "grader_source_digest" in fp1.diff_material(fp2)


def test_template_string_change_changes_fingerprint(tmp_path: Path):
    fp1 = _fp(tmp_path)
    fp2 = _fp(tmp_path, chat_template="{{messages}}<|im_end|>")
    assert fp1.value() != fp2.value()
    assert "template_digest" in fp1.diff_material(fp2)


def test_digest_file_helpers(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello")
    assert digest_file(f) == "sha256:" + hashlib.sha256(b"hello").hexdigest()
    assert digest_file(None) is None
    with pytest.raises(FileNotFoundError):
        digest_file(tmp_path / "missing.txt")
    # multi-file grader: order-independent, content-sensitive
    g1 = tmp_path / "a.py"; g1.write_text("A")
    g2 = tmp_path / "b.py"; g2.write_text("B")
    d_ab = digest_files([g1, g2])
    d_ba = digest_files([g2, g1])
    assert d_ab == d_ba and d_ab.startswith("sha256:")
    g2.write_text("B2")
    assert digest_files([g1, g2]) != d_ab


def test_precomputed_digest_for_in_memory_dataset(tmp_path: Path):
    # HF dataset with no single file: caller hashes serialized rows, passes the digest directly
    fp = _fp(tmp_path, task_data_path=None, task_data_digest="sha256:deadbeef")
    assert fp._canonical_payload()["task_data_digest"] == "sha256:deadbeef"


# --------------------------------------------------------------------------------------------
# model revision resolution
# --------------------------------------------------------------------------------------------
def test_resolve_model_revision_pinned_passthrough():
    sha = "b" * 40
    assert resolve_model_revision("org/m", sha) == sha  # already a commit -> unchanged
    # offline / no-network: branch name passes through unchanged (no HF_HUB_OFFLINE network hit)
    assert resolve_model_revision("org/m", "main", allow_network=False) == "main"
    assert resolve_model_revision("org/m", None, allow_network=False) is None


def test_model_revision_is_material(tmp_path: Path):
    fp1 = _fp(tmp_path, model_revision="a" * 40)
    fp2 = _fp(tmp_path, model_revision="c" * 40)
    assert fp1.value() != fp2.value()
    assert "model_revision" in fp1.diff_material(fp2)


# --------------------------------------------------------------------------------------------
# DECISION MATRIX — one row per MATERIAL field -> refuse; one per IGNORED field -> resume
# --------------------------------------------------------------------------------------------
# Each entry: (override_for_run_2) where run_1 uses _base_kwargs. The override changes exactly
# one MATERIAL input; decide() must REFUSE and name the field.
_MATERIAL_DELTAS = {
    "model_repo": dict(model_repo="org/other"),
    "model_revision": dict(model_revision="f" * 40),
    "task_name": dict(task_name="AIME24"),
    "grader_version": dict(grader_version="hendrycks_math.v2"),
    "apply_chat_template": dict(apply_chat_template=False),
    "temperature": dict(decoding={"temperature": 1.0, "top_p": 1.0, "do_sample": False,
                                  "max_new_tokens": 32768, "n": 1, "num_samples": 1}),
    "top_p": dict(decoding={"temperature": 0.7, "top_p": 0.95, "do_sample": False,
                            "max_new_tokens": 32768, "n": 1, "num_samples": 1}),
    "max_tokens": dict(decoding={"temperature": 0.7, "top_p": 1.0, "do_sample": False,
                                 "max_new_tokens": 8192, "n": 1, "num_samples": 1}),
    "do_sample": dict(decoding={"temperature": 0.7, "top_p": 1.0, "do_sample": True,
                                "max_new_tokens": 32768, "n": 1, "num_samples": 1}),
    "n": dict(decoding={"temperature": 0.7, "top_p": 1.0, "do_sample": False,
                        "max_new_tokens": 32768, "n": 8, "num_samples": 1}),
    "num_samples": dict(decoding={"temperature": 0.7, "top_p": 1.0, "do_sample": False,
                                  "max_new_tokens": 32768, "n": 1, "num_samples": 128}),
    "seed_set": dict(seed_set=[1, 2, 3, 4]),
    "max_model_len": dict(max_model_len=65536),
    "num_fewshot": dict(num_fewshot=5),
    "passk_batch_size": dict(passk_batch_size=32),
    "rendered_config": dict(rendered_config={"task": "MATH500", "limit": 100}),
}


@pytest.mark.parametrize("field_name,override", list(_MATERIAL_DELTAS.items()),
                         ids=list(_MATERIAL_DELTAS))
def test_material_delta_refuses(tmp_path: Path, field_name, override):
    run_dir = tmp_path / "run"
    fp1 = _fp(tmp_path)
    assert ResumeManager(run_dir, fp1, mode="auto").decide() == "fresh"

    fp2 = _fp(tmp_path, **override)
    assert fp1.value() != fp2.value(), f"{field_name} delta must change the fingerprint"
    with pytest.raises(ResumeRefused) as exc:
        ResumeManager(run_dir, fp2, mode="auto").decide()
    # the changed field (or its normalized/digest name) is named in the error
    msg = str(exc.value)
    named = field_name
    if field_name in ("temperature", "top_p", "max_tokens", "do_sample", "n", "num_samples"):
        named = "max_tokens" if field_name == "max_tokens" else field_name
    if field_name == "apply_chat_template":
        # template digest also drops out when off; the bool itself is named
        named = "apply_chat_template"
    assert named in msg, f"refuse error should name {named}; got: {msg}"


# IGNORED (cosmetic): changing it must keep the SAME fingerprint and RESUME.
_IGNORED_DELTAS = {
    "output_path": dict(output_path="/scratch/run-2"),
    "job_name": dict(job_name="rerun"),
    "run_tag": dict(run_tag="v2"),
    "batch_size": dict(batch_size=512),
    "max_batch_size": dict(max_batch_size=1024),
    "gpu_memory_utilization": dict(gpu_memory_utilization=0.95),
    "tensor_parallel_size": dict(tensor_parallel_size=8),
    "verbosity": dict(verbosity="DEBUG"),
    "log_level": dict(log_level="INFO"),
    "slurm_job_id": dict(slurm_job_id="99999"),
    "started_at": dict(started_at="2026-06-15T00:00:00"),
    "wall_clock": dict(wall_clock=3600),
}


@pytest.mark.parametrize("field_name,cosmetic", list(_IGNORED_DELTAS.items()),
                         ids=list(_IGNORED_DELTAS))
def test_ignored_delta_resumes(tmp_path: Path, field_name, cosmetic):
    run_dir = tmp_path / "run"
    # build run-1 fingerprint and inject the cosmetic field into the stored inputs
    fp1 = _fp(tmp_path)
    fp1.inputs.update(cosmetic)
    m1 = ResumeManager(run_dir, fp1, mode="auto")
    assert m1.decide() == "fresh"
    m1.record({"task": "MATH500", "problem_idx": 0}, {"v": 0})

    # run-2: same material set, DIFFERENT cosmetic value -> identical fingerprint, RESUME
    fp2 = _fp(tmp_path)
    # flip the cosmetic field to a different value (keep type, change value)
    val = list(cosmetic.values())[0]
    flipped = (val + "_x") if isinstance(val, str) else (val + 1 if isinstance(val, int) else 0.5)
    fp2.inputs[field_name] = flipped
    assert fp1.value() == fp2.value(), f"{field_name} must not change the fingerprint"
    m2 = ResumeManager(run_dir, fp2, mode="auto")
    assert m2.decide() == "resume", f"{field_name} cosmetic delta must resume"
    assert m2.should_skip({"task": "MATH500", "problem_idx": 0})


# --------------------------------------------------------------------------------------------
# TP / rank-layout mismatch -> refuse (decision #3)
# --------------------------------------------------------------------------------------------
def test_rank_layout_mismatch_refuses_with_real_fingerprint(tmp_path: Path):
    run_dir = tmp_path / "run"
    fp = _fp(tmp_path)
    m1 = ResumeManager(run_dir, fp, mode="auto", world_size=1, rank=0)
    m1.decide()
    m1.record({"task": "MATH500", "problem_idx": 0}, {"v": 0})
    # same material fingerprint, but now world_size=4 -> per-rank layout mismatch -> refuse
    m2 = ResumeManager(run_dir, fp, mode="auto", world_size=4, rank=0)
    with pytest.raises(ResumeRefused) as exc:
        m2.decide()
    assert "rank" in str(exc.value).lower()


# --------------------------------------------------------------------------------------------
# force-fresh + off
# --------------------------------------------------------------------------------------------
def test_force_fresh_overrides_material_refuse(tmp_path: Path):
    run_dir = tmp_path / "run"
    fp1 = _fp(tmp_path)
    m1 = ResumeManager(run_dir, fp1, mode="auto")
    m1.decide()
    m1.record({"task": "MATH500", "problem_idx": 0}, {"v": 0})

    # a material delta that would refuse under auto...
    fp2 = _fp(tmp_path, decoding={"temperature": 1.0, "top_p": 1.0, "do_sample": True,
                                  "max_new_tokens": 32768, "n": 8, "num_samples": 8})
    with pytest.raises(ResumeRefused):
        ResumeManager(run_dir, fp2, mode="auto").decide()
    # ...is wiped + restarted under force-fresh
    m3 = ResumeManager(run_dir, fp2, mode="force-fresh")
    assert m3.decide() == "fresh"
    assert m3.done_units() == set()
    # the rewritten fingerprint matches the NEW material set
    stored = (run_dir / "resume" / "fingerprint.json").read_text()
    assert fp2.value() in stored


def test_off_ignores_material_delta(tmp_path: Path):
    run_dir = tmp_path / "run"
    fp1 = _fp(tmp_path)
    seed = ResumeManager(run_dir, fp1, mode="auto")
    seed.decide()
    seed.record({"task": "MATH500", "problem_idx": 0}, {"v": 0})
    # off: pure no-op, never reads/validates the prior fingerprint, never refuses
    fp2 = _fp(tmp_path, temperature=1.0) if False else _fp(tmp_path, model_repo="org/other")
    m = ResumeManager(run_dir, fp2, mode="off")
    assert m.decide() == "fresh"
    assert m.should_skip({"task": "MATH500", "problem_idx": 0}) is False
