import argparse

from eval.judges import JudgeConfig
from eval.resume.wiring import build_resume_wiring


class _FakeLM:
    rank = 0
    world_size = 1


def _args(tmp_path, judge_config):
    return argparse.Namespace(
        resume_mode="auto",
        output_path=str(tmp_path),
        model_args="pretrained=some/model,revision=abc",
        model_name=None,
        max_tokens="256",
        gen_kwargs="temperature=0.7,top_p=1.0",
        num_samples=1,
        apply_chat_template=False,
        num_fewshot=None,
        seed=[0, 1234, 1234, 1234],
        annotator_model=judge_config.model if judge_config else "auto",
        judge_config=judge_config,
        limit=None,
        predict_only=False,
        fewshot_as_multiturn=False,
        system_instruction=None,
        passk_batch_size=None,
    )


def test_judge_provider_changes_resume_fingerprint(tmp_path):
    openai_args = _args(tmp_path, JudgeConfig.from_model("gpt-5.5"))
    deepseek_args = _args(tmp_path, JudgeConfig.from_model("deepseek-v4-pro"))

    openai_fp = build_resume_wiring(openai_args, _FakeLM())("ProofTask").fingerprint.value()
    deepseek_fp = build_resume_wiring(deepseek_args, _FakeLM())("ProofTask").fingerprint.value()

    assert openai_fp != deepseek_fp


def test_judge_reasoning_effort_changes_resume_fingerprint(tmp_path):
    medium = _args(tmp_path, JudgeConfig.from_model("gpt-5.5", reasoning_effort="medium"))
    high = _args(tmp_path, JudgeConfig.from_model("gpt-5.5", reasoning_effort="high"))

    medium_fp = build_resume_wiring(medium, _FakeLM())("ProofTask").fingerprint.value()
    high_fp = build_resume_wiring(high, _FakeLM())("ProofTask").fingerprint.value()

    assert medium_fp != high_fp


def test_judge_api_key_value_does_not_enter_resume_fingerprint(tmp_path, monkeypatch):
    args = _args(tmp_path, JudgeConfig.from_model("gpt-5.5"))

    monkeypatch.setenv("OPENAI_API_KEY", "sk-one")
    fp1 = build_resume_wiring(args, _FakeLM())("ProofTask").fingerprint.value()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-two")
    fp2 = build_resume_wiring(args, _FakeLM())("ProofTask").fingerprint.value()

    assert fp1 == fp2
