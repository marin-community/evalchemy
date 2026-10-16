"""Alpaca corpus metrics retain their source indices under concurrent completion."""

import asyncio
from dataclasses import replace
import re

import pandas as pd
import pytest
from aiohttp import ClientSession, web
from alpaca_eval import constants, utils
from alpaca_eval.annotators import SinglePairwiseAnnotator
from alpaca_eval.metrics import get_length_controlled_winrate, glm_winrate

from eval.chat_benchmarks.alpaca_eval.eval_instruct import AlpacaBenchmark
from eval.contracts.eval_spec import EvalSpec, GraderSpec, ReducerSpec, SourceRef
from eval.execution.engine import ItemResult
from eval.execution.metrics import SourceMetrics
from eval.execution.judge import JudgeBackend
from eval.graders.alpaca import alpaca, alpaca_preference
from eval.graders.answer_equivalence import JudgeConfig
from eval.sources.common import SourceOptions


@pytest.fixture
def alpaca_calibration(tmp_path, monkeypatch):
    calibration = tmp_path / "df_gamed.csv"
    pd.DataFrame([
        {
            "model": model, "index": index,
            "instruction_difficulty": index / 5 - 2,
            "std_delta_len": length, "not_gamed_baseline": False,
            "preference": 0.2 + index / 50 if length < 0 else 0.8 - index / 50,
        }
        for model, length in (("short", -1), ("long", 1))
        for index in range(20)
    ]).to_csv(calibration, index=False)

    def download(*, filename, **kwargs):
        assert filename == "df_gamed.csv"
        return str(calibration)

    monkeypatch.setattr(glm_winrate, "hf_hub_download", download)


def _ranking(prompt):
    if not isinstance(prompt, str):
        prompt = "\n".join(message["content"] for message in prompt)
    ordinal = int(re.search(r"case-(\d+)", prompt).group(1))
    rank = 1 if ordinal % 2 else 2
    return repr([{"model": "model_1", "rank": rank}, {"model": "model_2", "rank": 3 - rank}])


@pytest.mark.parametrize("reverse_completions", [False, True])
def test_alpaca_corpus_score_matches_original_indexed_annotations(alpaca_calibration, reverse_completions):
    name = "alpaca_eval_gpt4_0613"
    config = next(iter(utils.load_configs(name, relative_to=constants.EVALUATORS_CONFIG_DIR).values()))
    pairs = [
        {
            "index": ordinal, "instruction": f"case-{ordinal}",
            "output_1": "reference " * (ordinal % 5 + 1),
            "output_2": "candidate " * (ordinal % 7 + 1),
            "generator_1": "reference", "generator_2": "candidate",
        }
        for ordinal in (17, 2, 11, 4, 0, 13, 8, 19, 6, 1)
    ]

    def completions(prompts, **kwargs):
        return {"completions": [_ranking(prompt) for prompt in prompts]}

    original = SinglePairwiseAnnotator(**{**config, "fn_completions": completions})
    original_annotations = original(pd.DataFrame(pairs)).to_dict("records")
    for annotation in original_annotations:
        annotation["annotator"] = name
    expected = get_length_controlled_winrate(original_annotations, save_weights_dir=None)

    results = []
    for pair in pairs:
        spec = EvalSpec(
            "alpaca_eval", SourceRef("fixture"), pair["index"], pair["index"], (),
            GraderSpec("alpaca", {"config": name, "baseline": "reference"}),
            doc={"instruction": pair["instruction"]}, target=pair["output_1"],
        )
        plan = alpaca(spec, [pair["output_2"]])
        grade = alpaca_preference(replace(spec, grader=plan.grader), [_ranking(request.prompt) for request in plan.requests])
        results.append(ItemResult(spec, (), grade, {"sample_id": str(pair["index"])}))
    metrics = SourceMetrics(repeated=False, policy=ReducerSpec(name="alpaca", trial_metric=None))
    for result in reversed(results) if reverse_completions else results:
        metrics.add(result)

    assert metrics.finish("alpaca_eval")["results"]["alpaca_eval"] == pytest.approx(expected)


@pytest.mark.parametrize("model", ["auto", "gpt-4", "gpt-4-1106-preview", "gpt-4o-mini-2024-07-18"])
@pytest.mark.parametrize("rank", [1, 2])
def test_alpaca_wire_model_and_saved_reply_match_original_annotator(monkeypatch, model, rank):
    row = {"instruction": "case-17", "output": "reference response", "generator": "reference"}
    monkeypatch.setattr(constants, "ALPACAEVAL_REFERENCE_OUTPUTS", lambda: [row])
    source = AlpacaBenchmark(annotator_model=model)
    monkeypatch.setattr(source, "load_dataset", lambda: [row])
    spec = next(source.iter_specs(SourceOptions("alpaca_eval", source.SOURCE)))
    config = next(iter(utils.load_configs(source.annotator_conf, relative_to=constants.EVALUATORS_CONFIG_DIR).values()))
    choice = {
        "index": 0, "finish_reason": "stop",
        "message": {"role": "assistant", "content": repr([
            {"model": "model_1", "rank": rank}, {"model": "model_2", "rank": 3 - rank},
        ])},
        "logprobs": {"content": [{"top_logprobs": [
            {"token": "m", "logprob": -0.2 if rank == 1 else -1.6},
            {"token": "M", "logprob": -1.6 if rank == 1 else -0.2},
        ]}]},
    }
    original_requests = []

    def completions(prompts, model_name, **kwargs):
        original_requests.append({"model": model_name, "messages": utils.prompt_to_chatml(prompts[0]), **kwargs})
        return {"completions": [choice["message"]["content"]], "completions_all": [choice]}

    original = SinglePairwiseAnnotator(**{**config, "fn_completions": completions})
    [expected] = original(pd.DataFrame([{
        "index": 0, "instruction": row["instruction"], "output_1": row["output"], "output_2": "candidate response",
        "generator_1": "reference", "generator_2": "candidate",
    }])).to_dict("records")
    plan = alpaca(spec, ["candidate response"])
    requests = []

    async def handler(request):
        requests.append(await request.json())
        return web.json_response({"choices": [choice]})

    async def run():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            await web.TCPSite(runner, "127.0.0.1", 0).start()
            base_url = f"http://127.0.0.1:{runner.addresses[0][1]}/v1"
            async with ClientSession() as session:
                backend = JudgeBackend(JudgeConfig("generic-fallback", base_url, "test-key"), session)
                return [await backend.execute(request) for request in plan.requests]
        finally:
            await runner.cleanup()

    replies = asyncio.run(run())
    grade = alpaca_preference(replace(spec, grader=plan.grader), replies)

    assert requests == original_requests
    assert source.judge_model == original_requests[0]["model"]
    assert grade.metrics["preference"] == pytest.approx(expected["preference"])
