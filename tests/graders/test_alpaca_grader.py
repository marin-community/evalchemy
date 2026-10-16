"""Alpaca corpus metrics retain their source indices under concurrent completion."""

from dataclasses import replace
import re

import pandas as pd
import pytest
from alpaca_eval import constants, utils
from alpaca_eval.annotators import SinglePairwiseAnnotator
from alpaca_eval.metrics import get_length_controlled_winrate, glm_winrate

from eval.contracts.eval_spec import EvalSpec, GraderSpec, ReducerSpec, SourceRef
from eval.execution.engine import ItemResult
from eval.execution.metrics import SourceMetrics
from eval.graders.alpaca import alpaca, alpaca_preference


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
