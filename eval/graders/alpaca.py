"""AlpacaEval prompt processors and parsers, without its execution driver."""

from functools import lru_cache

from eval.contracts.eval_spec import Grade, GraderSpec, JudgePlan, ModelRequest, RequestKind


@lru_cache
def annotation_config(name):
    from alpaca_eval import constants, utils

    configs = utils.load_configs(name, relative_to=constants.EVALUATORS_CONFIG_DIR)
    if len(configs) != 1:
        raise ValueError("an AlpacaEval source must select one annotator")
    annotator, config = next(iter(configs.items()))
    return annotator, config


def annotator_for(name):
    from alpaca_eval.annotators import SinglePairwiseAnnotator

    _, configuration = annotation_config(name)
    config = dict(configuration)
    config["batch_size"] = 1
    return SinglePairwiseAnnotator(**config)


def alpaca(spec, responses):
    import pandas as pd
    from alpaca_eval import utils

    name = spec.grader.reference["config"]
    annotator = annotator_for(name)
    frame = pd.DataFrame([{
        "index": spec.ordinal,
        "instruction": spec.doc["instruction"], "output_1": spec.target, "output_2": str(responses[-1]),
        "generator_1": spec.grader.reference["baseline"], "generator_2": "candidate",
    }])
    frame = annotator._preprocess(frame)
    prompts, frame = annotator._make_prompts(frame)
    settings = dict(annotator.completions_kwargs)
    settings.pop("model_name", None)
    requests = tuple(ModelRequest(RequestKind.GENERATE, utils.prompt_to_chatml(prompt), settings) for prompt in prompts)
    return JudgePlan(requests, GraderSpec("alpaca_preference", {"config": name, "frame": frame.to_dict("records")}))


def alpaca_preference(spec, responses):
    import pandas as pd

    name = spec.grader.reference["config"]
    annotator = annotator_for(name)
    completions = [response.response.raw_choice if annotator.completion_key == "completions_all" else str(response)
                   for response in responses]
    preferences, _ = annotator._parse_completions(completions)
    frame = pd.DataFrame(spec.grader.reference["frame"])
    frame["preference"] = preferences
    frame = annotator._postprocess(frame)
    if frame["preference"].isna().any():
        raise ValueError("AlpacaEval judge returned no valid preference")
    row = frame.to_dict("records")[0]
    row["annotator"] = annotation_config(name)[0]
    return Grade({"preference": float(row["preference"])}, {"alpaca_annotation": row,
                                                            "judge_raw": [str(response) for response in responses]})
