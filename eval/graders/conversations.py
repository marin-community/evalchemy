"""Named graders for fixed conversations."""

import re

from eval.contracts.eval_spec import Grade, GraderSpec, JudgePlan, ModelRequest, RequestKind


def mtbench(spec, responses):
    reference = spec.grader.reference
    requests = []
    for index, template in enumerate(reference["templates"][:len(responses)]):
        if index == 0:
            values = {"question": spec.doc["turns"][0], "answer": str(responses[0])}
        else:
            values = {"question_1": spec.doc["turns"][0], "question_2": spec.doc["turns"][1],
                      "answer_1": str(responses[0]), "answer_2": str(responses[1])}
        values.update({f"ref_answer_{i + 1}": answer for i, answer in enumerate(reference["answers"])})
        prompt = [{"role": "system", "content": template["system_prompt"]},
                  {"role": "user", "content": template["prompt_template"].format(**values)}]
        requests.append(ModelRequest(RequestKind.GENERATE, prompt, {"temperature": 0, "max_gen_toks": 2048}))
    return JudgePlan(tuple(requests), GraderSpec("turn_ratings"))


def turn_ratings(spec, responses):
    scores = []
    metrics = {}
    for index, response in enumerate(responses):
        match = re.search(r"\[\[(\d+\.?\d*)\]\]", str(response)) or re.search(r"\[(\d+\.?\d*)\]", str(response))
        if match is None:
            continue
        score = float(match.group(1))
        scores.append(score)
        metrics[f"Turn {index + 1}"] = score
    if not scores:
        return Grade({}, {"judge_raw": [str(response) for response in responses]})
    metrics["Average"] = sum(scores) / len(scores)
    return Grade(metrics, {"judge_raw": [str(response) for response in responses]}, {"Average": len(scores)})


def livebench(spec, responses):
    from eval.chat_benchmarks.LiveBench.livebench.common import MatchSingle
    from eval.chat_benchmarks.LiveBench.livebench.gen_ground_truth_judgment import play_a_match_gt

    answer = {"choices": [{"turns": [str(response).strip() for response in responses]}]}
    result = play_a_match_gt(MatchSingle(dict(spec.doc), "candidate", answer), output_file=None)
    return Grade({"accuracy": 100 * float(result["score"])})
