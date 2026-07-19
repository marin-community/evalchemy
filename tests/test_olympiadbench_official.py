import pytest

from eval.chat_benchmarks.OlympiadBench_official import eval_instruct as olympiad
from eval.chat_benchmarks.OlympiadBench_official.auto_scoring_judge import AutoScoringJudge


def _row(**overrides):
    row = {
        "id": 1606,
        "question": "What is 40 + 2?",
        "solution": ["Add the numbers."],
        "final_answer": ["42"],
        "context": "Answer as an integer.",
        "image_1": None,
        "modality": "Text-only",
        "difficulty": "Competition",
        "is_multiple_answer": False,
        "unit": None,
        "answer_type": "Numerical",
        "error": None,
        "question_type": "Open-ended",
        "subfield": "Algebra",
        "subject": "Math",
        "language": "English",
    }
    row.update(overrides)
    return row


def _install_dataset(monkeypatch, rows_by_subset):
    calls = []

    def fake_load_dataset(dataset_name, subset, split, revision, cache_dir):
        calls.append((dataset_name, subset, split, revision, cache_dir))
        return rows_by_subset[subset]

    monkeypatch.setattr(olympiad, "load_dataset", fake_load_dataset)
    return calls


class _RecordingLM:
    def __init__(self, output_batches):
        self.output_batches = output_batches
        self.rank = 0
        self.world_size = 1
        self.calls = []
        self.messages = []

    def apply_chat_template(self, messages):
        self.messages.append(messages)
        return messages

    def generate_until(self, instances):
        instances = list(instances)
        self.calls.append(instances)
        outputs = self.output_batches[len(self.calls) - 1]
        return [outputs[instance.idx] for instance in instances]


def test_official_scorer_matches_reference_answer_forms():
    scorer = AutoScoringJudge()

    assert scorer.judge("1.000", "1.000001", precision=1e-3)
    assert scorer.judge("\\boxed{\\frac{1}{2}}", "0.5")
    assert scorer.judge("1,2", "2,1")
    assert not scorer.judge("1.000", "1.1", precision=1e-3)


def test_load_questions_uses_pinned_english_text_subsets(monkeypatch):
    rows_by_subset = {
        "OE_TO_maths_en_COMP": [
            _row(
                id=2349,
                final_answer=["$221,$8$"],
                is_multiple_answer=True,
            )
        ],
        "OE_TO_physics_en_COMP": [
            _row(
                id=1065,
                subject="Physics",
                subfield="Modern Physics",
                final_answer=["$\\frac{1}{3}$ , 0"],
                is_multiple_answer=True,
                error="0",
            ),
            _row(
                id=1214,
                subject="Physics",
                subfield="Mechanics",
                final_answer=[
                    "$\\frac{1}{2}x^2$, $3.63 \\cdot 10^{-11}$",
                    "$0.5x^2$, $3.63 \\cdot 10^{-11}$",
                ],
                is_multiple_answer=True,
                answer_type="Expression,Numerical",
                error=",1e-12",
            ),
        ],
    }
    calls = _install_dataset(monkeypatch, rows_by_subset)

    questions = olympiad.OlympiadBenchOfficialBenchmark(debug=True).load_questions()

    assert [question["subset"] for question in questions] == [
        "OE_TO_maths_en_COMP",
        "OE_TO_physics_en_COMP",
        "OE_TO_physics_en_COMP",
    ]
    assert [question["subject"] for question in questions] == ["Math", "Physics", "Physics"]
    assert questions[0]["answer"] == ["221,8"]
    assert questions[1]["answer"] == ["\\frac{1}{3} , 0"]
    assert questions[2]["answer"] == [
        "\\frac{1}{2}x^2, 3.63 \\cdot 10^{-11}",
        "0.5x^2, 3.63 \\cdot 10^{-11}",
    ]
    assert calls == [
        (
            olympiad.DEFAULT_DATASET,
            subset,
            olympiad.DEFAULT_SPLIT,
            olympiad.DEFAULT_DATASET_REVISION,
            None,
        )
        for subset in olympiad.ENGLISH_TEXT_SUBSETS
    ]


@pytest.mark.parametrize(
    "invalid_row",
    [
        _row(modality="Multimodal"),
        _row(question_type="Theorem proof"),
        _row(language="Chinese"),
        _row(image_1="unexpected-image"),
    ],
)
def test_load_questions_rejects_rows_outside_english_text_scope(monkeypatch, invalid_row):
    _install_dataset(
        monkeypatch,
        {
            "OE_TO_maths_en_COMP": [invalid_row],
            "OE_TO_physics_en_COMP": [_row(subject="Physics")],
        },
    )

    with pytest.raises(ValueError):
        olympiad.OlympiadBenchOfficialBenchmark(debug=True).load_questions()


def test_load_questions_fails_when_pinned_dataset_count_drifts(monkeypatch):
    _install_dataset(
        monkeypatch,
        {
            "OE_TO_maths_en_COMP": [_row()],
            "OE_TO_physics_en_COMP": [_row(subject="Physics")],
        },
    )

    with pytest.raises(ValueError, match="Expected 674 rows"):
        olympiad.OlympiadBenchOfficialBenchmark().load_questions()


def test_generate_and_evaluate_reports_subject_accuracy(monkeypatch):
    _install_dataset(
        monkeypatch,
        {
            "OE_TO_maths_en_COMP": [_row(solution=["SECRET_SOLUTION"], final_answer=["42"])],
            "OE_TO_physics_en_COMP": [
                _row(
                    id=802,
                    subject="Physics",
                    question="What is 2 + 2?",
                    context="Use SI units.",
                    final_answer=["4"],
                    unit="m/s",
                )
            ],
        },
    )
    model = _RecordingLM([["Reasoning. \\boxed{42}", "Incorrect. \\boxed{5}"]])
    benchmark = olympiad.OlympiadBenchOfficialBenchmark(debug=True)

    generated = benchmark.generate_responses(model)
    scored = benchmark.evaluate_responses(generated)
    samples = benchmark.to_samples(scored)

    assert scored["benchmark_scope"] == "english_open_ended_text_only"
    assert scored["benchmark_protocol"] == "openbmb_english_oe_to_v1"
    assert scored["reference_implementation_revision"] == olympiad.REFERENCE_IMPLEMENTATION_REVISION
    assert scored["dataset_revision"] == olympiad.DEFAULT_DATASET_REVISION
    assert scored["num_total"] == 2
    assert scored["num_solved"] == 1
    assert scored["accuracy"] == 0.5
    assert scored["accuracy_subject_Math"] == 1.0
    assert scored["accuracy_subject_Physics"] == 0.0
    assert "SECRET_SOLUTION" not in model.messages[0][0]["content"]
    assert "Answer as an integer.\nWhat is 40 + 2?" in model.messages[0][0]["content"]
    assert "International Math competition" in model.messages[0][0]["content"]
    assert "The answer of The problem should be a numerical value." in model.messages[0][0]["content"]
    assert r'So the final answer is \boxed{answer}."' in model.messages[0][0]["content"]
    assert r'So the final answer is \boxed{answer}(unit)."' in model.messages[1][0]["content"]
    assert r"the unit of the answer should not be included in \boxed{}" in model.messages[1][0]["content"]
    assert model.calls[0][0].args[1]["temperature"] == 0.0
    assert model.calls[0][0].args[1]["do_sample"] is False
    assert model.calls[0][0].metadata["expected_answers"] == ["42"]
    assert samples[0]["arguments"][0][0] == model.messages[0][0]["content"]


def test_evaluate_responses_requires_complete_candidate_and_applies_precision():
    benchmark = olympiad.OlympiadBenchOfficialBenchmark()
    results = {
        "examples": [
            {
                "answer": ["$1$, $2$"],
                "error": ",1e-3",
                "model_output": "Only one component: \\boxed{1}",
                "subject": "Physics",
            },
            {
                "answer": ["$1$, $2$"],
                "error": ",1e-3",
                "model_output": "\\boxed{2.0005}\n\\boxed{1}",
                "subject": "Physics",
            },
            {
                "answer": ["7", "8"],
                "error": None,
                "model_output": "Alternative form: \\boxed{8}",
                "subject": "Math",
            },
        ]
    }

    scored = benchmark.evaluate_responses(results)

    assert [example["is_correct"] for example in scored["examples"]] == [
        False,
        True,
        True,
    ]
    assert scored["num_solved"] == 2
    assert scored["accuracy"] == pytest.approx(2 / 3)


def test_pass_at_k_scores_raw_samples_and_reports_subject_metrics(monkeypatch):
    _install_dataset(
        monkeypatch,
        {
            "OE_TO_maths_en_COMP": [_row(final_answer=["42"])],
            "OE_TO_physics_en_COMP": [
                _row(
                    id=802,
                    subject="Physics",
                    final_answer=["$1$, $2$"],
                    is_multiple_answer=False,
                    answer_type="Equation,Numerical",
                )
            ],
        },
    )
    model = _RecordingLM(
        [
            ["\\boxed{42}", "\\boxed{5}"],
            ["\\boxed{0}", "\\boxed{1}\n\\boxed{2}"],
        ]
    )
    benchmark = olympiad.OlympiadBenchOfficialBenchmark(
        debug=True,
        num_samples=2,
        pass_at_k=[1, 2],
    )

    scored = benchmark.evaluate_responses(benchmark.generate_responses(model))

    assert scored["num_correct"] == [1, 1]
    assert scored["pass@1"] == 0.5
    assert scored["pass@2"] == 1.0
    assert scored["pass@1_subject_Math"] == 0.5
    assert scored["pass@1_subject_Physics"] == 0.5
    assert "answers in order being an equation, a numerical value" in model.messages[1][0]["content"]
    assert "multiple answers connected with commas" in model.messages[1][0]["content"]
    assert all(
        instance.args[1]["temperature"] == 0.7 and instance.args[1]["do_sample"] is True
        for call in model.calls
        for instance in call
    )
