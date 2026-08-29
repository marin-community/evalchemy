from eval.chat_benchmarks.NUPA.eval_instruct import NUPABenchmark, flatten_nupa_row, split_prompt_answer
from eval.task import TaskManager


def test_split_prompt_answer_keeps_equals_in_prompt():
    prompt, answer = split_prompt_answer("Get the maximal number: 9.11 and 9.9 = 9.9")
    assert prompt == "Get the maximal number: 9.11 and 9.9 ="
    assert answer == "9.9"


def test_flatten_nupa_row_adds_metadata():
    rows = flatten_nupa_row(
        {
            "max_Float_Float_Float": {
                "3": [
                    "Directly return the answer as a float without any comma separator, like 10.4 . "
                    "Get the maximal number: 9.11 and 9.9 = 9.9"
                ]
            }
        },
        split="test",
    )
    assert rows == [
        {
            "id": "test:max_Float_Float_Float:3:000000",
            "task_name": "max_Float_Float_Float",
            "operation": "max",
            "answer_format": "Float",
            "digit": 3,
            "length_bucket": "S",
            "prompt": "Directly return the answer as a float without any comma separator, like 10.4 . "
            "Get the maximal number: 9.11 and 9.9 =",
            "answer": "9.9",
        }
    ]


def test_flatten_nupa_row_maps_scalar_int_output_to_integer_format():
    rows = flatten_nupa_row(
        {"get_digit_Integer_int_int": {"3": ["Get digit: 123 and 2 = 2"]}},
        split="test",
    )

    assert rows[0]["answer_format"] == "Integer"


def test_nupa_benchmark_aggregates_overall_task_and_bucket_metrics():
    benchmark = NUPABenchmark(debug=True)
    results = {
        "examples": [
            {
                "task_name": "max_Float_Float_Float",
                "length_bucket": "S",
                "answer_format": "Float",
                "answer": "9.9",
                "output": "9.9",
            },
            {
                "task_name": "max_Float_Float_Float",
                "length_bucket": "S",
                "answer_format": "Float",
                "answer": "9.9",
                "output": "9.11",
            },
        ]
    }
    metrics = benchmark.evaluate_responses(results)
    assert metrics["exact_match"] == 0.5
    assert metrics["task:max_Float_Float_Float/exact_match"] == 0.5
    assert metrics["bucket:S/exact_match"] == 0.5
    assert metrics["task:max_Float_Float_Float/bucket:S/exact_match"] == 0.5
    assert metrics["dataset_num_samples"] == 2.0


def test_task_manager_loads_nupa_native_benchmark():
    task_manager = TaskManager(task_list=["NUPA"])
    assert task_manager.is_valid_task("NUPA")
