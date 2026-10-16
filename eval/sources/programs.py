"""Data transformations shared by program-evaluation sources."""

import base64
import json
import os
import pickle
import zlib

from datasets import concatenate_datasets


def difficulty(rating):
    if not rating:
        return "Easy"
    if rating < 1000 or rating > 3500:
        return "Easy"
    return "Medium" if rating < 1300 else "Hard"


def exact_tests(tests):
    normalized = []
    for test in tests:
        output = test["output"].strip()
        if output.endswith("-"):
            output = output[:output.rfind("-")].rstrip()
        normalized.append({"input": test["input"], "output": output})
    return normalized


def live_code_records(rows):
    # Mapping in shards avoids Arrow's offset overflow on the decoded private tests.
    shards = []
    for index in range(4):
        shard = rows.shard(num_shards=4, index=index)
        shard = shard.map(
            lambda row: {"private_test_cases": _private_tests(row["private_test_cases"])},
            num_proc=os.cpu_count(),
        )
        shards.append(shard.map(_live_code_record, remove_columns=rows.column_names))
    return concatenate_datasets(shards)


def _private_tests(encoded_data):
    decoded = zlib.decompress(base64.b64decode(encoded_data))
    return json.loads(pickle.loads(decoded))


def _live_code_record(row):
    public_tests = json.loads(row["public_test_cases"])
    return {
        "prompt": row["question_content"],
        "test": row["private_test_cases"],
        "entry_point": row["starter_code"],
        "task_id": row["question_id"],
        "is_stdin": any(test.get("testtype") == "stdin" for test in public_tests),
        "public_test_cases": row["public_test_cases"],
        "difficulty": row["difficulty"],
    }
