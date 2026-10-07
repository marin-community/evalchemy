"""The optimization panel must remain a broad subset of the policy panel."""

import json
from collections import Counter
from importlib import import_module


def test_nupa200_membership_and_runtime_selection():
    small = import_module("eval.chat_benchmarks.NUPA200-Loose.eval_instruct")
    parent = import_module("eval.chat_benchmarks.NUPA5K-Loose.panel")
    manifest = [json.loads(line) for line in small.MANIFEST_PATH.read_text().splitlines()]
    identities = {parent.SourceIdentity(**record) for record in manifest}
    policy_ids = set(parent.load_nupa5k_manifest())
    assert len(identities) == 200
    assert identities < policy_ids
    counts = Counter(record["task_name"] for record in manifest)
    assert len(counts) == 44
    assert set(counts.values()) == {4, 5}
    benchmark = small.NUPA200LooseBenchmark()
    records = benchmark._load_records()
    assert len(records) == 200
    assert {record["id"] for record in records} == {identity.as_record_id("test") for identity in identities}
    benchmark.set_evaluation_limits(limit=17)
    assert benchmark._load_records() == records[:17]
