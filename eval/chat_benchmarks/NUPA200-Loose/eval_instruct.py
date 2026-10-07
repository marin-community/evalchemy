"""Fixed 200-example optimization subset of the policy NUPA5K panel."""

import json
from importlib import import_module
from pathlib import Path

parent = import_module("eval.chat_benchmarks.NUPA5K-Loose.eval_instruct")
panel = import_module("eval.chat_benchmarks.NUPA5K-Loose.panel")
MANIFEST_PATH = Path(__file__).with_name("data") / "nupa200_manifest.jsonl"


class NUPA200LooseBenchmark(parent.NUPA5KLooseBenchmark):
    """Use the parent prompt and scorer on a fixed, task-balanced subset."""

    def benchmark_size(self) -> int:
        return 200

    def _load_records(self) -> list[dict]:
        identities = tuple(
            panel.SourceIdentity(**json.loads(line))
            for line in MANIFEST_PATH.read_text().splitlines()
        )
        source = Path(self.source_file) if self.source_file else parent.nupa.download_nupa_source(
            dataset_name=self.dataset_name,
            dataset_revision=self.dataset_revision,
            split=self.dataset_split,
        )
        records = panel.load_panel_records(source, split=self.dataset_split, identities=identities)
        return self.limit_samples(records)
