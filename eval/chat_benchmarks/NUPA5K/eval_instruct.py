"""Fixed deterministic 5,000-record panel from the canonical NUPA source."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from eval.chat_benchmarks.NUPA.eval_instruct import (
    DEFAULT_SPLIT,
    SOURCE_DATASET_NAME,
    SOURCE_DATASET_REVISION,
    NUPABenchmark,
    download_nupa_source,
)

from .panel import NUPA5K_SIZE, load_nupa5k_manifest, load_panel_records


class NUPA5KBenchmark(NUPABenchmark):
    """Evaluate the fixed stratified NUPA5K convenience panel."""

    def __init__(
        self,
        dataset_name: str = SOURCE_DATASET_NAME,
        dataset_revision: str = SOURCE_DATASET_REVISION,
        dataset_split: str = DEFAULT_SPLIT,
        source_file: str | None = None,
        max_tokens: int = 256,
        debug: bool = False,
        logger: logging.Logger | None = None,
        system_instruction: str | None = None,
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset_revision=dataset_revision,
            dataset_split=dataset_split,
            max_tokens=max_tokens,
            debug=debug,
            logger=logger,
            system_instruction=system_instruction,
        )
        self.source_file = source_file

    def benchmark_size(self) -> int:
        if self.debug:
            size = super().benchmark_size()
            assert size is not None
            return size
        return NUPA5K_SIZE

    def _load_records(self) -> list[dict[str, Any]]:
        if self.debug:
            return super()._load_records()
        source = (
            Path(self.source_file)
            if self.source_file
            else download_nupa_source(
                dataset_name=self.dataset_name,
                dataset_revision=self.dataset_revision,
                split=self.dataset_split,
            )
        )
        records = load_panel_records(
            source,
            split=self.dataset_split,
            identities=load_nupa5k_manifest(),
        )
        return self.limit_samples(records)
