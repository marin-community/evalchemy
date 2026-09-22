"""Rebuild the checked-in NUPA5K identity manifest from the pinned source."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.chat_benchmarks.NUPA5K.panel import MANIFEST_PATH, NUPA5K_SIZE, build_nupa5k_identities, write_manifest
from eval.chat_benchmarks.NUPA.eval_instruct import (
    DEFAULT_SPLIT,
    SOURCE_DATASET_NAME,
    SOURCE_DATASET_REVISION,
    download_nupa_source,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=SOURCE_DATASET_NAME)
    parser.add_argument("--revision", default=SOURCE_DATASET_REVISION)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--panel-size", type=int, default=NUPA5K_SIZE)
    args = parser.parse_args()

    source = args.source_file or download_nupa_source(
        dataset_name=args.dataset_name,
        dataset_revision=args.revision,
        split=args.split,
    )
    identities = build_nupa5k_identities(source, panel_size=args.panel_size)
    digest = write_manifest(args.output, identities)
    print(f"Wrote {len(identities)} NUPA5K identities to {args.output} (sha256={digest})")


if __name__ == "__main__":
    main()
