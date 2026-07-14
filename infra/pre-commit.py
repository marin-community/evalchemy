#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     # Pin to an exact marin-style revision so every contributor and CI run
#     # uses the same checks. Bump <REV> to adopt a new version.
#     "marin-style @ git+https://github.com/marin-community/marin-style@999ea41d48c05db6f78552ddfa8d3cc6d49cd37f",
# ]
# ///
"""Consumer-repo pre-commit shim. Delegates to the shared marin-style checks.

Committed at `infra/pre-commit.py`; the `commit` skill and CI invoke it directly.
"""

from marin_style.precommit import main

if __name__ == "__main__":
    raise SystemExit(main())
