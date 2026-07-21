#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     # Pin to an exact marin-style revision so every contributor and CI run
#     # uses the same checks. Bump the hash to adopt a new version.
#     "marin-style @ git+https://github.com/marin-community/marin-style@ccbf03e7ca58486d61fff7a4e73031673d7fd8a4",
# ]
# ///
"""Consumer-repo pre-commit shim. Delegates to the shared marin-style checks.

Committed at `infra/pre-commit.py`; the `commit` skill and CI invoke it directly.
"""

from marin_style.precommit import main

if __name__ == "__main__":
    raise SystemExit(main())
