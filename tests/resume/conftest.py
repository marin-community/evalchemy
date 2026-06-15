"""Make the repo root importable so `import eval.resume` resolves to the in-tree package.

The resume package is pure stdlib (no torch / lm_eval / numpy), so these tests run on any
Python with pytest, independent of the evalchemy GPU env.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
