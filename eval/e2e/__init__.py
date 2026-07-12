# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""End-to-end test harness: serve a model (via Marin) and evaluate it with evalchemy.

See ``eval/e2e/README.md``. Two CLIs, split by audience:

* :mod:`eval.e2e.run_evals`  -- **human** tool: serve a model (or attach to one),
  run the eval, print the results. No pass/fail.
* :mod:`eval.e2e.validate`   -- **CI** tool: ``check`` a run against a golden
  baseline (exit 0/1), or ``record`` a new baseline.

Supporting library:

* :mod:`eval.e2e.models`      -- pydantic models for config / baseline / results.
* :mod:`eval.e2e.compare`     -- the gate (``EvalResults`` vs ``Baseline``).
* :mod:`eval.e2e.providers`   -- resolve a served model into an :class:`ServedModel`.
* :mod:`eval.e2e.eval_args`   -- build the ``eval.eval`` argv for a served endpoint.

Only :mod:`eval.e2e.providers` (the ``marin-serve`` provider) touches Marin; the
``endpoint`` provider and everything downstream of it are dependency-free so the
harness's core is unit-testable on stock (hosted) CI with no accelerator.
"""
