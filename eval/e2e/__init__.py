# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""End-to-end test harness: serve a model (via Marin) and evaluate it with evalchemy.

See ``eval/e2e/README.md``. The public surface is:

* :mod:`eval.e2e.eval_args`  -- build the ``eval.eval`` argv for a served endpoint.
* :mod:`eval.e2e.compare`    -- gate a ``results_*.json`` against a baseline.
* :mod:`eval.e2e.providers`  -- resolve a served model into an :class:`ServedModel`.
* :mod:`eval.e2e.run_e2e`    -- the one-shot orchestrator (CLI entrypoint).

Only :mod:`eval.e2e.providers` (the ``marin-serve`` provider) touches Marin; the
``endpoint`` provider and everything downstream of it are dependency-free so the
harness's core is unit-testable on stock (hosted) CI with no accelerator.
"""
