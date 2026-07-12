# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavior test for run_evals' one non-obvious rule: sample-limit resolution.

Everything else run_evals does is thin plumbing -- click flag parsing, and
threading values into a provider / EvalInvocation -- or is covered at its real
contract layer (argv construction in test_eval_args, provider commands in
test_providers). The only surprising behavior that lives *here* is that
``--limit 0`` (or negative) means "run the FULL task", because lm-eval reads a
missing ``--limit`` as "all samples". A regression there silently caps a full
run at 0 samples (or vice-versa), so it is worth pinning.
"""

import pytest

from eval.e2e.run_evals import resolve_limit


@pytest.mark.parametrize(
    "cli_limit, config_limit, expected",
    [
        (None, None, 200),  # no CLI, no config -> the human default (a real-ish eval)
        (None, 20, 20),  # config supplies it (CI sets 20 for a fast smoke)
        (50, 20, 50),  # CLI overrides config
        (0, 20, None),  # 0 is the escape hatch: run the FULL task, not a cap of 0
        (-1, None, None),  # negative is also "full task"
    ],
)
def test_resolve_limit(cli_limit, config_limit, expected):
    assert resolve_limit(cli_limit, config_limit) == expected
