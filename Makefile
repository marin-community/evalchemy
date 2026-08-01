PYTHON_VERSION ?= 3.12

# Evalchemy uses uv (https://docs.astral.sh/uv/). `uv sync` resolves from the
# committed uv.lock into a local .venv. The base install is the lean, torch-free
# endpoint-eval core (see pyproject); add extras as needed:
#   uv sync                       # the lean core
#   uv sync --extra mtbench       # + one benchmark's deps (extra == the dir name)
#   uv sync --extra benchmarks    # + every benchmark under eval/chat_benchmarks
#   uv sync --extra vllm          # + local vLLM inference engine
# `make install` adds dev (pytest, pre-commit) and serve-eval (the runner + gate,
# which `make test`'s tests/e2e suite exercises).
install:
	@echo "Syncing dependencies with uv (Python $(PYTHON_VERSION))"
	uv sync --python $(PYTHON_VERSION) --extra dev --extra serve-eval
	@echo "Installing pre-commit hooks"
	uv run pre-commit install

# Refresh the lockfile after changing dependencies in pyproject.toml.
lock:
	uv lock --python $(PYTHON_VERSION)

test:
	uv run --python $(PYTHON_VERSION) --extra dev --extra serve-eval pytest

.PHONY: install lock test
