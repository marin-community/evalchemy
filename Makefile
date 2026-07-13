PYTHON_VERSION ?= 3.11

# Evalchemy uses uv (https://docs.astral.sh/uv/). `uv sync` resolves from the
# committed uv.lock into a local .venv. The base install is vLLM-free (see
# pyproject); add extras as needed:
#   uv sync                 # base + dev
#   uv sync --extra vllm    # + local vLLM inference engine
install:
	@echo "Syncing dependencies with uv (Python $(PYTHON_VERSION))"
	uv sync --python $(PYTHON_VERSION)
	@echo "Installing pre-commit hooks"
	uv run pre-commit install

# Refresh the lockfile after changing dependencies in pyproject.toml.
lock:
	uv lock --python $(PYTHON_VERSION)

test:
	uv run --python $(PYTHON_VERSION) pytest

.PHONY: install lock test
