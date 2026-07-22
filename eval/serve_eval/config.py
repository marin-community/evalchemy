# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""The run config for the serve-and-eval runner (e.g. ``configs/qwen-tiny.yaml``).

Describes what to serve and how to evaluate it. Layered by :meth:`RunConfig.load`:
CLI overrides win, then ``E2E_*`` env vars, then the yaml file, then these defaults.
The gate spec a run is checked against lives with the regression gate
(:mod:`eval.regression`).
"""

from __future__ import annotations

import os
from typing import Any, ClassVar, Dict, List, Optional, Union

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource


class RunConfig(BaseSettings):
    """What to serve and evaluate."""

    model_config = SettingsConfigDict(env_prefix="E2E_", extra="forbid", protected_namespaces=())

    _yaml_path: ClassVar[Optional[str]] = None

    # what to serve
    model: Optional[str] = None
    model_revision: Optional[str] = None
    tokenizer: Optional[str] = None

    # the evalchemy run
    tasks: List[str] = Field(default_factory=lambda: ["gsm8k"])
    apply_chat_template: bool = False
    limit: Optional[int] = 200
    num_fewshot: Optional[int] = None
    batch_size: Union[int, str] = 1
    seed: Optional[int] = 1234
    gen_kwargs: Optional[str] = None
    extra_model_args: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)

    # marin-serve provider
    cluster: str = "marin"
    tpu: str = "v5litepod-8"
    region: Optional[str] = None
    wait_timeout_s: float = 1800.0
    timeout_hours: float = 2.0
    marin_workspace: Optional[str] = None

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        sources = [init_settings, env_settings]
        if cls._yaml_path:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=cls._yaml_path))
        return tuple(sources)

    @classmethod
    def load(cls, path: Optional[str] = None, **overrides: Any) -> "RunConfig":
        """Load from ``path`` (if it exists) + ``E2E_*`` env + CLI ``overrides``.

        ``None`` overrides are dropped so an unset CLI flag can't clobber the file.
        """
        cls._yaml_path = path if (path and os.path.exists(path)) else None
        return cls(**{k: v for k, v in overrides.items() if v is not None})
