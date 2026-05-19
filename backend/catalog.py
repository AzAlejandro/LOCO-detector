from __future__ import annotations

from .plugins.experiments import build_default_experiments
from .registry import ExperimentRegistry


def build_registry() -> ExperimentRegistry:
    reg = ExperimentRegistry()
    for exp in build_default_experiments():
        reg.register(exp)
    return reg
