from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np


ImplementationStatus = Literal['native', 'fallback', 'proxy']
RunStatusLevel = Literal['success', 'warning', 'error']


@dataclass(frozen=True)
class ExperimentInfo:
    experiment_id: str
    group: str
    display_name: str
    description: str
    default_params: dict[str, Any]
    implementation_status: ImplementationStatus = 'native'
    requirements_hint: str = ''


@dataclass
class RunContext:
    image_rgb: np.ndarray
    image_gray_f: np.ndarray
    labels: np.ndarray
    params: dict[str, Any]


@dataclass
class ExperimentOutput:
    prior_map: np.ndarray
    mask: np.ndarray
    meta: dict[str, Any]
    status_level: RunStatusLevel = 'success'
    prob_maps: dict[str, np.ndarray] | None = None


class ExperimentPlugin(Protocol):
    info: ExperimentInfo

    def run(self, ctx: RunContext) -> ExperimentOutput:
        ...
