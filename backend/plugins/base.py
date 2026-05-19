from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RunContext:
    image_rgb: np.ndarray
    image_gray_f: np.ndarray
    labels: np.ndarray
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunOutput:
    mask: np.ndarray
    prior_map: np.ndarray | None = None
    prob_maps: dict[str, np.ndarray] | None = None
    meta: dict[str, Any] | None = None
    status_level: str = 'success'
