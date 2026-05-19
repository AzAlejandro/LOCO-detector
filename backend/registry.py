from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from .plugins.base import ExperimentPlugin


class ExperimentRegistry:
    def __init__(self) -> None:
        self._experiments: dict[str, ExperimentPlugin] = {}

    def register(self, exp: ExperimentPlugin) -> None:
        eid = str(exp.info.experiment_id).strip().lower()
        if not eid:
            raise ValueError('experiment_id invalido')
        self._experiments[eid] = exp

    def get(self, experiment_id: str) -> ExperimentPlugin:
        key = str(experiment_id or '').strip().lower()
        if key not in self._experiments:
            raise HTTPException(status_code=400, detail=f'Experimento no soportado: {experiment_id}')
        return self._experiments[key]

    def list(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for key in sorted(self._experiments.keys()):
            info = self._experiments[key].info
            out.append(
                {
                    'experiment_id': info.experiment_id,
                    'group': info.group,
                    'display_name': info.display_name,
                    'description': info.description,
                    'default_params': dict(info.default_params),
                    'implementation_status': info.implementation_status,
                    'requirements_hint': info.requirements_hint,
                }
            )
        return out
