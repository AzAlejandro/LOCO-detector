from __future__ import annotations

from typing import Any

import numpy as np


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def profile_npz_payload(points: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    rows: list[tuple[float, float, float, float, float, float]] = []
    candidate_counts: list[tuple[int, int, int]] = []
    for point in points:
        point_index = int(point.get('point_index', -1))
        edge = dict(point.get('edge_pairs') or {})
        for prof in list(edge.get('profiles') or []):
            rows.append(
                (
                    float(point_index),
                    float(prof.get('profile_index', -1)),
                    float(prof.get('diameter_px', 0.0)),
                    float(prof.get('pair_score', 0.0)),
                    float(prof.get('edge_score', 0.0)),
                    1.0 if bool(prof.get('accepted_final')) else 0.0,
                )
            )
        for item in list((edge.get('raw') or {}).get('edge_candidates_by_profile') or []):
            candidate_counts.append(
                (
                    int(point_index),
                    int(item.get('profile_index', -1)),
                    int(len(item.get('left_candidates') or []) + len(item.get('right_candidates') or [])),
                )
            )
    return {
        'profile_rows': np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 6), dtype=np.float32),
        'candidate_counts': np.asarray(candidate_counts, dtype=np.int32) if candidate_counts else np.zeros((0, 3), dtype=np.int32),
    }
