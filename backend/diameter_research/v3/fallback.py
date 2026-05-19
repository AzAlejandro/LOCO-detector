from __future__ import annotations

from typing import Any

import numpy as np

from ..profiles import sample_bilinear


def _param_bool(params: dict[str, Any] | None, key: str, default: bool) -> bool:
    value = (params or {}).get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def _param_float(params: dict[str, Any] | None, key: str, default: float) -> float:
    try:
        return float((params or {}).get(key, default))
    except Exception:
        return float(default)


def _unit(v: np.ndarray, fallback: tuple[float, float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n < 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return a / n


def measure_diameter_fallback(
    *,
    support_weight: np.ndarray,
    center_xy: tuple[float, float],
    orientation: dict[str, Any],
    geometry: dict[str, Any],
    recenter: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    if not _param_bool(params, 'fallback_enabled', True):
        return {'status': 'skipped', 'reason': 'fallback_disabled'}
    if str(geometry.get('geometry_status') or '') != 'geometry_simple':
        return {'status': 'skipped', 'reason': 'geometry_not_simple'}
    if str(recenter.get('recenter_status') or 'ok') not in {'ok'}:
        return {'status': 'skipped', 'reason': 'recenter_not_stable'}

    weight = np.asarray(support_weight, dtype=np.float32)
    normal = _unit(np.asarray(orientation.get('normal', [0.0, 1.0]), dtype=np.float64), (0.0, 1.0))
    half = max(8.0, _param_float(params, 'profile_length_px', 80.0) * 0.5)
    distances = np.linspace(-half, half, int(round(half * 2)) + 1, dtype=np.float32)
    xs = float(center_xy[0]) + distances.astype(np.float64) * float(normal[0])
    ys = float(center_xy[1]) + distances.astype(np.float64) * float(normal[1])
    signal = sample_bilinear(weight, xs, ys, default=0.0)
    center_idx = int(np.argmin(np.abs(distances)))
    threshold = max(0.18, float(np.percentile(signal[signal > 0], 30)) if np.any(signal > 0) else 0.18)
    inside = signal >= threshold
    if not inside[center_idx]:
        return {'status': 'failed', 'reason': 'center_outside_support'}

    left_idx = center_idx
    while left_idx > 0 and bool(inside[left_idx - 1]):
        left_idx -= 1
    right_idx = center_idx
    while right_idx < len(inside) - 1 and bool(inside[right_idx + 1]):
        right_idx += 1
    width = float(distances[right_idx] - distances[left_idx])
    if width < 2.0:
        return {'status': 'failed', 'reason': 'fallback_width_too_small', 'diameter_px': width}
    left_xy = [float(center_xy[0] + float(distances[left_idx]) * normal[0]), float(center_xy[1] + float(distances[left_idx]) * normal[1])]
    right_xy = [float(center_xy[0] + float(distances[right_idx]) * normal[0]), float(center_xy[1] + float(distances[right_idx]) * normal[1])]
    score = float(np.clip(np.mean(signal[left_idx:right_idx + 1]), 0.0, 1.0))
    return {
        'status': 'ok',
        'reason': '',
        'diameter_px': width,
        'left_edge_xy': left_xy,
        'right_edge_xy': right_xy,
        'fallback_score': score,
        'edge_pair_score': score,
        'profile_consensus': 0.55,
        'valid_profiles': 1,
        'total_profiles': 1,
    }
