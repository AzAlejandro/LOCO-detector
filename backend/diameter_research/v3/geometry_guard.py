from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _param_float(params: dict[str, Any] | None, key: str, default: float) -> float:
    try:
        return float((params or {}).get(key, default))
    except Exception:
        return float(default)


def _param_int(params: dict[str, Any] | None, key: str, default: int) -> int:
    try:
        return int(round(float((params or {}).get(key, default))))
    except Exception:
        return int(default)


def _window(mask: np.ndarray, point_xy: tuple[float, float], radius: int) -> np.ndarray:
    h, w = mask.shape[:2]
    x = int(round(float(point_xy[0])))
    y = int(round(float(point_xy[1])))
    r = max(4, int(radius))
    return mask[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)]


def evaluate_local_geometry_ambiguity(
    *,
    gray_f: np.ndarray,
    support_refined: np.ndarray,
    center_xy: tuple[float, float],
    orientation: dict[str, Any],
    profiles: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, Any]:
    hard = (np.asarray(support_refined) > 0).astype(np.uint8)
    radius = max(8, _param_int(params, 'geometry_window_px', 48) // 2)
    crop = _window(hard, center_xy, radius)
    components = 0
    if np.any(crop):
        ncc, _cc, stats, _cent = cv2.connectedComponentsWithStats(crop, connectivity=8)
        for cid in range(1, int(ncc)):
            if int(stats[cid, cv2.CC_STAT_AREA]) >= 8:
                components += 1

    coherence = float(orientation.get('orientation_coherence', orientation.get('confidence', 0.0)) or 0.0)
    min_coherence = _param_float(params, 'min_orientation_coherence', 0.18)
    accepted = [float(p.get('diameter_px', 0.0)) for p in profiles if bool(p.get('accepted_final')) and float(p.get('diameter_px', 0.0)) > 0]
    width_spread = 0.0
    width_bimodal = False
    if accepted:
        vals = np.sort(np.asarray(accepted, dtype=np.float32))
        med = max(1e-6, float(np.median(vals)))
        width_spread = float((float(np.max(vals)) - float(np.min(vals))) / med)
        if len(vals) >= 4:
            gaps = np.diff(vals)
            width_bimodal = bool(gaps.size and float(np.max(gaps)) / med > _param_float(params, 'bimodal_width_gap_ratio', 0.22))

    gray = np.asarray(gray_f, dtype=np.float32)
    image_crop = _window(gray, center_xy, radius)
    texture_std = float(np.std(image_crop)) if image_crop.size else 0.0

    flags: list[str] = []
    status = 'geometry_simple'
    if components >= 2:
        flags.append('multiple_components_nearby')
        status = 'geometry_ambiguous'
    if coherence < min_coherence:
        flags.append('low_orientation_coherence')
        status = 'geometry_ambiguous' if status != 'crossing_likely' else status
    if width_bimodal or width_spread > 0.50:
        flags.append('width_inconsistent')
        status = 'geometry_ambiguous'
    if components <= 1 and coherence < min_coherence * 0.75 and texture_std > 0.03:
        flags.append('crossing_likely')
        status = 'crossing_likely'
    elif status == 'geometry_simple' and (width_spread > 0.28 or coherence < min_coherence * 1.4):
        status = 'geometry_complex_but_measurable'

    return {
        'geometry_status': status,
        'quality_flags': sorted(set(flags)),
        'components_nearby': int(components),
        'orientation_coherence': float(coherence),
        'width_spread_ratio': float(width_spread),
        'width_bimodal': bool(width_bimodal),
        'texture_std': float(texture_std),
    }
