from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..profiles import sample_bilinear


def _param_int(params: dict[str, Any] | None, key: str, default: int) -> int:
    try:
        return int(round(float((params or {}).get(key, default))))
    except Exception:
        return int(default)


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


def _symmetry_score(gray: np.ndarray, center: tuple[float, float], normal: np.ndarray) -> float:
    distances = np.asarray([2.0, 4.0, 6.0], dtype=np.float32)
    xs_l = float(center[0]) - distances * float(normal[0])
    ys_l = float(center[1]) - distances * float(normal[1])
    xs_r = float(center[0]) + distances * float(normal[0])
    ys_r = float(center[1]) + distances * float(normal[1])
    left = sample_bilinear(gray, xs_l, ys_l, default=0.0)
    right = sample_bilinear(gray, xs_r, ys_r, default=0.0)
    diff = float(np.mean(np.abs(left - right)))
    return float(np.clip(1.0 - diff / 0.35, 0.0, 1.0))


def recenter_point_on_local_axis(
    *,
    gray_f: np.ndarray,
    support_weight: np.ndarray,
    support_refined: np.ndarray,
    point_xy: tuple[float, float],
    orientation: dict[str, Any],
    params: dict[str, Any],
) -> tuple[tuple[float, float], dict[str, Any]]:
    gray = np.asarray(gray_f, dtype=np.float32)
    weight = np.asarray(support_weight, dtype=np.float32)
    hard = (np.asarray(support_refined) > 0).astype(np.uint8)
    h, w = gray.shape[:2]
    x0 = float(np.clip(point_xy[0], 0, max(0, w - 1)))
    y0 = float(np.clip(point_xy[1], 0, max(0, h - 1)))
    radius = max(0, _param_int(params, 'recenter_radius_px', 6))
    max_shift = max(1.0, _param_float(params, 'max_recenter_shift_px', 8.0))
    normal = _unit(np.asarray(orientation.get('normal', [0.0, 1.0]), dtype=np.float64), (0.0, 1.0))

    dist = cv2.distanceTransform(hard, cv2.DIST_L2, 5) if np.any(hard) else np.zeros_like(gray, dtype=np.float32)
    if np.any(dist):
        dist = dist / max(float(np.max(dist)), 1e-6)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2, sigmaY=1.2)
    local_abs = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=6.0, sigmaY=6.0))
    local_abs = local_abs / max(float(np.percentile(local_abs, 98)), 1e-6)

    candidates: list[dict[str, Any]] = []
    best_xy = (x0, y0)
    best_score = -1e9
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            move = float(np.hypot(dx, dy))
            if move > radius or move > max_shift:
                continue
            x = float(np.clip(x0 + dx, 0, max(0, w - 1)))
            y = float(np.clip(y0 + dy, 0, max(0, h - 1)))
            ws = float(sample_bilinear(weight, np.asarray([x]), np.asarray([y]), default=0.0)[0])
            ds = float(sample_bilinear(dist, np.asarray([x]), np.asarray([y]), default=0.0)[0])
            rs = float(sample_bilinear(local_abs, np.asarray([x]), np.asarray([y]), default=0.0)[0])
            sym = _symmetry_score(blur, (x, y), normal)
            score = 0.34 * ws + 0.28 * ds + 0.22 * sym + 0.16 * rs - 0.10 * (move / max(1.0, max_shift))
            candidates.append(
                {
                    'x': x,
                    'y': y,
                    'score': float(score),
                    'support_weight': float(ws),
                    'distance_score': float(ds),
                    'symmetry_score': float(sym),
                    'ridge_score': float(rs),
                    'shift_px': float(move),
                }
            )
            if score > best_score:
                best_score = score
                best_xy = (x, y)
    candidates.sort(key=lambda c: float(c.get('score', 0.0)), reverse=True)
    shift = float(np.hypot(best_xy[0] - x0, best_xy[1] - y0))
    top_scores = [float(c.get('score', 0.0)) for c in candidates[:4]]
    ambiguous = len(top_scores) >= 2 and abs(top_scores[0] - top_scores[1]) < 0.025
    status = 'ambiguous' if ambiguous else ('large_shift' if shift > max_shift else 'ok')
    return best_xy, {
        'original_xy': [float(x0), float(y0)],
        'recentered_xy': [float(best_xy[0]), float(best_xy[1])],
        'shift_px': float(shift),
        'recenter_shift_px': float(shift),
        'recenter_score': float(best_score if np.isfinite(best_score) else 0.0),
        'recenter_status': status,
        'candidates': candidates[:96],
    }
