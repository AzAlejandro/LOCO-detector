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


def _unit(v: np.ndarray, fallback: tuple[float, float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n < 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return a / n


def _crop_bounds(point_xy: tuple[float, float], shape_hw: tuple[int, int], window_px: int) -> tuple[int, int, int, int]:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    half = max(4, int(window_px) // 2)
    x = int(round(float(point_xy[0])))
    y = int(round(float(point_xy[1])))
    return max(0, x - half), max(0, y - half), min(w, x + half + 1), min(h, y + half + 1)


def estimate_local_orientation_from_image(
    *,
    gray_f: np.ndarray,
    support: np.ndarray,
    point_xy: tuple[float, float],
    params: dict[str, Any],
) -> dict[str, Any]:
    gray = np.asarray(gray_f, dtype=np.float32)
    hard = (np.asarray(support) > 0).astype(np.float32)
    h, w = gray.shape[:2]
    window = max(13, _param_int(params, 'geometry_window_px', 48))
    x0, y0, x1, y1 = _crop_bounds(point_xy, (h, w), window)
    crop = gray[y0:y1, x0:x1]
    mask = hard[y0:y1, x0:x1]
    if crop.size == 0:
        tangent = np.asarray([1.0, 0.0], dtype=np.float64)
        normal = np.asarray([0.0, 1.0], dtype=np.float64)
        return {
            'source': 'structure_tensor_image',
            'tangent': tangent.tolist(),
            'normal': normal.tolist(),
            'orientation_coherence': 0.0,
            'confidence': 0.0,
            'status': 'image_orientation_failed',
        }

    sigma = max(0.1, _param_float(params, 'orientation_image_sigma', 1.4))
    smooth = cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma, sigmaY=sigma)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    weights = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, sigma), sigmaY=max(1.0, sigma))
    if float(np.sum(weights)) < 1e-6:
        weights = np.ones_like(smooth, dtype=np.float32)
    weights = weights / max(float(np.sum(weights)), 1e-6)
    jxx = float(np.sum(weights * gx * gx))
    jyy = float(np.sum(weights * gy * gy))
    jxy = float(np.sum(weights * gx * gy))
    mat = np.asarray([[jxx, jxy], [jxy, jyy]], dtype=np.float64)
    if not np.isfinite(mat).all():
        normal = np.asarray([0.0, 1.0], dtype=np.float64)
        vals = np.asarray([0.0, 0.0], dtype=np.float64)
    else:
        vals, vecs = np.linalg.eigh(mat)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        normal = _unit(vecs[:, 0], (0.0, 1.0))
    tangent = _unit(np.asarray([-normal[1], normal[0]]), (1.0, 0.0))
    if tangent[0] < 0:
        tangent = -tangent
        normal = -normal
    denom = float(vals[0] + vals[1] + 1e-9)
    coherence = float(np.clip((float(vals[0]) - float(vals[1])) / denom, 0.0, 1.0)) if denom > 0 else 0.0
    min_coherence = _param_float(params, 'min_orientation_coherence', 0.18)
    status = 'ok' if coherence >= min_coherence else 'low_coherence'
    return {
        'source': 'structure_tensor_image',
        'tangent': [float(tangent[0]), float(tangent[1])],
        'normal': [float(normal[0]), float(normal[1])],
        'orientation_coherence': float(coherence),
        'confidence': float(coherence),
        'status': status,
        'bbox_xyxy': [int(x0), int(y0), int(x1), int(y1)],
        'support_pixels': int(np.sum(mask > 0)),
    }
