from __future__ import annotations

from typing import Any

import cv2
import numpy as np


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


def _window_bounds(x: float, y: float, shape_hw: tuple[int, int], window_px: int) -> tuple[int, int, int, int]:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    half = max(2, int(window_px) // 2)
    x0 = max(0, int(round(x)) - half)
    y0 = max(0, int(round(y)) - half)
    x1 = min(w, int(round(x)) + half + 1)
    y1 = min(h, int(round(y)) + half + 1)
    return x0, y0, x1, y1


def _pca_orientation(support_crop: np.ndarray, x0: int, y0: int) -> dict[str, Any] | None:
    ys, xs = np.where(np.asarray(support_crop) > 0)
    if len(xs) < 8:
        return None
    coords = np.column_stack([xs.astype(np.float64) + float(x0), ys.astype(np.float64) + float(y0)])
    coords -= np.mean(coords, axis=0, keepdims=True)
    cov = np.cov(coords, rowvar=False)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return None
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    tangent = _unit(vecs[:, 0], (1.0, 0.0))
    if tangent[0] < 0:
        tangent = -tangent
    normal = _unit(np.array([-tangent[1], tangent[0]]), (0.0, 1.0))
    denom = float(vals[0] + vals[1] + 1e-9)
    conf = float(np.clip((float(vals[0]) - float(vals[1])) / denom, 0.0, 1.0))
    return {
        'source': 'pca_support',
        'tangent': [float(tangent[0]), float(tangent[1])],
        'normal': [float(normal[0]), float(normal[1])],
        'confidence': conf,
        'support_pixels': int(len(xs)),
    }


def _structure_tensor_orientation(gray_crop: np.ndarray) -> dict[str, Any]:
    crop = np.asarray(gray_crop, dtype=np.float32)
    if crop.size == 0:
        tangent = np.asarray([1.0, 0.0], dtype=np.float64)
        normal = np.asarray([0.0, 1.0], dtype=np.float64)
        return {
            'source': 'structure_tensor',
            'tangent': [float(tangent[0]), float(tangent[1])],
            'normal': [float(normal[0]), float(normal[1])],
            'confidence': 0.0,
            'support_pixels': 0,
        }
    crop = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.2, sigmaY=1.2)
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    mat = np.asarray([[jxx, jxy], [jxy, jyy]], dtype=np.float64)
    if not np.isfinite(mat).all():
        tangent = np.asarray([1.0, 0.0], dtype=np.float64)
        normal = np.asarray([0.0, 1.0], dtype=np.float64)
        conf = 0.0
    else:
        vals, vecs = np.linalg.eigh(mat)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        normal = _unit(vecs[:, 0], (0.0, 1.0))
        tangent = _unit(np.array([-normal[1], normal[0]]), (1.0, 0.0))
        if tangent[0] < 0:
            tangent = -tangent
            normal = -normal
        denom = float(vals[0] + vals[1] + 1e-9)
        conf = float(np.clip((float(vals[0]) - float(vals[1])) / denom, 0.0, 1.0))
    return {
        'source': 'structure_tensor',
        'tangent': [float(tangent[0]), float(tangent[1])],
        'normal': [float(normal[0]), float(normal[1])],
        'confidence': conf,
        'support_pixels': 0,
    }


def estimate_orientation(
    *,
    gray_u8: np.ndarray,
    support: np.ndarray,
    point_xy: tuple[float, float],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gray = np.asarray(gray_u8)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_f = np.asarray(gray, dtype=np.float32) / 255.0
    h, w = gray_f.shape[:2]
    x, y = float(point_xy[0]), float(point_xy[1])
    window = max(9, _param_int(params, 'local_window_px', 41))
    x0, y0, x1, y1 = _window_bounds(x, y, (h, w), window)
    support_crop = np.asarray(support, dtype=np.uint8)[y0:y1, x0:x1]
    pca = _pca_orientation(support_crop, x0, y0)
    if pca is not None:
        return pca
    out = _structure_tensor_orientation(gray_f[y0:y1, x0:x1])
    out['support_pixels'] = int(np.sum(support_crop > 0))
    return out
