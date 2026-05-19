from __future__ import annotations

from typing import Any

import cv2
import numpy as np


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


def _odd_kernel(px: float, *, minimum: int = 5) -> int:
    k = max(minimum, int(round(float(px))))
    if k % 2 == 0:
        k += 1
    return k


def _normalize01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    mn = float(np.min(a)) if a.size else 0.0
    mx = float(np.max(a)) if a.size else 0.0
    if mx <= mn + 1e-6:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - mn) / (mx - mn)).astype(np.float32)


def build_local_preprocess_diagnostics(
    *,
    gray_f: np.ndarray,
    roi_bbox: tuple[int, int, int, int],
    params: dict[str, Any],
) -> dict[str, Any]:
    if not _param_bool(params, 'local_preprocess_diagnostics_enabled', False):
        return {'enabled': False}
    gray = np.asarray(gray_f, dtype=np.float32)
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in roi_bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return {'enabled': True, 'status': 'empty_roi'}
    crop = gray[y0:y1, x0:x1]

    radius = _odd_kernel(_param_float(params, 'rolling_ball_radius_px', 28.0))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    background = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernel)
    background_corrected = _normalize01(crop - background)

    clip = max(0.1, _param_float(params, 'clahe_clip_limit', 1.6))
    tile_px = max(4, int(round(_param_float(params, 'clahe_tile_grid_px', 24.0))))
    tiles = (max(1, int(np.ceil(crop.shape[1] / tile_px))), max(1, int(np.ceil(crop.shape[0] / tile_px))))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    clahe_img = clahe.apply((_normalize01(crop) * 255.0).astype(np.uint8)).astype(np.float32) / 255.0

    sigma = max(0.3, _param_float(params, 'ridge_sigma_px', 1.2))
    smooth = cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma, sigmaY=sigma)
    gxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3)
    gyy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3)
    gxy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3)
    trace = gxx + gyy
    det_term = np.sqrt(np.maximum((gxx - gyy) * (gxx - gyy) + 4.0 * gxy * gxy, 0.0))
    lambda_dark = 0.5 * (trace - det_term)
    lambda_bright = 0.5 * (trace + det_term)
    ridge_response = _normalize01(np.maximum(np.abs(lambda_dark), np.abs(lambda_bright)))

    return {
        'enabled': True,
        'status': 'ok',
        'bbox_xyxy': [int(x0), int(y0), int(x1), int(y1)],
        'rolling_ball_radius_px': int(radius),
        'clahe_clip_limit': float(clip),
        'clahe_tile_grid': [int(tiles[0]), int(tiles[1])],
        'ridge_sigma_px': float(sigma),
        'background_mean': float(np.mean(background)) if background.size else 0.0,
        'corrected_mean': float(np.mean(background_corrected)) if background_corrected.size else 0.0,
        'clahe_mean': float(np.mean(clahe_img)) if clahe_img.size else 0.0,
        'ridge_mean': float(np.mean(ridge_response)) if ridge_response.size else 0.0,
        'ridge_p95': float(np.percentile(ridge_response, 95)) if ridge_response.size else 0.0,
    }
