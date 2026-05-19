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


def _param_int(params: dict[str, Any] | None, key: str, default: int) -> int:
    try:
        return int(round(float((params or {}).get(key, default))))
    except Exception:
        return int(default)


def _bounds(point_xy: tuple[float, float], shape_hw: tuple[int, int], radius: int) -> tuple[int, int, int, int]:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    x = int(round(float(point_xy[0])))
    y = int(round(float(point_xy[1])))
    r = max(4, int(radius))
    return max(0, x - r), max(0, y - r), min(w, x + r + 1), min(h, y + r + 1)


def _refine_support_crop(raw: np.ndarray, weight: np.ndarray, gray_crop: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    hard = (np.asarray(raw) > 0).astype(np.uint8)
    wgt = np.asarray(weight, dtype=np.float32)
    if not np.any(hard):
        return hard, np.zeros_like(wgt, dtype=np.float32), {'support_status': 'empty', 'raw_pixels': 0, 'refined_pixels': 0}

    enabled = _param_bool(params, 'support_refine_enabled', True)
    if not enabled:
        return hard, np.where(hard > 0, wgt, 0.0).astype(np.float32), {
            'support_status': 'refine_disabled',
            'raw_pixels': int(np.sum(hard)),
            'refined_pixels': int(np.sum(hard)),
        }

    strength = float(np.clip(_param_float(params, 'support_refine_strength', 0.35), 0.0, 0.85))
    dist = cv2.distanceTransform(hard, cv2.DIST_L2, 5)
    positive = dist[dist > 0]
    max_dist = float(np.max(positive)) if positive.size else 0.0
    thin_mode = _param_bool(params, 'thin_fiber_support_mode', True)
    thin_threshold = max(1.0, _param_float(params, 'thin_fiber_threshold_px', 8.0))
    thin = bool(thin_mode and max_dist * 2.0 <= thin_threshold)

    if thin:
        dist_threshold = max(0.55, max_dist * 0.18)
    else:
        dist_threshold = max(0.75, float(np.percentile(positive, 35)) * strength) if positive.size else 0.75

    core = (dist >= dist_threshold) & (hard > 0)
    strong_weight = (wgt >= max(0.50, float(np.percentile(wgt[hard > 0], 55)) if np.any(hard) else 0.5)) & (hard > 0)

    gray = np.asarray(gray_crop, dtype=np.float32)
    if gray.size and np.any(hard):
        smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2, sigmaY=1.2)
        local = smooth[hard > 0]
        med = float(np.median(local)) if local.size else float(np.median(smooth))
        contrast = np.abs(smooth - med)
        contrast_thr = float(np.percentile(contrast[hard > 0], 35)) if np.any(hard) else 0.0
        image_supported = (contrast >= contrast_thr) & (hard > 0)
    else:
        image_supported = hard > 0

    refined = ((core | (strong_weight & image_supported))).astype(np.uint8)
    if int(np.sum(refined)) < max(4, int(np.sum(hard) * 0.18)):
        refined = core.astype(np.uint8) if np.any(core) else hard
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if not np.any(refined):
        refined = core.astype(np.uint8) if np.any(core) else hard

    refined_weight = np.where(refined > 0, np.maximum(wgt, 0.20), 0.0).astype(np.float32)
    status = 'thin_fiber_refined' if thin else 'refined'
    return refined.astype(np.uint8), refined_weight, {
        'support_status': status,
        'raw_pixels': int(np.sum(hard)),
        'refined_pixels': int(np.sum(refined)),
        'max_distance_px': float(max_dist),
        'thin_fiber_support_mode': bool(thin),
        'refine_strength': float(strength),
    }


def build_local_support_roi(
    *,
    gray_f: np.ndarray,
    support: np.ndarray,
    support_weight: np.ndarray,
    point_xy: tuple[float, float],
    params: dict[str, Any],
) -> dict[str, Any]:
    hard = (np.asarray(support) > 0).astype(np.uint8)
    weight = np.asarray(support_weight, dtype=np.float32)
    gray = np.asarray(gray_f, dtype=np.float32)
    h, w = hard.shape[:2]
    radius = max(8, _param_int(params, 'local_roi_radius_px', 56))
    x0, y0, x1, y1 = _bounds(point_xy, (h, w), radius)
    raw_crop = hard[y0:y1, x0:x1]
    weight_crop = weight[y0:y1, x0:x1]
    gray_crop = gray[y0:y1, x0:x1]
    refined_crop, refined_weight_crop, meta = _refine_support_crop(raw_crop, weight_crop, gray_crop, params)

    raw_global = np.zeros_like(hard, dtype=np.uint8)
    refined_global = np.zeros_like(hard, dtype=np.uint8)
    refined_weight_global = np.zeros_like(weight, dtype=np.float32)
    raw_global[y0:y1, x0:x1] = raw_crop
    refined_global[y0:y1, x0:x1] = refined_crop
    refined_weight_global[y0:y1, x0:x1] = refined_weight_crop

    meta.update(
        {
            'bbox_xyxy': [int(x0), int(y0), int(x1), int(y1)],
            'radius_px': int(radius),
            'point_xy': [float(point_xy[0]), float(point_xy[1])],
        }
    )
    return {
        'bbox': (x0, y0, x1, y1),
        'support_raw_local': raw_crop.astype(np.uint8),
        'support_refined_local': refined_crop.astype(np.uint8),
        'support_raw_global': raw_global,
        'support_refined_global': refined_global,
        'support_refined_weight_global': refined_weight_global,
        'meta': meta,
    }
