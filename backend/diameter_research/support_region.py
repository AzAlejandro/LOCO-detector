from __future__ import annotations

from typing import Any

import cv2
import numpy as np


DEFAULT_SUPPORT_PARAMS = {
    'support_high_threshold': 0.70,
    'support_low_threshold': 0.35,
    'support_dilation_px': 5,
}


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


def _resize_like(arr: np.ndarray, shape_hw: tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    a = np.asarray(arr)
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if a.shape[:2] == (h, w):
        return a
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(a, (w, h), interpolation=interp)


def _disk_kernel(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _clean_support(mask: np.ndarray) -> np.ndarray:
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.size == 0 or not np.any(m):
        return m
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _disk_kernel(1))
    ncc, cc, stats, _cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m, dtype=np.uint8)
    min_area = max(8, int(round(m.size * 0.00005)))
    for cid in range(1, int(ncc)):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[cc == cid] = 1
    if np.any(out):
        return out
    return m


def support_from_scribbles(labels: np.ndarray | None, shape_hw: tuple[int, int], params: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if labels is None:
        labels_u8 = np.zeros((h, w), dtype=np.uint8)
    else:
        labels_u8 = _resize_like(np.asarray(labels, dtype=np.uint8), (h, w), nearest=True)
    fg = labels_u8 == 1
    dilation = max(3, _param_int(params, 'support_dilation_px', DEFAULT_SUPPORT_PARAMS['support_dilation_px']) * 2)
    support = cv2.dilate(fg.astype(np.uint8), _disk_kernel(dilation), iterations=1)
    support = _clean_support(support)
    meta = {
        'source': 'scribbles',
        'fallback': 'scribbles_support',
        'support_pixels': int(np.sum(support > 0)),
        'fg_scribble_pixels': int(np.sum(fg)),
        'support_dilation_px': int(dilation),
    }
    return support.astype(np.uint8), meta


def support_from_prior(prior_map: np.ndarray, shape_hw: tuple[int, int], params: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    prior = _resize_like(np.asarray(prior_map, dtype=np.float32), (h, w), nearest=False)
    prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
    if prior.size and float(np.nanmax(prior)) > 1.5:
        prior = prior / 255.0
    prior = np.clip(prior, 0.0, 1.0)

    high = float(np.clip(_param_float(params, 'support_high_threshold', DEFAULT_SUPPORT_PARAMS['support_high_threshold']), 0.0, 1.0))
    low = float(np.clip(_param_float(params, 'support_low_threshold', DEFAULT_SUPPORT_PARAMS['support_low_threshold']), 0.0, high))
    dilation = max(0, _param_int(params, 'support_dilation_px', DEFAULT_SUPPORT_PARAMS['support_dilation_px']))

    possible = prior >= low
    sure = prior >= high
    if dilation > 0 and np.any(sure):
        sure = cv2.dilate(sure.astype(np.uint8), _disk_kernel(dilation), iterations=1) > 0
    support = sure & possible
    if not np.any(support):
        support = possible
    support = _clean_support(support.astype(np.uint8))
    meta = {
        'source': 'prior',
        'fallback': '',
        'support_pixels': int(np.sum(support > 0)),
        'prior_min': float(np.min(prior)) if prior.size else 0.0,
        'prior_max': float(np.max(prior)) if prior.size else 0.0,
        'support_high_threshold': float(high),
        'support_low_threshold': float(low),
        'support_dilation_px': int(dilation),
    }
    return support.astype(np.uint8), meta


def build_support_region(
    *,
    prior_map: np.ndarray | None,
    labels: np.ndarray | None,
    shape_hw: tuple[int, int],
    params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if prior_map is not None and np.asarray(prior_map).size:
        support, meta = support_from_prior(prior_map, shape_hw, params)
        if np.any(support > 0):
            return support, meta
    support, meta = support_from_scribbles(labels, shape_hw, params)
    return support, meta
