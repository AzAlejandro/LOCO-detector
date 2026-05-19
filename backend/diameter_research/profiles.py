from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d


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


def sample_bilinear(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray, *, default: float = 0.0) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError('sample_bilinear espera matriz 2D')
    h, w = a.shape
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    out = np.full(x.shape, float(default), dtype=np.float32)
    valid = (x >= 0.0) & (y >= 0.0) & (x <= (w - 1)) & (y <= (h - 1))
    if not np.any(valid):
        return out
    xv = x[valid]
    yv = y[valid]
    x0 = np.floor(xv).astype(np.int32)
    y0 = np.floor(yv).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (xv - x0).astype(np.float32)
    wy = (yv - y0).astype(np.float32)
    af = a.astype(np.float32, copy=False)
    v00 = af[y0, x0]
    v10 = af[y0, x1]
    v01 = af[y1, x0]
    v11 = af[y1, x1]
    out[valid] = (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    )
    return out


def _edge_candidates(
    distances: np.ndarray,
    intensity: np.ndarray,
    support_signal: np.ndarray,
    params: dict[str, Any] | None,
) -> tuple[int | None, int | None, float, dict[str, float]]:
    sigma = max(0.0, _param_float(params, 'grad_smooth_sigma', 1.0))
    smooth = gaussian_filter1d(np.asarray(intensity, dtype=np.float32), sigma=sigma, mode='nearest') if sigma > 0 else np.asarray(intensity, dtype=np.float32)
    grad = np.gradient(smooth, distances.astype(np.float32))
    grad_abs = np.abs(grad)
    p95 = float(np.percentile(grad_abs, 95)) if grad_abs.size else 0.0
    grad_score = grad_abs / max(p95, 1e-6)

    support_smooth = gaussian_filter1d(np.asarray(support_signal, dtype=np.float32), sigma=max(0.75, sigma), mode='nearest')
    support_grad = np.abs(np.gradient(support_smooth, distances.astype(np.float32)))
    sg95 = float(np.percentile(support_grad, 95)) if support_grad.size else 0.0
    support_score = support_grad / max(sg95, 1e-6)
    score = grad_score + 0.25 * support_score

    center_gap = max(1.0, _param_float(params, 'center_exclusion_px', 1.5))
    left_mask = distances < -center_gap
    right_mask = distances > center_gap
    if not np.any(left_mask) or not np.any(right_mask):
        return None, None, 0.0, {'left_score': 0.0, 'right_score': 0.0}

    left_indices = np.where(left_mask)[0]
    right_indices = np.where(right_mask)[0]
    li = int(left_indices[int(np.argmax(score[left_indices]))])
    ri = int(right_indices[int(np.argmax(score[right_indices]))])
    left_score = float(score[li])
    right_score = float(score[ri])
    return li, ri, float(0.5 * (left_score + right_score)), {'left_score': left_score, 'right_score': right_score}


def detect_profile_edges(
    *,
    distances: np.ndarray,
    intensity: np.ndarray,
    support_signal: np.ndarray,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    left_idx, right_idx, edge_score, score_meta = _edge_candidates(distances, intensity, support_signal, params)
    edge_min_score = _param_float(params, 'edge_min_score', 0.18)
    max_asym = _param_float(params, 'max_profile_asymmetry', 0.75)
    min_width = _param_float(params, 'min_width_px', 2.0)
    max_width = _param_float(params, 'max_width_px', max(4.0, float(np.max(distances) - np.min(distances)) * 0.95))
    support_min_inside = _param_float(params, 'support_min_inside_ratio', 0.15)

    reject = ''
    accepted = True
    left_d = right_d = diameter = 0.0
    inside_ratio = 0.0
    if left_idx is None or right_idx is None:
        accepted = False
        reject = 'missing_edge'
    else:
        left_d = float(distances[left_idx])
        right_d = float(distances[right_idx])
        if left_d >= right_d:
            accepted = False
            reject = 'edge_order'
        else:
            diameter = float(right_d - left_d)
            inside_mask = (distances >= left_d) & (distances <= right_d)
            inside_ratio = float(np.mean(np.asarray(support_signal)[inside_mask] > 0.15)) if np.any(inside_mask) else 0.0
            asym = abs(abs(left_d) - abs(right_d)) / max(abs(left_d) + abs(right_d), 1e-6)
            if edge_score < edge_min_score:
                accepted = False
                reject = 'edge_score'
            elif diameter < min_width or diameter > max_width:
                accepted = False
                reject = 'width_range'
            elif asym > max_asym:
                accepted = False
                reject = 'asymmetry'
            elif inside_ratio < support_min_inside:
                accepted = False
                reject = 'low_support_intersection'

    return {
        'accepted': bool(accepted),
        'reject_reason': reject,
        'diameter_px': float(diameter),
        'left_distance_px': float(left_d),
        'right_distance_px': float(right_d),
        'edge_score': float(edge_score),
        'inside_ratio': float(inside_ratio),
        **score_meta,
    }


def sample_profile(
    *,
    gray_f: np.ndarray,
    support: np.ndarray,
    center_xy: tuple[float, float],
    tangent: np.ndarray,
    normal: np.ndarray,
    offset_px: float,
    profile_length_px: float,
) -> dict[str, Any]:
    half = max(2.0, float(profile_length_px) * 0.5)
    samples = max(9, int(round(profile_length_px)) + 1)
    distances = np.linspace(-half, half, samples, dtype=np.float32)
    t = _unit(tangent, (1.0, 0.0))
    n = _unit(normal, (0.0, 1.0))
    cx = float(center_xy[0]) + float(offset_px) * float(t[0])
    cy = float(center_xy[1]) + float(offset_px) * float(t[1])
    xs = cx + distances.astype(np.float64) * float(n[0])
    ys = cy + distances.astype(np.float64) * float(n[1])
    intensity = sample_bilinear(np.asarray(gray_f, dtype=np.float32), xs, ys, default=0.0)
    support_signal = sample_bilinear((np.asarray(support) > 0).astype(np.float32), xs, ys, default=0.0)
    return {
        'distances': distances,
        'xs': xs.astype(np.float32),
        'ys': ys.astype(np.float32),
        'intensity': intensity.astype(np.float32),
        'support': support_signal.astype(np.float32),
        'offset_px': float(offset_px),
    }


def measure_profiles(
    *,
    gray_f: np.ndarray,
    support: np.ndarray,
    center_xy: tuple[float, float],
    tangent: list[float] | tuple[float, float] | np.ndarray,
    normal: list[float] | tuple[float, float] | np.ndarray,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    count = max(1, _param_int(params, 'profile_count', 7))
    spacing = _param_float(params, 'profile_spacing_px', 2.0)
    length = _param_float(params, 'profile_length_px', 80.0)
    offsets = (np.arange(count, dtype=np.float32) - (count - 1) * 0.5) * float(spacing)
    t = _unit(np.asarray(tangent, dtype=np.float64), (1.0, 0.0))
    n = _unit(np.asarray(normal, dtype=np.float64), (0.0, 1.0))
    profiles: list[dict[str, Any]] = []
    for idx, offset in enumerate(offsets):
        prof = sample_profile(
            gray_f=gray_f,
            support=support,
            center_xy=center_xy,
            tangent=t,
            normal=n,
            offset_px=float(offset),
            profile_length_px=length,
        )
        det = detect_profile_edges(
            distances=prof['distances'],
            intensity=prof['intensity'],
            support_signal=prof['support'],
            params=params,
        )
        left_xy = [
            float(center_xy[0] + float(offset) * t[0] + det['left_distance_px'] * n[0]),
            float(center_xy[1] + float(offset) * t[1] + det['left_distance_px'] * n[1]),
        ]
        right_xy = [
            float(center_xy[0] + float(offset) * t[0] + det['right_distance_px'] * n[0]),
            float(center_xy[1] + float(offset) * t[1] + det['right_distance_px'] * n[1]),
        ]
        profiles.append(
            {
                'profile_index': int(idx),
                'offset_px': float(offset),
                'accepted': bool(det['accepted']),
                'accepted_final': bool(det['accepted']),
                'reject_reason': str(det['reject_reason']),
                'diameter_px': float(det['diameter_px']),
                'left_distance_px': float(det['left_distance_px']),
                'right_distance_px': float(det['right_distance_px']),
                'edge_score': float(det['edge_score']),
                'left_score': float(det.get('left_score', 0.0)),
                'right_score': float(det.get('right_score', 0.0)),
                'inside_ratio': float(det['inside_ratio']),
                'left_xy': left_xy,
                'right_xy': right_xy,
            }
        )
    return profiles


def aggregate_profiles(profiles: list[dict[str, Any]], params: dict[str, Any] | None = None) -> dict[str, Any]:
    accepted = [p for p in profiles if bool(p.get('accepted')) and float(p.get('diameter_px', 0.0)) > 0.0]
    total = len(profiles)
    min_valid = max(1, _param_int(params, 'min_valid_profiles', 3))
    if not accepted:
        return {
            'status': 'failed',
            'reason': 'no_valid_profiles',
            'diameter_px': None,
            'valid_profiles': 0,
            'total_profiles': int(total),
            'mad_px': None,
            'edge_score_mean': 0.0,
            'kept_profiles': [],
        }

    vals = np.asarray([float(p['diameter_px']) for p in accepted], dtype=np.float32)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    max_mad_scale = _param_float(params, 'max_mad_scale', 2.5)
    if mad <= 1e-6:
        keep_mask = np.ones_like(vals, dtype=bool)
    else:
        robust_sigma = max(1e-6, 1.4826 * mad)
        keep_mask = np.abs(vals - med) <= (float(max_mad_scale) * robust_sigma)
    kept = [p for p, keep in zip(accepted, keep_mask) if bool(keep)]
    for p, keep in zip(accepted, keep_mask):
        p['accepted_final'] = bool(keep)
        if not bool(keep):
            p['reject_reason'] = 'mad_outlier'

    if len(kept) < min_valid:
        return {
            'status': 'failed',
            'reason': 'too_few_profiles',
            'diameter_px': None,
            'valid_profiles': int(len(kept)),
            'total_profiles': int(total),
            'mad_px': float(mad),
            'edge_score_mean': float(np.mean([float(p.get('edge_score', 0.0)) for p in kept])) if kept else 0.0,
            'kept_profiles': kept,
        }
    kept_vals = np.asarray([float(p['diameter_px']) for p in kept], dtype=np.float32)
    return {
        'status': 'ok',
        'reason': '',
        'diameter_px': float(np.median(kept_vals)),
        'valid_profiles': int(len(kept)),
        'total_profiles': int(total),
        'mad_px': float(np.median(np.abs(kept_vals - np.median(kept_vals)))),
        'edge_score_mean': float(np.mean([float(p.get('edge_score', 0.0)) for p in kept])),
        'kept_profiles': kept,
    }
