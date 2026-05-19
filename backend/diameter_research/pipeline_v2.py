from __future__ import annotations

from datetime import datetime
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..image_codec import to_gray_u8, to_uint8_rgb
from .orientation import estimate_orientation
from .pipeline import DEFAULT_PARAMS as V1_DEFAULT_PARAMS
from .pipeline import build_overlay
from .profiles import aggregate_profiles, sample_bilinear
from .support_region import build_support_region


METHOD_ID_V2 = 'hybrid_profile_diameter_v2'

DEFAULT_PARAMS_V2: dict[str, Any] = {
    **V1_DEFAULT_PARAMS,
    'orientation_sweep_deg': [-10, -5, 0, 5, 10],
    'recenter_radius_px': 6,
    'recenter_step_px': 1,
    'support_component_radius_px': 48,
    'edge_candidate_count': 4,
    'halo_guard_px': 3,
    'orientation_instability_threshold': 0.28,
    'bimodal_width_gap_ratio': 0.22,
    'min_point_confidence': 0.45,
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


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


def _normalize_params(params: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULT_PARAMS_V2)
    for key, value in dict(params or {}).items():
        if key in out:
            out[key] = value
    sweep = out.get('orientation_sweep_deg', [-10, -5, 0, 5, 10])
    if isinstance(sweep, str):
        sweep = [x.strip() for x in sweep.split(',') if x.strip()]
    try:
        out['orientation_sweep_deg'] = [float(x) for x in list(sweep)]
    except Exception:
        out['orientation_sweep_deg'] = [-10.0, -5.0, 0.0, 5.0, 10.0]
    if 0.0 not in [float(x) for x in out['orientation_sweep_deg']]:
        out['orientation_sweep_deg'] = [0.0, *out['orientation_sweep_deg']]
    return out


def _unit(v: np.ndarray, fallback: tuple[float, float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n < 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return a / n


def _rotate(v: np.ndarray, degrees: float) -> np.ndarray:
    a = float(np.deg2rad(degrees))
    c = float(np.cos(a))
    s = float(np.sin(a))
    x, y = float(v[0]), float(v[1])
    return _unit(np.asarray([c * x - s * y, s * x + c * y]), (1.0, 0.0))


def _sanitize_points(points: list[dict[str, Any]] | None, shape_hw: tuple[int, int]) -> list[dict[str, Any]]:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(points or []):
        try:
            x = float(raw.get('x'))
            y = float(raw.get('y'))
        except Exception:
            continue
        if not np.isfinite([x, y]).all():
            continue
        out.append(
            {
                'point_index': int(raw.get('point_index', idx) if isinstance(raw, dict) else idx),
                'x': float(np.clip(x, 0.0, max(0, w - 1))),
                'y': float(np.clip(y, 0.0, max(0, h - 1))),
            }
        )
    return out


def _resize_like(arr: np.ndarray, shape_hw: tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    a = np.asarray(arr)
    if a.shape[:2] == (h, w):
        return a
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(a, (w, h), interpolation=interp)


def _normalize_prior(prior_map: np.ndarray | None, shape_hw: tuple[int, int]) -> np.ndarray | None:
    if prior_map is None or not np.asarray(prior_map).size:
        return None
    prior = _resize_like(np.asarray(prior_map, dtype=np.float32), shape_hw)
    prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
    if prior.size and float(np.max(prior)) > 1.5:
        prior = prior / 255.0
    return np.clip(prior, 0.0, 1.0)


def build_weighted_support(
    *,
    prior_map: np.ndarray | None,
    labels: np.ndarray | None,
    shape_hw: tuple[int, int],
    params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    hard, meta = build_support_region(prior_map=prior_map, labels=labels, shape_hw=shape_hw, params=params)
    hard = (np.asarray(hard) > 0).astype(np.uint8)
    prior = _normalize_prior(prior_map, shape_hw)
    if prior is not None:
        weight = np.maximum(prior, hard.astype(np.float32) * 0.45)
        source = 'prior_weight'
    else:
        if not np.any(hard):
            weight = np.zeros(shape_hw, dtype=np.float32)
        else:
            dist = cv2.distanceTransform(hard.astype(np.uint8), cv2.DIST_L2, 5)
            denom = max(float(np.max(dist)), 1e-6)
            weight = hard.astype(np.float32) * (0.35 + 0.65 * (dist / denom))
            weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=1.0, sigmaY=1.0)
        source = 'scribbles_weight'
    weight = np.clip(weight.astype(np.float32), 0.0, 1.0)

    halo_scribble_px = 0
    background_scribble_px = 0
    negative_suppressed_px = 0
    if labels is not None:
        lab = _resize_like(np.asarray(labels, dtype=np.uint8), shape_hw, nearest=True)
        halo = lab == 2
        background = lab == 3
        negative = halo | background
        halo_scribble_px = int(np.sum(halo))
        background_scribble_px = int(np.sum(background))
        if int(np.sum(negative)) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            halo_band = cv2.dilate(halo.astype(np.uint8), kernel, iterations=1).astype(bool)
            bg_band = cv2.dilate(background.astype(np.uint8), kernel, iterations=1).astype(bool)
            neg_band = halo_band | bg_band
            negative_suppressed_px = int(np.sum(neg_band & (weight > 0.05)))
            weight = np.where(halo_band, weight * 0.32, weight).astype(np.float32)
            weight = np.where(bg_band, weight * 0.12, weight).astype(np.float32)
            hard = np.where(neg_band & (hard > 0), 0, hard).astype(np.uint8)

    stats = {
        'source': source,
        'min': float(np.min(weight)) if weight.size else 0.0,
        'max': float(np.max(weight)) if weight.size else 0.0,
        'mean': float(np.mean(weight)) if weight.size else 0.0,
        'nonzero_px': int(np.sum(weight > 0.05)),
        'hard_px': int(np.sum(hard > 0)),
        'halo_scribble_px': halo_scribble_px,
        'background_scribble_px': background_scribble_px,
        'negative_scribble_px': int(halo_scribble_px + background_scribble_px),
        'negative_suppressed_px': negative_suppressed_px,
    }
    meta = dict(meta or {})
    meta['support_weight_stats'] = stats
    return hard, weight, meta


def isolate_local_component(
    support: np.ndarray,
    support_weight: np.ndarray,
    point_xy: tuple[float, float],
    params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    hard = (np.asarray(support) > 0).astype(np.uint8)
    weight = np.asarray(support_weight, dtype=np.float32)
    h, w = hard.shape[:2]
    x = float(np.clip(point_xy[0], 0, max(0, w - 1)))
    y = float(np.clip(point_xy[1], 0, max(0, h - 1)))
    radius = max(4, _param_int(params, 'support_component_radius_px', 48))
    yy, xx = np.indices((h, w))
    disk = ((xx - x) ** 2 + (yy - y) ** 2) <= float(radius * radius)
    local_hard = hard & disk.astype(np.uint8)

    meta: dict[str, Any] = {
        'selected_component_id': 0,
        'local_support_pixels': int(np.sum(local_hard > 0)),
        'components_nearby': 0,
        'multi_component_nearby': False,
        'selection': 'empty',
    }
    if not np.any(local_hard):
        return local_hard.astype(np.uint8), np.zeros_like(weight, dtype=np.float32), meta

    ncc, cc, stats, cent = cv2.connectedComponentsWithStats(hard, connectivity=8)
    rx = int(round(x))
    ry = int(round(y))
    selected = int(cc[ry, rx]) if 0 <= ry < h and 0 <= rx < w else 0
    if selected <= 0:
        ys, xs = np.where(local_hard > 0)
        if len(xs):
            scores = sample_bilinear(weight, xs.astype(np.float32), ys.astype(np.float32), default=0.0)
            d2 = (xs.astype(np.float32) - x) ** 2 + (ys.astype(np.float32) - y) ** 2
            best = int(np.argmax(scores - 0.002 * d2))
            selected = int(cc[int(ys[best]), int(xs[best])])
    if selected <= 0:
        selected_mask = local_hard.astype(bool)
        meta['selection'] = 'local_disk'
    else:
        selected_mask = (cc == selected) & disk
        meta['selection'] = 'connected_component'
        meta['selected_component_id'] = int(selected)

    significant = 0
    for cid in range(1, int(ncc)):
        comp = (cc == cid) & disk
        area = int(np.sum(comp))
        if area >= 12:
            significant += 1
    meta['components_nearby'] = int(significant)
    meta['multi_component_nearby'] = bool(significant >= 2)
    meta['local_support_pixels'] = int(np.sum(selected_mask))
    if selected > 0 and selected < len(stats):
        meta['selected_component_area_px'] = int(stats[selected, cv2.CC_STAT_AREA])
        meta['selected_component_centroid'] = [float(cent[selected, 0]), float(cent[selected, 1])]

    local_mask = selected_mask.astype(np.uint8)
    local_weight = np.where(local_mask > 0, weight, 0.0).astype(np.float32)
    return local_mask, local_weight, meta


def recenter_point(
    *,
    support_weight: np.ndarray,
    local_support: np.ndarray,
    point_xy: tuple[float, float],
    params: dict[str, Any] | None = None,
) -> tuple[tuple[float, float], dict[str, Any]]:
    weight = np.asarray(support_weight, dtype=np.float32)
    hard = (np.asarray(local_support) > 0).astype(np.uint8)
    h, w = weight.shape[:2]
    x0 = float(np.clip(point_xy[0], 0, max(0, w - 1)))
    y0 = float(np.clip(point_xy[1], 0, max(0, h - 1)))
    radius = max(0, _param_int(params, 'recenter_radius_px', 6))
    step = max(1, _param_int(params, 'recenter_step_px', 1))

    if np.any(hard):
        dist = cv2.distanceTransform(hard.astype(np.uint8), cv2.DIST_L2, 5)
        dist = dist / max(float(np.max(dist)), 1e-6)
    else:
        dist = np.zeros_like(weight, dtype=np.float32)

    candidates: list[dict[str, Any]] = []
    best_xy = (x0, y0)
    best_score = -1e9
    for dy in range(-radius, radius + 1, step):
        for dx in range(-radius, radius + 1, step):
            if (dx * dx + dy * dy) > radius * radius:
                continue
            x = float(np.clip(x0 + dx, 0, max(0, w - 1)))
            y = float(np.clip(y0 + dy, 0, max(0, h - 1)))
            ws = float(sample_bilinear(weight, np.asarray([x]), np.asarray([y]), default=0.0)[0])
            ds = float(sample_bilinear(dist, np.asarray([x]), np.asarray([y]), default=0.0)[0])
            move_penalty = 0.015 * (float(np.hypot(dx, dy)) / max(1.0, float(radius)))
            score = 0.62 * ds + 0.38 * ws - move_penalty
            candidates.append({'x': x, 'y': y, 'score': float(score), 'support_weight': ws, 'ridge_score': ds})
            if score > best_score:
                best_score = score
                best_xy = (x, y)
    candidates.sort(key=lambda c: float(c.get('score', 0.0)), reverse=True)
    shift = float(np.hypot(best_xy[0] - x0, best_xy[1] - y0))
    return best_xy, {
        'original_xy': [float(x0), float(y0)],
        'recentered_xy': [float(best_xy[0]), float(best_xy[1])],
        'recenter_shift_px': shift,
        'best_score': float(best_score if np.isfinite(best_score) else 0.0),
        'candidates': candidates[:80],
    }


def _edge_candidates_1d(
    distances: np.ndarray,
    intensity: np.ndarray,
    support_signal: np.ndarray,
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    sigma = max(0.0, _param_float(params, 'grad_smooth_sigma', 1.0))
    smooth = gaussian_filter1d(np.asarray(intensity, dtype=np.float32), sigma=sigma, mode='nearest') if sigma > 0 else np.asarray(intensity, dtype=np.float32)
    bg_sigma = max(3.0, float(len(smooth)) / 10.0)
    corrected = smooth - gaussian_filter1d(smooth, sigma=bg_sigma, mode='nearest')
    grad = np.gradient(corrected, distances.astype(np.float32))
    grad_abs = np.abs(grad)
    p95 = float(np.percentile(grad_abs, 95)) if grad_abs.size else 0.0
    grad_score = grad_abs / max(p95, 1e-6)
    support_smooth = gaussian_filter1d(np.asarray(support_signal, dtype=np.float32), sigma=max(0.75, sigma), mode='nearest')
    support_grad = np.abs(np.gradient(support_smooth, distances.astype(np.float32)))
    sg95 = float(np.percentile(support_grad, 95)) if support_grad.size else 0.0
    support_score = support_grad / max(sg95, 1e-6)
    combined = grad_score + 0.65 * support_score + 0.20 * support_smooth

    count = max(1, _param_int(params, 'edge_candidate_count', 4))
    center_gap = max(1.0, _param_float(params, 'center_exclusion_px', 1.5))
    min_separation = max(1.0, _param_float(params, 'halo_guard_px', 3.0) * 0.65)

    def pick(side: str) -> list[dict[str, Any]]:
        mask = distances < -center_gap if side == 'left' else distances > center_gap
        idxs = list(np.where(mask)[0])
        idxs.sort(key=lambda i: float(combined[i]), reverse=True)
        out: list[dict[str, Any]] = []
        for i in idxs:
            d = float(distances[i])
            if any(abs(d - float(c['distance_px'])) < min_separation for c in out):
                continue
            out.append(
                {
                    'distance_px': d,
                    'score': float(combined[i]),
                    'grad_score': float(grad_score[i]),
                    'support_score': float(support_score[i]),
                    'support_weight': float(support_smooth[i]),
                    'gradient': float(grad[i]),
                }
            )
            if len(out) >= count:
                break
        return out

    return pick('left'), pick('right'), corrected.astype(np.float32), grad.astype(np.float32)


def _score_pair(left: dict[str, Any], right: dict[str, Any], distances: np.ndarray, support_signal: np.ndarray, params: dict[str, Any], target_width: float | None = None) -> dict[str, Any]:
    ld = float(left['distance_px'])
    rd = float(right['distance_px'])
    width = rd - ld
    if width <= 0:
        return {'accepted': False, 'score': -1e9, 'diameter_px': 0.0}
    min_width = _param_float(params, 'min_width_px', 2.0)
    max_width = _param_float(params, 'max_width_px', max(4.0, float(np.max(distances) - np.min(distances)) * 0.95))
    if width < min_width or width > max_width:
        return {'accepted': False, 'score': -1e9, 'diameter_px': float(width)}
    inside = (distances >= ld) & (distances <= rd)
    inside_weight = float(np.mean(np.asarray(support_signal)[inside])) if np.any(inside) else 0.0
    center_offset = abs((ld + rd) * 0.5)
    asym = abs(abs(ld) - abs(rd)) / max(abs(ld) + abs(rd), 1e-6)
    max_asym = _param_float(params, 'max_profile_asymmetry', 0.75)
    halo_guard = max(0.0, _param_float(params, 'halo_guard_px', 3.0))
    outside_penalty = max(0.0, (abs(ld) - (float(np.max(distances)) - halo_guard)) / max(1.0, halo_guard))
    outside_penalty += max(0.0, (abs(rd) - (float(np.max(distances)) - halo_guard)) / max(1.0, halo_guard))
    width_penalty = 0.0
    if target_width is not None and target_width > 0:
        width_penalty = min(1.0, abs(width - float(target_width)) / max(float(target_width), 1e-6))
    edge_score = 0.5 * (float(left['score']) + float(right['score']))
    score = edge_score + 0.85 * inside_weight - 0.45 * center_offset / max(width, 1e-6) - 0.45 * max(0.0, asym - max_asym) - 0.35 * width_penalty - 0.20 * outside_penalty
    return {
        'accepted': bool(asym <= max_asym),
        'score': float(score),
        'diameter_px': float(width),
        'left_distance_px': float(ld),
        'right_distance_px': float(rd),
        'edge_score': float(edge_score),
        'inside_ratio': float(inside_weight),
        'center_offset_px': float(center_offset),
        'asymmetry': float(asym),
        'width_penalty': float(width_penalty),
    }


def _profile_for_pair(center_xy: tuple[float, float], tangent: np.ndarray, normal: np.ndarray, offset_px: float, pair: dict[str, Any], profile_index: int, reject_reason: str = '') -> dict[str, Any]:
    cx = float(center_xy[0]) + float(offset_px) * float(tangent[0])
    cy = float(center_xy[1]) + float(offset_px) * float(tangent[1])
    ld = float(pair.get('left_distance_px', 0.0))
    rd = float(pair.get('right_distance_px', 0.0))
    left_xy = [float(cx + ld * normal[0]), float(cy + ld * normal[1])]
    right_xy = [float(cx + rd * normal[0]), float(cy + rd * normal[1])]
    accepted = bool(pair.get('accepted', False))
    return {
        'profile_index': int(profile_index),
        'offset_px': float(offset_px),
        'accepted': bool(accepted),
        'accepted_final': bool(accepted),
        'reject_reason': '' if accepted else reject_reason,
        'diameter_px': float(pair.get('diameter_px', 0.0)),
        'left_distance_px': float(ld),
        'right_distance_px': float(rd),
        'edge_score': float(pair.get('edge_score', 0.0)),
        'left_score': float(pair.get('left_score', 0.0)),
        'right_score': float(pair.get('right_score', 0.0)),
        'inside_ratio': float(pair.get('inside_ratio', 0.0)),
        'pair_score': float(pair.get('score', 0.0)),
        'center_offset_px': float(pair.get('center_offset_px', 0.0)),
        'asymmetry': float(pair.get('asymmetry', 0.0)),
        'left_xy': left_xy,
        'right_xy': right_xy,
    }


def measure_profiles_v2(
    *,
    gray_f: np.ndarray,
    support_weight: np.ndarray,
    center_xy: tuple[float, float],
    tangent: list[float] | tuple[float, float] | np.ndarray,
    normal: list[float] | tuple[float, float] | np.ndarray,
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    count = max(1, _param_int(params, 'profile_count', 7))
    spacing = _param_float(params, 'profile_spacing_px', 2.0)
    length = _param_float(params, 'profile_length_px', 80.0)
    offsets = (np.arange(count, dtype=np.float32) - (count - 1) * 0.5) * float(spacing)
    t = _unit(np.asarray(tangent, dtype=np.float64), (1.0, 0.0))
    n = _unit(np.asarray(normal, dtype=np.float64), (0.0, 1.0))
    half = max(2.0, float(length) * 0.5)
    samples = max(9, int(round(length)) + 1)
    distances = np.linspace(-half, half, samples, dtype=np.float32)

    profile_data: list[dict[str, Any]] = []
    initial_widths: list[float] = []
    for idx, offset in enumerate(offsets):
        cx = float(center_xy[0]) + float(offset) * float(t[0])
        cy = float(center_xy[1]) + float(offset) * float(t[1])
        xs = cx + distances.astype(np.float64) * float(n[0])
        ys = cy + distances.astype(np.float64) * float(n[1])
        intensity = sample_bilinear(np.asarray(gray_f, dtype=np.float32), xs, ys, default=0.0)
        support_signal = sample_bilinear(np.asarray(support_weight, dtype=np.float32), xs, ys, default=0.0)
        left_cands, right_cands, corrected, grad = _edge_candidates_1d(distances, intensity, support_signal, params)
        pairs: list[dict[str, Any]] = []
        for li, left in enumerate(left_cands):
            for ri, right in enumerate(right_cands):
                pair = _score_pair(left, right, distances, support_signal, params)
                pair['left_candidate_index'] = int(li)
                pair['right_candidate_index'] = int(ri)
                pair['left_score'] = float(left.get('score', 0.0))
                pair['right_score'] = float(right.get('score', 0.0))
                pairs.append(pair)
        pairs.sort(key=lambda p: float(p.get('score', -1e9)), reverse=True)
        if pairs and pairs[0].get('accepted', False):
            initial_widths.append(float(pairs[0]['diameter_px']))
        profile_data.append(
            {
                'profile_index': int(idx),
                'offset_px': float(offset),
                'distances': distances,
                'support_signal': support_signal,
                'intensity': intensity,
                'corrected_intensity': corrected,
                'gradient': grad,
                'left_candidates': left_cands,
                'right_candidates': right_cands,
                'pairs': pairs[:24],
            }
        )

    target_width = float(np.median(initial_widths)) if initial_widths else None
    profiles: list[dict[str, Any]] = []
    edge_candidates_by_profile: list[dict[str, Any]] = []
    selected_edge_path: list[dict[str, Any]] = []
    for pdata in profile_data:
        scored: list[dict[str, Any]] = []
        for pair in pdata['pairs']:
            scored_pair = _score_pair(
                pdata['left_candidates'][int(pair['left_candidate_index'])],
                pdata['right_candidates'][int(pair['right_candidate_index'])],
                pdata['distances'],
                pdata['support_signal'],
                params,
                target_width=target_width,
            )
            scored_pair['left_candidate_index'] = int(pair['left_candidate_index'])
            scored_pair['right_candidate_index'] = int(pair['right_candidate_index'])
            scored_pair['left_score'] = float(pdata['left_candidates'][int(pair['left_candidate_index'])].get('score', 0.0))
            scored_pair['right_score'] = float(pdata['right_candidates'][int(pair['right_candidate_index'])].get('score', 0.0))
            scored.append(scored_pair)
        scored.sort(key=lambda p: float(p.get('score', -1e9)), reverse=True)
        chosen = scored[0] if scored else {'accepted': False, 'score': 0.0, 'diameter_px': 0.0}
        reject = 'missing_edge_candidates' if not scored else 'pair_score'
        min_score = _param_float(params, 'edge_min_score', 0.18)
        chosen['accepted'] = bool(chosen.get('accepted', False) and float(chosen.get('score', 0.0)) >= min_score)
        if not chosen['accepted'] and scored:
            reject = 'coupled_pair_score'
        prof = _profile_for_pair(
            center_xy,
            t,
            n,
            float(pdata['offset_px']),
            chosen,
            int(pdata['profile_index']),
            reject_reason=reject,
        )
        profiles.append(prof)
        edge_candidates_by_profile.append(
            {
                'profile_index': int(pdata['profile_index']),
                'left_candidates': pdata['left_candidates'],
                'right_candidates': pdata['right_candidates'],
                'selected_pair': chosen,
            }
        )
        selected_edge_path.append(
            {
                'profile_index': int(pdata['profile_index']),
                'accepted': bool(prof['accepted']),
                'left_xy': prof['left_xy'],
                'right_xy': prof['right_xy'],
                'diameter_px': float(prof['diameter_px']),
                'pair_score': float(prof['pair_score']),
            }
        )

    raw = {
        'edge_candidates_by_profile': edge_candidates_by_profile,
        'selected_edge_path': selected_edge_path,
        'target_width_px': target_width,
    }
    return profiles, raw


def _candidate_quality(agg: dict[str, Any], profiles: list[dict[str, Any]], orientation_conf: float, angle_penalty: float = 0.0) -> float:
    total = max(1, int(agg.get('total_profiles', len(profiles))))
    valid = int(agg.get('valid_profiles', 0))
    valid_ratio = valid / total
    if agg.get('diameter_px') is None:
        return float(0.15 * valid_ratio + 0.05 * orientation_conf - angle_penalty)
    diam = max(1e-6, float(agg.get('diameter_px') or 0.0))
    mad = float(agg.get('mad_px') or 0.0)
    dispersion = float(np.clip(1.0 - (mad / max(diam * 0.35, 1e-6)), 0.0, 1.0))
    edge = float(np.clip(float(agg.get('edge_score_mean', 0.0)) / 1.6, 0.0, 1.0))
    inside_vals = [float(p.get('inside_ratio', 0.0)) for p in profiles if bool(p.get('accepted_final'))]
    inside = float(np.mean(inside_vals)) if inside_vals else 0.0
    return float(np.clip(0.30 * valid_ratio + 0.25 * dispersion + 0.20 * edge + 0.15 * inside + 0.10 * orientation_conf - angle_penalty, 0.0, 1.0))


def evaluate_orientation_sweep(
    *,
    gray_f: np.ndarray,
    support_weight: np.ndarray,
    center_xy: tuple[float, float],
    base_orientation: dict[str, Any],
    params: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_t = _unit(np.asarray(base_orientation.get('tangent', [1.0, 0.0]), dtype=np.float64), (1.0, 0.0))
    base_conf = float(np.clip(float(base_orientation.get('confidence', 0.0)), 0.0, 1.0))
    candidates: list[dict[str, Any]] = []
    for deg in list(params.get('orientation_sweep_deg') or [0.0]):
        delta = float(deg)
        t = _rotate(base_t, delta)
        n = _unit(np.asarray([-t[1], t[0]]), (0.0, 1.0))
        profiles, raw = measure_profiles_v2(
            gray_f=gray_f,
            support_weight=support_weight,
            center_xy=center_xy,
            tangent=t,
            normal=n,
            params=params,
        )
        agg = aggregate_profiles(profiles, params)
        quality = _candidate_quality(agg, profiles, base_conf, angle_penalty=abs(delta) / 240.0)
        candidates.append(
            {
                'delta_deg': float(delta),
                'score': float(quality),
                'status': str(agg.get('status') or 'failed'),
                'diameter_px': None if agg.get('diameter_px') is None else float(agg['diameter_px']),
                'valid_profiles': int(agg.get('valid_profiles', 0)),
                'total_profiles': int(agg.get('total_profiles', len(profiles))),
                'mad_px': None if agg.get('mad_px') is None else float(agg['mad_px']),
                'edge_score_mean': float(agg.get('edge_score_mean', 0.0)),
                'tangent': [float(t[0]), float(t[1])],
                'normal': [float(n[0]), float(n[1])],
                'profiles': profiles,
                'aggregate': agg,
                'raw': raw,
            }
        )
    candidates.sort(key=lambda c: float(c.get('score', 0.0)), reverse=True)
    best = candidates[0] if candidates else {}
    return best, candidates


def _representative_edges(kept_profiles: list[dict[str, Any]]) -> tuple[list[float] | None, list[float] | None]:
    if not kept_profiles:
        return None, None
    left = np.asarray([p.get('left_xy', [np.nan, np.nan]) for p in kept_profiles], dtype=np.float32)
    right = np.asarray([p.get('right_xy', [np.nan, np.nan]) for p in kept_profiles], dtype=np.float32)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return None, None
    return [float(v) for v in np.median(left, axis=0)], [float(v) for v in np.median(right, axis=0)]


def _confidence_v2(agg: dict[str, Any], best_orientation_score: float, base_orientation: dict[str, Any]) -> float:
    if agg.get('diameter_px') is None:
        return 0.0
    total = max(1, int(agg.get('total_profiles', 0)))
    valid = int(agg.get('valid_profiles', 0))
    valid_ratio = min(1.0, valid / total)
    diam = max(1e-6, float(agg.get('diameter_px', 0.0)))
    mad = float(agg.get('mad_px') or 0.0)
    dispersion = float(np.clip(1.0 - (mad / max(diam * 0.30, 1e-6)), 0.0, 1.0))
    edge = float(np.clip(float(agg.get('edge_score_mean', 0.0)) / 1.6, 0.0, 1.0))
    orient = float(np.clip(float(base_orientation.get('confidence', 0.0)), 0.0, 1.0))
    sweep = float(np.clip(best_orientation_score, 0.0, 1.0))
    return float(np.clip(0.30 * valid_ratio + 0.22 * dispersion + 0.18 * edge + 0.15 * orient + 0.15 * sweep, 0.0, 1.0))


def _quality_flags(
    *,
    agg: dict[str, Any],
    profiles: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    component_meta: dict[str, Any],
    confidence: float,
    params: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    if int(component_meta.get('local_support_pixels', 0)) < 12:
        flags.append('support_weak')
    if bool(component_meta.get('multi_component_nearby', False)):
        flags.append('multiple_components_nearby')
    diameters = [float(p.get('diameter_px', 0.0)) for p in profiles if bool(p.get('accepted_final')) and float(p.get('diameter_px', 0.0)) > 0.0]
    if diameters:
        vals = np.sort(np.asarray(diameters, dtype=np.float32))
        median = max(1e-6, float(np.median(vals)))
        if len(vals) >= 4:
            gaps = np.diff(vals)
            if gaps.size and float(np.max(gaps)) / median > _param_float(params, 'bimodal_width_gap_ratio', 0.22):
                flags.append('width_bimodal')
        spread = (float(np.max(vals)) - float(np.min(vals))) / median
        if spread > _param_float(params, 'bimodal_width_gap_ratio', 0.22) * 2.2:
            flags.append('width_unstable')
    else:
        flags.append('edge_ambiguous')

    ok_candidate_widths = [float(c['diameter_px']) for c in candidates if c.get('diameter_px') is not None and c.get('status') == 'ok']
    if ok_candidate_widths:
        best_width = max(1e-6, float(agg.get('diameter_px') or np.median(ok_candidate_widths)))
        orientation_spread = (max(ok_candidate_widths) - min(ok_candidate_widths)) / best_width
        if orientation_spread > _param_float(params, 'orientation_instability_threshold', 0.28):
            flags.append('orientation_unstable')
    if confidence < _param_float(params, 'min_point_confidence', 0.45):
        flags.append('low_confidence')
    if agg.get('status') != 'ok':
        flags.append(str(agg.get('reason') or 'profile_rejected'))
    return sorted(set(flags))


def _quality_label(status: str, confidence: float, flags: list[str]) -> str:
    if status == 'rejected':
        if 'multiple_components_nearby' in flags or 'width_bimodal' in flags:
            return 'geometry_ambiguous'
        if 'orientation_unstable' in flags:
            return 'orientation_unstable'
        if 'edge_ambiguous' in flags:
            return 'edge_ambiguous'
        if 'support_weak' in flags:
            return 'support_weak'
        return 'low_confidence'
    if confidence >= 0.75 and not flags:
        return 'high_confidence'
    return 'medium_confidence'


def run_hybrid_profile_diameter_v2(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    rgb = to_uint8_rgb(image_rgb)
    if rgb is None:
        raise ValueError('Imagen invalida.')
    gray_u8 = to_gray_u8(rgb)
    if gray_u8 is None:
        raise ValueError('No se pudo convertir imagen a gris.')
    shape_hw = rgb.shape[:2]
    effective_params = _normalize_params(params)
    clean_points = _sanitize_points(points, shape_hw)
    if not clean_points:
        raise ValueError('No hay puntos validos para medir.')

    support, support_weight, support_meta = build_weighted_support(prior_map=prior_map, labels=labels, shape_hw=shape_hw, params=effective_params)
    if not np.any(support > 0):
        raise ValueError('No se pudo construir soporte desde prior ni scribbles.')

    gray_f = gray_u8.astype(np.float32) / 255.0
    results: list[dict[str, Any]] = []
    diagnostics_points: list[dict[str, Any]] = []
    for point in clean_points:
        original_xy = (float(point['x']), float(point['y']))
        local_support, local_weight, component_meta = isolate_local_component(support, support_weight, original_xy, effective_params)
        recentered_xy, recenter_diag = recenter_point(support_weight=local_weight, local_support=local_support, point_xy=original_xy, params=effective_params)
        base_orientation = estimate_orientation(gray_u8=gray_u8, support=local_support, point_xy=recentered_xy, params=effective_params)
        best, orientation_candidates = evaluate_orientation_sweep(
            gray_f=gray_f,
            support_weight=local_weight,
            center_xy=recentered_xy,
            base_orientation=base_orientation,
            params=effective_params,
        )
        profiles = list(best.get('profiles') or [])
        agg = dict(best.get('aggregate') or aggregate_profiles(profiles, effective_params))
        kept = list(agg.get('kept_profiles') or [])
        left_xy, right_xy = _representative_edges(kept)
        confidence = _confidence_v2(agg, float(best.get('score', 0.0)), base_orientation)
        flags = _quality_flags(
            agg=agg,
            profiles=profiles,
            candidates=orientation_candidates,
            component_meta=component_meta,
            confidence=confidence,
            params=effective_params,
        )
        reject_flags = {'support_weak', 'multiple_components_nearby', 'width_bimodal', 'orientation_unstable', 'edge_ambiguous', 'low_confidence'}
        rejected = agg.get('status') != 'ok' or any(flag in reject_flags for flag in flags)
        status = 'rejected' if rejected else 'ok'
        reason = ','.join(flags) if rejected else ''
        label = _quality_label(status, confidence, flags if rejected else [])
        result = {
            'method_id': METHOD_ID_V2,
            'point_index': int(point['point_index']),
            'x': float(recentered_xy[0]),
            'y': float(recentered_xy[1]),
            'original_xy': [float(original_xy[0]), float(original_xy[1])],
            'recentered_xy': [float(recentered_xy[0]), float(recentered_xy[1])],
            'recenter_shift_px': float(recenter_diag.get('recenter_shift_px', 0.0)),
            'status': status,
            'reason': reason,
            'quality_label': label,
            'diameter_px': None if rejected or agg.get('diameter_px') is None else float(agg['diameter_px']),
            'confidence': float(confidence),
            'stability_score': float(best.get('score', 0.0)),
            'valid_profiles': int(agg.get('valid_profiles', 0)),
            'total_profiles': int(agg.get('total_profiles', len(profiles))),
            'mad_px': None if agg.get('mad_px') is None else float(agg['mad_px']),
            'edge_score_mean': float(agg.get('edge_score_mean', 0.0)),
            'orientation_delta_deg': float(best.get('delta_deg', 0.0)),
            'left_edge_xy': None if rejected else left_xy,
            'right_edge_xy': None if rejected else right_xy,
            'orientation': {
                'source': str(base_orientation.get('source', '')),
                'tangent': best.get('tangent', base_orientation.get('tangent', [1.0, 0.0])),
                'normal': best.get('normal', base_orientation.get('normal', [0.0, 1.0])),
                'confidence': float(base_orientation.get('confidence', 0.0)),
                'base_tangent': base_orientation.get('tangent', [1.0, 0.0]),
                'base_normal': base_orientation.get('normal', [0.0, 1.0]),
            },
            'profiles': profiles,
            'quality_flags': flags,
        }
        results.append(_json_ready(result))
        diagnostics_points.append(
            _json_ready(
                {
                    'point_index': int(point['point_index']),
                    'original_xy': result['original_xy'],
                    'recenter': recenter_diag,
                    'support_component': component_meta,
                    'base_orientation': base_orientation,
                    'orientation_candidates': [
                        {k: v for k, v in cand.items() if k not in {'profiles', 'aggregate', 'raw'}}
                        for cand in orientation_candidates
                    ],
                    'edge_candidates_by_profile': (best.get('raw') or {}).get('edge_candidates_by_profile', []),
                    'selected_edge_path': (best.get('raw') or {}).get('selected_edge_path', []),
                    'quality_flags': flags,
                }
            )
        )

    overlay = build_overlay(image_rgb=rgb, support=support, results=results, params=effective_params)
    meta = {
        'method': METHOD_ID_V2,
        'method_id': METHOD_ID_V2,
        'experiment_id': METHOD_ID_V2,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_mode': str(source_mode or 'prior'),
        'params_effective': _json_ready(effective_params),
        'support': _json_ready(support_meta),
        'support_weight_stats': _json_ready((support_meta or {}).get('support_weight_stats') or {}),
        'points_requested': int(len(clean_points)),
        'points_ok': int(sum(1 for r in results if r.get('status') == 'ok')),
        'points_rejected': int(sum(1 for r in results if r.get('status') == 'rejected')),
        'image_shape': [int(v) for v in rgb.shape],
    }
    diagnostics_v2 = {
        'method_id': METHOD_ID_V2,
        'support_meta': _json_ready(support_meta),
        'support_weight_stats': _json_ready((support_meta or {}).get('support_weight_stats') or {}),
        'points': diagnostics_points,
    }
    return {
        'experiment_id': METHOD_ID_V2,
        'method_id': METHOD_ID_V2,
        'overlay': overlay,
        'support_region': support.astype(np.uint8),
        'support_weight': support_weight.astype(np.float32),
        'results': results,
        'meta': meta,
        'diagnostics': {
            'results': results,
            'support_meta': _json_ready(support_meta),
            'support_weight_stats': _json_ready((support_meta or {}).get('support_weight_stats') or {}),
            'params_effective': _json_ready(effective_params),
            'diagnostics_v2': diagnostics_v2,
        },
    }
