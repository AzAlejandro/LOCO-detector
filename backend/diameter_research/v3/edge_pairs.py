from __future__ import annotations

from typing import Any

import numpy as np

from ..profiles import aggregate_profiles
from ..pipeline_v2 import _representative_edges, measure_profiles_v2


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


def _normalize_edge_params(params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params or {})
    out['edge_candidate_count'] = int(round(float(out.get('edge_pair_candidate_count', out.get('edge_candidate_count', 5)))))
    out['edge_min_score'] = float(out.get('edge_pair_min_score', out.get('edge_min_score', 0.22)))
    if bool(out.get('antihalo_enabled', True)):
        out['halo_guard_px'] = max(float(out.get('halo_guard_px', 3.0)), 3.0)
    return out


def _unit(v: Any, fallback: tuple[float, float]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(arr))
    if not np.isfinite(n) or n < 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return arr / n


def _sample_nearest(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    h, w = a.shape[:2]
    xi = np.rint(xs).astype(np.int32)
    yi = np.rint(ys).astype(np.int32)
    valid = (xi >= 0) & (yi >= 0) & (xi < w) & (yi < h)
    out = np.zeros(xs.shape, dtype=np.float32)
    out[valid] = a[yi[valid], xi[valid]]
    return out


def _local_context(
    *,
    support_weight: np.ndarray,
    center_xy: tuple[float, float],
    orientation: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    length = max(8.0, _param_float(params, 'profile_length_px', 80.0))
    half = max(4.0, length * 0.5)
    samples = max(17, int(round(length)) + 1)
    distances = np.linspace(-half, half, samples, dtype=np.float32)
    normal = _unit(orientation.get('normal', [0.0, 1.0]), (0.0, 1.0))
    xs = float(center_xy[0]) + distances.astype(np.float64) * float(normal[0])
    ys = float(center_xy[1]) + distances.astype(np.float64) * float(normal[1])
    signal = _sample_nearest(support_weight, xs, ys)
    threshold = _param_float(params, 'overshoot_support_threshold', 0.15)
    inside = signal >= threshold
    center_idx = int(np.argmin(np.abs(distances)))
    left_idx = right_idx = center_idx
    if bool(inside[center_idx]):
        while left_idx > 0 and bool(inside[left_idx - 1]):
            left_idx -= 1
        while right_idx < len(inside) - 1 and bool(inside[right_idx + 1]):
            right_idx += 1
    elif np.any(inside):
        idxs = np.where(inside)[0]
        nearest = int(idxs[int(np.argmin(np.abs(idxs - center_idx)))])
        left_idx = right_idx = nearest
        while left_idx > 0 and bool(inside[left_idx - 1]):
            left_idx -= 1
        while right_idx < len(inside) - 1 and bool(inside[right_idx + 1]):
            right_idx += 1
    context_width = float(max(0.0, distances[right_idx] - distances[left_idx])) if np.any(inside) else 0.0

    h, w = np.asarray(support_weight).shape[:2]
    radius = int(round(max(6.0, _param_float(params, 'overshoot_density_radius_px', 24.0))))
    cx = int(round(float(center_xy[0])))
    cy = int(round(float(center_xy[1])))
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    crop = np.asarray(support_weight, dtype=np.float32)[y0:y1, x0:x1]
    density = float(np.mean(crop >= threshold)) if crop.size else 0.0
    return {
        'context_width_px': context_width,
        'support_center_weight': float(signal[center_idx]) if signal.size else 0.0,
        'support_density': density,
        'support_threshold': float(threshold),
        'profile_length_requested_px': float(length),
    }


def _adapt_profile_length(edge_params: dict[str, Any], context: dict[str, Any], orientation: dict[str, Any]) -> dict[str, Any]:
    out = dict(edge_params)
    requested = max(8.0, _param_float(out, 'profile_length_px', 80.0))
    effective = requested
    reason = 'disabled'
    if _param_bool(out, 'adaptive_profile_length_enabled', False):
        context_width = float(context.get('context_width_px', 0.0) or 0.0)
        min_len = max(8.0, _param_float(out, 'adaptive_profile_min_length_px', 24.0))
        margin = max(0.0, _param_float(out, 'overshoot_margin_px', 6.0))
        scale = max(1.2, _param_float(out, 'adaptive_profile_context_scale', 2.4))
        coherence = float(orientation.get('orientation_coherence', orientation.get('confidence', 1.0)) or 0.0)
        dense = float(context.get('support_density', 0.0)) >= _param_float(out, 'adaptive_profile_dense_threshold', 0.22)
        low_coh = coherence <= _param_float(out, 'adaptive_profile_coherence_threshold', 0.30)
        if context_width > 0.0:
            effective = min(effective, max(min_len, context_width * scale + 2.0 * margin))
            reason = 'context_width'
        if dense or low_coh:
            effective = min(effective, max(min_len, requested * _param_float(out, 'adaptive_profile_risk_scale', 0.72)))
            reason = 'dense_or_low_coherence' if reason == 'disabled' else f'{reason},dense_or_low_coherence'
    out['profile_length_px'] = float(effective)
    context['profile_length_effective_px'] = float(effective)
    context['profile_length_reason'] = reason
    return out


def _edge_geometry_control(
    *,
    status: str,
    diameter_px: float | None,
    kept_profiles: list[dict[str, Any]],
    context: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    if not _param_bool(params, 'local_geometry_control_enabled', False):
        return {'enabled': False, 'status': 'disabled', **context}
    flags: list[str] = []
    context_width = float(context.get('context_width_px', 0.0) or 0.0)
    support_path_mean = float(np.mean([float(p.get('inside_ratio', 0.0)) for p in kept_profiles])) if kept_profiles else 0.0
    max_ratio = max(1.0, _param_float(params, 'overshoot_max_context_width_ratio', 1.85))
    margin = max(0.0, _param_float(params, 'overshoot_margin_px', 6.0))
    max_plausible = float(context_width * max_ratio + margin) if context_width > 0.0 else 0.0
    if status == 'ok' and diameter_px is not None and max_plausible > 0.0 and float(diameter_px) > max_plausible:
        flags.append('width_exceeds_local_context')
    min_path = float(np.clip(_param_float(params, 'overshoot_support_path_min', 0.18), 0.0, 1.0))
    if status == 'ok' and support_path_mean < min_path:
        flags.append('support_path_weak')
    control_status = 'ok' if not flags else 'overshoot_risk'
    return {
        'enabled': True,
        'status': control_status,
        'flags': flags,
        'context_width_px': context_width,
        'max_plausible_width_px': max_plausible,
        'support_path_mean': support_path_mean,
        'profile_length_requested_px': float(context.get('profile_length_requested_px', 0.0) or 0.0),
        'profile_length_effective_px': float(context.get('profile_length_effective_px', context.get('profile_length_requested_px', 0.0)) or 0.0),
        'profile_length_reason': str(context.get('profile_length_reason', '')),
        'support_density': float(context.get('support_density', 0.0) or 0.0),
    }


def measure_diameter_from_edge_pairs(
    *,
    gray_f: np.ndarray,
    support_weight: np.ndarray,
    center_xy: tuple[float, float],
    orientation: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    edge_params = _normalize_edge_params(params)
    context = _local_context(
        support_weight=np.asarray(support_weight, dtype=np.float32),
        center_xy=center_xy,
        orientation=orientation,
        params=edge_params,
    )
    edge_params = _adapt_profile_length(edge_params, context, orientation)
    profiles, raw = measure_profiles_v2(
        gray_f=np.asarray(gray_f, dtype=np.float32),
        support_weight=np.asarray(support_weight, dtype=np.float32),
        center_xy=center_xy,
        tangent=orientation.get('tangent', [1.0, 0.0]),
        normal=orientation.get('normal', [0.0, 1.0]),
        params=edge_params,
    )
    agg = aggregate_profiles(profiles, edge_params)
    kept = list(agg.get('kept_profiles') or [])
    left_xy, right_xy = _representative_edges(kept)
    valid = int(agg.get('valid_profiles', 0))
    total = max(1, int(agg.get('total_profiles', len(profiles))))
    pair_scores = [float(p.get('pair_score', 0.0)) for p in kept]
    edge_pair_score = float(np.mean(pair_scores)) if pair_scores else 0.0
    profile_consensus = float(valid / total)
    if agg.get('diameter_px') is not None:
        diam = max(1e-6, float(agg.get('diameter_px') or 0.0))
        mad = float(agg.get('mad_px') or 0.0)
        profile_consensus *= float(np.clip(1.0 - mad / max(diam * 0.35, 1e-6), 0.0, 1.0))
    status = 'ok' if agg.get('status') == 'ok' and left_xy and right_xy else 'failed'
    diameter = None if agg.get('diameter_px') is None else float(agg.get('diameter_px'))
    control = _edge_geometry_control(
        status=status,
        diameter_px=diameter,
        kept_profiles=kept,
        context=context,
        params=edge_params,
    )
    control_flags = list(control.get('flags') or [])
    if control_flags:
        status = 'failed'
        edge_pair_score *= 0.35
        profile_consensus *= 0.35
    reason = '' if status == 'ok' else str(agg.get('reason') or 'edge_pair_failed')
    if control_flags:
        reason = 'overshoot_geometry:' + ','.join(str(f) for f in control_flags)
    return {
        'status': status,
        'reason': reason,
        'diameter_px': diameter,
        'left_edge_xy': left_xy,
        'right_edge_xy': right_xy,
        'profiles': profiles,
        'raw': raw,
        'aggregate': agg,
        'local_geometry_control': control,
        'geometry_control_status': str(control.get('status', 'disabled')),
        'profile_length_effective_px': float(control.get('profile_length_effective_px', context.get('profile_length_effective_px', context.get('profile_length_requested_px', 0.0))) or 0.0),
        'context_width_px': float(control.get('context_width_px', context.get('context_width_px', 0.0)) or 0.0),
        'support_path_mean': float(control.get('support_path_mean', 0.0) or 0.0),
        'edge_pair_score': float(edge_pair_score),
        'profile_consensus': float(profile_consensus),
        'valid_profiles': valid,
        'total_profiles': total,
        'mad_px': None if agg.get('mad_px') is None else float(agg.get('mad_px')),
        'edge_score_mean': float(agg.get('edge_score_mean', 0.0)),
        'min_score': _param_float(edge_params, 'edge_min_score', 0.22),
    }
