from __future__ import annotations

from typing import Any

import numpy as np


METHODOLOGY_NONE = ''
METHODOLOGY_SMALL_LARGE = 'small_large'
METHODOLOGY_HALO_AWARE = 'halo_aware'
METHODOLOGY_RIDGE_ANCHORED = 'ridge_anchored'
METHODOLOGY_FLUX_AWARE = 'flux_aware'
METHODOLOGY_CONTOUR_REFINE = 'contour_refine'
METHODOLOGY_CURVELET_AIDED = 'curvelet_aided'


def _param_float(params: dict[str, Any] | None, key: str, default: float) -> float:
    try:
        return float((params or {}).get(key, default))
    except Exception:
        return float(default)


def _sample_nearest(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    h, w = a.shape[:2]
    xi = np.rint(xs).astype(np.int32)
    yi = np.rint(ys).astype(np.int32)
    valid = (xi >= 0) & (yi >= 0) & (xi < w) & (yi < h)
    out = np.zeros(xs.shape, dtype=np.float32)
    out[valid] = a[yi[valid], xi[valid]]
    return out


def _unit(v: Any, fallback: tuple[float, float]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(arr))
    if not np.isfinite(n) or n < 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return arr / n


def _crop_bounds(shape_hw: tuple[int, int], center_xy: tuple[float, float], radius: float) -> tuple[int, int, int, int]:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    cx = int(round(float(center_xy[0])))
    cy = int(round(float(center_xy[1])))
    r = int(round(max(4.0, float(radius))))
    return max(0, cx - r), min(w, cx + r + 1), max(0, cy - r), min(h, cy + r + 1)


def _normal_support_profile(
    *,
    support_weight: np.ndarray,
    center_xy: tuple[float, float],
    orientation: dict[str, Any],
    length_px: float,
    threshold: float,
) -> dict[str, Any]:
    half = max(4.0, float(length_px) * 0.5)
    samples = max(17, int(round(float(length_px))) + 1)
    distances = np.linspace(-half, half, samples, dtype=np.float32)
    normal = _unit(orientation.get('normal', [0.0, 1.0]), (0.0, 1.0))
    xs = float(center_xy[0]) + distances.astype(np.float64) * float(normal[0])
    ys = float(center_xy[1]) + distances.astype(np.float64) * float(normal[1])
    support_signal = _sample_nearest(support_weight, xs, ys)
    inside = support_signal >= float(threshold)
    center_idx = int(np.argmin(np.abs(distances)))
    width = 0.0
    if np.any(inside):
        if bool(inside[center_idx]):
            left_idx = right_idx = center_idx
        else:
            idxs = np.where(inside)[0]
            nearest = int(idxs[int(np.argmin(np.abs(idxs - center_idx)))])
            left_idx = right_idx = nearest
        while left_idx > 0 and bool(inside[left_idx - 1]):
            left_idx -= 1
        while right_idx < len(inside) - 1 and bool(inside[right_idx + 1]):
            right_idx += 1
        width = float(max(0.0, distances[right_idx] - distances[left_idx]))
    return {
        'distances': distances,
        'support_signal': support_signal,
        'context_width_px': float(width),
        'support_center_weight': float(support_signal[center_idx]) if support_signal.size else 0.0,
    }


def classify_local_context(
    *,
    gray_f: np.ndarray,
    support_weight: np.ndarray,
    support_refined: np.ndarray,
    center_xy: tuple[float, float],
    orientation: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    profile_len = _param_float(params, 'profile_length_px', 80.0)
    support_threshold = _param_float(params, 'overshoot_support_threshold', 0.15)
    normal_profile = _normal_support_profile(
        support_weight=np.asarray(support_weight, dtype=np.float32),
        center_xy=center_xy,
        orientation=orientation,
        length_px=profile_len,
        threshold=support_threshold,
    )
    context_width = float(normal_profile.get('context_width_px', 0.0) or 0.0)
    thin_threshold = _param_float(params, 'thin_fiber_threshold_px', 8.0)
    radius = _param_float(params, 'geometry_window_px', 48.0) * 0.5
    x0, x1, y0, y1 = _crop_bounds(np.asarray(gray_f).shape[:2], center_xy, radius)
    crop = np.asarray(gray_f, dtype=np.float32)[y0:y1, x0:x1]
    support_crop = np.asarray(support_weight, dtype=np.float32)[y0:y1, x0:x1]
    density = float(np.mean(support_crop >= support_threshold)) if support_crop.size else 0.0
    if crop.size:
        gy, gx = np.gradient(crop.astype(np.float32))
        grad_mag = np.sqrt(gx * gx + gy * gy)
        grad_p90 = float(np.percentile(grad_mag, 90))
        contrast = float(np.percentile(crop, 90) - np.percentile(crop, 10))
    else:
        grad_mag = np.zeros((1, 1), dtype=np.float32)
        grad_p90 = 0.0
        contrast = 0.0

    distances = np.asarray(normal_profile.get('distances'), dtype=np.float32)
    signal = np.asarray(normal_profile.get('support_signal'), dtype=np.float32)
    halo_score = 0.0
    if distances.size and signal.size:
        support_grad = np.abs(np.gradient(signal, distances))
        edge_band = max(3.0, context_width * 0.5)
        inner = np.abs(np.abs(distances) - edge_band) <= 4.0
        outer = np.abs(distances) > (edge_band + _param_float(params, 'halo_guard_px', 3.0))
        inner_strength = float(np.mean(support_grad[inner])) if np.any(inner) else 0.0
        outer_strength = float(np.mean(support_grad[outer])) if np.any(outer) else 0.0
        halo_score = float(np.clip(outer_strength / max(inner_strength + outer_strength, 1e-6), 0.0, 1.0))

    coherence = float(orientation.get('orientation_coherence', orientation.get('confidence', 0.0)) or 0.0)
    thin_score = float(np.clip((thin_threshold * 1.45 - context_width) / max(thin_threshold * 1.45, 1e-6), 0.0, 1.0)) if context_width > 0 else 0.35
    wide_score = float(np.clip((context_width - thin_threshold * 2.0) / max(thin_threshold * 3.0, 1e-6), 0.0, 1.0))
    neighbor_score = float(np.clip(density * (1.15 - 0.45 * coherence), 0.0, 1.0))
    low_contrast_score = float(np.clip((0.12 - contrast) / 0.12, 0.0, 1.0))

    labels: list[str] = []
    if thin_score >= 0.55:
        labels.append('thin_fiber')
    if wide_score >= 0.35:
        labels.append('wide_fiber')
    if neighbor_score >= 0.26:
        labels.append('dense_neighborhood')
    if halo_score >= 0.34:
        labels.append('halo_suspect')
    if low_contrast_score >= 0.45 or grad_p90 < 0.025:
        labels.append('low_contrast')
    if not labels:
        labels.append('clean_local')

    return {
        'local_context_label': str(labels[0]),
        'local_context_labels': labels,
        'local_context_scores': {
            'thin_score': float(thin_score),
            'wide_score': float(wide_score),
            'halo_score': float(halo_score),
            'neighbor_density': float(neighbor_score),
            'low_contrast_score': float(low_contrast_score),
            'orientation_coherence': float(coherence),
            'gradient_p90': float(grad_p90),
            'contrast_p80': float(contrast),
            'support_density': float(density),
        },
        'preliminary_width_px': float(context_width),
        'support_center_weight': float(normal_profile.get('support_center_weight', 0.0) or 0.0),
        'support_refined_pixels': int(np.sum(np.asarray(support_refined) > 0)),
    }


def params_for_methodology(methodology_id: str, params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    method = str(methodology_id or '')
    out = dict(params or {})
    labels = set(str(x) for x in list(context.get('local_context_labels') or []))
    width = float(context.get('preliminary_width_px', 0.0) or 0.0)
    requested = max(8.0, _param_float(out, 'profile_length_px', 80.0))
    thin_threshold = _param_float(out, 'thin_fiber_threshold_px', 8.0)

    if method == METHODOLOGY_SMALL_LARGE:
        out['local_geometry_control_enabled'] = True
        out['adaptive_profile_length_enabled'] = True
        if 'thin_fiber' in labels:
            target = max(_param_float(out, 'adaptive_profile_min_length_px', 24.0), width * 3.1 + 12.0 if width > 0 else requested * 0.55)
            out['profile_length_px'] = float(min(requested, target))
            out['overshoot_max_context_width_ratio'] = min(_param_float(out, 'overshoot_max_context_width_ratio', 1.85), 1.45)
            out['overshoot_margin_px'] = min(_param_float(out, 'overshoot_margin_px', 6.0), 4.0)
            out['edge_pair_candidate_count'] = max(5, int(round(_param_float(out, 'edge_pair_candidate_count', 5))))
        elif 'wide_fiber' in labels:
            out['profile_length_px'] = float(max(requested, min(96.0, width * 2.2 + 14.0 if width > 0 else requested)))
            out['halo_guard_px'] = max(_param_float(out, 'halo_guard_px', 3.0), 4.0)
        out['small_large_enabled'] = True

    elif method == METHODOLOGY_HALO_AWARE:
        out['antihalo_enabled'] = True
        out['halo_guard_px'] = max(_param_float(out, 'halo_guard_px', 3.0), 5.0)
        out['edge_pair_candidate_count'] = max(6, int(round(_param_float(out, 'edge_pair_candidate_count', 5))))
        out['overshoot_max_context_width_ratio'] = min(_param_float(out, 'overshoot_max_context_width_ratio', 1.85), 1.62)
        out['halo_aware_enabled'] = True

    elif method == METHODOLOGY_RIDGE_ANCHORED:
        out['local_preprocess_diagnostics_enabled'] = True
        out['ridge_anchor_enabled'] = True
        out['edge_pair_candidate_count'] = max(5, int(round(_param_float(out, 'edge_pair_candidate_count', 5))))

    elif method == METHODOLOGY_FLUX_AWARE:
        out['flux_aware_enabled'] = True
        out['local_geometry_control_enabled'] = True
        out['adaptive_profile_length_enabled'] = True
        out['overshoot_max_context_width_ratio'] = min(_param_float(out, 'overshoot_max_context_width_ratio', 1.85), 1.55)

    elif method == METHODOLOGY_CONTOUR_REFINE:
        out['fallback_enabled'] = True
        out['contour_refine_enabled'] = True
        out['edge_pair_min_score'] = max(0.12, _param_float(out, 'edge_pair_min_score', 0.22) * 0.86)

    elif method == METHODOLOGY_CURVELET_AIDED:
        out['local_preprocess_diagnostics_enabled'] = True
        out['curvelet_aided_enabled'] = True
        out['edge_pair_candidate_count'] = max(5, int(round(_param_float(out, 'edge_pair_candidate_count', 5))))
        if width > 0 and width < thin_threshold * 1.5:
            out['profile_length_px'] = float(min(requested, max(24.0, width * 3.3 + 10.0)))

    return out


def methodology_diagnostics(
    *,
    methodology_id: str,
    context: dict[str, Any],
    measurement: dict[str, Any],
    edge_measurement: dict[str, Any],
    orientation: dict[str, Any],
    geometry: dict[str, Any],
    preprocess: dict[str, Any] | None = None,
) -> dict[str, Any]:
    method = str(methodology_id or '')
    labels = list(context.get('local_context_labels') or [])
    scores = dict(context.get('local_context_scores') or {})
    diameter = measurement.get('diameter_px')
    diameter_f = None if diameter is None else float(diameter)
    context_width = float(measurement.get('context_width_px', context.get('preliminary_width_px', 0.0)) or 0.0)
    support_path = float(measurement.get('support_path_mean', 0.0) or 0.0)
    profiles = list(edge_measurement.get('profiles') or measurement.get('profiles') or [])
    center_offsets = [abs(float(p.get('center_offset_px', 0.0) or 0.0)) for p in profiles if p.get('accepted')]
    center_offset = float(np.median(center_offsets)) if center_offsets else 0.0
    coherence = float(orientation.get('orientation_coherence', orientation.get('confidence', 0.0)) or 0.0)
    neighbor = float(scores.get('neighbor_density', 0.0) or 0.0)
    halo_score = float(scores.get('halo_score', 0.0) or 0.0)
    gradient_p90 = float(scores.get('gradient_p90', 0.0) or 0.0)
    contrast = float(scores.get('contrast_p80', 0.0) or 0.0)

    size_route = 'thin_fiber' if 'thin_fiber' in labels else 'wide_fiber' if 'wide_fiber' in labels else 'normal'
    halo_status = 'halo_suspect' if halo_score >= 0.34 or (diameter_f is not None and context_width > 0 and diameter_f > context_width * 1.42 and support_path < 0.38) else 'not_suspect'
    ridge_response = float(np.clip(0.55 * coherence + 0.25 * min(1.0, gradient_p90 / 0.12) + 0.20 * min(1.0, contrast / 0.30), 0.0, 1.0))
    ridge_status = 'used' if method == METHODOLOGY_RIDGE_ANCHORED else 'diagnostic'
    if ridge_response < 0.32 or center_offset > max(3.0, context_width * 0.32):
        ridge_status = 'weak_anchor'
    axis_flux = float(np.clip(coherence * (1.0 - 0.35 * neighbor), 0.0, 1.0))
    neighbor_flux = float(np.clip(neighbor * (1.1 - 0.4 * coherence), 0.0, 1.0))
    flux_status = 'ambiguous' if neighbor_flux > axis_flux + 0.18 else 'axis_supported'
    curvelet_score = float(np.clip(0.45 * min(1.0, gradient_p90 / 0.12) + 0.35 * coherence + 0.20 * min(1.0, support_path / 0.45), 0.0, 1.0))
    curvelet_status = 'diagnostic_only' if method != METHODOLOGY_CURVELET_AIDED else ('weak_structure' if curvelet_score < 0.34 else 'used')
    contour_status = 'not_used'
    if method == METHODOLOGY_CONTOUR_REFINE:
        contour_status = 'used' if str(measurement.get('status') or '') == 'ok' and str(edge_measurement.get('status') or '') != 'ok' else 'not_needed'

    flags: list[str] = []
    confidence_multiplier = 1.0
    reject_reason = ''
    selected_policy = 'baseline'

    if method == METHODOLOGY_SMALL_LARGE:
        selected_policy = f'{size_route}_policy'
        if size_route == 'thin_fiber':
            confidence_multiplier *= 0.96
        if size_route == 'thin_fiber' and diameter_f is not None and context_width > 0 and diameter_f > context_width * 1.65:
            flags.append('small_large_width_incoherent')
            confidence_multiplier *= 0.72

    elif method == METHODOLOGY_HALO_AWARE:
        selected_policy = 'internal_edge_preferred' if halo_status == 'halo_suspect' else 'edge_pair'
        if halo_status == 'halo_suspect':
            flags.append('halo_suspect')
            confidence_multiplier *= 0.86
            if diameter_f is not None and context_width > 0 and diameter_f > context_width * 1.75 and support_path < 0.22:
                reject_reason = 'halo_external_edge_risk'

    elif method == METHODOLOGY_RIDGE_ANCHORED:
        selected_policy = 'ridge_center_validated'
        if ridge_status == 'weak_anchor':
            flags.append('ridge_anchor_weak')
            confidence_multiplier *= 0.78
            if center_offset > max(5.0, context_width * 0.45):
                reject_reason = 'ridge_anchor_inconsistent'

    elif method == METHODOLOGY_FLUX_AWARE:
        selected_policy = 'axis_flux_validated'
        if flux_status == 'ambiguous':
            flags.append('flux_neighbor_risk')
            confidence_multiplier *= 0.80
            if neighbor_flux > axis_flux + 0.34:
                reject_reason = 'neighbor_flux_stronger'

    elif method == METHODOLOGY_CONTOUR_REFINE:
        selected_policy = 'edge_pair_or_contour_refine'
        if contour_status == 'used':
            confidence_multiplier *= 0.88

    elif method == METHODOLOGY_CURVELET_AIDED:
        selected_policy = 'structure_map_validated'
        if curvelet_status == 'weak_structure':
            flags.append('curvelet_structure_weak')
            confidence_multiplier *= 0.84

    return {
        'methodology_id': method,
        'selected_methodology': method,
        'local_context_label': str(context.get('local_context_label') or ''),
        'local_context_labels': labels,
        'local_context_scores': scores,
        'methodology_reason': ','.join(labels) if labels else 'clean_local',
        'selected_edge_policy': selected_policy,
        'size_route': size_route,
        'halo_status': halo_status,
        'ridge_anchor_status': ridge_status,
        'ridge_response': float(ridge_response),
        'edge_pair_center_offset_px': float(center_offset),
        'flux_status': flux_status,
        'axis_flux_score': float(axis_flux),
        'neighbor_flux_score': float(neighbor_flux),
        'contour_refine_status': contour_status,
        'curvelet_status': curvelet_status,
        'curvelet_edge_score': float(curvelet_score),
        'confidence_multiplier': float(confidence_multiplier),
        'flags': flags,
        'reject_reason': reject_reason,
        'preprocess_keys': sorted(list(dict(preprocess or {}).keys())),
        'geometry_status': str(geometry.get('geometry_status') or ''),
    }
