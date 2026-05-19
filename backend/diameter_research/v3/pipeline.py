from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from ...image_codec import to_gray_u8, to_uint8_rgb
from ..pipeline import DEFAULT_PARAMS as V1_DEFAULT_PARAMS
from ..pipeline import build_overlay
from ..pipeline_v2 import build_weighted_support
from ..pipeline_v2 import _sanitize_points
from .diagnostics import json_ready, profile_npz_payload
from .edge_pairs import measure_diameter_from_edge_pairs
from .fallback import measure_diameter_fallback
from .geometry_guard import evaluate_local_geometry_ambiguity
from .local_preprocess import build_local_preprocess_diagnostics
from .local_orientation import estimate_local_orientation_from_image
from .methodologies import (
    METHODOLOGY_CONTOUR_REFINE,
    METHODOLOGY_CURVELET_AIDED,
    METHODOLOGY_FLUX_AWARE,
    METHODOLOGY_HALO_AWARE,
    METHODOLOGY_NONE,
    METHODOLOGY_RIDGE_ANCHORED,
    METHODOLOGY_SMALL_LARGE,
    classify_local_context,
    methodology_diagnostics,
    params_for_methodology,
)
from .multiscale import multiscale_decision
from .recenter import recenter_point_on_local_axis
from .support_roi import build_local_support_roi


METHOD_ID_V3 = 'hybrid_profile_diameter_v3'
METHOD_ID_V3_1 = 'hybrid_profile_diameter_v3_1'
METHOD_ID_V3_2 = 'hybrid_profile_diameter_v3_2'
METHOD_ID_V3_2_AUTO = 'hybrid_profile_diameter_v3_2_auto'
METHOD_ID_V3_2_SMALL_MASK = 'hybrid_profile_diameter_v3_2_small_mask'
METHOD_ID_V3_2_LARGE_IMAGE = 'hybrid_profile_diameter_v3_2_large_image'
METHOD_ID_CIRCLE_SQUARE = 'circle_square_mask_diameter'
METHOD_ID_MANUAL_DUAL_SIDE = 'manual_dual_side_caliper'
METHOD_ID_MANUAL_LINE_DIRECT = 'manual_line_direct_caliper'
METHOD_ID_ELLIPSE_FIT = 'ellipse_oriented_fit'
METHOD_ID_LOCO = 'loco_circle_probe'
METHOD_ID_V3_3 = 'hybrid_profile_diameter_v3_3'
METHOD_ID_V3_3A = 'hybrid_profile_diameter_v3_3a'
METHOD_ID_V3_3B = 'hybrid_profile_diameter_v3_3b'
METHOD_ID_V3_3C = 'hybrid_profile_diameter_v3_3c'
METHOD_ID_V3_3D = 'hybrid_profile_diameter_v3_3d'
METHOD_ID_V3_2_SMALL_LARGE = 'hybrid_profile_diameter_v3_2_small_large'
METHOD_ID_V3_2_HALO_AWARE = 'hybrid_profile_diameter_v3_2_halo_aware'
METHOD_ID_V3_2_RIDGE_ANCHORED = 'hybrid_profile_diameter_v3_2_ridge_anchored'
METHOD_ID_V3_2_FLUX_AWARE = 'hybrid_profile_diameter_v3_2_flux_aware'
METHOD_ID_V3_2_CONTOUR_REFINE = 'hybrid_profile_diameter_v3_2_contour_refine'
METHOD_ID_V3_2_CURVELET_AIDED = 'hybrid_profile_diameter_v3_2_curvelet_aided'
V3_METHOD_IDS = {
    METHOD_ID_V3,
    METHOD_ID_V3_1,
    METHOD_ID_V3_2,
    METHOD_ID_V3_2_AUTO,
    METHOD_ID_V3_2_SMALL_MASK,
    METHOD_ID_V3_2_LARGE_IMAGE,
    METHOD_ID_CIRCLE_SQUARE,
    METHOD_ID_MANUAL_DUAL_SIDE,
    METHOD_ID_MANUAL_LINE_DIRECT,
    METHOD_ID_ELLIPSE_FIT,
    METHOD_ID_LOCO,
    METHOD_ID_V3_3,
    METHOD_ID_V3_3A,
    METHOD_ID_V3_3B,
    METHOD_ID_V3_3C,
    METHOD_ID_V3_3D,
    METHOD_ID_V3_2_SMALL_LARGE,
    METHOD_ID_V3_2_HALO_AWARE,
    METHOD_ID_V3_2_RIDGE_ANCHORED,
    METHOD_ID_V3_2_FLUX_AWARE,
    METHOD_ID_V3_2_CONTOUR_REFINE,
    METHOD_ID_V3_2_CURVELET_AIDED,
}

DEFAULT_PARAMS_V3: dict[str, Any] = {
    **V1_DEFAULT_PARAMS,
    'orientation_sweep_deg': [-10, -5, 0, 5, 10],
    'local_roi_radius_px': 56,
    'support_refine_enabled': True,
    'support_refine_strength': 0.35,
    'thin_fiber_support_mode': True,
    'thin_fiber_threshold_px': 8,
    'orientation_image_sigma': 1.4,
    'min_orientation_coherence': 0.18,
    'recenter_radius_px': 6,
    'max_recenter_shift_px': 8,
    'geometry_window_px': 48,
    'edge_pair_candidate_count': 5,
    'edge_pair_min_score': 0.22,
    'fallback_enabled': True,
    'multiscale_enabled': True,
    'upscale_factors': [2, 3],
    'upscale_method': 'bicubic',
    'antihalo_enabled': True,
    'min_point_confidence': 0.45,
    'bimodal_width_gap_ratio': 0.22,
    'geometry_guard_enabled': True,
    'small_diameter_flag_enabled': True,
    'local_geometry_control_enabled': False,
    'adaptive_profile_length_enabled': False,
    'overshoot_max_context_width_ratio': 1.85,
    'overshoot_margin_px': 6,
    'overshoot_support_path_min': 0.18,
    'overshoot_support_threshold': 0.15,
    'adaptive_profile_context_scale': 2.4,
    'adaptive_profile_min_length_px': 24,
    'adaptive_profile_dense_threshold': 0.22,
    'adaptive_profile_coherence_threshold': 0.30,
    'adaptive_profile_risk_scale': 0.72,
    'local_preprocess_diagnostics_enabled': False,
    'rolling_ball_radius_px': 28,
    'clahe_clip_limit': 1.6,
    'clahe_tile_grid_px': 24,
    'ridge_sigma_px': 1.2,
    'methodology_id': METHODOLOGY_NONE,
    'small_large_enabled': False,
    'halo_aware_enabled': False,
    'ridge_anchor_enabled': False,
    'flux_aware_enabled': False,
    'contour_refine_enabled': False,
    'curvelet_aided_enabled': False,
    'fiber_size_mode': 'large',
    'mask_driven_enabled': False,
    'mask_driven_method': 'caliper_raycast',
    'mask_local_radius_px': 32,
    'mask_recenter_radius_px': 6,
    'mask_trace_step_px': 0.5,
    'mask_max_trace_px': 0,
    'mask_ray_count': 36,
    'mask_min_width_px': 2,
    'mask_min_confidence': 0.35,
    'mask_caliper_raycast_max_delta_px': 5,
    'auto_small_distance_threshold_px': 6.5,
    'auto_small_context_width_px': 14,
    'interactive_geometry_method': '',
    'circle_square_seed_mode': 'manual_circle',
    'circle_square_seed_radius_px': 8,
    'circle_square_max_radius_px': 26,
    'circle_square_radius_step_px': 1,
    'circle_square_length_factor': 0.9,
    'circle_square_width_factor': 0.7,
    'circle_square_samples': 7,
    'circle_square_aggregation': 'median',
    'circle_square_recenter_seed': True,
    'circle_square_max_recenter_shift_px': 5,
    'circle_square_geometry_id': '',
    'circle_square_circles_by_point': [],
    'manual_geometry_id': '',
    'interactive_geometry_id': '',
    'ellipse_roi_radius_px': 42,
    'manual_caliper_refine': True,
    'manual_left_x': None,
    'manual_left_y': None,
    'manual_right_x': None,
    'manual_right_y': None,
    'manual_lines_by_point': [],
    'manual_line_mask_min_run_px': 1.0,
    'manual_line_mask_gap_tolerance_px': 0.0,
    'loco_roi_radius_px': 36,
    'loco_max_radius_px': 26,
    'loco_radius_step_px': 1,
    'loco_seed_radius_px': 0,
    'loco_seed_radius_window_px': 8,
    'loco_circle_samples': 128,
    'loco_recenter_enabled': True,
    'loco_max_recenter_shift_px': 5,
    'loco_symmetry_threshold': 0.62,
    'loco_reject_threshold': 0.42,
    'loco_mode': 'refine',
    'loco_aggregation': 'median',
    'loco_seed_radii_by_point': {},
}


VARIANT_DEFAULTS: dict[str, dict[str, Any]] = {
    METHOD_ID_V3_1: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': False,
        'multiscale_enabled': False,
        'antihalo_enabled': False,
        'geometry_guard_enabled': False,
        'small_diameter_flag_enabled': False,
        'local_geometry_control_enabled': False,
        'adaptive_profile_length_enabled': False,
    },
    METHOD_ID_V3_2: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': False,
        'multiscale_enabled': False,
        'antihalo_enabled': False,
        'geometry_guard_enabled': False,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': True,
        'adaptive_profile_length_enabled': True,
    },
    METHOD_ID_V3_2_AUTO: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': False,
        'multiscale_enabled': False,
        'antihalo_enabled': True,
        'geometry_guard_enabled': False,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': True,
        'adaptive_profile_length_enabled': True,
        'fiber_size_mode': 'auto',
        'mask_driven_enabled': True,
        'mask_driven_method': 'caliper_raycast',
    },
    METHOD_ID_V3_2_SMALL_MASK: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': False,
        'multiscale_enabled': False,
        'antihalo_enabled': True,
        'geometry_guard_enabled': False,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': True,
        'adaptive_profile_length_enabled': True,
        'fiber_size_mode': 'small',
        'mask_driven_enabled': True,
        'mask_driven_method': 'caliper_raycast',
    },
    METHOD_ID_V3_2_LARGE_IMAGE: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': False,
        'multiscale_enabled': False,
        'antihalo_enabled': True,
        'geometry_guard_enabled': False,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': True,
        'adaptive_profile_length_enabled': True,
        'fiber_size_mode': 'large',
        'mask_driven_enabled': False,
        'mask_driven_method': 'edge_pair',
    },
    METHOD_ID_CIRCLE_SQUARE: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': False,
        'multiscale_enabled': False,
        'antihalo_enabled': False,
        'geometry_guard_enabled': False,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': False,
        'adaptive_profile_length_enabled': False,
        'fiber_size_mode': 'interactive',
        'mask_driven_enabled': True,
        'interactive_geometry_method': 'circle_square_mask_diameter',
    },
    METHOD_ID_MANUAL_DUAL_SIDE: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': False,
        'multiscale_enabled': False,
        'antihalo_enabled': False,
        'geometry_guard_enabled': False,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': False,
        'adaptive_profile_length_enabled': False,
        'fiber_size_mode': 'interactive',
        'mask_driven_enabled': True,
        'interactive_geometry_method': 'manual_dual_side_caliper',
    },
    METHOD_ID_MANUAL_LINE_DIRECT: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': False,
        'multiscale_enabled': False,
        'antihalo_enabled': False,
        'geometry_guard_enabled': False,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': False,
        'adaptive_profile_length_enabled': False,
        'fiber_size_mode': 'interactive',
        'mask_driven_enabled': False,
        'interactive_geometry_method': 'manual_line_direct_caliper',
    },
    METHOD_ID_ELLIPSE_FIT: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': False,
        'multiscale_enabled': False,
        'antihalo_enabled': False,
        'geometry_guard_enabled': False,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': False,
        'adaptive_profile_length_enabled': False,
        'fiber_size_mode': 'interactive',
        'mask_driven_enabled': True,
        'interactive_geometry_method': 'ellipse_oriented_fit',
    },
    METHOD_ID_LOCO: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': False,
        'multiscale_enabled': False,
        'antihalo_enabled': False,
        'geometry_guard_enabled': False,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': False,
        'adaptive_profile_length_enabled': False,
        'fiber_size_mode': 'interactive',
        'mask_driven_enabled': True,
        'interactive_geometry_method': 'loco_circle_probe',
    },
    METHOD_ID_V3_3: {
        'support_refine_enabled': True,
        'thin_fiber_support_mode': True,
        'fallback_enabled': True,
        'multiscale_enabled': True,
        'antihalo_enabled': True,
        'geometry_guard_enabled': True,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': True,
        'adaptive_profile_length_enabled': True,
    },
    METHOD_ID_V3_3A: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': False,
        'multiscale_enabled': False,
        'antihalo_enabled': False,
        'geometry_guard_enabled': True,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': True,
        'adaptive_profile_length_enabled': True,
    },
    METHOD_ID_V3_3B: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': True,
        'multiscale_enabled': False,
        'antihalo_enabled': False,
        'geometry_guard_enabled': False,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': True,
        'adaptive_profile_length_enabled': True,
    },
    METHOD_ID_V3_3C: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': False,
        'multiscale_enabled': True,
        'antihalo_enabled': False,
        'geometry_guard_enabled': False,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': True,
        'adaptive_profile_length_enabled': True,
    },
    METHOD_ID_V3_3D: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': False,
        'multiscale_enabled': False,
        'antihalo_enabled': True,
        'geometry_guard_enabled': False,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': True,
        'adaptive_profile_length_enabled': True,
    },
    METHOD_ID_V3_2_SMALL_LARGE: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': False,
        'multiscale_enabled': False,
        'antihalo_enabled': False,
        'geometry_guard_enabled': False,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': True,
        'adaptive_profile_length_enabled': True,
        'methodology_id': METHODOLOGY_SMALL_LARGE,
        'small_large_enabled': True,
    },
    METHOD_ID_V3_2_HALO_AWARE: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': False,
        'multiscale_enabled': False,
        'antihalo_enabled': True,
        'geometry_guard_enabled': False,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': True,
        'adaptive_profile_length_enabled': True,
        'methodology_id': METHODOLOGY_HALO_AWARE,
        'halo_aware_enabled': True,
    },
    METHOD_ID_V3_2_RIDGE_ANCHORED: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': False,
        'multiscale_enabled': False,
        'antihalo_enabled': False,
        'geometry_guard_enabled': False,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': True,
        'adaptive_profile_length_enabled': True,
        'local_preprocess_diagnostics_enabled': True,
        'methodology_id': METHODOLOGY_RIDGE_ANCHORED,
        'ridge_anchor_enabled': True,
    },
    METHOD_ID_V3_2_FLUX_AWARE: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': False,
        'multiscale_enabled': False,
        'antihalo_enabled': False,
        'geometry_guard_enabled': False,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': True,
        'adaptive_profile_length_enabled': True,
        'methodology_id': METHODOLOGY_FLUX_AWARE,
        'flux_aware_enabled': True,
    },
    METHOD_ID_V3_2_CONTOUR_REFINE: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': True,
        'multiscale_enabled': False,
        'antihalo_enabled': False,
        'geometry_guard_enabled': False,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': True,
        'adaptive_profile_length_enabled': True,
        'methodology_id': METHODOLOGY_CONTOUR_REFINE,
        'contour_refine_enabled': True,
    },
    METHOD_ID_V3_2_CURVELET_AIDED: {
        'support_refine_enabled': False,
        'thin_fiber_support_mode': False,
        'fallback_enabled': False,
        'multiscale_enabled': False,
        'antihalo_enabled': False,
        'geometry_guard_enabled': False,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': True,
        'adaptive_profile_length_enabled': True,
        'local_preprocess_diagnostics_enabled': True,
        'methodology_id': METHODOLOGY_CURVELET_AIDED,
        'curvelet_aided_enabled': True,
    },
    METHOD_ID_V3: {
        'support_refine_enabled': True,
        'thin_fiber_support_mode': True,
        'fallback_enabled': True,
        'multiscale_enabled': True,
        'antihalo_enabled': True,
        'geometry_guard_enabled': True,
        'small_diameter_flag_enabled': True,
        'local_geometry_control_enabled': True,
        'adaptive_profile_length_enabled': True,
    },
}

VARIANT_LOCKED_KEYS = {
    'support_refine_enabled',
    'thin_fiber_support_mode',
    'fallback_enabled',
    'multiscale_enabled',
    'antihalo_enabled',
    'geometry_guard_enabled',
    'small_diameter_flag_enabled',
    'local_geometry_control_enabled',
    'adaptive_profile_length_enabled',
    'methodology_id',
    'small_large_enabled',
    'halo_aware_enabled',
    'ridge_anchor_enabled',
    'flux_aware_enabled',
    'contour_refine_enabled',
    'curvelet_aided_enabled',
    'fiber_size_mode',
    'mask_driven_enabled',
    'mask_driven_method',
    'interactive_geometry_method',
}


def _param_float(params: dict[str, Any] | None, key: str, default: float) -> float:
    try:
        return float((params or {}).get(key, default))
    except Exception:
        return float(default)


def _param_bool(params: dict[str, Any] | None, key: str, default: bool) -> bool:
    value = (params or {}).get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def _resize_mask_like(arr: np.ndarray, shape_hw: tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    a = np.asarray(arr)
    if a.shape[:2] == (h, w):
        return a
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(a, (w, h), interpolation=interp)


def _interactive_support_mask(
    *,
    prior_map: np.ndarray | None,
    labels: np.ndarray | None,
    fallback_support: np.ndarray,
    shape_hw: tuple[int, int],
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the mask shown and used by manual geometry methods.

    Manual methods should not require a high-confidence prior seed. A thin fiber
    can be visible and intentionally marked while still living only in the low
    probability band, so use the inclusive low-threshold mask for measurement.
    """
    support = (np.asarray(fallback_support) > 0).astype(np.uint8)
    meta: dict[str, Any] = {
        'interactive_support_source': 'fallback_support',
        'interactive_support_pixels': int(np.sum(support > 0)),
    }
    if prior_map is not None and np.asarray(prior_map).size:
        prior = _resize_mask_like(np.asarray(prior_map, dtype=np.float32), shape_hw)
        prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
        if prior.size and float(np.nanmax(prior)) > 1.5:
            prior = prior / 255.0
        prior = np.clip(prior, 0.0, 1.0)
        low = float(np.clip(_param_float(params, 'support_low_threshold', 0.35), 0.0, 1.0))
        support = (prior >= low).astype(np.uint8)
        meta.update(
            {
                'interactive_support_source': 'prior_low_threshold',
                'interactive_support_low_threshold': float(low),
                'interactive_support_prior_min': float(np.min(prior)) if prior.size else 0.0,
                'interactive_support_prior_max': float(np.max(prior)) if prior.size else 0.0,
                'interactive_support_pixels': int(np.sum(support > 0)),
            }
        )
    if labels is not None and meta.get('interactive_support_source') != 'prior_low_threshold':
        lab = _resize_mask_like(np.asarray(labels, dtype=np.uint8), shape_hw, nearest=True)
        fiber = lab == 1
        background = lab == 3
        if np.any(fiber):
            support = np.maximum(support, fiber.astype(np.uint8))
        if np.any(background):
            support = np.where(background, 0, support).astype(np.uint8)
        meta.update(
            {
                'interactive_support_fiber_scribble_px': int(np.sum(fiber)),
                'interactive_support_background_scribble_px': int(np.sum(background)),
                'interactive_support_pixels': int(np.sum(support > 0)),
            }
        )
    return support.astype(np.uint8), meta


def _normalize_params(params: dict[str, Any] | None, method_id: str = METHOD_ID_V3_3) -> dict[str, Any]:
    out = dict(DEFAULT_PARAMS_V3)
    variant_defaults = VARIANT_DEFAULTS.get(str(method_id), VARIANT_DEFAULTS[METHOD_ID_V3_3])
    out.update(variant_defaults)
    for key, value in dict(params or {}).items():
        if key in out:
            out[key] = value
    for key in VARIANT_LOCKED_KEYS:
        if key in variant_defaults:
            out[key] = variant_defaults[key]
    if 'methodology_id' not in variant_defaults:
        out['methodology_id'] = METHODOLOGY_NONE
        for key in (
            'small_large_enabled',
            'halo_aware_enabled',
            'ridge_anchor_enabled',
            'flux_aware_enabled',
            'contour_refine_enabled',
            'curvelet_aided_enabled',
        ):
            out[key] = False
    sweep = out.get('orientation_sweep_deg', [-10, -5, 0, 5, 10])
    if isinstance(sweep, str):
        sweep = [x.strip() for x in sweep.split(',') if x.strip()]
    try:
        out['orientation_sweep_deg'] = [float(x) for x in list(sweep)]
    except Exception:
        out['orientation_sweep_deg'] = [-10.0, -5.0, 0.0, 5.0, 10.0]
    if 0.0 not in [float(x) for x in out['orientation_sweep_deg']]:
        out['orientation_sweep_deg'] = [0.0, *out['orientation_sweep_deg']]
    if isinstance(out.get('upscale_factors'), str):
        out['upscale_factors'] = [x.strip() for x in str(out['upscale_factors']).split(',') if x.strip()]
    return out


def _unit(v: Any, fallback: tuple[float, float]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(arr))
    if not np.isfinite(n) or n < 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return arr / n


def _rotate(v: np.ndarray, degrees: float) -> np.ndarray:
    a = float(np.deg2rad(degrees))
    c = float(np.cos(a))
    s = float(np.sin(a))
    x, y = float(v[0]), float(v[1])
    return _unit([c * x - s * y, s * x + c * y], (1.0, 0.0))


def _sweep_edge_pairs(
    *,
    gray_f: np.ndarray,
    support_weight: np.ndarray,
    center_xy: tuple[float, float],
    orientation: dict[str, Any],
    params: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_t = _unit(orientation.get('tangent', [1.0, 0.0]), (1.0, 0.0))
    candidates: list[dict[str, Any]] = []
    for delta in list(params.get('orientation_sweep_deg') or [0.0]):
        deg = float(delta)
        t = _rotate(base_t, deg)
        n = _unit([-t[1], t[0]], (0.0, 1.0))
        orient = dict(orientation)
        orient['tangent'] = [float(t[0]), float(t[1])]
        orient['normal'] = [float(n[0]), float(n[1])]
        measured = measure_diameter_from_edge_pairs(
            gray_f=gray_f,
            support_weight=support_weight,
            center_xy=center_xy,
            orientation=orient,
            params=params,
        )
        score = float(measured.get('profile_consensus', 0.0)) + 0.35 * float(measured.get('edge_pair_score', 0.0)) - abs(deg) / 360.0
        if measured.get('status') != 'ok':
            score -= 0.35
        measured['orientation_delta_deg'] = float(deg)
        measured['orientation'] = orient
        measured['candidate_score'] = float(score)
        candidates.append(measured)
    candidates.sort(key=lambda item: float(item.get('candidate_score', -1e9)), reverse=True)
    return (candidates[0] if candidates else {'status': 'failed', 'reason': 'no_orientation_candidates'}), candidates


def _mask_at(mask: np.ndarray, x: float, y: float) -> bool:
    h, w = np.asarray(mask).shape[:2]
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    return 0 <= xi < w and 0 <= yi < h and bool(mask[yi, xi] > 0)


def _mask_line_inside_ratio(mask: np.ndarray, p0: tuple[float, float], p1: tuple[float, float]) -> float:
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dist = float(np.hypot(x1 - x0, y1 - y0))
    samples = max(3, int(round(dist * 2.0)) + 1)
    xs = np.linspace(x0, x1, samples, dtype=np.float32)
    ys = np.linspace(y0, y1, samples, dtype=np.float32)
    vals = [_mask_at(mask, float(x), float(y)) for x, y in zip(xs, ys)]
    return float(np.mean(vals)) if vals else 0.0


def _circle_mask_quadrilateral(
    *,
    mask: np.ndarray,
    center_xy: tuple[float, float],
    radius_px: float,
    fallback_tangent: np.ndarray,
) -> dict[str, Any]:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    radius = max(1.0, float(radius_px))
    sample_count = int(max(192, min(1440, round(2.0 * math.pi * radius * 6.0))))
    angles = np.linspace(0.0, 2.0 * math.pi, sample_count, endpoint=False, dtype=np.float64)
    xs = cx + np.cos(angles) * radius
    ys = cy + np.sin(angles) * radius
    inside = np.asarray([_mask_at(mask, float(x), float(y)) for x, y in zip(xs, ys)], dtype=bool)
    if not np.any(inside):
        return {'status': 'failed', 'reason': 'circle_square_circle_no_mask_intersection', 'inside_ratio': 0.0, 'intersection_count': 0}
    if bool(np.all(inside)):
        return {'status': 'failed', 'reason': 'circle_square_circle_fully_inside_mask', 'inside_ratio': 1.0, 'intersection_count': 0}

    false_idxs = np.where(~inside)[0]
    rot = int(false_idxs[0])
    rolled = np.roll(inside, -rot)
    runs: list[dict[str, Any]] = []
    in_run = False
    start = 0
    for j in range(1, sample_count + 1):
        prev = bool(rolled[j - 1])
        cur = bool(rolled[j % sample_count]) if j < sample_count else bool(rolled[0])
        if not prev and cur and not in_run:
            start = j
            in_run = True
        if prev and not cur and in_run:
            end = j - 1
            length = max(1, end - start + 1)
            step = 2.0 * math.pi / float(sample_count)
            start_angle = ((start - 0.5 + rot) % sample_count) * step
            end_angle = ((end + 0.5 + rot) % sample_count) * step
            p0 = np.asarray([cx + math.cos(start_angle) * radius, cy + math.sin(start_angle) * radius], dtype=np.float64)
            p1 = np.asarray([cx + math.cos(end_angle) * radius, cy + math.sin(end_angle) * radius], dtype=np.float64)
            runs.append({'length': int(length), 'angles': [float(start_angle), float(end_angle)], 'points': [p0, p1], 'midpoint': (p0 + p1) * 0.5})
            in_run = False

    runs = [r for r in runs if int(r.get('length', 0)) >= 2]
    if len(runs) < 2:
        return {
            'status': 'failed',
            'reason': 'circle_square_circle_intersections_not_four',
            'inside_ratio': float(np.mean(inside)),
            'intersection_count': int(len(runs) * 2),
            'inside_arc_count': int(len(runs)),
        }
    runs.sort(key=lambda item: int(item.get('length', 0)), reverse=True)
    selected = runs[:2]
    arc_midpoints = [np.asarray(r['midpoint'], dtype=np.float64) for r in selected]
    tangent = _unit(arc_midpoints[1] - arc_midpoints[0], tuple(float(v) for v in _unit(fallback_tangent, (1.0, 0.0))))
    normal = _unit([-tangent[1], tangent[0]], (0.0, 1.0))
    points = [np.asarray(p, dtype=np.float64) for r in selected for p in r['points']]
    projections = [float(np.dot(p - np.asarray([cx, cy], dtype=np.float64), normal)) for p in points]
    order = np.argsort(projections)
    low = [points[int(order[0])], points[int(order[1])]]
    high = [points[int(order[2])], points[int(order[3])]]
    low.sort(key=lambda p: float(np.dot(p, tangent)))
    high.sort(key=lambda p: float(np.dot(p, tangent)))
    low0, low1 = low
    high0, high1 = high
    quad = [high0, high1, low1, low0]
    width_center = float(np.linalg.norm(((high0 + high1) * 0.5) - ((low0 + low1) * 0.5)))
    length_center = float(np.linalg.norm(((high0 + low0) * 0.5) - ((high1 + low1) * 0.5)))
    return {
        'status': 'ok',
        'reason': '',
        'inside_ratio': float(np.mean(inside)),
        'intersection_count': 4,
        'inside_arc_count': int(len(selected)),
        'quad_vertices_xy': [[float(p[0]), float(p[1])] for p in quad],
        'high_edge': [[float(high0[0]), float(high0[1])], [float(high1[0]), float(high1[1])]],
        'low_edge': [[float(low0[0]), float(low0[1])], [float(low1[0]), float(low1[1])]],
        'tangent': [float(tangent[0]), float(tangent[1])],
        'normal': [float(normal[0]), float(normal[1])],
        'width_center_px': width_center,
        'length_center_px': length_center,
        'selected_arc_lengths': [int(r.get('length', 0)) for r in selected],
    }


def _mask_recenter_by_distance_transform(
    *,
    support_mask: np.ndarray,
    point_xy: tuple[float, float],
    params: dict[str, Any],
) -> tuple[tuple[float, float], dict[str, Any]]:
    mask = (np.asarray(support_mask, dtype=np.uint8) > 0).astype(np.uint8)
    if not np.any(mask):
        return point_xy, {'status': 'failed', 'reason': 'mask_empty', 'recenter_shift_px': 0.0}
    h, w = mask.shape[:2]
    radius = max(1.0, _param_float(params, 'mask_recenter_radius_px', 10.0))
    cx = float(np.clip(float(point_xy[0]), 0.0, max(0, w - 1)))
    cy = float(np.clip(float(point_xy[1]), 0.0, max(0, h - 1)))
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

    x0 = max(0, int(np.floor(cx - radius)))
    x1 = min(w, int(np.ceil(cx + radius)) + 1)
    y0 = max(0, int(np.floor(cy - radius)))
    y1 = min(h, int(np.ceil(cy + radius)) + 1)
    crop = dt[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disk = ((xx.astype(np.float32) - cx) ** 2 + (yy.astype(np.float32) - cy) ** 2) <= radius ** 2
    candidate = (crop > 0) & disk
    if np.any(candidate):
        max_dt = float(np.max(crop[candidate]))
        stable = candidate & (crop >= max_dt * 0.96)
        d2 = (xx.astype(np.float32) - cx) ** 2 + (yy.astype(np.float32) - cy) ** 2
        local_scores = np.where(stable, d2, np.inf)
        flat_idx = int(np.argmin(local_scores))
        ly, lx = np.unravel_index(flat_idx, crop.shape)
        out_xy = (float(x0 + lx), float(y0 + ly))
        status = 'ok'
        reason = ''
    else:
        coords = np.argwhere(mask > 0)
        d2 = (coords[:, 1].astype(np.float64) - cx) ** 2 + (coords[:, 0].astype(np.float64) - cy) ** 2
        best = coords[int(np.argmin(d2))]
        out_xy = (float(best[1]), float(best[0]))
        status = 'nearest_support'
        reason = 'no_mask_inside_recenter_window'

    shift = float(np.hypot(out_xy[0] - cx, out_xy[1] - cy))
    ox = int(np.clip(round(out_xy[0]), 0, max(0, w - 1)))
    oy = int(np.clip(round(out_xy[1]), 0, max(0, h - 1)))
    return out_xy, {
        'status': status,
        'reason': reason,
        'recenter_shift_px': shift,
        'mask_center_distance_px': float(dt[oy, ox]),
        'mask_local_max_distance_px': float(np.max(crop)) if crop.size else 0.0,
    }


def _trace_mask_edge(
    *,
    support_mask: np.ndarray,
    center_xy: tuple[float, float],
    direction: np.ndarray,
    params: dict[str, Any],
) -> dict[str, Any]:
    mask = (np.asarray(support_mask, dtype=np.uint8) > 0).astype(np.uint8)
    direction = _unit(direction, (1.0, 0.0))
    step = max(0.25, _param_float(params, 'mask_trace_step_px', 0.5))
    max_len_param = _param_float(params, 'mask_max_trace_px', 0.0)
    max_len = max_len_param if max_len_param > 0.0 else _param_float(params, 'mask_local_radius_px', _param_float(params, 'profile_length_px', 80.0) * 0.6)
    max_len = max(3.0, max_len)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    if not _mask_at(mask, cx, cy):
        return {'status': 'failed', 'reason': 'center_outside_mask'}
    last_xy = (cx, cy)
    last_dist = 0.0
    dist = step
    while dist <= max_len:
        x = cx + float(direction[0]) * dist
        y = cy + float(direction[1]) * dist
        if not _mask_at(mask, x, y):
            return {'status': 'ok', 'edge_xy': [float(last_xy[0]), float(last_xy[1])], 'distance_px': float(last_dist)}
        last_xy = (float(x), float(y))
        last_dist = float(dist)
        dist += step
    return {
        'status': 'failed',
        'reason': 'trace_did_not_exit_mask',
        'edge_xy': [float(last_xy[0]), float(last_xy[1])],
        'distance_px': float(last_dist),
    }


def _mask_caliper_measurement(
    *,
    support_mask: np.ndarray,
    center_xy: tuple[float, float],
    orientation: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    mask = (np.asarray(support_mask, dtype=np.uint8) > 0).astype(np.uint8)
    normal = _unit(orientation.get('normal', [0.0, 1.0]), (0.0, 1.0))
    left = _trace_mask_edge(support_mask=mask, center_xy=center_xy, direction=-normal, params=params)
    right = _trace_mask_edge(support_mask=mask, center_xy=center_xy, direction=normal, params=params)
    if left.get('status') != 'ok' or right.get('status') != 'ok':
        return {
            'status': 'failed',
            'reason': str(left.get('reason') or right.get('reason') or 'mask_caliper_failed'),
            'diameter_px': None,
        }
    left_xy = tuple(float(v) for v in left.get('edge_xy', [center_xy[0], center_xy[1]]))
    right_xy = tuple(float(v) for v in right.get('edge_xy', [center_xy[0], center_xy[1]]))
    dl = float(left.get('distance_px', 0.0) or 0.0)
    dr = float(right.get('distance_px', 0.0) or 0.0)
    diameter = float(dl + dr)
    min_width = max(0.0, _param_float(params, 'mask_min_width_px', 2.0))
    if diameter < min_width:
        return {'status': 'failed', 'reason': 'mask_width_too_small', 'diameter_px': None}

    inside = _mask_line_inside_ratio(mask, left_xy, right_xy)
    symmetry = float(1.0 - abs(dl - dr) / max(diameter, 1e-6))
    center_dist = 0.0
    try:
        dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        xi = int(np.clip(round(center_xy[0]), 0, mask.shape[1] - 1))
        yi = int(np.clip(round(center_xy[1]), 0, mask.shape[0] - 1))
        center_dist = float(dt[yi, xi])
    except Exception:
        center_dist = 0.0
    center_score = float(np.clip(center_dist / max(diameter * 0.45, 1e-6), 0.0, 1.0))
    orient_score = float(np.clip(float(orientation.get('orientation_coherence', orientation.get('confidence', 0.5)) or 0.5), 0.0, 1.0))
    confidence = float(np.clip(0.36 * inside + 0.26 * symmetry + 0.22 * center_score + 0.16 * orient_score, 0.0, 1.0))
    status = 'ok' if confidence >= _param_float(params, 'mask_min_confidence', 0.35) else 'failed'
    reason = '' if status == 'ok' else 'mask_confidence_low'
    return {
        'status': status,
        'reason': reason,
        'diameter_px': diameter if status == 'ok' else None,
        'raw_diameter_px': diameter,
        'left_edge_xy': [float(left_xy[0]), float(left_xy[1])],
        'right_edge_xy': [float(right_xy[0]), float(right_xy[1])],
        'mask_inside_ratio': inside,
        'mask_symmetry_score': symmetry,
        'mask_center_score': center_score,
        'mask_confidence': confidence,
        'edge_pair_score': confidence,
        'profile_consensus': inside,
        'valid_profiles': 1 if status == 'ok' else 0,
        'total_profiles': 1,
        'mad_px': 0.0,
        'edge_score_mean': confidence,
        'geometry_control_status': 'mask_caliper',
        'profile_length_effective_px': diameter,
        'context_width_px': diameter,
        'support_path_mean': inside,
    }


def _mask_raycast_measurement(
    *,
    support_mask: np.ndarray,
    center_xy: tuple[float, float],
    orientation: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    mask = (np.asarray(support_mask, dtype=np.uint8) > 0).astype(np.uint8)
    ray_count = int(max(8, round(_param_float(params, 'mask_ray_count', 36.0))))
    ray_count = ray_count + (ray_count % 2)
    tangent = _unit(orientation.get('tangent', [1.0, 0.0]), (1.0, 0.0))
    normal = _unit(orientation.get('normal', [-tangent[1], tangent[0]]), (0.0, 1.0))
    widths: list[dict[str, Any]] = []
    for idx, angle in enumerate(np.linspace(0.0, np.pi, max(4, ray_count // 2), endpoint=False)):
        direction = np.asarray([float(np.cos(angle)), float(np.sin(angle))], dtype=np.float64)
        a = _trace_mask_edge(support_mask=mask, center_xy=center_xy, direction=direction, params=params)
        b = _trace_mask_edge(support_mask=mask, center_xy=center_xy, direction=-direction, params=params)
        if a.get('status') != 'ok' or b.get('status') != 'ok':
            continue
        width = float(a.get('distance_px', 0.0) or 0.0) + float(b.get('distance_px', 0.0) or 0.0)
        if width <= 0:
            continue
        align = float(abs(np.dot(_unit(direction, (1.0, 0.0)), normal)))
        widths.append(
            {
                'idx': int(idx),
                'angle_rad': float(angle),
                'width_px': width,
                'normal_alignment': align,
                'left_edge_xy': b.get('edge_xy'),
                'right_edge_xy': a.get('edge_xy'),
            }
        )
    if not widths:
        return {'status': 'failed', 'reason': 'no_raycast_widths', 'diameter_px': None, 'ray_candidates': []}
    values = np.asarray([float(w['width_px']) for w in widths], dtype=np.float64)
    median_width = float(np.median(values))
    best = max(widths, key=lambda item: float(item.get('normal_alignment', 0.0)) - abs(float(item.get('width_px', 0.0)) - median_width) / max(median_width, 1e-6) * 0.18)
    mad = float(np.median(np.abs(values - median_width))) if values.size else 0.0
    consistency = float(np.clip(1.0 - mad / max(median_width * 0.35, 1e-6), 0.0, 1.0))
    align = float(np.clip(best.get('normal_alignment', 0.0), 0.0, 1.0))
    confidence = float(np.clip(0.58 * consistency + 0.42 * align, 0.0, 1.0))
    return {
        'status': 'ok' if confidence >= _param_float(params, 'mask_min_confidence', 0.35) else 'failed',
        'reason': '' if confidence >= _param_float(params, 'mask_min_confidence', 0.35) else 'raycast_confidence_low',
        'diameter_px': float(best['width_px']) if confidence >= _param_float(params, 'mask_min_confidence', 0.35) else None,
        'raw_diameter_px': float(best['width_px']),
        'median_width_px': median_width,
        'mad_px': mad,
        'left_edge_xy': best.get('left_edge_xy'),
        'right_edge_xy': best.get('right_edge_xy'),
        'ray_candidates': widths,
        'mask_confidence': confidence,
        'edge_pair_score': confidence,
        'profile_consensus': consistency,
        'valid_profiles': int(len(widths)),
        'total_profiles': int(max(1, ray_count // 2)),
        'edge_score_mean': confidence,
        'geometry_control_status': 'mask_raycast',
        'profile_length_effective_px': float(best['width_px']),
        'context_width_px': median_width,
        'support_path_mean': consistency,
    }


def _run_mask_driven_measurement(
    *,
    support_mask: np.ndarray,
    point_xy: tuple[float, float],
    orientation: dict[str, Any],
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    center_xy, recenter_diag = _mask_recenter_by_distance_transform(
        support_mask=support_mask,
        point_xy=point_xy,
        params=params,
    )
    if recenter_diag.get('status') == 'failed':
        measurement = {
            'status': 'failed',
            'reason': str(recenter_diag.get('reason') or 'mask_recenter_failed'),
            'diameter_px': None,
        }
        return measurement, {'recenter': recenter_diag, 'caliper': {}, 'raycast': {}, 'selected': 'none'}

    caliper = _mask_caliper_measurement(
        support_mask=support_mask,
        center_xy=center_xy,
        orientation=orientation,
        params=params,
    )
    raycast = _mask_raycast_measurement(
        support_mask=support_mask,
        center_xy=center_xy,
        orientation=orientation,
        params=params,
    )
    selected = dict(caliper)
    selected_method = 'mask_caliper'
    if caliper.get('status') != 'ok' and raycast.get('status') == 'ok':
        selected = dict(raycast)
        selected_method = 'mask_raycast'
    elif caliper.get('status') == 'ok' and raycast.get('status') == 'ok':
        cd = float(caliper.get('raw_diameter_px', caliper.get('diameter_px', 0.0)) or 0.0)
        rd = float(raycast.get('raw_diameter_px', raycast.get('diameter_px', 0.0)) or 0.0)
        delta = abs(cd - rd)
        selected['mask_raycast_delta_px'] = float(delta)
        selected['mask_raycast_diameter_px'] = rd
        if delta > _param_float(params, 'mask_caliper_raycast_max_delta_px', 5.0):
            selected['edge_pair_score'] = float(selected.get('edge_pair_score', 0.0)) * 0.72
            selected['profile_consensus'] = float(selected.get('profile_consensus', 0.0)) * 0.72
            selected['mask_confidence'] = float(selected.get('mask_confidence', 0.0)) * 0.72
            selected['reason'] = 'mask_caliper_raycast_disagree'
            if selected['mask_confidence'] < _param_float(params, 'mask_min_confidence', 0.35):
                selected['status'] = 'failed'
                selected['diameter_px'] = None
    selected['measurement_mode'] = selected_method if selected.get('status') == 'ok' else 'rejected'
    selected['mask_method'] = selected_method
    selected['x'] = float(center_xy[0])
    selected['y'] = float(center_xy[1])
    selected['mask_center_xy'] = [float(center_xy[0]), float(center_xy[1])]
    selected['mask_center_shift_px'] = float(recenter_diag.get('recenter_shift_px', 0.0) or 0.0)
    selected['mask_center_distance_px'] = float(recenter_diag.get('mask_center_distance_px', 0.0) or 0.0)
    selected['mask_caliper_diameter_px'] = None if caliper.get('raw_diameter_px') is None else float(caliper.get('raw_diameter_px'))
    selected['mask_raycast_diameter_px'] = None if raycast.get('raw_diameter_px') is None else float(raycast.get('raw_diameter_px'))
    return selected, {
        'recenter': recenter_diag,
        'caliper': caliper,
        'raycast': raycast,
        'selected': selected_method,
    }


def _trimmed_mean(values: np.ndarray, trim: float = 0.18) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    if arr.size == 0:
        return 0.0
    k = int(np.floor(arr.size * trim))
    if k > 0 and arr.size > 2 * k:
        arr = arr[k:-k]
    return float(np.mean(arr))


def _circle_transition_candidates(
    *,
    mask: np.ndarray,
    center_xy: tuple[float, float],
    radius: float,
    samples: int,
) -> dict[str, Any]:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    n = int(max(32, samples))
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float64)
    xs = cx + np.cos(angles) * float(radius)
    ys = cy + np.sin(angles) * float(radius)
    inside = np.asarray([_mask_at(mask, float(x), float(y)) for x, y in zip(xs, ys)], dtype=bool)
    transition_indices = np.where(inside != np.roll(inside, 1))[0]
    intersections: list[dict[str, Any]] = []
    for idx in transition_indices.tolist():
        prev_idx = int((idx - 1) % n)
        angle = float((angles[idx] + angles[prev_idx]) * 0.5)
        if angle < 0.0:
            angle += float(2.0 * np.pi)
        intersections.append(
            {
                'idx': int(idx),
                'angle_rad': angle,
                'xy': [float(cx + np.cos(angle) * radius), float(cy + np.sin(angle) * radius)],
            }
        )

    best_pair: dict[str, Any] | None = None
    best_symmetry = 0.0
    if len(intersections) >= 2:
        for a_idx in range(len(intersections)):
            for b_idx in range(a_idx + 1, len(intersections)):
                a = float(intersections[a_idx]['angle_rad'])
                b = float(intersections[b_idx]['angle_rad'])
                delta = abs(a - b) % (2.0 * np.pi)
                delta = min(delta, 2.0 * np.pi - delta)
                symmetry = float(np.clip(1.0 - abs(delta - np.pi) / max(np.pi, 1e-6), 0.0, 1.0))
                if symmetry > best_symmetry:
                    best_symmetry = symmetry
                    best_pair = {
                        'indices': [int(a_idx), int(b_idx)],
                        'angle_delta_rad': float(delta),
                        'left_edge_xy': intersections[a_idx]['xy'],
                        'right_edge_xy': intersections[b_idx]['xy'],
                    }

    transition_count = int(len(intersections))
    inside_ratio = float(np.mean(inside)) if inside.size else 0.0
    transition_score = 0.0
    if transition_count in {2, 4}:
        transition_score = 1.0
    elif transition_count > 0:
        transition_score = float(np.clip(1.0 - abs(transition_count - 4) / 8.0, 0.0, 0.82))
    inside_score = float(np.clip(1.0 - abs(inside_ratio - 0.50) / 0.50, 0.0, 1.0))
    return {
        'radius_px': float(radius),
        'inside_ratio': inside_ratio,
        'transition_count': transition_count,
        'intersections': intersections,
        'best_pair': best_pair,
        'symmetry_score': float(best_symmetry),
        'transition_score': float(transition_score),
        'inside_score': float(inside_score),
    }


def _loco_radius_sequence(params: dict[str, Any]) -> list[float]:
    max_radius = max(2.0, _param_float(params, 'loco_max_radius_px', 26.0))
    step = max(0.5, _param_float(params, 'loco_radius_step_px', 1.0))
    seed_radius = _param_float(params, 'loco_seed_radius_px', 0.0)
    if seed_radius > 0.0:
        window = max(step, _param_float(params, 'loco_seed_radius_window_px', 8.0))
        start = max(1.0, seed_radius - window)
        end = min(max_radius, seed_radius + window)
    else:
        start = 1.0
        end = max_radius
    values = list(np.arange(start, end + step * 0.5, step, dtype=np.float64))
    if seed_radius > 0.0 and 1.0 <= seed_radius <= max_radius:
        values.append(float(seed_radius))
    unique = sorted({round(float(v), 4) for v in values if 1.0 <= float(v) <= max_radius})
    return [float(v) for v in unique]


def _loco_refine_from_radius(
    *,
    support_mask: np.ndarray,
    center_xy: tuple[float, float],
    orientation: dict[str, Any],
    radius: float,
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    mask = (np.asarray(support_mask, dtype=np.uint8) > 0).astype(np.uint8)
    tangent = _unit(orientation.get('tangent', [1.0, 0.0]), (1.0, 0.0))
    normal = _unit(orientation.get('normal', [-tangent[1], tangent[0]]), (0.0, 1.0))
    half_len = max(2.0, float(radius) * 0.90)
    half_width = max(2.0, float(radius) * 1.15)
    sample_count = int(max(3, round(_param_float(params, 'circle_square_samples', 7.0))))
    offsets = np.linspace(-half_len, half_len, sample_count, dtype=np.float64)
    trace_params = {**params, 'mask_max_trace_px': float(half_width)}
    cx, cy = float(center_xy[0]), float(center_xy[1])
    samples: list[dict[str, Any]] = []
    for idx, offset in enumerate(offsets):
        sample_center = (float(cx + tangent[0] * offset), float(cy + tangent[1] * offset))
        if not _mask_at(mask, sample_center[0], sample_center[1]):
            continue
        left = _trace_mask_edge(support_mask=mask, center_xy=sample_center, direction=-normal, params=trace_params)
        right = _trace_mask_edge(support_mask=mask, center_xy=sample_center, direction=normal, params=trace_params)
        if left.get('status') != 'ok' or right.get('status') != 'ok':
            continue
        left_xy = tuple(float(v) for v in left.get('edge_xy', sample_center))
        right_xy = tuple(float(v) for v in right.get('edge_xy', sample_center))
        width = float(left.get('distance_px', 0.0) or 0.0) + float(right.get('distance_px', 0.0) or 0.0)
        if width <= 0.0:
            continue
        samples.append(
            {
                'idx': int(idx),
                'offset_px': float(offset),
                'center_xy': [float(sample_center[0]), float(sample_center[1])],
                'width_px': float(width),
                'left_edge_xy': [float(left_xy[0]), float(left_xy[1])],
                'right_edge_xy': [float(right_xy[0]), float(right_xy[1])],
                'inside_ratio': _mask_line_inside_ratio(mask, left_xy, right_xy),
            }
        )
    if not samples:
        return {'status': 'failed', 'reason': 'loco_refine_no_transverse_samples', 'diameter_px': None}, {'samples': []}
    widths = np.asarray([float(item['width_px']) for item in samples], dtype=np.float64)
    aggregation = str(params.get('loco_aggregation') or 'median')
    diameter = _trimmed_mean(widths) if aggregation == 'trimmed_mean' else float(np.median(widths))
    mad = float(np.median(np.abs(widths - float(np.median(widths)))))
    consistency = float(np.clip(1.0 - mad / max(diameter * 0.35, 1e-6), 0.0, 1.0))
    inside = float(np.mean([float(item.get('inside_ratio', 0.0)) for item in samples]))
    valid_ratio = float(len(samples) / max(1, sample_count))
    confidence = float(np.clip(0.46 * consistency + 0.32 * inside + 0.22 * valid_ratio, 0.0, 1.0))
    representative = min(samples, key=lambda item: abs(float(item['width_px']) - diameter))
    return {
        'status': 'ok',
        'reason': '',
        'diameter_px': float(diameter),
        'raw_diameter_px': float(diameter),
        'left_edge_xy': representative.get('left_edge_xy'),
        'right_edge_xy': representative.get('right_edge_xy'),
        'mad_px': float(mad),
        'profile_consensus': consistency,
        'support_path_mean': inside,
        'valid_profiles': int(len(samples)),
        'total_profiles': int(sample_count),
        'edge_score_mean': confidence,
        'edge_pair_score': confidence,
        'profile_length_effective_px': float(half_width * 2.0),
        'context_width_px': float(diameter),
        'square_half_length_px': float(half_len),
        'square_half_width_px': float(half_width),
        'square_samples_valid': int(len(samples)),
        'square_samples_total': int(sample_count),
    }, {'samples': samples, 'aggregation': aggregation}


def _loco_circle_probe_measurement(
    *,
    support_mask: np.ndarray,
    point_xy: tuple[float, float],
    point_index: int,
    orientation: dict[str, Any],
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    mask = (np.asarray(support_mask, dtype=np.uint8) > 0).astype(np.uint8)
    if not np.any(mask):
        return {'status': 'rejected', 'reason': 'mask_empty', 'diameter_px': None, 'measurement_mode': 'rejected'}, {}

    point_params = dict(params)
    seed_map = point_params.get('loco_seed_radii_by_point')
    if isinstance(seed_map, dict):
        for key in (str(point_index), str(int(point_index))):
            if key in seed_map:
                try:
                    point_params['loco_seed_radius_px'] = float(seed_map[key])
                except Exception:
                    pass
                break

    recenter_enabled = _param_bool(point_params, 'loco_recenter_enabled', True)
    if recenter_enabled:
        recenter_params = {
            **point_params,
            'mask_recenter_radius_px': min(
                _param_float(point_params, 'loco_roi_radius_px', 36.0),
                _param_float(point_params, 'loco_max_recenter_shift_px', 5.0),
            ),
        }
        center_xy, recenter_diag = _mask_recenter_by_distance_transform(support_mask=mask, point_xy=point_xy, params=recenter_params)
    else:
        center_xy = point_xy
        recenter_diag = {'status': 'disabled', 'reason': '', 'recenter_shift_px': 0.0, 'mask_center_distance_px': 0.0}
    if not _mask_at(mask, center_xy[0], center_xy[1]):
        return {
            'status': 'rejected',
            'reason': 'loco_seed_outside_mask',
            'diameter_px': None,
            'measurement_mode': 'rejected',
            'loco_recenter_shift_px': float(recenter_diag.get('recenter_shift_px', 0.0) or 0.0),
        }, {'recenter': recenter_diag}

    radii = _loco_radius_sequence(point_params)
    sample_count = int(max(32, round(_param_float(point_params, 'loco_circle_samples', 128.0))))
    candidates = [
        _circle_transition_candidates(mask=mask, center_xy=center_xy, radius=float(radius), samples=sample_count)
        for radius in radii
    ]
    shift = float(recenter_diag.get('recenter_shift_px', 0.0) or 0.0)
    shift_penalty = float(np.clip(shift / max(_param_float(point_params, 'loco_max_recenter_shift_px', 5.0), 1e-6), 0.0, 1.0)) * 0.16
    max_radius = max(1.0, _param_float(point_params, 'loco_max_radius_px', 26.0))
    for idx, cand in enumerate(candidates):
        prev_c = candidates[idx - 1] if idx > 0 else cand
        next_c = candidates[idx + 1] if idx + 1 < len(candidates) else cand
        stability = 1.0 - min(
            1.0,
            (
                abs(float(cand['inside_ratio']) - float(prev_c['inside_ratio']))
                + abs(float(cand['inside_ratio']) - float(next_c['inside_ratio']))
                + abs(int(cand['transition_count']) - int(prev_c['transition_count'])) / 8.0
                + abs(int(cand['transition_count']) - int(next_c['transition_count'])) / 8.0
            )
            * 0.5,
        )
        branch_penalty = max(0.0, int(cand['transition_count']) - 4) * 0.045
        radius_penalty = float(cand['radius_px']) / max_radius * 0.08
        score = (
            0.38 * float(cand['symmetry_score'])
            + 0.22 * float(cand['transition_score'])
            + 0.18 * float(cand['inside_score'])
            + 0.14 * float(stability)
            + 0.08 * float(1.0 - min(1.0, branch_penalty * 3.0))
            - branch_penalty
            - radius_penalty
            - shift_penalty
        )
        cand['stability_score'] = float(np.clip(stability, 0.0, 1.0))
        cand['circle_symmetry_score'] = float(np.clip(score, 0.0, 1.0))
    best = max(candidates, key=lambda item: float(item.get('circle_symmetry_score', 0.0))) if candidates else None
    if not best:
        return {'status': 'rejected', 'reason': 'loco_no_radius_candidates', 'diameter_px': None, 'measurement_mode': 'rejected'}, {'recenter': recenter_diag, 'radius_candidates': []}

    best_score = float(best.get('circle_symmetry_score', 0.0))
    reject_threshold = _param_float(point_params, 'loco_reject_threshold', 0.42)
    if best_score < reject_threshold:
        return {
            'status': 'rejected',
            'reason': 'loco_symmetry_low',
            'diameter_px': None,
            'measurement_mode': 'rejected',
            'mask_method': 'loco_circle_probe',
            'loco_best_radius_px': float(best['radius_px']),
            'loco_symmetry_score': best_score,
            'loco_intersection_count': int(best.get('transition_count', 0)),
            'loco_recenter_shift_px': shift,
            'radius_candidates': candidates,
        }, {'recenter': recenter_diag, 'radius_candidates': candidates, 'best': best}

    mode = str(point_params.get('loco_mode') or 'refine').strip().lower()
    normal = _unit(orientation.get('normal', [0.0, 1.0]), (0.0, 1.0))
    direct_left = [float(center_xy[0] - normal[0] * float(best['radius_px'])), float(center_xy[1] - normal[1] * float(best['radius_px']))]
    direct_right = [float(center_xy[0] + normal[0] * float(best['radius_px'])), float(center_xy[1] + normal[1] * float(best['radius_px']))]
    direct = {
        'status': 'ok',
        'reason': '',
        'diameter_px': float(best['radius_px']) * 2.0,
        'raw_diameter_px': float(best['radius_px']) * 2.0,
        'left_edge_xy': direct_left,
        'right_edge_xy': direct_right,
        'mad_px': 0.0,
        'profile_consensus': float(best.get('stability_score', 0.0)),
        'support_path_mean': float(best.get('inside_ratio', 0.0)),
        'valid_profiles': 1,
        'total_profiles': 1,
        'edge_score_mean': best_score,
        'edge_pair_score': best_score,
        'profile_length_effective_px': float(best['radius_px']) * 2.0,
        'context_width_px': float(best['radius_px']) * 2.0,
    }
    refine: dict[str, Any] = {'status': 'failed', 'reason': 'loco_refine_skipped'}
    refine_diag: dict[str, Any] = {}
    selected = dict(direct)
    measurement_mode = 'loco_direct'
    if mode == 'refine':
        refine, refine_diag = _loco_refine_from_radius(
            support_mask=mask,
            center_xy=center_xy,
            orientation=orientation,
            radius=float(best['radius_px']),
            params=point_params,
        )
        if refine.get('status') == 'ok':
            selected = dict(refine)
            measurement_mode = 'loco_refine'

    confidence = float(np.clip(0.62 * best_score + 0.38 * float(selected.get('edge_pair_score', best_score) or 0.0), 0.0, 1.0))
    selected.update(
        {
            'status': 'ok' if confidence >= reject_threshold else 'rejected',
            'reason': '' if confidence >= reject_threshold else 'loco_confidence_low',
            'diameter_px': float(selected.get('diameter_px')) if confidence >= reject_threshold else None,
            'measurement_mode': measurement_mode if confidence >= reject_threshold else 'rejected',
            'mask_method': 'loco_circle_probe',
            'mask_confidence': confidence,
            'mask_center_xy': [float(center_xy[0]), float(center_xy[1])],
            'mask_center_shift_px': shift,
            'mask_center_distance_px': float(recenter_diag.get('mask_center_distance_px', 0.0) or 0.0),
            'loco_mode': mode,
            'loco_seed_radius_px': None if _param_float(point_params, 'loco_seed_radius_px', 0.0) <= 0.0 else float(_param_float(point_params, 'loco_seed_radius_px', 0.0)),
            'loco_best_radius_px': float(best['radius_px']),
            'loco_symmetry_score': best_score,
            'loco_intersection_count': int(best.get('transition_count', 0)),
            'loco_recenter_shift_px': shift,
            'radius_candidates': candidates,
            'loco_intersections_xy': [item.get('xy') for item in list(best.get('intersections') or [])],
            'edge_score_mean': confidence,
            'edge_pair_score': confidence,
            'geometry_control_status': 'loco_circle_probe',
        }
    )
    return selected, {
        'recenter': recenter_diag,
        'radius_candidates': candidates,
        'best': best,
        'direct': direct,
        'refine': refine,
        'refine_diag': refine_diag,
        'mode': mode,
    }


def _draw_loco_geometry_overlay(image_rgb: np.ndarray, results: list[dict[str, Any]]) -> np.ndarray:
    out = to_uint8_rgb(image_rgb).copy()
    for result in results:
        center = result.get('recentered_xy') or result.get('original_xy')
        if not center:
            continue
        try:
            cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
        except Exception:
            continue
        radius = result.get('loco_best_radius_px')
        if radius is not None:
            try:
                cv2.circle(out, (cx, cy), max(1, int(round(float(radius)))), (255, 234, 0), 1, lineType=cv2.LINE_AA)
            except Exception:
                pass
        for xy in list(result.get('loco_intersections_xy') or []):
            try:
                ix, iy = int(round(float(xy[0]))), int(round(float(xy[1])))
                cv2.circle(out, (ix, iy), 2, (255, 132, 0), thickness=-1, lineType=cv2.LINE_AA)
            except Exception:
                continue
        cv2.circle(out, (cx, cy), 3, (255, 23, 68), thickness=-1, lineType=cv2.LINE_AA)
    return out


def _circle_square_mask_measurement(
    *,
    support_mask: np.ndarray,
    point_xy: tuple[float, float],
    orientation: dict[str, Any],
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    mask = (np.asarray(support_mask, dtype=np.uint8) > 0).astype(np.uint8)
    if not np.any(mask):
        return {'status': 'failed', 'reason': 'mask_empty', 'diameter_px': None}, {}
    seed_mode = str(params.get('circle_square_seed_mode') or 'manual_circle')
    manual_circle_locked = seed_mode == 'manual_circle'
    if not manual_circle_locked:
        return {
            'status': 'failed',
            'reason': 'circle_square_requires_manual_circle',
            'diameter_px': None,
            'measurement_mode': 'rejected',
            'mask_method': 'circle_square_mask_diameter',
        }, {
            'seed_mode': seed_mode,
            'manual_circle_required': True,
        }
    recenter_enabled = _param_bool(params, 'circle_square_recenter_seed', True) and not manual_circle_locked
    if recenter_enabled:
        recenter_params = {
            **params,
            'mask_recenter_radius_px': min(
                _param_float(params, 'mask_recenter_radius_px', 6.0),
                _param_float(params, 'circle_square_max_recenter_shift_px', 5.0),
            ),
        }
        center_xy, recenter_diag = _mask_recenter_by_distance_transform(support_mask=mask, point_xy=point_xy, params=recenter_params)
    else:
        center_xy = point_xy
        recenter_diag = {
            'status': 'locked_manual_circle' if manual_circle_locked else 'disabled',
            'recenter_shift_px': 0.0,
            'mask_center_distance_px': 0.0,
        }
    if not manual_circle_locked and not _mask_at(mask, center_xy[0], center_xy[1]):
        return {'status': 'failed', 'reason': 'seed_outside_mask', 'diameter_px': None}, {'recenter': recenter_diag}

    tangent = _unit(orientation.get('tangent', [1.0, 0.0]), (1.0, 0.0))
    normal = _unit(orientation.get('normal', [-tangent[1], tangent[0]]), (0.0, 1.0))
    start_radius = max(1.0, _param_float(params, 'circle_square_seed_radius_px', 8.0) if seed_mode == 'manual_circle' else float(recenter_diag.get('mask_center_distance_px', 0.0) or 3.0))
    max_radius = start_radius if manual_circle_locked else max(start_radius + 1.0, _param_float(params, 'circle_square_max_radius_px', 26.0))
    step = max(0.5, _param_float(params, 'circle_square_radius_step_px', 1.0))
    chosen_radius = start_radius
    circle_diag: list[dict[str, Any]] = []
    angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    radius = start_radius
    while radius <= max_radius:
        xs = cx + np.cos(angles) * radius
        ys = cy + np.sin(angles) * radius
        inside = np.asarray([_mask_at(mask, float(x), float(y)) for x, y in zip(xs, ys)], dtype=bool)
        inside_ratio = float(np.mean(inside))
        transitions = int(np.sum(inside != np.roll(inside, 1)))
        circle_diag.append({'radius_px': float(radius), 'inside_ratio': inside_ratio, 'transitions': transitions})
        chosen_radius = float(radius)
        if manual_circle_locked or (transitions >= 2 and inside_ratio < 0.92):
            break
        radius += step

    measurement_mask = mask
    if manual_circle_locked:
        yy, xx = np.indices(mask.shape)
        circle_bound = ((xx.astype(np.float64) - cx) ** 2 + (yy.astype(np.float64) - cy) ** 2) <= (float(chosen_radius) + 0.5) ** 2
        measurement_mask = (mask.astype(bool) & circle_bound).astype(np.uint8)
        if not np.any(measurement_mask):
            return {
                'status': 'failed',
                'reason': 'circle_square_empty_inside_manual_circle',
                'diameter_px': None,
                'measurement_mode': 'rejected',
                'mask_method': 'circle_square_mask_diameter',
            }, {'recenter': recenter_diag, 'circle_growth': circle_diag, 'samples': []}

    mask_pixels = int(mask.sum())
    measurement_mask_pixels = int(measurement_mask.sum())
    samples = int(max(3, round(_param_float(params, 'circle_square_samples', 9.0))))
    widths: list[dict[str, Any]] = []
    circle_quad_diag: dict[str, Any] = {}
    if manual_circle_locked:
        circle_quad_diag = _circle_mask_quadrilateral(mask=mask, center_xy=(cx, cy), radius_px=chosen_radius, fallback_tangent=tangent)
        if circle_quad_diag.get('status') != 'ok':
            return {
                'status': 'failed',
                'reason': str(circle_quad_diag.get('reason') or 'circle_square_circle_intersections_failed'),
                'diameter_px': None,
                'measurement_mode': 'rejected',
                'mask_method': 'circle_square_mask_diameter',
                'mask_center_xy': [float(cx), float(cy)],
                'mask_center_shift_px': float(recenter_diag.get('recenter_shift_px', 0.0) or 0.0),
                'mask_center_distance_px': float(recenter_diag.get('mask_center_distance_px', 0.0) or 0.0),
                'circle_radius_px': float(chosen_radius),
                'square_half_length_px': 0.0,
                'square_half_width_px': 0.0,
                'square_vertices_xy': None,
                'circle_square_manual_locked': True,
                'circle_square_mask_pixels': mask_pixels,
                'circle_square_measurement_mask_pixels': measurement_mask_pixels,
                'circle_square_intersection_count': int(circle_quad_diag.get('intersection_count', 0) or 0),
                'circle_square_inside_arc_count': int(circle_quad_diag.get('inside_arc_count', 0) or 0),
                'square_samples_valid': 0,
                'square_samples_total': int(samples),
            }, {'recenter': recenter_diag, 'circle_growth': circle_diag, 'circle_quad': circle_quad_diag, 'samples': []}
        quad_vertices = list(circle_quad_diag.get('quad_vertices_xy') or [])
        tangent = np.asarray(circle_quad_diag.get('tangent') or tangent, dtype=np.float64)
        normal = np.asarray(circle_quad_diag.get('normal') or normal, dtype=np.float64)
        high0 = np.asarray(circle_quad_diag['high_edge'][0], dtype=np.float64)
        high1 = np.asarray(circle_quad_diag['high_edge'][1], dtype=np.float64)
        low0 = np.asarray(circle_quad_diag['low_edge'][0], dtype=np.float64)
        low1 = np.asarray(circle_quad_diag['low_edge'][1], dtype=np.float64)
        half_len = max(0.0, float(circle_quad_diag.get('length_center_px', 0.0) or 0.0) * 0.5)
        half_width = max(0.0, float(circle_quad_diag.get('width_center_px', 0.0) or 0.0) * 0.5)
        for idx, tval in enumerate(np.linspace(0.0, 1.0, samples, dtype=np.float64)):
            high_xy = high0 * (1.0 - tval) + high1 * tval
            low_xy = low0 * (1.0 - tval) + low1 * tval
            width = float(np.linalg.norm(high_xy - low_xy))
            if width <= 0:
                continue
            inside_ratio = _mask_line_inside_ratio(measurement_mask, (float(low_xy[0]), float(low_xy[1])), (float(high_xy[0]), float(high_xy[1])))
            if inside_ratio < 0.35:
                continue
            center_line = (high_xy + low_xy) * 0.5
            widths.append(
                {
                    'idx': int(idx),
                    'offset_px': float((tval - 0.5) * max(float(circle_quad_diag.get('length_center_px', 0.0) or 0.0), 0.0)),
                    'center_xy': [float(center_line[0]), float(center_line[1])],
                    'width_px': width,
                    'left_edge_xy': [float(low_xy[0]), float(low_xy[1])],
                    'right_edge_xy': [float(high_xy[0]), float(high_xy[1])],
                    'inside_ratio': inside_ratio,
                }
            )
    else:
        half_len = max(2.0, chosen_radius * max(0.25, _param_float(params, 'circle_square_length_factor', 1.35)))
        half_width = max(2.0, chosen_radius * max(0.25, _param_float(params, 'circle_square_width_factor', 1.0)))
        quad_vertices = [
            [float(cx + tangent[0] * half_len + normal[0] * half_width), float(cy + tangent[1] * half_len + normal[1] * half_width)],
            [float(cx + tangent[0] * half_len - normal[0] * half_width), float(cy + tangent[1] * half_len - normal[1] * half_width)],
            [float(cx - tangent[0] * half_len - normal[0] * half_width), float(cy - tangent[1] * half_len - normal[1] * half_width)],
            [float(cx - tangent[0] * half_len + normal[0] * half_width), float(cy - tangent[1] * half_len + normal[1] * half_width)],
        ]
        offsets = np.linspace(-half_len, half_len, samples, dtype=np.float64)
        trace_params = {**params, 'mask_max_trace_px': float(half_width)}
        for idx, offset in enumerate(offsets):
            sample_center = (float(cx + tangent[0] * offset), float(cy + tangent[1] * offset))
            if not _mask_at(measurement_mask, sample_center[0], sample_center[1]):
                continue
            left = _trace_mask_edge(support_mask=measurement_mask, center_xy=sample_center, direction=-normal, params=trace_params)
            right = _trace_mask_edge(support_mask=measurement_mask, center_xy=sample_center, direction=normal, params=trace_params)
            if left.get('status') != 'ok' or right.get('status') != 'ok':
                continue
            left_xy = tuple(float(v) for v in left.get('edge_xy', sample_center))
            right_xy = tuple(float(v) for v in right.get('edge_xy', sample_center))
            width = float(left.get('distance_px', 0.0) or 0.0) + float(right.get('distance_px', 0.0) or 0.0)
            if width <= 0:
                continue
            widths.append(
                {
                    'idx': int(idx),
                    'offset_px': float(offset),
                    'center_xy': [float(sample_center[0]), float(sample_center[1])],
                    'width_px': width,
                    'left_edge_xy': [float(left_xy[0]), float(left_xy[1])],
                    'right_edge_xy': [float(right_xy[0]), float(right_xy[1])],
                    'inside_ratio': _mask_line_inside_ratio(measurement_mask, left_xy, right_xy),
                }
            )

    if not widths:
        return {
            'status': 'failed',
            'reason': 'circle_square_no_valid_transverse_samples',
            'diameter_px': None,
            'measurement_mode': 'rejected',
            'mask_method': 'circle_square_mask_diameter',
            'mask_center_xy': [float(cx), float(cy)],
            'mask_center_shift_px': float(recenter_diag.get('recenter_shift_px', 0.0) or 0.0),
            'mask_center_distance_px': float(recenter_diag.get('mask_center_distance_px', 0.0) or 0.0),
            'circle_radius_px': float(chosen_radius),
            'square_half_length_px': float(half_len),
            'square_half_width_px': float(half_width),
            'square_vertices_xy': quad_vertices,
            'circle_square_manual_locked': bool(manual_circle_locked),
            'circle_square_mask_pixels': mask_pixels,
            'circle_square_measurement_mask_pixels': measurement_mask_pixels,
            'circle_square_intersection_count': int(circle_quad_diag.get('intersection_count', 0) or 0),
            'circle_square_inside_arc_count': int(circle_quad_diag.get('inside_arc_count', 0) or 0),
            'square_samples_valid': 0,
            'square_samples_total': int(samples),
        }, {'recenter': recenter_diag, 'circle_growth': circle_diag, 'circle_quad': circle_quad_diag, 'samples': []}

    value_items = list(widths)
    if manual_circle_locked:
        center_sorted = sorted(widths, key=lambda item: (abs(float(item.get('offset_px', 0.0))), abs(float(item.get('width_px', 0.0)))))
        central_n = min(len(center_sorted), 3)
        value_items = center_sorted[:central_n]
    values = np.asarray([float(item['width_px']) for item in value_items], dtype=np.float64)
    aggregation = str(params.get('circle_square_aggregation') or 'median')
    diameter = _trimmed_mean(values) if aggregation == 'trimmed_mean' else float(np.median(values))
    mad = float(np.median(np.abs(values - float(np.median(values)))))
    consistency = float(np.clip(1.0 - mad / max(diameter * 0.35, 1e-6), 0.0, 1.0))
    inside_mean = float(np.mean([float(item.get('inside_ratio', 0.0)) for item in widths]))
    valid_ratio = float(len(widths) / max(1, samples))
    confidence = float(np.clip(0.42 * consistency + 0.34 * inside_mean + 0.24 * valid_ratio, 0.0, 1.0))
    if manual_circle_locked:
        representative = min(widths, key=lambda item: (abs(float(item.get('offset_px', 0.0))), abs(float(item['width_px']) - diameter)))
    else:
        representative = min(widths, key=lambda item: abs(float(item['width_px']) - diameter))
    status = 'ok' if confidence >= _param_float(params, 'mask_min_confidence', 0.35) else 'failed'
    return {
        'status': status,
        'reason': '' if status == 'ok' else 'circle_square_confidence_low',
        'diameter_px': float(diameter) if status == 'ok' else None,
        'raw_diameter_px': float(diameter),
        'left_edge_xy': representative.get('left_edge_xy'),
        'right_edge_xy': representative.get('right_edge_xy'),
        'debug_left_edge_xy': representative.get('left_edge_xy'),
        'debug_right_edge_xy': representative.get('right_edge_xy'),
        'measurement_mode': 'circle_square_mask' if status == 'ok' else 'rejected',
        'mask_method': 'circle_square_mask_diameter',
        'mask_confidence': confidence,
        'mask_center_xy': [float(cx), float(cy)],
        'mask_center_shift_px': float(recenter_diag.get('recenter_shift_px', 0.0) or 0.0),
        'mask_center_distance_px': float(recenter_diag.get('mask_center_distance_px', 0.0) or 0.0),
        'circle_radius_px': float(chosen_radius),
        'square_half_length_px': float(half_len),
        'square_half_width_px': float(half_width),
        'square_vertices_xy': quad_vertices,
        'circle_square_manual_locked': bool(manual_circle_locked),
        'circle_square_mask_pixels': mask_pixels,
        'circle_square_measurement_mask_pixels': measurement_mask_pixels,
        'circle_square_intersection_count': int(circle_quad_diag.get('intersection_count', 0) or 0),
        'circle_square_inside_arc_count': int(circle_quad_diag.get('inside_arc_count', 0) or 0),
        'circle_square_display_offset_px': float(representative.get('offset_px', 0.0) or 0.0),
        'circle_square_aggregation_samples': int(len(value_items)),
        'square_samples_valid': int(len(widths)),
        'square_samples_total': int(samples),
        'edge_pair_score': confidence,
        'profile_consensus': consistency,
        'valid_profiles': int(len(widths)),
        'total_profiles': int(samples),
        'mad_px': mad,
        'edge_score_mean': confidence,
        'geometry_control_status': 'circle_square',
        'profile_length_effective_px': float(half_width * 2.0),
        'context_width_px': float(diameter),
        'support_path_mean': inside_mean,
    }, {
        'recenter': recenter_diag,
        'circle_growth': circle_diag,
        'circle_quad': circle_quad_diag,
        'samples': widths,
        'aggregation': aggregation,
        'manual_circle_locked': bool(manual_circle_locked),
        'quad_vertices_xy': quad_vertices,
    }


def _manual_dual_side_measurement(
    *,
    support_mask: np.ndarray,
    orientation: dict[str, Any],
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = [params.get('manual_left_x'), params.get('manual_left_y'), params.get('manual_right_x'), params.get('manual_right_y')]
    try:
        lx, ly, rx, ry = [float(v) for v in raw]
    except Exception:
        return {'status': 'failed', 'reason': 'manual_dual_side_requires_manual_line', 'diameter_px': None}, {}
    if not np.isfinite([lx, ly, rx, ry]).all():
        return {'status': 'failed', 'reason': 'manual_dual_side_invalid_line', 'diameter_px': None}, {}
    center = ((lx + rx) * 0.5, (ly + ry) * 0.5)
    direction = _unit([rx - lx, ry - ly], (0.0, 1.0))
    mask = (np.asarray(support_mask, dtype=np.uint8) > 0).astype(np.uint8)
    recenter_diag = {'status': 'manual_midpoint', 'recenter_shift_px': 0.0}
    manual_dist = float(np.hypot(rx - lx, ry - ly))
    refine_with_mask = _param_bool(params, 'manual_caliper_refine', True)
    if not refine_with_mask:
        status = 'ok' if manual_dist >= 0.5 else 'failed'
        confidence = 1.0 if status == 'ok' else 0.0
        return {
            'status': status,
            'reason': '' if status == 'ok' else 'manual_line_too_short',
            'diameter_px': float(manual_dist) if status == 'ok' else None,
            'raw_diameter_px': float(manual_dist),
            'left_edge_xy': [float(lx), float(ly)],
            'right_edge_xy': [float(rx), float(ry)],
            'measurement_mode': 'manual_direct_line' if status == 'ok' else 'rejected',
            'mask_method': 'manual_line_direct_caliper',
            'mask_confidence': confidence,
            'mask_center_xy': [float(center[0]), float(center[1])],
            'mask_center_shift_px': 0.0,
            'manual_input_diameter_px': manual_dist,
            'edge_pair_score': confidence,
            'profile_consensus': confidence,
            'valid_profiles': 1 if status == 'ok' else 0,
            'total_profiles': 1,
            'mad_px': 0.0,
            'edge_score_mean': confidence,
            'geometry_control_status': 'manual_line_direct',
            'profile_length_effective_px': float(manual_dist),
            'context_width_px': float(manual_dist),
            'support_path_mean': 1.0,
        }, {'manual_input': {'left_xy': [lx, ly], 'right_xy': [rx, ry], 'diameter_px': manual_dist}}
    if manual_dist < 0.5:
        return {'status': 'failed', 'reason': 'manual_line_too_short', 'diameter_px': None}, {'manual_input': {'left_xy': [lx, ly], 'right_xy': [rx, ry], 'diameter_px': manual_dist}}
    sample_count = int(np.clip(math.ceil(manual_dist / 0.25) + 1, 8, 4096))
    ts = np.linspace(0.0, 1.0, sample_count)
    xs = lx + (rx - lx) * ts
    ys = ly + (ry - ly) * ts
    inside_samples = np.asarray([_mask_at(mask, float(x), float(y)) for x, y in zip(xs, ys)], dtype=bool)
    gap_tolerance_px = max(0.0, _param_float(params, 'manual_line_mask_gap_tolerance_px', 0.0))
    if gap_tolerance_px > 0:
        max_gap_samples = int(round(gap_tolerance_px / max(manual_dist / max(sample_count - 1, 1), 1e-6)))
        if max_gap_samples > 0:
            i = 0
            while i < sample_count:
                if inside_samples[i]:
                    i += 1
                    continue
                j = i
                while j < sample_count and not inside_samples[j]:
                    j += 1
                if i > 0 and j < sample_count and (j - i) <= max_gap_samples:
                    inside_samples[i:j] = True
                i = j
    min_run_px = max(0.0, _param_float(params, 'manual_line_mask_min_run_px', 1.0))
    sample_step_px = manual_dist / max(sample_count - 1, 1)
    run_start = -1
    run_end = -1
    i = 0
    while i < sample_count:
        if not inside_samples[i]:
            i += 1
            continue
        j = i
        while j + 1 < sample_count and bool(inside_samples[j + 1]):
            j += 1
        if float((j - i + 1) * sample_step_px) >= min_run_px:
            run_start = i
            run_end = j
            break
        i = j + 1
    if run_start >= 0:
        start_idx = int(run_start)
        end_idx = int(run_end)

        def point_at(t: float) -> list[float]:
            return [float(lx + (rx - lx) * t), float(ly + (ry - ly) * t)]

        def boundary_t(t_inside: float, t_outside: float) -> float:
            lo = float(min(t_inside, t_outside))
            hi = float(max(t_inside, t_outside))
            inside_is_low = bool(_mask_at(mask, *point_at(lo)))
            for _ in range(10):
                mid = (lo + hi) * 0.5
                mid_inside = bool(_mask_at(mask, *point_at(mid)))
                if mid_inside == inside_is_low:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) * 0.5

        if start_idx <= 0:
            t_left = 0.0
        else:
            t_left = boundary_t(float(ts[start_idx]), float(ts[start_idx - 1]))
        if end_idx >= sample_count - 1:
            t_right = 1.0
        else:
            t_right = boundary_t(float(ts[end_idx]), float(ts[end_idx + 1]))
        if t_right < t_left:
            t_left, t_right = t_right, t_left
        left_xy = point_at(t_left)
        right_xy = point_at(t_right)
        diameter = float(manual_dist * max(0.0, t_right - t_left))
        center = ((left_xy[0] + right_xy[0]) * 0.5, (left_xy[1] + right_xy[1]) * 0.5)
        recenter_diag = {
            'status': 'manual_first_mask_interval',
            'recenter_shift_px': 0.0,
            'manual_line_interval_start_t': float(t_left),
            'manual_line_interval_end_t': float(t_right),
            'manual_line_interval_samples': int(end_idx - start_idx + 1),
            'manual_line_mask_min_run_px': float(min_run_px),
            'manual_line_mask_gap_tolerance_px': float(gap_tolerance_px),
        }
    else:
        left_xy = [lx, ly]
        right_xy = [rx, ry]
        diameter = 0.0
        recenter_diag = {'status': 'manual_line_no_mask_intersection', 'recenter_shift_px': 0.0}
    inside = _mask_line_inside_ratio(mask, tuple(left_xy), tuple(right_xy))
    agreement = float(np.clip(1.0 - abs(diameter - manual_dist) / max(manual_dist, diameter, 1e-6), 0.0, 1.0))
    confidence = float(np.clip(0.58 * inside + 0.42 * agreement, 0.0, 1.0))
    status = 'ok' if diameter >= _param_float(params, 'mask_min_width_px', 2.0) and confidence >= _param_float(params, 'mask_min_confidence', 0.35) else 'failed'
    return {
        'status': status,
        'reason': '' if status == 'ok' else 'manual_dual_side_confidence_low',
        'diameter_px': float(diameter) if status == 'ok' else None,
        'raw_diameter_px': float(diameter),
        'left_edge_xy': [float(left_xy[0]), float(left_xy[1])],
        'right_edge_xy': [float(right_xy[0]), float(right_xy[1])],
        'measurement_mode': ('manual_mask_line' if refine_with_mask else 'manual_direct_line') if status == 'ok' else 'rejected',
        'mask_method': 'manual_dual_side_caliper' if refine_with_mask else 'manual_line_direct_caliper',
        'mask_confidence': confidence,
        'mask_center_xy': [float(center[0]), float(center[1])],
        'mask_center_shift_px': float(recenter_diag.get('recenter_shift_px', 0.0) or 0.0),
        'manual_input_diameter_px': manual_dist,
        'edge_pair_score': confidence,
        'profile_consensus': agreement,
        'valid_profiles': 1 if status == 'ok' else 0,
        'total_profiles': 1,
        'mad_px': 0.0,
        'edge_score_mean': confidence,
        'geometry_control_status': 'manual_dual_side' if refine_with_mask else 'manual_line_direct',
        'profile_length_effective_px': float(diameter),
        'context_width_px': float(diameter),
        'support_path_mean': inside,
    }, {'recenter': recenter_diag, 'manual_input': {'left_xy': [lx, ly], 'right_xy': [rx, ry], 'diameter_px': manual_dist}}


def _ellipse_oriented_fit_measurement(
    *,
    support_mask: np.ndarray,
    point_xy: tuple[float, float],
    orientation: dict[str, Any],
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    mask = (np.asarray(support_mask, dtype=np.uint8) > 0).astype(np.uint8)
    if not np.any(mask):
        return {'status': 'failed', 'reason': 'mask_empty', 'diameter_px': None}, {}
    center_xy, recenter_diag = _mask_recenter_by_distance_transform(support_mask=mask, point_xy=point_xy, params=params)
    h, w = mask.shape[:2]
    cx, cy = float(center_xy[0]), float(center_xy[1])
    radius = max(5.0, _param_float(params, 'ellipse_roi_radius_px', 42.0))
    labels_count, cc = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    xi = int(np.clip(round(cx), 0, max(0, w - 1)))
    yi = int(np.clip(round(cy), 0, max(0, h - 1)))
    label = int(cc[yi, xi]) if labels_count > 1 else 0
    yy, xx = np.mgrid[0:h, 0:w]
    disk = ((xx.astype(np.float32) - cx) ** 2 + (yy.astype(np.float32) - cy) ** 2) <= radius ** 2
    comp = (cc == label) if label > 0 else (mask > 0)
    candidate = np.argwhere((mask > 0) & comp & disk)
    if candidate.shape[0] < 5:
        return {'status': 'failed', 'reason': 'ellipse_not_enough_mask_pixels', 'diameter_px': None}, {'recenter': recenter_diag}
    pts = candidate[:, [1, 0]].astype(np.float32).reshape(-1, 1, 2)
    try:
        (ecx, ecy), (a, b), angle = cv2.fitEllipse(pts)
    except Exception:
        return {'status': 'failed', 'reason': 'ellipse_fit_failed', 'diameter_px': None}, {'recenter': recenter_diag}
    major = float(max(a, b))
    minor = float(min(a, b))
    if minor < _param_float(params, 'mask_min_width_px', 2.0):
        return {'status': 'failed', 'reason': 'ellipse_minor_too_small', 'diameter_px': None}, {'recenter': recenter_diag}
    theta = np.deg2rad(float(angle))
    axis = np.asarray([float(np.cos(theta)), float(np.sin(theta))], dtype=np.float64)
    if a >= b:
        minor_dir = _unit([-axis[1], axis[0]], (0.0, 1.0))
    else:
        minor_dir = _unit(axis, (0.0, 1.0))
    left_xy = [float(ecx - minor_dir[0] * minor * 0.5), float(ecy - minor_dir[1] * minor * 0.5)]
    right_xy = [float(ecx + minor_dir[0] * minor * 0.5), float(ecy + minor_dir[1] * minor * 0.5)]
    fill_ratio = float(candidate.shape[0] / max(1.0, np.pi * (major * 0.5) * (minor * 0.5)))
    axis_ratio = float(np.clip(minor / max(major, 1e-6), 0.0, 1.0))
    confidence = float(np.clip(0.52 * min(fill_ratio, 1.0) + 0.28 * (1.0 - axis_ratio) + 0.20 * float(orientation.get('orientation_coherence', 0.5) or 0.5), 0.0, 1.0))
    status = 'ok' if confidence >= _param_float(params, 'mask_min_confidence', 0.35) else 'failed'
    return {
        'status': status,
        'reason': '' if status == 'ok' else 'ellipse_confidence_low',
        'diameter_px': minor if status == 'ok' else None,
        'raw_diameter_px': minor,
        'left_edge_xy': left_xy,
        'right_edge_xy': right_xy,
        'measurement_mode': 'ellipse_fit' if status == 'ok' else 'rejected',
        'mask_method': 'ellipse_oriented_fit',
        'mask_confidence': confidence,
        'mask_center_xy': [float(ecx), float(ecy)],
        'mask_center_shift_px': float(recenter_diag.get('recenter_shift_px', 0.0) or 0.0),
        'ellipse_major_px': major,
        'ellipse_minor_px': minor,
        'ellipse_angle_deg': float(angle),
        'edge_pair_score': confidence,
        'profile_consensus': float(min(fill_ratio, 1.0)),
        'valid_profiles': int(candidate.shape[0]),
        'total_profiles': int(candidate.shape[0]),
        'mad_px': 0.0,
        'edge_score_mean': confidence,
        'geometry_control_status': 'ellipse_fit',
        'profile_length_effective_px': minor,
        'context_width_px': minor,
        'support_path_mean': float(min(fill_ratio, 1.0)),
    }, {'recenter': recenter_diag, 'ellipse': {'center_xy': [ecx, ecy], 'major_px': major, 'minor_px': minor, 'angle_deg': float(angle), 'pixel_count': int(candidate.shape[0])}}


def _auto_prefers_small_route(
    *,
    measurement: dict[str, Any],
    local_context: dict[str, Any],
    support_meta: dict[str, Any],
    params: dict[str, Any],
) -> tuple[bool, str]:
    diameter = measurement.get('diameter_px')
    context_width = float(measurement.get('context_width_px', local_context.get('context_width_px', 0.0)) or 0.0)
    center_dist = float(support_meta.get('max_distance_px', support_meta.get('mask_center_distance_px', 0.0)) or 0.0)
    if measurement.get('status') != 'ok':
        return True, 'image_route_failed'
    if diameter is not None and float(diameter) <= _param_float(params, 'thin_fiber_threshold_px', 8.0):
        return True, 'thin_diameter'
    if context_width > 0.0 and context_width <= _param_float(params, 'auto_small_context_width_px', 14.0):
        return True, 'small_context_width'
    if center_dist > 0.0 and center_dist <= _param_float(params, 'auto_small_distance_threshold_px', 6.5):
        return True, 'small_distance_transform'
    return False, 'large_image_stable'


def _confidence(
    *,
    measurement: dict[str, Any],
    geometry: dict[str, Any],
    orientation: dict[str, Any],
    support_meta: dict[str, Any],
    measurement_mode: str,
    small_diameter_suspect: bool,
) -> float:
    valid = float(measurement.get('valid_profiles', 0.0))
    total = max(1.0, float(measurement.get('total_profiles', 1.0)))
    valid_ratio = float(np.clip(valid / total, 0.0, 1.0))
    consensus = float(np.clip(float(measurement.get('profile_consensus', 0.0)), 0.0, 1.0))
    edge = float(np.clip(float(measurement.get('edge_pair_score', 0.0)) / 1.4, 0.0, 1.0))
    orient = float(np.clip(float(orientation.get('orientation_coherence', orientation.get('confidence', 0.0)) or 0.0), 0.0, 1.0))
    raw_px = max(1.0, float(support_meta.get('raw_pixels', 0) or 0))
    refined_px = float(support_meta.get('refined_pixels', 0) or 0)
    support = float(np.clip(refined_px / raw_px, 0.0, 1.0))
    geom_status = str(geometry.get('geometry_status') or '')
    geom = 1.0 if geom_status == 'geometry_simple' else 0.74 if geom_status == 'geometry_complex_but_measurable' else 0.25
    conf = 0.23 * valid_ratio + 0.22 * consensus + 0.18 * edge + 0.17 * orient + 0.10 * support + 0.10 * geom
    if measurement_mode == 'fallback':
        conf *= 0.78
    if small_diameter_suspect:
        conf = min(conf, 0.86)
    return float(np.clip(conf, 0.0, 1.0))


def _quality_label(status: str, confidence: float, flags: list[str], measurement_mode: str) -> str:
    if status == 'rejected':
        if 'crossing_likely' in flags:
            return 'crossing_likely'
        if 'geometry_ambiguous' in flags or 'multiple_components_nearby' in flags or 'width_inconsistent' in flags:
            return 'geometry_ambiguous'
        if 'support_weak' in flags:
            return 'support_weak'
        if 'orientation_unstable' in flags:
            return 'orientation_unstable'
        return 'edge_ambiguous'
    if measurement_mode == 'fallback':
        return 'fallback_confidence'
    if 'small_diameter_suspect' in flags:
        return 'small_diameter_suspect'
    if confidence >= 0.78:
        return 'high_confidence'
    if confidence >= 0.58:
        return 'medium_confidence'
    return 'low_confidence'


def _run_hybrid_profile_diameter_v3_variant(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
    method_id: str = METHOD_ID_V3_3,
) -> dict[str, Any]:
    method_id = str(method_id if method_id in V3_METHOD_IDS else METHOD_ID_V3_3)
    rgb = to_uint8_rgb(image_rgb)
    if rgb is None:
        raise ValueError('Imagen invalida.')
    gray_u8 = to_gray_u8(rgb)
    if gray_u8 is None:
        raise ValueError('No se pudo convertir imagen a gris.')
    shape_hw = rgb.shape[:2]
    effective_params = _normalize_params(params, method_id=method_id)
    variant_stage = 'v3_3' if method_id == METHOD_ID_V3 else method_id.replace('hybrid_profile_diameter_', '')
    clean_points = _sanitize_points(points, shape_hw)
    if not clean_points:
        raise ValueError('No hay puntos validos para medir.')

    support, support_weight, support_meta = build_weighted_support(prior_map=prior_map, labels=labels, shape_hw=shape_hw, params=effective_params)
    if not np.any(support > 0):
        raise ValueError('No se pudo construir soporte desde prior ni scribbles.')
    interactive_support, interactive_support_meta = _interactive_support_mask(
        prior_map=prior_map,
        labels=labels,
        fallback_support=support,
        shape_hw=shape_hw,
        params=effective_params,
    )
    support_meta = {**dict(support_meta or {}), **interactive_support_meta}

    gray_f = gray_u8.astype(np.float32) / 255.0
    results: list[dict[str, Any]] = []
    diagnostics_points: list[dict[str, Any]] = []
    support_refined_union = np.zeros(shape_hw, dtype=np.uint8)

    for point_idx, point in enumerate(clean_points):
        original_xy = (float(point['x']), float(point['y']))
        roi = build_local_support_roi(
            gray_f=gray_f,
            support=support,
            support_weight=support_weight,
            point_xy=original_xy,
            params=effective_params,
        )
        preprocess_diag = build_local_preprocess_diagnostics(
            gray_f=gray_f,
            roi_bbox=tuple(roi.get('bbox') or (0, 0, 0, 0)),
            params=effective_params,
        )
        local_support = np.asarray(roi['support_refined_global'], dtype=np.uint8)
        local_weight = np.asarray(roi['support_refined_weight_global'], dtype=np.float32)
        if not np.any(local_support):
            local_support = np.asarray(roi['support_raw_global'], dtype=np.uint8)
            local_weight = np.where(local_support > 0, support_weight, 0.0).astype(np.float32)
        support_refined_union = np.maximum(support_refined_union, local_support.astype(np.uint8))

        orientation = estimate_local_orientation_from_image(
            gray_f=gray_f,
            support=local_support,
            point_xy=original_xy,
            params=effective_params,
        )
        recentered_xy, recenter_diag = recenter_point_on_local_axis(
            gray_f=gray_f,
            support_weight=local_weight,
            support_refined=local_support,
            point_xy=original_xy,
            orientation=orientation,
            params=effective_params,
        )
        orientation = estimate_local_orientation_from_image(
            gray_f=gray_f,
            support=local_support,
            point_xy=recentered_xy,
            params=effective_params,
        )
        methodology_id = str(effective_params.get('methodology_id') or METHODOLOGY_NONE)
        local_context = classify_local_context(
            gray_f=gray_f,
            support_weight=local_weight,
            support_refined=local_support,
            center_xy=recentered_xy,
            orientation=orientation,
            params=effective_params,
        )
        point_params = params_for_methodology(methodology_id, effective_params, local_context)
        manual_lines = point_params.get('manual_lines_by_point')
        if isinstance(manual_lines, list) and point_idx < len(manual_lines) and isinstance(manual_lines[point_idx], dict):
            line_params = manual_lines[point_idx]
            for line_key in ('manual_left_x', 'manual_left_y', 'manual_right_x', 'manual_right_y', 'manual_geometry_id'):
                if line_key in line_params:
                    point_params[line_key] = line_params[line_key]
        manual_circles = point_params.get('circle_square_circles_by_point')
        if isinstance(manual_circles, list) and point_idx < len(manual_circles) and isinstance(manual_circles[point_idx], dict):
            circle_params = manual_circles[point_idx]
            for circle_key in ('circle_square_seed_radius_px', 'circle_square_center_x', 'circle_square_center_y', 'circle_square_geometry_id'):
                if circle_key in circle_params:
                    point_params[circle_key] = circle_params[circle_key]
        edge_measurement, edge_candidates = _sweep_edge_pairs(
            gray_f=gray_f,
            support_weight=local_weight,
            center_xy=recentered_xy,
            orientation=orientation,
            params=point_params,
        )
        selected_orientation = dict(edge_measurement.get('orientation') or orientation)
        profiles = list(edge_measurement.get('profiles') or [])
        geometry = evaluate_local_geometry_ambiguity(
            gray_f=gray_f,
            support_refined=local_support,
            center_xy=recentered_xy,
            orientation=orientation,
            profiles=profiles,
            params=point_params,
        )

        measurement = edge_measurement
        measurement_mode = 'edge_pair' if edge_measurement.get('status') == 'ok' else 'rejected'
        if edge_measurement.get('status') != 'ok':
            fallback = measure_diameter_fallback(
                support_weight=local_weight,
                center_xy=recentered_xy,
                orientation=selected_orientation,
                geometry=geometry,
                recenter=recenter_diag,
                params=point_params,
            )
            if fallback.get('status') == 'ok':
                measurement = fallback
                measurement_mode = 'fallback'
            else:
                edge_measurement['fallback'] = fallback
        fiber_size_mode = str(point_params.get('fiber_size_mode') or 'large').strip().lower()
        if fiber_size_mode not in {'auto', 'small', 'large', 'interactive'}:
            fiber_size_mode = 'large'
        mask_diag: dict[str, Any] = {}
        diameter_route = 'large_image'
        auto_reason = ''
        interactive_method = str(point_params.get('interactive_geometry_method') or '').strip()
        if interactive_method == 'circle_square_mask_diameter':
            interactive_measurement, mask_diag = _circle_square_mask_measurement(
                support_mask=interactive_support,
                point_xy=original_xy,
                orientation=selected_orientation,
                params=point_params,
            )
            measurement = interactive_measurement
            measurement_mode = str(interactive_measurement.get('measurement_mode') or 'rejected')
            diameter_route = 'circle_square_mask_diameter'
            fiber_size_mode = 'interactive'
            if interactive_measurement.get('mask_center_xy'):
                recentered_xy = (float(interactive_measurement['mask_center_xy'][0]), float(interactive_measurement['mask_center_xy'][1]))
        elif interactive_method in {'manual_dual_side_caliper', 'manual_line_direct_caliper'}:
            manual_params = point_params
            if interactive_method == 'manual_line_direct_caliper':
                manual_params = {**point_params, 'manual_caliper_refine': False}
            interactive_measurement, mask_diag = _manual_dual_side_measurement(
                support_mask=interactive_support if interactive_method == 'manual_dual_side_caliper' else local_support,
                orientation=selected_orientation,
                params=manual_params,
            )
            measurement = interactive_measurement
            measurement_mode = str(interactive_measurement.get('measurement_mode') or 'rejected')
            diameter_route = interactive_method
            fiber_size_mode = 'interactive'
            if interactive_measurement.get('mask_center_xy'):
                recentered_xy = (float(interactive_measurement['mask_center_xy'][0]), float(interactive_measurement['mask_center_xy'][1]))
        elif interactive_method == 'ellipse_oriented_fit':
            interactive_measurement, mask_diag = _ellipse_oriented_fit_measurement(
                support_mask=local_support,
                point_xy=original_xy,
                orientation=selected_orientation,
                params=point_params,
            )
            measurement = interactive_measurement
            measurement_mode = str(interactive_measurement.get('measurement_mode') or 'rejected')
            diameter_route = 'ellipse_oriented_fit'
            fiber_size_mode = 'interactive'
            if interactive_measurement.get('mask_center_xy'):
                recentered_xy = (float(interactive_measurement['mask_center_xy'][0]), float(interactive_measurement['mask_center_xy'][1]))
        elif interactive_method == 'loco_circle_probe':
            interactive_measurement, mask_diag = _loco_circle_probe_measurement(
                support_mask=local_support,
                point_xy=original_xy,
                point_index=int(point['point_index']),
                orientation=selected_orientation,
                params=point_params,
            )
            measurement = interactive_measurement
            measurement_mode = str(interactive_measurement.get('measurement_mode') or 'rejected')
            diameter_route = 'loco_circle_probe'
            fiber_size_mode = 'interactive'
            if interactive_measurement.get('mask_center_xy'):
                recentered_xy = (float(interactive_measurement['mask_center_xy'][0]), float(interactive_measurement['mask_center_xy'][1]))
        elif _param_bool(point_params, 'mask_driven_enabled', False) and fiber_size_mode in {'auto', 'small'}:
            use_mask_route = fiber_size_mode == 'small'
            if fiber_size_mode == 'auto':
                use_mask_route, auto_reason = _auto_prefers_small_route(
                    measurement=measurement,
                    local_context=local_context,
                    support_meta=dict(roi.get('meta') or {}),
                    params=point_params,
                )
            if use_mask_route:
                mask_measurement, mask_diag = _run_mask_driven_measurement(
                    support_mask=local_support,
                    point_xy=original_xy,
                    orientation=selected_orientation,
                    params=point_params,
                )
                measurement = mask_measurement
                measurement_mode = str(mask_measurement.get('measurement_mode') or 'rejected')
                diameter_route = 'small_mask' if fiber_size_mode == 'small' else 'auto_small_mask'
                if mask_measurement.get('mask_center_xy'):
                    recentered_xy = (float(mask_measurement['mask_center_xy'][0]), float(mask_measurement['mask_center_xy'][1]))
                    recenter_diag = {
                        **dict(recenter_diag or {}),
                        'mask_recenter': dict(mask_diag.get('recenter') or {}),
                        'recenter_shift_px': float(mask_measurement.get('mask_center_shift_px', 0.0) or 0.0),
                        'shift_px': float(mask_measurement.get('mask_center_shift_px', 0.0) or 0.0),
                    }
                    orientation = estimate_local_orientation_from_image(
                        gray_f=gray_f,
                        support=local_support,
                        point_xy=recentered_xy,
                        params=point_params,
                    )
                    selected_orientation = dict(selected_orientation)
                    selected_orientation.update(
                        {
                            'tangent': orientation.get('tangent', selected_orientation.get('tangent', [1.0, 0.0])),
                            'normal': orientation.get('normal', selected_orientation.get('normal', [0.0, 1.0])),
                        }
                    )
            elif fiber_size_mode == 'auto':
                diameter_route = 'auto_large_image'
        elif fiber_size_mode == 'small':
            diameter_route = 'small_requested_but_disabled'

        if measurement.get('mask_center_shift_px') is not None:
            recenter_diag = {
                **dict(recenter_diag or {}),
                'interactive_geometry': dict(mask_diag or {}),
                'recenter_shift_px': float(measurement.get('mask_center_shift_px', 0.0) or 0.0),
                'shift_px': float(measurement.get('mask_center_shift_px', 0.0) or 0.0),
            }

        methodology_diag = methodology_diagnostics(
            methodology_id=methodology_id,
            context=local_context,
            measurement=measurement,
            edge_measurement=edge_measurement,
            orientation=orientation,
            geometry=geometry,
            preprocess=preprocess_diag,
        )
        if methodology_id == METHODOLOGY_CONTOUR_REFINE and measurement_mode == 'fallback':
            measurement_mode = 'contour_refine'

        diameter = None if measurement.get('diameter_px') is None else float(measurement.get('diameter_px'))
        small_suspect = bool(
            _param_bool(point_params, 'small_diameter_flag_enabled', True)
            and diameter is not None
            and diameter <= _param_float(point_params, 'thin_fiber_threshold_px', 8.0)
        )
        multiscale = multiscale_decision(
            diameter_px=diameter,
            edge_status=str(edge_measurement.get('status') or ''),
            support_meta=dict(roi.get('meta') or {}),
            params=point_params,
        )

        flags = list(geometry.get('quality_flags') or [])
        flags.extend(list(methodology_diag.get('flags') or []))
        support_status = str((roi.get('meta') or {}).get('support_status') or '')
        if int((roi.get('meta') or {}).get('refined_pixels', 0)) < 8:
            flags.append('support_weak')
        geom_status = str(geometry.get('geometry_status') or '')
        geometry_guard_enabled = _param_bool(point_params, 'geometry_guard_enabled', True)
        if geometry_guard_enabled and geom_status in {'geometry_ambiguous', 'crossing_likely'}:
            flags.append(geom_status)
        if float(orientation.get('orientation_coherence', 0.0)) < _param_float(point_params, 'min_orientation_coherence', 0.18):
            flags.append('orientation_unstable')
        if small_suspect:
            flags.append('small_diameter_suspect')
        if measurement_mode == 'rejected' or measurement.get('status') != 'ok':
            flags.append(str(measurement.get('reason') or edge_measurement.get('reason') or 'edge_pair_failed'))
        if diameter_route in {'small_mask', 'auto_small_mask', 'circle_square_mask_diameter', 'manual_dual_side_caliper', 'manual_line_direct_caliper', 'ellipse_oriented_fit', 'loco_circle_probe'}:
            flags.append('mask_driven')
        methodology_reject = str(methodology_diag.get('reject_reason') or '')
        if methodology_reject:
            flags.append(methodology_reject)

        confidence = _confidence(
            measurement=measurement,
            geometry=geometry,
            orientation=orientation,
            support_meta=dict(roi.get('meta') or {}),
            measurement_mode=measurement_mode,
            small_diameter_suspect=small_suspect,
        )
        confidence *= float(methodology_diag.get('confidence_multiplier', 1.0) or 1.0)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        if confidence < _param_float(point_params, 'min_point_confidence', 0.45) and measurement_mode not in {'fallback', 'contour_refine'}:
            flags.append('low_confidence')
        reject = (
            measurement_mode == 'rejected'
            or (geometry_guard_enabled and geom_status in {'geometry_ambiguous', 'crossing_likely'})
            or 'support_weak' in flags
            or bool(methodology_reject)
            or ('low_confidence' in flags and measurement_mode not in {'fallback', 'contour_refine'})
        )
        status = 'rejected' if reject else 'ok'
        if status == 'rejected':
            measurement_mode = 'rejected'
            diameter = None
        reason = ','.join(sorted(set(str(f) for f in flags if str(f)))) if status == 'rejected' else ''
        label = _quality_label(status, confidence, sorted(set(flags)), measurement_mode)

        result = {
            'method_id': method_id,
            'interactive_geometry_id': str(
                point_params.get('interactive_geometry_id')
                or point_params.get('manual_geometry_id')
                or point_params.get('circle_square_geometry_id')
                or ''
            ),
            'variant_stage': variant_stage,
            'methodology_id': methodology_id,
            'local_context_label': str(methodology_diag.get('local_context_label') or ''),
            'methodology_reason': str(methodology_diag.get('methodology_reason') or ''),
            'selected_edge_policy': str(methodology_diag.get('selected_edge_policy') or ''),
            'size_route': diameter_route,
            'fiber_size_mode': fiber_size_mode,
            'auto_size_reason': auto_reason,
            'diameter_route': diameter_route,
            'mask_method': str(measurement.get('mask_method') or ''),
            'mask_confidence': None if measurement.get('mask_confidence') is None else float(measurement.get('mask_confidence') or 0.0),
            'mask_center_shift_px': None if measurement.get('mask_center_shift_px') is None else float(measurement.get('mask_center_shift_px') or 0.0),
            'mask_center_distance_px': None if measurement.get('mask_center_distance_px') is None else float(measurement.get('mask_center_distance_px') or 0.0),
            'mask_caliper_diameter_px': None if measurement.get('mask_caliper_diameter_px') is None else float(measurement.get('mask_caliper_diameter_px') or 0.0),
            'mask_raycast_diameter_px': None if measurement.get('mask_raycast_diameter_px') is None else float(measurement.get('mask_raycast_diameter_px') or 0.0),
            'circle_radius_px': None if measurement.get('circle_radius_px') is None else float(measurement.get('circle_radius_px') or 0.0),
            'square_half_length_px': None if measurement.get('square_half_length_px') is None else float(measurement.get('square_half_length_px') or 0.0),
            'square_half_width_px': None if measurement.get('square_half_width_px') is None else float(measurement.get('square_half_width_px') or 0.0),
            'square_vertices_xy': measurement.get('square_vertices_xy') if diameter_route == 'circle_square_mask_diameter' else None,
            'circle_square_manual_locked': bool(measurement.get('circle_square_manual_locked', False)),
            'circle_square_mask_pixels': None if measurement.get('circle_square_mask_pixels') is None else int(measurement.get('circle_square_mask_pixels') or 0),
            'circle_square_measurement_mask_pixels': None if measurement.get('circle_square_measurement_mask_pixels') is None else int(measurement.get('circle_square_measurement_mask_pixels') or 0),
            'circle_square_intersection_count': None if measurement.get('circle_square_intersection_count') is None else int(measurement.get('circle_square_intersection_count') or 0),
            'circle_square_inside_arc_count': None if measurement.get('circle_square_inside_arc_count') is None else int(measurement.get('circle_square_inside_arc_count') or 0),
            'circle_square_display_offset_px': None if measurement.get('circle_square_display_offset_px') is None else float(measurement.get('circle_square_display_offset_px') or 0.0),
            'circle_square_aggregation_samples': None if measurement.get('circle_square_aggregation_samples') is None else int(measurement.get('circle_square_aggregation_samples') or 0),
            'square_samples_valid': None if measurement.get('square_samples_valid') is None else int(measurement.get('square_samples_valid') or 0),
            'square_samples_total': None if measurement.get('square_samples_total') is None else int(measurement.get('square_samples_total') or 0),
            'manual_input_diameter_px': None if measurement.get('manual_input_diameter_px') is None else float(measurement.get('manual_input_diameter_px') or 0.0),
            'ellipse_major_px': None if measurement.get('ellipse_major_px') is None else float(measurement.get('ellipse_major_px') or 0.0),
            'ellipse_minor_px': None if measurement.get('ellipse_minor_px') is None else float(measurement.get('ellipse_minor_px') or 0.0),
            'ellipse_angle_deg': None if measurement.get('ellipse_angle_deg') is None else float(measurement.get('ellipse_angle_deg') or 0.0),
            'loco_best_radius_px': None if measurement.get('loco_best_radius_px') is None else float(measurement.get('loco_best_radius_px') or 0.0),
            'loco_symmetry_score': None if measurement.get('loco_symmetry_score') is None else float(measurement.get('loco_symmetry_score') or 0.0),
            'loco_intersection_count': None if measurement.get('loco_intersection_count') is None else int(measurement.get('loco_intersection_count') or 0),
            'loco_recenter_shift_px': None if measurement.get('mask_center_shift_px') is None else float(measurement.get('mask_center_shift_px') or 0.0),
            'loco_mode': str(measurement.get('loco_mode') or ''),
            'loco_seed_radius_px': None if measurement.get('loco_seed_radius_px') is None else float(measurement.get('loco_seed_radius_px') or 0.0),
            'radius_candidates': measurement.get('radius_candidates') if diameter_route == 'loco_circle_probe' else None,
            'loco_intersections_xy': measurement.get('loco_intersections_xy') if diameter_route == 'loco_circle_probe' else None,
            'halo_status': str(methodology_diag.get('halo_status') or ''),
            'ridge_anchor_status': str(methodology_diag.get('ridge_anchor_status') or ''),
            'flux_status': str(methodology_diag.get('flux_status') or ''),
            'contour_refine_status': str(methodology_diag.get('contour_refine_status') or ''),
            'curvelet_status': str(methodology_diag.get('curvelet_status') or ''),
            'point_index': int(point['point_index']),
            'x': float(recentered_xy[0]),
            'y': float(recentered_xy[1]),
            'original_xy': [float(original_xy[0]), float(original_xy[1])],
            'recentered_xy': [float(recentered_xy[0]), float(recentered_xy[1])],
            'recenter_shift_px': float(recenter_diag.get('recenter_shift_px', recenter_diag.get('shift_px', 0.0))),
            'measurement_mode': measurement_mode,
            'status': status,
            'reason': reason,
            'quality_label': label,
            'diameter_px': diameter,
            'confidence': float(confidence),
            'stability_score': float(measurement.get('profile_consensus', 0.0)),
            'valid_profiles': int(measurement.get('valid_profiles', edge_measurement.get('valid_profiles', 0)) or 0),
            'total_profiles': int(measurement.get('total_profiles', edge_measurement.get('total_profiles', 0)) or 0),
            'mad_px': None if measurement.get('mad_px') is None else float(measurement.get('mad_px')),
            'edge_score_mean': float(measurement.get('edge_score_mean', 0.0)),
            'edge_pair_score': float(measurement.get('edge_pair_score', 0.0)),
            'profile_consensus': float(measurement.get('profile_consensus', 0.0)),
            'geometry_control_status': str(measurement.get('geometry_control_status', 'disabled')),
            'profile_length_effective_px': float(measurement.get('profile_length_effective_px', effective_params.get('profile_length_px', 0.0)) or 0.0),
            'context_width_px': float(measurement.get('context_width_px', 0.0) or 0.0),
            'support_path_mean': float(measurement.get('support_path_mean', 0.0) or 0.0),
            'ridge_response': float(methodology_diag.get('ridge_response', 0.0) or 0.0),
            'edge_pair_center_offset_px': float(methodology_diag.get('edge_pair_center_offset_px', 0.0) or 0.0),
            'axis_flux_score': float(methodology_diag.get('axis_flux_score', 0.0) or 0.0),
            'neighbor_flux_score': float(methodology_diag.get('neighbor_flux_score', 0.0) or 0.0),
            'curvelet_edge_score': float(methodology_diag.get('curvelet_edge_score', 0.0) or 0.0),
            'orientation_delta_deg': float(edge_measurement.get('orientation_delta_deg', 0.0)),
            'used_upscale': bool(multiscale.get('used_upscale', False)),
            'scale_factor': int(multiscale.get('scale_factor', 1)),
            'upscale_method': str(multiscale.get('upscale_method', '')),
            'multiscale_status': str(multiscale.get('multiscale_status', '')),
            'orientation_coherence': float(orientation.get('orientation_coherence', 0.0)),
            'geometry_status': geom_status,
            'support_status': support_status,
            'small_diameter_suspect': bool(small_suspect),
            'left_edge_xy': None if status != 'ok' else measurement.get('left_edge_xy'),
            'right_edge_xy': None if status != 'ok' else measurement.get('right_edge_xy'),
            'debug_left_edge_xy': measurement.get('debug_left_edge_xy') or measurement.get('left_edge_xy'),
            'debug_right_edge_xy': measurement.get('debug_right_edge_xy') or measurement.get('right_edge_xy'),
            'orientation': {
                'source': str(orientation.get('source', 'structure_tensor_image')),
                'tangent': selected_orientation.get('tangent', orientation.get('tangent', [1.0, 0.0])),
                'normal': selected_orientation.get('normal', orientation.get('normal', [0.0, 1.0])),
                'confidence': float(orientation.get('orientation_coherence', 0.0)),
                'orientation_coherence': float(orientation.get('orientation_coherence', 0.0)),
                'status': str(orientation.get('status', '')),
            },
            'profiles': profiles,
            'quality_flags': sorted(set(flags)),
        }
        results.append(json_ready(result))

        diagnostics_points.append(
            {
                'point_index': int(point['point_index']),
                'original_xy': result['original_xy'],
                'support_roi': {
                    **dict(roi.get('meta') or {}),
                    'support_raw_local': np.asarray(roi['support_raw_local'], dtype=np.uint8),
                    'support_refined_local': np.asarray(roi['support_refined_local'], dtype=np.uint8),
                },
                'orientation': orientation,
                'recenter': recenter_diag,
                'geometry': geometry,
                'local_preprocess': preprocess_diag,
                'methodology': methodology_diag,
                'mask_driven': mask_diag,
                'local_context': local_context,
                'point_params_effective': json_ready(point_params),
                'edge_pairs': {
                    'selected': {k: v for k, v in edge_measurement.items() if k not in {'profiles', 'raw', 'aggregate'}},
                    'orientation_candidates': [
                        {k: v for k, v in cand.items() if k not in {'profiles', 'raw', 'aggregate'}}
                        for cand in edge_candidates
                    ],
                    'profiles': profiles,
                    'raw': edge_measurement.get('raw', {}),
                },
                'multiscale': multiscale,
                'result': result,
                'variant_stage': variant_stage,
            }
        )

    interactive_methods = {
        METHOD_ID_CIRCLE_SQUARE,
        METHOD_ID_MANUAL_DUAL_SIDE,
        METHOD_ID_MANUAL_LINE_DIRECT,
        METHOD_ID_ELLIPSE_FIT,
        METHOD_ID_LOCO,
    }
    # Interactive/manual methods should show the same complete mask the user selected.
    # The local ROI union is useful for internal diagnostics, but as a UI overlay it
    # looks like the mask was erased in patches after running points.
    support_for_overlay = interactive_support.astype(np.uint8) if method_id in interactive_methods else (support_refined_union if np.any(support_refined_union) else support.astype(np.uint8))
    overlay = build_overlay(image_rgb=rgb, support=support_for_overlay, results=results, params=effective_params)
    if method_id == METHOD_ID_LOCO:
        overlay = _draw_loco_geometry_overlay(overlay, results)
    meta = {
        'method': method_id,
        'method_id': method_id,
        'experiment_id': method_id,
        'variant_stage': variant_stage,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_mode': str(source_mode or 'prior'),
        'params_effective': json_ready(effective_params),
        'support': json_ready(support_meta),
        'points_requested': int(len(clean_points)),
        'points_ok': int(sum(1 for r in results if r.get('status') == 'ok')),
        'points_rejected': int(sum(1 for r in results if r.get('status') == 'rejected')),
        'image_shape': [int(v) for v in rgb.shape],
    }
    diagnostics_v3 = {
        'method_id': method_id,
        'variant_stage': variant_stage,
        'support_meta': json_ready(support_meta),
        'points': json_ready(diagnostics_points),
    }
    diagnostics_loco = {
        'method_id': method_id,
        'support_meta': json_ready(support_meta),
        'points': [
            {
                'point_index': int(item.get('point_index', -1)),
                'original_xy': item.get('original_xy'),
                'loco': dict(item.get('mask_driven') or {}),
                'result': item.get('result'),
            }
            for item in diagnostics_points
        ],
    } if method_id == METHOD_ID_LOCO else {}
    return {
        'experiment_id': method_id,
        'method_id': method_id,
        'overlay': overlay,
        'support_region': support_for_overlay.astype(np.uint8),
        'support_weight': support_weight.astype(np.float32),
        'results': results,
        'meta': meta,
        'diagnostics': {
            'results': results,
            'support_meta': json_ready(support_meta),
            'params_effective': json_ready(effective_params),
            'diagnostics_v3': diagnostics_v3,
            'diagnostics_loco': diagnostics_loco,
            'loco_overlay': overlay if method_id == METHOD_ID_LOCO else None,
            'profiles_raw_v3': profile_npz_payload(diagnostics_points),
        },
    }


def run_hybrid_profile_diameter_v3_1(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_V3_1,
    )


def run_hybrid_profile_diameter_v3_2(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_V3_2,
    )


def run_hybrid_profile_diameter_v3_2_auto(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_V3_2_AUTO,
    )


def run_hybrid_profile_diameter_v3_2_small_mask(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_V3_2_SMALL_MASK,
    )


def run_hybrid_profile_diameter_v3_2_large_image(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_V3_2_LARGE_IMAGE,
    )


def run_circle_square_mask_diameter(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_CIRCLE_SQUARE,
    )


def run_manual_dual_side_caliper(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_MANUAL_DUAL_SIDE,
    )


def run_manual_line_direct_caliper(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_MANUAL_LINE_DIRECT,
    )


def run_ellipse_oriented_fit(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_ELLIPSE_FIT,
    )


def run_loco_circle_probe(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_LOCO,
    )


def run_hybrid_profile_diameter_v3_3(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_V3_3,
    )


def run_hybrid_profile_diameter_v3_3a(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_V3_3A,
    )


def run_hybrid_profile_diameter_v3_3b(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_V3_3B,
    )


def run_hybrid_profile_diameter_v3_3c(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_V3_3C,
    )


def run_hybrid_profile_diameter_v3_3d(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_V3_3D,
    )


def run_hybrid_profile_diameter_v3_2_small_large(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_V3_2_SMALL_LARGE,
    )


def run_hybrid_profile_diameter_v3_2_halo_aware(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_V3_2_HALO_AWARE,
    )


def run_hybrid_profile_diameter_v3_2_ridge_anchored(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_V3_2_RIDGE_ANCHORED,
    )


def run_hybrid_profile_diameter_v3_2_flux_aware(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_V3_2_FLUX_AWARE,
    )


def run_hybrid_profile_diameter_v3_2_contour_refine(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_V3_2_CONTOUR_REFINE,
    )


def run_hybrid_profile_diameter_v3_2_curvelet_aided(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_V3_2_CURVELET_AIDED,
    )


def run_hybrid_profile_diameter_v3(
    *,
    image_rgb: np.ndarray,
    labels: np.ndarray | None,
    prior_map: np.ndarray | None,
    points: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    source_mode: str = 'prior',
) -> dict[str, Any]:
    return _run_hybrid_profile_diameter_v3_variant(
        image_rgb=image_rgb,
        labels=labels,
        prior_map=prior_map,
        points=points,
        params=params,
        source_mode=source_mode,
        method_id=METHOD_ID_V3,
    )
