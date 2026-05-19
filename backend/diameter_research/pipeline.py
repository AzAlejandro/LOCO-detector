from __future__ import annotations

from datetime import datetime
from typing import Any

import cv2
import numpy as np

from ..image_codec import to_gray_u8, to_uint8_rgb
from .orientation import estimate_orientation
from .profiles import aggregate_profiles, measure_profiles
from .support_region import build_support_region


METHOD_ID = 'hybrid_profile_diameter_v1'

DEFAULT_PARAMS: dict[str, Any] = {
    'support_high_threshold': 0.70,
    'support_low_threshold': 0.35,
    'support_dilation_px': 5,
    'local_window_px': 41,
    'profile_length_px': 80,
    'profile_count': 7,
    'profile_spacing_px': 2.0,
    'grad_smooth_sigma': 1.0,
    'edge_min_score': 0.18,
    'min_valid_profiles': 3,
    'max_profile_asymmetry': 0.75,
    'max_mad_scale': 2.5,
    'support_min_inside_ratio': 0.15,
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


def _normalize_params(params: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULT_PARAMS)
    for key, value in dict(params or {}).items():
        if key in out:
            out[key] = value
    return out


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


def _draw_line_rgb(img: np.ndarray, p0: list[float], p1: list[float], color: tuple[int, int, int], thickness: int = 1) -> None:
    a = (int(round(p0[0])), int(round(p0[1])))
    b = (int(round(p1[0])), int(round(p1[1])))
    cv2.line(img, a, b, color, thickness=thickness, lineType=cv2.LINE_AA)


def build_overlay(
    *,
    image_rgb: np.ndarray,
    support: np.ndarray,
    results: list[dict[str, Any]],
    params: dict[str, Any],
) -> np.ndarray:
    rgb = to_uint8_rgb(image_rgb)
    if rgb is None:
        raise ValueError('Imagen invalida para overlay.')
    out = rgb.copy()
    support_mask = np.asarray(support) > 0
    if np.any(support_mask):
        overlay = out.astype(np.float32)
        green = np.zeros_like(overlay)
        green[:, :, 1] = 210.0
        green[:, :, 0] = 15.0
        overlay[support_mask] = overlay[support_mask] * 0.76 + green[support_mask] * 0.24
        out = np.clip(overlay, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(support_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 140, 70), 1, lineType=cv2.LINE_AA)

    for result in results:
        if result.get('status') == 'ok':
            left = result.get('left_edge_xy') or None
            right = result.get('right_edge_xy') or None
            if left and right:
                _draw_line_rgb(out, left, right, (255, 234, 0), thickness=2)
    return out


def _representative_edges(kept_profiles: list[dict[str, Any]]) -> tuple[list[float] | None, list[float] | None]:
    if not kept_profiles:
        return None, None
    left = np.asarray([p.get('left_xy', [np.nan, np.nan]) for p in kept_profiles], dtype=np.float32)
    right = np.asarray([p.get('right_xy', [np.nan, np.nan]) for p in kept_profiles], dtype=np.float32)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return None, None
    return [float(v) for v in np.median(left, axis=0)], [float(v) for v in np.median(right, axis=0)]


def _confidence(agg: dict[str, Any], orientation: dict[str, Any]) -> float:
    if agg.get('status') != 'ok' or not agg.get('diameter_px'):
        return 0.0
    total = max(1, int(agg.get('total_profiles', 0)))
    valid = int(agg.get('valid_profiles', 0))
    valid_ratio = min(1.0, valid / total)
    diam = max(1e-6, float(agg.get('diameter_px', 0.0)))
    mad = float(agg.get('mad_px') or 0.0)
    dispersion = float(np.clip(1.0 - (mad / max(diam * 0.35, 1e-6)), 0.0, 1.0))
    edge = float(np.clip(float(agg.get('edge_score_mean', 0.0)) / 1.4, 0.0, 1.0))
    orient = float(np.clip(float(orientation.get('confidence', 0.0)), 0.0, 1.0))
    return float(np.clip(0.35 * valid_ratio + 0.25 * dispersion + 0.20 * edge + 0.20 * orient, 0.0, 1.0))


def run_hybrid_profile_diameter(
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

    support, support_meta = build_support_region(prior_map=prior_map, labels=labels, shape_hw=shape_hw, params=effective_params)
    if not np.any(support > 0):
        raise ValueError('No se pudo construir soporte desde prior ni scribbles.')

    gray_f = gray_u8.astype(np.float32) / 255.0
    results: list[dict[str, Any]] = []
    for point in clean_points:
        x = float(point['x'])
        y = float(point['y'])
        orient = estimate_orientation(gray_u8=gray_u8, support=support, point_xy=(x, y), params=effective_params)
        profiles = measure_profiles(
            gray_f=gray_f,
            support=support,
            center_xy=(x, y),
            tangent=orient.get('tangent', [1.0, 0.0]),
            normal=orient.get('normal', [0.0, 1.0]),
            params=effective_params,
        )
        agg = aggregate_profiles(profiles, effective_params)
        kept = list(agg.get('kept_profiles') or [])
        left_xy, right_xy = _representative_edges(kept)
        confidence = _confidence(agg, orient)
        result = {
            'point_index': int(point['point_index']),
            'x': float(x),
            'y': float(y),
            'status': str(agg.get('status') or 'failed'),
            'reason': str(agg.get('reason') or ''),
            'diameter_px': None if agg.get('diameter_px') is None else float(agg['diameter_px']),
            'confidence': float(confidence),
            'valid_profiles': int(agg.get('valid_profiles', 0)),
            'total_profiles': int(agg.get('total_profiles', len(profiles))),
            'mad_px': None if agg.get('mad_px') is None else float(agg['mad_px']),
            'edge_score_mean': float(agg.get('edge_score_mean', 0.0)),
            'left_edge_xy': left_xy,
            'right_edge_xy': right_xy,
            'orientation': orient,
            'profiles': profiles,
        }
        results.append(_json_ready(result))

    overlay = build_overlay(image_rgb=rgb, support=support, results=results, params=effective_params)
    meta = {
        'method': METHOD_ID,
        'experiment_id': METHOD_ID,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_mode': str(source_mode or 'prior'),
        'params_effective': _json_ready(effective_params),
        'support': _json_ready(support_meta),
        'points_requested': int(len(clean_points)),
        'points_ok': int(sum(1 for r in results if r.get('status') == 'ok')),
        'image_shape': [int(v) for v in rgb.shape],
    }
    return {
        'experiment_id': METHOD_ID,
        'overlay': overlay,
        'support_region': support.astype(np.uint8),
        'results': results,
        'meta': meta,
        'diagnostics': {
            'results': results,
            'support_meta': _json_ready(support_meta),
            'params_effective': _json_ready(effective_params),
        },
    }
