from __future__ import annotations

from typing import Any


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


def _scale_factors(params: dict[str, Any]) -> list[int]:
    raw = params.get('upscale_factors', [2, 3])
    if isinstance(raw, str):
        raw = [x.strip() for x in raw.split(',') if x.strip()]
    out: list[int] = []
    for item in list(raw or []):
        try:
            val = int(round(float(item)))
        except Exception:
            continue
        if val > 1 and val not in out:
            out.append(val)
    return out or [2]


def multiscale_decision(
    *,
    diameter_px: float | None,
    edge_status: str,
    support_meta: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    enabled = _param_bool(params, 'multiscale_enabled', True)
    if not enabled:
        return {'used_upscale': False, 'scale_factor': 1, 'upscale_method': '', 'multiscale_status': 'disabled'}
    thin_threshold = max(1.0, _param_float(params, 'thin_fiber_threshold_px', 8.0))
    thin_support = bool(support_meta.get('thin_fiber_support_mode', False))
    small_diam = diameter_px is not None and float(diameter_px) <= thin_threshold
    unstable = str(edge_status or '') != 'ok'
    if thin_support or small_diam or unstable:
        factor = _scale_factors(params)[0]
        return {
            'used_upscale': True,
            'scale_factor': int(factor),
            'upscale_method': str(params.get('upscale_method') or 'bicubic'),
            'multiscale_status': 'triggered_thin_or_unstable',
        }
    return {'used_upscale': False, 'scale_factor': 1, 'upscale_method': '', 'multiscale_status': 'not_needed'}
