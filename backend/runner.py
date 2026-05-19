from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from typing import Any

import numpy as np

from .image_codec import apply_mask_overlay, to_gray_u8, to_uint8_rgb
from .metrics import dice_iou, operational_metrics
from .plugins.base import RunContext
from .registry import ExperimentRegistry
from .scribble import scribble_label_counts


@dataclass
class RunArtifacts:
    run_id: str
    image_id: str
    experiment_id: str
    created_at: str
    input_image: np.ndarray
    scribble_labels: np.ndarray
    prior_map: np.ndarray
    mask: np.ndarray
    overlay: np.ndarray
    meta: dict[str, Any]
    class_prob_maps: dict[str, np.ndarray] | None = None


def now_str() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def compute_image_id(image_rgb: np.ndarray) -> str:
    rgb = to_uint8_rgb(image_rgb)
    if rgb is None:
        raise ValueError('Imagen invalida para hash.')
    h = hashlib.sha1(rgb.tobytes()).hexdigest()[:12]
    return f'img_{h}'


def new_run_id(experiment_id: str) -> str:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    token = hashlib.sha1(f'{experiment_id}_{stamp}'.encode('utf-8')).hexdigest()[:8]
    return f'{experiment_id}_{stamp}_{token}'


def _aux_operational_score(op: dict[str, Any], gt: dict[str, Any] | None = None) -> float:
    if isinstance(gt, dict) and 'iou' in gt:
        try:
            return float(gt.get('iou', 0.0))
        except Exception:
            pass
    try:
        leakage = float(op.get('leakage_to_bg', 0.0))
        frag = float(op.get('fragmentation_index', 0.0))
        compact = float(op.get('compactness', 0.0))
        border = float(op.get('border_touch_ratio', 0.0))
        score = (0.55 * compact) + (0.30 * (1.0 - min(1.0, leakage * 4.0))) + (0.15 * (1.0 - min(1.0, frag / 4.0)))
        score -= 0.08 * border
        return float(max(0.0, min(1.0, score)))
    except Exception:
        return 0.0


def _sanitize_exclude_rect(exclude_rect: dict[str, Any] | None, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if not isinstance(exclude_rect, dict):
        return None
    try:
        x = float(exclude_rect.get('x', 0))
        y = float(exclude_rect.get('y', 0))
        w = float(exclude_rect.get('w', 0))
        h = float(exclude_rect.get('h', 0))
    except Exception:
        return None

    if not np.isfinite([x, y, w, h]).all():
        return None
    if w <= 0 or h <= 0:
        return None

    ih, iw = int(shape[0]), int(shape[1])
    x0 = int(np.floor(max(0.0, min(float(iw), x))))
    y0 = int(np.floor(max(0.0, min(float(ih), y))))
    x1 = int(np.ceil(max(0.0, min(float(iw), x + w))))
    y1 = int(np.ceil(max(0.0, min(float(ih), y + h))))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _build_common_meta(
    *,
    registry: ExperimentRegistry,
    experiment_id: str,
    params: dict[str, Any] | None,
    meta_in: dict[str, Any],
    mask: np.ndarray,
    labels_u8: np.ndarray,
    runtime_ms: float,
    gt_mask: np.ndarray | None,
    exclude_box: tuple[int, int, int, int] | None,
    exclude_nonzero_before: int,
    exclude_nonzero_after: int,
) -> dict[str, Any]:
    exp = registry.get(experiment_id)
    op = operational_metrics(mask, labels_u8, runtime_ms)
    metrics: dict[str, Any] = {'operational': op}
    gt_metrics: dict[str, Any] | None = None
    if isinstance(gt_mask, np.ndarray):
        gt = (np.asarray(gt_mask) > 0).astype(np.uint8)
        d, i = dice_iou(mask, gt)
        gt_metrics = {'dice': float(d), 'iou': float(i)}
        metrics['gt'] = gt_metrics

    meta = dict(meta_in or {})
    if exclude_box is not None:
        x0, y0, x1, y1 = exclude_box
        meta['exclude_rect'] = {
            'x': int(x0),
            'y': int(y0),
            'w': int(x1 - x0),
            'h': int(y1 - y0),
            'area_px': int((x1 - x0) * (y1 - y0)),
            'mask_nonzero_before_px': int(exclude_nonzero_before),
            'mask_nonzero_after_px': int(exclude_nonzero_after),
        }
    else:
        meta['exclude_rect'] = None

    meta['metrics'] = metrics
    meta['scribble_label_schema'] = {
        '0': 'unlabeled',
        '1': 'fiber',
        '2': 'halo',
        '3': 'background',
    }
    meta['scribble_label_counts'] = scribble_label_counts(labels_u8)
    meta['run_status_level'] = str(meta.get('run_status_level', 'success'))
    meta['experiment'] = {
        'experiment_id': exp.info.experiment_id,
        'group': exp.info.group,
        'display_name': exp.info.display_name,
        'implementation_status': exp.info.implementation_status,
        'requirements_hint': exp.info.requirements_hint,
    }
    meta['params_effective'] = {**exp.info.default_params, **dict(params or {})}
    meta['aux_score'] = _aux_operational_score(op, gt_metrics)
    return meta


def run_experiment(
    *,
    registry: ExperimentRegistry,
    experiment_id: str,
    image_rgb: np.ndarray,
    labels: np.ndarray,
    params: dict[str, Any] | None = None,
    gt_mask: np.ndarray | None = None,
    exclude_rect: dict[str, Any] | None = None,
) -> RunArtifacts:
    exp = registry.get(experiment_id)
    rgb = to_uint8_rgb(image_rgb)
    if rgb is None:
        raise ValueError('Imagen invalida.')
    labels_u8 = np.asarray(labels, dtype=np.uint8)
    exclude_box = _sanitize_exclude_rect(exclude_rect, rgb.shape[:2])
    exclude_mask: np.ndarray | None = None
    if exclude_box is not None:
        x0, y0, x1, y1 = exclude_box
        exclude_mask = np.zeros(rgb.shape[:2], dtype=bool)
        exclude_mask[y0:y1, x0:x1] = True
        labels_u8 = labels_u8.copy()
        labels_u8[exclude_mask] = 0

    gray_u8 = to_gray_u8(rgb)
    if gray_u8 is None:
        raise ValueError('No se pudo convertir imagen a gris.')

    ctx = RunContext(
        image_rgb=rgb,
        image_gray_f=(gray_u8.astype(np.float32) / 255.0),
        labels=labels_u8,
        params=dict(params or {}),
    )

    out = exp.run(ctx)
    prior = np.clip(np.asarray(out.prior_map, dtype=np.float32), 0.0, 1.0)
    class_prob_maps: dict[str, np.ndarray] = {}
    for name, pmap in dict(out.prob_maps or {}).items():
        arr = np.clip(np.asarray(pmap, dtype=np.float32), 0.0, 1.0)
        if arr.shape[:2] == rgb.shape[:2]:
            class_prob_maps[str(name)] = arr
    mask = (np.asarray(out.mask) > 0).astype(np.uint8)
    exclude_nonzero_before = 0
    exclude_nonzero_after = 0
    if exclude_mask is not None:
        exclude_nonzero_before = int(np.sum(mask[exclude_mask] > 0))
        prior = prior.copy()
        mask = mask.copy()
        prior[exclude_mask] = 0.0
        mask[exclude_mask] = 0
        for key, arr in list(class_prob_maps.items()):
            clipped = arr.copy()
            clipped[exclude_mask] = 0.0
            class_prob_maps[key] = clipped
        exclude_nonzero_after = int(np.sum(mask[exclude_mask] > 0))
    overlay = apply_mask_overlay(rgb, mask, color=(0, 220, 80), alpha=0.35)
    meta_in = dict(out.meta or {})
    meta_in['run_status_level'] = str(out.status_level)
    meta = _build_common_meta(
        registry=registry,
        experiment_id=experiment_id,
        params=params,
        meta_in=meta_in,
        mask=mask,
        labels_u8=labels_u8,
        runtime_ms=float((out.meta or {}).get('runtime_ms', 0.0)),
        gt_mask=gt_mask,
        exclude_box=exclude_box,
        exclude_nonzero_before=exclude_nonzero_before,
        exclude_nonzero_after=exclude_nonzero_after,
    )

    return RunArtifacts(
        run_id=new_run_id(exp.info.experiment_id),
        image_id=compute_image_id(rgb),
        experiment_id=exp.info.experiment_id,
        created_at=now_str(),
        input_image=rgb,
        scribble_labels=labels_u8,
        prior_map=prior,
        mask=mask,
        overlay=overlay,
        meta=meta,
        class_prob_maps=class_prob_maps or None,
    )

