from __future__ import annotations

import csv
import base64
import hashlib
import json
import secrets
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import cv2
import joblib
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from ..image_codec import encode_display_b64, encode_gray_png_b64
from ..library_store import load_library_image
from ..persistence import list_runs as list_segmentation_runs
from ..persistence import load_run as load_segmentation_run
from ..persistence import load_scribble_draft
from ..runner import compute_image_id
from ..scribble import decode_scribble_b64
from ..session_store import store
from . import persistence as drp
from .pipeline import METHOD_ID, run_hybrid_profile_diameter
from .pipeline_v2 import METHOD_ID_V2, run_hybrid_profile_diameter_v2
from .v3 import (
    METHOD_ID_CIRCLE_SQUARE,
    METHOD_ID_ELLIPSE_FIT,
    METHOD_ID_LOCO,
    METHOD_ID_MANUAL_DUAL_SIDE,
    METHOD_ID_MANUAL_LINE_DIRECT,
    METHOD_ID_V3,
    METHOD_ID_V3_1,
    METHOD_ID_V3_2,
    METHOD_ID_V3_2_AUTO,
    METHOD_ID_V3_2_LARGE_IMAGE,
    METHOD_ID_V3_2_SMALL_MASK,
    METHOD_ID_V3_3,
    METHOD_ID_V3_3A,
    METHOD_ID_V3_3B,
    METHOD_ID_V3_3C,
    METHOD_ID_V3_3D,
    METHOD_ID_V3_2_CONTOUR_REFINE,
    METHOD_ID_V3_2_CURVELET_AIDED,
    METHOD_ID_V3_2_FLUX_AWARE,
    METHOD_ID_V3_2_HALO_AWARE,
    METHOD_ID_V3_2_RIDGE_ANCHORED,
    METHOD_ID_V3_2_SMALL_LARGE,
    V3_METHOD_IDS,
    run_circle_square_mask_diameter,
    run_ellipse_oriented_fit,
    run_loco_circle_probe,
    run_manual_line_direct_caliper,
    run_hybrid_profile_diameter_v3,
    run_hybrid_profile_diameter_v3_1,
    run_hybrid_profile_diameter_v3_2,
    run_hybrid_profile_diameter_v3_2_auto,
    run_hybrid_profile_diameter_v3_2_large_image,
    run_hybrid_profile_diameter_v3_2_small_mask,
    run_hybrid_profile_diameter_v3_2_contour_refine,
    run_hybrid_profile_diameter_v3_2_curvelet_aided,
    run_hybrid_profile_diameter_v3_2_flux_aware,
    run_hybrid_profile_diameter_v3_2_halo_aware,
    run_hybrid_profile_diameter_v3_2_ridge_anchored,
    run_hybrid_profile_diameter_v3_2_small_large,
    run_hybrid_profile_diameter_v3_3,
    run_hybrid_profile_diameter_v3_3a,
    run_hybrid_profile_diameter_v3_3b,
    run_hybrid_profile_diameter_v3_3c,
    run_hybrid_profile_diameter_v3_3d,
    run_manual_dual_side_caliper,
)
from .report import export_diameter_report
from .validation import attach_run, export_validation, list_cases, upsert_case


router = APIRouter(prefix='/api/diameter-research', tags=['diameter-research'])


class PointItem(BaseModel):
    x: float
    y: float


class PointsUpdateReq(BaseModel):
    session_id: str
    action: Literal['add', 'remove_last', 'remove_active', 'set_active', 'clear', 'replace']
    x: float | None = None
    y: float | None = None
    active_index: int | None = None
    points: list[PointItem] = Field(default_factory=list)


class PointsSaveReq(BaseModel):
    session_id: str
    image_id: str


class RunReq(BaseModel):
    session_id: str
    image_id: str
    method_id: Literal[
        'hybrid_profile_diameter_v1',
        'hybrid_profile_diameter_v2',
        'hybrid_profile_diameter_v3',
        'hybrid_profile_diameter_v3_1',
        'hybrid_profile_diameter_v3_2',
        'hybrid_profile_diameter_v3_2_auto',
        'hybrid_profile_diameter_v3_2_small_mask',
        'hybrid_profile_diameter_v3_2_large_image',
        'circle_square_mask_diameter',
        'manual_dual_side_caliper',
        'manual_line_direct_caliper',
        'ellipse_oriented_fit',
        'loco_circle_probe',
        'hybrid_profile_diameter_v3_3',
        'hybrid_profile_diameter_v3_3a',
        'hybrid_profile_diameter_v3_3b',
        'hybrid_profile_diameter_v3_3c',
        'hybrid_profile_diameter_v3_3d',
        'hybrid_profile_diameter_v3_2_small_large',
        'hybrid_profile_diameter_v3_2_halo_aware',
        'hybrid_profile_diameter_v3_2_ridge_anchored',
        'hybrid_profile_diameter_v3_2_flux_aware',
        'hybrid_profile_diameter_v3_2_contour_refine',
        'hybrid_profile_diameter_v3_2_curvelet_aided',
    ] = METHOD_ID
    source_mode: Literal['prior', 'prior_mask', 'scribbles'] = 'prior_mask'
    prior_run_id: str = ''
    points: list[PointItem] = Field(default_factory=list)
    active_only: bool = False
    params: dict[str, Any] = Field(default_factory=dict)
    scribble_map_b64: str = ''


class LocoPreviewReq(BaseModel):
    session_id: str
    image_id: str
    source_mode: Literal['prior', 'prior_mask', 'scribbles'] = 'prior_mask'
    prior_run_id: str = ''
    point: PointItem
    params: dict[str, Any] = Field(default_factory=dict)
    scribble_map_b64: str = ''
    step: int = 0
    candidate_index: int = -1


class LocoLabReq(BaseModel):
    session_id: str
    image_id: str
    source_mode: Literal['prior_mask'] = 'prior_mask'
    prior_run_id: str = ''
    method: Literal['circle_grid', 'fiber_first', 'both'] = 'circle_grid'
    params: dict[str, Any] = Field(default_factory=dict)
    scribble_map_b64: str = ''


class LocoLabFilterReq(BaseModel):
    session_id: str
    image_id: str
    proposals: list[dict[str, Any]] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class LocoLabMeasureReq(BaseModel):
    session_id: str
    image_id: str
    proposals: list[dict[str, Any]] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    prior_run_id: str = ''
    scribble_map_b64: str = ''


class LocoLabEvaluateReq(BaseModel):
    session_id: str
    image_id: str
    proposals: list[dict[str, Any]] = Field(default_factory=list)
    measurements: list[dict[str, Any]] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class LocoLabRunReq(BaseModel):
    session_id: str
    image_id: str
    proposals: list[dict[str, Any]] = Field(default_factory=list)
    measurements: list[dict[str, Any]] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    prior_run_id: str = ''
    scribble_map_b64: str = ''


class LocoDatasetCandidate(BaseModel):
    candidate_id: str
    center_x: float
    center_y: float
    radius_px: float
    label: Literal['valid', 'invalid', 'invalid_crossing', 'invalid_other'] | None = None


class LocoDatasetReq(BaseModel):
    session_id: str
    image_id: str
    source_mode: Literal['prior_mask'] = 'prior_mask'
    prior_run_id: str = ''
    candidates: list[LocoDatasetCandidate] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    scribble_map_b64: str = ''


class LocoDatasetAugmentReq(BaseModel):
    items: list[str] = Field(default_factory=list)
    label_filter: Literal['all', 'valid', 'invalid', 'invalid_crossing', 'invalid_other'] = 'all'
    pipeline: list[dict[str, Any]] = Field(default_factory=list)
    max_items: int = 12
    max_variants_per_source: int = 32
    passes_per_source: int = 4


class LocoTrainingReq(BaseModel):
    data_selection: Literal['original', 'augmented', 'all'] = 'all'
    test_size: float = 0.2
    random_seed: int = 42
    pixel_mode: Literal['square_64', 'circle_only'] = 'square_64'
    circle_prune_px: int = 0
    patch_size: int = 64
    uses_patch_zoom_factor: bool = False
    uses_source_radius_px: bool = False
    models: list[Literal['catboost', 'lightgbm', 'xgboost', 'extratrees']] = Field(default_factory=lambda: ['catboost', 'lightgbm', 'xgboost', 'extratrees'])
    multiclass_model: bool = False


class LocoTestCircleCandidate(BaseModel):
    candidate_id: str
    center_x: float
    center_y: float
    radius_px: float
    label: Literal['valid', 'invalid', 'invalid_crossing', 'invalid_other'] | None = None


class LocoTestCircleReq(BaseModel):
    session_id: str
    image_id: str
    source_mode: Literal['prior_mask'] = 'prior_mask'
    prior_run_id: str = ''
    training_run_id: str = 'latest'
    model_id: Literal['catboost', 'lightgbm', 'xgboost', 'extratrees'] = 'extratrees'
    threshold: float = 0.5
    candidates: list[LocoTestCircleCandidate] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    scribble_map_b64: str = ''


class LocoModelDetectReq(BaseModel):
    session_id: str
    image_id: str
    source_mode: Literal['prior_mask'] = 'prior_mask'
    prior_run_id: str = ''
    model_run_id: str = 'latest'
    model_id: Literal['catboost', 'lightgbm', 'xgboost', 'extratrees'] = 'extratrees'
    patch_size: int = 64
    grid_step: int = 10
    min_radius: float = 8.0
    max_radius: float = 32.0
    radius_step: float = 4.0
    threshold: float = 0.9
    use_radius_thresholds: bool = True
    small_threshold: float = 0.85
    medium_threshold: float = 0.9
    large_threshold: float = 0.95
    small_radius_limit: float = 14.0
    large_radius_limit: float = 24.0
    use_nms: bool = True
    nms_mode: Literal['distance_radius', 'circle_iou'] = 'circle_iou'
    nms_distance_factor: float = 0.5
    radius_similarity_factor: float = 0.4
    circle_iou_threshold: float = 0.4
    candidate_sampling_mode: Literal['row_major', 'random_seeded', 'tile_balanced'] = 'tile_balanced'
    candidate_random_seed: int = 42
    tile_size_px: int = 128
    candidate_max_per_tile: int = 0
    return_rejected: bool = False
    max_candidates: int = 8000
    max_return_rejected: int = 800
    scribble_map_b64: str = ''
    crossing_threshold: float = 0.5
    use_spatial_final_filter: bool = False
    spatial_final_tile_px: int = 128
    spatial_final_max_per_tile: int = 3
    spatial_final_min_center_distance_factor: float = 1.0


class LocoModelMeasureReq(BaseModel):
    session_id: str
    image_id: str
    source_mode: Literal['prior_mask'] = 'prior_mask'
    prior_run_id: str = ''
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    scribble_map_b64: str = ''


class ValidationCaseReq(BaseModel):
    session_id: str
    image_id: str
    case_id: str = ''
    point: PointItem
    category: str = 'otro'
    quality_manual: str = 'medium'
    manual_diameter_px: float | None = None
    manual_left_x: float | None = None
    manual_left_y: float | None = None
    manual_right_x: float | None = None
    manual_right_y: float | None = None
    measurement_decision: str = 'unreviewed'
    notes: str = ''
    result_comment: str = ''
    source_mode: Literal['prior', 'prior_mask', 'scribbles'] = 'prior_mask'
    prior_run_id: str = ''
    params: dict[str, Any] = Field(default_factory=dict)


class ValidationRunCaseReq(BaseModel):
    session_id: str
    image_id: str
    case_id: str
    source_mode: Literal['prior', 'prior_mask', 'scribbles'] = 'prior_mask'
    prior_run_id: str = ''
    params: dict[str, Any] = Field(default_factory=dict)
    scribble_map_b64: str = ''
    methods: list[Literal[
        'hybrid_profile_diameter_v1',
        'hybrid_profile_diameter_v2',
        'hybrid_profile_diameter_v3',
        'hybrid_profile_diameter_v3_1',
        'hybrid_profile_diameter_v3_2',
        'hybrid_profile_diameter_v3_2_auto',
        'hybrid_profile_diameter_v3_2_small_mask',
        'hybrid_profile_diameter_v3_2_large_image',
        'circle_square_mask_diameter',
        'manual_dual_side_caliper',
        'manual_line_direct_caliper',
        'ellipse_oriented_fit',
        'loco_circle_probe',
        'hybrid_profile_diameter_v3_3',
        'hybrid_profile_diameter_v3_3a',
        'hybrid_profile_diameter_v3_3b',
        'hybrid_profile_diameter_v3_3c',
        'hybrid_profile_diameter_v3_3d',
        'hybrid_profile_diameter_v3_2_small_large',
        'hybrid_profile_diameter_v3_2_halo_aware',
        'hybrid_profile_diameter_v3_2_ridge_anchored',
        'hybrid_profile_diameter_v3_2_flux_aware',
        'hybrid_profile_diameter_v3_2_contour_refine',
        'hybrid_profile_diameter_v3_2_curvelet_aided',
    ]] = Field(default_factory=lambda: [
        METHOD_ID,
        METHOD_ID_V2,
        METHOD_ID_V3_1,
        METHOD_ID_V3_2_AUTO,
        METHOD_ID_V3_2_SMALL_MASK,
        METHOD_ID_V3_2_LARGE_IMAGE,
        METHOD_ID_CIRCLE_SQUARE,
        METHOD_ID_MANUAL_DUAL_SIDE,
        METHOD_ID_MANUAL_LINE_DIRECT,
        METHOD_ID_ELLIPSE_FIT,
        METHOD_ID_LOCO,
    ])


def _require_session(session_id: str):
    sess = store.get(session_id)
    if sess is None:
        raise HTTPException(status_code=400, detail='Sesion invalida. Crea una sesion nueva.')
    return sess


def _require_active_image(session_id: str, image_id: str | None = None):
    sess = _require_session(session_id)
    if sess.image_rgb is None:
        raise HTTPException(status_code=400, detail='Carga una imagen antes de usar Diameter Research.')
    active_id = str(sess.image_id or '').strip()
    if image_id is not None and str(image_id or '').strip() != active_id:
        raise HTTPException(status_code=400, detail='image_id no corresponde a la imagen activa de la sesion.')
    return sess


def _clamp_points(points: list[dict[str, Any]], shape_hw: tuple[int, int]) -> list[dict[str, float]]:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    out: list[dict[str, float]] = []
    for idx, item in enumerate(points):
        try:
            x = float(item.get('x'))
            y = float(item.get('y'))
        except Exception:
            continue
        if not np.isfinite([x, y]).all():
            continue
        point_index = int(item.get('point_index', idx))
        out.append(
            {
                'x': float(np.clip(x, 0.0, max(0, w - 1))),
                'y': float(np.clip(y, 0.0, max(0, h - 1))),
                'point_index': int(point_index),
            }
        )
    return out


def _points_overlay(image_rgb: np.ndarray, points: list[dict[str, Any]], active_idx: int) -> str:
    rgb = np.asarray(image_rgb).copy()
    for idx, point in enumerate(points):
        x = int(round(float(point.get('x', 0.0))))
        y = int(round(float(point.get('y', 0.0))))
        cv2.circle(rgb, (x, y), 3, (255, 23, 68), thickness=-1, lineType=cv2.LINE_AA)
    overlay_b64, _mime = encode_display_b64(rgb)
    return overlay_b64


def _points_response(sess: Any, payload: dict[str, Any]) -> dict[str, Any]:
    points = list(payload.get('points') or [])
    active = int(payload.get('active_point_idx', -1))
    overlay_b64 = _points_overlay(sess.image_rgb, points, active) if sess.image_rgb is not None else ''
    return {
        'ok': True,
        'points': points,
        'active_point_idx': active,
        'overlay_b64': overlay_b64,
    }


def _labels_from_request_or_draft(req: RunReq, shape_hw: tuple[int, int]) -> np.ndarray:
    txt = str(req.scribble_map_b64 or '').strip()
    if txt:
        return decode_scribble_b64(txt, target_shape=shape_hw)
    draft = load_scribble_draft(req.image_id)
    if bool(draft.get('found', False)):
        labels = np.asarray(draft.get('labels'), dtype=np.uint8)
        if labels.shape != shape_hw:
            labels = cv2.resize(labels, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
        return labels
    return np.zeros(shape_hw, dtype=np.uint8)


def _prior_from_segmentation_run(
    image_id: str,
    run_id: str,
    shape_hw: tuple[int, int],
    *,
    use_mask: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    rid = str(run_id or '').strip()
    if not rid:
        return None, None, ''
    item = load_segmentation_run(rid)
    if str(item.get('image_id') or '') != str(image_id or ''):
        raise HTTPException(status_code=400, detail='prior_run_id no pertenece a la imagen activa.')
    if use_mask:
        prior = (np.asarray(item.get('mask')) > 0).astype(np.float32)
    else:
        prior = np.asarray(item.get('prior_prob'), dtype=np.float32)
    if prior.shape[:2] != shape_hw:
        prior = cv2.resize(prior, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR)
    labels = np.asarray(item.get('scribble_map'), dtype=np.uint8)
    if labels.shape[:2] != shape_hw:
        labels = cv2.resize(labels, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    class_maps = dict(item.get('class_prob_maps') or {})
    halo_prob = class_maps.get('halo_prob')
    background_prob = class_maps.get('background_prob')
    if isinstance(halo_prob, np.ndarray) and halo_prob.size:
        hp = np.asarray(halo_prob, dtype=np.float32)
        if hp.shape[:2] != shape_hw:
            hp = cv2.resize(hp, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR)
        labels = np.where((labels == 0) & (hp >= 0.72), 2, labels).astype(np.uint8)
    if isinstance(background_prob, np.ndarray) and background_prob.size:
        bp = np.asarray(background_prob, dtype=np.float32)
        if bp.shape[:2] != shape_hw:
            bp = cv2.resize(bp, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR)
        labels = np.where((labels == 0) & (bp >= 0.78), 3, labels).astype(np.uint8)
    return prior, labels, rid


def _latest_prior(
    image_id: str,
    shape_hw: tuple[int, int],
    *,
    use_mask: bool = False,
    prior_run_id: str = '',
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    selected = str(prior_run_id or '').strip()
    if selected and selected != 'latest':
        return _prior_from_segmentation_run(image_id, selected, shape_hw, use_mask=use_mask)
    for row in list_segmentation_runs(image_id):
        run_id = str(row.get('run_id') or '')
        if not run_id:
            continue
        try:
            return _prior_from_segmentation_run(image_id, run_id, shape_hw, use_mask=use_mask)
        except HTTPException:
            raise
        except Exception:
            continue
    return None, None, ''


def _run_payload(item: dict[str, Any]) -> dict[str, Any]:
    overlay_b64, overlay_mime = encode_display_b64(item.get('overlay'))
    input_b64, input_mime = encode_display_b64(item.get('input_image'))
    support = (np.asarray(item.get('support_region')) > 0).astype(np.uint8) * 255
    return {
        'ok': True,
        'run_id': item.get('run_id', ''),
        'image_id': item.get('image_id', ''),
        'experiment_id': item.get('experiment_id', METHOD_ID),
        'method_id': str((item.get('meta') or {}).get('method_id') or item.get('experiment_id') or METHOD_ID),
        'created_at': item.get('created_at', ''),
        'input_image_b64': input_b64,
        'input_image_mime': input_mime,
        'overlay_b64': overlay_b64,
        'overlay_mime': overlay_mime,
        'support_region_b64': encode_gray_png_b64(support),
        'results': list(item.get('results') or []),
        'meta': dict(item.get('meta') or {}),
        'diagnostics': dict(item.get('diagnostics') or {}),
    }


def _loco_lab_context(req: Any) -> tuple[Any, str, np.ndarray, np.ndarray, np.ndarray, str]:
    image_id = str(req.image_id or '').strip()
    sess = _require_session(req.session_id)
    if sess.image_rgb is None and image_id:
        try:
            rgb, meta = load_library_image(image_id)
            sess.image_rgb = rgb
            sess.image_name = str(meta.get('image_name') or 'image')
            sess.image_id = compute_image_id(rgb)
            sess.gt_mask = None
            sess.touch()
        except FileNotFoundError:
            pass
    sess = _require_active_image(req.session_id, image_id)
    image_check = compute_image_id(sess.image_rgb)
    if image_check != image_id:
        raise HTTPException(status_code=400, detail='image_id no corresponde al contenido de imagen activo.')

    shape_hw = sess.image_rgb.shape[:2]
    labels = _labels_from_request_or_draft(req, shape_hw)  # type: ignore[arg-type]
    prior, prior_labels, prior_run_id = _latest_prior(
        image_id,
        shape_hw,
        use_mask=True,
        prior_run_id=str(getattr(req, 'prior_run_id', '') or ''),
    )
    if not np.any(labels == 1) and prior_labels is not None:
        labels = prior_labels
    if prior is None:
        if int(np.sum(labels == 1)) <= 0:
            raise HTTPException(status_code=400, detail='No hay prior_mask ni scribbles de fibra para LOCO Lab.')
        support = (labels == 1).astype(np.uint8)
        prior = support.astype(np.float32)
    else:
        support = (np.asarray(prior) > 0.5).astype(np.uint8)
    if int(np.sum(support)) <= 0:
        raise HTTPException(status_code=400, detail='El prior_mask activo esta vacio.')
    return sess, image_id, labels, np.asarray(prior, dtype=np.float32), support, prior_run_id


def _loco_float_param(params: dict[str, Any], key: str, default: float, *, lo: float | None = None, hi: float | None = None) -> float:
    try:
        val = float(params.get(key, default))
    except Exception:
        val = float(default)
    if lo is not None:
        val = max(float(lo), val)
    if hi is not None:
        val = min(float(hi), val)
    return float(val)


def _loco_int_param(params: dict[str, Any], key: str, default: int, *, lo: int | None = None, hi: int | None = None) -> int:
    val = int(round(_loco_float_param(params, key, float(default), lo=float(lo) if lo is not None else None, hi=float(hi) if hi is not None else None)))
    return int(val)


def _circle_sample_points(center: tuple[float, float], radius: float, samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=False, dtype=np.float32)
    xs = float(center[0]) + float(radius) * np.cos(angles)
    ys = float(center[1]) + float(radius) * np.sin(angles)
    return xs.astype(np.float32), ys.astype(np.float32), angles


def _circle_mask_values(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    xi = np.clip(np.rint(xs).astype(np.int32), 0, max(0, w - 1))
    yi = np.clip(np.rint(ys).astype(np.int32), 0, max(0, h - 1))
    return (mask[yi, xi] > 0).astype(np.uint8)


def _transition_points(xs: np.ndarray, ys: np.ndarray, angles: np.ndarray, vals: np.ndarray) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    n = int(vals.size)
    if n <= 1:
        return points
    prev = np.roll(vals, 1)
    idxs = np.where(vals != prev)[0]
    for idx in idxs:
        j = (int(idx) - 1) % n
        x = float((xs[int(idx)] + xs[j]) * 0.5)
        y = float((ys[int(idx)] + ys[j]) * 0.5)
        a = float(angles[int(idx)])
        points.append({'x': x, 'y': y, 'angle': a})
    points.sort(key=lambda item: float(item.get('angle', 0.0)))
    return points


def _angular_symmetry_score(intersections: list[dict[str, Any]]) -> float:
    if len(intersections) < 4:
        return 0.0
    angles = np.array([float(p.get('angle', 0.0)) for p in intersections], dtype=np.float32)
    best = 0.0
    n = int(angles.size)
    for start in range(n):
        selected = np.array([angles[(start + round(k * n / 4)) % n] for k in range(4)], dtype=np.float32)
        selected.sort()
        gaps = np.diff(np.r_[selected, selected[0] + 2.0 * np.pi])
        err = float(np.mean(np.abs(gaps - (np.pi * 0.5))) / (np.pi * 0.5))
        best = max(best, float(np.clip(1.0 - err, 0.0, 1.0)))
    return float(best)


def _arc_continuity_score(vals: np.ndarray) -> float:
    if vals.size <= 0 or not np.any(vals):
        return 0.0
    doubled = np.r_[vals, vals]
    best = cur = 0
    for val in doubled:
        if int(val) > 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    best = min(best, int(vals.size))
    return float(best / max(1, int(vals.size)))


def _loco_circle_component_metrics(
    support: np.ndarray,
    center: tuple[float, float],
    radius: float,
    xs: np.ndarray,
    ys: np.ndarray,
) -> dict[str, Any]:
    h, w = support.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    r = max(0.5, float(radius))
    pad = int(np.ceil(r)) + 2
    x0 = max(0, int(np.floor(cx)) - pad)
    x1 = min(w, int(np.floor(cx)) + pad + 1)
    y0 = max(0, int(np.floor(cy)) - pad)
    y1 = min(h, int(np.floor(cy)) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return {
            'component_count_inside_circle': 0,
            'boundary_component_count': 0,
            'dominant_component_ratio': 0.0,
            'dominant_boundary_ratio': 0.0,
            'dominant_boundary_arc_count': 0,
            'center_inside_mask': False,
            'center_mask_distance_px': float('inf'),
            'component_bridge_score': 0.0,
        }
    local = (support[y0:y1, x0:x1] > 0).astype(np.uint8)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disk = ((xx.astype(np.float32) - cx) ** 2 + (yy.astype(np.float32) - cy) ** 2) <= (r * r)
    local = (local & disk.astype(np.uint8)).astype(np.uint8)
    if int(np.sum(local)) <= 0:
        return {
            'component_count_inside_circle': 0,
            'boundary_component_count': 0,
            'dominant_component_ratio': 0.0,
            'dominant_boundary_ratio': 0.0,
            'dominant_boundary_arc_count': 0,
            'center_inside_mask': False,
            'center_mask_distance_px': float('inf'),
            'component_bridge_score': 0.0,
        }
    n_labels, labels_cc, stats, _centroids = cv2.connectedComponentsWithStats(local, connectivity=8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, int(n_labels))]
    dominant_label = int(np.argmax(areas) + 1) if areas else 0
    total_area = max(1, int(np.sum(local)))
    dominant_ratio = float(max(areas) / total_area) if areas else 0.0

    sx = np.clip(np.rint(xs).astype(np.int32) - x0, 0, max(0, x1 - x0 - 1))
    sy = np.clip(np.rint(ys).astype(np.int32) - y0, 0, max(0, y1 - y0 - 1))
    boundary_labels = labels_cc[sy, sx]
    positive_boundary = boundary_labels[boundary_labels > 0]
    unique_boundary = np.unique(positive_boundary) if positive_boundary.size else np.array([], dtype=np.int32)
    boundary_component_count = int(unique_boundary.size)
    dominant_boundary_vals = (boundary_labels == dominant_label).astype(np.uint8) if dominant_label > 0 else np.zeros_like(boundary_labels, dtype=np.uint8)
    dominant_boundary_ratio = float(np.sum(dominant_boundary_vals) / max(1, int(np.sum(boundary_labels > 0))))
    if np.any(dominant_boundary_vals):
        transitions = int(np.sum(dominant_boundary_vals != np.roll(dominant_boundary_vals, 1)))
        dominant_arc_count = 1 if transitions == 0 else max(1, transitions // 2)
    else:
        dominant_arc_count = 0

    local_cx = int(np.clip(round(cx) - x0, 0, max(0, x1 - x0 - 1)))
    local_cy = int(np.clip(round(cy) - y0, 0, max(0, y1 - y0 - 1)))
    center_inside = bool(local[local_cy, local_cx] > 0)
    inv = (1 - local).astype(np.uint8)
    dist_to_mask = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    center_dist = float(dist_to_mask[local_cy, local_cx])
    bridge_score = 1.0
    if boundary_component_count != 1:
        bridge_score *= 0.35
    bridge_score *= float(np.clip(dominant_ratio, 0.0, 1.0))
    bridge_score *= float(np.clip(dominant_boundary_ratio, 0.0, 1.0))
    if dominant_arc_count != 2:
        bridge_score *= 0.45
    if center_dist > max(1.5, min(3.0, r * 0.25)):
        bridge_score *= 0.55
    return {
        'component_count_inside_circle': int(n_labels - 1),
        'boundary_component_count': boundary_component_count,
        'dominant_component_ratio': dominant_ratio,
        'dominant_boundary_ratio': dominant_boundary_ratio,
        'dominant_boundary_arc_count': int(dominant_arc_count),
        'center_inside_mask': center_inside,
        'center_mask_distance_px': center_dist,
        'component_bridge_score': float(np.clip(bridge_score, 0.0, 1.0)),
    }


def _score_loco_circle(
    support: np.ndarray,
    *,
    proposal_id: str,
    method: str,
    center: tuple[float, float],
    radius: float,
    params: dict[str, Any],
    component_id: int | None = None,
) -> dict[str, Any]:
    samples = _loco_int_param(params, 'circle_samples', 128, lo=32, hi=512)
    h, w = support.shape[:2]
    x, y = float(center[0]), float(center[1])
    r = float(radius)
    if r < 0.5 or x < 0 or y < 0 or x >= w or y >= h:
        return {
            'proposal_id': proposal_id,
            'method': method,
            'center_xy': [x, y],
            'radius_px': r,
            'score': 0.0,
            'status': 'rejected',
            'reason': 'out_of_bounds',
            'mask_ratio': 0.0,
            'intersection_count': 0,
            'intersection_points': [],
            'symmetry_score': 0.0,
            'continuity_score': 0.0,
            'component_id': component_id,
        }

    xs, ys, angles = _circle_sample_points((x, y), r, samples)
    vals = _circle_mask_values(support, xs, ys)
    ratio = float(np.mean(vals))
    intersections = _transition_points(xs, ys, angles, vals)
    cuts = len(intersections)
    symmetry = _angular_symmetry_score(intersections)
    continuity = _arc_continuity_score(vals)
    component_metrics = _loco_circle_component_metrics(support, (x, y), r, xs, ys)
    bridge_score = float(component_metrics.get('component_bridge_score', 0.0))
    target_ratio = _loco_float_param(params, 'target_mask_ratio', 0.5, lo=0.05, hi=0.95)
    ratio_score = float(np.clip(1.0 - abs(ratio - target_ratio) / max(0.1, target_ratio), 0.0, 1.0))
    cut_score = float(np.clip(1.0 - abs(cuts - 4) / 8.0, 0.0, 1.0))
    excessive_cuts_penalty = float(np.clip((cuts - 8) / 12.0, 0.0, 1.0))
    radius_penalty = float(np.clip((r - _loco_float_param(params, 'radius_max_px', 18.0, lo=1.0)) / max(1.0, r), 0.0, 1.0))
    score = (0.26 * cut_score) + (0.20 * ratio_score) + (0.22 * symmetry) + (0.10 * continuity) + (0.22 * bridge_score)
    score = float(np.clip(score - 0.22 * excessive_cuts_penalty - 0.18 * radius_penalty, 0.0, 1.0))
    min_ratio = _loco_float_param(params, 'mask_required_ratio', 0.1, lo=0.0, hi=1.0)
    min_score = _loco_float_param(params, 'min_score', 0.42, lo=0.0, hi=1.0)
    max_cuts = _loco_int_param(params, 'max_intersections', 12, lo=1, hi=256)
    min_bridge = _loco_float_param(params, 'min_component_bridge_score', 0.12, lo=0.0, hi=1.0)
    require_four = bool(params.get('require_four_cuts', True))
    reasons: list[str] = []
    if ratio < min_ratio:
        reasons.append('low_mask_ratio')
    if require_four and cuts != 4:
        reasons.append('not_four_cuts')
    elif cuts < 4:
        reasons.append('too_few_cuts')
    if cuts > max_cuts:
        reasons.append('too_many_cuts')
    if int(component_metrics.get('boundary_component_count', 0)) > 1:
        reasons.append('multiple_boundary_components')
    if float(component_metrics.get('center_mask_distance_px', 0.0)) > max(1.5, min(3.0, r * 0.25)):
        reasons.append('center_not_on_mask_chain')
    if bridge_score < min_bridge:
        reasons.append('low_component_bridge')
    if score < min_score:
        reasons.append('low_score')
    out = {
        'proposal_id': proposal_id,
        'method': method,
        'center_xy': [x, y],
        'radius_px': r,
        'score': score,
        'status': 'accepted' if not reasons else 'rejected',
        'reason': 'ok' if not reasons else ','.join(reasons),
        'mask_ratio': ratio,
        'intersection_count': cuts,
        'intersection_points': [{'x': float(p['x']), 'y': float(p['y']), 'angle': float(p['angle'])} for p in intersections],
        'symmetry_score': symmetry,
        'continuity_score': continuity,
        'component_id': component_id,
    }
    out.update(component_metrics)
    return out


def _loco_grid_proposals(support: np.ndarray, params: dict[str, Any]) -> list[dict[str, Any]]:
    h, w = support.shape[:2]
    stride = _loco_int_param(params, 'grid_stride_px', 18, lo=3, hi=256)
    rmin = _loco_float_param(params, 'radius_min_px', 3.0, lo=1.0, hi=512.0)
    rmax = _loco_float_param(params, 'radius_max_px', 18.0, lo=rmin, hi=512.0)
    rstep = _loco_float_param(params, 'radius_step_px', 2.0, lo=0.5, hi=max(0.5, rmax))
    max_candidates = _loco_int_param(params, 'max_candidates', 600, lo=1, hi=8000)
    support_u8 = (support > 0).astype(np.uint8)
    proposals: list[dict[str, Any]] = []
    idx = 0
    ys = range(stride // 2, h, stride)
    xs = range(stride // 2, w, stride)
    for y in ys:
        for x in xs:
            rr = int(max(2, round(rmax)))
            y0, y1 = max(0, y - rr), min(h, y + rr + 1)
            x0, x1 = max(0, x - rr), min(w, x + rr + 1)
            tile = support_u8[y0:y1, x0:x1]
            if int(np.sum(tile)) <= 0:
                continue
            yy, xx = np.where(tile > 0)
            center = (float(x0 + np.mean(xx)), float(y0 + np.mean(yy))) if xx.size else (float(x), float(y))
            r = rmin
            while r <= rmax + 1e-6:
                proposals.append(
                    _score_loco_circle(
                        support_u8,
                        proposal_id=f'grid_{idx:06d}',
                        method='circle_grid',
                        center=center,
                        radius=float(r),
                        params=params,
                    )
                )
                idx += 1
                r += rstep
    proposals.sort(key=lambda item: float(item.get('score', 0.0)), reverse=True)
    return proposals[:max_candidates]


def _loco_fiber_first_proposals(support: np.ndarray, params: dict[str, Any]) -> list[dict[str, Any]]:
    support_u8 = (support > 0).astype(np.uint8)
    h, w = support_u8.shape[:2]
    n_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(support_u8, connectivity=8)
    min_area = _loco_int_param(params, 'component_min_area_px', 8, lo=1, hi=h * w)
    max_candidates = _loco_int_param(params, 'max_candidates', 600, lo=1, hi=8000)
    rmin = _loco_float_param(params, 'radius_min_px', 3.0, lo=1.0, hi=512.0)
    rmax = _loco_float_param(params, 'radius_max_px', 18.0, lo=rmin, hi=512.0)
    proposals: list[dict[str, Any]] = []
    idx = 0
    dist = cv2.distanceTransform(support_u8, cv2.DIST_L2, 3)
    for comp_id in range(1, int(n_labels)):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x0 = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y0 = int(stats[comp_id, cv2.CC_STAT_TOP])
        ww = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        hh = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        comp_mask = labels_cc[y0:y0 + hh, x0:x0 + ww] == comp_id
        comp_dist = np.where(comp_mask, dist[y0:y0 + hh, x0:x0 + ww], 0.0)
        _, max_val, _, max_loc = cv2.minMaxLoc(comp_dist.astype(np.float32))
        cx, cy = centroids[comp_id]
        centers = [(float(cx), float(cy))]
        if max_val > 0:
            centers.insert(0, (float(x0 + max_loc[0]), float(y0 + max_loc[1])))
        base_r = float(max(rmin, min(rmax, max_val * 1.4 if max_val > 0 else min(ww, hh) * 0.5)))
        radii = sorted({float(np.clip(base_r * factor, rmin, rmax)) for factor in (0.8, 1.0, 1.25)})
        for center in centers[:2]:
            for radius in radii:
                proposal = _score_loco_circle(
                    support_u8,
                    proposal_id=f'fiber_{idx:06d}',
                    method='fiber_first',
                    center=center,
                    radius=radius,
                    params=params,
                    component_id=comp_id,
                )
                proposal['component_bbox'] = [x0, y0, ww, hh]
                proposal['component_area_px'] = area
                proposals.append(proposal)
                idx += 1
    proposals.sort(key=lambda item: float(item.get('score', 0.0)), reverse=True)
    return proposals[:max_candidates]


def _filter_loco_proposals(proposals: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    min_score = _loco_float_param(params, 'min_score', 0.42, lo=0.0, hi=1.0)
    min_mask_ratio = _loco_float_param(params, 'mask_required_ratio', 0.1, lo=0.0, hi=1.0)
    max_intersections = _loco_int_param(params, 'max_intersections', 12, lo=1, hi=256)
    min_bridge = _loco_float_param(params, 'min_component_bridge_score', 0.12, lo=0.0, hi=1.0)
    require_four = bool(params.get('require_four_cuts', False))
    out: list[dict[str, Any]] = []
    for proposal in proposals:
        item = dict(proposal)
        reasons: list[str] = []
        if float(item.get('score', 0.0)) < min_score:
            reasons.append('low_score')
        if float(item.get('mask_ratio', 0.0)) < min_mask_ratio:
            reasons.append('low_mask_ratio')
        cuts = int(item.get('intersection_count', 0) or 0)
        if cuts > max_intersections:
            reasons.append('too_many_cuts')
        if require_four and cuts != 4:
            reasons.append('not_four_cuts')
        if int(item.get('boundary_component_count', 0) or 0) > 1:
            reasons.append('multiple_boundary_components')
        radius = float(item.get('radius_px') or 0.0)
        center_dist = float(item.get('center_mask_distance_px', 0.0) or 0.0)
        if center_dist > max(1.5, min(3.0, radius * 0.25)):
            reasons.append('center_not_on_mask_chain')
        if float(item.get('component_bridge_score', 0.0) or 0.0) < min_bridge:
            reasons.append('low_component_bridge')
        item['status'] = 'accepted' if not reasons else 'rejected'
        item['reason'] = 'ok' if not reasons else ','.join(reasons)
        out.append(item)
    return out


def _measure_loco_circle_square(support: np.ndarray, proposal: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    item = dict(proposal)
    pid = str(item.get('proposal_id') or '')
    center = item.get('center_xy') or [0.0, 0.0]
    radius = float(item.get('radius_px') or 0.0)
    intersections = list(item.get('intersection_points') or [])
    if len(intersections) < 4:
        rescored = _score_loco_circle(
            support,
            proposal_id=pid,
            method=str(item.get('method') or 'circle_grid'),
            center=(float(center[0]), float(center[1])),
            radius=radius,
            params=params,
            component_id=item.get('component_id'),
        )
        intersections = list(rescored.get('intersection_points') or [])
        item.update(rescored)
    if len(intersections) < 4:
        return {
            'proposal_id': pid,
            'status': 'rejected',
            'quality_label': 'rejected',
            'reason': 'not_enough_circle_mask_intersections',
            'diameter_px': None,
            'circle_xy': [float(center[0]), float(center[1])],
            'radius_px': radius,
            'intersection_points': intersections,
        }
    intersections = sorted(intersections, key=lambda p: float(p.get('angle', 0.0)))
    n = len(intersections)
    idxs = [int(round(k * n / 4.0)) % n for k in range(4)]
    vertices = [
        {'x': float(intersections[i].get('x', 0.0)), 'y': float(intersections[i].get('y', 0.0))}
        for i in idxs
    ]
    pairs = [(vertices[0], vertices[2]), (vertices[1], vertices[3])]
    distances = [
        float(np.hypot(float(a['x']) - float(b['x']), float(a['y']) - float(b['y'])))
        for a, b in pairs
    ]
    if not distances or min(distances) <= 0:
        return {
            'proposal_id': pid,
            'status': 'rejected',
            'quality_label': 'rejected',
            'reason': 'invalid_quadrilateral_distance',
            'diameter_px': None,
            'circle_xy': [float(center[0]), float(center[1])],
            'radius_px': radius,
            'intersection_points': intersections,
            'quadrilateral_vertices': vertices,
        }
    best_pair_idx = int(np.argmin(distances))
    left, right = pairs[best_pair_idx]
    diameter = float(distances[best_pair_idx])
    score = float(item.get('score', 0.0))
    quality = 'high_confidence' if score >= 0.72 else ('medium_confidence' if score >= 0.5 else 'low_confidence')
    return {
        'proposal_id': pid,
        'method_id': 'loco_lab_circle_square',
        'method': str(item.get('method') or ''),
        'status': 'ok',
        'quality_label': quality,
        'reason': 'ok',
        'diameter_px': diameter,
        'confidence': score,
        'score': score,
        'circle_xy': [float(center[0]), float(center[1])],
        'radius_px': radius,
        'intersection_points': intersections,
        'quadrilateral_vertices': vertices,
        'left_edge_xy': [float(left['x']), float(left['y'])],
        'right_edge_xy': [float(right['x']), float(right['y'])],
        'mask_ratio': float(item.get('mask_ratio', 0.0)),
        'intersection_count': int(item.get('intersection_count', len(intersections))),
        'symmetry_score': float(item.get('symmetry_score', 0.0)),
    }


LOCO_DATASET_FEATURE_NAMES = [
    'radio_px',
    'n_cortes',
    'area_mask_ratio',
    'n_componentes_dentro_circulo',
    'distancia_opuesta_1_rel',
    'distancia_opuesta_2_rel',
    'angulo_entre_lados',
    'simetria_cuadrilatero',
    'ancho_estimado_rel',
    'largo_estimado_rel',
    'relacion_largo_ancho',
    'variabilidad_del_ancho',
    'porcentaje_mascara_en_borde_del_circulo',
]

LOCO_TRAINING_EXTRA_FEATURE_NAMES = [
    'continuity_score',
    'boundary_component_count',
    'dominant_component_ratio',
    'dominant_boundary_ratio',
    'dominant_boundary_arc_count',
    'center_inside_mask',
    'center_mask_distance_px',
    'component_bridge_score',
]

LOCO_TRAINING_FEATURE_NAMES = [*LOCO_DATASET_FEATURE_NAMES, *LOCO_TRAINING_EXTRA_FEATURE_NAMES]

LOCO_VECTOR_PIXEL_MODES = {'square_64', 'circle_only'}
LOCO_VECTOR_EXTRA_FEATURE_PATCH_ZOOM = 'patch_zoom_factor'


def _loco_vector_config_from_req(req: Any) -> dict[str, Any]:
    pixel_mode = str(getattr(req, 'pixel_mode', 'square_64') or 'square_64').strip()
    if pixel_mode not in LOCO_VECTOR_PIXEL_MODES:
        pixel_mode = 'square_64'
    patch_size = int(np.clip(int(getattr(req, 'patch_size', 64) or 64), 16, 256))
    prune = int(np.clip(int(getattr(req, 'circle_prune_px', 0) or 0), 0, max(0, patch_size // 2 - 1)))
    return {
        'pixel_mode': pixel_mode,
        'circle_prune_px': prune if pixel_mode == 'circle_only' else 0,
        'patch_size': patch_size,
        'uses_patch_zoom_factor': bool(getattr(req, 'uses_patch_zoom_factor', False)),
        'uses_source_radius_px': bool(getattr(req, 'uses_source_radius_px', False)),
    }


def _loco_vector_config_from_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict((meta or {}).get('vector_config') or {})
    pixel_mode = str(raw.get('pixel_mode') or (meta or {}).get('pixel_mode') or 'square_64')
    if pixel_mode not in LOCO_VECTOR_PIXEL_MODES:
        pixel_mode = 'square_64'
    patch_size = int(np.clip(int(raw.get('patch_size') or (meta or {}).get('patch_size') or 64), 16, 256))
    prune = int(np.clip(int(raw.get('circle_prune_px') or (meta or {}).get('circle_prune_px') or 0), 0, max(0, patch_size // 2 - 1)))
    return {
        'pixel_mode': pixel_mode,
        'circle_prune_px': prune if pixel_mode == 'circle_only' else 0,
        'patch_size': patch_size,
        'uses_patch_zoom_factor': bool(raw.get('uses_patch_zoom_factor') or (meta or {}).get('uses_patch_zoom_factor') or False),
        'uses_source_radius_px': bool(raw.get('uses_source_radius_px') or (meta or {}).get('uses_source_radius_px') or False),
    }


def _loco_circle_pixel_mask(patch_size: int = 64, prune_px: int = 0) -> np.ndarray:
    size = int(np.clip(int(patch_size or 64), 16, 256))
    prune = float(np.clip(float(prune_px or 0), 0.0, max(0.0, size / 2.0 - 1.0)))
    yy, xx = np.indices((size, size), dtype=np.float32)
    center = size / 2.0
    radius = max(0.0, size / 2.0 - prune)
    return ((xx + 0.5 - center) ** 2 + (yy + 0.5 - center) ** 2) <= (radius ** 2)


def _loco_pixel_feature_count(config: dict[str, Any]) -> int:
    patch_size = int(config.get('patch_size') or 64)
    if str(config.get('pixel_mode') or 'square_64') == 'circle_only':
        return int(np.sum(_loco_circle_pixel_mask(patch_size, int(config.get('circle_prune_px') or 0))))
    return int(patch_size * patch_size)


def _loco_tabular_feature_names(config: dict[str, Any]) -> list[str]:
    names = list(LOCO_TRAINING_FEATURE_NAMES)
    if bool(config.get('uses_patch_zoom_factor')):
        names.append(LOCO_VECTOR_EXTRA_FEATURE_PATCH_ZOOM)
    return names


def _loco_vector_feature_order(config: dict[str, Any]) -> list[str]:
    return [f'mask_pixel_{idx}' for idx in range(_loco_pixel_feature_count(config))] + _loco_tabular_feature_names(config)


def _loco_source_radius_from_item(item: dict[str, Any] | None, features: dict[str, Any]) -> float:
    item = item or {}
    for key in ('source_radius_px', 'radius_for_group', 'radius_px'):
        val = item.get(key)
        parsed = _float_or_nan(val)
        if np.isfinite(parsed) and parsed > 0:
            return float(parsed)
    parsed = _float_or_nan(features.get('source_radius_px'))
    if np.isfinite(parsed) and parsed > 0:
        return float(parsed)
    parsed = _float_or_nan(features.get('radio_px'))
    if np.isfinite(parsed) and parsed > 0:
        return float(parsed)
    return float('nan')


def _candidate_to_dict(c: LocoDatasetCandidate) -> dict[str, Any]:
    return {
        'candidate_id': str(c.candidate_id or '').strip(),
        'center_x': float(c.center_x),
        'center_y': float(c.center_y),
        'radius_px': float(c.radius_px),
        'label': str(c.label or '').strip(),
    }


def _circle_disk_patch(support: np.ndarray, center: tuple[float, float], radius: float, *, patch_size: int = 64) -> tuple[np.ndarray, float]:
    h, w = support.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    r = max(1.0, float(radius))
    side = max(3, int(np.ceil(2.0 * r)))
    x0 = int(np.floor(cx - r))
    y0 = int(np.floor(cy - r))
    src = np.zeros((side, side), dtype=np.uint8)
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(w, x0 + side)
    sy1 = min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        dx0 = sx0 - x0
        dy0 = sy0 - y0
        src[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = (support[sy0:sy1, sx0:sx1] > 0).astype(np.uint8)
    yy, xx = np.indices(src.shape, dtype=np.float32)
    local_cx = cx - x0
    local_cy = cy - y0
    disk = ((xx + 0.5 - local_cx) ** 2 + (yy + 0.5 - local_cy) ** 2) <= (r ** 2)
    src = np.where(disk, src, 0).astype(np.uint8)
    disk_count = max(1, int(np.sum(disk)))
    area_ratio = float(np.sum(src > 0) / disk_count)
    resized = cv2.resize(src * 255, (int(patch_size), int(patch_size)), interpolation=cv2.INTER_NEAREST)
    resized = (resized > 127).astype(np.uint8) * 255
    return resized, area_ratio


def _write_gray_png(path: Any, arr: np.ndarray) -> None:
    ok, buf = cv2.imencode('.png', np.asarray(arr, dtype=np.uint8))
    if not ok:
        raise RuntimeError(f'No se pudo codificar PNG: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())


def _write_rgb_png(path: Any, arr: np.ndarray) -> None:
    rgb = np.asarray(arr, dtype=np.uint8)
    if rgb.ndim == 2:
        _write_gray_png(path, rgb)
        return
    bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.png', bgr)
    if not ok:
        raise RuntimeError(f'No se pudo codificar PNG: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())


def _loco_dataset_features_for_candidate(
    support: np.ndarray,
    cand: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(cand.get('candidate_id') or '').strip()
    cx = float(cand.get('center_x') or 0.0)
    cy = float(cand.get('center_y') or 0.0)
    radius = max(0.0, float(cand.get('radius_px') or 0.0))
    score = _score_loco_circle(
        support,
        proposal_id=candidate_id,
        method='manual_loco_dataset',
        center=(cx, cy),
        radius=radius,
        params={**params, 'radius_max_px': max(radius, float(params.get('radius_max_px', radius) or radius))},
    )
    _patch, area_ratio = _circle_disk_patch(
        support,
        (cx, cy),
        radius,
        patch_size=_loco_int_param(params, 'patch_size', 64, lo=16, hi=256),
    )
    intersections = sorted(list(score.get('intersection_points') or []), key=lambda p: float(p.get('angle', 0.0)))
    distances: list[float] = []
    angle_between: float | None = None
    if len(intersections) >= 4 and radius > 0:
        n = len(intersections)
        idxs = [int(round(k * n / 4.0)) % n for k in range(4)]
        vertices = [
            {'x': float(intersections[i].get('x', 0.0)), 'y': float(intersections[i].get('y', 0.0))}
            for i in idxs
        ]
        vectors: list[np.ndarray] = []
        for a, b in [(vertices[0], vertices[2]), (vertices[1], vertices[3])]:
            vec = np.array([float(b['x']) - float(a['x']), float(b['y']) - float(a['y'])], dtype=np.float32)
            dist = float(np.linalg.norm(vec))
            if dist > 1e-6:
                distances.append(dist)
                vectors.append(vec / dist)
        if len(vectors) >= 2:
            dot = float(np.clip(abs(float(np.dot(vectors[0], vectors[1]))), 0.0, 1.0))
            angle_between = float(np.degrees(np.arccos(dot)))
    distances_sorted = sorted(distances)
    width = distances_sorted[0] if distances_sorted else None
    length = distances_sorted[-1] if distances_sorted else None
    dist_mean = float(np.mean(distances)) if distances else None
    features = {
        'radio_px': radius,
        'n_cortes': int(score.get('intersection_count') or 0),
        'area_mask_ratio': area_ratio,
        'n_componentes_dentro_circulo': int(score.get('component_count_inside_circle') or 0),
        'distancia_opuesta_1_rel': float(distances_sorted[0] / radius) if len(distances_sorted) >= 1 and radius > 0 else None,
        'distancia_opuesta_2_rel': float(distances_sorted[1] / radius) if len(distances_sorted) >= 2 and radius > 0 else None,
        'angulo_entre_lados': angle_between,
        'simetria_cuadrilatero': float(score.get('symmetry_score') or 0.0),
        'ancho_estimado_rel': float(width / radius) if width is not None and radius > 0 else None,
        'largo_estimado_rel': float(length / radius) if length is not None and radius > 0 else None,
        'relacion_largo_ancho': float(length / width) if width and length and width > 1e-6 else None,
        'variabilidad_del_ancho': float(np.std(distances) / dist_mean) if distances and dist_mean and dist_mean > 1e-6 else None,
        'porcentaje_mascara_en_borde_del_circulo': float(score.get('mask_ratio') or 0.0),
    }
    return {
        'candidate_id': candidate_id,
        'center_x': cx,
        'center_y': cy,
        'radius_px': radius,
        'label': str(cand.get('label') or ''),
        'features': features,
        'diagnostics': score,
    }


def _read_loco_dataset_metadata(csv_path: Any, fieldnames: list[str]) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open('r', newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        rows = []
        for row in reader:
            rows.append({k: row.get(k, '') for k in fieldnames})
        return rows


def _read_csv_flexible(csv_path: Any) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open('r', newline='', encoding='utf-8') as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float('nan')
    text = str(value).strip()
    if text == '' or text.lower() in {'nan', 'none', 'null'}:
        return float('nan')
    try:
        return float(text)
    except Exception:
        return float('nan')


def _bool_feature(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in {'true', '1', 'yes', 'si'}:
        return 1.0
    if text in {'false', '0', 'no'}:
        return 0.0
    return float('nan')


def _loco_dataset_root() -> Any:
    return drp.OUTPUT_ROOT / 'datasets' / 'loco_circle_dataset' / 'main'


def _safe_loco_rel_path(rel: str) -> str:
    parts = [p for p in str(rel or '').replace('\\', '/').split('/') if p and p not in {'.', '..'}]
    return '/'.join(parts)


def _read_gray_png(path: Any) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail=f'No se pudo leer PNG: {path}')
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST) if img.shape[:2] != (64, 64) else img
    return (img > 127).astype(np.uint8) * 255


def _gray_png_b64(arr: np.ndarray) -> str:
    ok, buf = cv2.imencode('.png', (np.asarray(arr) > 0).astype(np.uint8) * 255)
    if not ok:
        raise RuntimeError('No se pudo codificar PNG.')
    return base64.b64encode(buf.tobytes()).decode('ascii')


def _loco_augmented_root(root: Any | None = None) -> Any:
    base = root if root is not None else _loco_dataset_root()
    return base / 'augmented'


def _load_loco_dataset_rows() -> tuple[Any, list[dict[str, Any]]]:
    root = _loco_dataset_root()
    csv_path = root / 'metadata.csv'
    fieldnames = ['dataset_id', 'image_id', 'candidate_id', 'center_x', 'center_y', 'radius_px', 'label_text', 'label_numeric', 'mask_patch_path', *LOCO_DATASET_FEATURE_NAMES]
    return root, _read_loco_dataset_metadata(csv_path, fieldnames)


def _loco_dataset_items() -> dict[str, Any]:
    root, rows = _load_loco_dataset_rows()
    items: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get('label_text') or '').strip()
        label_norm = _loco_normalize_label(label)
        rel = _safe_loco_rel_path(str(row.get('mask_patch_path') or ''))
        if label_norm not in {'valid', 'invalid_crossing', 'invalid_other'} or not rel:
            continue
        # Try the normalized path first, fall back to legacy path
        path = root / rel
        if not path.exists() and label != label_norm:
            alt_rel = rel.replace(f'{label}/', f'{label_norm}/')
            path = root / alt_rel
        if not path.exists():
            continue
        candidate_id = str(row.get('candidate_id') or '').strip()
        image_id = str(row.get('image_id') or '').strip()
        item_id = f'{image_id}::{candidate_id}'
        source_features = {name: row.get(name, '') for name in LOCO_DATASET_FEATURE_NAMES}
        try:
            source_b64 = _gray_png_b64(_read_gray_png(path))
        except Exception:
            source_b64 = ''
        items.append(
            {
                'item_id': item_id,
                'image_id': image_id,
                'candidate_id': candidate_id,
                'label': label_norm,
                'radius_px': row.get('radius_px') or '',
                'source_path': rel,
                'source_b64': source_b64,
                'source_features': source_features,
            }
        )
    aug_root = _loco_augmented_root(root)
    valid_aug = len(list((aug_root / 'valid').glob('*.png'))) if (aug_root / 'valid').exists() else 0
    crossing_aug = len(list((aug_root / 'invalid_crossing').glob('*.png'))) if (aug_root / 'invalid_crossing').exists() else 0
    other_aug = len(list((aug_root / 'invalid_other').glob('*.png'))) if (aug_root / 'invalid_other').exists() else 0
    counts = {
        'total': len(items),
        'valid': sum(1 for item in items if item['label'] == 'valid'),
        'invalid_crossing': sum(1 for item in items if item['label'] == 'invalid_crossing'),
        'invalid_other': sum(1 for item in items if item['label'] == 'invalid_other'),
        'augmented_valid': valid_aug,
        'augmented_crossing': crossing_aug,
        'augmented_other': other_aug,
        'augmented_total': valid_aug + crossing_aug + other_aug,
    }
    return {'root': root, 'items': items, 'counts': counts}


def _loco_parse_csv_list(value: Any, fallback: list[Any]) -> list[Any]:
    if isinstance(value, list):
        return [x for x in value if str(x).strip()]
    text = str(value or '').strip()
    if not text:
        return fallback
    return [x.strip() for x in text.split(',') if x.strip()]


def _loco_block_probability(block: dict[str, Any]) -> float:
    params = dict(block.get('params') or {})
    raw = params.get('probability', params.get('p', 1.0))
    try:
        return float(np.clip(float(raw), 0.0, 1.0))
    except Exception:
        return 1.0


def _loco_pipeline_hash(pipeline: list[dict[str, Any]]) -> str:
    raw = json.dumps(pipeline, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]


def _circle_clip64(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.uint8)
    yy, xx = np.indices((64, 64), dtype=np.float32)
    disk = ((xx + 0.5 - 32.0) ** 2 + (yy + 0.5 - 32.0) ** 2) <= (32.0 ** 2)
    return np.where(disk, out, 0).astype(np.uint8)


def _threshold64(arr: np.ndarray) -> np.ndarray:
    out = cv2.resize(np.asarray(arr, dtype=np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST) if arr.shape[:2] != (64, 64) else np.asarray(arr, dtype=np.uint8)
    out = (out > 127).astype(np.uint8) * 255
    return _circle_clip64(out)


def _rotate_binary(arr: np.ndarray, angle: float) -> np.ndarray:
    angle = float(angle) % 360.0
    if abs(angle) < 1e-6:
        return _threshold64(arr)
    if abs(angle - 90.0) < 1e-6:
        return np.rot90(_threshold64(arr), 1).copy()
    if abs(angle - 180.0) < 1e-6:
        return np.rot90(_threshold64(arr), 2).copy()
    if abs(angle - 270.0) < 1e-6:
        return np.rot90(_threshold64(arr), 3).copy()
    mat = cv2.getRotationMatrix2D((31.5, 31.5), angle, 1.0)
    rot = cv2.warpAffine(_threshold64(arr), mat, (64, 64), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return _threshold64(rot)


def _perturb_binary(arr: np.ndarray, seed_text: str, amount: float = 0.015) -> np.ndarray:
    src = (_threshold64(arr) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    edge = (cv2.dilate(src, kernel, iterations=1) - cv2.erode(src, kernel, iterations=1)) > 0
    near = cv2.dilate(src, kernel, iterations=1) > 0
    rng = np.random.default_rng(int(hashlib.sha1(seed_text.encode('utf-8')).hexdigest()[:8], 16))
    out = src.copy()
    edge_coords = np.argwhere(edge & (src > 0))
    add_coords = np.argwhere(near & (src == 0))
    remove_n = min(len(edge_coords), max(1, int(round(float(amount) * max(1, int(np.sum(src)))))))
    add_n = min(len(add_coords), max(1, int(round(float(amount) * max(1, int(np.sum(src)))))))
    if len(edge_coords) and remove_n > 0:
        picks = edge_coords[rng.choice(len(edge_coords), size=remove_n, replace=False)]
        out[picks[:, 0], picks[:, 1]] = 0
    if len(add_coords) and add_n > 0:
        picks = add_coords[rng.choice(len(add_coords), size=add_n, replace=False)]
        out[picks[:, 0], picks[:, 1]] = 1
    return _threshold64(out.astype(np.uint8) * 255)


def _perturb_binary_rng(arr: np.ndarray, rng: np.random.Generator, amount: float = 0.015) -> np.ndarray:
    src = (_threshold64(arr) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    edge = (cv2.dilate(src, kernel, iterations=1) - cv2.erode(src, kernel, iterations=1)) > 0
    near = cv2.dilate(src, kernel, iterations=1) > 0
    out = src.copy()
    edge_coords = np.argwhere(edge & (src > 0))
    add_coords = np.argwhere(near & (src == 0))
    remove_n = min(len(edge_coords), max(1, int(round(float(amount) * max(1, int(np.sum(src)))))))
    add_n = min(len(add_coords), max(1, int(round(float(amount) * max(1, int(np.sum(src)))))))
    if len(edge_coords) and remove_n > 0:
        picks = edge_coords[rng.choice(len(edge_coords), size=remove_n, replace=False)]
        out[picks[:, 0], picks[:, 1]] = 0
    if len(add_coords) and add_n > 0:
        picks = add_coords[rng.choice(len(add_coords), size=add_n, replace=False)]
        out[picks[:, 0], picks[:, 1]] = 1
    return _threshold64(out.astype(np.uint8) * 255)


def _apply_loco_aug_block_random(img: np.ndarray, block: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, str]:
    btype = str(block.get('type') or '').strip()
    params = dict(block.get('params') or {})
    src = _threshold64(img)
    kernel = np.ones((3, 3), dtype=np.uint8)
    if btype == 'rotate':
        raw_angles = []
        for raw in _loco_parse_csv_list(params.get('angles'), [90, 180, 270]):
            try:
                raw_angles.append(float(raw) % 360.0)
            except Exception:
                continue
        angles = raw_angles or [90.0, 180.0, 270.0]
        angle = float(rng.choice(np.array(angles, dtype=np.float32)))
        return _rotate_binary(src, angle), f'rotate:{angle:g}'
    if btype == 'flip':
        modes = [str(x).strip() for x in _loco_parse_csv_list(params.get('modes'), ['h', 'v', 'hv']) if str(x).strip() in {'h', 'v', 'hv'}]
        mode = str(rng.choice(np.array(modes or ['h', 'v', 'hv'], dtype=object)))
        if mode == 'h':
            aug = cv2.flip(src, 1)
        elif mode == 'v':
            aug = cv2.flip(src, 0)
        else:
            aug = cv2.flip(src, -1)
        return _threshold64(aug), f'flip:{mode}'
    if btype == 'morphology':
        ops = [str(x).strip() for x in _loco_parse_csv_list(params.get('ops'), ['erode1', 'dilate1', 'open1', 'close1']) if str(x).strip() in {'erode1', 'dilate1', 'open1', 'close1'}]
        op = str(rng.choice(np.array(ops or ['erode1', 'dilate1', 'open1', 'close1'], dtype=object)))
        binary = (src > 0).astype(np.uint8) * 255
        if op == 'erode1':
            aug = cv2.erode(binary, kernel, iterations=1)
        elif op == 'dilate1':
            aug = cv2.dilate(binary, kernel, iterations=1)
        elif op == 'open1':
            aug = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        else:
            aug = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return _threshold64(aug), f'morph:{op}'
    if btype == 'perturb':
        lo = float(params.get('amount_min', 0.005) or 0.005)
        hi = float(params.get('amount_max', params.get('amount', 0.02)) or 0.02)
        lo, hi = sorted((float(np.clip(lo, 0.0, 0.08)), float(np.clip(hi, 0.0, 0.08))))
        amount = float(rng.uniform(lo, hi if hi > lo else lo + 1e-6))
        return _perturb_binary_rng(src, rng, amount=amount), f'perturb:{amount:.4f}'
    if btype == 'resize_method':
        methods = [str(x).strip() for x in _loco_parse_csv_list(params.get('methods'), ['nearest', 'bilinear_threshold', 'area_threshold']) if str(x).strip() in {'nearest', 'bilinear_threshold', 'area_threshold'}]
        method = str(rng.choice(np.array(methods or ['nearest', 'bilinear_threshold', 'area_threshold'], dtype=object)))
        size_min = int(np.clip(int(params.get('target_size_min', 40) or 40), 24, 63))
        size_max = int(np.clip(int(params.get('target_size_max', params.get('target_size', 56)) or 56), 24, 63))
        size_min, size_max = sorted((size_min, size_max))
        mid = int(rng.integers(size_min, size_max + 1))
        if method == 'nearest':
            small = cv2.resize(src, (mid, mid), interpolation=cv2.INTER_NEAREST)
            aug = cv2.resize(small, (64, 64), interpolation=cv2.INTER_NEAREST)
        elif method == 'bilinear_threshold':
            small = cv2.resize(src, (mid, mid), interpolation=cv2.INTER_LINEAR)
            aug = cv2.resize(small, (64, 64), interpolation=cv2.INTER_LINEAR)
        else:
            small = cv2.resize(src, (mid, mid), interpolation=cv2.INTER_AREA)
            aug = cv2.resize(small, (64, 64), interpolation=cv2.INTER_AREA)
        return _threshold64(aug), f'resize:{method}:{mid}'
    if btype == 'resolution':
        sizes = []
        for raw in _loco_parse_csv_list(params.get('sizes'), [48, 40]):
            try:
                sizes.append(int(np.clip(int(raw), 24, 63)))
            except Exception:
                continue
        size = int(rng.choice(np.array(sizes or [48, 40], dtype=np.int32)))
        small = cv2.resize(src, (size, size), interpolation=cv2.INTER_AREA)
        aug = cv2.resize(small, (64, 64), interpolation=cv2.INTER_NEAREST)
        return _threshold64(aug), f'resolution:{size}'
    return src, f'skip_unknown:{btype or "none"}'


def _apply_loco_aug_block(variants: list[dict[str, Any]], block: dict[str, Any], source_key: str) -> list[dict[str, Any]]:
    btype = str(block.get('type') or '').strip()
    params = dict(block.get('params') or {})
    out: list[dict[str, Any]] = []
    kernel = np.ones((3, 3), dtype=np.uint8)
    for variant in variants:
        img = _threshold64(variant['image'])
        chain = list(variant.get('chain') or [])
        if btype == 'rotate':
            values = []
            for raw in _loco_parse_csv_list(params.get('angles'), [90, 180, 270]):
                try:
                    angle = float(raw)
                except Exception:
                    continue
                values.append(angle % 360.0)
            for angle in values:
                out.append({'image': _rotate_binary(img, angle), 'chain': [*chain, f'rotate:{angle:g}']})
        elif btype == 'flip':
            modes = [str(x).strip() for x in _loco_parse_csv_list(params.get('modes'), ['h', 'v', 'hv'])]
            for mode in modes:
                if mode == 'h':
                    aug = cv2.flip(img, 1)
                elif mode == 'v':
                    aug = cv2.flip(img, 0)
                elif mode == 'hv':
                    aug = cv2.flip(img, -1)
                else:
                    continue
                out.append({'image': _threshold64(aug), 'chain': [*chain, f'flip:{mode}']})
        elif btype == 'morphology':
            ops = [str(x).strip() for x in _loco_parse_csv_list(params.get('ops'), ['erode1', 'dilate1', 'open1', 'close1'])]
            for op in ops:
                binary = (img > 0).astype(np.uint8) * 255
                if op == 'erode1':
                    aug = cv2.erode(binary, kernel, iterations=1)
                elif op == 'dilate1':
                    aug = cv2.dilate(binary, kernel, iterations=1)
                elif op == 'open1':
                    aug = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
                elif op == 'close1':
                    aug = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
                else:
                    continue
                out.append({'image': _threshold64(aug), 'chain': [*chain, f'morph:{op}']})
        elif btype == 'perturb':
            amount = float(params.get('amount', 0.015) or 0.015)
            aug = _perturb_binary(img, f'{source_key}|{"|".join(chain)}|perturb|{amount}', amount=amount)
            out.append({'image': aug, 'chain': [*chain, f'perturb:{amount:g}']})
        elif btype == 'resize_method':
            methods = [str(x).strip() for x in _loco_parse_csv_list(params.get('methods'), ['nearest', 'bilinear_threshold', 'area_threshold'])]
            mid = int(params.get('target_size', 48) or 48)
            mid = int(np.clip(mid, 24, 63))
            for method in methods:
                if method == 'nearest':
                    small = cv2.resize(img, (mid, mid), interpolation=cv2.INTER_NEAREST)
                    aug = cv2.resize(small, (64, 64), interpolation=cv2.INTER_NEAREST)
                elif method == 'bilinear_threshold':
                    small = cv2.resize(img, (mid, mid), interpolation=cv2.INTER_LINEAR)
                    aug = cv2.resize(small, (64, 64), interpolation=cv2.INTER_LINEAR)
                elif method == 'area_threshold':
                    small = cv2.resize(img, (mid, mid), interpolation=cv2.INTER_AREA)
                    aug = cv2.resize(small, (64, 64), interpolation=cv2.INTER_AREA)
                else:
                    continue
                out.append({'image': _threshold64(aug), 'chain': [*chain, f'resize:{method}:{mid}']})
        elif btype == 'resolution':
            sizes = []
            for raw in _loco_parse_csv_list(params.get('sizes'), [48, 40]):
                try:
                    size = int(raw)
                except Exception:
                    continue
                sizes.append(int(np.clip(size, 24, 63)))
            for size in sizes:
                small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                aug = cv2.resize(small, (64, 64), interpolation=cv2.INTER_NEAREST)
                out.append({'image': _threshold64(aug), 'chain': [*chain, f'resolution:{size}']})
        else:
            out.append({'image': img, 'chain': chain})
    return out


def _generate_loco_aug_variants(source_img: np.ndarray, pipeline: list[dict[str, Any]], source_key: str, max_variants: int) -> list[dict[str, Any]]:
    variants = [{'image': _threshold64(source_img), 'chain': ['source']}]
    for block in pipeline:
        variants = _apply_loco_aug_block(variants, block, source_key)
        if len(variants) > max_variants:
            variants = variants[:max_variants]
    return variants[:max(1, max_variants)]


def _generate_loco_aug_random_variants(
    source_img: np.ndarray,
    pipeline: list[dict[str, Any]],
    source_key: str,
    *,
    passes: int,
    max_variants: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    source = _threshold64(source_img)
    n_passes = int(np.clip(int(passes or 1), 1, max(1, max_variants)))
    for pass_idx in range(n_passes):
        img = source.copy()
        chain = ['source', f'pass:{pass_idx + 1}']
        applied = 0
        for block in pipeline:
            probability = _loco_block_probability(block)
            if probability <= 0.0 or float(rng.random()) > probability:
                chain.append(f'skip:{str(block.get("type") or "block")}')
                continue
            img, desc = _apply_loco_aug_block_random(img, block, rng)
            chain.append(desc)
            applied += 1
        if applied <= 0:
            chain.append('identity')
        out.append({'image': _threshold64(img), 'chain': chain, 'source_key': source_key})
    return out[:max(1, max_variants)]


def _loco_augmented_features_for_patch(patch: np.ndarray, candidate_id: str, label: str) -> dict[str, Any]:
    support = (_threshold64(patch) > 0).astype(np.uint8)
    cand = {
        'candidate_id': candidate_id,
        'center_x': 31.5,
        'center_y': 31.5,
        'radius_px': 31.5,
        'label': label,
    }
    item = _loco_dataset_features_for_candidate(
        support,
        cand,
        {
            'patch_size': 64,
            'circle_samples': 128,
            'require_four_cuts': False,
            'radius_max_px': 31.5,
            'mask_required_ratio': 0.0,
            'min_score': 0.0,
            'max_intersections': 256,
        },
    )
    return item


def _loco_feature_json_path(root: Any, label: str, image_id: str, candidate_id: str) -> Any:
    base_name = f'{image_id}_{candidate_id}'
    safe_name = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in base_name)
    return root / 'features' / label / f'{safe_name}.json'


def _loco_normalize_label(label: str) -> str:
    """Normalize legacy 'invalid' to 'invalid_other' for backward compatibility."""
    lt = str(label).strip().lower()
    if lt == 'invalid':
        return 'invalid_other'
    return lt


def _loco_label_maps(label_text: str) -> tuple[int, int]:
    """Return (label_binary, label_multiclass) for a given label_text."""
    lt = _loco_normalize_label(label_text)
    if lt == 'valid':
        return (1, 0)
    elif lt == 'invalid_crossing':
        return (0, 1)
    else:  # invalid_other
        return (0, 2)


def _loco_training_read_original(root: Any) -> list[dict[str, Any]]:
    rows = _read_csv_flexible(root / 'metadata.csv')
    items: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get('label_text') or '').strip()
        label_norm = _loco_normalize_label(label)
        if label_norm not in {'valid', 'invalid_crossing', 'invalid_other'}:
            continue
        # For legacy rows with label_text='invalid', try reading from 'invalid/' dir first,
        # then fall back to 'invalid_other/'
        rel = _safe_loco_rel_path(str(row.get('mask_patch_path') or ''))
        path = root / rel
        if not rel or not path.exists():
            # Legacy fallback: try the normalized path
            if label != label_norm:
                alt_rel = rel.replace(f'{label}/', f'{label_norm}/')
                alt_path = root / alt_rel
                if alt_path.exists():
                    rel = alt_rel
                    path = alt_path
                else:
                    continue
            else:
                continue
        image_id = str(row.get('image_id') or '')
        candidate_id = str(row.get('candidate_id') or '')
        diagnostics: dict[str, Any] = {}
        feature_json = _loco_feature_json_path(root, label_norm, image_id, candidate_id)
        if not feature_json.exists() and label != label_norm:
            # Legacy fallback for feature json
            feature_json = _loco_feature_json_path(root, label, image_id, candidate_id)
        if feature_json.exists():
            try:
                payload = json.loads(feature_json.read_text(encoding='utf-8'))
                diagnostics = dict(payload.get('diagnostics') or {})
            except Exception:
                diagnostics = {}
        label_binary, label_multiclass = _loco_label_maps(label_norm)
        items.append({
            'dataset_kind': 'original',
            'item_id': f'original::{image_id}::{candidate_id}',
            'group_id': f'{image_id}::{candidate_id}',
            'image_id': image_id,
            'candidate_id': candidate_id,
            'label_text': label_norm,
            'label_binary': label_binary,
            'label_multiclass': label_multiclass,
            'label_numeric': label_binary,  # keep for backward compat
            'patch_path': path,
            'patch_rel': rel,
            'radius_for_group': _float_or_nan(row.get('radius_px')),
            'source_radius_px': _float_or_nan(row.get('radius_px')),
            'features': {name: row.get(name, '') for name in LOCO_DATASET_FEATURE_NAMES},
            'diagnostics': diagnostics,
        })
    return items


def _loco_training_read_augmented(root: Any) -> list[dict[str, Any]]:
    aug_root = _loco_augmented_root(root)
    rows = _read_csv_flexible(aug_root / 'augmented_metadata.csv')
    items: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get('source_label') or '').strip()
        label_norm = _loco_normalize_label(label)
        if label_norm not in {'valid', 'invalid_crossing', 'invalid_other'}:
            continue
        rel = _safe_loco_rel_path(str(row.get('augmented_path') or ''))
        path = aug_root / rel
        if not rel or not path.exists():
            continue
        image_id = str(row.get('source_image_id') or '')
        candidate_id = str(row.get('source_candidate_id') or '')
        diagnostics: dict[str, Any] = {}
        feature_rel = _safe_loco_rel_path(str(row.get('feature_path') or ''))
        if feature_rel and (aug_root / feature_rel).exists():
            try:
                payload = json.loads((aug_root / feature_rel).read_text(encoding='utf-8'))
                diagnostics = dict(payload.get('augmented_diagnostics') or {})
            except Exception:
                diagnostics = {}
        label_binary, label_multiclass = _loco_label_maps(label_norm)
        items.append({
            'dataset_kind': 'augmented',
            'item_id': f'augmented::{rel}',
            'group_id': f'{image_id}::{candidate_id}',
            'image_id': image_id,
            'candidate_id': candidate_id,
            'label_text': label_norm,
            'label_binary': label_binary,
            'label_multiclass': label_multiclass,
            'label_numeric': label_binary,  # keep for backward compat
            'patch_path': path,
            'patch_rel': f'augmented/{rel}',
            'radius_for_group': _float_or_nan(row.get('source_radius_px')),
            'source_radius_px': _float_or_nan(row.get('source_radius_px')),
            'features': {name: row.get(name, '') for name in LOCO_DATASET_FEATURE_NAMES},
            'diagnostics': diagnostics,
        })
    return items


def _loco_training_items(selection: str) -> list[dict[str, Any]]:
    root = _loco_dataset_root()
    original = _loco_training_read_original(root) if selection in {'original', 'all'} else []
    augmented = _loco_training_read_augmented(root) if selection in {'augmented', 'all'} else []
    return [*original, *augmented]


def _loco_training_vector(item: dict[str, Any], vector_config: dict[str, Any]) -> np.ndarray:
    patch = (_read_gray_png(item['patch_path']) > 0).astype(np.float32)
    return _loco_vector_from_patch_features(patch, item.get('features') or {}, item.get('diagnostics') or {}, item=item, vector_config=vector_config)


def _loco_patch_pixels(patch_flat_or_img: np.ndarray, vector_config: dict[str, Any]) -> np.ndarray:
    config = dict(vector_config or {})
    patch_size = int(config.get('patch_size') or 64)
    arr = np.asarray(patch_flat_or_img, dtype=np.float32)
    if arr.ndim == 2:
        patch = arr
    else:
        flat = arr.reshape(-1)
        if flat.size == patch_size * patch_size:
            patch = flat.reshape((patch_size, patch_size))
        else:
            side = int(round(float(np.sqrt(max(1, flat.size)))))
            patch = flat.reshape((side, side)) if side * side == flat.size else flat.reshape((1, -1))
    if patch.shape[:2] != (patch_size, patch_size):
        patch = cv2.resize(patch.astype(np.float32), (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
    patch = (patch > 0).astype(np.float32)
    if str(config.get('pixel_mode') or 'square_64') == 'circle_only':
        mask = _loco_circle_pixel_mask(patch_size, int(config.get('circle_prune_px') or 0))
        return patch[mask].astype(np.float32).reshape(-1)
    return patch.reshape(-1).astype(np.float32)


def _loco_vector_from_patch_features(
    patch_flat_or_img: np.ndarray,
    features: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    item: dict[str, Any] | None = None,
    vector_config: dict[str, Any] | None = None,
) -> np.ndarray:
    config = _loco_vector_config_from_meta({'vector_config': vector_config or {}})
    patch = _loco_patch_pixels(patch_flat_or_img, config)
    features = dict(features or {})
    diagnostics = dict(diagnostics or {})
    source_radius = _loco_source_radius_from_item(item, features)
    values: list[float] = []
    for name in LOCO_DATASET_FEATURE_NAMES:
        raw_value = features.get(name)
        if name == 'radio_px' and bool(config.get('uses_source_radius_px')) and np.isfinite(source_radius):
            raw_value = source_radius
        values.append(_float_or_nan(raw_value))
    for name in LOCO_TRAINING_EXTRA_FEATURE_NAMES:
        if name == 'center_inside_mask':
            values.append(_bool_feature(diagnostics.get(name)))
        else:
            values.append(_float_or_nan(diagnostics.get(name)))
    if bool(config.get('uses_patch_zoom_factor')):
        values.append(float(32.0 / source_radius) if np.isfinite(source_radius) and source_radius > 0 else float('nan'))
    vec = np.concatenate([patch, np.asarray(values, dtype=np.float32)], axis=0)
    vec[np.isinf(vec)] = np.nan
    return vec


def _loco_group_split(items: list[dict[str, Any]], test_size: float, random_seed: int, *, multiclass: bool = False) -> tuple[list[int], list[int]]:
    group_to_indices: dict[str, list[int]] = {}
    group_to_label: dict[str, int] = {}
    for idx, item in enumerate(items):
        gid = str(item.get('group_id') or item.get('item_id') or idx)
        group_to_indices.setdefault(gid, []).append(idx)
        if multiclass:
            group_to_label.setdefault(gid, int(item.get('label_multiclass') or 0))
        else:
            group_to_label.setdefault(gid, int(item.get('label_numeric') or 0))
    groups = sorted(group_to_indices)
    labels = np.asarray([group_to_label[g] for g in groups], dtype=np.int32)
    n_classes = len(set(labels.tolist()))
    if len(groups) < 4 or n_classes < 2:
        raise HTTPException(status_code=400, detail='Dataset insuficiente para split estratificado.')
    bincount = np.bincount(labels, minlength=n_classes)
    if min(bincount[:n_classes]) < 2:
        raise HTTPException(status_code=400, detail='Dataset insuficiente para split estratificado (clase minoritaria < 2 grupos).')
    train_groups, test_groups = train_test_split(
        groups,
        test_size=float(np.clip(test_size, 0.05, 0.5)),
        random_state=int(random_seed),
        stratify=labels,
    )
    train_idx = [idx for gid in train_groups for idx in group_to_indices[gid]]
    test_idx = [idx for gid in test_groups for idx in group_to_indices[gid]]
    return train_idx, test_idx


def _loco_model_instance(model_id: str, random_seed: int, *, multiclass: bool = False) -> Any:
    if model_id == 'extratrees':
        return ExtraTreesClassifier(n_estimators=350, random_state=random_seed, class_weight='balanced', n_jobs=-1)
    if model_id == 'catboost':
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:
            raise RuntimeError(f'catboost_unavailable: {exc}') from exc
        loss = 'MultiClass' if multiclass else 'Logloss'
        return CatBoostClassifier(iterations=350, depth=6, learning_rate=0.05, loss_function=loss, random_seed=random_seed, verbose=False, allow_writing_files=False)
    if model_id == 'xgboost':
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise RuntimeError(f'xgboost_unavailable: {exc}') from exc
        obj = 'multi:softprob' if multiclass else 'binary:logistic'
        ev = 'mlogloss' if multiclass else 'logloss'
        return XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, objective=obj, eval_metric=ev, random_state=random_seed, n_jobs=-1)
    if model_id == 'lightgbm':
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:
            raise RuntimeError(f'lightgbm_unavailable: {exc}') from exc
        return LGBMClassifier(n_estimators=350, learning_rate=0.05, num_leaves=31, random_state=random_seed, class_weight='balanced', n_jobs=-1, verbose=-1)
    raise RuntimeError(f'model_unknown: {model_id}')


def _positive_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], average='binary', pos_label=1, zero_division=0)
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    return {
        'precision_valid': float(precision),
        'recall_valid': float(recall),
        'f1_valid': float(f1),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
    }


def _patch_b64_from_path(path: Any) -> str:
    return _gray_png_b64(_read_gray_png(path))


def _training_runs_root() -> Any:
    return drp.OUTPUT_ROOT / 'training_runs'


def _list_loco_training_runs() -> list[dict[str, Any]]:
    root = _training_runs_root()
    if not root.exists():
        return []
    items: list[dict[str, Any]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        meta_path = path / 'run_meta.json'
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except Exception:
            meta = {}
        models_dir = path / 'models'
        models = []
        multiclass_models = []
        if models_dir.exists():
            for model_file in models_dir.glob('*_model.pkl'):
                mid = model_file.name.replace('_model.pkl', '')
                # Check if this is a multiclass model (suffixed with _multiclass)
                if mid.endswith('_multiclass'):
                    multiclass_models.append(mid.replace('_multiclass', ''))
                else:
                    models.append(mid)
        items.append({
            'run_id': path.name,
            'created_at': str(meta.get('created_at') or ''),
            'data_selection': str(meta.get('data_selection') or ''),
            'sample_count': int(meta.get('sample_count') or 0),
            'train_count': int(meta.get('train_count') or 0),
            'test_count': int(meta.get('test_count') or 0),
            'feature_count': int(meta.get('feature_count') or 0),
            'pixel_mode': str((meta.get('vector_config') or {}).get('pixel_mode') or meta.get('pixel_mode') or 'square_64'),
            'circle_prune_px': int((meta.get('vector_config') or {}).get('circle_prune_px') or meta.get('circle_prune_px') or 0),
            'uses_patch_zoom_factor': bool((meta.get('vector_config') or {}).get('uses_patch_zoom_factor') or meta.get('uses_patch_zoom_factor') or False),
            'models': sorted(models),
            'multiclass_models': sorted(multiclass_models),
            'has_multiclass': bool(meta.get('has_multiclass') or False) or bool(multiclass_models),
            'run_dir': str(path),
            '_mtime': path.stat().st_mtime,
        })
    items.sort(key=lambda x: float(x.get('_mtime') or 0.0), reverse=True)
    for item in items:
        item.pop('_mtime', None)
    return items


def _resolve_training_run_id(training_run_id: str) -> str:
    rid = str(training_run_id or '').strip()
    if rid and rid != 'latest':
        return rid
    runs = _list_loco_training_runs()
    if not runs:
        raise HTTPException(status_code=400, detail='No hay training runs disponibles.')
    return str(runs[0]['run_id'])


def _load_training_model(training_run_id: str, model_id: str, *, model_kind: str = 'binary') -> tuple[str, Any, dict[str, Any]]:
    rid = _resolve_training_run_id(training_run_id)
    run_root = _training_runs_root() / rid
    suffix = '_multiclass_model.pkl' if model_kind == 'multiclass' else '_model.pkl'
    model_path = run_root / 'models' / f'{model_id}{suffix}'
    if not model_path.exists():
        if model_kind == 'multiclass':
            raise HTTPException(status_code=400, detail=f'Modelo multiclase no encontrado: {rid}/{model_id}')
        raise HTTPException(status_code=400, detail=f'Modelo no encontrado: {rid}/{model_id}')
    meta_path = run_root / 'run_meta.json'
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    except Exception:
        meta = {}
    return rid, joblib.load(model_path), meta


def _loco_model_radius_group(radius: float, small_limit: float, large_limit: float) -> str:
    r = float(radius)
    if r < float(small_limit):
        return 'small'
    if r < float(large_limit):
        return 'medium'
    return 'large'


def _loco_model_threshold(req: LocoModelDetectReq, radius: float) -> tuple[float, str]:
    if not bool(req.use_radius_thresholds):
        return float(np.clip(float(req.threshold), 0.01, 0.99)), 'all'
    group = _loco_model_radius_group(radius, req.small_radius_limit, req.large_radius_limit)
    if group == 'small':
        th = req.small_threshold
    elif group == 'medium':
        th = req.medium_threshold
    else:
        th = req.large_threshold
    return float(np.clip(float(th), 0.01, 0.99)), group


def _loco_model_predict_valid_scores(model: Any, vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.asarray([], dtype=np.float32)
    X = np.vstack(vectors).astype(np.float32)
    expected = getattr(model, 'n_features_in_', None)
    if expected is not None and int(expected) != int(X.shape[1]):
        raise HTTPException(
            status_code=400,
            detail=f'El modelo espera {int(expected)} columnas, pero la inferencia genero {int(X.shape[1])}. Reentrena con el dataset actual.',
        )
    if hasattr(model, 'predict_proba'):
        probs = np.asarray(model.predict_proba(X), dtype=np.float32)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1].astype(np.float32)
        return probs.reshape(-1).astype(np.float32)
    return np.asarray(model.predict(X), dtype=np.float32).reshape(-1)


def _loco_candidate_specs(
    support_u8: np.ndarray,
    radii: list[float],
    *,
    step: int,
    max_candidates: int,
    mode: str,
    seed: int,
    tile_size_px: int,
    max_per_tile: int = 0,
) -> tuple[list[tuple[float, float, float]], int]:
    h, w = support_u8.shape[:2]
    specs: list[tuple[float, float, float]] = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            if support_u8[y, x] <= 0:
                continue
            for radius in radii:
                specs.append((float(x), float(y), float(radius)))
    total = len(specs)
    cap = int(np.clip(int(max_candidates or 8000), 1, 500000))
    if total <= cap:
        return specs, total
    mode = str(mode or 'row_major')
    rng = np.random.default_rng(int(seed))
    if mode == 'random_seeded':
        order = rng.permutation(total)[:cap]
        return [specs[int(idx)] for idx in order], total
    if mode == 'tile_balanced':
        tile_size = int(np.clip(int(tile_size_px or 128), 32, 2048))
        buckets: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
        for spec in specs:
            x, y, _radius = spec
            buckets.setdefault((int(x) // tile_size, int(y) // tile_size), []).append(spec)
        keys = list(buckets)
        rng.shuffle(keys)
        for key in keys:
            rng.shuffle(buckets[key])
        # Apply per-tile cap if set (>0)
        if max_per_tile > 0:
            for key in keys:
                bucket = buckets[key]
                if len(bucket) > max_per_tile:
                    buckets[key] = bucket[:max_per_tile]
        selected: list[tuple[float, float, float]] = []
        cursors = {key: 0 for key in keys}
        while len(selected) < cap and keys:
            active: list[tuple[int, int]] = []
            for key in keys:
                idx = cursors[key]
                bucket = buckets[key]
                if idx < len(bucket):
                    selected.append(bucket[idx])
                    cursors[key] = idx + 1
                    active.append(key)
                    if len(selected) >= cap:
                        break
            keys = active
        return selected, total
    return specs[:cap], total


def _circle_intersection_area(r1: float, r2: float, d: float) -> float:
    r1 = max(0.0, float(r1))
    r2 = max(0.0, float(r2))
    d = max(0.0, float(d))
    if r1 <= 0.0 or r2 <= 0.0:
        return 0.0
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return float(np.pi * min(r1, r2) ** 2)
    a1 = r1 ** 2 * np.arccos(np.clip((d ** 2 + r1 ** 2 - r2 ** 2) / (2.0 * d * r1), -1.0, 1.0))
    a2 = r2 ** 2 * np.arccos(np.clip((d ** 2 + r2 ** 2 - r1 ** 2) / (2.0 * d * r2), -1.0, 1.0))
    a3 = 0.5 * np.sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
    return float(a1 + a2 - a3)


def _circle_iou_xy(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax = float(a.get('center_x') or 0.0)
    ay = float(a.get('center_y') or 0.0)
    ar = max(1e-6, float(a.get('radius_px') or 0.0))
    bx = float(b.get('center_x') or 0.0)
    by = float(b.get('center_y') or 0.0)
    br = max(1e-6, float(b.get('radius_px') or 0.0))
    d = float(np.hypot(ax - bx, ay - by))
    inter = _circle_intersection_area(ar, br, d)
    union = float(np.pi * ar ** 2 + np.pi * br ** 2 - inter)
    return float(inter / union) if union > 0 else 0.0


def _loco_circle_nms(
    accepted: list[dict[str, Any]],
    *,
    nms_mode: str = 'distance_radius',
    nms_distance_factor: float,
    radius_similarity_factor: float,
    circle_iou_threshold: float = 0.4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not accepted:
        return [], []
    n = len(accepted)
    # Pre-compute centers and radii for fast access
    cx = np.asarray([float(c.get('center_x') or 0.0) for c in accepted], dtype=np.float64)
    cy = np.asarray([float(c.get('center_y') or 0.0) for c in accepted], dtype=np.float64)
    cr = np.asarray([max(1e-6, float(c.get('radius_px') or 0.0)) for c in accepted], dtype=np.float64)
    scores = np.asarray([float(c.get('valid_score') or 0.0) for c in accepted], dtype=np.float64)
    order = np.argsort(-scores)  # descending by score
    is_circle_iou = str(nms_mode or 'distance_radius') == 'circle_iou'
    iou_th = float(np.clip(float(circle_iou_threshold), 0.0, 1.0))
    dist_factor = float(nms_distance_factor)
    rad_factor = float(radius_similarity_factor)

    # Pre-compute pairwise overlap matrix using vectorized operations
    # overlap[i, j] = True if circle j should be removed when circle i is kept (i has higher score)
    # We only compute for i < j in score order
    overlap: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        idx_i = order[i]
        xi, yi, ri = cx[idx_i], cy[idx_i], cr[idx_i]
        for j in range(i + 1, n):
            idx_j = order[j]
            xj, yj, rj = cx[idx_j], cy[idx_j], cr[idx_j]
            d = float(np.hypot(xi - xj, yi - yj))
            min_r = min(ri, rj)
            max_r = max(ri, rj)
            # Containment check
            if (d + min_r) <= max_r + 1e-9:
                overlap[idx_i].add(idx_j)
                continue
            if is_circle_iou:
                iou_val = _circle_iou_xy(accepted[idx_i], accepted[idx_j])
                if iou_val >= iou_th:
                    overlap[idx_i].add(idx_j)
                elif d < dist_factor * min_r and abs(ri - rj) < rad_factor * min_r:
                    overlap[idx_i].add(idx_j)
            else:
                if d < dist_factor * min_r and abs(ri - rj) < rad_factor * min_r:
                    overlap[idx_i].add(idx_j)

    # Greedy NMS using pre-computed overlap matrix
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    active = set(range(n))
    for i in range(n):
        idx_i = order[i]
        if idx_i not in active:
            continue
        kept.append(dict(accepted[idx_i]))
        # Remove all circles that overlap with this kept circle
        to_remove = overlap[idx_i] & active
        for idx_j in to_remove:
            dropped = dict(accepted[idx_j])
            dropped['status'] = 'removed_by_nms'
            dropped['reason'] = 'overlap_with_higher_score'
            dropped['nms_kept_candidate_id'] = str(accepted[idx_i].get('candidate_id') or '')
            removed.append(dropped)
            active.discard(idx_j)
        active.discard(idx_i)
    return kept, removed


def _loco_spatial_final_filter(
    accepted: list[dict[str, Any]],
    image_h: int,
    image_w: int,
    *,
    tile_px: int = 128,
    max_per_tile: int = 3,
    min_center_distance_factor: float = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply spatial filtering to accepted circles after NMS.

    Divides the image into tiles of `tile_px` size, groups accepted circles
    by tile, sorts by valid_score descending within each tile, and keeps only
    `max_per_tile` circles per tile. Optionally enforces a minimum center
    distance between kept circles within the same tile.

    Returns:
        (filtered_accepted, removed_by_spatial, tile_stats)
        - filtered_accepted: circles that passed the spatial filter
        - removed_by_spatial: circles removed by the spatial filter
        - tile_stats: dict with per-tile summary information
    """
    if not accepted:
        return [], [], {'tiles': {}, 'total_before': 0, 'total_after': 0, 'removed': 0}

    tile_px = max(16, int(tile_px))
    max_per_tile = max(1, int(max_per_tile))
    min_center_distance_factor = max(0.0, float(min_center_distance_factor))

    # Group circles by tile
    tiles: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for cand in accepted:
        cx = float(cand.get('center_x', 0))
        cy = float(cand.get('center_y', 0))
        tx = int(cx // tile_px)
        ty = int(cy // tile_px)
        key = (tx, ty)
        if key not in tiles:
            tiles[key] = []
        tiles[key].append(cand)

    filtered: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    tile_stats: dict[str, Any] = {}
    total_before = len(accepted)

    for (tx, ty), candidates in tiles.items():
        # Sort by valid_score descending
        sorted_cands = sorted(
            candidates,
            key=lambda c: float(c.get('valid_score', 0.0)),
            reverse=True,
        )

        kept: list[dict[str, Any]] = []
        rejected_in_tile: list[dict[str, Any]] = []

        for cand in sorted_cands:
            if len(kept) >= max_per_tile:
                rejected_in_tile.append(cand)
                continue

            # Enforce minimum center distance from already-kept circles
            if min_center_distance_factor > 0 and kept:
                cx = float(cand.get('center_x', 0))
                cy = float(cand.get('center_y', 0))
                r = float(cand.get('radius_px', 1))
                too_close = False
                for k in kept:
                    kx = float(k.get('center_x', 0))
                    ky = float(k.get('center_y', 0))
                    kr = float(k.get('radius_px', 1))
                    avg_r = (r + kr) / 2.0
                    d = ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5
                    if d < avg_r * min_center_distance_factor:
                        too_close = True
                        break
                if too_close:
                    rejected_in_tile.append(cand)
                    continue

            kept.append(cand)

        filtered.extend(kept)
        removed.extend(rejected_in_tile)

        tile_key = f'{tx},{ty}'
        tile_stats[tile_key] = {
            'tile_x': int(tx * tile_px),
            'tile_y': int(ty * tile_px),
            'tile_w': min(tile_px, image_w - tx * tile_px),
            'tile_h': min(tile_px, image_h - ty * tile_px),
            'candidates_in_tile': len(candidates),
            'kept': len(kept),
            'removed': len(rejected_in_tile),
        }

    # Mark removed circles
    for cand in removed:
        cand['status'] = 'rejected'
        existing = str(cand.get('reason', ''))
        if 'removed_by_spatial' not in existing:
            cand['reason'] = f"{existing}|removed_by_spatial" if existing else 'removed_by_spatial'

    return filtered, removed, {
        'tiles': tile_stats,
        'total_before': total_before,
        'total_after': len(filtered),
        'removed': len(removed),
    }


def _loco_model_summary(
    *,
    total_candidates: int,
    sampled_candidates: int,
    discarded_empty: int,
    evaluated_candidates: int,
    accepted_before_nms: int,
    accepted_after_nms: int,
    accepted_after_spatial: int | None = None,
    rejected_by_threshold: int,
    removed_by_nms: int,
    removed_by_spatial: int = 0,
    scores: np.ndarray,
    accepted: list[dict[str, Any]],
) -> dict[str, Any]:
    hist_counts, hist_edges = np.histogram(scores, bins=np.linspace(0.0, 1.0, 11)) if scores.size else (np.zeros(10, dtype=np.int32), np.linspace(0.0, 1.0, 11))
    by_group: dict[str, int] = {'small': 0, 'medium': 0, 'large': 0}
    for cand in accepted:
        group = str(cand.get('radius_group') or 'medium')
        by_group[group] = by_group.get(group, 0) + 1
    result = {
        'total_candidates': int(total_candidates),
        'sampled_candidates': int(sampled_candidates),
        'discarded_empty': int(discarded_empty),
        'evaluated_candidates': int(evaluated_candidates),
        'accepted_before_nms': int(accepted_before_nms),
        'accepted_after_nms': int(accepted_after_nms),
        'accepted_after_spatial': int(accepted_after_spatial) if accepted_after_spatial is not None else int(accepted_after_nms),
        'rejected_by_threshold': int(rejected_by_threshold),
        'removed_by_nms': int(removed_by_nms),
        'removed_by_spatial': int(removed_by_spatial),
        'score_mean': float(np.mean(scores)) if scores.size else None,
        'score_median': float(np.median(scores)) if scores.size else None,
        'score_histogram': [
            {'bin_start': float(hist_edges[i]), 'bin_end': float(hist_edges[i + 1]), 'count': int(hist_counts[i])}
            for i in range(len(hist_counts))
        ],
        'accepted_by_radius_group': by_group,
    }
    return result


def _sample_rejected_balanced(
    rejected_all: list[dict[str, Any]],
    *,
    max_return: int,
    tile_size_px: int = 128,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample rejected circles evenly across spatial tiles for balanced visualization.

    Groups rejected circles by spatial tile (using tile_size_px), then performs
    round-robin selection across tiles up to max_return. This ensures the returned
    sample represents all regions of the image, not just the top rows.

    Args:
        rejected_all: Full list of rejected circles (each with center_x, center_y).
        max_return: Maximum number of circles to return.
        tile_size_px: Tile size in pixels for spatial grouping.
        seed: Random seed for shuffling within each tile.

    Returns:
        A list of up to max_return rejected circles, balanced across tiles.
    """
    if not rejected_all or max_return <= 0:
        return []
    if len(rejected_all) <= max_return:
        return list(rejected_all)

    tile_size = max(16, int(tile_size_px))
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for cand in rejected_all:
        x = int(round(float(cand.get('center_x') or 0.0)))
        y = int(round(float(cand.get('center_y') or 0.0)))
        tx = x // tile_size
        ty = y // tile_size
        buckets.setdefault((tx, ty), []).append(cand)

    keys = list(buckets)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(buckets[key])

    selected: list[dict[str, Any]] = []
    cursors = {key: 0 for key in keys}
    while len(selected) < max_return and keys:
        active: list[tuple[int, int]] = []
        for key in keys:
            idx = cursors[key]
            bucket = buckets[key]
            if idx < len(bucket):
                selected.append(bucket[idx])
                cursors[key] = idx + 1
                active.append(key)
                if len(selected) >= max_return:
                    break
        keys = active
    return selected


def _loco_model_overlay(image_rgb: np.ndarray, accepted: list[dict[str, Any]], rejected: list[dict[str, Any]] | None = None) -> np.ndarray:
    rgb = np.asarray(image_rgb, dtype=np.uint8).copy()
    rejected_sample = _sample_rejected_balanced(
        list(rejected or []),
        max_return=1000,
        tile_size_px=128,
        seed=42,
    )
    for cand in rejected_sample:
        x = int(round(float(cand.get('center_x') or 0.0)))
        y = int(round(float(cand.get('center_y') or 0.0)))
        r = int(round(float(cand.get('radius_px') or 0.0)))
        reason = str(cand.get('reason') or '')
        color = (96, 140, 180) if 'nms' in reason else ((150, 150, 150) if reason == 'below_threshold' else (180, 50, 45))
        if r > 0:
            cv2.circle(rgb, (x, y), r, color, thickness=1, lineType=cv2.LINE_AA)
    for cand in accepted[:2000]:
        x = int(round(float(cand.get('center_x') or 0.0)))
        y = int(round(float(cand.get('center_y') or 0.0)))
        r = int(round(float(cand.get('radius_px') or 0.0)))
        score = float(cand.get('valid_score') or 0.0)
        color = (0, 185, 80) if score >= 0.9 else ((235, 190, 0) if score >= 0.8 else (0, 150, 130))
        if r > 0:
            cv2.circle(rgb, (x, y), r, color, thickness=1, lineType=cv2.LINE_AA)
    return rgb


def _write_loco_model_csv(path: Any, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        'candidate_id',
        'center_x',
        'center_y',
        'radius_px',
        'valid_score',
        'threshold_used',
        'radius_group',
        'status',
        'reason',
        'nms_kept_candidate_id',
        'circle_iou_with_kept',
        'n_cortes',
        'area_mask_ratio',
        'n_componentes_dentro_circulo',
        'simetria_cuadrilatero',
        'continuity_score',
        'component_bridge_score',
        'dominant_component_ratio',
        'boundary_component_count',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            features = dict(row.get('features') or {})
            diagnostics = dict(row.get('diagnostics') or {})
            payload = {k: row.get(k, '') for k in fieldnames}
            payload.update({
                'n_cortes': features.get('n_cortes', ''),
                'area_mask_ratio': features.get('area_mask_ratio', ''),
                'n_componentes_dentro_circulo': features.get('n_componentes_dentro_circulo', ''),
                'simetria_cuadrilatero': features.get('simetria_cuadrilatero', ''),
                'continuity_score': diagnostics.get('continuity_score', ''),
                'component_bridge_score': diagnostics.get('component_bridge_score', ''),
                'dominant_component_ratio': diagnostics.get('dominant_component_ratio', ''),
                'boundary_component_count': diagnostics.get('boundary_component_count', ''),
            })
            writer.writerow({k: ('' if payload.get(k) is None else payload.get(k)) for k in fieldnames})


def _loco_json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _loco_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _loco_json_safe(value.item())
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, int) or isinstance(value, str) or isinstance(value, bool) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _loco_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_loco_json_safe(v) for v in value]
    return value


def _loco_lab_summary(proposals: list[dict[str, Any]], measurements: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [p for p in proposals if str(p.get('status') or '') == 'accepted']
    ok = [m for m in measurements if str(m.get('status') or '') == 'ok' and m.get('diameter_px') is not None]
    diameters = np.array([float(m.get('diameter_px')) for m in ok], dtype=np.float32) if ok else np.array([], dtype=np.float32)
    by_method: dict[str, dict[str, int]] = {}
    for p in proposals:
        method = str(p.get('method') or 'unknown')
        row = by_method.setdefault(method, {'total': 0, 'accepted': 0, 'rejected': 0})
        row['total'] += 1
        if str(p.get('status') or '') == 'accepted':
            row['accepted'] += 1
        else:
            row['rejected'] += 1
    return {
        'proposal_count': len(proposals),
        'accepted_count': len(accepted),
        'rejected_count': max(0, len(proposals) - len(accepted)),
        'measurement_count': len(measurements),
        'measurement_ok_count': len(ok),
        'diameter_mean_px': float(np.mean(diameters)) if diameters.size else None,
        'diameter_median_px': float(np.median(diameters)) if diameters.size else None,
        'by_method': by_method,
    }


def _loco_lab_overlay(image_rgb: np.ndarray, proposals: list[dict[str, Any]], measurements: list[dict[str, Any]]) -> np.ndarray:
    rgb = np.asarray(image_rgb, dtype=np.uint8).copy()
    for p in proposals[:1200]:
        center = p.get('center_xy') or [0.0, 0.0]
        x, y = int(round(float(center[0]))), int(round(float(center[1])))
        r = int(round(float(p.get('radius_px') or 0.0)))
        score = float(p.get('score', 0.0))
        if str(p.get('status') or '') == 'accepted':
            color = (0, 180, 70) if score >= 0.62 else (220, 160, 0)
        else:
            color = (200, 40, 40)
        if r > 0:
            cv2.circle(rgb, (x, y), r, color, thickness=1, lineType=cv2.LINE_AA)
    for m in measurements:
        if str(m.get('status') or '') != 'ok':
            continue
        verts = list(m.get('quadrilateral_vertices') or [])
        if len(verts) >= 4:
            pts = np.array([[int(round(float(v.get('x', 0)))), int(round(float(v.get('y', 0))))] for v in verts], dtype=np.int32)
            cv2.polylines(rgb, [pts], isClosed=True, color=(255, 170, 0), thickness=1, lineType=cv2.LINE_AA)
        left = m.get('left_edge_xy') or []
        right = m.get('right_edge_xy') or []
        if len(left) >= 2 and len(right) >= 2:
            cv2.line(
                rgb,
                (int(round(float(left[0]))), int(round(float(left[1])))),
                (int(round(float(right[0]))), int(round(float(right[1])))),
                (255, 235, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
    return rgb


@router.post('/points/update')
def points_update(req: PointsUpdateReq) -> dict[str, Any]:
    sess = _require_active_image(req.session_id)
    image_id = str(sess.image_id or '')
    state = drp.load_points(image_id)
    points = _clamp_points(list(state.get('points') or []), sess.image_rgb.shape[:2])
    active = int(state.get('active_point_idx', 0 if points else -1))

    if req.action == 'add':
        if req.x is None or req.y is None:
            raise HTTPException(status_code=400, detail='x/y requeridos para agregar punto.')
        points.append({'x': float(req.x), 'y': float(req.y)})
        points = _clamp_points(points, sess.image_rgb.shape[:2])
        active = len(points) - 1
    elif req.action == 'remove_last':
        if points:
            points.pop()
        active = min(active, len(points) - 1) if points else -1
    elif req.action == 'remove_active':
        if 0 <= active < len(points):
            points.pop(active)
        active = min(active, len(points) - 1) if points else -1
    elif req.action == 'set_active':
        idx = int(req.active_index if req.active_index is not None else -1)
        if idx < 0 or idx >= len(points):
            raise HTTPException(status_code=400, detail='active_index fuera de rango.')
        active = idx
    elif req.action == 'clear':
        points = []
        active = -1
    elif req.action == 'replace':
        points = _clamp_points([p.model_dump() for p in req.points], sess.image_rgb.shape[:2])
        idx = int(req.active_index if req.active_index is not None else (0 if points else -1))
        active = min(max(idx, -1), len(points) - 1) if points else -1

    payload = drp.save_points(image_id, points, active)
    return _points_response(sess, payload)


@router.post('/points/save')
def points_save(req: PointsSaveReq) -> dict[str, Any]:
    sess = _require_active_image(req.session_id, req.image_id)
    payload = drp.load_points(req.image_id)
    saved = drp.save_points(req.image_id, list(payload.get('points') or []), int(payload.get('active_point_idx', -1)))
    return _points_response(sess, saved)


@router.get('/points/load')
def points_load(session_id: str, image_id: str) -> dict[str, Any]:
    sess = _require_active_image(session_id, image_id)
    payload = drp.load_points(image_id)
    return _points_response(sess, payload)


@router.post('/run')
def run(req: RunReq) -> dict[str, Any]:
    sess = _require_active_image(req.session_id, req.image_id)
    image_id = str(req.image_id or '').strip()
    image_check = compute_image_id(sess.image_rgb)
    if image_check != image_id:
        raise HTTPException(status_code=400, detail='image_id no corresponde al contenido de imagen activo.')

    shape_hw = sess.image_rgb.shape[:2]
    saved_points = drp.load_points(image_id)
    points = [{'x': float(p.x), 'y': float(p.y), 'point_index': idx} for idx, p in enumerate(req.points)]
    if not points:
        points = list(saved_points.get('points') or [])
    points = _clamp_points(points, shape_hw)
    if req.active_only:
        active = int(saved_points.get('active_point_idx', 0 if points else -1))
        if not (0 <= active < len(points)):
            active = 0 if points else -1
        points = [points[active]] if active >= 0 else []
    if not points:
        raise HTTPException(status_code=400, detail='Agrega al menos un punto de medicion.')

    labels = _labels_from_request_or_draft(req, shape_hw)
    prior = None
    prior_run_id = ''
    source_mode = 'prior_mask'
    if source_mode in {'prior', 'prior_mask'}:
        prior, prior_labels, prior_run_id = _latest_prior(
            image_id,
            shape_hw,
            use_mask=True,
            prior_run_id=req.prior_run_id,
        )
        if not np.any(labels == 1) and prior_labels is not None:
            labels = prior_labels

    fg_count = int(np.sum(labels == 1))
    if prior is None and fg_count <= 0:
        raise HTTPException(status_code=400, detail='No hay prior disponible ni scribbles de fibra para construir soporte.')

    try:
        if req.method_id == METHOD_ID_V3_1:
            pipeline_out = run_hybrid_profile_diameter_v3_1(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_V3_2:
            pipeline_out = run_hybrid_profile_diameter_v3_2(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_V3_2_AUTO:
            pipeline_out = run_hybrid_profile_diameter_v3_2_auto(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_V3_2_SMALL_MASK:
            pipeline_out = run_hybrid_profile_diameter_v3_2_small_mask(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_V3_2_LARGE_IMAGE:
            pipeline_out = run_hybrid_profile_diameter_v3_2_large_image(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_CIRCLE_SQUARE:
            pipeline_out = run_circle_square_mask_diameter(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_MANUAL_DUAL_SIDE:
            pipeline_out = run_manual_dual_side_caliper(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_MANUAL_LINE_DIRECT:
            pipeline_out = run_manual_line_direct_caliper(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_ELLIPSE_FIT:
            pipeline_out = run_ellipse_oriented_fit(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_LOCO:
            pipeline_out = run_loco_circle_probe(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_V3_2_SMALL_LARGE:
            pipeline_out = run_hybrid_profile_diameter_v3_2_small_large(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_V3_2_HALO_AWARE:
            pipeline_out = run_hybrid_profile_diameter_v3_2_halo_aware(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_V3_2_RIDGE_ANCHORED:
            pipeline_out = run_hybrid_profile_diameter_v3_2_ridge_anchored(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_V3_2_FLUX_AWARE:
            pipeline_out = run_hybrid_profile_diameter_v3_2_flux_aware(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_V3_2_CONTOUR_REFINE:
            pipeline_out = run_hybrid_profile_diameter_v3_2_contour_refine(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_V3_2_CURVELET_AIDED:
            pipeline_out = run_hybrid_profile_diameter_v3_2_curvelet_aided(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_V3_3:
            pipeline_out = run_hybrid_profile_diameter_v3_3(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_V3_3A:
            pipeline_out = run_hybrid_profile_diameter_v3_3a(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_V3_3B:
            pipeline_out = run_hybrid_profile_diameter_v3_3b(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_V3_3C:
            pipeline_out = run_hybrid_profile_diameter_v3_3c(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_V3_3D:
            pipeline_out = run_hybrid_profile_diameter_v3_3d(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_V3:
            pipeline_out = run_hybrid_profile_diameter_v3(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        elif req.method_id == METHOD_ID_V2:
            pipeline_out = run_hybrid_profile_diameter_v2(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
        else:
            pipeline_out = run_hybrid_profile_diameter(
                image_rgb=sess.image_rgb,
                labels=labels,
                prior_map=prior,
                points=points,
                params=dict(req.params or {}),
                source_mode=req.source_mode,
            )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'No se pudo ejecutar Diameter Research: {exc}') from exc

    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    method_id = str(pipeline_out.get('method_id') or pipeline_out.get('experiment_id') or req.method_id or METHOD_ID)
    run_id = drp.new_run_id(method_id)
    meta = dict(pipeline_out.get('meta') or {})
    meta['run_id'] = run_id
    meta['image_id'] = image_id
    meta['created_at'] = created_at
    meta['method_id'] = method_id
    meta['prior_run_id'] = prior_run_id
    meta['requested_prior_run_id'] = str(req.prior_run_id or '')
    meta['fallback'] = str((meta.get('support') or {}).get('fallback') or '')
    meta['points_ok'] = int(sum(1 for r in pipeline_out.get('results', []) if r.get('status') == 'ok'))

    art = drp.DiameterRunArtifacts(
        run_id=run_id,
        image_id=image_id,
        experiment_id=method_id,
        created_at=created_at,
        input_image=sess.image_rgb,
        scribble_labels=labels,
        prior_prob=np.zeros(shape_hw, dtype=np.float32) if prior is None else np.asarray(prior, dtype=np.float32),
        support_region=np.asarray(pipeline_out.get('support_region'), dtype=np.uint8),
        overlay=np.asarray(pipeline_out.get('overlay'), dtype=np.uint8),
        results=list(pipeline_out.get('results') or []),
        diagnostics=dict(pipeline_out.get('diagnostics') or {}),
        meta=meta,
    )
    save_meta = drp.save_diameter_run(art)
    item = drp.load_diameter_run(run_id)
    payload = _run_payload(item)
    payload['meta']['save'] = save_meta
    return payload


@router.post('/loco/preview')
def loco_preview(req: LocoPreviewReq) -> dict[str, Any]:
    sess = _require_active_image(req.session_id, req.image_id)
    image_id = str(req.image_id or '').strip()
    image_check = compute_image_id(sess.image_rgb)
    if image_check != image_id:
        raise HTTPException(status_code=400, detail='image_id no corresponde al contenido de imagen activo.')

    shape_hw = sess.image_rgb.shape[:2]
    labels = _labels_from_request_or_draft(req, shape_hw)  # type: ignore[arg-type]
    prior = None
    prior_run_id = ''
    source_mode = 'prior_mask'
    if source_mode in {'prior', 'prior_mask'}:
        prior, prior_labels, prior_run_id = _latest_prior(
            image_id,
            shape_hw,
            use_mask=True,
            prior_run_id=req.prior_run_id,
        )
        if not np.any(labels == 1) and prior_labels is not None:
            labels = prior_labels
    if prior is None and int(np.sum(labels == 1)) <= 0:
        raise HTTPException(status_code=400, detail='No hay prior disponible ni scribbles de fibra para construir soporte.')

    point = {'x': float(req.point.x), 'y': float(req.point.y), 'point_index': 0}
    try:
        pipeline_out = run_loco_circle_probe(
            image_rgb=sess.image_rgb,
            labels=labels,
            prior_map=prior,
            points=[point],
            params=dict(req.params or {}),
            source_mode=req.source_mode,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'No se pudo generar preview LOCO: {exc}') from exc

    overlay_b64, overlay_mime = encode_display_b64(pipeline_out.get('overlay'))
    support = (np.asarray(pipeline_out.get('support_region')) > 0).astype(np.uint8) * 255
    results = list(pipeline_out.get('results') or [])
    result = dict(results[0] if results else {})
    candidates = list(result.get('radius_candidates') or [])
    best_radius = result.get('loco_best_radius_px')
    best_idx = -1
    if best_radius is not None and candidates:
        try:
            best_idx = int(np.argmin([abs(float(c.get('radius_px', 0.0)) - float(best_radius)) for c in candidates]))
        except Exception:
            best_idx = 0
    requested_idx = int(req.candidate_index)
    candidate_idx = requested_idx if 0 <= requested_idx < len(candidates) else best_idx
    return {
        'ok': True,
        'image_id': image_id,
        'method_id': METHOD_ID_LOCO,
        'source_mode': req.source_mode,
        'prior_run_id': prior_run_id,
        'preview_step': int(req.step),
        'candidate_index': int(candidate_idx),
        'best_candidate_index': int(best_idx),
        'overlay_b64': overlay_b64,
        'overlay_mime': overlay_mime,
        'support_region_b64': encode_gray_png_b64(support),
        'result': result,
        'radius_candidates': candidates,
        'meta': dict(pipeline_out.get('meta') or {}),
        'diagnostics': dict(pipeline_out.get('diagnostics') or {}),
    }


@router.post('/loco/proposals')
def loco_lab_proposals(req: LocoLabReq) -> dict[str, Any]:
    _sess, image_id, _labels, _prior, support, prior_run_id = _loco_lab_context(req)
    params = dict(req.params or {})
    method = str(req.method or 'circle_grid')
    proposals: list[dict[str, Any]] = []
    if method in {'circle_grid', 'both'}:
        proposals.extend(_loco_grid_proposals(support, params))
    if method in {'fiber_first', 'both'}:
        proposals.extend(_loco_fiber_first_proposals(support, params))
    proposals.sort(key=lambda item: float(item.get('score', 0.0)), reverse=True)
    max_candidates = _loco_int_param(params, 'max_candidates', 600, lo=1, hi=8000)
    proposals = proposals[:max_candidates]
    return {
        'ok': True,
        'image_id': image_id,
        'prior_run_id': prior_run_id,
        'method': method,
        'support_region_b64': encode_gray_png_b64((support > 0).astype(np.uint8) * 255),
        'proposals': proposals,
        'summary': _loco_lab_summary(proposals, []),
    }


@router.post('/loco/filter')
def loco_lab_filter(req: LocoLabFilterReq) -> dict[str, Any]:
    _require_active_image(req.session_id, req.image_id)
    proposals = _filter_loco_proposals(list(req.proposals or []), dict(req.params or {}))
    return {
        'ok': True,
        'image_id': str(req.image_id or ''),
        'proposals': proposals,
        'summary': _loco_lab_summary(proposals, []),
    }


@router.post('/loco/measure')
def loco_lab_measure(req: LocoLabMeasureReq) -> dict[str, Any]:
    _sess, image_id, _labels, _prior, support, prior_run_id = _loco_lab_context(req)
    params = dict(req.params or {})
    proposals = list(req.proposals or [])
    selected = [p for p in proposals if str(p.get('status') or '') == 'accepted']
    if not selected:
        selected = proposals[:_loco_int_param(params, 'measure_limit', 120, lo=1, hi=2000)]
    limit = _loco_int_param(params, 'measure_limit', 120, lo=1, hi=2000)
    measurements = [_measure_loco_circle_square(support, p, params) for p in selected[:limit]]
    return {
        'ok': True,
        'image_id': image_id,
        'prior_run_id': prior_run_id,
        'measurements': measurements,
        'summary': _loco_lab_summary(proposals, measurements),
    }


@router.post('/loco/evaluate')
def loco_lab_evaluate(req: LocoLabEvaluateReq) -> dict[str, Any]:
    _require_active_image(req.session_id, req.image_id)
    proposals = list(req.proposals or [])
    measurements = list(req.measurements or [])
    return {
        'ok': True,
        'image_id': str(req.image_id or ''),
        'summary': _loco_lab_summary(proposals, measurements),
    }


@router.post('/loco/run')
def loco_lab_run(req: LocoLabRunReq) -> dict[str, Any]:
    sess, image_id, labels, prior, support, prior_run_id = _loco_lab_context(req)
    params = dict(req.params or {})
    proposals = list(req.proposals or [])
    measurements = list(req.measurements or [])
    if not measurements:
        selected = [p for p in proposals if str(p.get('status') or '') == 'accepted']
        if not selected:
            selected = proposals[:_loco_int_param(params, 'measure_limit', 120, lo=1, hi=2000)]
        measurements = [_measure_loco_circle_square(support, p, params) for p in selected[:_loco_int_param(params, 'measure_limit', 120, lo=1, hi=2000)]]
    summary = _loco_lab_summary(proposals, measurements)
    if not proposals and not measurements:
        raise HTTPException(status_code=400, detail='No hay propuestas ni mediciones LOCO para guardar.')

    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    method_id = 'loco_lab'
    run_id = drp.new_run_id(method_id)
    overlay = _loco_lab_overlay(sess.image_rgb, proposals, measurements)
    results: list[dict[str, Any]] = []
    for idx, m in enumerate(measurements):
        center = m.get('circle_xy') or [0.0, 0.0]
        results.append(
            {
                'point_index': idx,
                'x': float(center[0]) if len(center) >= 2 else 0.0,
                'y': float(center[1]) if len(center) >= 2 else 0.0,
                'method_id': method_id,
                'route': str(m.get('method') or 'loco_lab'),
                'measurement_mode': 'loco_lab_circle_square',
                'status': str(m.get('status') or 'rejected'),
                'quality': str(m.get('quality_label') or 'rejected'),
                'quality_label': str(m.get('quality_label') or 'rejected'),
                'reason': str(m.get('reason') or ''),
                'diameter_px': m.get('diameter_px'),
                'confidence': float(m.get('confidence') or m.get('score') or 0.0),
                'stability': 1.0 if str(m.get('status') or '') == 'ok' else 0.0,
                'loco_best_radius_px': m.get('radius_px'),
                'loco_symmetry_score': m.get('symmetry_score'),
                'loco_intersection_count': m.get('intersection_count'),
                'left_edge_xy': m.get('left_edge_xy'),
                'right_edge_xy': m.get('right_edge_xy'),
                'quadrilateral_vertices': m.get('quadrilateral_vertices'),
                'proposal_id': m.get('proposal_id'),
            }
        )
    meta = {
        'run_id': run_id,
        'image_id': image_id,
        'created_at': created_at,
        'method_id': method_id,
        'prior_run_id': prior_run_id,
        'requested_prior_run_id': str(req.prior_run_id or ''),
        'points_ok': int(summary.get('measurement_ok_count') or 0),
        'proposal_count': int(summary.get('proposal_count') or 0),
        'accepted_count': int(summary.get('accepted_count') or 0),
        'params': params,
    }
    art = drp.DiameterRunArtifacts(
        run_id=run_id,
        image_id=image_id,
        experiment_id=method_id,
        created_at=created_at,
        input_image=sess.image_rgb,
        scribble_labels=labels,
        prior_prob=np.asarray(prior, dtype=np.float32),
        support_region=(support > 0).astype(np.uint8),
        overlay=overlay,
        results=results,
        diagnostics={
            'diagnostics_loco': {'proposals': proposals, 'measurements': measurements, 'summary': summary},
            'loco_lab': {'proposals': proposals, 'measurements': measurements, 'summary': summary},
            'loco_overlay': overlay,
        },
        meta=meta,
    )
    save_meta = drp.save_diameter_run(art)
    item = drp.load_diameter_run(run_id)
    payload = _run_payload(item)
    payload['meta']['save'] = save_meta
    return payload


@router.post('/loco-dataset/features')
def loco_dataset_features(req: LocoDatasetReq) -> dict[str, Any]:
    _sess, image_id, _labels, _prior, support, prior_run_id = _loco_lab_context(req)
    params = dict(req.params or {})
    candidates = [_candidate_to_dict(c) for c in (req.candidates or [])]
    rows = [_loco_dataset_features_for_candidate(support, cand, params) for cand in candidates]
    return {
        'ok': True,
        'image_id': image_id,
        'prior_run_id': prior_run_id,
        'feature_names': LOCO_DATASET_FEATURE_NAMES,
        'items': rows,
    }


@router.post('/loco-dataset/save')
def loco_dataset_save(req: LocoDatasetReq) -> dict[str, Any]:
    sess, image_id, _labels, _prior, support, prior_run_id = _loco_lab_context(req)
    params = dict(req.params or {})
    patch_size = _loco_int_param(params, 'patch_size', 64, lo=16, hi=256)
    candidates = [_candidate_to_dict(c) for c in (req.candidates or [])]
    if not candidates:
        raise HTTPException(status_code=400, detail='No hay candidatos para guardar.')
    for cand in candidates:
        if str(cand.get('label') or '') not in {'valid', 'invalid', 'invalid_crossing', 'invalid_other'}:
            raise HTTPException(status_code=400, detail=f"Candidato sin etiqueta valida: {cand.get('candidate_id') or '-'}")
        if float(cand.get('radius_px') or 0.0) < 1.0:
            raise HTTPException(status_code=400, detail=f"Candidato con radio invalido: {cand.get('candidate_id') or '-'}")

    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    dataset_id = 'main'
    root = drp.OUTPUT_ROOT / 'datasets' / 'loco_circle_dataset' / dataset_id
    feature_dir = root / 'features'
    (root / 'valid').mkdir(parents=True, exist_ok=True)
    (root / 'invalid_crossing').mkdir(parents=True, exist_ok=True)
    (root / 'invalid_other').mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    # Label mapping: label_text -> label_binary, label_multiclass
    def _label_map(lt: str) -> tuple[int, int]:
        if lt == 'valid':
            return (1, 0)
        elif lt == 'invalid_crossing':
            return (0, 1)
        else:  # invalid_other or legacy invalid
            return (0, 2)

    new_rows: list[dict[str, Any]] = []
    for idx, cand in enumerate(candidates):
        cid = str(cand.get('candidate_id') or f'cand{idx + 1:04d}').strip()
        label_text = str(cand.get('label') or '')
        # Normalize legacy 'invalid' to 'invalid_other'
        if label_text == 'invalid':
            label_text = 'invalid_other'
        label_binary, label_multiclass = _label_map(label_text)
        cx = float(cand.get('center_x') or 0.0)
        cy = float(cand.get('center_y') or 0.0)
        radius = float(cand.get('radius_px') or 0.0)
        patch, _area_ratio = _circle_disk_patch(support, (cx, cy), radius, patch_size=patch_size)
        base_name = f'{image_id}_{cid}'
        safe_name = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in base_name)
        patch_rel = f'{label_text}/{safe_name}.png'
        feature_rel = f'features/{label_text}/{safe_name}.json'
        # Clean up old paths for this candidate across all label dirs
        for old_label in ('valid', 'invalid', 'invalid_crossing', 'invalid_other'):
            for old_rel in (
                f'{old_label}/{safe_name}.png',
                f'features/{old_label}/{safe_name}.json',
            ):
                if old_rel not in {patch_rel, feature_rel}:
                    old_path = root / old_rel
                    if old_path.exists():
                        old_path.unlink()
        _write_gray_png(root / patch_rel, patch)
        item = _loco_dataset_features_for_candidate(support, cand, {**params, 'patch_size': patch_size})
        feature_payload = {
            'dataset_id': dataset_id,
            'image_id': image_id,
            'image_name': str(getattr(sess, 'image_name', '') or ''),
            'candidate_id': cid,
            'center_x': cx,
            'center_y': cy,
            'radius_px': radius,
            'label': label_text,
            'label_binary': label_binary,
            'label_multiclass': label_multiclass,
            'mask_patch_path': patch_rel,
            'features': item['features'],
            'diagnostics': item.get('diagnostics') or {},
        }
        feature_path = root / feature_rel
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        feature_path.write_text(json.dumps(feature_payload, ensure_ascii=False, indent=2), encoding='utf-8')
        row = {
            'dataset_id': dataset_id,
            'image_id': image_id,
            'candidate_id': cid,
            'center_x': cx,
            'center_y': cy,
            'radius_px': radius,
            'label_text': label_text,
            'label_binary': label_binary,
            'label_multiclass': label_multiclass,
            'mask_patch_path': patch_rel,
        }
        row.update(item['features'])
        new_rows.append(row)

    csv_path = root / 'metadata.csv'
    fieldnames = ['dataset_id', 'image_id', 'candidate_id', 'center_x', 'center_y', 'radius_px', 'label_text', 'label_binary', 'label_multiclass', 'mask_patch_path', *LOCO_DATASET_FEATURE_NAMES]
    existing_rows = _read_loco_dataset_metadata(csv_path, fieldnames)
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in existing_rows:
        key = (str(row.get('image_id') or ''), str(row.get('candidate_id') or ''))
        if key[0] and key[1]:
            merged[key] = row
    for row in new_rows:
        merged[(str(row.get('image_id') or ''), str(row.get('candidate_id') or ''))] = row
    rows = list(merged.values())
    with csv_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if row.get(k) is None else row.get(k)) for k in fieldnames})

    warnings: list[str] = []
    valid_count = sum(1 for row in rows if str(row.get('label_text') or '') == 'valid')
    crossing_count = sum(1 for row in rows if str(row.get('label_text') or '') == 'invalid_crossing')
    other_count = sum(1 for row in rows if str(row.get('label_text') or '') == 'invalid_other')
    if valid_count <= 0:
        warnings.append('dataset_without_valid_examples')
    if crossing_count <= 0 and other_count <= 0:
        warnings.append('dataset_without_invalid_examples')
    old_meta: dict[str, Any] = {}
    meta_path = root / 'dataset_meta.json'
    if meta_path.exists():
        try:
            old_meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except Exception:
            old_meta = {}
    meta = {
        'dataset_id': dataset_id,
        'created_at': str(old_meta.get('created_at') or created_at),
        'updated_at': created_at,
        'image_count': len({str(row.get('image_id') or '') for row in rows if str(row.get('image_id') or '')}),
        'last_image_id': image_id,
        'last_image_name': str(getattr(sess, 'image_name', '') or ''),
        'prior_run_id': prior_run_id,
        'requested_prior_run_id': str(req.prior_run_id or ''),
        'patch_size': patch_size,
        'candidate_count': len(rows),
        'last_write_candidate_count': len(candidates),
        'valid_count': valid_count,
        'crossing_count': crossing_count,
        'other_count': other_count,
        'feature_names': LOCO_DATASET_FEATURE_NAMES,
        'layout': {
            'valid_patches': 'valid/',
            'crossing_patches': 'invalid_crossing/',
            'other_patches': 'invalid_other/',
            'features': 'features/{valid,invalid_crossing,invalid_other}/',
        },
        'params': params,
        'warnings': warnings,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return {
        'ok': True,
        'dataset_id': dataset_id,
        'dataset_dir': str(root),
        'candidate_count': len(rows),
        'last_write_candidate_count': len(candidates),
        'valid_count': valid_count,
        'crossing_count': crossing_count,
        'other_count': other_count,
        'warnings': warnings,
    }


@router.post('/loco-dataset/clean-legacy-invalid')
def loco_dataset_clean_legacy_invalid() -> dict[str, Any]:
    """Remove all legacy 'invalid' examples from the dataset.

    Legacy 'invalid' examples (label_text='invalid') were created before the
    split into invalid_crossing and invalid_other. This endpoint removes them
    entirely so the user can re-label them properly.
    """
    root = _loco_dataset_root()
    csv_path = root / 'metadata.csv'
    fieldnames = ['dataset_id', 'image_id', 'candidate_id', 'center_x', 'center_y', 'radius_px', 'label_text', 'label_binary', 'label_multiclass', 'mask_patch_path', *LOCO_DATASET_FEATURE_NAMES]
    rows = _read_loco_dataset_metadata(csv_path, fieldnames)

    kept: list[dict[str, Any]] = []
    removed_count = 0
    removed_ids: list[str] = []

    for row in rows:
        label = str(row.get('label_text') or '').strip()
        if label == 'invalid':
            removed_count += 1
            image_id = str(row.get('image_id') or '')
            candidate_id = str(row.get('candidate_id') or '')
            removed_ids.append(f'{image_id}::{candidate_id}')
            # Delete the PNG patch
            rel = _safe_loco_rel_path(str(row.get('mask_patch_path') or ''))
            if rel:
                png_path = root / rel
                if png_path.exists():
                    png_path.unlink()
            # Delete the feature JSON
            if image_id and candidate_id:
                feat_path = _loco_feature_json_path(root, 'invalid', image_id, candidate_id)
                if feat_path.exists():
                    feat_path.unlink()
        else:
            kept.append(row)

    # Rewrite metadata.csv without legacy invalid rows
    with csv_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in kept:
            writer.writerow({k: ('' if row.get(k) is None else row.get(k)) for k in fieldnames})

    # Clean up empty directories
    invalid_dir = root / 'invalid'
    if invalid_dir.exists():
        remaining = list(invalid_dir.glob('*.png'))
        if not remaining:
            try:
                invalid_dir.rmdir()
            except Exception:
                pass
    feat_invalid_dir = root / 'features' / 'invalid'
    if feat_invalid_dir.exists():
        remaining = list(feat_invalid_dir.glob('*.json'))
        if not remaining:
            try:
                feat_invalid_dir.rmdir()
            except Exception:
                pass

    # Update dataset_meta.json
    meta_path = root / 'dataset_meta.json'
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except Exception:
            meta = {}

    valid_count = sum(1 for row in kept if str(row.get('label_text') or '') == 'valid')
    crossing_count = sum(1 for row in kept if str(row.get('label_text') or '') == 'invalid_crossing')
    other_count = sum(1 for row in kept if str(row.get('label_text') or '') == 'invalid_other')

    meta['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    meta['candidate_count'] = len(kept)
    meta['valid_count'] = valid_count
    meta['crossing_count'] = crossing_count
    meta['other_count'] = other_count
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    return {
        'ok': True,
        'removed_count': removed_count,
        'remaining_count': len(kept),
        'valid_count': valid_count,
        'crossing_count': crossing_count,
        'other_count': other_count,
        'removed_ids': removed_ids,
    }


@router.get('/loco-dataset/augment/items')
def loco_dataset_augment_items() -> dict[str, Any]:
    payload = _loco_dataset_items()
    return {
        'ok': True,
        'dataset_id': 'main',
        'dataset_dir': str(payload['root']),
        'items': payload['items'],
        'counts': payload['counts'],
    }


def _select_loco_aug_items(req: LocoDatasetAugmentReq) -> tuple[Any, list[dict[str, Any]]]:
    payload = _loco_dataset_items()
    root = payload['root']
    all_items = list(payload['items'])
    label_filter = str(req.label_filter or 'all')
    if label_filter in {'valid', 'invalid', 'invalid_crossing', 'invalid_other'}:
        # Normalize legacy 'invalid' filter to 'invalid_other'
        norm_filter = _loco_normalize_label(label_filter)
        all_items = [item for item in all_items if item.get('label') == norm_filter]
    selected_ids = {str(x) for x in (req.items or []) if str(x).strip()}
    if selected_ids:
        all_items = [item for item in all_items if str(item.get('item_id')) in selected_ids]
    return root, all_items


@router.post('/loco-dataset/augment/preview')
def loco_dataset_augment_preview(req: LocoDatasetAugmentReq) -> dict[str, Any]:
    root, items = _select_loco_aug_items(req)
    limit = int(np.clip(int(req.max_items or 12), 1, 48))
    pipeline = list(req.pipeline or [])
    max_variants = int(np.clip(int(req.max_variants_per_source or 32), 1, 128))
    passes = int(np.clip(int(req.passes_per_source or 4), 1, max_variants))
    random_seed = int(secrets.randbits(63))
    rng = np.random.default_rng(random_seed)
    previews: list[dict[str, Any]] = []
    for item in items[:limit]:
        source_path = root / _safe_loco_rel_path(str(item.get('source_path') or ''))
        if not source_path.exists():
            continue
        src = _read_gray_png(source_path)
        variants = _generate_loco_aug_random_variants(src, pipeline, str(item.get('item_id') or ''), passes=passes, max_variants=max_variants, rng=rng)
        previews.append(
            {
                'item': item,
                'source_b64': _gray_png_b64(src),
                'variants': [
                    {
                        'chain': v['chain'],
                        'image_b64': _gray_png_b64(v['image']),
                    }
                    for v in variants
                ],
            }
        )
    return {
        'ok': True,
        'dataset_id': 'main',
        'pipeline_hash': _loco_pipeline_hash(pipeline),
        'random_seed': random_seed,
        'passes_per_source': passes,
        'preview_count': len(previews),
        'source_count': len(items),
        'variant_count': sum(len(p.get('variants') or []) for p in previews),
        'items': previews,
    }


@router.post('/loco-dataset/augment/apply')
def loco_dataset_augment_apply(req: LocoDatasetAugmentReq) -> dict[str, Any]:
    if not req.pipeline:
        raise HTTPException(status_code=400, detail='Agrega al menos un bloque de aumentacion antes de aplicar.')
    root, items = _select_loco_aug_items(req)
    if not items:
        raise HTTPException(status_code=400, detail='No hay ejemplos del dataset main para aumentar.')
    pipeline = list(req.pipeline or [])
    pipeline_hash = _loco_pipeline_hash(pipeline)
    max_variants = int(np.clip(int(req.max_variants_per_source or 32), 1, 128))
    passes = int(np.clip(int(req.passes_per_source or 4), 1, max_variants))
    random_seed = int(secrets.randbits(63))
    rng = np.random.default_rng(random_seed)
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    aug_root = _loco_augmented_root(root)
    for rel_dir in ('valid', 'invalid_crossing', 'invalid_other', 'features/valid', 'features/invalid_crossing', 'features/invalid_other'):
        (aug_root / rel_dir).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for item in items:
        label = str(item.get('label') or '')
        label_norm = _loco_normalize_label(label)
        if label_norm not in {'valid', 'invalid_crossing', 'invalid_other'}:
            continue
        label = label_norm
        source_rel = _safe_loco_rel_path(str(item.get('source_path') or ''))
        source_path = root / source_rel
        if not source_path.exists():
            continue
        src = _read_gray_png(source_path)
        variants = _generate_loco_aug_random_variants(src, pipeline, str(item.get('item_id') or ''), passes=passes, max_variants=max_variants, rng=rng)
        base = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in f"{item.get('image_id')}_{item.get('candidate_id')}")
        for idx, variant in enumerate(variants):
            chain = [str(x) for x in (variant.get('chain') or [])]
            chain_hash = hashlib.sha1('|'.join(chain).encode('utf-8')).hexdigest()[:10]
            seed_tag = f'{random_seed:x}'[-8:]
            file_name = f'{base}__aug_{pipeline_hash}_{seed_tag}_{idx + 1:03d}_{chain_hash}.png'
            rel_png = f'{label}/{file_name}'
            rel_json = f'features/{label}/{file_name[:-4]}.json'
            aug_img = _threshold64(variant['image'])
            _write_gray_png(aug_root / rel_png, aug_img)
            aug_feature_item = _loco_augmented_features_for_patch(aug_img, f'{item.get("candidate_id") or ""}__aug_{idx + 1:03d}', label)
            aug_features = dict(aug_feature_item.get('features') or {})
            feature_payload = {
                'dataset_id': 'main',
                'augmented_dataset': 'augmented',
                'source_candidate_id': str(item.get('candidate_id') or ''),
                'source_image_id': str(item.get('image_id') or ''),
                'source_label': label,
                'source_path': source_rel,
                'source_radius_px': item.get('radius_px') or '',
                'source_features': item.get('source_features') or {},
                'augmented_path': rel_png,
                'augmented_features': aug_features,
                'augmented_diagnostics': aug_feature_item.get('diagnostics') or {},
                'transform_chain': chain,
                'pipeline_hash': pipeline_hash,
                'random_seed': random_seed,
                'created_at': created_at,
            }
            (aug_root / rel_json).write_text(json.dumps(feature_payload, ensure_ascii=False, indent=2), encoding='utf-8')
            row = {
                'dataset_id': 'main',
                'source_image_id': item.get('image_id') or '',
                'source_candidate_id': item.get('candidate_id') or '',
                'source_label': label,
                'source_path': source_rel,
                'source_radius_px': item.get('radius_px') or '',
                'augmented_path': rel_png,
                'feature_path': rel_json,
                'transform_chain': '|'.join(chain),
                'pipeline_hash': pipeline_hash,
                'random_seed': random_seed,
                'created_at': created_at,
            }
            row.update(aug_features)
            rows.append(row)

    csv_path = aug_root / 'augmented_metadata.csv'
    fieldnames = ['dataset_id', 'source_image_id', 'source_candidate_id', 'source_label', 'source_path', 'source_radius_px', 'augmented_path', 'feature_path', 'transform_chain', 'pipeline_hash', 'random_seed', 'created_at', *LOCO_DATASET_FEATURE_NAMES]
    existing: dict[str, dict[str, Any]] = {}
    if csv_path.exists():
        with csv_path.open('r', newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = str(row.get('augmented_path') or '')
                if key:
                    existing[key] = {k: row.get(k, '') for k in fieldnames}
    for row in rows:
        existing[str(row.get('augmented_path') or '')] = row
    merged_rows = list(existing.values())
    with csv_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})

    valid_count = len(list((aug_root / 'valid').glob('*.png'))) if (aug_root / 'valid').exists() else 0
    crossing_count = len(list((aug_root / 'invalid_crossing').glob('*.png'))) if (aug_root / 'invalid_crossing').exists() else 0
    other_count = len(list((aug_root / 'invalid_other').glob('*.png'))) if (aug_root / 'invalid_other').exists() else 0
    meta = {
        'dataset_id': 'main',
        'augmented_dataset': 'augmented',
        'updated_at': created_at,
        'pipeline_hash': pipeline_hash,
        'random_seed': random_seed,
        'pipeline': pipeline,
        'passes_per_source': passes,
        'source_count': len(items),
        'last_write_count': len(rows),
        'augmented_count': valid_count + crossing_count + other_count,
        'valid_count': valid_count,
        'crossing_count': crossing_count,
        'other_count': other_count,
        'layout': {
            'valid': 'augmented/valid/',
            'crossing': 'augmented/invalid_crossing/',
            'other': 'augmented/invalid_other/',
            'features': 'augmented/features/{valid,invalid_crossing,invalid_other}/',
            'metadata': 'augmented/augmented_metadata.csv',
        },
    }
    (aug_root / 'augmentation_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return {
        'ok': True,
        'dataset_id': 'main',
        'augmented_dir': str(aug_root),
        'last_write_count': len(rows),
        'augmented_count': valid_count + crossing_count + other_count,
        'valid_count': valid_count,
        'crossing_count': crossing_count,
        'other_count': other_count,
        'pipeline_hash': pipeline_hash,
        'random_seed': random_seed,
    }


@router.post('/loco-dataset/augment/clear')
def loco_dataset_augment_clear() -> dict[str, Any]:
    root = _loco_dataset_root()
    aug_root = _loco_augmented_root(root)
    if aug_root.exists():
        shutil.rmtree(aug_root)
    return {
        'ok': True,
        'dataset_id': 'main',
        'augmented_dir': str(aug_root),
        'augmented_count': 0,
        'valid_count': 0,
        'crossing_count': 0,
        'other_count': 0,
    }


@router.post('/loco-training/train')
def loco_training_train(req: LocoTrainingReq) -> dict[str, Any]:
    selection = 'all' if req.data_selection == 'all' else ('original' if req.data_selection == 'original' else 'augmented')
    vector_config = _loco_vector_config_from_req(req)
    items = _loco_training_items(selection)
    if not items:
        raise HTTPException(status_code=400, detail='No hay ejemplos LOCO para entrenar.')
    labels = np.asarray([int(item.get('label_numeric') or 0) for item in items], dtype=np.int32)
    if len(set(labels.tolist())) < 2:
        raise HTTPException(status_code=400, detail='Se necesitan ejemplos valid e invalid para entrenar.')

    train_idx, test_idx = _loco_group_split(items, req.test_size, req.random_seed)
    X_all = np.vstack([_loco_training_vector(item, vector_config) for item in items]).astype(np.float32)
    y_all = labels
    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    run_id = f"loco_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.sha1(f'{created_at}|{len(items)}|{req.random_seed}'.encode()).hexdigest()[:8]}"
    run_root = drp.OUTPUT_ROOT / 'training_runs' / run_id
    (run_root / 'models').mkdir(parents=True, exist_ok=True)

    model_labels = {
        'catboost': 'CatBoost',
        'lightgbm': 'LightGBM',
        'xgboost': 'XGBoost',
        'extratrees': 'ExtraTrees',
    }
    summary_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    radius_rows: list[dict[str, Any]] = []
    confusion: dict[str, Any] = {}
    error_review: list[dict[str, Any]] = []
    model_payloads: dict[str, Any] = {}
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    test_items = [items[i] for i in test_idx]
    test_radii = np.asarray([_float_or_nan(item.get('radius_for_group')) for item in test_items], dtype=np.float32)
    finite_radii = test_radii[np.isfinite(test_radii)]
    q1, q2 = (np.nan, np.nan)
    if finite_radii.size >= 3:
        q1, q2 = np.quantile(finite_radii, [1 / 3, 2 / 3])

    for model_id in req.models:
        model_name = model_labels.get(model_id, model_id)
        try:
            model = _loco_model_instance(model_id, int(req.random_seed))
            model.fit(X_train, y_train)
            if hasattr(model, 'predict_proba'):
                y_prob = np.asarray(model.predict_proba(X_test))[:, 1]
            else:
                y_prob = np.asarray(model.predict(X_test), dtype=np.float32)
            base = _positive_metrics(y_test, y_prob, 0.5)
            try:
                pr_auc = float(average_precision_score(y_test, y_prob))
            except Exception:
                pr_auc = float('nan')
            base.update({
                'model': model_name,
                'model_id': model_id,
                'status': 'ok',
                'pr_auc': pr_auc,
                'test_samples': int(len(y_test)),
                'train_samples': int(len(y_train)),
            })
            summary_rows.append(base)
            confusion[model_id] = {'tn': base['tn'], 'fp': base['fp'], 'fn': base['fn'], 'tp': base['tp']}
            joblib.dump(model, run_root / 'models' / f'{model_id}_model.pkl')

            th_metrics: list[dict[str, Any]] = []
            for th in thresholds:
                row = _positive_metrics(y_test, y_prob, th)
                row.update({'model': model_name, 'model_id': model_id, 'threshold': th})
                threshold_rows.append(row)
                th_metrics.append(row)
            precision_sorted = sorted(th_metrics, key=lambda r: (float(r['precision_valid']), float(r['recall_valid']), float(r['threshold'])), reverse=True)
            f1_sorted = sorted(th_metrics, key=lambda r: (float(r['f1_valid']), float(r['precision_valid'])), reverse=True)
            model_payloads[model_id] = {
                'model': model_name,
                'model_id': model_id,
                'recommended_threshold_precision': precision_sorted[0]['threshold'] if precision_sorted else 0.5,
                'recommended_threshold_f1': f1_sorted[0]['threshold'] if f1_sorted else 0.5,
            }

            if np.isfinite(q1) and np.isfinite(q2):
                group_names = []
                for r in test_radii:
                    if not np.isfinite(r):
                        group_names.append('unknown')
                    elif r <= q1:
                        group_names.append('small')
                    elif r <= q2:
                        group_names.append('medium')
                    else:
                        group_names.append('large')
                for group in ['small', 'medium', 'large']:
                    mask = np.asarray([g == group for g in group_names], dtype=bool)
                    if not np.any(mask):
                        continue
                    row = _positive_metrics(y_test[mask], y_prob[mask], 0.5)
                    row.update({'model': model_name, 'model_id': model_id, 'radius_group': group, 'n_samples': int(np.sum(mask))})
                    radius_rows.append(row)

            y_pred = (y_prob >= 0.5).astype(np.int32)
            for local_idx, item in enumerate(test_items):
                real = int(y_test[local_idx])
                pred = int(y_pred[local_idx])
                if real == pred:
                    continue
                err_type = 'False Positives' if real == 0 and pred == 1 else 'False Negatives'
                try:
                    patch_b64 = _patch_b64_from_path(item['patch_path'])
                except Exception:
                    patch_b64 = ''
                error_review.append({
                    'model': model_name,
                    'model_id': model_id,
                    'error_type': err_type,
                    'candidate_id': item.get('candidate_id') or '',
                    'item_id': item.get('item_id') or '',
                    'dataset_kind': item.get('dataset_kind') or '',
                    'label_real': int(real),
                    'prediction': int(pred),
                    'probability_valid': float(y_prob[local_idx]),
                    'radius_px': None if not np.isfinite(test_radii[local_idx]) else float(test_radii[local_idx]),
                    'patch_path': str(item.get('patch_rel') or ''),
                    'patch_b64': patch_b64,
                })
        except Exception as exc:
            summary_rows.append({
                'model': model_name,
                'model_id': model_id,
                'status': 'unavailable',
                'reason': str(exc),
                'precision_valid': None,
                'recall_valid': None,
                'f1_valid': None,
                'pr_auc': None,
                'accuracy': None,
                'balanced_accuracy': None,
                'test_samples': int(len(y_test)),
                'train_samples': int(len(y_train)),
            })
            confusion[model_id] = {'status': 'unavailable', 'reason': str(exc)}

    # ── Multiclass training (if requested) ──────────────────────────────────
    multiclass_summary_rows: list[dict[str, Any]] = []
    multiclass_confusion: dict[str, Any] = {}
    multiclass_class_metrics_rows: list[dict[str, Any]] = []
    crossing_metrics_rows: list[dict[str, Any]] = []
    combined_decision_rows: list[dict[str, Any]] = []
    best_combined_thresholds_rows: list[dict[str, Any]] = []
    multiclass_radius_rows: list[dict[str, Any]] = []
    multiclass_error_review_rows: list[dict[str, Any]] = []
    combined_error_review_rows: list[dict[str, Any]] = []

    if req.multiclass_model:
        y_all_multiclass = np.asarray([int(item.get('label_multiclass') or 0) for item in items], dtype=np.int32)
        if len(set(y_all_multiclass.tolist())) < 2:
            multiclass_summary_rows.append({
                'model': 'N/A',
                'model_id': 'N/A',
                'status': 'unavailable',
                'reason': 'Se necesitan al menos 2 clases distintas para entrenar modelo multiclase.',
            })
        else:
            train_idx_mc, test_idx_mc = _loco_group_split(items, req.test_size, req.random_seed, multiclass=True)
            X_train_mc = X_all[train_idx_mc]
            y_train_mc = y_all_multiclass[train_idx_mc]
            X_test_mc = X_all[test_idx_mc]
            y_test_mc = y_all_multiclass[test_idx_mc]
            test_items_mc = [items[i] for i in test_idx_mc]
            test_radii_mc = np.asarray([_float_or_nan(item.get('radius_for_group')) for item in test_items_mc], dtype=np.float32)
            finite_radii_mc = test_radii_mc[np.isfinite(test_radii_mc)]
            q1_mc, q2_mc = (np.nan, np.nan)
            if finite_radii_mc.size >= 3:
                q1_mc, q2_mc = np.quantile(finite_radii_mc, [1 / 3, 2 / 3])

            for model_id in req.models:
                model_name = model_labels.get(model_id, model_id)
                try:
                    model_mc = _loco_model_instance(model_id, int(req.random_seed), multiclass=True)
                    model_mc.fit(X_train_mc, y_train_mc)
                    if hasattr(model_mc, 'predict_proba'):
                        y_prob_mc = np.asarray(model_mc.predict_proba(X_test_mc))
                    else:
                        y_prob_mc = np.asarray(model_mc.predict(X_test_mc), dtype=np.float32)
                    y_pred_mc = np.argmax(y_prob_mc, axis=1)

                    cm = confusion_matrix(y_test_mc, y_pred_mc, labels=[0, 1, 2])
                    report = classification_report(y_test_mc, y_pred_mc, labels=[0, 1, 2], output_dict=True, zero_division=0)

                    multiclass_summary_rows.append({
                        'model': model_name,
                        'model_id': model_id,
                        'status': 'ok',
                        'accuracy': float(report.get('accuracy', 0)),
                        'precision_valid': float(report.get('0', {}).get('precision', 0)),
                        'recall_valid': float(report.get('0', {}).get('recall', 0)),
                        'f1_valid': float(report.get('0', {}).get('f1-score', 0)),
                        'precision_crossing': float(report.get('1', {}).get('precision', 0)),
                        'recall_crossing': float(report.get('1', {}).get('recall', 0)),
                        'f1_crossing': float(report.get('1', {}).get('f1-score', 0)),
                        'precision_other': float(report.get('2', {}).get('precision', 0)),
                        'recall_other': float(report.get('2', {}).get('recall', 0)),
                        'f1_other': float(report.get('2', {}).get('f1-score', 0)),
                        'test_samples': int(len(y_test_mc)),
                        'train_samples': int(len(y_train_mc)),
                    })
                    multiclass_confusion[model_id] = cm.tolist()
                    joblib.dump(model_mc, run_root / 'models' / f'{model_id}_multiclass_model.pkl')

                    # ── 1. Per-class metrics in long format ──────────────────────────
                    for cls_id, cls_name in [(0, 'valid'), (1, 'invalid_crossing'), (2, 'invalid_other')]:
                        cls_report = report.get(str(cls_id), {})
                        multiclass_class_metrics_rows.append({
                            'model': model_name,
                            'model_id': model_id,
                            'class': cls_name,
                            'class_id': cls_id,
                            'precision': float(cls_report.get('precision', 0)),
                            'recall': float(cls_report.get('recall', 0)),
                            'f1': float(cls_report.get('f1-score', 0)),
                            'support': int(cls_report.get('support', 0)),
                        })

                    # ── 2. Crossing rejection metrics ────────────────────────────────
                    crossing_mask = y_test_mc == 1
                    crossing_total = int(np.sum(crossing_mask))
                    crossing_rejected = 0
                    crossing_accepted_as_valid = 0
                    if crossing_total > 0:
                        crossing_prob = y_prob_mc[crossing_mask, 1]  # prob of crossing
                        crossing_pred = y_pred_mc[crossing_mask]
                        crossing_rejected = int(np.sum(crossing_pred != 0))  # not predicted as valid
                        crossing_accepted_as_valid = int(np.sum(crossing_pred == 0))  # predicted as valid
                    crossing_false_accept_rate = (crossing_accepted_as_valid / crossing_total) if crossing_total > 0 else float('nan')
                    crossing_rejection_rate = (crossing_rejected / crossing_total) if crossing_total > 0 else float('nan')
                    crossing_metrics_rows.append({
                        'model': model_name,
                        'model_id': model_id,
                        'crossing_total': crossing_total,
                        'crossing_rejected': crossing_rejected,
                        'crossing_accepted_as_valid': crossing_accepted_as_valid,
                        'crossing_false_accept_rate': round(crossing_false_accept_rate, 4) if not np.isnan(crossing_false_accept_rate) else None,
                        'crossing_rejection_rate': round(crossing_rejection_rate, 4) if not np.isnan(crossing_rejection_rate) else None,
                    })

                    # ── 3. Combined decision threshold grid ──────────────────────────
                    # Use binary model scores for combined decision
                    binary_model_path = run_root / 'models' / f'{model_id}_model.pkl'
                    has_binary_model = binary_model_path.exists()
                    if has_binary_model:
                        try:
                            binary_model = joblib.load(binary_model_path)
                            if hasattr(binary_model, 'predict_proba'):
                                y_prob_binary = np.asarray(binary_model.predict_proba(X_test_mc))[:, 1]
                            else:
                                y_prob_binary = np.asarray(binary_model.predict(X_test_mc), dtype=np.float32)
                        except Exception:
                            y_prob_binary = None
                    else:
                        y_prob_binary = None

                    valid_thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
                    crossing_thresholds = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50]

                    if y_prob_binary is not None:
                        for vt in valid_thresholds:
                            for ct in crossing_thresholds:
                                # Combined decision: binary valid_score >= vt AND crossing_prob <= ct
                                crossing_prob_col = y_prob_mc[:, 1]  # probability of crossing class
                                combined_valid = (y_prob_binary >= vt) & (crossing_prob_col <= ct)
                                combined_pred = combined_valid.astype(np.int32)
                                # Evaluate: treat real valid (0) as positive, real crossing/other (1,2) as negative
                                combined_real = (y_test_mc == 0).astype(np.int32)
                                tp = int(np.sum((combined_pred == 1) & (combined_real == 1)))
                                fp = int(np.sum((combined_pred == 1) & (combined_real == 0)))
                                fn = int(np.sum((combined_pred == 0) & (combined_real == 1)))
                                tn = int(np.sum((combined_pred == 0) & (combined_real == 0)))
                                precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
                                recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
                                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float('nan')
                                accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else float('nan')

                                # Crossing-specific: how many real crossing (1) are accepted as valid
                                real_crossing = y_test_mc == 1
                                crossing_accepted = int(np.sum(combined_pred[real_crossing] == 1)) if np.any(real_crossing) else 0
                                crossing_total_local = int(np.sum(real_crossing))
                                crossing_accept_rate = crossing_accepted / crossing_total_local if crossing_total_local > 0 else float('nan')

                                combined_decision_rows.append({
                                    'model': model_name,
                                    'model_id': model_id,
                                    'valid_threshold': vt,
                                    'crossing_threshold': ct,
                                    'precision_valid': round(precision, 4) if not np.isnan(precision) else None,
                                    'recall_valid': round(recall, 4) if not np.isnan(recall) else None,
                                    'f1_valid': round(f1, 4) if not np.isnan(f1) else None,
                                    'accuracy': round(accuracy, 4) if not np.isnan(accuracy) else None,
                                    'tp': tp,
                                    'fp': fp,
                                    'fn': fn,
                                    'tn': tn,
                                    'crossing_accepted_as_valid': crossing_accepted,
                                    'crossing_total': crossing_total_local,
                                    'crossing_accept_rate': round(crossing_accept_rate, 4) if not np.isnan(crossing_accept_rate) else None,
                                })

                        # ── 4. Best combined thresholds ──────────────────────────────
                        if combined_decision_rows:
                            model_combos = [r for r in combined_decision_rows if r['model_id'] == model_id]
                            if model_combos:
                                by_precision = sorted(model_combos, key=lambda r: float(r['precision_valid'] or 0), reverse=True)
                                by_crossing = sorted(model_combos, key=lambda r: float(r['crossing_accept_rate'] or 1))  # lower is better
                                by_f1 = sorted(model_combos, key=lambda r: float(r['f1_valid'] or 0), reverse=True)
                                best_combined_thresholds_rows.append({
                                    'model': model_name,
                                    'model_id': model_id,
                                    'criterion': 'best_by_precision_valid',
                                    'valid_threshold': by_precision[0]['valid_threshold'],
                                    'crossing_threshold': by_precision[0]['crossing_threshold'],
                                    'precision_valid': by_precision[0]['precision_valid'],
                                    'recall_valid': by_precision[0]['recall_valid'],
                                    'f1_valid': by_precision[0]['f1_valid'],
                                    'crossing_accept_rate': by_precision[0]['crossing_accept_rate'],
                                })
                                best_combined_thresholds_rows.append({
                                    'model': model_name,
                                    'model_id': model_id,
                                    'criterion': 'best_by_crossing_rejection',
                                    'valid_threshold': by_crossing[0]['valid_threshold'],
                                    'crossing_threshold': by_crossing[0]['crossing_threshold'],
                                    'precision_valid': by_crossing[0]['precision_valid'],
                                    'recall_valid': by_crossing[0]['recall_valid'],
                                    'f1_valid': by_crossing[0]['f1_valid'],
                                    'crossing_accept_rate': by_crossing[0]['crossing_accept_rate'],
                                })
                                best_combined_thresholds_rows.append({
                                    'model': model_name,
                                    'model_id': model_id,
                                    'criterion': 'best_by_f1_valid',
                                    'valid_threshold': by_f1[0]['valid_threshold'],
                                    'crossing_threshold': by_f1[0]['crossing_threshold'],
                                    'precision_valid': by_f1[0]['precision_valid'],
                                    'recall_valid': by_f1[0]['recall_valid'],
                                    'f1_valid': by_f1[0]['f1_valid'],
                                    'crossing_accept_rate': by_f1[0]['crossing_accept_rate'],
                                })

                    # ── 5. Multiclass radius group metrics ───────────────────────────
                    if np.isfinite(q1_mc) and np.isfinite(q2_mc):
                        group_names_mc = []
                        for r in test_radii_mc:
                            if not np.isfinite(r):
                                group_names_mc.append('unknown')
                            elif r <= q1_mc:
                                group_names_mc.append('small')
                            elif r <= q2_mc:
                                group_names_mc.append('medium')
                            else:
                                group_names_mc.append('large')
                        for group in ['small', 'medium', 'large']:
                            mask = np.asarray([g == group for g in group_names_mc], dtype=bool)
                            if not np.any(mask):
                                continue
                            y_group = y_test_mc[mask]
                            y_pred_group = y_pred_mc[mask]
                            for cls_id, cls_name in [(0, 'valid'), (1, 'invalid_crossing'), (2, 'invalid_other')]:
                                cls_real = (y_group == cls_id).astype(np.int32)
                                cls_pred = (y_pred_group == cls_id).astype(np.int32)
                                tp_g = int(np.sum((cls_pred == 1) & (cls_real == 1)))
                                fp_g = int(np.sum((cls_pred == 1) & (cls_real == 0)))
                                fn_g = int(np.sum((cls_pred == 0) & (cls_real == 1)))
                                precision_g = tp_g / (tp_g + fp_g) if (tp_g + fp_g) > 0 else float('nan')
                                recall_g = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else float('nan')
                                f1_g = 2 * precision_g * recall_g / (precision_g + recall_g) if (precision_g + recall_g) > 0 else float('nan')
                                multiclass_radius_rows.append({
                                    'model': model_name,
                                    'model_id': model_id,
                                    'radius_group': group,
                                    'class': cls_name,
                                    'class_id': cls_id,
                                    'n_samples': int(np.sum(mask)),
                                    'precision': round(precision_g, 4) if not np.isnan(precision_g) else None,
                                    'recall': round(recall_g, 4) if not np.isnan(recall_g) else None,
                                    'f1': round(f1_g, 4) if not np.isnan(f1_g) else None,
                                })

                    # ── 6. Multiclass error review ───────────────────────────────────
                    for local_idx, item in enumerate(test_items_mc):
                        real_mc = int(y_test_mc[local_idx])
                        pred_mc = int(y_pred_mc[local_idx])
                        if real_mc == pred_mc:
                            continue
                        cls_names = {0: 'valid', 1: 'invalid_crossing', 2: 'invalid_other'}
                        err_type = f"{cls_names.get(real_mc, '?')}_predicted_as_{cls_names.get(pred_mc, '?')}"
                        try:
                            patch_b64 = _patch_b64_from_path(item['patch_path'])
                        except Exception:
                            patch_b64 = ''
                        multiclass_error_review_rows.append({
                            'model': model_name,
                            'model_id': model_id,
                            'error_type': err_type,
                            'candidate_id': item.get('candidate_id') or '',
                            'item_id': item.get('item_id') or '',
                            'dataset_kind': item.get('dataset_kind') or '',
                            'label_real': real_mc,
                            'label_real_name': cls_names.get(real_mc, '?'),
                            'prediction': pred_mc,
                            'prediction_name': cls_names.get(pred_mc, '?'),
                            'prob_valid': round(float(y_prob_mc[local_idx, 0]), 4),
                            'prob_crossing': round(float(y_prob_mc[local_idx, 1]), 4),
                            'prob_other': round(float(y_prob_mc[local_idx, 2]), 4),
                            'radius_px': None if not np.isfinite(test_radii_mc[local_idx]) else float(test_radii_mc[local_idx]),
                            'patch_path': str(item.get('patch_rel') or ''),
                            'patch_b64': patch_b64,
                        })

                    # ── 7. Combined decision error review ───────────────────────────
                    if y_prob_binary is not None:
                        # Use default thresholds for combined error review
                        default_vt = 0.5
                        default_ct = 0.5
                        crossing_prob_col = y_prob_mc[:, 1]
                        combined_valid_dec = (y_prob_binary >= default_vt) & (crossing_prob_col <= default_ct)
                        combined_pred_dec = combined_valid_dec.astype(np.int32)
                        combined_real_dec = (y_test_mc == 0).astype(np.int32)
                        for local_idx, item in enumerate(test_items_mc):
                            real_dec = int(combined_real_dec[local_idx])
                            pred_dec = int(combined_pred_dec[local_idx])
                            if real_dec == pred_dec:
                                continue
                            real_mc = int(y_test_mc[local_idx])
                            real_mc_name = cls_names.get(real_mc, '?')
                            rejection_reason = ''
                            if pred_dec == 0 and real_dec == 1:
                                # Real valid rejected
                                if y_prob_binary[local_idx] < default_vt:
                                    rejection_reason = f'binary_score_below_{default_vt}'
                                elif y_prob_mc[local_idx, 1] > default_ct:
                                    rejection_reason = f'crossing_prob_above_{default_ct}'
                                else:
                                    rejection_reason = 'combined_rejection'
                            elif pred_dec == 1 and real_dec == 0:
                                # Non-valid accepted
                                if real_mc == 1:
                                    rejection_reason = 'crossing_accepted_as_valid'
                                else:
                                    rejection_reason = 'other_accepted_as_valid'
                            try:
                                patch_b64 = _patch_b64_from_path(item['patch_path'])
                            except Exception:
                                patch_b64 = ''
                            combined_error_review_rows.append({
                                'model': model_name,
                                'model_id': model_id,
                                'error_type': 'False Positives' if real_dec == 0 and pred_dec == 1 else 'False Negatives',
                                'error_subtype': rejection_reason,
                                'candidate_id': item.get('candidate_id') or '',
                                'item_id': item.get('item_id') or '',
                                'dataset_kind': item.get('dataset_kind') or '',
                                'label_real_binary': real_dec,
                                'label_real_multiclass': real_mc,
                                'label_real_multiclass_name': real_mc_name,
                                'prediction_binary': pred_dec,
                                'binary_valid_score': round(float(y_prob_binary[local_idx]), 4),
                                'multiclass_prob_valid': round(float(y_prob_mc[local_idx, 0]), 4),
                                'multiclass_prob_crossing': round(float(y_prob_mc[local_idx, 1]), 4),
                                'multiclass_prob_other': round(float(y_prob_mc[local_idx, 2]), 4),
                                'radius_px': None if not np.isfinite(test_radii_mc[local_idx]) else float(test_radii_mc[local_idx]),
                                'patch_path': str(item.get('patch_rel') or ''),
                                'patch_b64': patch_b64,
                            })

                except Exception as exc:
                    multiclass_summary_rows.append({
                        'model': model_name,
                        'model_id': model_id,
                        'status': 'unavailable',
                        'reason': str(exc),
                    })
                    multiclass_confusion[model_id] = {'status': 'unavailable', 'reason': str(exc)}

    metric_fieldnames = ['model', 'model_id', 'status', 'reason', 'precision_valid', 'recall_valid', 'f1_valid', 'pr_auc', 'accuracy', 'balanced_accuracy', 'train_samples', 'test_samples', 'tn', 'fp', 'fn', 'tp']
    with (run_root / 'metrics_summary.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=metric_fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, '') for k in metric_fieldnames})
    threshold_fieldnames = ['model', 'model_id', 'threshold', 'precision_valid', 'recall_valid', 'f1_valid', 'accuracy', 'balanced_accuracy', 'tn', 'fp', 'fn', 'tp']
    with (run_root / 'threshold_metrics.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=threshold_fieldnames)
        writer.writeheader()
        for row in threshold_rows:
            writer.writerow({k: row.get(k, '') for k in threshold_fieldnames})
    radius_fieldnames = ['model', 'model_id', 'radius_group', 'n_samples', 'precision_valid', 'recall_valid', 'f1_valid', 'accuracy', 'balanced_accuracy', 'fp', 'fn']
    with (run_root / 'radius_group_metrics.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=radius_fieldnames)
        writer.writeheader()
        for row in radius_rows:
            writer.writerow({k: row.get(k, '') for k in radius_fieldnames})
    error_fieldnames = ['model', 'model_id', 'error_type', 'candidate_id', 'item_id', 'dataset_kind', 'label_real', 'prediction', 'probability_valid', 'radius_px', 'patch_path']
    with (run_root / 'error_review.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=error_fieldnames)
        writer.writeheader()
        for row in error_review:
            writer.writerow({k: row.get(k, '') for k in error_fieldnames})
    # Save multiclass confusion matrices if available
    if multiclass_confusion:
        (run_root / 'multiclass_confusion_matrices.json').write_text(json.dumps(multiclass_confusion, ensure_ascii=False, indent=2), encoding='utf-8')
    # Save multiclass summary CSV if available
    if multiclass_summary_rows:
        mc_fieldnames = ['model', 'model_id', 'status', 'reason', 'accuracy', 'precision_valid', 'recall_valid', 'f1_valid', 'precision_crossing', 'recall_crossing', 'f1_crossing', 'precision_other', 'recall_other', 'f1_other', 'test_samples', 'train_samples']
        with (run_root / 'multiclass_metrics_summary.csv').open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=mc_fieldnames)
            writer.writeheader()
            for row in multiclass_summary_rows:
                writer.writerow({k: row.get(k, '') for k in mc_fieldnames})

    # ── Save new multiclass metrics files ─────────────────────────────────
    # 1. Confusion matrix 3x3 (already saved as multiclass_confusion_matrices.json)
    # 2. Per-class metrics in long format
    if multiclass_class_metrics_rows:
        mc_class_fieldnames = ['model', 'model_id', 'class', 'class_id', 'precision', 'recall', 'f1', 'support']
        with (run_root / 'multiclass_class_metrics.csv').open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=mc_class_fieldnames)
            writer.writeheader()
            for row in multiclass_class_metrics_rows:
                writer.writerow({k: row.get(k, '') for k in mc_class_fieldnames})
    # 3. Crossing rejection metrics
    if crossing_metrics_rows:
        crossing_fieldnames = ['model', 'model_id', 'crossing_total', 'crossing_rejected', 'crossing_accepted_as_valid', 'crossing_false_accept_rate', 'crossing_rejection_rate']
        with (run_root / 'crossing_metrics.csv').open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=crossing_fieldnames)
            writer.writeheader()
            for row in crossing_metrics_rows:
                writer.writerow({k: row.get(k, '') for k in crossing_fieldnames})
    # 4. Combined decision thresholds
    if combined_decision_rows:
        combined_fieldnames = ['model', 'model_id', 'valid_threshold', 'crossing_threshold', 'precision_valid', 'recall_valid', 'f1_valid', 'accuracy', 'tp', 'fp', 'fn', 'tn', 'crossing_accepted_as_valid', 'crossing_total', 'crossing_accept_rate']
        with (run_root / 'combined_decision_thresholds.csv').open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=combined_fieldnames)
            writer.writeheader()
            for row in combined_decision_rows:
                writer.writerow({k: row.get(k, '') for k in combined_fieldnames})
    # 5. Best combined thresholds
    if best_combined_thresholds_rows:
        best_combined_fieldnames = ['model', 'model_id', 'criterion', 'valid_threshold', 'crossing_threshold', 'precision_valid', 'recall_valid', 'f1_valid', 'crossing_accept_rate']
        with (run_root / 'best_combined_thresholds.csv').open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=best_combined_fieldnames)
            writer.writeheader()
            for row in best_combined_thresholds_rows:
                writer.writerow({k: row.get(k, '') for k in best_combined_fieldnames})
    # 6. Multiclass radius group metrics
    if multiclass_radius_rows:
        mc_radius_fieldnames = ['model', 'model_id', 'radius_group', 'class', 'class_id', 'n_samples', 'precision', 'recall', 'f1']
        with (run_root / 'multiclass_radius_group_metrics.csv').open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=mc_radius_fieldnames)
            writer.writeheader()
            for row in multiclass_radius_rows:
                writer.writerow({k: row.get(k, '') for k in mc_radius_fieldnames})
    # 7a. Multiclass error review
    if multiclass_error_review_rows:
        mc_err_fieldnames = ['model', 'model_id', 'error_type', 'candidate_id', 'item_id', 'dataset_kind', 'label_real', 'label_real_name', 'prediction', 'prediction_name', 'prob_valid', 'prob_crossing', 'prob_other', 'radius_px', 'patch_path']
        with (run_root / 'error_review_multiclass.csv').open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=mc_err_fieldnames)
            writer.writeheader()
            for row in multiclass_error_review_rows:
                writer.writerow({k: row.get(k, '') for k in mc_err_fieldnames})
    # 7b. Combined decision error review
    if combined_error_review_rows:
        combined_err_fieldnames = ['model', 'model_id', 'error_type', 'error_subtype', 'candidate_id', 'item_id', 'dataset_kind', 'label_real_binary', 'label_real_multiclass', 'label_real_multiclass_name', 'prediction_binary', 'binary_valid_score', 'multiclass_prob_valid', 'multiclass_prob_crossing', 'multiclass_prob_other', 'radius_px', 'patch_path']
        with (run_root / 'error_review_combined.csv').open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=combined_err_fieldnames)
            writer.writeheader()
            for row in combined_error_review_rows:
                writer.writerow({k: row.get(k, '') for k in combined_err_fieldnames})

    (run_root / 'confusion_matrices.json').write_text(json.dumps(confusion, ensure_ascii=False, indent=2), encoding='utf-8')
    meta = {
        'run_id': run_id,
        'created_at': created_at,
        'data_selection': selection,
        'test_size': float(req.test_size),
        'random_seed': int(req.random_seed),
        'feature_count': int(X_all.shape[1]),
        'pixel_feature_count': _loco_pixel_feature_count(vector_config),
        'tabular_features': _loco_tabular_feature_names(vector_config),
        'pixel_mode': str(vector_config.get('pixel_mode')),
        'circle_prune_px': int(vector_config.get('circle_prune_px') or 0),
        'patch_size': int(vector_config.get('patch_size') or 64),
        'uses_patch_zoom_factor': bool(vector_config.get('uses_patch_zoom_factor')),
        'uses_source_radius_px': bool(vector_config.get('uses_source_radius_px')),
        'vector_config': {
            **vector_config,
            'pixel_feature_count': _loco_pixel_feature_count(vector_config),
            'feature_names': _loco_tabular_feature_names(vector_config),
            'feature_order': _loco_vector_feature_order(vector_config),
            'total_feature_count': int(X_all.shape[1]),
            'radio_px_semantics': 'source_radius_px' if bool(vector_config.get('uses_source_radius_px')) else 'stored_radio_px',
        },
        'sample_count': len(items),
        'train_count': len(train_idx),
        'test_count': len(test_idx),
        'group_count': len({item.get('group_id') for item in items}),
        'positive_class': 'valid',
        'models': list(req.models),
        'multiclass_model': bool(req.multiclass_model),
        'has_multiclass': bool(multiclass_summary_rows and any(r.get('status') == 'ok' for r in multiclass_summary_rows)),
    }
    (run_root / 'run_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return {
        'ok': True,
        'run_id': run_id,
        'run_dir': str(run_root),
        'meta': meta,
        'metrics_summary': summary_rows,
        'threshold_metrics': threshold_rows,
        'radius_group_metrics': radius_rows,
        'confusion_matrices': confusion,
        'error_review': error_review[:400],
        'model_recommendations': model_payloads,
        'multiclass_metrics_summary': multiclass_summary_rows,
        'multiclass_confusion_matrices': multiclass_confusion,
        'multiclass_class_metrics': multiclass_class_metrics_rows,
        'crossing_metrics': crossing_metrics_rows,
        'combined_decision_thresholds': combined_decision_rows,
        'best_combined_thresholds': best_combined_thresholds_rows,
        'multiclass_radius_group_metrics': multiclass_radius_rows,
        'error_review_multiclass': multiclass_error_review_rows[:400],
        'error_review_combined': combined_error_review_rows[:400],
    }


@router.get('/loco-training/runs')
def loco_training_runs() -> dict[str, Any]:
    return {'ok': True, 'items': _list_loco_training_runs()}


@router.post('/loco-training/test-circles')
def loco_training_test_circles(req: LocoTestCircleReq) -> dict[str, Any]:
    sess, image_id, _labels, _prior, support, prior_run_id = _loco_lab_context(req)
    candidates = [_candidate_to_dict(c) for c in (req.candidates or [])]
    if not candidates:
        raise HTTPException(status_code=400, detail='Dibuja al menos un circulo para testear.')
    for cand in candidates:
        label = str(cand.get('label') or '')
        if label not in {'valid', 'invalid', 'invalid_crossing', 'invalid_other'}:
            raise HTTPException(status_code=400, detail=f"Candidato con etiqueta desconocida: {cand.get('candidate_id') or '-'}")
        if float(cand.get('radius_px') or 0.0) < 1.0:
            raise HTTPException(status_code=400, detail=f"Candidato con radio invalido: {cand.get('candidate_id') or '-'}")
    model_id = str(req.model_id or 'extratrees')
    resolved_run_id, model, model_meta = _load_training_model(req.training_run_id, model_id)
    vector_config = _loco_vector_config_from_meta(model_meta)
    threshold = float(np.clip(float(req.threshold or 0.5), 0.01, 0.99))
    patch_size = int(vector_config.get('patch_size') or 64)

    # Try to load multiclass model if available
    multiclass_model = None
    try:
        _, multiclass_model, _ = _load_training_model(req.training_run_id, model_id, model_kind='multiclass')
    except HTTPException:
        pass

    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for cand in candidates:
        cx = float(cand.get('center_x') or 0.0)
        cy = float(cand.get('center_y') or 0.0)
        radius = float(cand.get('radius_px') or 0.0)
        label = str(cand.get('label') or '')
        patch, _area_ratio = _circle_disk_patch(support, (cx, cy), radius, patch_size=patch_size)
        feature_item = _loco_dataset_features_for_candidate(support, cand, {'patch_size': patch_size, 'circle_samples': 128, 'require_four_cuts': False})
        features = dict(feature_item.get('features') or {})
        diagnostics = dict(feature_item.get('diagnostics') or {})
        vector_item = {'source_radius_px': radius, 'radius_for_group': radius, 'radius_px': radius}
        vectors.append(_loco_vector_from_patch_features((patch > 0).astype(np.float32), features, diagnostics, item=vector_item, vector_config=vector_config))
        # Normalize label for backward compatibility
        norm_label = _loco_normalize_label(label)
        label_binary, label_multiclass = _loco_label_maps(norm_label)
        rows.append({
            'candidate_id': str(cand.get('candidate_id') or ''),
            'center_x': cx,
            'center_y': cy,
            'radius_px': radius,
            'label': norm_label,
            'label_numeric': label_binary,
            'label_multiclass': label_multiclass,
            'features': features,
            'diagnostics': diagnostics,
            'patch_b64': _gray_png_b64(patch),
        })
    X = np.vstack(vectors).astype(np.float32)
    y_true_binary = np.asarray([int(row['label_numeric']) for row in rows], dtype=np.int32)
    y_prob = _loco_model_predict_valid_scores(model, vectors)
    y_pred = (y_prob >= threshold).astype(np.int32)

    # Multiclass prediction
    multiclass_predictions: list[dict[str, float]] | None = None
    if multiclass_model is not None:
        if hasattr(multiclass_model, 'predict_proba'):
            mc_probs = np.asarray(multiclass_model.predict_proba(X))
        else:
            mc_probs_raw = np.asarray(multiclass_model.predict(X), dtype=np.float32)
            mc_probs = np.zeros((len(mc_probs_raw), 3), dtype=np.float32)
            for i in range(len(mc_probs_raw)):
                cls = int(mc_probs_raw[i])
                if 0 <= cls < 3:
                    mc_probs[i, cls] = 1.0
        multiclass_predictions = [
            {
                'prob_valid': float(mc_probs[i, 0]),
                'prob_crossing': float(mc_probs[i, 1]),
                'prob_other': float(mc_probs[i, 2]),
                'predicted_class': int(np.argmax(mc_probs[i])),
            }
            for i in range(len(rows))
        ]

    predictions: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        pred_label = 'valid' if int(y_pred[idx]) == 1 else 'invalid'
        real_label = str(row['label'])
        error_type = ''
        if real_label in ('invalid_crossing', 'invalid_other') and pred_label == 'valid':
            error_type = 'False Positives'
        elif real_label == 'valid' and pred_label == 'invalid':
            error_type = 'False Negatives'
        pred_entry = {
            **row,
            'prediction': pred_label,
            'prediction_numeric': int(y_pred[idx]),
            'probability_valid': float(y_prob[idx]),
            'correct': bool(int(y_pred[idx]) == int(y_true_binary[idx])),
            'error_type': error_type,
        }
        if multiclass_predictions is not None:
            pred_entry['multiclass'] = multiclass_predictions[idx]
        predictions.append(pred_entry)
    metrics = _positive_metrics(y_true_binary, y_prob, threshold) if len(set(y_true_binary.tolist())) >= 2 else {
        'precision_valid': None,
        'recall_valid': None,
        'f1_valid': None,
        'accuracy': float(accuracy_score(y_true_binary, y_pred)),
        'balanced_accuracy': None,
        'tn': int(np.sum((y_true_binary == 0) & (y_pred == 0))),
        'fp': int(np.sum((y_true_binary == 0) & (y_pred == 1))),
        'fn': int(np.sum((y_true_binary == 1) & (y_pred == 0))),
        'tp': int(np.sum((y_true_binary == 1) & (y_pred == 1))),
    }
    return {
        'ok': True,
        'image_id': image_id,
        'prior_run_id': prior_run_id,
        'training_run_id': resolved_run_id,
        'model_id': model_id,
        'threshold': threshold,
        'has_multiclass': multiclass_model is not None,
        'vector_config': _loco_json_safe({
            **vector_config,
            'pixel_feature_count': _loco_pixel_feature_count(vector_config),
            'tabular_features': _loco_tabular_feature_names(vector_config),
            'feature_count': int(X.shape[1]),
        }),
        'candidate_count': len(predictions),
        'predictions': predictions,
        'metrics': metrics,
    }


@router.post('/loco-models/detect-circles')
def loco_models_detect_circles(req: LocoModelDetectReq) -> dict[str, Any]:
    sess, image_id, _labels, _prior, support, prior_run_id = _loco_lab_context(req)
    model_id = str(req.model_id or 'extratrees')
    resolved_run_id, model, model_meta = _load_training_model(req.model_run_id, model_id)
    vector_config = _loco_vector_config_from_meta(model_meta)

    # Try to load multiclass model if available
    multiclass_model = None
    try:
        _, multiclass_model, _ = _load_training_model(req.model_run_id, model_id, model_kind='multiclass')
    except HTTPException:
        pass

    h, w = support.shape[:2]
    step = int(np.clip(int(req.grid_step or 10), 2, 128))
    rmin = float(np.clip(float(req.min_radius or 1.0), 1.0, 512.0))
    rmax = float(np.clip(float(req.max_radius or rmin), rmin, 512.0))
    rstep = float(np.clip(float(req.radius_step or 1.0), 0.5, max(0.5, rmax)))
    patch_size = int(vector_config.get('patch_size') or np.clip(int(req.patch_size or 64), 16, 256))
    max_candidates = int(np.clip(int(req.max_candidates or 8000), 1, 500000))
    max_return_rejected = int(np.clip(int(req.max_return_rejected or 800), 0, 8000))
    radii = [float(r) for r in np.arange(rmin, rmax + 1e-6, rstep)]
    if not radii:
        radii = [rmin]

    params = {
        'patch_size': patch_size,
        'circle_samples': 128,
        'require_four_cuts': False,
        'radius_max_px': rmax,
        'mask_required_ratio': 0.0,
        'min_score': 0.0,
        'max_intersections': 256,
        'min_component_bridge_score': 0.0,
    }
    support_u8 = (support > 0).astype(np.uint8)
    candidate_max_per_tile = int(np.clip(int(req.candidate_max_per_tile or 0), 0, 60000))
    specs, total_candidates = _loco_candidate_specs(
        support_u8,
        radii,
        step=step,
        max_candidates=max_candidates,
        mode=str(req.candidate_sampling_mode or 'row_major'),
        seed=int(req.candidate_random_seed or 42),
        tile_size_px=int(req.tile_size_px or 128),
        max_per_tile=candidate_max_per_tile,
    )
    discarded_empty = 0
    vectors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    cheap_rejected: list[dict[str, Any]] = []
    candidate_seq = 0
    for x, y, radius in specs:
        cid = f'auto_{candidate_seq:06d}'
        candidate_seq += 1
        patch, area_ratio = _circle_disk_patch(support_u8, (float(x), float(y)), radius, patch_size=patch_size)
        if area_ratio <= 0.001 or int(np.sum(patch > 0)) <= 0:
            discarded_empty += 1
            if req.return_rejected and len(cheap_rejected) < max_return_rejected:
                cheap_rejected.append({
                    'candidate_id': cid,
                    'center_x': float(x),
                    'center_y': float(y),
                    'center_xy': [float(x), float(y)],
                    'radius_px': float(radius),
                    'valid_score': 0.0,
                    'threshold_used': float(req.threshold),
                    'radius_group': _loco_model_radius_group(radius, req.small_radius_limit, req.large_radius_limit),
                    'status': 'rejected',
                    'reason': 'empty_mask',
                    'features': {},
                    'diagnostics': {},
                })
            continue
        cand = {
            'candidate_id': cid,
            'center_x': float(x),
            'center_y': float(y),
            'radius_px': float(radius),
            'label': '',
        }
        feature_item = _loco_dataset_features_for_candidate(support_u8, cand, params)
        features = dict(feature_item.get('features') or {})
        diagnostics = dict(feature_item.get('diagnostics') or {})
        vector_item = {'source_radius_px': float(radius), 'radius_for_group': float(radius), 'radius_px': float(radius)}
        vectors.append(_loco_vector_from_patch_features((patch > 0).astype(np.float32), features, diagnostics, item=vector_item, vector_config=vector_config))
        threshold_used, radius_group = _loco_model_threshold(req, radius)
        rows.append({
            'candidate_id': cid,
            'center_x': float(x),
            'center_y': float(y),
            'center_xy': [float(x), float(y)],
            'radius_px': float(radius),
            'threshold_used': threshold_used,
            'radius_group': radius_group,
            'status': 'pending',
            'reason': '',
            'features': features,
            'diagnostics': diagnostics,
        })

    scores = _loco_model_predict_valid_scores(model, vectors)

    # Multiclass scores (if multiclass model available)
    multiclass_scores: list[dict[str, float]] | None = None
    if multiclass_model is not None:
        if hasattr(multiclass_model, 'predict_proba'):
            mc_probs = np.asarray(multiclass_model.predict_proba(np.vstack(vectors).astype(np.float32)))
        else:
            mc_probs_raw = np.asarray(multiclass_model.predict(np.vstack(vectors).astype(np.float32)), dtype=np.float32)
            mc_probs = np.zeros((len(mc_probs_raw), 3), dtype=np.float32)
            for i in range(len(mc_probs_raw)):
                cls = int(mc_probs_raw[i])
                if 0 <= cls < 3:
                    mc_probs[i, cls] = 1.0
        multiclass_scores = [
            {
                'prob_valid': float(mc_probs[i, 0]),
                'prob_crossing': float(mc_probs[i, 1]),
                'prob_other': float(mc_probs[i, 2]),
                'predicted_class': int(np.argmax(mc_probs[i])),
            }
            for i in range(len(rows))
        ]

    crossing_threshold = float(np.clip(float(req.crossing_threshold or 0.5), 0.0, 1.0))
    accepted_before_nms: list[dict[str, Any]] = []
    rejected_threshold: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        score = float(scores[idx]) if idx < scores.size else 0.0
        item = dict(row)
        item['valid_score'] = score

        # Combined decision rule:
        #   binary score >= threshold AND (no multiclass OR crossing_prob <= crossing_threshold)
        passes_binary = score >= float(item.get('threshold_used') or req.threshold)
        if multiclass_scores is not None and idx < len(multiclass_scores):
            item['multiclass'] = multiclass_scores[idx]
            passes_multiclass = multiclass_scores[idx]['prob_crossing'] <= crossing_threshold
        else:
            item['multiclass'] = None
            passes_multiclass = True

        if passes_binary and passes_multiclass:
            item['status'] = 'accepted'
            item['reason'] = 'accepted'
            accepted_before_nms.append(item)
        else:
            reasons = []
            if not passes_binary:
                reasons.append('below_threshold')
            if not passes_multiclass:
                reasons.append('crossing_detected')
            item['status'] = 'rejected'
            item['reason'] = '|'.join(reasons)
            rejected_threshold.append(item)

    if req.use_nms:
        accepted_after_nms, removed_by_nms = _loco_circle_nms(
            accepted_before_nms,
            nms_mode=str(req.nms_mode or 'distance_radius'),
            nms_distance_factor=float(np.clip(float(req.nms_distance_factor or 0.5), 0.05, 10.0)),
            radius_similarity_factor=float(np.clip(float(req.radius_similarity_factor or 0.4), 0.0, 10.0)),
            circle_iou_threshold=float(np.clip(float(req.circle_iou_threshold or 0.4), 0.0, 1.0)),
        )
    else:
        accepted_after_nms = accepted_before_nms
        removed_by_nms = []

    # --- Spatial final filter (after NMS) ---
    removed_by_spatial: list[dict[str, Any]] = []
    spatial_tile_stats: dict[str, Any] = {}
    if req.use_spatial_final_filter and accepted_after_nms:
        accepted_after_spatial, removed_by_spatial, spatial_result = _loco_spatial_final_filter(
            accepted_after_nms,
            h, w,
            tile_px=int(req.spatial_final_tile_px or 128),
            max_per_tile=int(req.spatial_final_max_per_tile or 3),
            min_center_distance_factor=float(req.spatial_final_min_center_distance_factor or 1.0),
        )
        spatial_tile_stats = spatial_result.get('tiles', {})
        accepted_final = accepted_after_spatial
    else:
        accepted_final = accepted_after_nms

    rejected_all = [*rejected_threshold, *removed_by_nms, *removed_by_spatial, *cheap_rejected]
    if req.return_rejected:
        if max_return_rejected <= 0:
            # Return all rejected (no limit)
            returned_rejected = list(rejected_all)
        else:
            returned_rejected = _sample_rejected_balanced(
                rejected_all,
                max_return=max_return_rejected,
                tile_size_px=int(req.tile_size_px or 128),
                seed=int(req.candidate_random_seed or 42),
            )
    else:
        returned_rejected = []
    summary = _loco_model_summary(
        total_candidates=total_candidates,
        sampled_candidates=len(specs),
        discarded_empty=discarded_empty,
        evaluated_candidates=len(rows),
        accepted_before_nms=len(accepted_before_nms),
        accepted_after_nms=len(accepted_after_nms),
        accepted_after_spatial=len(accepted_final),
        rejected_by_threshold=len(rejected_threshold),
        removed_by_nms=len(removed_by_nms),
        removed_by_spatial=len(removed_by_spatial),
        scores=scores,
        accepted=accepted_final,
    )
    if spatial_tile_stats:
        summary['spatial_tiles'] = spatial_tile_stats

    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    run_id = f"loco_model_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.sha1(f'{image_id}|{resolved_run_id}|{model_id}|{created_at}'.encode()).hexdigest()[:8]}"
    run_root = drp.OUTPUT_ROOT / 'model_inference_runs' / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    all_model_rows = [*accepted_before_nms, *rejected_threshold, *removed_by_nms, *removed_by_spatial, *cheap_rejected]
    _write_loco_model_csv(run_root / 'candidates_all.csv', all_model_rows)
    _write_loco_model_csv(run_root / 'candidates_accepted.csv', accepted_before_nms)
    _write_loco_model_csv(run_root / 'candidates_after_nms.csv', accepted_after_nms)
    _write_loco_model_csv(run_root / 'candidates_after_spatial.csv', accepted_final)
    _write_loco_model_csv(run_root / 'candidates_rejected.csv', rejected_all)
    overlay = _loco_model_overlay(sess.image_rgb, accepted_final, returned_rejected)
    _write_rgb_png(run_root / 'overlay_preview.png', overlay)
    run_meta = {
        'run_id': run_id,
        'created_at': created_at,
        'image_id': image_id,
        'image_name': str(getattr(sess, 'image_name', '') or ''),
        'prior_run_id': prior_run_id,
        'requested_prior_run_id': str(req.prior_run_id or ''),
        'model_run_id': resolved_run_id,
        'model_id': model_id,
        'vector_config': _loco_json_safe({
            **vector_config,
            'pixel_feature_count': _loco_pixel_feature_count(vector_config),
            'tabular_features': _loco_tabular_feature_names(vector_config),
        }),
        'has_multiclass': multiclass_model is not None,
        'crossing_threshold': crossing_threshold,
        'thresholds': {
            'threshold': float(req.threshold),
            'use_radius_thresholds': bool(req.use_radius_thresholds),
            'small_threshold': float(req.small_threshold),
            'medium_threshold': float(req.medium_threshold),
            'large_threshold': float(req.large_threshold),
            'small_radius_limit': float(req.small_radius_limit),
            'large_radius_limit': float(req.large_radius_limit),
        },
        'grid_step': step,
        'candidate_sampling': {
            'mode': str(req.candidate_sampling_mode),
            'random_seed': int(req.candidate_random_seed),
            'tile_size_px': int(req.tile_size_px),
            'max_per_tile': candidate_max_per_tile,
            'pool_count': int(total_candidates),
            'sampled_count': int(len(specs)),
        },
        'radii': {'min_radius': rmin, 'max_radius': rmax, 'radius_step': rstep, 'values': radii},
        'nms_params': {
            'use_nms': bool(req.use_nms),
            'nms_mode': str(req.nms_mode),
            'nms_distance_factor': float(req.nms_distance_factor),
            'radius_similarity_factor': float(req.radius_similarity_factor),
            'circle_iou_threshold': float(req.circle_iou_threshold),
        },
        'spatial_filter_params': {
            'use_spatial_final_filter': bool(req.use_spatial_final_filter),
            'spatial_final_tile_px': int(req.spatial_final_tile_px or 128),
            'spatial_final_max_per_tile': int(req.spatial_final_max_per_tile or 3),
            'spatial_final_min_center_distance_factor': float(req.spatial_final_min_center_distance_factor or 1.0),
        },
        'summary': summary,
    }
    (run_root / 'run_meta.json').write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding='utf-8')
    overlay_b64, overlay_mime = encode_display_b64(overlay)
    return {
        'ok': True,
        'run_id': run_id,
        'run_dir': str(run_root),
        'image_id': image_id,
        'prior_run_id': prior_run_id,
        'model_run_id': resolved_run_id,
        'model_id': model_id,
        'has_multiclass': multiclass_model is not None,
        'crossing_threshold': crossing_threshold,
        'summary': _loco_json_safe(summary),
        'accepted': _loco_json_safe(accepted_final),
        'rejected': _loco_json_safe(returned_rejected),
        'overlay_b64': overlay_b64,
        'overlay_mime': overlay_mime,
    }


@router.post('/loco-models/measure-accepted')
def loco_models_measure_accepted(req: LocoModelMeasureReq) -> dict[str, Any]:
    sess, image_id, labels, prior, support, prior_run_id = _loco_lab_context(req)
    params = dict(req.params or {})
    raw_candidates = list(req.candidates or [])
    if not raw_candidates:
        raise HTTPException(status_code=400, detail='No hay circulos aceptados para medir.')
    proposals: list[dict[str, Any]] = []
    for idx, cand in enumerate(raw_candidates):
        cx = float(cand.get('center_x', cand.get('x', 0.0)) or 0.0)
        cy = float(cand.get('center_y', cand.get('y', 0.0)) or 0.0)
        radius = float(cand.get('radius_px') or 0.0)
        if radius <= 0:
            continue
        score = float(cand.get('valid_score', cand.get('score', 0.0)) or 0.0)
        proposals.append({
            'proposal_id': str(cand.get('candidate_id') or cand.get('proposal_id') or f'model_{idx:06d}'),
            'candidate_id': str(cand.get('candidate_id') or cand.get('proposal_id') or f'model_{idx:06d}'),
            'method': 'loco_model_circle_detector',
            'center_xy': [cx, cy],
            'center_x': cx,
            'center_y': cy,
            'radius_px': radius,
            'score': score,
            'valid_score': score,
            'status': 'accepted',
            'reason': 'accepted_by_model',
        })
    if not proposals:
        raise HTTPException(status_code=400, detail='No hay circulos aceptados validos para medir.')
    limit = _loco_int_param(params, 'measure_limit', 300, lo=1, hi=5000)
    measure_params = {
        **params,
        'circle_samples': int(params.get('circle_samples') or 128),
        'require_four_cuts': False,
        'min_score': 0.0,
        'min_component_bridge_score': 0.0,
        'max_intersections': int(params.get('max_intersections') or 256),
    }
    measurements = [_measure_loco_circle_square(support, p, measure_params) for p in proposals[:limit]]
    summary = _loco_lab_summary(proposals, measurements)

    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    method_id = 'loco_model_circle_detector'
    run_id = drp.new_run_id(method_id)
    overlay = _loco_lab_overlay(sess.image_rgb, proposals, measurements)
    results: list[dict[str, Any]] = []
    for idx, m in enumerate(measurements):
        center = m.get('circle_xy') or [0.0, 0.0]
        proposal_id = str(m.get('proposal_id') or '')
        proposal = next((p for p in proposals if str(p.get('proposal_id')) == proposal_id), {})
        results.append({
            'point_index': idx,
            'x': float(center[0]) if len(center) >= 2 else 0.0,
            'y': float(center[1]) if len(center) >= 2 else 0.0,
            'method_id': method_id,
            'route': 'loco_model_circle_square',
            'measurement_mode': 'model_circle_square',
            'status': str(m.get('status') or 'rejected'),
            'quality': str(m.get('quality_label') or 'rejected'),
            'quality_label': str(m.get('quality_label') or 'rejected'),
            'reason': str(m.get('reason') or ''),
            'diameter_px': m.get('diameter_px'),
            'confidence': float(m.get('confidence') or m.get('score') or proposal.get('valid_score') or 0.0),
            'stability': 1.0 if str(m.get('status') or '') == 'ok' else 0.0,
            'loco_best_radius_px': m.get('radius_px'),
            'loco_symmetry_score': m.get('symmetry_score'),
            'loco_intersection_count': m.get('intersection_count'),
            'left_edge_xy': m.get('left_edge_xy'),
            'right_edge_xy': m.get('right_edge_xy'),
            'quadrilateral_vertices': m.get('quadrilateral_vertices'),
            'proposal_id': proposal_id,
            'valid_score': proposal.get('valid_score'),
        })
    meta = {
        'run_id': run_id,
        'image_id': image_id,
        'created_at': created_at,
        'method_id': method_id,
        'prior_run_id': prior_run_id,
        'requested_prior_run_id': str(req.prior_run_id or ''),
        'points_ok': int(summary.get('measurement_ok_count') or 0),
        'proposal_count': int(summary.get('proposal_count') or 0),
        'accepted_count': int(summary.get('accepted_count') or 0),
        'params': params,
    }
    art = drp.DiameterRunArtifacts(
        run_id=run_id,
        image_id=image_id,
        experiment_id=method_id,
        created_at=created_at,
        input_image=sess.image_rgb,
        scribble_labels=labels,
        prior_prob=np.asarray(prior, dtype=np.float32),
        support_region=(support > 0).astype(np.uint8),
        overlay=overlay,
        results=results,
        diagnostics={
            'loco_model_detector': {'proposals': proposals, 'measurements': measurements, 'summary': summary},
            'loco_overlay': overlay,
        },
        meta=meta,
    )
    save_meta = drp.save_diameter_run(art)
    item = drp.load_diameter_run(run_id)
    payload = _run_payload(item)
    payload['meta']['save'] = save_meta
    return payload


# ── Calibration ──────────────────────────────────────────────────────────

CALIBRATION_DIR = Path(__file__).resolve().parents[2] / 'data' / 'calibration'


def _calibration_safe_id(text: str) -> str:
    raw = str(text or '').strip()
    out: list[str] = []
    for ch in raw:
        if ch.isalnum() or ch in {'_', '-'}:
            out.append(ch)
        else:
            out.append('_')
    return ''.join(out).strip('_')


class CalibrationSaveReq(BaseModel):
    image_id: str
    enabled: bool
    unit: str
    known_value_px: float
    pixel_distance: float
    nm_per_px: float


class CalibrationDeleteReq(BaseModel):
    image_id: str


@router.post('/calibration/save')
def calibration_save(req: CalibrationSaveReq) -> dict[str, Any]:
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    path = CALIBRATION_DIR / f'{_calibration_safe_id(req.image_id)}.json'
    data = {
        'image_id': req.image_id,
        'enabled': req.enabled,
        'unit': req.unit,
        'known_value_px': req.known_value_px,
        'pixel_distance': req.pixel_distance,
        'nm_per_px': req.nm_per_px,
    }
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return {'ok': True, 'image_id': req.image_id}


@router.get('/calibration/load')
def calibration_load(image_id: str) -> dict[str, Any]:
    if not str(image_id or '').strip():
        raise HTTPException(status_code=400, detail='image_id requerido.')
    path = CALIBRATION_DIR / f'{_calibration_safe_id(image_id)}.json'
    if not path.exists():
        return {'ok': True, 'image_id': image_id, 'calibration': None}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Error al leer calibracion: {exc}') from exc
    return {'ok': True, 'image_id': image_id, 'calibration': data}


@router.post('/calibration/delete')
def calibration_delete(req: CalibrationDeleteReq) -> dict[str, Any]:
    path = CALIBRATION_DIR / f'{_calibration_safe_id(req.image_id)}.json'
    if path.exists():
        path.unlink()
    return {'ok': True, 'image_id': req.image_id}


@router.get('/results/list')
def results_list(image_id: str) -> dict[str, Any]:
    if not str(image_id or '').strip():
        raise HTTPException(status_code=400, detail='image_id requerido.')
    return {'ok': True, 'image_id': image_id, 'items': drp.list_diameter_runs(image_id)}


@router.get('/results/get')
def results_get(run_id: str) -> dict[str, Any]:
    if not str(run_id or '').strip():
        raise HTTPException(status_code=400, detail='run_id requerido.')
    try:
        item = drp.load_diameter_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _run_payload(item)


@router.get('/reports/export')
def reports_export(image_id: str) -> dict[str, Any]:
    if not str(image_id or '').strip():
        raise HTTPException(status_code=400, detail='image_id requerido.')
    try:
        info = export_diameter_report(image_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'No se pudo exportar reporte Diameter Research: {exc}') from exc
    return {'ok': True, **info}


@router.post('/validation/case/upsert')
def validation_case_upsert(req: ValidationCaseReq) -> dict[str, Any]:
    sess = _require_active_image(req.session_id, req.image_id)
    case = upsert_case(
        {
            'case_id': req.case_id,
            'image_id': req.image_id,
            'image_name': str(getattr(sess, 'image_name', '') or ''),
            'point': {'x': float(req.point.x), 'y': float(req.point.y)},
            'category': req.category,
            'quality_manual': req.quality_manual,
            'manual_diameter_px': req.manual_diameter_px,
            'manual_left_x': req.manual_left_x,
            'manual_left_y': req.manual_left_y,
            'manual_right_x': req.manual_right_x,
            'manual_right_y': req.manual_right_y,
            'measurement_decision': req.measurement_decision,
            'notes': req.notes,
            'result_comment': req.result_comment,
            'source_mode': req.source_mode,
            'prior_run_id': req.prior_run_id,
            'params': dict(req.params or {}),
        }
    )
    return {'ok': True, 'case': case}


@router.get('/validation/cases')
def validation_cases(image_id: str) -> dict[str, Any]:
    if not str(image_id or '').strip():
        raise HTTPException(status_code=400, detail='image_id requerido.')
    return {'ok': True, 'image_id': image_id, 'items': list_cases(image_id)}


@router.post('/validation/run-case')
def validation_run_case(req: ValidationRunCaseReq) -> dict[str, Any]:
    _require_active_image(req.session_id, req.image_id)
    case_rows = [c for c in list_cases(req.image_id) if str(c.get('case_id') or '') == str(req.case_id or '')]
    if not case_rows:
        raise HTTPException(status_code=404, detail=f'Caso no encontrado: {req.case_id}')
    case = dict(case_rows[0])
    point = dict(case.get('point') or {})
    valid_methods = {METHOD_ID, METHOD_ID_V2, *V3_METHOD_IDS}
    methods = [str(m) for m in req.methods if str(m) in valid_methods]
    if not methods:
        methods = [
            METHOD_ID,
            METHOD_ID_V2,
            METHOD_ID_V3_1,
            METHOD_ID_V3_2_AUTO,
            METHOD_ID_V3_2_SMALL_MASK,
            METHOD_ID_V3_2_LARGE_IMAGE,
            METHOD_ID_CIRCLE_SQUARE,
            METHOD_ID_MANUAL_DUAL_SIDE,
            METHOD_ID_MANUAL_LINE_DIRECT,
            METHOD_ID_ELLIPSE_FIT,
            METHOD_ID_LOCO,
        ]

    outputs: dict[str, Any] = {}
    for method_id in methods:
        run_payload = run(
            RunReq(
                session_id=req.session_id,
                image_id=req.image_id,
                method_id=method_id,  # type: ignore[arg-type]
                source_mode=req.source_mode,
                prior_run_id=req.prior_run_id or str(case.get('prior_run_id') or ''),
                points=[PointItem(x=float(point.get('x', 0.0)), y=float(point.get('y', 0.0)))],
                active_only=False,
                params=dict(req.params or case.get('params') or {}),
                scribble_map_b64=req.scribble_map_b64,
            )
        )
        case = attach_run(req.case_id, run_payload, params=dict(req.params or case.get('params') or {}), source_mode=req.source_mode)
        outputs[method_id] = run_payload
    return {'ok': True, 'case': case, 'runs': outputs}


@router.get('/validation/export')
def validation_export(image_id: str) -> dict[str, Any]:
    if not str(image_id or '').strip():
        raise HTTPException(status_code=400, detail='image_id requerido.')
    info = export_validation(image_id)
    return {'ok': True, **info}
