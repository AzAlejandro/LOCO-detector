from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import secrets
import shutil
from typing import Any, Literal

import cv2
import joblib
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from .features import build_features
from .image_codec import apply_mask_overlay, encode_display_b64, encode_gray_png_b64, to_uint8_rgb
from .library_store import list_library_images, load_library_image, load_library_thumbnail, normalize_structured_tags
from .persistence import OUTPUT_ROOT, load_scribble_draft, load_run, save_run
from .runner import RunArtifacts, compute_image_id, new_run_id, now_str
from .scribble import labels_to_visual, scribble_label_counts
from .session_store import store


router = APIRouter(prefix='/api/assist-models', tags=['assist-models'])

MODELS_ROOT = OUTPUT_ROOT / 'assist_models'
MODELS_DIR = MODELS_ROOT / 'models'
REGISTRY_PATH = MODELS_ROOT / 'registry.json'

LABEL_NAMES = {
    1: 'fiber',
    2: 'halo',
    3: 'background',
}


class TrainModelReq(BaseModel):
    session_id: str = ''
    model_name: str = ''
    image_ids: list[str] = Field(default_factory=list)
    class_mode: Literal['multiclass', 'binary'] = 'multiclass'
    classifier: Literal['extratrees', 'rf', 'catboost', 'xgboost'] = 'extratrees'
    feature_variant: Literal['base', 'context'] = 'context'
    n_estimators: int = 120
    max_samples_per_class: int = 20000
    max_samples_per_image_class: int = 6000
    notes: str = ''
    origin_overrides: dict[str, str] = Field(default_factory=dict)


class PredictModelReq(BaseModel):
    session_id: str
    image_id: str
    model_id: str = ''
    min_confidence: float = 0.72
    fiber_confidence: float | None = None
    halo_confidence: float | None = None
    background_confidence: float | None = None
    include_fiber: bool = True
    include_halo: bool = True
    include_background: bool = False


class PredictMaskReq(BaseModel):
    session_id: str
    image_id: str
    model_id: str = ''
    min_confidence: float = 0.72
    fiber_confidence: float | None = None
    halo_confidence: float | None = None
    background_confidence: float | None = None
    include_fiber: bool = True
    include_halo: bool = True
    include_background: bool = False


class ModelIdReq(BaseModel):
    model_id: str


class ModelMetaUpdateReq(BaseModel):
    model_id: str
    model_name: str = ''
    model_tags: list[dict[str, Any]] = Field(default_factory=list)
    notes: str = ''


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _safe_id(text: str) -> str:
    raw = str(text or '').strip().lower()
    out: list[str] = []
    for ch in raw:
        if ch.isalnum() or ch in {'_', '-'}:
            out.append(ch)
        elif ch in {' ', '.', '/'}:
            out.append('_')
    sid = ''.join(out).strip('_')
    return sid or 'assist_model'


def _ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _model_dir(model_id: str) -> Path:
    mid = _safe_id(model_id)
    if not mid:
        raise ValueError('model_id invalido')
    return MODELS_DIR / mid


def _load_registry() -> dict[str, Any]:
    _ensure_dirs()
    if not REGISTRY_PATH.exists():
        return {'default_model_id': '', 'models': []}
    try:
        payload = dict(json.loads(REGISTRY_PATH.read_text(encoding='utf-8')) or {})
    except Exception:
        payload = {'default_model_id': '', 'models': []}
    payload.setdefault('default_model_id', '')
    payload.setdefault('models', [])
    return payload


def _save_registry(payload: dict[str, Any]) -> None:
    _ensure_dirs()
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _read_meta(model_id: str) -> dict[str, Any]:
    mpath = _model_dir(model_id) / 'meta.json'
    if not mpath.exists():
        raise FileNotFoundError(f'Modelo no encontrado: {model_id}')
    return dict(json.loads(mpath.read_text(encoding='utf-8')) or {})


def _write_meta(model_id: str, meta: dict[str, Any]) -> None:
    mpath = _model_dir(model_id) / 'meta.json'
    if not mpath.exists():
        raise FileNotFoundError(f'Modelo no encontrado: {model_id}')
    mpath.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')


def _dedupe_structured_tags(items: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tag in normalize_structured_tags(items):
        key = json.dumps(tag, sort_keys=True, ensure_ascii=False).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return out


def _model_training_param_tags(meta: dict[str, Any]) -> list[dict[str, Any]]:
    payload = meta or {}
    params = payload.get('params') or {}
    classifier_labels = {
        'extratrees': 'ExtraTrees',
        'rf': 'RandomForest',
        'catboost': 'CatBoost',
        'xgboost': 'XGBoost',
    }
    class_labels = {
        'multiclass': 'fibra / halo / background',
        'binary': 'fibra vs resto',
    }
    tags: list[dict[str, Any]] = []

    def add(label: str) -> None:
        value = str(label or '').strip()
        if value:
            tags.append({'category': 'other', 'label': value})

    class_mode = str(payload.get('class_mode') or '').strip()
    if class_mode:
        add(f"Clases: {class_labels.get(class_mode, class_mode)}")
    classifier = str(payload.get('classifier') or '').strip()
    if classifier:
        add(f"Modelo: {classifier_labels.get(classifier, classifier)}")
    feature_variant = str(payload.get('feature_variant') or '').strip()
    if feature_variant:
        add(f"Features: {feature_variant}")
    if params.get('n_estimators') is not None:
        add(f"Arboles: {params.get('n_estimators')}")
    if params.get('max_samples_per_class') is not None:
        add(f"Max/clase: {params.get('max_samples_per_class')}")
    if params.get('max_samples_per_image_class') is not None:
        add(f"Max/imagen/clase: {params.get('max_samples_per_image_class')}")
    if payload.get('image_count') is not None:
        add(f"Imagenes: {payload.get('image_count')}")
    if payload.get('train_samples') is not None:
        add(f"Muestras: {payload.get('train_samples')}")
    add(f"Tuning: {'si' if bool(payload.get('is_tuning_trial')) else 'no'}")
    return _dedupe_structured_tags(tags)


def _model_image_tags(meta: dict[str, Any]) -> list[dict[str, Any]]:
    image_ids = {str(x or '').strip() for x in (meta.get('image_ids') or []) if str(x or '').strip()}
    tags: list[dict[str, Any]] = []
    for row in meta.get('images') or []:
        if not isinstance(row, dict):
            continue
        if row.get('structured_tags') or row.get('tags'):
            tags.extend(normalize_structured_tags(row.get('structured_tags') or row.get('tags') or []))
        iid = str(row.get('image_id') or '').strip()
        if iid:
            image_ids.add(iid)
    if image_ids:
        for image in list_library_images():
            if str(image.get('image_id') or '') not in image_ids:
                continue
            tags.extend(normalize_structured_tags(image.get('structured_tags') or image.get('tags') or []))
    return _dedupe_structured_tags(tags)


def _registry_item(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        'model_id': str(meta.get('model_id') or ''),
        'model_name': str(meta.get('model_name') or ''),
        'created_at': str(meta.get('created_at') or ''),
        'class_mode': str(meta.get('class_mode') or ''),
        'classes': list(meta.get('classes') or []),
        'feature_variant': str(meta.get('feature_variant') or ''),
        'classifier': str(meta.get('classifier') or ''),
        'image_count': int(meta.get('image_count') or 0),
        'train_samples': int(meta.get('train_samples') or 0),
        'metrics': dict(meta.get('metrics') or {}),
        'notes': str(meta.get('notes') or ''),
        'image_ids': [str(x or '') for x in (meta.get('image_ids') or []) if str(x or '')],
        'image_tags': _model_image_tags(meta),
        'model_tags': _dedupe_structured_tags(meta.get('model_tags') or []),
        'auto_model_tags': _model_training_param_tags(meta),
    }


def _sync_registry(default_model_id: str | None = None) -> dict[str, Any]:
    _ensure_dirs()
    current = _load_registry()
    models: list[dict[str, Any]] = []
    for d in sorted(MODELS_DIR.glob('*')):
        if not d.is_dir():
            continue
        mpath = d / 'meta.json'
        if not mpath.exists():
            continue
        try:
            models.append(_registry_item(dict(json.loads(mpath.read_text(encoding='utf-8')) or {})))
        except Exception:
            continue
    models.sort(key=lambda x: str(x.get('created_at') or ''), reverse=True)
    default_id = str(default_model_id if default_model_id is not None else current.get('default_model_id') or '').strip()
    valid_ids = {str(x.get('model_id') or '') for x in models}
    if default_id not in valid_ids:
        default_id = str(models[0].get('model_id') or '') if models else ''
    payload = {'default_model_id': default_id, 'models': models}
    _save_registry(payload)
    return payload


def _require_session(session_id: str) -> Any:
    sess = store.get(str(session_id or '').strip())
    if sess is None:
        raise HTTPException(status_code=404, detail='Sesion no encontrada.')
    return sess


def _dataset_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in list_library_images():
        row = dict(item)
        counts = {
            'fiber': int(row.get('draft_n_fg') or 0),
            'halo': int(row.get('draft_n_halo') or 0),
            'background': int(row.get('draft_n_bg') or 0),
        }
        row['class_counts'] = counts
        row['trainable_multiclass'] = bool(counts['fiber'] > 0 and (counts['halo'] > 0 or counts['background'] > 0))
        row['trainable_binary'] = bool(counts['fiber'] > 0 and (counts['halo'] > 0 or counts['background'] > 0))
        row['thumbnail_b64'] = ''
        row['thumbnail_mime'] = 'image/png'
        row['scribble_thumb_b64'] = ''
        row['scribble_thumb_mime'] = 'image/png'
        try:
            thumb = load_library_thumbnail(str(row.get('image_id') or ''), max_px=144)
            row['thumbnail_b64'], row['thumbnail_mime'] = encode_display_b64(thumb)
        except Exception:
            pass
        # Load scribble thumbnail from the current scribble_origin
        try:
            current_origin = str(row.get('scribble_origin', '') or '')
            draft = load_scribble_draft(str(row.get('image_id') or ''), origin=current_origin)
            if bool(draft.get('found')):
                labels = np.asarray(draft.get('labels'), dtype=np.uint8)
                visual = labels_to_visual(labels)
                h, w = visual.shape[:2]
                scale = min(1.0, 144.0 / float(max(1, h, w)))
                if scale < 0.999:
                    visual = cv2.resize(
                        visual,
                        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                        interpolation=cv2.INTER_NEAREST,
                    )
                row['scribble_thumb_b64'], row['scribble_thumb_mime'] = encode_display_b64(visual)
        except Exception:
            pass
        rows.append(row)
    return rows


def _target_labels(labels: np.ndarray, class_mode: str) -> np.ndarray:
    lab = np.asarray(labels, dtype=np.uint8)
    out = np.zeros_like(lab, dtype=np.uint8)
    if str(class_mode) == 'binary':
        out[lab == 1] = 1
        out[(lab == 2) | (lab == 3)] = 3
        return out
    out[lab == 1] = 1
    out[lab == 2] = 2
    out[lab == 3] = 3
    return out


def _collect_training_data(req: TrainModelReq) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, int]]:
    image_ids = [str(x).strip() for x in req.image_ids if str(x).strip()]
    if not image_ids:
        raise HTTPException(status_code=400, detail='Selecciona al menos una imagen con scribbles.')

    rng = np.random.RandomState(42)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    image_rows: list[dict[str, Any]] = []
    total_counts = {'fiber': 0, 'halo': 0, 'background': 0, 'binary_negative': 0}
    per_img_limit = max(100, int(req.max_samples_per_image_class))

    # Build origin override map
    origin_overrides = {str(k).strip(): str(v).strip() for k, v in (req.origin_overrides or {}).items()}

    for image_id in image_ids:
        try:
            image, meta = load_library_image(image_id)
            origin = origin_overrides.get(image_id, '')
            draft = load_scribble_draft(image_id, origin=origin)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f'No se pudo cargar {image_id}: {exc}') from exc
        if not bool(draft.get('found')):
            continue
        labels = np.asarray(draft.get('labels'), dtype=np.uint8)
        if labels.shape[:2] != image.shape[:2]:
            labels = cv2.resize(labels, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        ymap = _target_labels(labels, req.class_mode)
        mask = ymap > 0
        if int(np.sum(mask)) < 2:
            continue
        classes = sorted(int(x) for x in np.unique(ymap[mask]).tolist())
        if 1 not in classes or len(classes) < 2:
            continue

        feats = build_features(image, variant=req.feature_variant)
        flat_x = feats.reshape(-1, feats.shape[-1])
        flat_y = ymap.reshape(-1)
        selected: list[np.ndarray] = []
        for cls in classes:
            idx = np.where(flat_y == cls)[0]
            if idx.size == 0:
                continue
            n = min(int(idx.size), per_img_limit)
            selected.append(rng.choice(idx, size=n, replace=False))
        if not selected:
            continue
        sel = np.concatenate(selected)
        xs.append(flat_x[sel])
        ys.append(flat_y[sel])
        counts = scribble_label_counts(labels)
        total_counts['fiber'] += int(counts['fiber'])
        total_counts['halo'] += int(counts['halo'])
        total_counts['background'] += int(counts['background'])
        total_counts['binary_negative'] += int(counts['halo'] + counts['background'])
        image_rows.append(
            {
                'image_id': image_id,
                'image_name': str(meta.get('image_name') or ''),
                'shape_hw': [int(image.shape[0]), int(image.shape[1])],
                'class_counts': counts,
                'sampled_px': int(sel.size),
            }
        )

    if not xs or not ys:
        raise HTTPException(status_code=400, detail='No hay imagenes seleccionadas con scribbles entrenables.')

    x_all = np.vstack(xs).astype(np.float32, copy=False)
    y_all = np.concatenate(ys).astype(np.uint8, copy=False)
    classes = sorted(int(x) for x in np.unique(y_all).tolist())
    if 1 not in classes or len(classes) < 2:
        raise HTTPException(status_code=400, detail='El dataset necesita fibra y al menos una clase negativa.')

    max_per_class = max(200, int(req.max_samples_per_class))
    selected_final: list[np.ndarray] = []
    for cls in classes:
        idx = np.where(y_all == cls)[0]
        n = min(int(idx.size), max_per_class)
        selected_final.append(rng.choice(idx, size=n, replace=False))
    sel_final = np.concatenate(selected_final)
    rng.shuffle(sel_final)
    return x_all[sel_final], y_all[sel_final], image_rows, total_counts


def _fit_classifier(req: TrainModelReq, x: np.ndarray, y: np.ndarray):
    kind = str(req.classifier or 'extratrees').lower()
    n_estimators = int(max(20, req.n_estimators))
    if kind == 'rf':
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced_subsample',
        )
    elif kind == 'catboost':
        loss = 'MultiClass' if str(req.class_mode) == 'multiclass' else 'Logloss'
        clf = CatBoostClassifier(
            iterations=n_estimators,
            depth=6,
            learning_rate=0.05,
            loss_function=loss,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
    elif kind == 'xgboost':
        objective = 'multi:softprob' if str(req.class_mode) == 'multiclass' else 'binary:logistic'
        num_class = int(len(np.unique(y))) if str(req.class_mode) == 'multiclass' else None
        clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.05,
            objective=objective,
            num_class=num_class,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        clf = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced',
        )
    clf.fit(x, y)
    return clf


def _train_metrics(clf: Any, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    classes, counts = np.unique(y, return_counts=True)
    metrics: dict[str, Any] = {
        'class_sample_counts': {LABEL_NAMES.get(int(c), str(int(c))): int(n) for c, n in zip(classes, counts)},
    }
    try:
        if y.size >= 50 and all(int(np.sum(y == cls)) >= 2 for cls in classes):
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.2,
                random_state=123,
                stratify=y,
            )
            shadow = clone(clf)
            shadow.fit(x_train, y_train)
            pred = shadow.predict(x_test)
            metrics['holdout_accuracy'] = float(accuracy_score(y_test, pred))
            metrics['holdout_samples'] = int(y_test.size)
    except Exception:
        pred = clf.predict(x)
        metrics['train_accuracy'] = float(accuracy_score(y, pred))
    if 'holdout_accuracy' not in metrics and 'train_accuracy' not in metrics:
        pred = clf.predict(x)
        metrics['train_accuracy'] = float(accuracy_score(y, pred))
    return metrics


def _overlay_suggestion(image_rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rgb = to_uint8_rgb(image_rgb)
    if rgb is None:
        raise ValueError('Imagen invalida.')
    lab = np.asarray(labels, dtype=np.uint8)
    if lab.shape[:2] != rgb.shape[:2]:
        lab = cv2.resize(lab, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = rgb.astype(np.float32)
    colors = {
        1: np.asarray([0, 229, 255], dtype=np.float32),
        2: np.asarray([255, 132, 0], dtype=np.float32),
        3: np.asarray([136, 75, 220], dtype=np.float32),
    }
    for cls, color in colors.items():
        mask = lab == cls
        if np.any(mask):
            out[mask] = out[mask] * 0.52 + color * 0.48
    return np.clip(out, 0, 255).astype(np.uint8)


def _class_probabilities(clf: Any, x_all: np.ndarray, shape_hw: tuple[int, int]) -> dict[int, np.ndarray]:
    proba = clf.predict_proba(x_all)
    classes = [int(c) for c in list(getattr(clf, 'classes_', []))]
    out: dict[int, np.ndarray] = {}
    if proba.ndim != 2 or not classes:
        pred = np.asarray(clf.predict(x_all)).reshape(-1)
        for cls in sorted(int(c) for c in np.unique(pred).tolist()):
            out[cls] = (pred == cls).astype(np.float32).reshape(shape_hw)
        return out
    for idx, cls in enumerate(classes):
        out[cls] = proba[:, idx].astype(np.float32).reshape(shape_hw)
    return out


def _prediction_thresholds(req: PredictModelReq | PredictMaskReq) -> dict[int, float]:
    fallback = float(np.clip(req.min_confidence, 0.001, 0.999))

    def one(value: float | None) -> float:
        if value is None:
            return fallback
        return float(np.clip(value, 0.001, 0.999))

    return {
        1: one(req.fiber_confidence),
        2: one(req.halo_confidence),
        3: one(req.background_confidence),
    }


def _prediction_suggestion(
    probs: dict[int, np.ndarray],
    allowed: set[int],
    thresholds: dict[int, float],
    shape_hw: tuple[int, int],
) -> np.ndarray:
    suggestion = np.zeros(shape_hw, dtype=np.uint8)
    if not probs:
        return suggestion
    valid_classes = [cls for cls in (1, 2, 3) if cls in allowed and cls in probs]
    if not valid_classes:
        return suggestion

    # Build one independent threshold mask per class. The only shared step is resolving
    # overlaps because the final scribble map can store a single class per pixel.
    margin_stack: list[np.ndarray] = []
    for cls in valid_classes:
        prob = np.asarray(probs[cls], dtype=np.float32)
        threshold = thresholds.get(cls, 0.72)
        margin_stack.append(np.where(prob >= threshold, prob - threshold, -1.0))
    margins = np.stack(margin_stack, axis=-1)
    best_idx = np.argmax(margins, axis=-1)
    best_margin = np.max(margins, axis=-1)
    for idx, cls in enumerate(valid_classes):
        suggestion[(best_idx == idx) & (best_margin >= 0.0)] = int(cls)
    return suggestion


def _prediction_probability_summary(
    probs: dict[int, np.ndarray],
    allowed: set[int],
    thresholds: dict[int, float],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    names = {1: 'fiber', 2: 'halo', 3: 'background'}
    for cls, name in names.items():
        if cls not in probs:
            continue
        arr = np.asarray(probs[cls], dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        threshold = thresholds.get(cls, 0.72)
        if finite.size == 0:
            out[name] = {
                'enabled': bool(cls in allowed),
                'threshold': float(threshold),
                'above_threshold': 0,
            }
            continue
        quantiles = np.quantile(finite, [0.5, 0.75, 0.9, 0.95, 0.99])
        out[name] = {
            'enabled': bool(cls in allowed),
            'threshold': float(threshold),
            'above_threshold': int(np.sum(arr >= threshold)) if cls in allowed else 0,
            'max': float(np.max(finite)),
            'p50': float(quantiles[0]),
            'p75': float(quantiles[1]),
            'p90': float(quantiles[2]),
            'p95': float(quantiles[3]),
            'p99': float(quantiles[4]),
        }
    return out


@router.get('/dataset/images')
def dataset_images(session_id: str = '') -> dict[str, Any]:
    if str(session_id or '').strip():
        _require_session(session_id)
    return {'ok': True, 'items': _dataset_rows()}


@router.get('/dataset/preview')
def dataset_preview(image_id: str, origin: str = '') -> dict[str, Any]:
    """Return image preview with scribble overlay for a specific origin.
    If origin is empty, uses the auto-detected origin (most recent draft)."""
    iid = str(image_id or '').strip()
    if not iid:
        raise HTTPException(status_code=400, detail='image_id requerido.')
    try:
        image, meta = load_library_image(iid)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'No se pudo cargar imagen: {exc}') from exc
    image_b64, image_mime = encode_display_b64(image)
    scribble_b64 = ''
    scribble_mime = 'image/png'
    counts: dict[str, int] = {}
    try:
        # Pass the origin parameter to load the specific origin's scribble
        draft = load_scribble_draft(iid, origin=origin)
        print(f"[DBG-PREVIEW] ENTER: image_id={iid}, origin_param={origin!r}, found={draft.get('found')}")
        if bool(draft.get('found')):
            labels = np.asarray(draft.get('labels'), dtype=np.uint8)
            if labels.shape[:2] != image.shape[:2]:
                labels = cv2.resize(labels, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            scribble_b64, scribble_mime = encode_display_b64(labels_to_visual(labels))
            counts = scribble_label_counts(labels)
    except Exception:
        scribble_b64 = ''
    return {
        'ok': True,
        'image_id': iid,
        'image_name': str(meta.get('image_name') or iid),
        'source_path': str(meta.get('source_path') or ''),
        'source_mtime': str(meta.get('source_mtime') or meta.get('updated_at') or ''),
        'image_b64': image_b64,
        'image_mime': image_mime,
        'scribble_b64': scribble_b64,
        'scribble_mime': scribble_mime,
        'counts': counts,
    }


@router.get('/list')
def list_models() -> dict[str, Any]:
    payload = _sync_registry()
    return {'ok': True, **payload}


@router.post('/train')
def train_model(req: TrainModelReq) -> dict[str, Any]:
    if str(req.session_id or '').strip():
        _require_session(req.session_id)
    x, y, image_rows, counts = _collect_training_data(req)
    clf = _fit_classifier(req, x, y)
    metrics = _train_metrics(clf, x, y)

    model_name = str(req.model_name or '').strip() or f'assist_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    token = secrets.token_hex(4)
    model_id = f'{_safe_id(model_name)}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{token}'
    out_dir = _model_dir(model_id)
    out_dir.mkdir(parents=True, exist_ok=False)
    model_path = out_dir / 'model.joblib'
    joblib.dump(clf, model_path)

    classes = [int(c) for c in list(getattr(clf, 'classes_', []))]
    meta = {
        'model_id': model_id,
        'model_name': model_name,
        'created_at': _now(),
        'version': 1,
        'class_mode': str(req.class_mode),
        'classes': [{'id': int(c), 'name': LABEL_NAMES.get(int(c), str(int(c)))} for c in classes],
        'classifier': str(req.classifier),
        'feature_variant': str(req.feature_variant),
        'params': {
            'n_estimators': int(req.n_estimators),
            'max_samples_per_class': int(req.max_samples_per_class),
            'max_samples_per_image_class': int(req.max_samples_per_image_class),
        },
        'image_ids': [str(row.get('image_id') or '') for row in image_rows],
        'images': image_rows,
        'image_count': int(len(image_rows)),
        'train_samples': int(x.shape[0]),
        'label_counts_source_px': counts,
        'metrics': metrics,
        'notes': str(req.notes or ''),
        'artifact_path': str(model_path),
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    registry = _sync_registry(default_model_id=model_id)
    return {'ok': True, 'model': _registry_item(meta), 'meta': meta, 'default_model_id': registry.get('default_model_id', '')}


@router.post('/set-default')
def set_default_model(req: ModelIdReq) -> dict[str, Any]:
    mid = str(req.model_id or '').strip()
    _read_meta(mid)
    payload = _sync_registry(default_model_id=mid)
    return {'ok': True, **payload}


@router.post('/update-meta')
def update_model_meta(req: ModelMetaUpdateReq) -> dict[str, Any]:
    mid = str(req.model_id or '').strip()
    if not mid:
        raise HTTPException(status_code=400, detail='model_id requerido.')
    try:
        meta = _read_meta(mid)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail='Modelo no encontrado.')
    name = str(req.model_name or '').strip()
    if name:
        meta['model_name'] = name
    meta['model_tags'] = _dedupe_structured_tags(req.model_tags or [])
    meta['notes'] = str(req.notes or '').strip()
    meta['updated_at'] = _now()
    _write_meta(mid, meta)
    payload = _sync_registry()
    item = next((m for m in payload.get('models', []) if str(m.get('model_id') or '') == mid), _registry_item(meta))
    return {'ok': True, 'model': item, **payload}


@router.post('/delete')
def delete_model(req: ModelIdReq) -> dict[str, Any]:
    mid = str(req.model_id or '').strip()
    if not mid:
        raise HTTPException(status_code=400, detail='model_id requerido.')
    d = _model_dir(mid)
    if not d.exists():
        raise HTTPException(status_code=404, detail='Modelo no encontrado.')
    shutil.rmtree(d)
    payload = _sync_registry()
    return {'ok': True, **payload}


@router.post('/predict')
def predict_model(req: PredictModelReq) -> dict[str, Any]:
    sess = _require_session(req.session_id)
    if sess.image_rgb is None:
        raise HTTPException(status_code=400, detail='Carga una imagen antes de predecir con modelo.')
    image_id = str(req.image_id or '').strip()
    if image_id != str(sess.image_id or '').strip():
        raise HTTPException(status_code=400, detail='image_id no corresponde a la imagen activa.')
    model_id = str(req.model_id or '').strip() or str(_sync_registry().get('default_model_id') or '')
    if not model_id:
        raise HTTPException(status_code=400, detail='Selecciona o entrena un modelo.')
    meta = _read_meta(model_id)
    model_path = _model_dir(model_id) / 'model.joblib'
    if not model_path.exists():
        raise HTTPException(status_code=404, detail='Archivo de modelo no encontrado.')
    clf = joblib.load(model_path)

    rgb = to_uint8_rgb(sess.image_rgb)
    feats = build_features(rgb, variant=str(meta.get('feature_variant') or 'context'))
    h, w, c = feats.shape
    probs = _class_probabilities(clf, feats.reshape(-1, c), (h, w))
    min_conf = float(np.clip(req.min_confidence, 0.001, 0.999))
    thresholds = _prediction_thresholds(req)
    allowed = set()
    if req.include_fiber:
        allowed.add(1)
    if req.include_halo:
        allowed.add(2)
    if req.include_background:
        allowed.add(3)
    suggestion = _prediction_suggestion(probs, allowed, thresholds, (h, w))

    preview = _overlay_suggestion(rgb, suggestion)
    maps_b64: dict[str, str] = {}
    for cls, name in ((1, 'fiber_prob'), (2, 'halo_prob'), (3, 'background_prob')):
        if cls in probs:
            maps_b64[name] = encode_gray_png_b64((np.clip(probs[cls], 0.0, 1.0) * 255.0).astype(np.uint8))
    preview_b64, preview_mime = encode_display_b64(preview)
    counts = scribble_label_counts(suggestion)
    probability_summary = _prediction_probability_summary(probs, allowed, thresholds)
    return {
        'ok': True,
        'model_id': model_id,
        'model_name': str(meta.get('model_name') or model_id),
        'image_id': image_id,
        'suggestion_b64': encode_gray_png_b64(labels_to_visual(suggestion)),
        'preview_b64': preview_b64,
        'preview_mime': preview_mime,
        'score_maps_b64': maps_b64,
        'counts': counts,
        'min_confidence': min_conf,
        'class_confidence': {
            'fiber': thresholds[1],
            'halo': thresholds[2],
            'background': thresholds[3],
        },
        'probability_summary': probability_summary,
        'conflict_policy': 'independent_threshold_max_margin',
        'meta': {
            'class_mode': str(meta.get('class_mode') or ''),
            'classes': list(meta.get('classes') or []),
            'feature_variant': str(meta.get('feature_variant') or ''),
            'classifier': str(meta.get('classifier') or ''),
        },
    }

@router.post('/predict-mask')
def predict_mask(req: PredictMaskReq) -> dict[str, Any]:
    """Predict a mask and save it as an experiment run visible in 'Revisión de resultados'."""
    sess = _require_session(req.session_id)
    if sess.image_rgb is None:
        raise HTTPException(status_code=400, detail='Carga una imagen antes de predecir.')
    image_id = str(req.image_id or '').strip()
    if image_id != str(sess.image_id or '').strip():
        raise HTTPException(status_code=400, detail='image_id no corresponde a la imagen activa.')
    model_id = str(req.model_id or '').strip() or str(_sync_registry().get('default_model_id') or '')
    if not model_id:
        raise HTTPException(status_code=400, detail='Selecciona o entrena un modelo.')
    meta = _read_meta(model_id)
    model_path = _model_dir(model_id) / 'model.joblib'
    if not model_path.exists():
        raise HTTPException(status_code=404, detail='Archivo de modelo no encontrado.')
    clf = joblib.load(model_path)

    rgb = to_uint8_rgb(sess.image_rgb)
    feats = build_features(rgb, variant=str(meta.get('feature_variant') or 'context'))
    h, w, c = feats.shape
    probs = _class_probabilities(clf, feats.reshape(-1, c), (h, w))
    min_conf = float(np.clip(req.min_confidence, 0.001, 0.999))
    thresholds = _prediction_thresholds(req)
    allowed = set()
    if req.include_fiber:
        allowed.add(1)
    if req.include_halo:
        allowed.add(2)
    if req.include_background:
        allowed.add(3)
    suggestion = _prediction_suggestion(probs, allowed, thresholds, (h, w))
    probability_summary = _prediction_probability_summary(probs, allowed, thresholds)

    # Build overlay from the suggestion mask
    overlay = apply_mask_overlay(rgb, suggestion, color=(0, 220, 80), alpha=0.35)

    # Build a RunArtifacts and save it as an experiment run
    run_id = new_run_id('assist_model_predict')
    art = RunArtifacts(
        run_id=run_id,
        image_id=image_id,
        experiment_id='assist_model_predict',
        created_at=now_str(),
        input_image=rgb,
        scribble_labels=np.zeros((h, w), dtype=np.uint8),
        prior_map=np.zeros((h, w), dtype=np.float32),
        mask=suggestion,
        overlay=overlay,
        meta={
            'model_id': model_id,
            'model_name': str(meta.get('model_name') or model_id),
            'classifier': str(meta.get('classifier') or ''),
            'class_mode': str(meta.get('class_mode') or ''),
            'min_confidence': min_conf,
            'class_confidence': {
                'fiber': thresholds[1],
                'halo': thresholds[2],
                'background': thresholds[3],
            },
            'probability_summary': probability_summary,
            'conflict_policy': 'independent_threshold_max_margin',
            'experiment_label': 'Prediccion de mascara',
            'run_status_level': 'success',
            # Include experiment metadata so save_run() writes group/display_name to the index.
            # This mirrors what _build_common_meta() does in runner.py for normal experiments.
            'experiment': {
                'experiment_id': 'assist_model_predict',
                'group': 'F',
                'display_name': 'Prediccion de mascara (modelo asistente)',
                'implementation_status': 'native',
                'requirements_hint': '',
            },
        },
        class_prob_maps={str(k): v for k, v in probs.items()},
    )
    save_run(art)

    # Load back and format as payload (inline _run_to_payload logic)
    run_item = load_run(run_id)
    labels_vis = labels_to_visual(np.asarray(run_item['scribble_map'], dtype=np.uint8))
    run_meta = dict(run_item.get('meta') or {})
    input_b64, input_mime = encode_display_b64(run_item['input_image'])
    overlay_b64, overlay_mime = encode_display_b64(run_item['overlay'])
    payload = {
        'run_id': run_item.get('run_id', ''),
        'image_id': run_item.get('image_id', ''),
        'experiment_id': run_item.get('experiment_id', ''),
        'created_at': run_item.get('created_at', ''),
        'run_status_level': str(run_meta.get('run_status_level', 'success')),
        'input_image_b64': input_b64,
        'input_image_mime': input_mime,
        'scribble_map_b64': encode_gray_png_b64(labels_vis),
        'prior_b64': encode_gray_png_b64((np.clip(np.asarray(run_item['prior_prob'], dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)),
        'mask_b64': encode_gray_png_b64((np.asarray(run_item['mask']) > 0).astype(np.uint8) * 255),
        'overlay_b64': overlay_b64,
        'overlay_mime': overlay_mime,
        'meta': run_meta,
    }
    return {'ok': True, 'run': payload}
