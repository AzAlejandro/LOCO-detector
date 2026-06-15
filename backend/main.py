from __future__ import annotations

import io
import json
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from .assist_models import router as assist_models_router
from .catalog import build_registry
from .diameter_research.api import router as diameter_research_router
from .image_codec import decode_png_b64, encode_display_b64, encode_gray_png_b64, to_gray_u8, to_uint8_rgb
from .library_store import (
    clear_prior_cache,
    delete_library_image,
    list_library_images,
    load_library_image,
    load_library_mask_thumbnail,
    load_library_thumbnail,
    register_library_image,
    save_prior_cache,
)
from .project_transfer import router as project_transfer_router
from .projects_api import router as projects_router
from .projects_store import get_active_project, list_projects, project_tags_for_image
from .persistence import (
    append_review,
    clear_results_for_image,
    clear_scribble_draft,
    delete_runs_for_image,
    export_report,
    list_reviews,
    list_runs,
    load_run,
    load_scribble_draft,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    save_run,
    save_scribble_draft,
)
from .runner import compute_image_id, run_experiment
from .scribble import decode_scribble_b64, has_fg_bg, labels_to_visual
from .session_store import store


app = FastAPI(title='scribble_research backend', version='0.2.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
app.include_router(diameter_research_router)
app.include_router(assist_models_router)
app.include_router(project_transfer_router)
app.include_router(projects_router)

registry = build_registry()
DEFAULT_MAX_RESOLUTION_PX = 900
UI_PREFS_PATH = OUTPUT_ROOT / 'ui_prefs.json'
LOCAL_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


class SessionReq(BaseModel):
    session_id: str | None = None


class ExcludeRectReq(BaseModel):
    x: float
    y: float
    w: float
    h: float


class RunReq(BaseModel):
    session_id: str
    experiment_id: str
    params: dict[str, Any] = Field(default_factory=dict)
    scribble_map_b64: str = ''
    gt_mask_b64: str = ''
    exclude_rect: ExcludeRectReq | None = None
    save_mode: str = 'overwrite'  # overwrite | append


class RunBatchReq(BaseModel):
    session_id: str
    experiment_ids: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    params_by_experiment: dict[str, dict[str, Any]] = Field(default_factory=dict)
    param_sweep: str = 'high'  # single | high
    scribble_map_b64: str = ''
    gt_mask_b64: str = ''
    exclude_rect: ExcludeRectReq | None = None
    save_mode: str = 'overwrite'  # overwrite | append


class ReviewMarkReq(BaseModel):
    run_id: str
    image_id: str = ''
    decision: str
    note: str = ''


class ResultsClearReq(BaseModel):
    session_id: str
    image_id: str


class LibraryLoadReq(BaseModel):
    session_id: str
    image_id: str
    restore_scribbles: bool = True
    scribble_origin: str = ''


class LibraryDeleteReq(BaseModel):
    session_id: str = ''
    image_id: str


class LocalImagePrefsReq(BaseModel):
    start_dir: str = ''


class LocalImageLoadReq(BaseModel):
    session_id: str
    path: str
    scale_percent: float = 100.0


class OpenFolderReq(BaseModel):
    path: str = ''
    kind: str = 'outputs'  # outputs | library | custom


class LocalImageSelectFolderReq(BaseModel):
    initial_dir: str = ''


class ScribbleDraftSaveReq(BaseModel):
    session_id: str
    image_id: str
    scribble_map_b64: str = ''
    scribble_origin: str = ''


class ScribbleDraftClearReq(BaseModel):
    session_id: str
    image_id: str
    origin: str = ''


def _ok(status: str, *, level: str = 'info', payload: dict[str, Any] | None = None, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        'ok': True,
        'status': status,
        'status_level': level,
        'payload': payload or {},
        'meta': meta or {},
    }


def _load_ui_prefs() -> dict[str, Any]:
    try:
        if UI_PREFS_PATH.exists():
            return dict(json.loads(UI_PREFS_PATH.read_text(encoding='utf-8')) or {})
    except Exception:
        pass
    return {}


def _save_ui_prefs(payload: dict[str, Any]) -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    UI_PREFS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload


def _resolve_existing_dir(path_text: str) -> Path:
    raw = str(path_text or '').strip().strip('"').strip("'")
    if not raw:
        raise HTTPException(status_code=400, detail='Ruta requerida.')
    # Normalize backslashes for Windows
    raw = raw.replace('/', '\\')
    path = Path(raw).expanduser()
    try:
        path = path.resolve()
    except Exception:
        path = path.absolute()
    if not path.exists():
        raise HTTPException(status_code=400, detail=f'La ruta no existe: {path}')
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f'La ruta no es una carpeta: {path}')
    return path


def _default_tutorial_local_image_dir() -> Path:
    return PROJECT_ROOT / 'frontend' / 'public' / 'tutorial'


def _select_directory_with_dialog(initial_dir: str = '') -> Path:
    if os.name != 'nt':
        raise HTTPException(status_code=400, detail='Elegir directorio solo esta soportado automaticamente en Windows.')
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'No se pudo abrir el selector de carpetas: {exc}') from exc

    resolved_initial = ''
    candidates = [
        str(initial_dir or '').strip(),
        str(_load_ui_prefs().get('start_dir') or '').strip(),
        str(_default_tutorial_local_image_dir()),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            resolved_initial = str(_resolve_existing_dir(candidate))
            break
        except HTTPException:
            continue

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.update()
    try:
        selected = filedialog.askdirectory(
            parent=root,
            mustexist=True,
            title='Elegir directorio de imagenes',
            initialdir=resolved_initial or None,
        )
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    if not selected:
        raise HTTPException(status_code=400, detail='Seleccion de directorio cancelada.')
    return _resolve_existing_dir(selected)


def _resize_percent(image: np.ndarray, p: float) -> np.ndarray:
    p = float(np.clip(p, 1.0, 100.0))
    if p >= 99.9:
        return image
    h, w = image.shape[:2]
    nw = max(1, int(round(w * p / 100.0)))
    nh = max(1, int(round(h * p / 100.0)))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def _load_image_from_path_to_session(sess, path: Path, scale_percent: float = 100.0) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f'Imagen no encontrada: {path}')
    if path.suffix.lower() not in LOCAL_IMAGE_EXTS:
        raise HTTPException(status_code=400, detail=f'Extension no soportada: {path.suffix}')
    raw = path.read_bytes()
    if not raw:
        raise HTTPException(status_code=400, detail='Archivo de imagen vacio.')
    img = _decode_upload(raw, path.name)
    rgb = to_uint8_rgb(img)
    if rgb is None:
        raise HTTPException(status_code=400, detail='Formato de imagen no soportado.')
    rgb, cap_applied, cap_scale = _cap_image_to_max_resolution(rgb, DEFAULT_MAX_RESOLUTION_PX)
    rgb = _resize_percent(rgb, float(scale_percent))

    sess.image_rgb = rgb
    sess.image_name = str(path.name)
    sess.image_id = compute_image_id(rgb)
    sess.gt_mask = None
    sess.touch()
    source_mtime = ''
    try:
        source_mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        source_mtime = ''
    active_project = get_active_project()
    project_tags, project_ids, structured_tags = project_tags_for_image(active_project)
    register_library_image(
        sess.image_id,
        sess.image_name,
        rgb,
        source_path=str(path),
        source_mtime=source_mtime,
        tags=project_tags,
        project_ids=project_ids,
        structured_tags=structured_tags,
    )

    prefs = _load_ui_prefs()
    prefs['start_dir'] = str(path.parent)
    _save_ui_prefs(prefs)

    image_b64, image_mime = encode_display_b64(rgb)
    return {
        'session_id': sess.session_id,
        'image_id': sess.image_id,
        'image_name': sess.image_name,
        'image_shape': [int(v) for v in rgb.shape],
        'image_b64': image_b64,
        'image_mime': image_mime,
        'has_gt_mask': False,
        'scale_percent': float(np.clip(float(scale_percent), 1.0, 100.0)),
        'max_resolution_px': int(DEFAULT_MAX_RESOLUTION_PX),
        'cap_applied': bool(cap_applied),
        'cap_scale': float(cap_scale),
        'source_path': str(path),
    }


def _project_for_upload(project_id: str = '') -> dict[str, Any] | None:
    pid = str(project_id or '').strip()
    if not pid:
        return get_active_project()
    try:
        state = list_projects()
        for project in state.get('projects') or []:
            if str(project.get('project_id') or '') == pid:
                return dict(project)
    except Exception:
        pass
    return None


def _load_uploaded_image_to_session(
    sess,
    raw: bytes,
    filename: str,
    *,
    scale_percent: float = 100.0,
    project_id: str = '',
    source_label: str = '',
) -> dict[str, Any]:
    if not raw:
        raise HTTPException(status_code=400, detail='Archivo de imagen vacio.')
    name = str(filename or 'image').strip() or 'image'
    ext = Path(name).suffix.lower()
    if ext and ext not in LOCAL_IMAGE_EXTS:
        raise HTTPException(status_code=400, detail=f'Extension no soportada: {ext}')
    img = _decode_upload(raw, name)
    rgb = to_uint8_rgb(img)
    if rgb is None:
        raise HTTPException(status_code=400, detail='Formato de imagen no soportado.')
    rgb, cap_applied, cap_scale = _cap_image_to_max_resolution(rgb, DEFAULT_MAX_RESOLUTION_PX)
    rgb = _resize_percent(rgb, float(scale_percent))

    sess.image_rgb = rgb
    sess.image_name = name
    sess.image_id = compute_image_id(rgb)
    sess.gt_mask = None
    sess.touch()

    project = _project_for_upload(project_id)
    project_tags, project_ids, structured_tags = project_tags_for_image(project)
    register_library_image(
        sess.image_id,
        sess.image_name,
        rgb,
        source_path=str(source_label or name),
        tags=project_tags,
        project_ids=project_ids,
        structured_tags=structured_tags,
    )

    image_b64, image_mime = encode_display_b64(rgb)
    return {
        'session_id': sess.session_id,
        'image_id': sess.image_id,
        'image_name': sess.image_name,
        'image_shape': [int(v) for v in rgb.shape],
        'image_b64': image_b64,
        'image_mime': image_mime,
        'has_gt_mask': False,
        'scale_percent': float(np.clip(float(scale_percent), 1.0, 100.0)),
        'max_resolution_px': int(DEFAULT_MAX_RESOLUTION_PX),
        'cap_applied': bool(cap_applied),
        'cap_scale': float(cap_scale),
        'source_path': str(source_label or name),
        'project_id': str(project.get('project_id') or '') if project else '',
    }


def _cap_image_to_max_resolution(image: np.ndarray, max_resolution_px: int) -> tuple[np.ndarray, bool, float]:
    img = np.asarray(image)
    if img.ndim < 2:
        return image, False, 1.0
    h, w = img.shape[:2]
    m = int(max(1, max_resolution_px))
    longest = max(int(h), int(w))
    if longest <= m:
        return image, False, 1.0
    scale = float(m) / float(longest)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return resized, True, float(scale)


def _normalize_uploaded_tiff_array(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    while out.ndim > 3:
        out = out[0]
    if out.ndim == 3 and out.shape[0] in (3, 4) and out.shape[2] not in (3, 4):
        out = np.moveaxis(out, 0, -1)
    if out.ndim == 3 and out.shape[2] not in (3, 4):
        out = out[0]
    return np.asarray(out)


def _decode_upload(data: bytes, filename: str | None = None) -> np.ndarray:
    name = str(filename or '').strip().lower()
    ext = Path(name).suffix.lower()
    image: np.ndarray | None = None

    if ext in {'.tif', '.tiff'}:
        try:
            import tifffile

            tif = tifffile.imread(io.BytesIO(data))
            image = _normalize_uploaded_tiff_array(np.asarray(tif))
        except Exception:
            image = None

    if image is None:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.ndim == 3 and img.shape[2] in (3, 4):
                code = cv2.COLOR_BGR2RGB if img.shape[2] == 3 else cv2.COLOR_BGRA2RGBA
                img = cv2.cvtColor(img, code)
            return np.asarray(img)
        try:
            pil = Image.open(BytesIO(data))
            image = np.asarray(pil)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f'No se pudo decodificar imagen: {exc}') from exc

    return np.asarray(image)


def _decode_gt_mask(gt_mask_b64: str, target_shape: tuple[int, int]) -> np.ndarray | None:
    txt = str(gt_mask_b64 or '').strip()
    if not txt:
        return None
    arr = decode_png_b64(txt)
    if arr is None:
        return None
    if arr.ndim == 3:
        g = to_gray_u8(arr)
        if g is None:
            return None
        arr = g
    arr_u8 = np.asarray(arr, dtype=np.uint8)
    if arr_u8.shape != target_shape:
        arr_u8 = cv2.resize(arr_u8, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (arr_u8 > 0).astype(np.uint8)


def _sanitize_exclude_rect(rect: ExcludeRectReq | dict[str, Any] | None, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if rect is None:
        return None
    if isinstance(rect, ExcludeRectReq):
        data: dict[str, Any] = rect.model_dump()
    elif isinstance(rect, dict):
        data = rect
    else:
        return None
    try:
        x = float(data.get('x', 0.0))
        y = float(data.get('y', 0.0))
        w = float(data.get('w', 0.0))
        h = float(data.get('h', 0.0))
    except Exception:
        return None
    if w <= 0.0 or h <= 0.0:
        return None
    ih, iw = int(shape[0]), int(shape[1])
    x0 = int(np.floor(max(0.0, min(float(iw), x))))
    y0 = int(np.floor(max(0.0, min(float(ih), y))))
    x1 = int(np.ceil(max(0.0, min(float(iw), x + w))))
    y1 = int(np.ceil(max(0.0, min(float(ih), y + h))))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _apply_exclude_rect_to_labels(labels: np.ndarray, rect: ExcludeRectReq | dict[str, Any] | None) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.uint8)
    box = _sanitize_exclude_rect(rect, arr.shape[:2])
    if box is None:
        return arr
    x0, y0, x1, y1 = box
    out = arr.copy()
    out[y0:y1, x0:x1] = 0
    return out


def _require_session(session_id: str):
    sess = store.get(session_id)
    if sess is None:
        raise HTTPException(status_code=400, detail='Sesion invalida. Crea una sesion nueva.')
    return sess


REVIEW_DECISIONS = {'s', 'a', 'b', 'c', 'unusable', 'ok', 'bad'}


def _run_to_payload(run_item: dict[str, Any]) -> dict[str, Any]:
    labels_vis = labels_to_visual(np.asarray(run_item['scribble_map'], dtype=np.uint8))
    meta = dict(run_item.get('meta') or {})
    input_b64, input_mime = encode_display_b64(run_item['input_image'])
    overlay_b64, overlay_mime = encode_display_b64(run_item['overlay'])
    return {
        'run_id': run_item.get('run_id', ''),
        'image_id': run_item.get('image_id', ''),
        'experiment_id': run_item.get('experiment_id', ''),
        'created_at': run_item.get('created_at', ''),
        'run_status_level': str(meta.get('run_status_level', 'success')),
        'input_image_b64': input_b64,
        'input_image_mime': input_mime,
        'scribble_map_b64': encode_gray_png_b64(labels_vis),
        'prior_b64': encode_gray_png_b64((np.clip(np.asarray(run_item['prior_prob'], dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)),
        'mask_b64': encode_gray_png_b64((np.asarray(run_item['mask']) > 0).astype(np.uint8) * 255),
        'overlay_b64': overlay_b64,
        'overlay_mime': overlay_mime,
        'meta': meta,
    }


_TRIPLET_PARAM_PROFILES: dict[str, dict[str, tuple[Any, Any, Any]]] = {
    'extratrees_pixel': {'n_estimators': (80, 160, 300), 'threshold': (0.35, 0.50, 0.65)},
    'rf_pixel': {'n_estimators': (60, 120, 240), 'threshold': (0.35, 0.50, 0.65)},
    'xgboost_pixel': {'n_estimators': (80, 180, 320), 'threshold': (0.35, 0.50, 0.65)},
    'catboost_pixel': {'n_estimators': (100, 220, 360), 'threshold': (0.35, 0.50, 0.65)},
    'extratrees_balanced': {'n_estimators': (120, 180, 320), 'threshold': (0.35, 0.50, 0.65)},
    'context_features_variant': {'n_estimators': (120, 180, 320), 'threshold': (0.35, 0.50, 0.65)},
    'classifier_morph_min': {'n_estimators': (120, 160, 260), 'threshold': (0.35, 0.50, 0.65), 'closing_radius': (0, 1, 2)},
}


def _build_param_high(experiment_id: str, base_params: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    eid = str(experiment_id or '').strip().lower()
    ranges = _TRIPLET_PARAM_PROFILES.get(eid, {})
    if not ranges:
        p = dict(base_params)
        p['__profile_name'] = 'high'
        return [('high', p)]

    p = dict(base_params)
    for k, vals in ranges.items():
        if len(vals) >= 1:
            p[k] = vals[-1]
    p['__profile_name'] = 'high'
    return [('high', p)]


def _merge_meta(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    out = dict(dst or {})
    for k, v in (src or {}).items():
        out[k] = v
    return out


def _execute_single_run(
    *,
    sess: Any,
    experiment_id: str,
    params: dict[str, Any],
    labels_effective: np.ndarray,
    gt: np.ndarray | None,
    exclude_rect: ExcludeRectReq | None,
    save_mode: str = 'overwrite',
) -> tuple[dict[str, Any], dict[str, Any]]:
    n_fg, n_bg, ok = has_fg_bg(labels_effective)
    if not ok:
        raise HTTPException(status_code=400, detail=f'Se necesitan marcas de fibra (n={n_fg}) y halo/background (n={n_bg}).')
    art = run_experiment(
        registry=registry,
        experiment_id=experiment_id,
        image_rgb=sess.image_rgb,
        labels=labels_effective,
        params=params,
        gt_mask=gt,
        exclude_rect=exclude_rect.model_dump() if exclude_rect else None,
    )

    mode = str(save_mode or 'overwrite').strip().lower()
    if mode not in {'overwrite', 'append'}:
        mode = 'overwrite'
    if mode == 'overwrite':
        params_eff = dict((art.meta or {}).get('params_effective') or {})
        profile_name = str(params_eff.get('__profile_name') or '')
        delete_runs_for_image(art.image_id, experiment_id=art.experiment_id, profile_name=profile_name)

    save_meta = save_run(art)
    if str(sess.image_id or '').strip() == str(art.image_id or '').strip():
        save_prior_cache(
            art.image_id,
            prior_map=art.prior_map,
            prior_overlay=art.overlay,
            experiment_id=art.experiment_id,
            params_effective=dict((art.meta or {}).get('params_effective') or {}),
            run_id=art.run_id,
            class_prob_maps=dict(art.class_prob_maps or {}),
        )
    run_item = load_run(art.run_id)
    payload = _run_to_payload(run_item)
    payload['save'] = save_meta
    return payload, save_meta


@app.post('/api/session/new')
def api_session_new(req: SessionReq | None = None) -> dict[str, Any]:
    sess = store.new(req.session_id if isinstance(req, SessionReq) else None)
    return _ok('Sesion creada.', level='success', payload={'session_id': sess.session_id})


@app.post('/api/image/load')
async def api_image_load(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    gt_file: UploadFile | None = File(default=None),
    scale_percent: float = Form(default=100.0),
) -> dict[str, Any]:
    sess = _require_session(session_id)
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail='Archivo de imagen vacio.')
    img = _decode_upload(raw, file.filename)
    rgb = to_uint8_rgb(img)
    if rgb is None:
        raise HTTPException(status_code=400, detail='Formato de imagen no soportado.')
    rgb, cap_applied, cap_scale = _cap_image_to_max_resolution(rgb, DEFAULT_MAX_RESOLUTION_PX)
    rgb = _resize_percent(rgb, float(scale_percent))

    gt = None
    if gt_file is not None:
        gt_raw = await gt_file.read()
        if gt_raw:
            gt_img = _decode_upload(gt_raw, gt_file.filename)
            gt_gray = gt_img if gt_img.ndim == 2 else to_gray_u8(gt_img)
            if gt_gray is not None:
                if gt_gray.shape[:2] != rgb.shape[:2]:
                    gt_gray = cv2.resize(gt_gray, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                gt = (np.asarray(gt_gray) > 0).astype(np.uint8)

    sess.image_rgb = rgb
    sess.image_name = str(file.filename or 'image')
    sess.image_id = compute_image_id(rgb)
    sess.gt_mask = gt
    sess.touch()
    active_project = get_active_project()
    project_tags, project_ids, structured_tags = project_tags_for_image(active_project)
    register_library_image(sess.image_id, sess.image_name, rgb, tags=project_tags, project_ids=project_ids, structured_tags=structured_tags)

    image_b64, image_mime = encode_display_b64(rgb)
    payload = {
        'session_id': sess.session_id,
        'image_id': sess.image_id,
        'image_name': sess.image_name,
        'image_shape': [int(v) for v in rgb.shape],
        'image_b64': image_b64,
        'image_mime': image_mime,
        'has_gt_mask': bool(isinstance(gt, np.ndarray)),
        'scale_percent': float(np.clip(float(scale_percent), 1.0, 100.0)),
        'max_resolution_px': int(DEFAULT_MAX_RESOLUTION_PX),
        'cap_applied': bool(cap_applied),
        'cap_scale': float(cap_scale),
    }
    return _ok('Imagen cargada.', level='success', payload=payload)


@app.get('/api/library/images')
def api_library_images(session_id: str = '') -> dict[str, Any]:
    rows = list_library_images()
    items: list[dict[str, Any]] = []
    for row in rows:
        image_id = str(row.get('image_id') or '')
        thumb_b64 = ''
        thumb_mime = 'image/png'
        try:
            thumb = load_library_thumbnail(image_id)
            thumb_b64, thumb_mime = encode_display_b64(thumb)
        except Exception:
            thumb_b64 = ''
        mask_thumb_b64 = ''
        mask_thumb_mime = 'image/png'
        try:
            mask_thumb = load_library_mask_thumbnail(image_id)
            if mask_thumb is not None:
                mask_thumb_b64, mask_thumb_mime = encode_display_b64(mask_thumb)
        except Exception:
            mask_thumb_b64 = ''
        item = dict(row)
        item['thumbnail_b64'] = thumb_b64
        item['thumbnail_mime'] = thumb_mime
        item['mask_thumbnail_b64'] = mask_thumb_b64
        item['mask_thumbnail_mime'] = mask_thumb_mime
        items.append(item)
    return _ok('Imagenes guardadas listadas.', payload={'items': items})


@app.post('/api/library/load')
def api_library_load(req: LibraryLoadReq) -> dict[str, Any]:
    sess = _require_session(req.session_id)
    image_id = str(req.image_id or '').strip()
    if not image_id:
        raise HTTPException(status_code=400, detail='image_id requerido.')
    try:
        rgb, meta = load_library_image(image_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    loaded_id = compute_image_id(rgb)
    sess.image_rgb = rgb
    sess.image_name = str(meta.get('image_name') or 'image')
    sess.image_id = loaded_id
    sess.gt_mask = None
    sess.touch()

    draft_payload: dict[str, Any] = {'found': False, 'image_id': loaded_id}
    if req.restore_scribbles:
        try:
            origin_to_load = str(req.scribble_origin or '').strip().lower()
            draft = load_scribble_draft(image_id, origin=origin_to_load)
            if bool(draft.get('found', False)):
                labels = np.asarray(draft.get('labels'), dtype=np.uint8)
                if labels.shape != rgb.shape[:2]:
                    labels = cv2.resize(labels, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                labels = np.where(np.isin(labels, [1, 2, 3]), labels, 0).astype(np.uint8)
                draft_payload = {
                    'found': True,
                    'image_id': loaded_id,
                    'scribble_map_b64': encode_gray_png_b64(labels_to_visual(labels)),
                    'meta': dict(draft.get('meta') or {}),
                }
        except Exception:
            draft_payload = {'found': False, 'image_id': loaded_id}

    image_b64, image_mime = encode_display_b64(rgb)
    payload = {
        'session_id': sess.session_id,
        'image_id': loaded_id,
        'requested_image_id': image_id,
        'image_name': sess.image_name,
        'image_shape': [int(v) for v in rgb.shape],
        'image_b64': image_b64,
        'image_mime': image_mime,
        'has_gt_mask': False,
        'library_meta': dict(meta),
        'scribble_draft': draft_payload,
    }
    return _ok('Imagen guardada cargada.', level='success', payload=payload)


@app.post('/api/library/delete')
def api_library_delete(req: LibraryDeleteReq) -> dict[str, Any]:
    image_id = str(req.image_id or '').strip()
    if not image_id:
        raise HTTPException(status_code=400, detail='image_id requerido.')
    if req.session_id:
        sess = _require_session(req.session_id)
        if str(getattr(sess, 'image_id', '') or '') == image_id:
            sess.image_rgb = None
            sess.image_name = ''
            sess.image_id = ''
            sess.gt_mask = None
            sess.touch()
    deleted: dict[str, Any] = {}
    try:
        deleted['library'] = delete_library_image(image_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        deleted['draft'] = clear_scribble_draft(image_id)
    except Exception:
        deleted['draft'] = {}
    try:
        deleted['runs'] = clear_results_for_image(image_id)
    except Exception:
        deleted['runs'] = {}
    return _ok('Imagen guardada eliminada.', level='success', payload=deleted)


@app.get('/api/local-images/prefs')
def api_local_image_prefs() -> dict[str, Any]:
    prefs = _load_ui_prefs()
    return _ok('Preferencias locales cargadas.', payload={'start_dir': str(prefs.get('start_dir') or '')})


@app.post('/api/local-images/prefs')
def api_local_image_prefs_save(req: LocalImagePrefsReq) -> dict[str, Any]:
    path = _resolve_existing_dir(req.start_dir)
    prefs = _load_ui_prefs()
    prefs['start_dir'] = str(path)
    _save_ui_prefs(prefs)
    return _ok('Ruta inicial guardada.', level='success', payload={'start_dir': str(path)})


@app.get('/api/local-images/tutorial-path')
def api_local_image_tutorial_path() -> dict[str, Any]:
    path = _default_tutorial_local_image_dir()
    image_name = 'overview-reference.png'
    image_path = path / image_name
    return _ok(
        'Ruta tutorial cargada.',
        payload={
            'start_dir': str(path),
            'image_name': image_name,
            'exists': bool(path.exists() and path.is_dir()),
            'image_exists': bool(image_path.exists() and image_path.is_file()),
        },
    )


@app.post('/api/local-images/select-folder')
def api_local_images_select_folder(req: LocalImageSelectFolderReq) -> dict[str, Any]:
    path = _select_directory_with_dialog(req.initial_dir)
    prefs = _load_ui_prefs()
    prefs['start_dir'] = str(path)
    _save_ui_prefs(prefs)
    return _ok('Directorio seleccionado.', level='success', payload={'start_dir': str(path)})


@app.get('/api/local-images/list')
def api_local_images_list(start_dir: str = '', recursive: bool = True, limit: int = 400) -> dict[str, Any]:
    path = _resolve_existing_dir(start_dir or str(_load_ui_prefs().get('start_dir') or ''))
    max_items = int(np.clip(int(limit or 400), 1, 2000))
    iterator = path.rglob('*') if bool(recursive) else path.glob('*')
    items: list[dict[str, Any]] = []
    for p in iterator:
        if len(items) >= max_items:
            break
        if not p.is_file() or p.suffix.lower() not in LOCAL_IMAGE_EXTS:
            continue
        try:
            stat = p.stat()
            rel = str(p.relative_to(path))
            updated_at = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            items.append(
                {
                    'name': p.name,
                    'path': str(p),
                    'relative_path': rel,
                    'size_bytes': int(stat.st_size),
                    'updated_at': updated_at,
                }
            )
        except Exception:
            continue
    items.sort(key=lambda x: str(x.get('relative_path') or ''))
    return _ok('Imagenes locales listadas.', payload={'start_dir': str(path), 'items': items, 'truncated': len(items) >= max_items})


@app.post('/api/local-images/load')
def api_local_image_load(req: LocalImageLoadReq) -> dict[str, Any]:
    sess = _require_session(req.session_id)
    path = Path(str(req.path or '').strip().strip('"')).expanduser()
    try:
        path = path.resolve()
    except Exception:
        path = path.absolute()
    payload = _load_image_from_path_to_session(sess, path, req.scale_percent)
    return _ok('Imagen local cargada.', level='success', payload=payload)


@app.post('/api/local-images/upload-browser')
async def api_local_image_upload_browser(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    project_id: str = Form(default=''),
    source_label: str = Form(default=''),
    scale_percent: float = Form(default=100.0),
) -> dict[str, Any]:
    sess = _require_session(session_id)
    raw = await file.read()
    label = str(source_label or file.filename or '').strip()
    payload = _load_uploaded_image_to_session(
        sess,
        raw,
        str(file.filename or 'image'),
        scale_percent=scale_percent,
        project_id=project_id,
        source_label=label,
    )
    return _ok('Imagen cargada desde navegador.', level='success', payload=payload)


@app.post('/api/system/open-folder')
def api_system_open_folder(req: OpenFolderReq) -> dict[str, Any]:
    kind = str(req.kind or 'outputs').strip().lower()
    if kind == 'library':
        target = OUTPUT_ROOT / 'library'
    elif kind == 'custom':
        target = _resolve_existing_dir(req.path)
    else:
        target = OUTPUT_ROOT
    target.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == 'nt':
            os.startfile(str(target))  # type: ignore[attr-defined]
        else:
            raise RuntimeError('Abrir carpeta solo esta soportado automaticamente en Windows.')
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'No se pudo abrir carpeta: {exc}') from exc
    return _ok('Carpeta abierta.', level='success', payload={'path': str(target)})


@app.get('/api/experiments/catalog')
def api_experiments_catalog() -> dict[str, Any]:
    all_items = registry.list()
    return _ok('Catalogo cargado.', payload={'experiments': all_items})


@app.post('/api/experiments/run')
def api_experiment_run(req: RunReq) -> dict[str, Any]:
    sess = _require_session(req.session_id)
    if sess.image_rgb is None:
        raise HTTPException(status_code=400, detail='Carga una imagen antes de ejecutar experimentos.')

    labels = decode_scribble_b64(req.scribble_map_b64, target_shape=sess.image_rgb.shape[:2])
    labels_effective = _apply_exclude_rect_to_labels(labels, req.exclude_rect)

    gt = _decode_gt_mask(req.gt_mask_b64, target_shape=sess.image_rgb.shape[:2])
    if gt is None:
        gt = sess.gt_mask

    try:
        params = dict(req.params or {})
        params.setdefault('__profile_name', 'single')
        payload, _save_meta = _execute_single_run(
            sess=sess,
            experiment_id=req.experiment_id,
            params=params,
            labels_effective=labels_effective,
            gt=gt,
            exclude_rect=req.exclude_rect,
            save_mode=req.save_mode,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Error al ejecutar experimento: {exc}') from exc

    run_level = str(payload.get('run_status_level', 'success'))
    return _ok('Experimento ejecutado.', level=run_level, payload=payload)


@app.post('/api/scribble/draft/save')
def api_scribble_draft_save(req: ScribbleDraftSaveReq) -> dict[str, Any]:
    sess = _require_session(req.session_id)
    if sess.image_rgb is None:
        raise HTTPException(status_code=400, detail='Carga una imagen antes de guardar draft.')
    image_id = str(req.image_id or '').strip()
    if not image_id:
        raise HTTPException(status_code=400, detail='image_id requerido.')
    if image_id != str(sess.image_id or '').strip():
        raise HTTPException(status_code=400, detail='image_id no corresponde a la imagen activa de la sesion.')
    labels = decode_scribble_b64(req.scribble_map_b64, target_shape=sess.image_rgb.shape[:2])
    meta = save_scribble_draft(image_id, labels, origin=req.scribble_origin)
    return _ok(
        'Draft de scribble guardado.',
        level='success',
        payload={
            'image_id': image_id,
            'updated_at': str(meta.get('updated_at', '')),
            'n_fg': int(meta.get('n_fg', 0)),
            'n_halo': int(meta.get('n_halo', 0)),
            'n_bg': int(meta.get('n_bg', 0)),
            'format_version': str(meta.get('format_version', 'v3_multiclass_halo')),
            'scribble_origin': str(meta.get('scribble_origin', '')),
        },
    )


@app.get('/api/scribble/draft/load')
def api_scribble_draft_load(session_id: str, image_id: str, origin: str = '') -> dict[str, Any]:
    sess = _require_session(session_id)
    if sess.image_rgb is None:
        raise HTTPException(status_code=400, detail='Carga una imagen antes de restaurar draft.')
    sid = str(image_id or '').strip()
    if not sid:
        raise HTTPException(status_code=400, detail='image_id requerido.')
    origin_key = str(origin or '').strip().lower()
    payload = load_scribble_draft(sid, origin=origin_key)
    if not bool(payload.get('found', False)):
        return _ok('Sin draft para la imagen.', payload={'found': False, 'image_id': sid})

    labels = np.asarray(payload.get('labels'), dtype=np.uint8)
    if labels.shape != sess.image_rgb.shape[:2]:
        labels = cv2.resize(labels, (sess.image_rgb.shape[1], sess.image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    labels = np.where(np.isin(labels, [1, 2, 3]), labels, 0).astype(np.uint8)
    vis = labels_to_visual(labels)
    meta = dict(payload.get('meta') or {})
    return _ok(
        'Draft restaurado.',
        payload={
            'found': True,
            'image_id': sid,
            'scribble_map_b64': encode_gray_png_b64(vis),
            'meta': meta,
        },
    )


@app.post('/api/scribble/draft/clear')
def api_scribble_draft_clear(req: ScribbleDraftClearReq) -> dict[str, Any]:
    _require_session(req.session_id)
    sid = str(req.image_id or '').strip()
    if not sid:
        raise HTTPException(status_code=400, detail='image_id requerido.')
    info = clear_scribble_draft(sid, origin=str(req.origin or '').strip().lower())
    return _ok('Draft limpiado.', level='success', payload=info)


@app.post('/api/experiments/run-batch')
def api_experiment_run_batch(req: RunBatchReq) -> dict[str, Any]:
    sess = _require_session(req.session_id)
    if sess.image_rgb is None:
        raise HTTPException(status_code=400, detail='Carga una imagen antes de ejecutar batch.')
    ids = [str(x).strip() for x in req.experiment_ids if str(x).strip()]
    if not ids:
        raise HTTPException(status_code=400, detail='experiment_ids vacio.')

    labels = decode_scribble_b64(req.scribble_map_b64, target_shape=sess.image_rgb.shape[:2])
    labels_effective = _apply_exclude_rect_to_labels(labels, req.exclude_rect)

    gt = _decode_gt_mask(req.gt_mask_b64, target_shape=sess.image_rgb.shape[:2])
    if gt is None:
        gt = sess.gt_mask

    items: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    raw_sweep_mode = str(req.param_sweep or 'high').strip().lower()
    sweep_mode = 'high' if raw_sweep_mode in {'extremes_mid', 'high', 'high_only'} else 'single'

    for eid in ids:
        base_params = dict(req.params or {})
        base_params.update(dict((req.params_by_experiment or {}).get(eid) or {}))
        if sweep_mode == 'high':
            param_sets = _build_param_high(eid, base_params)
        else:
            p = dict(base_params)
            p['__profile_name'] = 'single'
            param_sets = [('single', p)]

        for profile_name, params in param_sets:
            try:
                payload, _ = _execute_single_run(
                    sess=sess,
                    experiment_id=eid,
                    params=dict(params or {}),
                    labels_effective=labels_effective,
                    gt=gt,
                    exclude_rect=req.exclude_rect,
                    save_mode=req.save_mode,
                )
                payload['batch_profile'] = profile_name
                items.append(payload)
                if str(payload.get('run_status_level', 'success')) == 'warning':
                    warnings.append(
                        {
                            'experiment_id': eid,
                            'profile': profile_name,
                            'run_id': payload.get('run_id', ''),
                            'reason': str((payload.get('meta') or {}).get('blocker_reason', 'fallback')),
                        }
                    )
            except Exception as exc:
                failures.append({'experiment_id': eid, 'profile': profile_name, 'error': str(exc)})

    if items and (warnings or failures):
        level = 'warning'
    elif items:
        level = 'success'
    else:
        level = 'error'

    return _ok(
        'Batch ejecutado.',
        level=level,
        payload={
            'items': items,
            'warnings': warnings,
            'failures': failures,
            'param_sweep': sweep_mode,
            'requested_experiments': len(ids),
            'executed_runs': len(items),
        },
    )


@app.get('/api/results/list')
def api_results_list(image_id: str) -> dict[str, Any]:
    if not str(image_id or '').strip():
        raise HTTPException(status_code=400, detail='image_id requerido.')
    rows = list_runs(image_id)
    return _ok('Resultados listados.', payload={'image_id': image_id, 'items': rows})


@app.post('/api/results/clear')
def api_results_clear(req: ResultsClearReq) -> dict[str, Any]:
    sess = _require_session(req.session_id)
    image_id = str(req.image_id or '').strip()
    if not image_id:
        raise HTTPException(status_code=400, detail='image_id requerido.')
    active_id = str(getattr(sess, 'image_id', '') or '').strip()
    if active_id and image_id != active_id:
        raise HTTPException(status_code=400, detail='image_id no corresponde a la imagen activa de la sesion.')
    info = clear_results_for_image(image_id)
    try:
        clear_prior_cache(image_id)
    except Exception:
        pass
    return _ok('Resultados de revision borrados.', level='success', payload=info)


@app.get('/api/results/get')
def api_results_get(run_id: str) -> dict[str, Any]:
    if not str(run_id or '').strip():
        raise HTTPException(status_code=400, detail='run_id requerido.')
    try:
        item = load_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    payload = _run_to_payload(item)
    level = str(payload.get('run_status_level', 'success'))
    return _ok('Resultado cargado.', level=level, payload=payload)


@app.get('/api/results/mask-thumb')
def api_results_mask_thumb(run_id: str) -> dict[str, Any]:
    if not str(run_id or '').strip():
        raise HTTPException(status_code=400, detail='run_id requerido.')
    try:
        item = load_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    mask = (np.asarray(item['mask']) > 0).astype(np.uint8) * 255
    h, w = mask.shape[:2]
    max_dim = 200
    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    thumb_b64 = encode_gray_png_b64(mask)
    return _ok('Miniatura de mascara cargada.', payload={
        'run_id': item.get('run_id', ''),
        'mask_thumb_b64': thumb_b64,
        'width': int(mask.shape[1]),
        'height': int(mask.shape[0]),
    })


@app.post('/api/review/mark')
def api_review_mark(req: ReviewMarkReq) -> dict[str, Any]:
    decision = str(req.decision or '').strip().lower()
    if decision not in REVIEW_DECISIONS:
        raise HTTPException(status_code=400, detail='decision debe ser s, a, b, c, unusable, ok o bad.')

    run_id = str(req.run_id or '').strip()
    if not run_id:
        raise HTTPException(status_code=400, detail='run_id requerido.')

    try:
        run_item = load_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    image_id = str(req.image_id or '').strip() or str(run_item.get('image_id') or '')
    item = append_review(image_id, run_id, str(run_item.get('experiment_id') or ''), decision, req.note)
    return _ok('Revision guardada.', level='success', payload={'item': item})


@app.get('/api/review/list')
def api_review_list(image_id: str) -> dict[str, Any]:
    if not str(image_id or '').strip():
        raise HTTPException(status_code=400, detail='image_id requerido.')
    rows = list_reviews(image_id)
    return _ok('Revisiones listadas.', payload={'image_id': image_id, 'items': rows})


@app.get('/api/reports/export')
def api_reports_export(image_id: str) -> dict[str, Any]:
    if not str(image_id or '').strip():
        raise HTTPException(status_code=400, detail='image_id requerido.')
    try:
        info = export_report(image_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'No se pudo exportar reporte: {exc}') from exc
    return _ok('Reporte exportado.', level='success', payload=info)


@app.get('/api/health')
def api_health() -> dict[str, Any]:
    return {
        'ok': True,
        'app': 'scribble_research',
        'version': '0.2.0',
        'capabilities': {
            'native_folder_picker': os.name == 'nt',
            'open_folder': os.name == 'nt',
            'runtime_os': os.name,
        },
    }
