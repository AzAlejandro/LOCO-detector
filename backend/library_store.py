from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import persistence as base_persistence
from .image_codec import to_uint8_rgb
from .persistence import OUTPUT_ROOT


LIBRARY_DIR = OUTPUT_ROOT / 'library'


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _safe_id(text: str) -> str:
    raw = str(text or '').strip()
    if not raw:
        return ''
    out: list[str] = []
    for ch in raw:
        if ch.isalnum() or ch in {'_', '-'}:
            out.append(ch)
        else:
            out.append('_')
    return ''.join(out).strip('_')


def _img_dir(image_id: str) -> Path:
    sid = _safe_id(image_id)
    if not sid:
        raise ValueError('image_id invalido')
    return LIBRARY_DIR / sid


def _meta_path(image_id: str) -> Path:
    return _img_dir(image_id) / 'meta.json'


def _prior_meta_path(image_id: str) -> Path:
    return _img_dir(image_id) / 'prior_meta.json'


def _write_png(path: Path, image: np.ndarray) -> None:
    arr = np.asarray(image)
    if arr.ndim == 2:
        ok, buf = cv2.imencode('.png', arr.astype(np.uint8))
    else:
        rgb = to_uint8_rgb(arr)
        if rgb is None:
            raise ValueError(f'Imagen invalida para {path.name}')
        ok, buf = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError(f'No se pudo codificar PNG: {path}')
    path.write_bytes(bytes(buf))


def _read_png(path: Path, grayscale: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
    arr = cv2.imread(str(path), flag)
    if arr is None:
        raise FileNotFoundError(f'No se pudo leer {path}')
    if not grayscale and arr.ndim == 3 and arr.shape[2] in (3, 4):
        code = cv2.COLOR_BGR2RGB if arr.shape[2] == 3 else cv2.COLOR_BGRA2RGBA
        arr = cv2.cvtColor(arr, code)
    return np.asarray(arr)


def ensure_library_dirs() -> None:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding='utf-8')) or {})
    except Exception:
        return {}


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def register_library_image(image_id: str, image_name: str, image_rgb: np.ndarray, source_path: str = '', source_mtime: str = '') -> dict[str, Any]:
    ensure_library_dirs()
    rgb = to_uint8_rgb(image_rgb)
    if rgb is None:
        raise ValueError('Imagen invalida para registrar en biblioteca.')
    d = _img_dir(image_id)
    d.mkdir(parents=True, exist_ok=True)
    _write_png(d / 'image.png', rgb)

    meta = _load_json(_meta_path(image_id))
    out = {
        'image_id': str(image_id),
        'image_name': str(image_name or meta.get('image_name', 'image')),
        'updated_at': _now(),
        'shape_hw': [int(rgb.shape[0]), int(rgb.shape[1])],
        'has_prior_cache': bool(meta.get('has_prior_cache', False)),
        'latest_prior_run_id': str(meta.get('latest_prior_run_id', '')),
        'source_path': str(source_path or meta.get('source_path') or ''),
        'source_mtime': str(source_mtime or meta.get('source_mtime') or ''),
    }
    _save_json(_meta_path(image_id), out)
    return out


def _draft_meta_for(image_id: str) -> dict[str, Any]:
    sid = _safe_id(image_id)
    if not sid:
        return {}
    path = base_persistence.DRAFTS_DIR / sid / 'meta.json'
    return _load_json(path)


def list_library_images() -> list[dict[str, Any]]:
    ensure_library_dirs()
    items: list[dict[str, Any]] = []
    for d in sorted(LIBRARY_DIR.glob('*')):
        if not d.is_dir():
            continue
        meta = _load_json(d / 'meta.json')
        image_id = str(meta.get('image_id') or d.name)
        image_path = d / 'image.png'
        if not image_id or not image_path.exists():
            continue
        draft = _draft_meta_for(image_id)
        prior_meta = _load_json(d / 'prior_meta.json')
        shape = list(meta.get('shape_hw') or [])
        items.append(
            {
                'image_id': image_id,
                'image_name': str(meta.get('image_name') or 'image'),
                'updated_at': str(meta.get('updated_at') or ''),
                'source_path': str(meta.get('source_path') or ''),
                'source_mtime': str(meta.get('source_mtime') or ''),
                'shape_hw': shape,
                'has_scribble_draft': bool(draft),
                'draft_updated_at': str(draft.get('updated_at') or ''),
                'draft_n_fg': int(draft.get('n_fg') or 0),
                'draft_n_halo': int(draft.get('n_halo') or 0),
                'draft_n_bg': int(draft.get('n_bg') or 0),
                'has_prior_cache': bool(meta.get('has_prior_cache', False)),
                'latest_prior_run_id': str(meta.get('latest_prior_run_id') or ''),
                'latest_prior_experiment_id': str(prior_meta.get('experiment_id') or ''),
                'prior_updated_at': str(prior_meta.get('updated_at') or ''),
            }
        )
    items.sort(key=lambda x: str(x.get('updated_at') or x.get('draft_updated_at') or ''), reverse=True)
    return items


def load_library_image(image_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    ensure_library_dirs()
    image_id = str(image_id or '').strip()
    img = _read_png(_img_dir(image_id) / 'image.png', grayscale=False)
    meta = _load_json(_meta_path(image_id))
    meta.setdefault('image_id', image_id)
    meta.setdefault('image_name', 'image')
    meta.setdefault('shape_hw', [int(img.shape[0]), int(img.shape[1])])
    return np.asarray(img, dtype=np.uint8), meta


def load_library_thumbnail(image_id: str, max_px: int = 220) -> np.ndarray:
    img, _meta = load_library_image(image_id)
    h, w = img.shape[:2]
    m = max(32, int(max_px))
    scale = min(1.0, float(m) / float(max(1, h, w)))
    if scale >= 0.999:
        return img
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def delete_library_image(image_id: str) -> dict[str, Any]:
    ensure_library_dirs()
    sid = _safe_id(image_id)
    if not sid:
        raise ValueError('image_id invalido')
    d = LIBRARY_DIR / sid
    existed = d.exists()
    if existed:
        shutil.rmtree(d)
    return {'image_id': str(image_id), 'deleted': bool(existed), 'path': str(d)}


def save_prior_cache(
    image_id: str,
    *,
    prior_map: np.ndarray,
    prior_overlay: np.ndarray,
    experiment_id: str,
    params_effective: dict[str, Any],
    run_id: str,
    class_prob_maps: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    ensure_library_dirs()
    d = _img_dir(image_id)
    d.mkdir(parents=True, exist_ok=True)

    p = np.clip(np.asarray(prior_map, dtype=np.float32), 0.0, 1.0)
    prior_u8 = (p * 255.0).astype(np.uint8)
    _write_png(d / 'prior_prob.png', prior_u8)
    _write_png(d / 'prior_overlay.png', prior_overlay)
    saved_class_maps: list[str] = []
    clean_maps: dict[str, np.ndarray] = {}
    for key, value in dict(class_prob_maps or {}).items():
        arr = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
        if arr.shape[:2] != p.shape[:2]:
            continue
        name = str(key)
        saved_class_maps.append(name)
        clean_maps[name] = arr
        _write_png(d / f'{name}.png', (arr * 255.0).astype(np.uint8))
    if clean_maps:
        np.savez_compressed(d / 'class_prob_maps.npz', **clean_maps)

    prior_meta = {
        'image_id': str(image_id),
        'run_id': str(run_id),
        'experiment_id': str(experiment_id),
        'params_effective': dict(params_effective or {}),
        'class_prob_maps': saved_class_maps,
        'updated_at': _now(),
    }
    _save_json(_prior_meta_path(image_id), prior_meta)

    meta = _load_json(_meta_path(image_id))
    meta.update(
        {
            'image_id': str(image_id),
            'updated_at': _now(),
            'has_prior_cache': True,
            'latest_prior_run_id': str(run_id),
        }
    )
    _save_json(_meta_path(image_id), meta)
    return meta


def clear_prior_cache(image_id: str) -> dict[str, Any]:
    ensure_library_dirs()
    d = _img_dir(image_id)
    prior_prob = d / 'prior_prob.png'
    prior_overlay = d / 'prior_overlay.png'
    prior_meta = _prior_meta_path(image_id)
    deleted = []
    for p in (prior_prob, prior_overlay, prior_meta):
        if p.exists():
            p.unlink(missing_ok=True)
            deleted.append(p.name)
    for extra in d.glob('*_prob.png'):
        if extra.name == 'prior_prob.png':
            continue
        extra.unlink(missing_ok=True)
        deleted.append(extra.name)
    cpath = d / 'class_prob_maps.npz'
    if cpath.exists():
        cpath.unlink(missing_ok=True)
        deleted.append(cpath.name)
    meta = _load_json(_meta_path(image_id))
    if meta:
        meta.update(
            {
                'updated_at': _now(),
                'has_prior_cache': False,
                'latest_prior_run_id': '',
            }
        )
        _save_json(_meta_path(image_id), meta)
    return {'image_id': str(image_id), 'deleted': deleted}
