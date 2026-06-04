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


def _normalize_list(items: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    raw_items = items if isinstance(items, list) else str(items or '').split(',')
    for item in raw_items:
        text = str(item or '').strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _tag_label(tag: Any) -> str:
    if isinstance(tag, dict):
        category = str(tag.get('category') or 'other').strip()
        if category == 'size':
            value = str(tag.get('value') or '').strip()
            unit = str(tag.get('unit') or '').strip()
            return f'Tamaño: {value} {unit}'.strip()
        return str(tag.get('label') or tag.get('value') or '').strip()
    return str(tag or '').strip()


def normalize_structured_tags(items: Any) -> list[dict[str, Any]]:
    raw_items = items if isinstance(items, list) else str(items or '').split(',')
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_items:
        if isinstance(raw, dict):
            category = str(raw.get('category') or 'other').strip() or 'other'
            if category == 'size':
                value = str(raw.get('value') or '').strip()
                unit = str(raw.get('unit') or '').strip()
                if not value:
                    continue
                item = {'category': 'size', 'label': f'Tamaño: {value} {unit}'.strip(), 'value': value, 'unit': unit}
            else:
                label = str(raw.get('label') or raw.get('value') or '').strip()
                if not label:
                    continue
                item = {'category': category, 'label': label}
        else:
            label = str(raw or '').strip()
            if not label:
                continue
            item = {'category': 'other', 'label': label}
        key = json.dumps(item, sort_keys=True, ensure_ascii=False).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _legacy_tags_from_structured(items: Any) -> list[str]:
    return _normalize_list([_tag_label(item) for item in normalize_structured_tags(items)])


def _merge_lists(*lists: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for items in lists:
        for item in _normalize_list(items):
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
    return out


def register_library_image(
    image_id: str,
    image_name: str,
    image_rgb: np.ndarray,
    source_path: str = '',
    source_mtime: str = '',
    tags: Any = None,
    project_ids: Any = None,
    structured_tags: Any = None,
) -> dict[str, Any]:
    ensure_library_dirs()
    rgb = to_uint8_rgb(image_rgb)
    if rgb is None:
        raise ValueError('Imagen invalida para registrar en biblioteca.')
    d = _img_dir(image_id)
    d.mkdir(parents=True, exist_ok=True)
    _write_png(d / 'image.png', rgb)

    meta = _load_json(_meta_path(image_id))
    merged_structured = normalize_structured_tags([
        *normalize_structured_tags(meta.get('structured_tags') or meta.get('tags') or []),
        *normalize_structured_tags(structured_tags or tags or []),
    ])
    out = {
        'image_id': str(image_id),
        'image_name': str(image_name or meta.get('image_name', 'image')),
        'updated_at': _now(),
        'shape_hw': [int(rgb.shape[0]), int(rgb.shape[1])],
        'has_prior_cache': bool(meta.get('has_prior_cache', False)),
        'latest_prior_run_id': str(meta.get('latest_prior_run_id', '')),
        'source_path': str(source_path or meta.get('source_path') or ''),
        'source_mtime': str(source_mtime or meta.get('source_mtime') or ''),
        'tags': _merge_lists(meta.get('tags') or [], tags or [], _legacy_tags_from_structured(merged_structured)),
        'structured_tags': merged_structured,
        'project_ids': _merge_lists(meta.get('project_ids') or [], project_ids or []),
    }
    _save_json(_meta_path(image_id), out)
    return out


def update_library_image_tags(image_id: str, tags: Any = None, project_ids: Any = None, structured_tags: Any = None) -> dict[str, Any]:
    ensure_library_dirs()
    sid = _safe_id(image_id)
    if not sid:
        raise ValueError('image_id invalido')
    meta_path = _meta_path(image_id)
    meta = _load_json(meta_path)
    if not meta:
        raise FileNotFoundError(f'No existe metadata para {image_id}')
    normalized_structured = normalize_structured_tags(structured_tags if structured_tags is not None else tags or [])
    meta['tags'] = _normalize_list(tags or _legacy_tags_from_structured(normalized_structured))
    meta['structured_tags'] = normalized_structured
    meta['project_ids'] = _normalize_list(project_ids or [])
    meta['updated_at'] = str(meta.get('updated_at') or _now())
    _save_json(meta_path, meta)
    return meta


def _draft_meta_for(image_id: str, origin: str = '') -> dict[str, Any]:
    """Return draft meta for a specific origin, or the most recent across all origins.
    Does NOT depend on scribble_origin stored in library meta.json."""
    sid = _safe_id(image_id)
    if not sid:
        return {}
    base_dir = base_persistence.DRAFTS_DIR / sid

    # If a specific origin is requested, return that one
    if origin in ('manual', 'modelo', 'modelo_modificado'):
        meta = _load_json(base_dir / origin / 'meta.json')
        if meta:
            meta['scribble_origin'] = origin
            return meta

    # First check for legacy root-level draft (pre-multi-origin)
    legacy_meta = _load_json(base_dir / 'meta.json')
    if legacy_meta and (base_dir / 'scribble_map.npz').exists():
        legacy_meta['scribble_origin'] = 'manual'
        return legacy_meta

    # Return most recent across all origins
    best: dict[str, Any] = {}
    for candidate in ('manual', 'modelo', 'modelo_modificado'):
        candidate_dir = base_dir / candidate
        meta = _load_json(candidate_dir / 'meta.json')
        if meta:
            meta['scribble_origin'] = candidate
            if not best or str(meta.get('updated_at', '')) > str(best.get('updated_at', '')):
                best = meta
    return best


def _list_available_origins(image_id: str) -> list[str]:
    """Return which origins have scribble files for this image."""
    sid = _safe_id(image_id)
    if not sid:
        return []
    base_dir = base_persistence.DRAFTS_DIR / sid
    available: list[str] = []
    # Check for legacy root-level draft
    if (base_dir / 'scribble_map.npz').exists():
        available.append('manual')
    # Check origin subdirectories
    for candidate in ('manual', 'modelo', 'modelo_modificado'):
        if (base_dir / candidate / 'scribble_map.npz').exists():
            if candidate not in available:
                available.append(candidate)
    return available


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
        available_origins = _list_available_origins(image_id)
        items.append(
            {
                'image_id': image_id,
                'image_name': str(meta.get('image_name') or 'image'),
                'updated_at': str(meta.get('updated_at') or ''),
                'source_path': str(meta.get('source_path') or ''),
                'source_mtime': str(meta.get('source_mtime') or ''),
                'tags': _normalize_list(meta.get('tags') or []),
                'structured_tags': normalize_structured_tags(meta.get('structured_tags') or meta.get('tags') or []),
                'project_ids': _normalize_list(meta.get('project_ids') or []),
                'shape_hw': shape,
                'has_scribble_draft': bool(draft),
                'draft_updated_at': str(draft.get('updated_at') or ''),
                'draft_n_fg': int(draft.get('n_fg') or 0),
                'draft_n_halo': int(draft.get('n_halo') or 0),
                'draft_n_bg': int(draft.get('n_bg') or 0),
                'scribble_origin': str(draft.get('scribble_origin', '')),
                'scribble_origins_available': available_origins,
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


def load_library_mask_thumbnail(image_id: str, max_px: int = 220) -> np.ndarray | None:
    """Load the predicted mask overlay (prior_overlay.png) as a thumbnail, if it exists."""
    ensure_library_dirs()
    sid = _safe_id(image_id)
    if not sid:
        return None
    mask_path = _img_dir(image_id) / 'prior_overlay.png'
    if not mask_path.exists():
        return None
    try:
        mask = _read_png(mask_path, grayscale=False)
    except Exception:
        return None
    h, w = mask.shape[:2]
    m = max(32, int(max_px))
    scale = min(1.0, float(m) / float(max(1, h, w)))
    if scale >= 0.999:
        return mask
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA)


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
