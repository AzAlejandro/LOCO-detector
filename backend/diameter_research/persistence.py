from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..image_codec import to_uint8_rgb
from .pipeline import METHOD_ID


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / 'outputs' / 'scribble_research' / 'diameter_research'
POINTS_DIR = OUTPUT_ROOT / 'points'
LOCO_DATASET_CIRCLES_DIR = OUTPUT_ROOT / 'loco_dataset_circles'
RUNS_DIR = OUTPUT_ROOT / 'runs'
INDEX_DIR = OUTPUT_ROOT / 'index'
REPORTS_DIR = OUTPUT_ROOT / 'reports'


@dataclass
class DiameterRunArtifacts:
    run_id: str
    image_id: str
    experiment_id: str
    created_at: str
    input_image: np.ndarray
    scribble_labels: np.ndarray
    prior_prob: np.ndarray
    support_region: np.ndarray
    overlay: np.ndarray
    results: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    meta: dict[str, Any]


def ensure_dirs() -> None:
    POINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOCO_DATASET_CIRCLES_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_id(text: str) -> str:
    raw = str(text or '').strip()
    out = []
    for ch in raw:
        if ch.isalnum() or ch in {'_', '-'}:
            out.append(ch)
        else:
            out.append('_')
    return ''.join(out).strip('_')


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2), encoding='utf-8')


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding='utf-8')) or {})


def _write_png(path: Path, image: np.ndarray) -> None:
    arr = np.asarray(image)
    if arr.ndim == 2:
        ok, buf = cv2.imencode('.png', arr.astype(np.uint8))
    else:
        rgb = to_uint8_rgb(arr)
        if rgb is None:
            raise ValueError(f'No se pudo convertir imagen para {path.name}')
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


def _points_path(image_id: str) -> Path:
    sid = _safe_id(image_id)
    if not sid:
        raise ValueError('image_id invalido para puntos')
    return POINTS_DIR / f'{sid}.json'


def _loco_dataset_circles_path(image_id: str) -> Path:
    sid = _safe_id(image_id)
    if not sid:
        raise ValueError('image_id invalido para circulos LOCO')
    return LOCO_DATASET_CIRCLES_DIR / f'{sid}.json'


def normalize_points(points: list[dict[str, Any]] | None) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for item in points or []:
        try:
            x = float(item.get('x'))
            y = float(item.get('y'))
        except Exception:
            continue
        if not np.isfinite([x, y]).all():
            continue
        out.append({'x': float(x), 'y': float(y), 'circle_type': str(item.get('circle_type', '') or ''), 'radius_px': float(item.get('radius_px', 0) or 0)})
    return out


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _normalize_point_ref(item: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(item, dict):
        return None
    x = _finite_float(item.get('x'))
    y = _finite_float(item.get('y'))
    if x is None or y is None:
        return None
    return {'x': float(x), 'y': float(y)}


def _normalize_manual_line(item: dict[str, Any] | None, kind: str) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    start = _normalize_point_ref(item.get('start'))
    end = _normalize_point_ref(item.get('end'))
    if not start or not end:
        return None
    geometry_id = str(item.get('geometry_id') or item.get('manual_geometry_id') or '').strip()
    return {
        'start': start,
        'end': end,
        'method_id': 'manual_line_direct_caliper' if kind == 'direct' else 'manual_dual_side_caliper',
        'geometry_id': geometry_id,
    }


def _normalize_manual_circle(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    center = _normalize_point_ref(item.get('center'))
    radius = _finite_float(item.get('radius'))
    if not center or radius is None or radius < 1:
        return None
    geometry_id = str(item.get('geometry_id') or item.get('circle_square_geometry_id') or '').strip()
    return {
        'center': center,
        'radius': float(radius),
        'geometry_id': geometry_id,
        'consumed': bool(item.get('consumed', False)),
        'type': str(item.get('type') or item.get('circle_type') or ''),
    }


def normalize_geometry(geometry: dict[str, Any] | None) -> dict[str, Any]:
    raw = geometry if isinstance(geometry, dict) else {}
    mask_lines = [_normalize_manual_line(item, 'mask') for item in list(raw.get('mask_lines') or raw.get('maskLines') or [])]
    direct_lines = [_normalize_manual_line(item, 'direct') for item in list(raw.get('direct_lines') or raw.get('directLines') or [])]
    circles = [_normalize_manual_circle(item) for item in list(raw.get('circles') or [])]
    return {
        'mask_lines': [item for item in mask_lines if item],
        'direct_lines': [item for item in direct_lines if item],
        'circles': [item for item in circles if item],
        'mask_line_active_idx': int(raw.get('mask_line_active_idx', raw.get('maskLineActiveIdx', -1)) or -1),
        'direct_line_active_idx': int(raw.get('direct_line_active_idx', raw.get('directLineActiveIdx', -1)) or -1),
        'circle_active_idx': int(raw.get('circle_active_idx', raw.get('circleActiveIdx', -1)) or -1),
    }


def save_points(
    image_id: str,
    points: list[dict[str, Any]],
    active_point_idx: int | None = None,
    geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_dirs()
    pts = normalize_points(points)
    if active_point_idx is None:
        active = 0 if pts else -1
    else:
        active = int(active_point_idx)
        if active < 0 or active >= len(pts):
            active = 0 if pts else -1
    payload = {
        'image_id': str(image_id),
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'points': pts,
        'active_point_idx': int(active),
        'geometry': normalize_geometry(geometry),
    }
    _write_json(_points_path(image_id), payload)
    return payload


def load_points(image_id: str) -> dict[str, Any]:
    ensure_dirs()
    path = _points_path(image_id)
    if not path.exists():
        return {'found': False, 'image_id': str(image_id), 'points': [], 'active_point_idx': -1, 'geometry': normalize_geometry({})}
    payload = _read_json(path)
    pts = normalize_points(list(payload.get('points') or []))
    active = int(payload.get('active_point_idx', 0 if pts else -1))
    if active < 0 or active >= len(pts):
        active = 0 if pts else -1
    payload['found'] = True
    payload['points'] = pts
    payload['active_point_idx'] = active
    payload['geometry'] = normalize_geometry(payload.get('geometry') or {})
    payload.setdefault('image_id', str(image_id))
    return payload


def normalize_loco_dataset_circles(circles: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(circles or []):
        if not isinstance(item, dict):
            continue
        try:
            cx = float(item.get('center_x'))
            cy = float(item.get('center_y'))
            radius = float(item.get('radius_px'))
        except Exception:
            continue
        if not np.isfinite([cx, cy, radius]).all() or radius < 1:
            continue
        label = str(item.get('label') or 'invalid_other')
        if label not in {'valid', 'invalid_crossing', 'invalid_other'}:
            label = 'invalid_other'
        candidate_id = str(item.get('candidate_id') or f'circle_{idx + 1}').strip() or f'circle_{idx + 1}'
        out.append({
            'candidate_id': candidate_id,
            'center_x': float(cx),
            'center_y': float(cy),
            'radius_px': float(radius),
            'label': label,
        })
    return out


def _loco_dataset_circles_from_legacy_points(image_id: str) -> list[dict[str, Any]]:
    try:
        state = load_points(image_id)
    except Exception:
        return []
    circles: list[dict[str, Any]] = []
    for idx, point in enumerate(list(state.get('points') or [])):
        circle_type = str(point.get('circle_type') or '')
        radius = _finite_float(point.get('radius_px'), 0.0) or 0.0
        if radius < 1 or circle_type not in {'valid', 'crossing', 'other_valid'}:
            continue
        label = 'invalid_crossing' if circle_type == 'crossing' else ('invalid_other' if circle_type == 'other_valid' else 'valid')
        circles.append({
            'candidate_id': f'legacy_{_safe_id(image_id)}_{idx}',
            'center_x': point.get('x'),
            'center_y': point.get('y'),
            'radius_px': radius,
            'label': label,
        })
    return normalize_loco_dataset_circles(circles)


def save_loco_dataset_circles(
    image_id: str,
    circles: list[dict[str, Any]],
    active_circle_id: str = '',
) -> dict[str, Any]:
    ensure_dirs()
    normalized = normalize_loco_dataset_circles(circles)
    active = str(active_circle_id or '').strip()
    if active and not any(str(item.get('candidate_id')) == active for item in normalized):
        active = ''
    payload = {
        'image_id': str(image_id),
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'circles': normalized,
        'active_circle_id': active,
    }
    _write_json(_loco_dataset_circles_path(image_id), payload)
    return payload


def load_loco_dataset_circles(image_id: str, migrate_legacy: bool = True) -> dict[str, Any]:
    ensure_dirs()
    path = _loco_dataset_circles_path(image_id)
    if path.exists():
        payload = _read_json(path)
        circles = normalize_loco_dataset_circles(list(payload.get('circles') or []))
        active = str(payload.get('active_circle_id') or '').strip()
        if active and not any(str(item.get('candidate_id')) == active for item in circles):
            active = ''
        payload['found'] = True
        payload['image_id'] = str(payload.get('image_id') or image_id)
        payload['circles'] = circles
        payload['active_circle_id'] = active
        return payload
    if migrate_legacy:
        legacy = _loco_dataset_circles_from_legacy_points(image_id)
        if legacy:
            migrated = save_loco_dataset_circles(image_id, legacy, '')
            migrated['found'] = True
            migrated['migrated_from_points'] = True
            return migrated
    return {'found': False, 'image_id': str(image_id), 'circles': [], 'active_circle_id': ''}


def clear_loco_dataset_circles(image_id: str) -> dict[str, Any]:
    return save_loco_dataset_circles(image_id, [], '')


def new_run_id(method_id: str = METHOD_ID) -> str:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    token = hashlib.sha1(f'{method_id}_{stamp}'.encode('utf-8')).hexdigest()[:8]
    return f'diam_{method_id}_{stamp}_{token}'


def _image_index_path(image_id: str) -> Path:
    sid = _safe_id(image_id)
    if not sid:
        raise ValueError('image_id invalido para index')
    return INDEX_DIR / f'{sid}.json'


def save_diameter_run(art: DiameterRunArtifacts) -> dict[str, Any]:
    ensure_dirs()
    run_dir = RUNS_DIR / str(art.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)

    _write_png(run_dir / 'input_image.png', art.input_image)
    np.savez_compressed(run_dir / 'scribble_map.npz', scribble_map=np.asarray(art.scribble_labels, dtype=np.uint8))
    _write_png(run_dir / 'prior_prob.png', (np.clip(np.asarray(art.prior_prob, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8))
    _write_png(run_dir / 'support_region.png', (np.asarray(art.support_region) > 0).astype(np.uint8) * 255)
    _write_png(run_dir / 'overlay.png', art.overlay)

    meta = {
        'run_id': art.run_id,
        'image_id': art.image_id,
        'experiment_id': art.experiment_id,
        'created_at': art.created_at,
        'meta': art.meta,
        'results': art.results,
    }
    _write_json(run_dir / 'meta.json', meta)
    diagnostics = dict(art.diagnostics or {})
    diagnostics['results'] = art.results
    loco_overlay = diagnostics.pop('loco_overlay', None)
    if isinstance(loco_overlay, np.ndarray):
        _write_png(run_dir / 'loco_overlay.png', loco_overlay)
        diagnostics['loco_overlay_path'] = str(run_dir / 'loco_overlay.png')
    profiles_raw_v3 = diagnostics.pop('profiles_raw_v3', None)
    if isinstance(profiles_raw_v3, dict):
        arrays: dict[str, np.ndarray] = {}
        for key, value in profiles_raw_v3.items():
            try:
                arrays[str(key)] = np.asarray(value)
            except Exception:
                continue
        if arrays:
            np.savez_compressed(run_dir / 'profiles_raw_v3.npz', **arrays)
        diagnostics['profiles_raw_v3_path'] = str(run_dir / 'profiles_raw_v3.npz')
    _write_json(run_dir / 'diagnostics.json', diagnostics)
    diagnostics_v2 = diagnostics.get('diagnostics_v2')
    if isinstance(diagnostics_v2, dict):
        _write_json(run_dir / 'diagnostics_v2.json', diagnostics_v2)
    diagnostics_v3 = diagnostics.get('diagnostics_v3')
    if isinstance(diagnostics_v3, dict):
        _write_json(run_dir / 'diagnostics_v3.json', diagnostics_v3)
    diagnostics_loco = diagnostics.get('diagnostics_loco')
    if isinstance(diagnostics_loco, dict) and diagnostics_loco:
        _write_json(run_dir / 'diagnostics_loco.json', diagnostics_loco)

    ok_count = int(sum(1 for r in art.results if r.get('status') == 'ok'))
    idx_item = {
        'run_id': art.run_id,
        'created_at': art.created_at,
        'experiment_id': art.experiment_id,
        'method_id': str((art.meta or {}).get('method_id') or art.experiment_id),
        'method': art.experiment_id,
        'source_mode': str((art.meta or {}).get('source_mode') or ''),
        'point_count': int(len(art.results)),
        'points_ok': int(ok_count),
        'run_status_level': 'success' if ok_count > 0 else 'warning',
    }
    idx_path = _image_index_path(art.image_id)
    current = {'image_id': art.image_id, 'runs': []}
    if idx_path.exists():
        try:
            current = _read_json(idx_path)
        except Exception:
            current = {'image_id': art.image_id, 'runs': []}
    rows = list(current.get('runs') or [])
    rows.append(idx_item)
    _write_json(idx_path, {'image_id': art.image_id, 'runs': rows})

    return {
        'run_id': art.run_id,
        'image_id': art.image_id,
        'experiment_id': art.experiment_id,
        'created_at': art.created_at,
        'run_dir': str(run_dir),
    }


def list_diameter_runs(image_id: str) -> list[dict[str, Any]]:
    ensure_dirs()
    idx = _image_index_path(str(image_id or '').strip())
    if not idx.exists():
        return []
    payload = _read_json(idx)
    rows = list(payload.get('runs') or [])
    rows.sort(key=lambda x: str(x.get('created_at') or ''), reverse=True)
    return rows


def load_diameter_run(run_id: str) -> dict[str, Any]:
    ensure_dirs()
    run_dir = RUNS_DIR / str(run_id or '').strip()
    if not run_dir.exists():
        raise FileNotFoundError(f'Run no encontrado: {run_id}')
    meta = _read_json(run_dir / 'meta.json')
    diag_path = run_dir / 'diagnostics.json'
    diagnostics = _read_json(diag_path) if diag_path.exists() else {}
    diag_v2_path = run_dir / 'diagnostics_v2.json'
    if diag_v2_path.exists():
        diagnostics['diagnostics_v2'] = _read_json(diag_v2_path)
    diag_v3_path = run_dir / 'diagnostics_v3.json'
    if diag_v3_path.exists():
        diagnostics['diagnostics_v3'] = _read_json(diag_v3_path)
    diag_loco_path = run_dir / 'diagnostics_loco.json'
    if diag_loco_path.exists():
        diagnostics['diagnostics_loco'] = _read_json(diag_loco_path)
    loco_overlay_path = run_dir / 'loco_overlay.png'
    if loco_overlay_path.exists():
        diagnostics['loco_overlay_path'] = str(loco_overlay_path)
    profiles_raw_v3_path = run_dir / 'profiles_raw_v3.npz'
    if profiles_raw_v3_path.exists():
        diagnostics['profiles_raw_v3_path'] = str(profiles_raw_v3_path)
    z = np.load(str(run_dir / 'scribble_map.npz'))
    return {
        'run_id': str(meta.get('run_id') or run_id),
        'image_id': str(meta.get('image_id') or ''),
        'experiment_id': str(meta.get('experiment_id') or METHOD_ID),
        'created_at': str(meta.get('created_at') or ''),
        'meta': dict(meta.get('meta') or {}),
        'results': list(meta.get('results') or diagnostics.get('results') or []),
        'diagnostics': diagnostics,
        'input_image': _read_png(run_dir / 'input_image.png', grayscale=False),
        'scribble_map': np.asarray(z['scribble_map'], dtype=np.uint8),
        'prior_prob': _read_png(run_dir / 'prior_prob.png', grayscale=True).astype(np.float32) / 255.0,
        'support_region': (_read_png(run_dir / 'support_region.png', grayscale=True) > 0).astype(np.uint8),
        'overlay': _read_png(run_dir / 'overlay.png', grayscale=False),
    }
