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


def save_points(image_id: str, points: list[dict[str, Any]], active_point_idx: int | None = None) -> dict[str, Any]:
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
    }
    _write_json(_points_path(image_id), payload)
    return payload


def load_points(image_id: str) -> dict[str, Any]:
    ensure_dirs()
    path = _points_path(image_id)
    if not path.exists():
        return {'found': False, 'image_id': str(image_id), 'points': [], 'active_point_idx': -1}
    payload = _read_json(path)
    pts = normalize_points(list(payload.get('points') or []))
    active = int(payload.get('active_point_idx', 0 if pts else -1))
    if active < 0 or active >= len(pts):
        active = 0 if pts else -1
    payload['found'] = True
    payload['points'] = pts
    payload['active_point_idx'] = active
    payload.setdefault('image_id', str(image_id))
    return payload


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
