from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np

from .image_codec import to_uint8_rgb
from .runner import RunArtifacts
from .scribble import (
    LABEL_BACKGROUND,
    LABEL_FIBER,
    LABEL_HALO,
    labels_to_visual,
    scribble_label_counts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / 'outputs' / 'scribble_research'
RUNS_DIR = OUTPUT_ROOT / 'runs'
REVIEWS_DIR = OUTPUT_ROOT / 'reviews'
INDEX_DIR = OUTPUT_ROOT / 'index'
REPORTS_DIR = OUTPUT_ROOT / 'reports'
DRAFTS_DIR = OUTPUT_ROOT / 'drafts'


def ensure_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_id(text: str) -> str:
    raw = str(text or '').strip()
    if not raw:
        return ''
    out = []
    for ch in raw:
        if ch.isalnum() or ch in {'_', '-'}:
            out.append(ch)
        else:
            out.append('_')
    return ''.join(out).strip('_')


def _draft_dir(image_id: str) -> Path:
    sid = _safe_id(image_id)
    if not sid:
        raise ValueError('image_id invalido para draft')
    return DRAFTS_DIR / sid


def _labels_to_visual(labels: np.ndarray) -> np.ndarray:
    return labels_to_visual(labels)


def save_scribble_draft(image_id: str, labels: np.ndarray) -> dict[str, Any]:
    ensure_dirs()
    draft_dir = _draft_dir(image_id)
    draft_dir.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(labels, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError('labels de draft debe ser matriz 2D')
    out = np.zeros_like(arr, dtype=np.uint8)
    out[arr == LABEL_FIBER] = LABEL_FIBER
    out[arr == LABEL_HALO] = LABEL_HALO
    out[arr == LABEL_BACKGROUND] = LABEL_BACKGROUND

    np.savez_compressed(draft_dir / 'scribble_map.npz', scribble_map=out)
    _write_png(draft_dir / 'scribble_preview.png', _labels_to_visual(out))

    counts = scribble_label_counts(out)
    updated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    meta = {
        'image_id': str(image_id),
        'updated_at': updated_at,
        'format_version': 'v3_multiclass_halo',
        'label_schema': {
            '0': 'unlabeled',
            '1': 'fiber',
            '2': 'halo',
            '3': 'background',
        },
        'n_fg': counts['fiber'],
        'n_halo': counts['halo'],
        'n_bg': counts['background'],
        'n_unlabeled': counts['unlabeled'],
        'shape_hw': [int(out.shape[0]), int(out.shape[1])],
    }
    (draft_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return meta


def load_scribble_draft(image_id: str) -> dict[str, Any]:
    ensure_dirs()
    draft_dir = _draft_dir(image_id)
    zpath = draft_dir / 'scribble_map.npz'
    mpath = draft_dir / 'meta.json'
    if not zpath.exists():
        return {'found': False, 'image_id': str(image_id)}

    z = np.load(str(zpath))
    labels = np.asarray(z['scribble_map'], dtype=np.uint8)
    out = np.zeros_like(labels, dtype=np.uint8)

    meta: dict[str, Any] = {'image_id': str(image_id)}
    if mpath.exists():
        try:
            meta = dict(json.loads(mpath.read_text(encoding='utf-8')) or {})
        except Exception:
            meta = {'image_id': str(image_id)}
    version = str(meta.get('format_version') or '')
    legacy_binary = (not version) or version.startswith('v1') or version.startswith('v2')
    out[labels == LABEL_FIBER] = LABEL_FIBER
    if legacy_binary:
        out[labels == LABEL_HALO] = LABEL_BACKGROUND
    else:
        out[labels == LABEL_HALO] = LABEL_HALO
        out[labels == LABEL_BACKGROUND] = LABEL_BACKGROUND

    counts = scribble_label_counts(out)
    meta.setdefault('image_id', str(image_id))
    meta.setdefault('updated_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    meta.setdefault('format_version', 'v3_multiclass_halo' if not legacy_binary else 'v2_canonical_migrated_to_v3')
    meta.setdefault(
        'label_schema',
        {
            '0': 'unlabeled',
            '1': 'fiber',
            '2': 'halo',
            '3': 'background',
        },
    )
    meta['n_fg'] = counts['fiber']
    meta['n_halo'] = counts['halo']
    meta['n_bg'] = counts['background']
    meta['n_unlabeled'] = counts['unlabeled']
    meta.setdefault('shape_hw', [int(out.shape[0]), int(out.shape[1])])
    return {'found': True, 'image_id': str(image_id), 'labels': out, 'meta': meta}


def clear_scribble_draft(image_id: str) -> dict[str, Any]:
    ensure_dirs()
    draft_dir = _draft_dir(image_id)
    existed = bool(draft_dir.exists())
    if draft_dir.exists():
        for p in draft_dir.glob('*'):
            if p.is_file():
                p.unlink(missing_ok=True)
        try:
            draft_dir.rmdir()
        except OSError:
            pass
    return {'image_id': str(image_id), 'cleared': bool(existed)}


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


def _image_index_path(image_id: str) -> Path:
    return INDEX_DIR / f'{image_id}.json'


def _profile_name_from_meta(meta: dict[str, Any]) -> str:
    return str((meta.get('params_effective') or {}).get('__profile_name') or '')


def _rewrite_reviews_without(image_id: str, run_ids: set[str]) -> None:
    if not run_ids:
        return
    path = _reviews_path(image_id)
    if not path.exists():
        return
    kept: list[str] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except Exception:
            kept.append(raw)
            continue
        if str(item.get('run_id') or '') in run_ids:
            continue
        kept.append(json.dumps(item, ensure_ascii=False))
    if kept:
        path.write_text('\n'.join(kept) + '\n', encoding='utf-8')
    else:
        path.unlink(missing_ok=True)


def _delete_run_dir(run_id: str) -> bool:
    rid = str(run_id or '').strip()
    if not rid:
        return False
    run_dir = RUNS_DIR / rid
    if not run_dir.exists():
        return False
    shutil.rmtree(run_dir)
    return True


def delete_runs_for_image(
    image_id: str,
    *,
    experiment_id: str | None = None,
    profile_name: str | None = None,
) -> dict[str, Any]:
    ensure_dirs()
    iid = str(image_id or '').strip()
    idx_path = _image_index_path(iid)
    if not idx_path.exists():
        return {'image_id': iid, 'deleted_count': 0, 'deleted_run_ids': []}
    try:
        payload = json.loads(idx_path.read_text(encoding='utf-8'))
    except Exception:
        payload = {'image_id': iid, 'runs': []}

    target_exp = str(experiment_id or '').strip()
    target_profile = str(profile_name or '').strip()
    keep: list[dict[str, Any]] = []
    deleted: list[str] = []
    for row_raw in list(payload.get('runs') or []):
        row = dict(row_raw or {})
        rid = str(row.get('run_id') or '').strip()
        row_exp = str(row.get('experiment_id') or '').strip()
        row_profile = str(row.get('profile_name') or '').strip()
        if not row_profile and rid:
            try:
                loaded = load_run(rid)
                row_profile = _profile_name_from_meta(dict(loaded.get('meta') or {}))
            except Exception:
                row_profile = ''
        exp_match = not target_exp or row_exp == target_exp
        profile_match = not target_profile or row_profile == target_profile
        if exp_match and profile_match:
            if rid:
                deleted.append(rid)
                _delete_run_dir(rid)
            continue
        keep.append(row)

    if keep:
        idx_path.write_text(json.dumps({'image_id': iid, 'runs': keep}, ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        idx_path.unlink(missing_ok=True)
    _rewrite_reviews_without(iid, set(deleted))
    return {'image_id': iid, 'deleted_count': len(deleted), 'deleted_run_ids': deleted}


def clear_results_for_image(image_id: str) -> dict[str, Any]:
    return delete_runs_for_image(image_id)


def save_run(art: RunArtifacts) -> dict[str, Any]:
    ensure_dirs()
    run_dir = RUNS_DIR / art.run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    _write_png(run_dir / 'input_image.png', art.input_image)
    np.savez_compressed(run_dir / 'scribble_map.npz', scribble_map=np.asarray(art.scribble_labels, dtype=np.uint8))
    _write_png(run_dir / 'prior_prob.png', (np.clip(art.prior_map, 0.0, 1.0) * 255.0).astype(np.uint8))
    if art.class_prob_maps:
        clean_maps: dict[str, np.ndarray] = {}
        for key, value in dict(art.class_prob_maps or {}).items():
            arr = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
            if arr.shape[:2] == np.asarray(art.prior_map).shape[:2]:
                clean_maps[str(key)] = arr
                _write_png(run_dir / f'{str(key)}_prob.png', (arr * 255.0).astype(np.uint8))
        if clean_maps:
            np.savez_compressed(run_dir / 'class_prob_maps.npz', **clean_maps)
    _write_png(run_dir / 'mask.png', (np.asarray(art.mask) > 0).astype(np.uint8) * 255)
    _write_png(run_dir / 'overlay.png', art.overlay)

    meta = {
        'run_id': art.run_id,
        'image_id': art.image_id,
        'experiment_id': art.experiment_id,
        'created_at': art.created_at,
        'meta': art.meta,
    }
    (run_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    exp_meta = dict((art.meta or {}).get('experiment') or {})
    profile_name = _profile_name_from_meta(dict(art.meta or {}))
    idx_item = {
        'run_id': art.run_id,
        'created_at': art.created_at,
        'experiment_id': art.experiment_id,
        'profile_name': profile_name,
        'group': str(exp_meta.get('group') or ''),
        'display_name': str(exp_meta.get('display_name') or ''),
        'implementation_status': str(exp_meta.get('implementation_status') or ''),
        'run_status_level': str((art.meta or {}).get('run_status_level') or 'success'),
    }

    idx_path = _image_index_path(art.image_id)
    current: dict[str, Any] = {'image_id': art.image_id, 'runs': []}
    if idx_path.exists():
        try:
            current = json.loads(idx_path.read_text(encoding='utf-8'))
        except Exception:
            current = {'image_id': art.image_id, 'runs': []}
    runs = list(current.get('runs') or [])
    runs.append(idx_item)
    current = {'image_id': art.image_id, 'runs': runs}
    idx_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding='utf-8')

    return {
        'run_id': art.run_id,
        'image_id': art.image_id,
        'experiment_id': art.experiment_id,
        'created_at': art.created_at,
        'run_dir': str(run_dir),
    }


def list_runs(image_id: str) -> list[dict[str, Any]]:
    ensure_dirs()
    idx = _image_index_path(str(image_id or '').strip())
    if not idx.exists():
        return []
    payload = json.loads(idx.read_text(encoding='utf-8'))
    rows = list(payload.get('runs') or [])
    rows.sort(key=lambda x: str(x.get('created_at') or ''), reverse=True)
    return rows


def load_run(run_id: str) -> dict[str, Any]:
    ensure_dirs()
    run_dir = RUNS_DIR / str(run_id or '').strip()
    if not run_dir.exists():
        raise FileNotFoundError(f'Run no encontrado: {run_id}')

    meta_path = run_dir / 'meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'meta.json no encontrado en run {run_id}')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))

    z = np.load(str(run_dir / 'scribble_map.npz'))
    scribble = np.asarray(z['scribble_map'], dtype=np.uint8)
    run_meta = dict(meta.get('meta') or {})
    if not run_meta.get('scribble_label_schema') and np.any(scribble == LABEL_HALO) and not np.any(scribble == LABEL_BACKGROUND):
        scribble = np.where(scribble == LABEL_HALO, LABEL_BACKGROUND, scribble).astype(np.uint8)
    class_prob_maps: dict[str, np.ndarray] = {}
    cpath = run_dir / 'class_prob_maps.npz'
    if cpath.exists():
        try:
            zc = np.load(str(cpath))
            for key in zc.files:
                class_prob_maps[str(key)] = np.asarray(zc[key], dtype=np.float32)
        except Exception:
            class_prob_maps = {}

    return {
        'run_id': meta.get('run_id', run_id),
        'image_id': meta.get('image_id', ''),
        'experiment_id': meta.get('experiment_id', ''),
        'created_at': meta.get('created_at', ''),
        'meta': run_meta,
        'input_image': _read_png(run_dir / 'input_image.png', grayscale=False),
        'scribble_map': scribble,
        'prior_prob': _read_png(run_dir / 'prior_prob.png', grayscale=True).astype(np.float32) / 255.0,
        'class_prob_maps': class_prob_maps,
        'mask': (_read_png(run_dir / 'mask.png', grayscale=True) > 0).astype(np.uint8),
        'overlay': _read_png(run_dir / 'overlay.png', grayscale=False),
    }


def _reviews_path(image_id: str) -> Path:
    return REVIEWS_DIR / f'{image_id}.jsonl'


def append_review(image_id: str, run_id: str, experiment_id: str, decision: str, note: str = '') -> dict[str, Any]:
    ensure_dirs()
    item = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image_id': str(image_id),
        'run_id': str(run_id),
        'experiment_id': str(experiment_id),
        'decision': str(decision).lower(),
        'note': str(note or ''),
    }
    with _reviews_path(image_id).open('a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
    return item


def _read_reviews_raw(image_id: str) -> list[dict[str, Any]]:
    ensure_dirs()
    path = _reviews_path(image_id)
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except Exception:
            continue
    items.sort(key=lambda x: str(x.get('timestamp') or ''), reverse=True)
    return items


def _latest_review_map(reviews: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in reviews:
        rid = str(r.get('run_id') or '')
        if not rid or rid in out:
            continue
        out[rid] = r
    return out


def list_reviews(image_id: str) -> list[dict[str, Any]]:
    raw = _read_reviews_raw(image_id)
    dedup = _latest_review_map(raw)
    items = list(dedup.values())
    items.sort(key=lambda x: str(x.get('timestamp') or ''), reverse=True)
    return items


def export_report(image_id: str) -> dict[str, Any]:
    ensure_dirs()
    runs = list_runs(image_id)
    reviews_raw = _read_reviews_raw(image_id)
    review_map = _latest_review_map(reviews_raw)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = REPORTS_DIR / f'{image_id}_{stamp}'
    out_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    loaded: list[dict[str, Any]] = []
    for r in runs:
        run_id = str(r.get('run_id') or '')
        if not run_id:
            continue
        try:
            item = load_run(run_id)
        except Exception:
            continue
        loaded.append(item)
        m = item.get('meta', {})
        exp = dict(m.get('experiment') or {})
        op = dict((m.get('metrics') or {}).get('operational') or {})
        gt = dict((m.get('metrics') or {}).get('gt') or {})
        rev = review_map.get(run_id, {})

        rows.append(
            {
                'run_id': run_id,
                'experiment_id': item.get('experiment_id', ''),
                'group': exp.get('group', ''),
                'implementation_status': exp.get('implementation_status', ''),
                'created_at': item.get('created_at', ''),
                'decision_latest': rev.get('decision', ''),
                'note': rev.get('note', ''),
                'run_status_level': m.get('run_status_level', ''),
                'blocker_reason': m.get('blocker_reason', ''),
                'feature_profile': m.get('feature_profile', ''),
                'aux_score': m.get('aux_score', ''),
                'runtime_ms': op.get('runtime_ms', ''),
                'mask_area_px': op.get('mask_area_px', ''),
                'compactness': op.get('compactness', ''),
                'border_touch_ratio': op.get('border_touch_ratio', ''),
                'components_count': op.get('components_count', ''),
                'fragmentation_index': op.get('fragmentation_index', ''),
                'leakage_to_bg': op.get('leakage_to_bg', ''),
                'dice': gt.get('dice', ''),
                'iou': gt.get('iou', ''),
            }
        )

    csv_path = out_dir / 'summary.csv'
    if rows:
        fields = list(rows[0].keys())
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow(row)
    else:
        csv_path.write_text('', encoding='utf-8')

    json_path = out_dir / 'summary.json'
    summary_payload = {
        'image_id': image_id,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'rows': rows,
        'reviews_latest': list(review_map.values()),
        'reviews_raw': reviews_raw,
    }
    json_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding='utf-8')

    gallery_path = out_dir / 'gallery.png'
    _build_gallery(loaded, review_map, gallery_path)

    return {
        'image_id': image_id,
        'report_dir': str(out_dir),
        'summary_csv': str(csv_path),
        'summary_json': str(json_path),
        'gallery_png': str(gallery_path),
        'runs_count': len(rows),
    }


def _build_gallery(loaded_runs: list[dict[str, Any]], review_map: dict[str, dict[str, Any]], out_path: Path) -> None:
    if not loaded_runs:
        blank = np.full((240, 640, 3), 245, dtype=np.uint8)
        cv2.putText(blank, 'No runs disponibles', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
        _write_png(out_path, blank)
        return

    cards: list[np.ndarray] = []
    for run in loaded_runs:
        rid = str(run.get('run_id', ''))
        exp_id = str(run.get('experiment_id', ''))
        meta = dict(run.get('meta') or {})
        exp_meta = dict(meta.get('experiment') or {})
        status = str(meta.get('run_status_level', 'success')).upper()
        decision = str((review_map.get(rid) or {}).get('decision', '')).upper()

        base = to_uint8_rgb(run['input_image'])
        overlay = to_uint8_rgb(run['overlay'])
        mask = ((np.asarray(run['mask']) > 0).astype(np.uint8) * 255)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        h, w = base.shape[:2]
        target_h = 220
        target_w = int(round(w * (target_h / max(1, h))))
        b = cv2.resize(base, (target_w, target_h), interpolation=cv2.INTER_AREA)
        o = cv2.resize(overlay, (target_w, target_h), interpolation=cv2.INTER_AREA)
        m = cv2.resize(mask_rgb, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        row = np.hstack([b, o, m])
        pad = np.full((70, row.shape[1], 3), 250, dtype=np.uint8)
        cv2.putText(pad, f"{exp_id} | {rid}", (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(
            pad,
            f"G:{exp_meta.get('group','-')} | impl:{exp_meta.get('implementation_status','-')} | status:{status}",
            (12, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
        if decision:
            good = decision in {'S', 'A', 'OK'}
            unusable = decision in {'UNUSABLE', 'BAD'}
            color = (20, 150, 20) if good else ((200, 30, 30) if unusable else (180, 120, 20))
            cv2.putText(pad, f'DECISION: {decision}', (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
        cards.append(np.vstack([pad, row]))

    gallery = np.vstack(cards)
    _write_png(out_path, gallery)
