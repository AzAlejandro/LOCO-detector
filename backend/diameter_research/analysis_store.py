from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .. import persistence as scribble_persistence
from ..library_store import list_library_images, normalize_structured_tags
from . import persistence as drp


ANALYSIS_ROOT = scribble_persistence.OUTPUT_ROOT / 'diameter_analysis'
MEASUREMENTS_PATH = ANALYSIS_ROOT / 'measurements.json'
ANALYSES_DIR = ANALYSIS_ROOT / 'analyses'
EXPORTS_DIR = ANALYSIS_ROOT / 'exports'


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _ensure() -> None:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not MEASUREMENTS_PATH.exists():
        MEASUREMENTS_PATH.write_text(json.dumps({'measurements': []}, ensure_ascii=False, indent=2), encoding='utf-8')


def _safe_id(text: str) -> str:
    raw = str(text or '').strip()
    out: list[str] = []
    for ch in raw:
        if ch.isalnum() or ch in {'_', '-'}:
            out.append(ch)
        else:
            out.append('_')
    return ''.join(out).strip('_')[:120] or f'item_{uuid4().hex[:8]}'


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding='utf-8')) or {})
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _measurement_id(run_id: str, result: dict[str, Any], index: int) -> str:
    parts = [
        str(run_id or ''),
        str(result.get('result_id') or ''),
        str(result.get('point_index', index)),
        str(result.get('manual_geometry_id') or result.get('interactive_geometry_id') or result.get('circle_square_geometry_id') or ''),
        str(result.get('x', '')),
        str(result.get('y', '')),
    ]
    return _safe_id('_'.join(parts))


def _image_meta_by_id() -> dict[str, dict[str, Any]]:
    return {str(item.get('image_id') or ''): item for item in list_library_images()}


def _tag_key(tag: dict[str, Any]) -> str:
    category = str(tag.get('category') or 'other').strip().lower()
    label = str(tag.get('label') or tag.get('value') or '').strip().lower()
    value = str(tag.get('value') or '').strip().lower()
    unit = str(tag.get('unit') or '').strip().lower()
    return f'{category}::{label}::{value}::{unit}'


def _matches_filters(item: dict[str, Any], filters: dict[str, Any]) -> bool:
    image_ids = [str(v) for v in filters.get('image_ids') or [] if str(v).strip()]
    if image_ids and str(item.get('image_id') or '') not in image_ids:
        return False
    project_ids = [str(v) for v in filters.get('project_ids') or [] if str(v).strip()]
    if project_ids:
        item_projects = set(str(v) for v in item.get('project_ids') or [])
        if not item_projects.intersection(project_ids):
            return False
    required_tags = normalize_structured_tags(filters.get('structured_tags') or [])
    if required_tags:
        item_tag_keys = {_tag_key(tag) for tag in normalize_structured_tags(item.get('structured_tags') or [])}
        if not all(_tag_key(tag) in item_tag_keys for tag in required_tags):
            return False
    return True


def _calibration_factor_nm(calibration: dict[str, Any] | None) -> float:
    if not calibration:
        return 0.0
    if not bool(calibration.get('enabled')):
        return 0.0
    factor = float(calibration.get('unit_per_px') or 0.0)
    if not np.isfinite(factor) or factor <= 0:
        return 0.0
    return factor * 1000.0 if str(calibration.get('unit') or 'nm') == 'um' else factor


def convert_from_px(diameter_px: Any, calibration: dict[str, Any] | None, unit: str) -> float | None:
    try:
        px = float(diameter_px)
    except Exception:
        return None
    if not np.isfinite(px):
        return None
    target = str(unit or 'px')
    if target == 'px':
        return px
    nm_per_px = _calibration_factor_nm(calibration)
    if nm_per_px <= 0:
        return None
    value_nm = px * nm_per_px
    return value_nm / 1000.0 if target == 'um' else value_nm


def _with_conversions(row: dict[str, Any]) -> dict[str, Any]:
    calibration = row.get('calibration') if isinstance(row.get('calibration'), dict) else {}
    diameter_px = row.get('diameter_px')
    diameter_nm = convert_from_px(diameter_px, calibration, 'nm')
    diameter_um = convert_from_px(diameter_px, calibration, 'um')
    out = dict(row)
    out['diameter_nm'] = diameter_nm
    out['diameter_um'] = diameter_um
    out['calibration_status'] = 'calibrated' if diameter_nm is not None else 'sin_calibracion'
    return out


def _load_measurements() -> list[dict[str, Any]]:
    _ensure()
    payload = _read_json(MEASUREMENTS_PATH)
    return [dict(item or {}) for item in list(payload.get('measurements') or [])]


def _save_measurements(rows: list[dict[str, Any]]) -> None:
    _write_json(MEASUREMENTS_PATH, {'updated_at': _now(), 'measurements': rows})


def update_measurements_calibration_for_image(image_id: str, calibration: dict[str, Any] | None) -> dict[str, Any]:
    target_image_id = str(image_id or '').strip()
    if not target_image_id:
        return {'image_id': target_image_id, 'updated_count': 0}
    rows = _load_measurements()
    updated_count = 0
    next_rows: list[dict[str, Any]] = []
    for item in rows:
        row = dict(item)
        if str(row.get('image_id') or '') == target_image_id:
            row['calibration'] = dict(calibration or {})
            row = _with_conversions(row)
            updated_count += 1
        next_rows.append(row)
    if updated_count:
        _save_measurements(next_rows)
    return {'image_id': target_image_id, 'updated_count': updated_count}


def save_measurements_from_run(run_id: str, calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    _ensure()
    run = drp.load_diameter_run(str(run_id or '').strip())
    image_id = str(run.get('image_id') or '')
    meta = _image_meta_by_id().get(image_id, {})
    existing = _load_measurements()
    by_id = {str(item.get('measurement_id') or ''): item for item in existing if item.get('measurement_id')}
    saved: list[dict[str, Any]] = []
    for idx, result in enumerate(list(run.get('results') or [])):
        if result.get('diameter_px') is None:
            continue
        measurement_id = _measurement_id(str(run.get('run_id') or run_id), result, idx)
        row = {
            'measurement_id': measurement_id,
            'image_id': image_id,
            'image_name': str(meta.get('image_name') or image_id),
            'run_id': str(run.get('run_id') or run_id),
            'created_at': str(run.get('created_at') or ''),
            'saved_at': _now(),
            'method_id': str(result.get('method_id') or (run.get('meta') or {}).get('method_id') or run.get('experiment_id') or ''),
            'status': str(result.get('status') or ''),
            'reason': str(result.get('reason') or ''),
            'point_index': int(result.get('point_index', idx)),
            'x': None if result.get('x') is None else float(result.get('x') or 0.0),
            'y': None if result.get('y') is None else float(result.get('y') or 0.0),
            'diameter_px': float(result.get('diameter_px') or 0.0),
            'confidence': float(result.get('confidence') or 0.0),
            'diameter_route': str(result.get('diameter_route') or result.get('size_route') or ''),
            'project_ids': list(meta.get('project_ids') or []),
            'tags': list(meta.get('tags') or []),
            'structured_tags': normalize_structured_tags(meta.get('structured_tags') or meta.get('tags') or []),
            'calibration': dict(calibration or {}),
        }
        row = _with_conversions(row)
        by_id[measurement_id] = row
        saved.append(row)
    merged = list(by_id.values())
    merged.sort(key=lambda item: (str(item.get('image_id') or ''), str(item.get('created_at') or ''), int(item.get('point_index') or 0)))
    _save_measurements(merged)
    return {'image_id': image_id, 'run_id': run_id, 'saved_count': len(saved), 'measurements': saved}


def _stats(values: list[float]) -> dict[str, Any]:
    clean = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if clean.size < 1:
        return {'n': 0}
    return {
        'n': int(clean.size),
        'mean': float(np.mean(clean)),
        'median': float(np.median(clean)),
        'std': float(np.std(clean)),
        'min': float(np.min(clean)),
        'max': float(np.max(clean)),
        'p10': float(np.percentile(clean, 10)),
        'p25': float(np.percentile(clean, 25)),
        'p75': float(np.percentile(clean, 75)),
        'p90': float(np.percentile(clean, 90)),
        'iqr': float(np.percentile(clean, 75) - np.percentile(clean, 25)),
    }


def query_measurements(filters: dict[str, Any] | None = None, unit: str = 'nm', include_uncalibrated: bool = False) -> dict[str, Any]:
    filters = dict(filters or {})
    target_unit = str(unit or 'nm')
    if target_unit not in {'px', 'nm', 'um'}:
        target_unit = 'nm'
    rows: list[dict[str, Any]] = []
    for item in _load_measurements():
        if not _matches_filters(item, filters):
            continue
        value = convert_from_px(item.get('diameter_px'), item.get('calibration') if isinstance(item.get('calibration'), dict) else {}, target_unit)
        if value is None and not include_uncalibrated:
            continue
        row = dict(item)
        row['diameter_value'] = value
        row['diameter_unit'] = target_unit
        rows.append(row)
    by_image: dict[str, list[float]] = {}
    for row in rows:
        value = row.get('diameter_value')
        if value is None:
            continue
        by_image.setdefault(str(row.get('image_id') or ''), []).append(float(value))
    image_metrics = [
        {
            'image_id': image_id,
            'image_name': next((str(row.get('image_name') or image_id) for row in rows if str(row.get('image_id') or '') == image_id), image_id),
            'unit': target_unit,
            'stats': _stats(values),
        }
        for image_id, values in sorted(by_image.items())
    ]
    all_values = [float(row['diameter_value']) for row in rows if row.get('diameter_value') is not None]
    return {
        'items': rows,
        'unit': target_unit,
        'image_metrics': image_metrics,
        'global_metrics': {'unit': target_unit, 'stats': _stats(all_values)},
        'uncalibrated_count': int(sum(1 for row in rows if row.get('diameter_value') is None)),
    }


def summarize_measurements_by_image() -> list[dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for item in _load_measurements():
        image_id = str(item.get('image_id') or '').strip()
        if not image_id:
            continue
        row = summary.setdefault(image_id, {
            'image_id': image_id,
            'image_name': str(item.get('image_name') or image_id),
            'measurement_count': 0,
            'ok_count': 0,
            'calibrated_count': 0,
            'uncalibrated_count': 0,
            'latest_saved_at': '',
        })
        row['measurement_count'] += 1
        if str(item.get('status') or '').lower() in {'ok', 'accepted', 'success'}:
            row['ok_count'] += 1
        if item.get('diameter_nm') is not None or str(item.get('calibration_status') or '') == 'calibrated':
            row['calibrated_count'] += 1
        else:
            row['uncalibrated_count'] += 1
        saved_at = str(item.get('saved_at') or '')
        if saved_at > str(row.get('latest_saved_at') or ''):
            row['latest_saved_at'] = saved_at
    return sorted(summary.values(), key=lambda row: str(row.get('image_name') or row.get('image_id') or ''))


def list_analyses() -> list[dict[str, Any]]:
    _ensure()
    items: list[dict[str, Any]] = []
    for path in ANALYSES_DIR.glob('*.json'):
        data = _read_json(path)
        if data:
            items.append(data)
    items.sort(key=lambda item: str(item.get('updated_at') or item.get('created_at') or ''), reverse=True)
    return items


def save_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    _ensure()
    name = str(payload.get('name') or '').strip() or f'Analisis {datetime.now().strftime("%Y%m%d_%H%M%S")}'
    analysis_id = str(payload.get('analysis_id') or '').strip() or f'analysis_{uuid4().hex[:10]}'
    filters = dict(payload.get('filters') or {})
    unit = str(payload.get('unit') or 'nm')
    query = query_measurements(filters, unit=unit, include_uncalibrated=bool(payload.get('include_uncalibrated')))
    now = _now()
    item = {
        'analysis_id': analysis_id,
        'name': name,
        'created_at': str(payload.get('created_at') or now),
        'updated_at': now,
        'project_ids': list(payload.get('project_ids') or filters.get('project_ids') or []),
        'structured_tags': normalize_structured_tags([
            {'category': 'analysis', 'label': name},
            *(payload.get('structured_tags') or filters.get('structured_tags') or []),
        ]),
        'filters': filters,
        'unit': unit,
        'chart_config': dict(payload.get('chart_config') or {}),
        'summary': {
            'measurement_count': len(query.get('items') or []),
            'image_count': len(query.get('image_metrics') or []),
            'global_metrics': query.get('global_metrics') or {},
        },
    }
    _write_json(ANALYSES_DIR / f'{_safe_id(analysis_id)}.json', item)
    return item


def _rewrite_tag_items(items: Any, category: str, old_label: str, new_label: str = '', remove: bool = False) -> list[dict[str, Any]]:
    target = _tag_key({'category': category, 'label': old_label})
    out: list[dict[str, Any]] = []
    for tag in normalize_structured_tags(items):
        tag_category = str(tag.get('category') or 'other')
        label = str(tag.get('label') or '')
        if tag_category == 'size' and str(category) == 'unit':
            unit = str(tag.get('unit') or '')
            if _tag_key({'category': 'unit', 'label': unit}) == target:
                if remove:
                    tag.pop('unit', None)
                else:
                    tag['unit'] = str(new_label or '').strip()
                    tag['label'] = f"TamaÃ±o: {tag.get('value')}{' ' + tag['unit'] if tag.get('unit') else ''}"
        elif _tag_key({'category': tag_category, 'label': label}) == target:
            if remove:
                continue
            tag['label'] = str(new_label or '').strip()
        out.append(tag)
    return normalize_structured_tags(out)


def rewrite_analysis_tags(category: str, old_label: str, new_label: str = '', remove: bool = False) -> None:
    _ensure()
    measurements_changed = False
    measurements: list[dict[str, Any]] = []
    for row in _load_measurements():
        next_tags = _rewrite_tag_items(row.get('structured_tags') or [], category, old_label, new_label, remove)
        if next_tags != normalize_structured_tags(row.get('structured_tags') or []):
            row['structured_tags'] = next_tags
            row['tags'] = [str(tag.get('label') or tag.get('value') or '') for tag in next_tags if str(tag.get('label') or tag.get('value') or '').strip()]
            measurements_changed = True
        measurements.append(row)
    if measurements_changed:
        _save_measurements(measurements)
    for path in ANALYSES_DIR.glob('*.json'):
        item = _read_json(path)
        if not item:
            continue
        changed = False
        next_tags = _rewrite_tag_items(item.get('structured_tags') or [], category, old_label, new_label, remove)
        if next_tags != normalize_structured_tags(item.get('structured_tags') or []):
            item['structured_tags'] = next_tags
            changed = True
        filters = dict(item.get('filters') or {})
        filter_tags = _rewrite_tag_items(filters.get('structured_tags') or [], category, old_label, new_label, remove)
        if filter_tags != normalize_structured_tags(filters.get('structured_tags') or []):
            filters['structured_tags'] = filter_tags
            item['filters'] = filters
            changed = True
        if category == 'analysis' and str(item.get('name') or '').strip().lower() == str(old_label or '').strip().lower() and not remove:
            item['name'] = str(new_label or '').strip()
            changed = True
        if changed:
            item['updated_at'] = _now()
            _write_json(path, item)


def sync_image_metadata(image_id: str, project_ids: Any = None, tags: Any = None, structured_tags: Any = None) -> None:
    iid = str(image_id or '').strip()
    if not iid:
        return
    _ensure()
    normalized_structured = normalize_structured_tags(structured_tags or tags or [])
    normalized_tags = [str(tag.get('label') or tag.get('value') or '') for tag in normalized_structured if str(tag.get('label') or tag.get('value') or '').strip()]
    changed = False
    rows: list[dict[str, Any]] = []
    for row in _load_measurements():
        if str(row.get('image_id') or '') == iid:
            row['project_ids'] = [str(item) for item in list(project_ids or []) if str(item).strip()]
            row['structured_tags'] = normalized_structured
            row['tags'] = normalized_tags
            changed = True
        rows.append(row)
    if changed:
        _save_measurements(rows)


def export_analysis(filters: dict[str, Any] | None = None, unit: str = 'nm', include_uncalibrated: bool = False) -> dict[str, Any]:
    _ensure()
    query = query_measurements(filters or {}, unit=unit, include_uncalibrated=include_uncalibrated)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = EXPORTS_DIR / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / 'diameter_analysis.json'
    csv_path = out_dir / 'diameter_analysis.csv'
    _write_json(json_path, query)
    rows = list(query.get('items') or [])
    fieldnames = [
        'image_id', 'image_name', 'run_id', 'measurement_id', 'method_id', 'status', 'point_index',
        'x', 'y', 'diameter_px', 'diameter_value', 'diameter_unit', 'calibration_status',
    ]
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})
    return {
        'export_dir': str(out_dir),
        'summary_json': str(json_path),
        'summary_csv': str(csv_path),
        'measurement_count': len(rows),
        'image_count': len(query.get('image_metrics') or []),
    }
