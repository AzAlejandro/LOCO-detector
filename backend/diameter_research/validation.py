from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from . import persistence as drp


CATEGORIES = {
    'borde_limpio',
    'halo_moderado',
    'halo_fuerte',
    'interseccion',
    'fibra_curva',
    'bajo_contraste',
    'mala_segmentacion',
    'otro',
}


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _root() -> Path:
    root = drp.OUTPUT_ROOT / 'validation'
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cases_path() -> Path:
    return _root() / 'cases.json'


def _exports_dir() -> Path:
    path = _root() / 'exports'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_id(text: str) -> str:
    return drp._safe_id(text)


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


def _read_all() -> dict[str, Any]:
    path = _cases_path()
    if not path.exists():
        return {'cases': []}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        cases = list(data.get('cases') or [])
        return {'cases': cases}
    except Exception:
        return {'cases': []}


def _write_all(data: dict[str, Any]) -> None:
    _cases_path().write_text(json.dumps(_json_ready(data), ensure_ascii=False, indent=2), encoding='utf-8')


def _case_auto_id(image_id: str, x: float, y: float) -> str:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{_safe_id(image_id)}_p{int(round(x))}_{int(round(y))}_{stamp}'


def _coerce_float(value: Any) -> float | None:
    if value is None or value == '':
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _normalize_category(value: str) -> str:
    cat = str(value or '').strip().lower() or 'otro'
    return cat if cat in CATEGORIES else 'otro'


def _normalize_quality(value: str) -> str:
    q = str(value or '').strip().lower() or 'medium'
    return q if q in {'high', 'medium', 'low'} else 'medium'


def _normalize_measurement_decision(value: str) -> str:
    d = str(value or '').strip().lower() or 'unreviewed'
    return d if d in {'unreviewed', 'validated', 'rejected', 'uncertain'} else 'unreviewed'


def list_cases(image_id: str | None = None) -> list[dict[str, Any]]:
    rows = list(_read_all().get('cases') or [])
    if image_id:
        rows = [c for c in rows if str(c.get('image_id') or '') == str(image_id)]
    rows.sort(key=lambda c: str(c.get('updated_at') or c.get('created_at') or ''), reverse=True)
    return rows


def get_case(case_id: str) -> dict[str, Any] | None:
    cid = str(case_id or '').strip()
    for case in list(_read_all().get('cases') or []):
        if str(case.get('case_id') or '') == cid:
            return dict(case)
    return None


def upsert_case(payload: dict[str, Any]) -> dict[str, Any]:
    image_id = str(payload.get('image_id') or '').strip()
    if not image_id:
        raise ValueError('image_id requerido para caso de validacion')
    point = dict(payload.get('point') or {})
    x = _coerce_float(point.get('x'))
    y = _coerce_float(point.get('y'))
    if x is None or y is None:
        raise ValueError('point.x/point.y requeridos para caso de validacion')
    case_id = _safe_id(str(payload.get('case_id') or '').strip()) or _case_auto_id(image_id, x, y)
    now = _now()
    data = _read_all()
    cases = list(data.get('cases') or [])
    current = None
    rest = []
    for case in cases:
        if str(case.get('case_id') or '') == case_id:
            current = dict(case)
        else:
            rest.append(case)
    if current is None:
        current = {
            'case_id': case_id,
            'created_at': now,
            'runs': {},
        }
    current.update(
        {
            'case_id': case_id,
            'image_id': image_id,
            'image_name': str(payload.get('image_name') or current.get('image_name') or ''),
            'updated_at': now,
            'point': {'x': float(x), 'y': float(y)},
            'category': _normalize_category(str(payload.get('category') or current.get('category') or 'otro')),
            'quality_manual': _normalize_quality(str(payload.get('quality_manual') or current.get('quality_manual') or 'medium')),
            'manual_diameter_px': _coerce_float(payload.get('manual_diameter_px')),
            'manual_left_x': _coerce_float(payload.get('manual_left_x')),
            'manual_left_y': _coerce_float(payload.get('manual_left_y')),
            'manual_right_x': _coerce_float(payload.get('manual_right_x')),
            'manual_right_y': _coerce_float(payload.get('manual_right_y')),
            'measurement_decision': _normalize_measurement_decision(
                str(payload.get('measurement_decision') or current.get('measurement_decision') or 'unreviewed')
            ),
            'notes': str(payload.get('notes') or ''),
            'result_comment': str(payload.get('result_comment') or ''),
            'source_mode': str(payload.get('source_mode') or current.get('source_mode') or 'prior'),
            'prior_run_id': str(payload.get('prior_run_id') or current.get('prior_run_id') or ''),
            'params': dict(payload.get('params') or current.get('params') or {}),
        }
    )
    current.setdefault('runs', {})
    current['metrics'] = case_metrics(current)
    rest.append(current)
    _write_all({'cases': rest})
    return current


def _result_first(payload: dict[str, Any]) -> dict[str, Any]:
    results = list(payload.get('results') or [])
    return dict(results[0] if results else {})


def _error_for(manual: float | None, diameter: float | None) -> dict[str, Any]:
    if manual is None or diameter is None or manual <= 0:
        return {'absolute_error_px': None, 'relative_error_pct': None, 'useful': None}
    err = abs(float(diameter) - float(manual))
    rel = 100.0 * err / max(float(manual), 1e-9)
    useful = bool(err <= max(3.0, 0.10 * float(manual)))
    return {'absolute_error_px': float(err), 'relative_error_pct': float(rel), 'useful': useful}


def run_snapshot(run_payload: dict[str, Any], *, params: dict[str, Any] | None = None, source_mode: str = '') -> dict[str, Any]:
    result = _result_first(run_payload)
    method_id = str(run_payload.get('method_id') or result.get('method_id') or run_payload.get('experiment_id') or '')
    diameter = _coerce_float(result.get('diameter_px'))
    return {
        'run_id': str(run_payload.get('run_id') or ''),
        'created_at': str(run_payload.get('created_at') or _now()),
        'method_id': method_id,
        'experiment_id': str(run_payload.get('experiment_id') or method_id),
        'source_mode': str(source_mode or (run_payload.get('meta') or {}).get('source_mode') or ''),
        'params': dict(params or {}),
        'diameter_px': diameter,
        'status': str(result.get('status') or ''),
        'reason': str(result.get('reason') or ''),
        'quality_label': str(result.get('quality_label') or ''),
        'confidence': _coerce_float(result.get('confidence')),
        'stability_score': _coerce_float(result.get('stability_score')),
        'recenter_shift_px': _coerce_float(result.get('recenter_shift_px')),
        'orientation_delta_deg': _coerce_float(result.get('orientation_delta_deg')),
        'measurement_mode': str(result.get('measurement_mode') or ''),
        'used_upscale': bool(result.get('used_upscale', False)),
        'scale_factor': int(result.get('scale_factor') or 1),
        'orientation_coherence': _coerce_float(result.get('orientation_coherence')),
        'geometry_status': str(result.get('geometry_status') or ''),
        'support_status': str(result.get('support_status') or ''),
        'small_diameter_suspect': bool(result.get('small_diameter_suspect', False)),
        'edge_pair_score': _coerce_float(result.get('edge_pair_score')),
        'profile_consensus': _coerce_float(result.get('profile_consensus')),
        'methodology_id': str(result.get('methodology_id') or ''),
        'local_context_label': str(result.get('local_context_label') or ''),
        'methodology_reason': str(result.get('methodology_reason') or ''),
        'selected_edge_policy': str(result.get('selected_edge_policy') or ''),
        'size_route': str(result.get('size_route') or ''),
        'halo_status': str(result.get('halo_status') or ''),
        'ridge_anchor_status': str(result.get('ridge_anchor_status') or ''),
        'flux_status': str(result.get('flux_status') or ''),
        'contour_refine_status': str(result.get('contour_refine_status') or ''),
        'curvelet_status': str(result.get('curvelet_status') or ''),
        'valid_profiles': int(result.get('valid_profiles') or 0),
        'total_profiles': int(result.get('total_profiles') or 0),
        'point_index': int(result.get('point_index') or 0),
        'result': result,
        'trace': {
            'meta': dict(run_payload.get('meta') or {}),
            'diagnostics_keys': sorted(list(dict(run_payload.get('diagnostics') or {}).keys())),
        },
    }


def attach_run(case_id: str, run_payload: dict[str, Any], *, params: dict[str, Any] | None = None, source_mode: str = '') -> dict[str, Any]:
    case = get_case(case_id)
    if case is None:
        raise ValueError(f'Caso no encontrado: {case_id}')
    snapshot = run_snapshot(run_payload, params=params, source_mode=source_mode)
    manual = _coerce_float(case.get('manual_diameter_px'))
    snapshot.update(_error_for(manual, snapshot.get('diameter_px')))
    runs = dict(case.get('runs') or {})
    runs[str(snapshot.get('method_id') or snapshot.get('experiment_id') or 'unknown')] = snapshot
    case['runs'] = runs
    case['updated_at'] = _now()
    case['metrics'] = case_metrics(case)
    data = _read_all()
    rows = [c for c in list(data.get('cases') or []) if str(c.get('case_id') or '') != str(case_id)]
    rows.append(case)
    _write_all({'cases': rows})
    return case


def case_metrics(case: dict[str, Any]) -> dict[str, Any]:
    runs = dict(case.get('runs') or {})
    manual = _coerce_float(case.get('manual_diameter_px'))
    out: dict[str, Any] = {'manual_diameter_px': manual}
    for method_id, run in runs.items():
        out[str(method_id)] = _error_for(manual, _coerce_float(run.get('diameter_px')))
    return out


def _accepted(run: dict[str, Any]) -> bool:
    return _coerce_float(run.get('diameter_px')) is not None and str(run.get('status') or '').lower() != 'rejected'


def _method_summary(cases: list[dict[str, Any]], method_id: str) -> dict[str, Any]:
    total = len(cases)
    runs = [dict((c.get('runs') or {}).get(method_id) or {}) for c in cases]
    present = [r for r in runs if r.get('run_id')]
    accepted = [r for r in present if _accepted(r)]
    rejected = [r for r in present if r.get('run_id') and not _accepted(r)]
    errs = [_coerce_float(r.get('absolute_error_px')) for r in accepted]
    errs = [float(e) for e in errs if e is not None]
    rels = [_coerce_float(r.get('relative_error_pct')) for r in accepted]
    rels = [float(e) for e in rels if e is not None]
    useful = [r for r in accepted if bool(r.get('useful'))]
    return {
        'method_id': method_id,
        'total_cases': int(total),
        'runs_present': int(len(present)),
        'accepted': int(len(accepted)),
        'rejected': int(len(rejected)),
        'coverage_pct': float(100.0 * len(accepted) / total) if total else 0.0,
        'abstention_pct': float(100.0 * len(rejected) / total) if total else 0.0,
        'useful_coverage_pct': float(100.0 * len(useful) / total) if total else 0.0,
        'mae_px': float(np.mean(errs)) if errs else None,
        'median_error_px': float(median(errs)) if errs else None,
        'p90_error_px': float(np.percentile(np.asarray(errs), 90)) if errs else None,
        'mape_pct': float(np.mean(rels)) if rels else None,
    }


def validation_summary(image_id: str | None = None) -> dict[str, Any]:
    cases = list_cases(image_id)
    methods = [
        'hybrid_profile_diameter_v1',
        'hybrid_profile_diameter_v2',
        'hybrid_profile_diameter_v3_1',
        'hybrid_profile_diameter_v3_2',
        'hybrid_profile_diameter_v3_3a',
        'hybrid_profile_diameter_v3_3b',
        'hybrid_profile_diameter_v3_3c',
        'hybrid_profile_diameter_v3_3d',
        'hybrid_profile_diameter_v3_2_halo_aware',
        'hybrid_profile_diameter_v3_2_small_large',
        'hybrid_profile_diameter_v3_2_ridge_anchored',
        'hybrid_profile_diameter_v3_2_flux_aware',
        'hybrid_profile_diameter_v3_2_contour_refine',
        'hybrid_profile_diameter_v3_2_curvelet_aided',
        'hybrid_profile_diameter_v3_2_auto',
        'hybrid_profile_diameter_v3_2_small_mask',
        'hybrid_profile_diameter_v3_2_large_image',
        'circle_square_mask_diameter',
        'manual_dual_side_caliper',
        'ellipse_oriented_fit',
        'loco_circle_probe',
    ]
    by_category: list[dict[str, Any]] = []
    categories = sorted(set([str(c.get('category') or 'otro') for c in cases]) | CATEGORIES)
    for cat in categories:
        cat_cases = [c for c in cases if str(c.get('category') or 'otro') == cat]
        if not cat_cases:
            continue
        for method in methods:
            row = _method_summary(cat_cases, method)
            row['category'] = cat
            by_category.append(row)
    return {
        'image_id': str(image_id or ''),
        'created_at': _now(),
        'case_count': int(len(cases)),
        'methods': [_method_summary(cases, method) for method in methods],
        'by_category': by_category,
        'cases': cases,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.3f}'
    return str(value)


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        out.append('| ' + ' | '.join(_fmt(v).replace('\n', ' ') for v in row) + ' |')
    return '\n'.join(out)


def export_validation(image_id: str | None = None) -> dict[str, Any]:
    summary = validation_summary(image_id)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sid = _safe_id(image_id or 'all')
    out_dir = _exports_dir() / f'{sid}_{stamp}'
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / 'validation_summary.json'
    json_path.write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding='utf-8')

    csv_path = out_dir / 'validation_cases.csv'
    fields = [
        'case_id',
        'image_id',
        'category',
        'manual_diameter_px',
        'manual_left_x',
        'manual_left_y',
        'manual_right_x',
        'manual_right_y',
        'point_x',
        'point_y',
        'source_mode',
        'prior_run_id',
        'measurement_decision',
        'method_id',
        'run_id',
        'status',
        'diameter_px',
        'absolute_error_px',
        'relative_error_pct',
        'useful',
        'confidence',
        'quality_label',
        'methodology_id',
        'local_context_label',
        'methodology_reason',
        'selected_edge_policy',
        'size_route',
        'halo_status',
        'ridge_anchor_status',
        'flux_status',
        'contour_refine_status',
        'curvelet_status',
        'loco_best_radius_px',
        'loco_symmetry_score',
        'loco_intersection_count',
        'loco_recenter_shift_px',
        'loco_mode',
        'loco_seed_radius_px',
        'recenter_shift_px',
        'orientation_delta_deg',
        'stability_score',
        'notes',
        'result_comment',
    ]
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in summary['cases']:
            for method_id, run in dict(case.get('runs') or {}).items():
                point = dict(case.get('point') or {})
                writer.writerow(
                    {
                        'case_id': case.get('case_id', ''),
                        'image_id': case.get('image_id', ''),
                        'category': case.get('category', ''),
                        'manual_diameter_px': case.get('manual_diameter_px', ''),
                        'manual_left_x': case.get('manual_left_x', ''),
                        'manual_left_y': case.get('manual_left_y', ''),
                        'manual_right_x': case.get('manual_right_x', ''),
                        'manual_right_y': case.get('manual_right_y', ''),
                        'point_x': point.get('x', ''),
                        'point_y': point.get('y', ''),
                        'source_mode': case.get('source_mode', ''),
                        'prior_run_id': case.get('prior_run_id', ''),
                        'measurement_decision': case.get('measurement_decision', ''),
                        'method_id': method_id,
                        'run_id': run.get('run_id', ''),
                        'status': run.get('status', ''),
                        'diameter_px': run.get('diameter_px', ''),
                        'absolute_error_px': run.get('absolute_error_px', ''),
                        'relative_error_pct': run.get('relative_error_pct', ''),
                        'useful': run.get('useful', ''),
                        'confidence': run.get('confidence', ''),
                        'quality_label': run.get('quality_label', ''),
                        'methodology_id': run.get('methodology_id', ''),
                        'local_context_label': run.get('local_context_label', ''),
                        'methodology_reason': run.get('methodology_reason', ''),
                        'selected_edge_policy': run.get('selected_edge_policy', ''),
                        'size_route': run.get('size_route', ''),
                        'halo_status': run.get('halo_status', ''),
                        'ridge_anchor_status': run.get('ridge_anchor_status', ''),
                        'flux_status': run.get('flux_status', ''),
                        'contour_refine_status': run.get('contour_refine_status', ''),
                        'curvelet_status': run.get('curvelet_status', ''),
                        'loco_best_radius_px': run.get('loco_best_radius_px', ''),
                        'loco_symmetry_score': run.get('loco_symmetry_score', ''),
                        'loco_intersection_count': run.get('loco_intersection_count', ''),
                        'loco_recenter_shift_px': run.get('loco_recenter_shift_px', ''),
                        'loco_mode': run.get('loco_mode', ''),
                        'loco_seed_radius_px': run.get('loco_seed_radius_px', ''),
                        'recenter_shift_px': run.get('recenter_shift_px', ''),
                        'orientation_delta_deg': run.get('orientation_delta_deg', ''),
                        'stability_score': run.get('stability_score', ''),
                        'notes': case.get('notes', ''),
                        'result_comment': case.get('result_comment', ''),
                    }
                )

    md_path = out_dir / 'STEP4_AUTOFILL.md'
    method_rows = [
        [
            row['method_id'],
            row['total_cases'],
            row['runs_present'],
            row['accepted'],
            row['rejected'],
            row['coverage_pct'],
            row['abstention_pct'],
            row['useful_coverage_pct'],
            row['mae_px'],
            row['mape_pct'],
        ]
        for row in summary['methods']
    ]
    case_rows: list[list[Any]] = []
    for case in summary['cases']:
        point = dict(case.get('point') or {})
        v1 = dict((case.get('runs') or {}).get('hybrid_profile_diameter_v1') or {})
        v2 = dict((case.get('runs') or {}).get('hybrid_profile_diameter_v2') or {})
        v31 = dict((case.get('runs') or {}).get('hybrid_profile_diameter_v3_1') or {})
        v32 = dict((case.get('runs') or {}).get('hybrid_profile_diameter_v3_2') or {})
        v33a = dict((case.get('runs') or {}).get('hybrid_profile_diameter_v3_3a') or {})
        v33b = dict((case.get('runs') or {}).get('hybrid_profile_diameter_v3_3b') or {})
        v33c = dict((case.get('runs') or {}).get('hybrid_profile_diameter_v3_3c') or {})
        v33d = dict((case.get('runs') or {}).get('hybrid_profile_diameter_v3_3d') or {})
        halo = dict((case.get('runs') or {}).get('hybrid_profile_diameter_v3_2_halo_aware') or {})
        small = dict((case.get('runs') or {}).get('hybrid_profile_diameter_v3_2_small_large') or {})
        ridge = dict((case.get('runs') or {}).get('hybrid_profile_diameter_v3_2_ridge_anchored') or {})
        flux = dict((case.get('runs') or {}).get('hybrid_profile_diameter_v3_2_flux_aware') or {})
        contour = dict((case.get('runs') or {}).get('hybrid_profile_diameter_v3_2_contour_refine') or {})
        curvelet = dict((case.get('runs') or {}).get('hybrid_profile_diameter_v3_2_curvelet_aided') or {})
        loco = dict((case.get('runs') or {}).get('loco_circle_probe') or {})
        case_rows.append(
            [
                case.get('case_id', ''),
                case.get('category', ''),
                case.get('source_mode', ''),
                case.get('prior_run_id', ''),
                case.get('measurement_decision', ''),
                case.get('manual_diameter_px', ''),
                point.get('x', ''),
                point.get('y', ''),
                v1.get('diameter_px', ''),
                v1.get('absolute_error_px', ''),
                v2.get('status', ''),
                v2.get('diameter_px', ''),
                v2.get('absolute_error_px', ''),
                v2.get('quality_label', ''),
                v2.get('recenter_shift_px', ''),
                v31.get('measurement_mode', ''),
                v31.get('status', ''),
                v31.get('diameter_px', ''),
                v31.get('quality_label', ''),
                v32.get('measurement_mode', ''),
                v32.get('status', ''),
                v32.get('diameter_px', ''),
                v32.get('quality_label', ''),
                v33a.get('status', ''),
                v33a.get('diameter_px', ''),
                v33a.get('quality_label', ''),
                v33b.get('measurement_mode', ''),
                v33b.get('status', ''),
                v33b.get('diameter_px', ''),
                v33b.get('quality_label', ''),
                v33c.get('used_upscale', ''),
                v33c.get('status', ''),
                v33c.get('diameter_px', ''),
                v33c.get('quality_label', ''),
                v33d.get('status', ''),
                v33d.get('diameter_px', ''),
                v33d.get('quality_label', ''),
                halo.get('diameter_px', ''),
                halo.get('halo_status', ''),
                small.get('diameter_px', ''),
                small.get('size_route', ''),
                ridge.get('diameter_px', ''),
                ridge.get('ridge_anchor_status', ''),
                flux.get('diameter_px', ''),
                flux.get('flux_status', ''),
                contour.get('diameter_px', ''),
                contour.get('contour_refine_status', ''),
                curvelet.get('diameter_px', ''),
                curvelet.get('curvelet_status', ''),
                loco.get('measurement_mode', ''),
                loco.get('diameter_px', ''),
                loco.get('loco_best_radius_px', ''),
                loco.get('loco_symmetry_score', ''),
                case.get('result_comment', ''),
            ]
        )
    md = [
        '# Step 4 - Autorrelleno de validacion Diameter Research',
        '',
        f'Generado: {summary["created_at"]}',
        f'Image ID: {summary.get("image_id") or "all"}',
        f'Casos: {summary["case_count"]}',
        '',
        '## Resumen por metodo',
        '',
        _md_table(
            ['Metodo', 'Total', 'Runs', 'Aceptados', 'Rechazados', 'Coverage %', 'Abstention %', 'Useful coverage %', 'MAE px', 'MAPE %'],
            method_rows,
        ),
        '',
        '## Casos trazables',
        '',
        _md_table(
            [
                'case_id',
                'categoria',
                'source',
                'prior_run_id',
                'decision',
                'manual_px',
                'x',
                'y',
                'v1_px',
                'v1_err',
                'v2_status',
                'v2_px',
                'v2_err',
                'v2_quality',
                'shift_px',
                'v3.1_mode',
                'v3.1_status',
                'v3.1_px',
                'v3.1_quality',
                'v3.2_mode',
                'v3.2_status',
                'v3.2_px',
                'v3.2_quality',
                'v3.3a_status',
                'v3.3a_px',
                'v3.3a_quality',
                'v3.3b_mode',
                'v3.3b_status',
                'v3.3b_px',
                'v3.3b_quality',
                'v3.3c_upscale',
                'v3.3c_status',
                'v3.3c_px',
                'v3.3c_quality',
                'v3.3d_status',
                'v3.3d_px',
                'v3.3d_quality',
                'halo_px',
                'halo_status',
                'small_px',
                'small_route',
                'ridge_px',
                'ridge_status',
                'flux_px',
                'flux_status',
                'contour_px',
                'contour_status',
                'curvelet_px',
                'curvelet_status',
                'loco_mode',
                'loco_px',
                'loco_radius',
                'loco_symmetry',
                'comentario',
            ],
            case_rows,
        ),
        '',
        '## Pendiente humano',
        '',
        '- Completar o revisar `manual_diameter_px` donde falte.',
        '- Completar comentarios en `result_comment` para casos relevantes.',
        '- Revisar casos con `v2_status = rejected` y `quality_label` ambiguo.',
        '- Revisar casos con `recenter_shift_px` alto.',
        '',
        '## Decision log',
        '',
        '| Fecha | Decision | Evidencia | Consecuencia |',
        '| --- | --- | --- | --- |',
        '|  |  |  |  |',
        '',
    ]
    md_path.write_text('\n'.join(md), encoding='utf-8')

    return {
        'export_dir': str(out_dir),
        'summary_json': str(json_path),
        'cases_csv': str(csv_path),
        'autofill_md': str(md_path),
        'summary': summary,
    }
