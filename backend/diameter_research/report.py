from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import persistence as drp


def _row_for_result(run: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    orientation = dict(result.get('orientation') or {})
    return {
        'image_id': str(run.get('image_id') or ''),
        'run_id': str(run.get('run_id') or ''),
        'created_at': str(run.get('created_at') or ''),
        'experiment_id': str(run.get('experiment_id') or ''),
        'method_id': str(result.get('method_id') or (run.get('meta') or {}).get('method_id') or run.get('experiment_id') or ''),
        'source_mode': str((run.get('meta') or {}).get('source_mode') or ''),
        'point_index': int(result.get('point_index', -1)),
        'x': float(result.get('x', 0.0)),
        'y': float(result.get('y', 0.0)),
        'original_x': '' if not result.get('original_xy') else float(result.get('original_xy')[0]),
        'original_y': '' if not result.get('original_xy') else float(result.get('original_xy')[1]),
        'recenter_shift_px': '' if result.get('recenter_shift_px') is None else float(result.get('recenter_shift_px')),
        'orientation_delta_deg': '' if result.get('orientation_delta_deg') is None else float(result.get('orientation_delta_deg')),
        'stability_score': '' if result.get('stability_score') is None else float(result.get('stability_score')),
        'quality_label': str(result.get('quality_label') or ''),
        'status': str(result.get('status') or ''),
        'reason': str(result.get('reason') or ''),
        'diameter_px': '' if result.get('diameter_px') is None else float(result.get('diameter_px')),
        'confidence': float(result.get('confidence') or 0.0),
        'valid_profiles': int(result.get('valid_profiles') or 0),
        'total_profiles': int(result.get('total_profiles') or 0),
        'mad_px': '' if result.get('mad_px') is None else float(result.get('mad_px')),
        'edge_score_mean': float(result.get('edge_score_mean') or 0.0),
        'orientation_source': str(orientation.get('source') or ''),
        'orientation_confidence': float(orientation.get('confidence') or 0.0),
        'measurement_mode': str(result.get('measurement_mode') or ''),
        'used_upscale': '' if result.get('used_upscale') is None else bool(result.get('used_upscale')),
        'scale_factor': '' if result.get('scale_factor') is None else int(result.get('scale_factor') or 1),
        'orientation_coherence': '' if result.get('orientation_coherence') is None else float(result.get('orientation_coherence') or 0.0),
        'geometry_status': str(result.get('geometry_status') or ''),
        'support_status': str(result.get('support_status') or ''),
        'small_diameter_suspect': '' if result.get('small_diameter_suspect') is None else bool(result.get('small_diameter_suspect')),
        'edge_pair_score': '' if result.get('edge_pair_score') is None else float(result.get('edge_pair_score') or 0.0),
        'profile_consensus': '' if result.get('profile_consensus') is None else float(result.get('profile_consensus') or 0.0),
        'geometry_control_status': str(result.get('geometry_control_status') or ''),
        'profile_length_effective_px': '' if result.get('profile_length_effective_px') is None else float(result.get('profile_length_effective_px') or 0.0),
        'context_width_px': '' if result.get('context_width_px') is None else float(result.get('context_width_px') or 0.0),
        'support_path_mean': '' if result.get('support_path_mean') is None else float(result.get('support_path_mean') or 0.0),
        'methodology_id': str(result.get('methodology_id') or ''),
        'local_context_label': str(result.get('local_context_label') or ''),
        'methodology_reason': str(result.get('methodology_reason') or ''),
        'selected_edge_policy': str(result.get('selected_edge_policy') or ''),
        'size_route': str(result.get('size_route') or ''),
        'fiber_size_mode': str(result.get('fiber_size_mode') or ''),
        'auto_size_reason': str(result.get('auto_size_reason') or ''),
        'diameter_route': str(result.get('diameter_route') or result.get('size_route') or ''),
        'mask_method': str(result.get('mask_method') or ''),
        'mask_confidence': '' if result.get('mask_confidence') is None else float(result.get('mask_confidence') or 0.0),
        'mask_center_shift_px': '' if result.get('mask_center_shift_px') is None else float(result.get('mask_center_shift_px') or 0.0),
        'mask_center_distance_px': '' if result.get('mask_center_distance_px') is None else float(result.get('mask_center_distance_px') or 0.0),
        'mask_caliper_diameter_px': '' if result.get('mask_caliper_diameter_px') is None else float(result.get('mask_caliper_diameter_px') or 0.0),
        'mask_raycast_diameter_px': '' if result.get('mask_raycast_diameter_px') is None else float(result.get('mask_raycast_diameter_px') or 0.0),
        'circle_radius_px': '' if result.get('circle_radius_px') is None else float(result.get('circle_radius_px') or 0.0),
        'square_half_length_px': '' if result.get('square_half_length_px') is None else float(result.get('square_half_length_px') or 0.0),
        'square_half_width_px': '' if result.get('square_half_width_px') is None else float(result.get('square_half_width_px') or 0.0),
        'square_samples_valid': '' if result.get('square_samples_valid') is None else int(result.get('square_samples_valid') or 0),
        'square_samples_total': '' if result.get('square_samples_total') is None else int(result.get('square_samples_total') or 0),
        'manual_input_diameter_px': '' if result.get('manual_input_diameter_px') is None else float(result.get('manual_input_diameter_px') or 0.0),
        'ellipse_major_px': '' if result.get('ellipse_major_px') is None else float(result.get('ellipse_major_px') or 0.0),
        'ellipse_minor_px': '' if result.get('ellipse_minor_px') is None else float(result.get('ellipse_minor_px') or 0.0),
        'ellipse_angle_deg': '' if result.get('ellipse_angle_deg') is None else float(result.get('ellipse_angle_deg') or 0.0),
        'loco_best_radius_px': '' if result.get('loco_best_radius_px') is None else float(result.get('loco_best_radius_px') or 0.0),
        'loco_symmetry_score': '' if result.get('loco_symmetry_score') is None else float(result.get('loco_symmetry_score') or 0.0),
        'loco_intersection_count': '' if result.get('loco_intersection_count') is None else int(result.get('loco_intersection_count') or 0),
        'loco_recenter_shift_px': '' if result.get('loco_recenter_shift_px') is None else float(result.get('loco_recenter_shift_px') or 0.0),
        'loco_mode': str(result.get('loco_mode') or ''),
        'loco_seed_radius_px': '' if result.get('loco_seed_radius_px') is None else float(result.get('loco_seed_radius_px') or 0.0),
        'halo_status': str(result.get('halo_status') or ''),
        'ridge_anchor_status': str(result.get('ridge_anchor_status') or ''),
        'flux_status': str(result.get('flux_status') or ''),
        'contour_refine_status': str(result.get('contour_refine_status') or ''),
        'curvelet_status': str(result.get('curvelet_status') or ''),
        'ridge_response': '' if result.get('ridge_response') is None else float(result.get('ridge_response') or 0.0),
        'edge_pair_center_offset_px': '' if result.get('edge_pair_center_offset_px') is None else float(result.get('edge_pair_center_offset_px') or 0.0),
        'axis_flux_score': '' if result.get('axis_flux_score') is None else float(result.get('axis_flux_score') or 0.0),
        'neighbor_flux_score': '' if result.get('neighbor_flux_score') is None else float(result.get('neighbor_flux_score') or 0.0),
        'curvelet_edge_score': '' if result.get('curvelet_edge_score') is None else float(result.get('curvelet_edge_score') or 0.0),
    }


def _make_gallery(runs: list[dict[str, Any]], out_path: Path) -> None:
    tiles: list[np.ndarray] = []
    for run in runs[:12]:
        overlay = np.asarray(run.get('overlay'))
        if overlay.ndim != 3:
            continue
        rgb = overlay[:, :, :3].copy()
        h, w = rgb.shape[:2]
        max_w = 520
        scale = min(1.0, max_w / max(1, w))
        if scale < 1.0:
            rgb = cv2.resize(rgb, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        header = np.full((34, rgb.shape[1], 3), 255, dtype=np.uint8)
        text = f"{run.get('run_id', '')} | ok {int((run.get('meta') or {}).get('points_ok', 0))}"
        cv2.putText(header, text[:90], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (19, 36, 55), 1, cv2.LINE_AA)
        tiles.append(np.vstack([header, rgb]))
    if not tiles:
        blank = np.full((120, 520, 3), 255, dtype=np.uint8)
        cv2.putText(blank, 'Sin runs de Diameter Research', (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (95, 114, 137), 1, cv2.LINE_AA)
        tiles = [blank]
    width = max(tile.shape[1] for tile in tiles)
    padded = []
    for tile in tiles:
        if tile.shape[1] == width:
            padded.append(tile)
            continue
        pad = np.full((tile.shape[0], width - tile.shape[1], 3), 255, dtype=np.uint8)
        padded.append(np.hstack([tile, pad]))
    gallery = np.vstack(padded)
    drp._write_png(out_path, gallery)


def export_diameter_report(image_id: str) -> dict[str, Any]:
    drp.ensure_dirs()
    rows_meta = drp.list_diameter_runs(image_id)
    runs: list[dict[str, Any]] = []
    for row in rows_meta:
        try:
            runs.append(drp.load_diameter_run(str(row.get('run_id') or '')))
        except Exception:
            continue

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = drp.REPORTS_DIR / f'{drp._safe_id(image_id)}_{stamp}'
    report_dir.mkdir(parents=True, exist_ok=True)

    flat_rows: list[dict[str, Any]] = []
    for run in runs:
        for result in list(run.get('results') or []):
            flat_rows.append(_row_for_result(run, result))

    csv_path = report_dir / 'summary.csv'
    fieldnames = list(flat_rows[0].keys()) if flat_rows else [
        'image_id',
        'run_id',
        'created_at',
        'experiment_id',
        'method_id',
        'source_mode',
        'point_index',
        'x',
        'y',
        'original_x',
        'original_y',
        'recenter_shift_px',
        'orientation_delta_deg',
        'stability_score',
        'quality_label',
        'status',
        'reason',
        'diameter_px',
        'confidence',
        'valid_profiles',
        'total_profiles',
        'mad_px',
        'edge_score_mean',
        'orientation_source',
        'orientation_confidence',
        'measurement_mode',
        'used_upscale',
        'scale_factor',
        'orientation_coherence',
        'geometry_status',
        'support_status',
        'small_diameter_suspect',
        'edge_pair_score',
        'profile_consensus',
        'geometry_control_status',
        'profile_length_effective_px',
        'context_width_px',
        'support_path_mean',
        'methodology_id',
        'local_context_label',
        'methodology_reason',
        'selected_edge_policy',
        'size_route',
        'fiber_size_mode',
        'auto_size_reason',
        'diameter_route',
        'mask_method',
        'mask_confidence',
        'mask_center_shift_px',
        'mask_center_distance_px',
        'mask_caliper_diameter_px',
        'mask_raycast_diameter_px',
        'circle_radius_px',
        'square_half_length_px',
        'square_half_width_px',
        'square_samples_valid',
        'square_samples_total',
        'manual_input_diameter_px',
        'ellipse_major_px',
        'ellipse_minor_px',
        'ellipse_angle_deg',
        'loco_best_radius_px',
        'loco_symmetry_score',
        'loco_intersection_count',
        'loco_recenter_shift_px',
        'loco_mode',
        'loco_seed_radius_px',
        'halo_status',
        'ridge_anchor_status',
        'flux_status',
        'contour_refine_status',
        'curvelet_status',
        'ridge_response',
        'edge_pair_center_offset_px',
        'axis_flux_score',
        'neighbor_flux_score',
        'curvelet_edge_score',
    ]
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    json_path = report_dir / 'summary.json'
    json_payload = {
        'image_id': str(image_id),
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'run_count': int(len(runs)),
        'result_count': int(len(flat_rows)),
        'rows': flat_rows,
        'runs': [
            {
                'run_id': run.get('run_id'),
                'created_at': run.get('created_at'),
                'experiment_id': run.get('experiment_id'),
                'meta': run.get('meta'),
                'results': run.get('results'),
            }
            for run in runs
        ],
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding='utf-8')

    gallery_path = report_dir / 'gallery.png'
    _make_gallery(runs, gallery_path)

    return {
        'image_id': str(image_id),
        'report_dir': str(report_dir),
        'summary_csv': str(csv_path),
        'summary_json': str(json_path),
        'gallery_png': str(gallery_path),
        'run_count': int(len(runs)),
        'result_count': int(len(flat_rows)),
    }
