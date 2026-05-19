import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.diameter_research.orientation import estimate_orientation
from backend.diameter_research.pipeline import run_hybrid_profile_diameter
from backend.diameter_research.pipeline_v2 import (
    METHOD_ID_V2,
    build_weighted_support,
    evaluate_orientation_sweep,
    isolate_local_component,
    recenter_point,
    run_hybrid_profile_diameter_v2,
)
from backend.diameter_research.support_region import build_support_region
from backend.diameter_research.v3 import (
    METHOD_ID_V3,
    METHOD_ID_V3_1,
    METHOD_ID_V3_2,
    METHOD_ID_V3_3,
    METHOD_ID_V3_3A,
    METHOD_ID_V3_3B,
    METHOD_ID_V3_3C,
    METHOD_ID_V3_3D,
    METHOD_ID_V3_2_CONTOUR_REFINE,
    METHOD_ID_V3_2_CURVELET_AIDED,
    METHOD_ID_V3_2_FLUX_AWARE,
    METHOD_ID_V3_2_HALO_AWARE,
    METHOD_ID_V3_2_RIDGE_ANCHORED,
    METHOD_ID_V3_2_SMALL_LARGE,
    METHOD_ID_LOCO,
    run_hybrid_profile_diameter_v3,
    run_hybrid_profile_diameter_v3_1,
    run_hybrid_profile_diameter_v3_2,
    run_hybrid_profile_diameter_v3_2_contour_refine,
    run_hybrid_profile_diameter_v3_2_curvelet_aided,
    run_hybrid_profile_diameter_v3_2_flux_aware,
    run_hybrid_profile_diameter_v3_2_halo_aware,
    run_hybrid_profile_diameter_v3_2_ridge_anchored,
    run_hybrid_profile_diameter_v3_2_small_large,
    run_hybrid_profile_diameter_v3_3,
    run_hybrid_profile_diameter_v3_3a,
    run_hybrid_profile_diameter_v3_3b,
    run_hybrid_profile_diameter_v3_3c,
    run_hybrid_profile_diameter_v3_3d,
    run_loco_circle_probe,
)
from backend.diameter_research.v3.edge_pairs import measure_diameter_from_edge_pairs
from backend.diameter_research.v3.fallback import measure_diameter_fallback
from backend.diameter_research.v3.geometry_guard import evaluate_local_geometry_ambiguity
from backend.diameter_research.v3.local_orientation import estimate_local_orientation_from_image
from backend.diameter_research.v3.multiscale import multiscale_decision
from backend.diameter_research.v3.recenter import recenter_point_on_local_axis
from backend.diameter_research.v3.support_roi import build_local_support_roi
from backend.image_codec import encode_gray_png_b64
from backend.main import app


def _fiber_image(size: int = 128, radius_x: int = 18, radius_y: int = 44) -> np.ndarray:
    y, x = np.indices((size, size))
    cy = cx = size // 2
    fiber = (((x - cx) / radius_x) ** 2 + ((y - cy) / radius_y) ** 2) <= 1.0
    bg = np.clip((x * 0.15 + y * 0.10) / size, 0.0, 1.0)
    img = np.clip(0.18 + 0.30 * bg + 0.58 * fiber.astype(np.float32), 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def _scribbles(size: int = 128) -> np.ndarray:
    labels = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    labels[c - 3:c + 4, c - 3:c + 4] = 128
    labels[c - 16:c - 10, c + 10:c + 22] = 192
    labels[8:20, 8:20] = 255
    labels[size - 22:size - 10, size - 22:size - 10] = 255
    return labels


def _labels_from_visual(vis: np.ndarray) -> np.ndarray:
    labels = np.zeros_like(vis, dtype=np.uint8)
    labels[vis == 128] = 1
    labels[vis == 192] = 2
    labels[vis == 255] = 3
    return labels


def _patch_diameter_dirs(tmp_path, monkeypatch) -> None:
    import backend.diameter_research.persistence as drp
    import backend.diameter_research.report as drr

    root = tmp_path / 'diameter_research'
    monkeypatch.setattr(drp, 'OUTPUT_ROOT', root)
    monkeypatch.setattr(drp, 'POINTS_DIR', root / 'points')
    monkeypatch.setattr(drp, 'RUNS_DIR', root / 'runs')
    monkeypatch.setattr(drp, 'INDEX_DIR', root / 'index')
    monkeypatch.setattr(drp, 'REPORTS_DIR', root / 'reports')
    monkeypatch.setattr(drr.drp, 'OUTPUT_ROOT', root)
    monkeypatch.setattr(drr.drp, 'POINTS_DIR', root / 'points')
    monkeypatch.setattr(drr.drp, 'RUNS_DIR', root / 'runs')
    monkeypatch.setattr(drr.drp, 'INDEX_DIR', root / 'index')
    monkeypatch.setattr(drr.drp, 'REPORTS_DIR', root / 'reports')


def test_support_from_prior_and_scribble_fallback() -> None:
    prior = np.zeros((64, 64), dtype=np.float32)
    cv2.circle(prior, (32, 32), 12, 0.9, thickness=-1)
    support, meta = build_support_region(prior_map=prior, labels=None, shape_hw=(64, 64), params={})
    assert support.shape == (64, 64)
    assert int(support.sum()) > 0
    assert meta['source'] == 'prior'

    labels = np.zeros((64, 64), dtype=np.uint8)
    labels[31:34, 31:34] = 1
    support_fb, meta_fb = build_support_region(prior_map=None, labels=labels, shape_hw=(64, 64), params={})
    assert int(support_fb.sum()) > int(labels.sum())
    assert meta_fb['fallback'] == 'scribbles_support'


def test_orientation_returns_normalized_vector() -> None:
    image = _fiber_image(128)
    prior = np.zeros((128, 128), dtype=np.float32)
    cv2.ellipse(prior, (64, 64), (17, 43), 0, 0, 360, 1.0, thickness=-1)
    orient = estimate_orientation(gray_u8=image, support=(prior > 0).astype(np.uint8), point_xy=(64, 64), params={})
    t = np.asarray(orient['tangent'], dtype=np.float32)
    n = np.asarray(orient['normal'], dtype=np.float32)
    assert orient['source'] == 'pca_support'
    assert np.isclose(np.linalg.norm(t), 1.0, atol=1e-4)
    assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-4)
    assert abs(float(np.dot(t, n))) < 1e-3


def test_pipeline_synthetic_diameter_positive() -> None:
    image = _fiber_image(128)
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    prior = np.zeros((128, 128), dtype=np.float32)
    cv2.ellipse(prior, (64, 64), (18, 44), 0, 0, 360, 0.95, thickness=-1)
    out = run_hybrid_profile_diameter(
        image_rgb=rgb,
        labels=np.zeros((128, 128), dtype=np.uint8),
        prior_map=prior,
        points=[{'x': 64, 'y': 64}],
        params={'profile_length_px': 80, 'profile_count': 7},
        source_mode='prior',
    )
    assert out['results']
    assert out['results'][0]['status'] == 'ok'
    assert float(out['results'][0]['diameter_px']) > 0


def test_v2_weighted_support_and_local_component() -> None:
    prior = np.zeros((96, 96), dtype=np.float32)
    cv2.circle(prior, (32, 48), 10, 0.9, thickness=-1)
    cv2.circle(prior, (68, 48), 10, 0.85, thickness=-1)
    hard, weight, meta = build_weighted_support(prior_map=prior, labels=None, shape_hw=(96, 96), params={})
    assert hard.shape == prior.shape
    assert weight.shape == prior.shape
    assert float(weight.max()) <= 1.0
    assert meta['support_weight_stats']['nonzero_px'] > 0

    labels = np.zeros((96, 96), dtype=np.uint8)
    labels[44:52, 28:36] = 2
    _hard_neg, weight_neg, meta_neg = build_weighted_support(prior_map=prior, labels=labels, shape_hw=(96, 96), params={})
    assert meta_neg['support_weight_stats']['halo_scribble_px'] > 0
    assert meta_neg['support_weight_stats']['negative_scribble_px'] > 0
    assert meta_neg['support_weight_stats']['negative_suppressed_px'] > 0
    assert float(weight_neg[48, 32]) < float(weight[48, 32])

    local, local_weight, comp_meta = isolate_local_component(
        hard,
        weight,
        (32, 48),
        {'support_component_radius_px': 44},
    )
    assert int(local.sum()) > 0
    assert float(local_weight.max()) > 0.0
    assert comp_meta['components_nearby'] >= 2


def test_v2_recenter_moves_displaced_point_to_ridge() -> None:
    prior = np.zeros((128, 128), dtype=np.float32)
    cv2.ellipse(prior, (64, 64), (18, 44), 0, 0, 360, 1.0, thickness=-1)
    hard, weight, _ = build_weighted_support(prior_map=prior, labels=None, shape_hw=(128, 128), params={})
    local, local_weight, _ = isolate_local_component(hard, weight, (58, 64), {'support_component_radius_px': 48})
    recentered, diag = recenter_point(
        support_weight=local_weight,
        local_support=local,
        point_xy=(58, 64),
        params={'recenter_radius_px': 8, 'recenter_step_px': 1},
    )
    assert diag['recenter_shift_px'] > 0
    assert abs(recentered[0] - 64) < abs(58 - 64)


def test_v2_orientation_sweep_prefers_best_angle() -> None:
    image = _fiber_image(128)
    prior = np.zeros((128, 128), dtype=np.float32)
    cv2.ellipse(prior, (64, 64), (18, 44), 0, 0, 360, 1.0, thickness=-1)
    hard, weight, _ = build_weighted_support(prior_map=prior, labels=None, shape_hw=(128, 128), params={})
    base = {'source': 'test', 'tangent': [0.17365, 0.9848], 'normal': [-0.9848, 0.17365], 'confidence': 0.9}
    best, candidates = evaluate_orientation_sweep(
        gray_f=image.astype(np.float32) / 255.0,
        support_weight=weight * hard,
        center_xy=(64, 64),
        base_orientation=base,
        params={'orientation_sweep_deg': [-10, -5, 0, 5, 10], 'profile_length_px': 80, 'profile_count': 7},
    )
    assert len(candidates) == 5
    assert best['score'] == max(float(c['score']) for c in candidates)
    assert best['valid_profiles'] >= 3


def test_v2_pipeline_handles_halo_without_external_edge() -> None:
    size = 128
    y, x = np.indices((size, size))
    c = size // 2
    fiber = (((x - c) / 18) ** 2 + ((y - c) / 44) ** 2) <= 1.0
    halo = ((((x - c) / 26) ** 2 + ((y - c) / 52) ** 2) <= 1.0) & ~fiber
    image = np.full((size, size), 60, dtype=np.float32)
    image[halo] = 110
    image[fiber] = 190
    image = cv2.GaussianBlur(image, (0, 0), 1.1).astype(np.uint8)
    out = run_hybrid_profile_diameter_v2(
        image_rgb=cv2.cvtColor(image, cv2.COLOR_GRAY2RGB),
        labels=np.zeros((size, size), dtype=np.uint8),
        prior_map=fiber.astype(np.float32),
        points=[{'x': 64, 'y': 64}],
        params={'profile_length_px': 90, 'profile_count': 7},
        source_mode='prior',
    )
    result = out['results'][0]
    assert result['status'] == 'ok'
    assert result['method_id'] == METHOD_ID_V2
    assert 28.0 <= float(result['diameter_px']) <= 40.0
    assert result['quality_label'] in {'high_confidence', 'medium_confidence'}


def test_v2_rejects_ambiguous_nearby_components() -> None:
    size = 128
    image = np.full((size, size), 60, dtype=np.uint8)
    prior = np.zeros((size, size), dtype=np.float32)
    cv2.ellipse(prior, (50, 64), (8, 36), 0, 0, 360, 1.0, thickness=-1)
    cv2.ellipse(prior, (78, 64), (8, 36), 0, 0, 360, 1.0, thickness=-1)
    image[prior > 0] = 190
    out = run_hybrid_profile_diameter_v2(
        image_rgb=cv2.cvtColor(image, cv2.COLOR_GRAY2RGB),
        labels=np.zeros((size, size), dtype=np.uint8),
        prior_map=prior,
        points=[{'x': 64, 'y': 64}],
        params={'support_component_radius_px': 40, 'profile_length_px': 70, 'profile_count': 7},
        source_mode='prior',
    )
    result = out['results'][0]
    assert result['status'] == 'rejected'
    assert 'multiple_components_nearby' in result['quality_flags']


def test_v3_support_roi_refines_lateral_expansion() -> None:
    size = 128
    y, x = np.indices((size, size))
    c = size // 2
    fiber = (((x - c) / 5) ** 2 + ((y - c) / 42) ** 2) <= 1.0
    inflated = (((x - c) / 14) ** 2 + ((y - c) / 46) ** 2) <= 1.0
    gray = np.full((size, size), 0.18, dtype=np.float32)
    gray[fiber] = 0.82
    roi = build_local_support_roi(
        gray_f=gray,
        support=inflated.astype(np.uint8),
        support_weight=inflated.astype(np.float32),
        point_xy=(64, 64),
        params={'local_roi_radius_px': 52, 'support_refine_strength': 0.45, 'thin_fiber_threshold_px': 12},
    )
    assert int(np.sum(roi['support_raw_local'])) > int(np.sum(roi['support_refined_local']))
    assert roi['meta']['support_status'] in {'refined', 'thin_fiber_refined'}


def test_v3_orientation_recenter_edgepair_and_fallback() -> None:
    image = _fiber_image(128)
    gray = image.astype(np.float32) / 255.0
    support = np.zeros((128, 128), dtype=np.uint8)
    cv2.ellipse(support, (64, 64), (18, 44), 0, 0, 360, 1, thickness=-1)
    orient = estimate_local_orientation_from_image(gray_f=gray, support=support, point_xy=(64, 64), params={})
    assert orient['source'] == 'structure_tensor_image'
    assert np.isclose(np.linalg.norm(np.asarray(orient['tangent'])), 1.0, atol=1e-4)
    assert np.isclose(np.linalg.norm(np.asarray(orient['normal'])), 1.0, atol=1e-4)
    assert 'orientation_coherence' in orient

    recentered, recenter_diag = recenter_point_on_local_axis(
        gray_f=gray,
        support_weight=support.astype(np.float32),
        support_refined=support,
        point_xy=(58, 64),
        orientation=orient,
        params={'recenter_radius_px': 8, 'max_recenter_shift_px': 8},
    )
    assert recenter_diag['recenter_shift_px'] > 0
    assert abs(recentered[0] - 64) < abs(58 - 64)

    edge = measure_diameter_from_edge_pairs(
        gray_f=gray,
        support_weight=support.astype(np.float32),
        center_xy=(64, 64),
        orientation=orient,
        params={'profile_length_px': 80, 'profile_count': 7, 'edge_pair_min_score': 0.1},
    )
    assert edge['status'] == 'ok'
    assert float(edge['diameter_px']) > 0

    fb = measure_diameter_fallback(
        support_weight=support.astype(np.float32),
        center_xy=(64, 64),
        orientation=orient,
        geometry={'geometry_status': 'geometry_simple'},
        recenter={'recenter_status': 'ok'},
        params={'fallback_enabled': True, 'profile_length_px': 80},
    )
    assert fb['status'] == 'ok'
    assert float(fb['diameter_px']) > 0


def test_v3_geometry_guard_and_multiscale_flags() -> None:
    size = 128
    gray = np.full((size, size), 0.2, dtype=np.float32)
    support = np.zeros((size, size), dtype=np.uint8)
    cv2.line(support, (64, 20), (64, 108), 1, thickness=9)
    cv2.line(support, (20, 64), (108, 64), 1, thickness=9)
    gray[support > 0] = 0.8
    geom = evaluate_local_geometry_ambiguity(
        gray_f=gray,
        support_refined=support,
        center_xy=(64, 64),
        orientation={'orientation_coherence': 0.02},
        profiles=[],
        params={'min_orientation_coherence': 0.18, 'geometry_window_px': 48},
    )
    assert geom['geometry_status'] in {'geometry_ambiguous', 'crossing_likely'}

    ms = multiscale_decision(
        diameter_px=5.0,
        edge_status='ok',
        support_meta={'thin_fiber_support_mode': True},
        params={'multiscale_enabled': True, 'thin_fiber_threshold_px': 8, 'upscale_factors': [2, 3]},
    )
    assert ms['used_upscale'] is True
    assert ms['scale_factor'] == 2


def test_v3_pipeline_subversions_clean_thin_and_crossing_cases() -> None:
    image = _fiber_image(128)
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    prior = np.zeros((128, 128), dtype=np.float32)
    cv2.ellipse(prior, (64, 64), (18, 44), 0, 0, 360, 1.0, thickness=-1)
    out31 = run_hybrid_profile_diameter_v3_1(
        image_rgb=rgb,
        labels=np.zeros((128, 128), dtype=np.uint8),
        prior_map=prior,
        points=[{'x': 64, 'y': 64}],
        params={
            'profile_length_px': 80,
            'profile_count': 7,
            'edge_pair_min_score': 0.1,
            'min_point_confidence': 0.1,
            'support_refine_enabled': True,
            'fallback_enabled': True,
            'multiscale_enabled': True,
        },
        source_mode='prior',
    )
    result31 = out31['results'][0]
    assert result31['method_id'] == METHOD_ID_V3_1
    assert result31['variant_stage'] == 'v3_1'
    assert result31['status'] == 'ok'
    assert result31['measurement_mode'] == 'edge_pair'
    assert result31['support_status'] == 'refine_disabled'
    assert result31['used_upscale'] is False
    assert result31['small_diameter_suspect'] is False
    assert float(result31['diameter_px']) > 0

    thin = np.full((128, 128), 50, dtype=np.uint8)
    thin_prior = np.zeros((128, 128), dtype=np.float32)
    cv2.ellipse(thin_prior, (64, 64), (3, 38), 0, 0, 360, 1.0, thickness=-1)
    thin[thin_prior > 0] = 210
    thin_out = run_hybrid_profile_diameter_v3_2(
        image_rgb=cv2.cvtColor(thin, cv2.COLOR_GRAY2RGB),
        labels=np.zeros((128, 128), dtype=np.uint8),
        prior_map=thin_prior,
        points=[{'x': 64, 'y': 64}],
        params={
            'profile_length_px': 44,
            'profile_count': 7,
            'min_point_confidence': 0.1,
            'thin_fiber_threshold_px': 8,
            'multiscale_enabled': True,
            'fallback_enabled': True,
            'local_geometry_control_enabled': False,
            'adaptive_profile_length_enabled': False,
        },
        source_mode='prior',
    )
    thin_result = thin_out['results'][0]
    assert thin_result['method_id'] == METHOD_ID_V3_2
    assert thin_result['variant_stage'] == 'v3_2'
    assert thin_result['support_status'] == 'refine_disabled'
    assert thin_result['geometry_control_status'] in {'ok', 'overshoot_risk'}
    assert float(thin_result['profile_length_effective_px']) <= 44.0
    assert thin_result['used_upscale'] is False
    assert thin_result['scale_factor'] == 1
    if thin_result['diameter_px'] is not None:
        assert thin_result['small_diameter_suspect'] is True

    params_common = {'profile_length_px': 80, 'profile_count': 7, 'edge_pair_min_score': 0.1, 'min_point_confidence': 0.1}
    out33a = run_hybrid_profile_diameter_v3_3a(
        image_rgb=rgb,
        labels=np.zeros((128, 128), dtype=np.uint8),
        prior_map=prior,
        points=[{'x': 64, 'y': 64}],
        params={**params_common, 'fallback_enabled': True, 'geometry_guard_enabled': False},
        source_mode='prior',
    )
    assert out33a['results'][0]['method_id'] == METHOD_ID_V3_3A
    assert out33a['meta']['params_effective']['geometry_guard_enabled'] is True
    assert out33a['meta']['params_effective']['fallback_enabled'] is False
    assert out33a['meta']['params_effective']['multiscale_enabled'] is False

    out33b = run_hybrid_profile_diameter_v3_3b(
        image_rgb=rgb,
        labels=np.zeros((128, 128), dtype=np.uint8),
        prior_map=prior,
        points=[{'x': 64, 'y': 64}],
        params={**params_common, 'geometry_guard_enabled': True, 'multiscale_enabled': True},
        source_mode='prior',
    )
    assert out33b['results'][0]['method_id'] == METHOD_ID_V3_3B
    assert out33b['meta']['params_effective']['fallback_enabled'] is True
    assert out33b['meta']['params_effective']['geometry_guard_enabled'] is False
    assert out33b['meta']['params_effective']['multiscale_enabled'] is False

    out33c = run_hybrid_profile_diameter_v3_3c(
        image_rgb=rgb,
        labels=np.zeros((128, 128), dtype=np.uint8),
        prior_map=prior,
        points=[{'x': 64, 'y': 64}],
        params={**params_common, 'fallback_enabled': True, 'geometry_guard_enabled': True},
        source_mode='prior',
    )
    assert out33c['results'][0]['method_id'] == METHOD_ID_V3_3C
    assert out33c['meta']['params_effective']['multiscale_enabled'] is True
    assert out33c['meta']['params_effective']['fallback_enabled'] is False
    assert out33c['meta']['params_effective']['geometry_guard_enabled'] is False

    out33d = run_hybrid_profile_diameter_v3_3d(
        image_rgb=rgb,
        labels=np.zeros((128, 128), dtype=np.uint8),
        prior_map=prior,
        points=[{'x': 64, 'y': 64}],
        params={**params_common, 'fallback_enabled': True, 'geometry_guard_enabled': True, 'multiscale_enabled': True},
        source_mode='prior',
    )
    assert out33d['results'][0]['method_id'] == METHOD_ID_V3_3D
    assert out33d['meta']['params_effective']['antihalo_enabled'] is True
    assert out33d['meta']['params_effective']['fallback_enabled'] is False
    assert out33d['meta']['params_effective']['multiscale_enabled'] is False

    cross = np.full((128, 128), 50, dtype=np.uint8)
    cross_prior = np.zeros((128, 128), dtype=np.float32)
    cv2.line(cross_prior, (64, 20), (64, 108), 1.0, thickness=9)
    cv2.line(cross_prior, (20, 64), (108, 64), 1.0, thickness=9)
    cross[cross_prior > 0] = 210
    cross_out = run_hybrid_profile_diameter_v3_3(
        image_rgb=cv2.cvtColor(cross, cv2.COLOR_GRAY2RGB),
        labels=np.zeros((128, 128), dtype=np.uint8),
        prior_map=cross_prior,
        points=[{'x': 64, 'y': 64}],
        params={'profile_length_px': 70, 'profile_count': 7, 'min_point_confidence': 0.1},
        source_mode='prior',
    )
    cross_result = cross_out['results'][0]
    assert cross_result['method_id'] == METHOD_ID_V3_3
    assert cross_result['variant_stage'] == 'v3_3'
    assert cross_result['status'] == 'rejected'
    assert cross_result['geometry_status'] in {'geometry_ambiguous', 'crossing_likely'}

    alias_out = run_hybrid_profile_diameter_v3(
        image_rgb=rgb,
        labels=np.zeros((128, 128), dtype=np.uint8),
        prior_map=prior,
        points=[{'x': 64, 'y': 64}],
        params={'profile_length_px': 80, 'profile_count': 7, 'edge_pair_min_score': 0.1, 'min_point_confidence': 0.1},
        source_mode='prior',
    )
    assert alias_out['results'][0]['method_id'] == METHOD_ID_V3
    assert alias_out['results'][0]['variant_stage'] == 'v3_3'


def test_v3_step5_methodology_variants_are_traced() -> None:
    image = _fiber_image(128)
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    prior = np.zeros((128, 128), dtype=np.float32)
    cv2.ellipse(prior, (64, 64), (18, 44), 0, 0, 360, 1.0, thickness=-1)
    variants = [
        (METHOD_ID_V3_2_HALO_AWARE, 'halo_aware', run_hybrid_profile_diameter_v3_2_halo_aware),
        (METHOD_ID_V3_2_SMALL_LARGE, 'small_large', run_hybrid_profile_diameter_v3_2_small_large),
        (METHOD_ID_V3_2_RIDGE_ANCHORED, 'ridge_anchored', run_hybrid_profile_diameter_v3_2_ridge_anchored),
        (METHOD_ID_V3_2_FLUX_AWARE, 'flux_aware', run_hybrid_profile_diameter_v3_2_flux_aware),
        (METHOD_ID_V3_2_CONTOUR_REFINE, 'contour_refine', run_hybrid_profile_diameter_v3_2_contour_refine),
        (METHOD_ID_V3_2_CURVELET_AIDED, 'curvelet_aided', run_hybrid_profile_diameter_v3_2_curvelet_aided),
    ]
    for method_id, methodology_id, runner in variants:
        out = runner(
            image_rgb=rgb,
            labels=np.zeros((128, 128), dtype=np.uint8),
            prior_map=prior,
            points=[{'x': 64, 'y': 64}],
            params={'profile_length_px': 70, 'profile_count': 7, 'edge_pair_min_score': 0.1, 'min_point_confidence': 0.1},
            source_mode='prior',
        )
        result = out['results'][0]
        assert result['method_id'] == method_id
        assert result['methodology_id'] == methodology_id
        assert result['local_context_label']
        assert result['selected_edge_policy']
        assert methodology_id in out['meta']['params_effective']['methodology_id']
        assert 'methodology' in out['diagnostics']['diagnostics_v3']['points'][0]


def test_loco_circle_probe_tracks_radius_candidates() -> None:
    h, w = 80, 120
    image = np.full((h, w), 220, dtype=np.uint8)
    prior = np.zeros((h, w), dtype=np.float32)
    prior[35:46, 15:106] = 1.0
    image[prior > 0] = 80
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    out = run_loco_circle_probe(
        image_rgb=rgb,
        labels=np.zeros((h, w), dtype=np.uint8),
        prior_map=prior,
        points=[{'x': 60, 'y': 40}],
        params={'loco_max_radius_px': 14, 'loco_reject_threshold': 0.35, 'loco_mode': 'refine'},
        source_mode='prior_mask',
    )
    result = out['results'][0]
    assert result['method_id'] == METHOD_ID_LOCO
    assert result['measurement_mode'] in {'loco_refine', 'loco_direct'}
    assert result['diameter_px'] is not None
    assert float(result['loco_best_radius_px']) > 0
    assert float(result['loco_symmetry_score']) > 0.35
    assert int(result['loco_intersection_count']) >= 2
    assert out['diagnostics']['diagnostics_loco']['points'][0]['best']['radius_px'] == result['loco_best_radius_px']


def test_loco_seed_radius_limits_search_window() -> None:
    h, w = 80, 120
    image = np.full((h, w, 3), 220, dtype=np.uint8)
    prior = np.zeros((h, w), dtype=np.float32)
    prior[34:47, 20:101] = 1.0
    image[prior > 0] = 80

    out = run_loco_circle_probe(
        image_rgb=image,
        labels=None,
        prior_map=prior,
        points=[{'x': 60, 'y': 40}],
        params={
            'loco_seed_radii_by_point': {'0': 6.0},
            'loco_seed_radius_window_px': 2.0,
            'loco_radius_step_px': 1.0,
            'loco_max_radius_px': 18.0,
            'loco_reject_threshold': 0.2,
            'loco_mode': 'direct',
        },
        source_mode='prior_mask',
    )
    radii = [float(item['radius_px']) for item in out['results'][0]['radius_candidates']]
    assert radii
    assert min(radii) >= 4.0
    assert max(radii) <= 8.0


def test_persistence_save_list_load_run(tmp_path, monkeypatch) -> None:
    import backend.diameter_research.persistence as drp

    _patch_diameter_dirs(tmp_path, monkeypatch)
    image = cv2.cvtColor(_fiber_image(96), cv2.COLOR_GRAY2RGB)
    labels = np.zeros((96, 96), dtype=np.uint8)
    prior = np.zeros((96, 96), dtype=np.float32)
    cv2.circle(prior, (48, 48), 16, 0.9, thickness=-1)
    run_id = drp.new_run_id()
    art = drp.DiameterRunArtifacts(
        run_id=run_id,
        image_id='img_test',
        experiment_id='hybrid_profile_diameter_v1',
        created_at='2026-05-04 12:00:00',
        input_image=image,
        scribble_labels=labels,
        prior_prob=prior,
        support_region=(prior > 0).astype(np.uint8),
        overlay=image,
        results=[{'point_index': 0, 'status': 'ok', 'diameter_px': 12.0}],
        diagnostics={},
        meta={'source_mode': 'prior'},
    )
    drp.save_diameter_run(art)
    rows = drp.list_diameter_runs('img_test')
    assert rows and rows[0]['run_id'] == run_id
    loaded = drp.load_diameter_run(run_id)
    assert loaded['run_id'] == run_id
    assert loaded['overlay'].shape[:2] == (96, 96)


def test_diameter_api_flow(tmp_path, monkeypatch) -> None:
    import backend.persistence as p
    import backend.library_store as ls

    _patch_diameter_dirs(tmp_path, monkeypatch)
    monkeypatch.setattr(p, 'OUTPUT_ROOT', tmp_path / 'outputs')
    monkeypatch.setattr(p, 'RUNS_DIR', p.OUTPUT_ROOT / 'runs')
    monkeypatch.setattr(p, 'REVIEWS_DIR', p.OUTPUT_ROOT / 'reviews')
    monkeypatch.setattr(p, 'INDEX_DIR', p.OUTPUT_ROOT / 'index')
    monkeypatch.setattr(p, 'REPORTS_DIR', p.OUTPUT_ROOT / 'reports')
    monkeypatch.setattr(p, 'DRAFTS_DIR', p.OUTPUT_ROOT / 'drafts')
    monkeypatch.setattr(ls, 'LIBRARY_DIR', p.OUTPUT_ROOT / 'library')

    c = TestClient(app)
    sid = c.post('/api/session/new', json={}).json()['payload']['session_id']
    image = _fiber_image(128)
    ok, buf = cv2.imencode('.png', image)
    assert ok
    r1 = c.post('/api/image/load', data={'session_id': sid}, files={'file': ('fiber.png', bytes(buf), 'image/png')})
    assert r1.status_code == 200
    image_id = r1.json()['payload']['image_id']

    scrib_vis = _scribbles(128)
    rd = c.post('/api/scribble/draft/save', json={
        'session_id': sid,
        'image_id': image_id,
        'scribble_map_b64': encode_gray_png_b64(scrib_vis),
    })
    assert rd.status_code == 200

    lib = c.get(f'/api/library/images?session_id={sid}')
    assert lib.status_code == 200
    assert any(x['image_id'] == image_id and x['has_scribble_draft'] for x in lib.json()['payload']['items'])

    loaded = c.post('/api/library/load', json={'session_id': sid, 'image_id': image_id, 'restore_scribbles': True})
    assert loaded.status_code == 200
    assert loaded.json()['payload']['image_id'] == image_id
    assert loaded.json()['payload']['scribble_draft']['found'] is True

    exp = c.post('/api/experiments/run', json={
        'session_id': sid,
        'experiment_id': 'extratrees_pixel',
        'scribble_map_b64': encode_gray_png_b64(scrib_vis),
    })
    assert exp.status_code == 200
    seg_run_id = exp.json()['payload']['run_id']
    review = c.post('/api/review/mark', json={
        'run_id': seg_run_id,
        'image_id': image_id,
        'decision': 's',
        'note': 'top tier',
    })
    assert review.status_code == 200
    reviews = c.get(f'/api/review/list?image_id={image_id}')
    assert reviews.status_code == 200
    assert reviews.json()['payload']['items'][0]['decision'] == 's'

    pt = c.post('/api/diameter-research/points/update', json={
        'session_id': sid,
        'action': 'add',
        'x': 64,
        'y': 64,
    })
    assert pt.status_code == 200
    assert len(pt.json()['points']) == 1

    run = c.post('/api/diameter-research/run', json={
        'session_id': sid,
        'image_id': image_id,
        'source_mode': 'scribbles',
        'points': [{'x': 64, 'y': 64}],
        'active_only': False,
        'params': {'profile_length_px': 80, 'profile_count': 7},
        'scribble_map_b64': encode_gray_png_b64(scrib_vis),
    })
    assert run.status_code == 200
    payload = run.json()
    assert payload['ok'] is True
    assert payload['run_id']
    assert payload['overlay_b64']
    assert payload['support_region_b64']
    assert payload['results']
    assert payload['method_id'] == 'hybrid_profile_diameter_v1'
    assert payload['results'][0]['diameter_px'] is not None

    run_prior = c.post('/api/diameter-research/run', json={
        'session_id': sid,
        'image_id': image_id,
        'source_mode': 'prior',
        'prior_run_id': seg_run_id,
        'points': [{'x': 64, 'y': 64}],
        'active_only': False,
        'params': {'profile_length_px': 80, 'profile_count': 7},
        'scribble_map_b64': encode_gray_png_b64(scrib_vis),
    })
    assert run_prior.status_code == 200
    assert run_prior.json()['meta']['prior_run_id'] == seg_run_id

    run_v2 = c.post('/api/diameter-research/run', json={
        'session_id': sid,
        'image_id': image_id,
        'method_id': 'hybrid_profile_diameter_v2',
        'source_mode': 'scribbles',
        'points': [{'x': 64, 'y': 64}],
        'active_only': False,
        'params': {'profile_length_px': 80, 'profile_count': 7, 'min_point_confidence': 0.1},
        'scribble_map_b64': encode_gray_png_b64(scrib_vis),
    })
    assert run_v2.status_code == 200
    payload_v2 = run_v2.json()
    assert payload_v2['method_id'] == 'hybrid_profile_diameter_v2'
    assert payload_v2['results'][0]['method_id'] == 'hybrid_profile_diameter_v2'
    assert 'quality_label' in payload_v2['results'][0]
    assert 'recenter_shift_px' in payload_v2['results'][0]

    run_v3 = c.post('/api/diameter-research/run', json={
        'session_id': sid,
        'image_id': image_id,
        'method_id': 'hybrid_profile_diameter_v3_3',
        'source_mode': 'scribbles',
        'points': [{'x': 64, 'y': 64}],
        'active_only': False,
        'params': {'profile_length_px': 80, 'profile_count': 7, 'min_point_confidence': 0.1, 'edge_pair_min_score': 0.1},
        'scribble_map_b64': encode_gray_png_b64(scrib_vis),
    })
    assert run_v3.status_code == 200
    payload_v3 = run_v3.json()
    assert payload_v3['method_id'] == 'hybrid_profile_diameter_v3_3'
    assert payload_v3['results'][0]['method_id'] == 'hybrid_profile_diameter_v3_3'
    assert 'measurement_mode' in payload_v3['results'][0]
    assert 'geometry_status' in payload_v3['results'][0]

    before_preview = c.get(f'/api/diameter-research/results/list?image_id={image_id}')
    assert before_preview.status_code == 200
    before_count = len(before_preview.json()['items'])
    loco_preview = c.post('/api/diameter-research/loco/preview', json={
        'session_id': sid,
        'image_id': image_id,
        'source_mode': 'scribbles',
        'point': {'x': 64, 'y': 64},
        'params': {'loco_max_radius_px': 10, 'loco_reject_threshold': 0.1},
        'scribble_map_b64': encode_gray_png_b64(scrib_vis),
        'step': 4,
    })
    assert loco_preview.status_code == 200
    assert loco_preview.json()['method_id'] == METHOD_ID_LOCO
    assert loco_preview.json()['radius_candidates']
    after_preview = c.get(f'/api/diameter-research/results/list?image_id={image_id}')
    assert len(after_preview.json()['items']) == before_count

    run_loco = c.post('/api/diameter-research/run', json={
        'session_id': sid,
        'image_id': image_id,
        'method_id': METHOD_ID_LOCO,
        'source_mode': 'scribbles',
        'points': [{'x': 64, 'y': 64}],
        'active_only': False,
        'params': {'loco_max_radius_px': 10, 'loco_reject_threshold': 0.1},
        'scribble_map_b64': encode_gray_png_b64(scrib_vis),
    })
    assert run_loco.status_code == 200
    payload_loco = run_loco.json()
    assert payload_loco['method_id'] == METHOD_ID_LOCO
    assert payload_loco['results'][0]['method_id'] == METHOD_ID_LOCO
    assert 'loco_best_radius_px' in payload_loco['results'][0]
    assert 'diagnostics_loco' in payload_loco['diagnostics']

    case = c.post('/api/diameter-research/validation/case/upsert', json={
        'session_id': sid,
        'image_id': image_id,
        'case_id': 'case_api_001',
        'point': {'x': 64, 'y': 64},
        'category': 'borde_limpio',
        'quality_manual': 'high',
        'manual_diameter_px': 34.0,
        'manual_left_x': 47.0,
        'manual_left_y': 64.0,
        'manual_right_x': 81.0,
        'manual_right_y': 64.0,
        'measurement_decision': 'validated',
        'notes': 'caso sintetico',
        'result_comment': 'pendiente revision',
        'source_mode': 'scribbles',
        'prior_run_id': seg_run_id,
        'params': {'profile_length_px': 80, 'profile_count': 7, 'min_point_confidence': 0.1},
    })
    assert case.status_code == 200
    assert case.json()['case']['case_id'] == 'case_api_001'
    assert case.json()['case']['measurement_decision'] == 'validated'
    assert case.json()['case']['prior_run_id'] == seg_run_id

    vr = c.post('/api/diameter-research/validation/run-case', json={
        'session_id': sid,
        'image_id': image_id,
        'case_id': 'case_api_001',
        'source_mode': 'scribbles',
        'params': {'profile_length_px': 80, 'profile_count': 7, 'min_point_confidence': 0.1},
        'scribble_map_b64': encode_gray_png_b64(scrib_vis),
        'methods': [
            'hybrid_profile_diameter_v1',
            'hybrid_profile_diameter_v2',
            'hybrid_profile_diameter_v3_1',
            'hybrid_profile_diameter_v3_2',
            'hybrid_profile_diameter_v3_3a',
            'hybrid_profile_diameter_v3_3b',
            'hybrid_profile_diameter_v3_3c',
            'hybrid_profile_diameter_v3_3d',
            METHOD_ID_LOCO,
        ],
    })
    assert vr.status_code == 200
    vcase = vr.json()['case']
    assert 'hybrid_profile_diameter_v1' in vcase['runs']
    assert 'hybrid_profile_diameter_v2' in vcase['runs']
    assert 'hybrid_profile_diameter_v3_1' in vcase['runs']
    assert 'hybrid_profile_diameter_v3_2' in vcase['runs']
    assert 'hybrid_profile_diameter_v3_3a' in vcase['runs']
    assert 'hybrid_profile_diameter_v3_3b' in vcase['runs']
    assert 'hybrid_profile_diameter_v3_3c' in vcase['runs']
    assert 'hybrid_profile_diameter_v3_3d' in vcase['runs']
    assert METHOD_ID_LOCO in vcase['runs']
    assert vcase['runs']['hybrid_profile_diameter_v1']['absolute_error_px'] is not None

    vcases = c.get(f'/api/diameter-research/validation/cases?image_id={image_id}')
    assert vcases.status_code == 200
    assert any(x['case_id'] == 'case_api_001' for x in vcases.json()['items'])

    vex = c.get(f'/api/diameter-research/validation/export?image_id={image_id}')
    assert vex.status_code == 200
    assert vex.json()['autofill_md']
    assert vex.json()['cases_csv']

    listed = c.get(f'/api/diameter-research/results/list?image_id={image_id}')
    assert listed.status_code == 200
    assert listed.json()['items'][0]['method_id'] in {
        'hybrid_profile_diameter_v1',
        'hybrid_profile_diameter_v2',
        'hybrid_profile_diameter_v3',
        'hybrid_profile_diameter_v3_1',
        'hybrid_profile_diameter_v3_2',
        'hybrid_profile_diameter_v3_3',
        'hybrid_profile_diameter_v3_3a',
        'hybrid_profile_diameter_v3_3b',
        'hybrid_profile_diameter_v3_3c',
        'hybrid_profile_diameter_v3_3d',
        METHOD_ID_LOCO,
    }

    got = c.get(f"/api/diameter-research/results/get?run_id={payload['run_id']}")
    assert got.status_code == 200
    assert got.json()['results']

    report = c.get(f'/api/diameter-research/reports/export?image_id={image_id}')
    assert report.status_code == 200
    assert report.json()['summary_csv']
