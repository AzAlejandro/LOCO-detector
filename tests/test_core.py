import numpy as np

from backend.catalog import build_registry
from backend.persistence import clear_scribble_draft, load_scribble_draft, save_scribble_draft
from backend.scribble import normalize_scribble_labels


def test_registry_has_expected_experiments() -> None:
    reg = build_registry()
    items = reg.list()
    ids = {x['experiment_id'] for x in items}
    expected = {
        'extratrees_pixel',
        'rf_pixel',
        'xgboost_pixel',
        'catboost_pixel',
        'extratrees_balanced',
        'context_features_variant',
        'classifier_morph_min',
        'unet_small_patch',
    }
    assert expected == ids
    assert len(items) == 8

    for item in items:
        assert item['implementation_status'] in {'native', 'fallback', 'proxy'}
        assert 'default_params' in item


def test_scribble_normalization_is_canonical() -> None:
    raw = np.zeros((40, 40), dtype=np.uint8)
    raw[5:14, 5:14] = 130
    raw[15:22, 15:24] = 192
    raw[25:34, 25:34] = 240
    labels = normalize_scribble_labels(raw)
    assert labels is not None
    uniq = set(np.unique(labels).tolist())
    assert uniq.issubset({0, 1, 2, 3})
    assert int(np.sum(labels == 1)) > 0
    assert int(np.sum(labels == 2)) > 0
    assert int(np.sum(labels == 3)) > 0


def test_draft_persistence_is_canonical(tmp_path, monkeypatch) -> None:
    import backend.persistence as p

    monkeypatch.setattr(p, 'OUTPUT_ROOT', tmp_path / 'outputs')
    monkeypatch.setattr(p, 'RUNS_DIR', p.OUTPUT_ROOT / 'runs')
    monkeypatch.setattr(p, 'REVIEWS_DIR', p.OUTPUT_ROOT / 'reviews')
    monkeypatch.setattr(p, 'INDEX_DIR', p.OUTPUT_ROOT / 'index')
    monkeypatch.setattr(p, 'REPORTS_DIR', p.OUTPUT_ROOT / 'reports')
    monkeypatch.setattr(p, 'DRAFTS_DIR', p.OUTPUT_ROOT / 'drafts')

    raw = np.zeros((32, 32), dtype=np.uint8)
    raw[5:12, 5:12] = 1
    raw[14:18, 18:27] = 2
    raw[20:27, 18:27] = 3
    raw[0:2, 0:2] = 255  # ambiguo intencional, debe limpiarse a 0
    meta = save_scribble_draft('img_test', raw)
    assert int(meta['n_fg']) > 0
    assert int(meta['n_halo']) > 0
    assert int(meta['n_bg']) > 0

    loaded = load_scribble_draft('img_test')
    assert loaded['found'] is True
    labels = np.asarray(loaded['labels'], dtype=np.uint8)
    uniq = set(np.unique(labels).tolist())
    assert uniq.issubset({0, 1, 2, 3})
    assert int(np.sum(labels == 1)) > 0
    assert int(np.sum(labels == 2)) > 0
    assert int(np.sum(labels == 3)) > 0

    info = clear_scribble_draft('img_test')
    assert info['cleared'] is True
    loaded2 = load_scribble_draft('img_test')
    assert loaded2['found'] is False
