import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.image_codec import encode_gray_png_b64
from backend.main import app


def _img(size: int = 96) -> np.ndarray:
    y, x = np.indices((size, size))
    base = np.clip((x + y) / (2 * size), 0, 1)
    cx = size // 2
    cy = size // 2
    fiber = ((((x - cx) / max(3, size * 0.09)) ** 2 + ((y - cy) / max(6, size * 0.34)) ** 2) < 1.0).astype(np.float32)
    img = np.clip(0.35 * base + 0.65 * fiber, 0, 1)
    return (img * 255.0).astype(np.uint8)


def _img_b(size: int = 96) -> np.ndarray:
    y, x = np.indices((size, size))
    base = np.clip((x * 0.7 + y * 0.2) / (size), 0, 1)
    rr = (x - int(size * 0.35)) ** 2 + (y - int(size * 0.6)) ** 2
    blob = (rr < (size // 6) ** 2).astype(np.float32)
    img = np.clip(0.65 * base + 0.35 * blob, 0, 1)
    return (img * 255.0).astype(np.uint8)


def _scrib(size: int = 96) -> np.ndarray:
    s = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    s[c - 4:c + 4, c - 4:c + 4] = 128
    s[c - 13:c - 8, c + 8:c + 18] = 192
    s[6:16, 6:16] = 255
    s[size - 16:size - 6, size - 16:size - 6] = 255
    return s


def test_api_flow_run_review_export(tmp_path, monkeypatch) -> None:
    import backend.persistence as p
    import backend.library_store as ls
    import backend.assist_models as am
    import backend.main as m

    monkeypatch.setattr(p, 'OUTPUT_ROOT', tmp_path / 'outputs')
    monkeypatch.setattr(p, 'RUNS_DIR', p.OUTPUT_ROOT / 'runs')
    monkeypatch.setattr(p, 'REVIEWS_DIR', p.OUTPUT_ROOT / 'reviews')
    monkeypatch.setattr(p, 'INDEX_DIR', p.OUTPUT_ROOT / 'index')
    monkeypatch.setattr(p, 'REPORTS_DIR', p.OUTPUT_ROOT / 'reports')
    monkeypatch.setattr(p, 'DRAFTS_DIR', p.OUTPUT_ROOT / 'drafts')
    monkeypatch.setattr(ls, 'LIBRARY_DIR', p.OUTPUT_ROOT / 'library')
    monkeypatch.setattr(am, 'MODELS_ROOT', p.OUTPUT_ROOT / 'assist_models')
    monkeypatch.setattr(am, 'MODELS_DIR', am.MODELS_ROOT / 'models')
    monkeypatch.setattr(am, 'REGISTRY_PATH', am.MODELS_ROOT / 'registry.json')
    monkeypatch.setattr(m, 'UI_PREFS_PATH', p.OUTPUT_ROOT / 'ui_prefs.json')

    c = TestClient(app)
    r0 = c.post('/api/session/new', json={})
    assert r0.status_code == 200
    sid = r0.json()['payload']['session_id']

    cat = c.get('/api/experiments/catalog')
    assert cat.status_code == 200
    assert len(cat.json()['payload']['experiments']) == 9

    img = _img(96)
    ok, buf = cv2.imencode('.png', img)
    assert ok
    r1 = c.post('/api/image/load', data={'session_id': sid}, files={'file': ('a.png', bytes(buf), 'image/png')})
    assert r1.status_code == 200
    image_id = r1.json()['payload']['image_id']

    scrib = _scrib(96)
    rd = c.post('/api/scribble/draft/save', json={
        'session_id': sid,
        'image_id': image_id,
        'scribble_map_b64': encode_gray_png_b64(scrib),
    })
    assert rd.status_code == 200
    assert rd.json()['payload']['n_fg'] > 0
    assert rd.json()['payload']['n_halo'] > 0
    assert rd.json()['payload']['n_bg'] > 0

    rd2 = c.get(f'/api/scribble/draft/load?session_id={sid}&image_id={image_id}')
    assert rd2.status_code == 200
    assert rd2.json()['payload']['found'] is True
    assert rd2.json()['payload']['scribble_map_b64']

    library_no_session = c.get('/api/library/images')
    assert library_no_session.status_code == 200
    assert any(x['image_id'] == image_id for x in library_no_session.json()['payload']['items'])

    local_dir = tmp_path / 'local_images'
    local_dir.mkdir()
    local_img_path = local_dir / 'local.png'
    ok_local, local_buf = cv2.imencode('.png', _img_b(64))
    assert ok_local
    local_img_path.write_bytes(bytes(local_buf))
    prefs = c.post('/api/local-images/prefs', json={'start_dir': str(local_dir)})
    assert prefs.status_code == 200
    local_list = c.get(f'/api/local-images/list?start_dir={str(local_dir)}&recursive=true')
    assert local_list.status_code == 200
    assert any(x['path'] == str(local_img_path) for x in local_list.json()['payload']['items'])
    local_load = c.post('/api/local-images/load', json={'session_id': sid, 'path': str(local_img_path)})
    assert local_load.status_code == 200
    assert local_load.json()['payload']['image_id']
    reload_original = c.post('/api/library/load', json={'session_id': sid, 'image_id': image_id, 'restore_scribbles': True})
    assert reload_original.status_code == 200

    dataset = c.get(f'/api/assist-models/dataset/images?session_id={sid}')
    assert dataset.status_code == 200
    dataset_items = dataset.json()['items']
    assert any(x['image_id'] == image_id and x['class_counts']['halo'] > 0 for x in dataset_items)

    trained = c.post('/api/assist-models/train', json={
        'session_id': sid,
        'model_name': 'pytest_assist',
        'image_ids': [image_id],
        'class_mode': 'multiclass',
        'classifier': 'extratrees',
        'feature_variant': 'base',
        'n_estimators': 20,
        'max_samples_per_class': 400,
        'max_samples_per_image_class': 250,
    })
    assert trained.status_code == 200
    model_id = trained.json()['model']['model_id']
    assert model_id

    models = c.get('/api/assist-models/list')
    assert models.status_code == 200
    assert any(x['model_id'] == model_id for x in models.json()['models'])

    pred = c.post('/api/assist-models/predict', json={
        'session_id': sid,
        'image_id': image_id,
        'model_id': model_id,
        'min_confidence': 0.4,
        'include_fiber': True,
        'include_halo': True,
        'include_background': False,
    })
    assert pred.status_code == 200
    pred_payload = pred.json()
    assert pred_payload['suggestion_b64']
    assert pred_payload['preview_b64']
    assert 'fiber_prob' in pred_payload['score_maps_b64']

    r2 = c.post('/api/experiments/run', json={
        'session_id': sid,
        'experiment_id': 'extratrees_pixel',
        'params': {},
        'scribble_map_b64': encode_gray_png_b64(scrib),
    })
    assert r2.status_code == 200
    run_id = r2.json()['payload']['run_id']
    assert r2.json()['status_level'] in {'success', 'warning'}

    r3 = c.get(f'/api/results/list?image_id={image_id}')
    assert r3.status_code == 200
    assert any(x['run_id'] == run_id for x in r3.json()['payload']['items'])

    r2_overwrite = c.post('/api/experiments/run', json={
        'session_id': sid,
        'experiment_id': 'extratrees_pixel',
        'params': {},
        'scribble_map_b64': encode_gray_png_b64(scrib),
    })
    assert r2_overwrite.status_code == 200
    run_id_overwrite = r2_overwrite.json()['payload']['run_id']
    assert run_id_overwrite != run_id
    r3b = c.get(f'/api/results/list?image_id={image_id}')
    assert r3b.status_code == 200
    rows_after_overwrite = r3b.json()['payload']['items']
    assert any(x['run_id'] == run_id_overwrite for x in rows_after_overwrite)
    assert not any(x['run_id'] == run_id for x in rows_after_overwrite)
    run_id = run_id_overwrite

    rb = c.post('/api/experiments/run-batch', json={
        'session_id': sid,
        'experiment_ids': ['extratrees_pixel'],
        'param_sweep': 'high',
        'params': {},
        'params_by_experiment': {},
        'scribble_map_b64': encode_gray_png_b64(scrib),
    })
    assert rb.status_code == 200
    assert rb.json()['payload']['param_sweep'] == 'high'
    assert len(rb.json()['payload']['items']) >= 1
    rows_after_batch = c.get(f'/api/results/list?image_id={image_id}')
    assert rows_after_batch.status_code == 200
    batch_count = len(rows_after_batch.json()['payload']['items'])

    rb_overwrite = c.post('/api/experiments/run-batch', json={
        'session_id': sid,
        'experiment_ids': ['extratrees_pixel'],
        'param_sweep': 'high',
        'params': {},
        'params_by_experiment': {},
        'scribble_map_b64': encode_gray_png_b64(scrib),
    })
    assert rb_overwrite.status_code == 200
    rows_after_batch_overwrite = c.get(f'/api/results/list?image_id={image_id}')
    assert rows_after_batch_overwrite.status_code == 200
    assert len(rows_after_batch_overwrite.json()['payload']['items']) == batch_count

    img2 = _img_b(96)
    ok2, buf2 = cv2.imencode('.png', img2)
    assert ok2
    r1b = c.post('/api/image/load', data={'session_id': sid}, files={'file': ('b.png', bytes(buf2), 'image/png')})
    assert r1b.status_code == 200
    assert r1b.json()['payload']['image_id']

    r4 = c.post('/api/review/mark', json={'run_id': run_id, 'image_id': image_id, 'decision': 'ok', 'note': 'bien'})
    assert r4.status_code == 200

    r5 = c.get(f'/api/review/list?image_id={image_id}')
    assert r5.status_code == 200
    assert any(x['run_id'] == run_id for x in r5.json()['payload']['items'])

    r6 = c.get(f'/api/reports/export?image_id={image_id}')
    assert r6.status_code == 200
    payload = r6.json()['payload']
    assert payload['summary_csv']
    assert payload['summary_json']
    assert payload['gallery_png']

    reload_a = c.post('/api/library/load', json={'session_id': sid, 'image_id': image_id, 'restore_scribbles': True})
    assert reload_a.status_code == 200
    clear_results = c.post('/api/results/clear', json={'session_id': sid, 'image_id': image_id})
    assert clear_results.status_code == 200
    assert clear_results.json()['payload']['deleted_count'] >= 1
    listed_clear = c.get(f'/api/results/list?image_id={image_id}')
    assert listed_clear.status_code == 200
    assert listed_clear.json()['payload']['items'] == []
    reviews_clear = c.get(f'/api/review/list?image_id={image_id}')
    assert reviews_clear.status_code == 200
    assert reviews_clear.json()['payload']['items'] == []

    rd3 = c.post('/api/scribble/draft/clear', json={'session_id': sid, 'image_id': image_id})
    assert rd3.status_code == 200
    assert rd3.json()['payload']['cleared'] is True

    rd4 = c.get(f'/api/scribble/draft/load?session_id={sid}&image_id={image_id}')
    assert rd4.status_code == 200
    assert rd4.json()['payload']['found'] is False
