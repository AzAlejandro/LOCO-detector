"""Tests for scale calibration (px -> nm/um) endpoints and logic."""

from __future__ import annotations

import json

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


def _setup(tmp_path, monkeypatch) -> tuple[TestClient, str, str]:
    """Create a test client with patched paths, session, and loaded image.
    
    Returns (client, session_id, image_id).
    """
    import backend.persistence as p
    import backend.library_store as ls
    import backend.diameter_research.api as api
    import backend.diameter_research.persistence as drp

    monkeypatch.setattr(p, 'OUTPUT_ROOT', tmp_path / 'outputs')
    monkeypatch.setattr(p, 'RUNS_DIR', p.OUTPUT_ROOT / 'runs')
    monkeypatch.setattr(p, 'REVIEWS_DIR', p.OUTPUT_ROOT / 'reviews')
    monkeypatch.setattr(p, 'INDEX_DIR', p.OUTPUT_ROOT / 'index')
    monkeypatch.setattr(p, 'REPORTS_DIR', p.OUTPUT_ROOT / 'reports')
    monkeypatch.setattr(p, 'DRAFTS_DIR', p.OUTPUT_ROOT / 'drafts')
    monkeypatch.setattr(ls, 'LIBRARY_DIR', p.OUTPUT_ROOT / 'library')
    monkeypatch.setattr(drp, 'OUTPUT_ROOT', tmp_path / 'outputs')
    monkeypatch.setattr(drp, 'POINTS_DIR', drp.OUTPUT_ROOT / 'points')
    monkeypatch.setattr(drp, 'RUNS_DIR', drp.OUTPUT_ROOT / 'runs')
    monkeypatch.setattr(drp, 'INDEX_DIR', drp.OUTPUT_ROOT / 'index')
    monkeypatch.setattr(drp, 'REPORTS_DIR', drp.OUTPUT_ROOT / 'reports')
    monkeypatch.setattr(api, 'CALIBRATION_DIR', tmp_path / 'data' / 'calibration')

    c = TestClient(app)
    sid = c.post('/api/session/new', json={}).json()['payload']['session_id']

    img = _img()
    ok, buf = cv2.imencode('.png', img)
    assert ok
    r = c.post('/api/image/load', data={'session_id': sid}, files={'file': ('test.png', bytes(buf), 'image/png')})
    assert r.status_code == 200, r.text
    iid = r.json()['payload']['image_id']
    return c, sid, iid


def test_calibration_save_load_delete(tmp_path, monkeypatch) -> None:
    """Calibration save/load/delete round-trip."""
    import backend.diameter_research.api as api

    c, _sid, iid = _setup(tmp_path, monkeypatch)
    cal_dir = tmp_path / 'data' / 'calibration'

    # --- save ---
    save_res = c.post(
        '/api/diameter-research/calibration/save',
        json={
            'image_id': iid,
            'enabled': True,
            'unit': 'nm',
            'known_value_px': 100.0,
            'pixel_distance': 50.0,
            'nm_per_px': 2.0,
        },
    )
    assert save_res.status_code == 200, save_res.text
    assert save_res.json()['ok'] is True

    # verify file exists on disk
    safe = api._calibration_safe_id(iid)
    cal_file = cal_dir / f'{safe}.json'
    assert cal_file.exists(), f'Calibration file not found: {cal_file}'
    raw = json.loads(cal_file.read_text(encoding='utf-8'))
    assert raw['unit'] == 'nm'
    assert raw['nm_per_px'] == 2.0

    # --- load ---
    load_res = c.get(f'/api/diameter-research/calibration/load?image_id={iid}')
    assert load_res.status_code == 200, load_res.text
    body = load_res.json()
    assert body['ok'] is True
    assert body['calibration'] is not None
    assert body['calibration']['nm_per_px'] == 2.0

    # --- load missing (no calibration saved) ---
    load_miss = c.get('/api/diameter-research/calibration/load?image_id=nonexistent')
    assert load_miss.status_code == 200, load_miss.text
    assert load_miss.json()['calibration'] is None

    # --- delete ---
    del_res = c.post(
        '/api/diameter-research/calibration/delete',
        json={'image_id': iid},
    )
    assert del_res.status_code == 200, del_res.text
    assert del_res.json()['ok'] is True
    assert not cal_file.exists(), 'Calibration file should have been deleted'

    # --- load after delete ---
    load_after = c.get(f'/api/diameter-research/calibration/load?image_id={iid}')
    assert load_after.status_code == 200, load_after.text
    assert load_after.json()['calibration'] is None


def test_calibration_load_requires_image_id(tmp_path, monkeypatch) -> None:
    """GET /calibration/load without image_id returns 400."""
    import backend.diameter_research.api as api

    monkeypatch.setattr(api, 'CALIBRATION_DIR', tmp_path / 'data' / 'calibration')

    c = TestClient(app)
    res = c.get('/api/diameter-research/calibration/load?image_id=')
    assert res.status_code == 400


def test_calibration_save_overwrites_existing(tmp_path, monkeypatch) -> None:
    """Saving calibration twice overwrites the previous values."""
    c, _sid, iid = _setup(tmp_path, monkeypatch)

    # save first version
    c.post(
        '/api/diameter-research/calibration/save',
        json={'image_id': iid, 'enabled': True, 'unit': 'nm', 'known_value_px': 100.0, 'pixel_distance': 50.0, 'nm_per_px': 2.0},
    )

    # save second version (overwrite)
    c.post(
        '/api/diameter-research/calibration/save',
        json={'image_id': iid, 'enabled': False, 'unit': 'um', 'known_value_px': 200.0, 'pixel_distance': 80.0, 'nm_per_px': 2.5},
    )

    load_res = c.get(f'/api/diameter-research/calibration/load?image_id={iid}')
    assert load_res.status_code == 200
    cal = load_res.json()['calibration']
    assert cal['unit'] == 'um'
    assert cal['nm_per_px'] == 2.5
    assert cal['enabled'] is False


def test_calibration_safe_id_sanitizes_special_chars(tmp_path, monkeypatch) -> None:
    """_calibration_safe_id replaces special characters with underscores."""
    import backend.diameter_research.api as api

    monkeypatch.setattr(api, 'CALIBRATION_DIR', tmp_path / 'data' / 'calibration')

    dirty_id = 'my image: test@123!'
    c = TestClient(app)
    c.post(
        '/api/diameter-research/calibration/save',
        json={'image_id': dirty_id, 'enabled': True, 'unit': 'nm', 'known_value_px': 50.0, 'pixel_distance': 25.0, 'nm_per_px': 2.0},
    )

    safe = api._calibration_safe_id(dirty_id)
    cal_file = tmp_path / 'data' / 'calibration' / f'{safe}.json'
    assert cal_file.exists(), f'Expected {cal_file} to exist'
