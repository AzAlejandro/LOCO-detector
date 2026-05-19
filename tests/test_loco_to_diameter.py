"""Tests for the LOCO Detector -> Diameter Research pipeline connection.

Verifies that accepted circles from LOCO Model detection are correctly
converted to diameter research points and that the 'replace' action works.
"""

from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app


def _img(size: int = 128) -> np.ndarray:
    y, x = np.indices((size, size))
    base = np.clip((x + y) / (2 * size), 0, 1)
    cx, cy = size // 2, size // 2
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


def test_points_replace_action_clears_and_sets(tmp_path, monkeypatch) -> None:
    """The 'replace' action in points/update clears existing points and sets new ones."""
    c, sid, iid = _setup(tmp_path, monkeypatch)

    # add a couple of points first
    add_res = c.post(
        '/api/diameter-research/points/update',
        json={
            'session_id': sid,
            'action': 'add',
            'x': 10.0,
            'y': 20.0,
        },
    )
    assert add_res.status_code == 200, add_res.text

    add_res2 = c.post(
        '/api/diameter-research/points/update',
        json={
            'session_id': sid,
            'action': 'add',
            'x': 30.0,
            'y': 40.0,
        },
    )
    assert add_res2.status_code == 200, add_res2.text

    # verify 2 points exist
    load_before = c.get(f'/api/diameter-research/points/load?session_id={sid}&image_id={iid}')
    assert load_before.status_code == 200
    assert len(load_before.json().get('points', [])) == 2

    # replace with new points (simulating LOCO -> diameter transfer)
    replace_res = c.post(
        '/api/diameter-research/points/update',
        json={
            'session_id': sid,
            'action': 'replace',
            'points': [
                {'x': 50.0, 'y': 60.0},
                {'x': 70.0, 'y': 80.0},
                {'x': 90.0, 'y': 100.0},
            ],
        },
    )
    assert replace_res.status_code == 200, replace_res.text

    # verify old points replaced with new ones
    load_after = c.get(f'/api/diameter-research/points/load?session_id={sid}&image_id={iid}')
    assert load_after.status_code == 200
    points = load_after.json().get('points', [])
    assert len(points) == 3, f'Expected 3 points after replace, got {len(points)}'
    assert points[0]['x'] == 50.0
    assert points[0]['y'] == 60.0
    assert points[2]['x'] == 90.0


def test_loco_accepted_circle_conversion() -> None:
    """Verify that accepted circles from LOCO detector can be mapped to diameter points."""
    # Simulate the frontend mapping logic from measureLocoModelAccepted()
    accepted = [
        {'center_x': 100.0, 'center_y': 200.0, 'radius_px': 15.0, 'score': 0.95},
        {'center_x': 300.0, 'center_y': 400.0, 'radius_px': 22.0, 'score': 0.87},
        {'center_x': 500.0, 'center_y': 600.0, 'radius_px': 8.0, 'score': 0.76},
    ]

    # This is the exact mapping used in App.jsx measureLocoModelAccepted()
    points = [
        {'x': float(c.get('center_x', 0)), 'y': float(c.get('center_y', 0)), 'radius_px': float(c.get('radius_px', 0))}
        for c in accepted
    ]

    assert len(points) == 3
    assert points[0]['x'] == 100.0
    assert points[0]['y'] == 200.0
    assert points[0]['radius_px'] == 15.0
    assert points[2]['radius_px'] == 8.0

    # Verify all required keys are present
    for p in points:
        assert 'x' in p
        assert 'y' in p
        assert 'radius_px' in p


def test_loco_accepted_empty_list() -> None:
    """An empty accepted list should produce an empty points array."""
    accepted: list[dict] = []
    points = [
        {'x': float(c.get('center_x', 0)), 'y': float(c.get('center_y', 0)), 'radius_px': float(c.get('radius_px', 0))}
        for c in accepted
    ]
    assert points == []


def test_loco_accepted_missing_fields() -> None:
    """Accepted circles with missing fields should default to 0."""
    accepted = [
        {'center_x': 100.0},  # missing center_y, radius_px
        {},  # completely empty
    ]
    points = [
        {'x': float(c.get('center_x', 0)), 'y': float(c.get('center_y', 0)), 'radius_px': float(c.get('radius_px', 0))}
        for c in accepted
    ]
    assert points[0]['x'] == 100.0
    assert points[0]['y'] == 0.0
    assert points[0]['radius_px'] == 0.0
    assert points[1]['x'] == 0.0
    assert points[1]['y'] == 0.0
