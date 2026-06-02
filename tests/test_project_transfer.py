from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from backend import project_transfer as transfer


def _definitions(root: Path) -> dict[str, transfer.TransferCategory]:
    return {
        'library': transfer.TransferCategory(
            'library',
            'Biblioteca de imagenes',
            (transfer.TransferSource('library', root / 'library'),),
        ),
        'scribble_experiments': transfer.TransferCategory(
            'scribble_experiments',
            'Experimentos Scribble',
            (
                transfer.TransferSource('runs', root / 'runs'),
                transfer.TransferSource('reports', root / 'reports'),
            ),
        ),
    }


def test_export_and_import_preserve_or_replace_conflicts(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / 'source'
    (source_root / 'library' / 'img_1').mkdir(parents=True)
    (source_root / 'library' / 'img_1' / 'image.png').write_bytes(b'new-image')
    (source_root / 'runs' / 'run_1').mkdir(parents=True)
    (source_root / 'runs' / 'run_1' / 'meta.json').write_text('{"run": 1}', encoding='utf-8')

    monkeypatch.setattr(transfer, '_category_definitions', lambda: _definitions(source_root))
    zip_path = tmp_path / 'project.zip'
    manifest = transfer._create_export_zip(['library', 'scribble_experiments'], zip_path)

    assert len(manifest['entries']) == 2
    with zipfile.ZipFile(zip_path, 'r') as archive:
        names = set(archive.namelist())
    assert 'manifest.json' in names
    assert 'payload/library/library/img_1/image.png' in names
    assert all('diameter' not in name and 'model_inference' not in name for name in names)

    destination_root = tmp_path / 'destination'
    (destination_root / 'library' / 'img_1').mkdir(parents=True)
    conflict = destination_root / 'library' / 'img_1' / 'image.png'
    conflict.write_bytes(b'old-image')
    monkeypatch.setattr(transfer, '_category_definitions', lambda: _definitions(destination_root))

    imported_manifest = transfer._validate_import_zip(zip_path)
    summary = transfer._summarize_import(imported_manifest['entries'])
    assert summary['conflict_count'] == 1
    assert summary['new_count'] == 1

    preserved = transfer._apply_import(zip_path, imported_manifest['entries'], overwrite=False)
    assert preserved == {
        'imported_count': 1,
        'skipped_count': 1,
        'replaced_count': 0,
        'categories': {
            'library': {'imported': 0, 'skipped': 1, 'replaced': 0},
            'scribble_experiments': {'imported': 1, 'skipped': 0, 'replaced': 0},
        },
    }
    assert conflict.read_bytes() == b'old-image'

    replaced = transfer._apply_import(zip_path, imported_manifest['entries'], overwrite=True)
    assert replaced['replaced_count'] == 2
    assert conflict.read_bytes() == b'new-image'


def test_import_rejects_unsafe_relative_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(transfer, '_category_definitions', lambda: _definitions(tmp_path / 'destination'))
    zip_path = tmp_path / 'unsafe.zip'
    manifest = {
        'format_version': transfer.FORMAT_VERSION,
        'created_at': '2026-06-01 12:00:00',
        'categories': ['library'],
        'entries': [],
    }
    with zipfile.ZipFile(zip_path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(manifest))
        archive.writestr('../escape.txt', 'blocked')

    with pytest.raises(ValueError, match='Ruta relativa invalida'):
        transfer._validate_import_zip(zip_path)


def test_import_rejects_unknown_manifest_category(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(transfer, '_category_definitions', lambda: _definitions(tmp_path / 'destination'))
    zip_path = tmp_path / 'unknown.zip'
    manifest = {
        'format_version': transfer.FORMAT_VERSION,
        'created_at': '2026-06-01 12:00:00',
        'categories': ['diameter_runs'],
        'entries': [],
    }
    with zipfile.ZipFile(zip_path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(manifest))

    with pytest.raises(ValueError, match='categorias desconocidas'):
        transfer._validate_import_zip(zip_path)
