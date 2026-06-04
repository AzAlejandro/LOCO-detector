from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import time
from typing import Any
from uuid import uuid4
import zipfile
import zlib

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from . import persistence as scribble_persistence
from .diameter_research import persistence as diameter_persistence


router = APIRouter(prefix='/api/project-transfer', tags=['project-transfer'])

FORMAT_VERSION = 'loco-training-project-v1'
MAX_UNCOMPRESSED_BYTES = 20 * 1024 * 1024 * 1024
MAX_UPLOAD_BYTES = 20 * 1024 * 1024 * 1024
TOKEN_TTL_SECONDS = 24 * 60 * 60
TEMP_ROOT = Path(tempfile.gettempdir()) / 'loco_detector_project_transfer'


@dataclass(frozen=True)
class TransferSource:
    key: str
    root: Path


@dataclass(frozen=True)
class TransferCategory:
    key: str
    label: str
    sources: tuple[TransferSource, ...]


class ExportPrepareReq(BaseModel):
    categories: list[str] = Field(default_factory=list)


class ImportApplyReq(BaseModel):
    token: str
    overwrite: bool = False


_tokens: dict[str, dict[str, Any]] = {}


def _category_definitions() -> dict[str, TransferCategory]:
    base = scribble_persistence.OUTPUT_ROOT
    diam = diameter_persistence.OUTPUT_ROOT
    return {
        'library': TransferCategory('library', 'Biblioteca de imagenes', (TransferSource('library', base / 'library'),)),
        'projects': TransferCategory('projects', 'Proyectos y tags', (TransferSource('projects', base / 'projects'),)),
        'drafts': TransferCategory('drafts', 'Scribbles guardados', (TransferSource('drafts', base / 'drafts'),)),
        'scribble_experiments': TransferCategory(
            'scribble_experiments',
            'Experimentos Scribble',
            (
                TransferSource('runs', base / 'runs'),
                TransferSource('index', base / 'index'),
                TransferSource('reviews', base / 'reviews'),
                TransferSource('reports', base / 'reports'),
            ),
        ),
        'assist_models': TransferCategory('assist_models', 'Modelos de asistencia', (TransferSource('assist_models', base / 'assist_models'),)),
        'loco_dataset': TransferCategory('loco_dataset', 'Dataset LOCO', (TransferSource('datasets', diam / 'datasets'),)),
        'loco_training_runs': TransferCategory('loco_training_runs', 'Historial de entrenamiento LOCO', (TransferSource('training_runs', diam / 'training_runs'),)),
        'loco_saved_models': TransferCategory('loco_saved_models', 'Modelos LOCO guardados', (TransferSource('saved_training_models', diam / 'saved_training_models'),)),
    }


def _cleanup_token(token: str) -> None:
    item = _tokens.pop(str(token or ''), None)
    if not item:
        return
    root = Path(item.get('temp_dir') or '')
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


def _cleanup_expired_tokens() -> None:
    now = time.time()
    for token, item in list(_tokens.items()):
        if now - float(item.get('created_ts') or 0.0) > TOKEN_TTL_SECONDS:
            _cleanup_token(token)


def _new_temp_dir() -> Path:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='transfer_', dir=str(TEMP_ROOT)))


def _safe_relative_path(text: str) -> PurePosixPath:
    raw = str(text or '').strip()
    if not raw or '\\' in raw or ':' in raw:
        raise ValueError(f'Ruta relativa invalida: {raw!r}')
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {'', '.', '..'} for part in path.parts):
        raise ValueError(f'Ruta relativa invalida: {raw!r}')
    return path


def _safe_destination(category_key: str, source_key: str, relative_path: str) -> Path:
    categories = _category_definitions()
    category = categories.get(str(category_key or ''))
    if category is None:
        raise ValueError(f'Categoria desconocida: {category_key}')
    source = next((item for item in category.sources if item.key == source_key), None)
    if source is None:
        raise ValueError(f'Fuente desconocida para {category_key}: {source_key}')
    relative = _safe_relative_path(relative_path)
    root = source.root.resolve()
    target = root.joinpath(*relative.parts).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'Ruta fuera del destino permitido: {relative_path}') from exc
    return target


def _iter_source_files(source: TransferSource) -> list[tuple[Path, str]]:
    if not source.root.exists():
        return []
    rows: list[tuple[Path, str]] = []
    root = source.root.resolve()
    for path in sorted(source.root.rglob('*')):
        if path.is_symlink() or not path.is_file():
            continue
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError:
            continue
        rows.append((resolved, relative))
    return rows


def _category_summary(category: TransferCategory) -> dict[str, Any]:
    file_count = 0
    size_bytes = 0
    for source in category.sources:
        for path, _relative in _iter_source_files(source):
            file_count += 1
            size_bytes += int(path.stat().st_size)
    return {'key': category.key, 'label': category.label, 'file_count': file_count, 'size_bytes': size_bytes, 'selected': True}


def _selected_categories(keys: list[str]) -> list[TransferCategory]:
    definitions = _category_definitions()
    selected: list[TransferCategory] = []
    seen: set[str] = set()
    for key in keys:
        normalized = str(key or '').strip()
        if normalized in seen:
            continue
        category = definitions.get(normalized)
        if category is None:
            raise ValueError(f'Categoria desconocida: {normalized}')
        seen.add(normalized)
        selected.append(category)
    if not selected:
        raise ValueError('Selecciona al menos una categoria.')
    return selected


def _zip_archive_path(category: str, source: str, relative_path: str) -> str:
    relative = _safe_relative_path(relative_path)
    return PurePosixPath('payload', category, source, *relative.parts).as_posix()


def _create_export_zip(categories: list[str], zip_path: Path) -> dict[str, Any]:
    selected = _selected_categories(categories)
    entries: list[dict[str, Any]] = []
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for category in selected:
            for source in category.sources:
                for path, relative_path in _iter_source_files(source):
                    archive_path = _zip_archive_path(category.key, source.key, relative_path)
                    archive.write(path, archive_path)
                    info = archive.getinfo(archive_path)
                    entries.append(
                        {
                            'category': category.key,
                            'source': source.key,
                            'path': relative_path,
                            'archive_path': archive_path,
                            'size_bytes': int(info.file_size),
                            'crc32': int(info.CRC),
                        }
                    )
        manifest = {
            'format_version': FORMAT_VERSION,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'categories': [item.key for item in selected],
            'entries': entries,
        }
        archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK


def _read_member_crc(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[int, int]:
    crc = 0
    size = 0
    with archive.open(info, 'r') as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            crc = zlib.crc32(chunk, crc)
    return size, crc & 0xFFFFFFFF


def _validate_import_zip(zip_path: Path) -> dict[str, Any]:
    try:
        archive = zipfile.ZipFile(zip_path, 'r')
    except zipfile.BadZipFile as exc:
        raise ValueError('El archivo no es un ZIP valido.') from exc
    with archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        if len(names) != len(set(names)):
            raise ValueError('El ZIP contiene rutas duplicadas.')
        for info in members:
            if _is_symlink(info):
                raise ValueError(f'El ZIP contiene un enlace no permitido: {info.filename}')
            if not info.is_dir():
                _safe_relative_path(info.filename)
        if 'manifest.json' not in names:
            raise ValueError('Falta manifest.json en el ZIP.')
        manifest_info = archive.getinfo('manifest.json')
        if int(manifest_info.file_size) > 5 * 1024 * 1024:
            raise ValueError('manifest.json excede el tamano permitido.')
        try:
            manifest = dict(json.loads(archive.read('manifest.json').decode('utf-8')) or {})
        except Exception as exc:
            raise ValueError('manifest.json no contiene JSON valido.') from exc
        if str(manifest.get('format_version') or '') != FORMAT_VERSION:
            raise ValueError('Version de paquete no soportada.')
        manifest_categories = list(manifest.get('categories') or [])
        known_categories = _category_definitions()
        if not all(isinstance(key, str) and key in known_categories for key in manifest_categories):
            raise ValueError('El manifiesto contiene categorias desconocidas.')
        entries = list(manifest.get('entries') or [])
        if not all(isinstance(item, dict) for item in entries):
            raise ValueError('Las entradas del manifiesto son invalidas.')
        expected_names = {'manifest.json'}
        total_size = 0
        normalized_entries: list[dict[str, Any]] = []
        for raw in entries:
            category = str(raw.get('category') or '')
            source = str(raw.get('source') or '')
            relative_path = _safe_relative_path(str(raw.get('path') or '')).as_posix()
            archive_path = _zip_archive_path(category, source, relative_path)
            if archive_path != str(raw.get('archive_path') or ''):
                raise ValueError(f'Ruta de archivo inconsistente: {archive_path}')
            _safe_destination(category, source, relative_path)
            if archive_path in expected_names:
                raise ValueError(f'Entrada duplicada en manifiesto: {archive_path}')
            expected_names.add(archive_path)
            try:
                info = archive.getinfo(archive_path)
            except KeyError as exc:
                raise ValueError(f'Archivo declarado no encontrado: {archive_path}') from exc
            expected_size = int(raw.get('size_bytes') or 0)
            expected_crc = int(raw.get('crc32') or 0)
            if expected_size != int(info.file_size) or expected_crc != int(info.CRC):
                raise ValueError(f'Metadata ZIP inconsistente: {archive_path}')
            total_size += expected_size
            if total_size > MAX_UNCOMPRESSED_BYTES:
                raise ValueError('El ZIP excede el limite descomprimido de 20 GB.')
            actual_size, actual_crc = _read_member_crc(archive, info)
            if actual_size != expected_size or actual_crc != expected_crc:
                raise ValueError(f'CRC invalido: {archive_path}')
            normalized_entries.append(
                {
                    'category': category,
                    'source': source,
                    'path': relative_path,
                    'archive_path': archive_path,
                    'size_bytes': expected_size,
                    'crc32': expected_crc,
                }
            )
        file_names = {item.filename for item in members if not item.is_dir()}
        if file_names != expected_names:
            raise ValueError('El ZIP contiene archivos no declarados en manifest.json.')
        manifest['entries'] = normalized_entries
        return manifest


def _summarize_import(entries: list[dict[str, Any]]) -> dict[str, Any]:
    categories = _category_definitions()
    rows: dict[str, dict[str, Any]] = {}
    conflicts: list[str] = []
    for entry in entries:
        category_key = str(entry['category'])
        row = rows.setdefault(
            category_key,
            {
                'key': category_key,
                'label': categories[category_key].label,
                'file_count': 0,
                'new_count': 0,
                'conflict_count': 0,
                'size_bytes': 0,
                'conflicts': [],
            },
        )
        target = _safe_destination(category_key, str(entry['source']), str(entry['path']))
        row['file_count'] += 1
        row['size_bytes'] += int(entry['size_bytes'])
        if target.exists():
            label = f"{category_key}/{entry['source']}/{entry['path']}"
            row['conflict_count'] += 1
            row['conflicts'].append(label)
            conflicts.append(label)
        else:
            row['new_count'] += 1
    return {
        'categories': [rows[key] for key in sorted(rows)],
        'file_count': len(entries),
        'conflict_count': len(conflicts),
        'new_count': len(entries) - len(conflicts),
        'size_bytes': sum(int(item['size_bytes']) for item in entries),
        'conflicts': conflicts,
    }


def _copy_member_atomic(archive: zipfile.ZipFile, entry: dict[str, Any], target: Path) -> None:
    info = archive.getinfo(str(entry['archive_path']))
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.parent / f'.{target.name}.import-{uuid4().hex}.tmp'
    try:
        with archive.open(info, 'r') as source, temp_path.open('wb') as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        if int(temp_path.stat().st_size) != int(entry['size_bytes']):
            raise ValueError(f'Tamano importado invalido: {entry["archive_path"]}')
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _apply_import(zip_path: Path, entries: list[dict[str, Any]], overwrite: bool) -> dict[str, Any]:
    imported = 0
    skipped = 0
    replaced = 0
    per_category: dict[str, dict[str, int]] = {}
    with zipfile.ZipFile(zip_path, 'r') as archive:
        for entry in entries:
            category = str(entry['category'])
            row = per_category.setdefault(category, {'imported': 0, 'skipped': 0, 'replaced': 0})
            target = _safe_destination(category, str(entry['source']), str(entry['path']))
            exists = target.exists()
            if exists and not overwrite:
                skipped += 1
                row['skipped'] += 1
                continue
            _copy_member_atomic(archive, entry, target)
            imported += 1
            row['imported'] += 1
            if exists:
                replaced += 1
                row['replaced'] += 1
    return {'imported_count': imported, 'skipped_count': skipped, 'replaced_count': replaced, 'categories': per_category}


async def _save_upload(upload: UploadFile, path: Path) -> None:
    size = 0
    with path.open('wb') as destination:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise ValueError('El ZIP excede el limite de carga de 20 GB.')
            destination.write(chunk)


@router.get('/catalog')
def project_transfer_catalog() -> dict[str, Any]:
    _cleanup_expired_tokens()
    return {'ok': True, 'categories': [_category_summary(item) for item in _category_definitions().values()]}


@router.post('/export/prepare')
def project_transfer_export_prepare(req: ExportPrepareReq) -> dict[str, Any]:
    _cleanup_expired_tokens()
    temp_dir = _new_temp_dir()
    token = uuid4().hex
    zip_name = f"loco_training_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = temp_dir / zip_name
    try:
        manifest = _create_export_zip(req.categories, zip_path)
    except ValueError as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _tokens[token] = {'kind': 'export', 'temp_dir': str(temp_dir), 'zip_path': str(zip_path), 'created_ts': time.time()}
    return {
        'ok': True,
        'token': token,
        'file_name': zip_name,
        'size_bytes': int(zip_path.stat().st_size),
        'file_count': len(manifest['entries']),
        'categories': list(manifest['categories']),
    }


@router.get('/export/download')
def project_transfer_export_download(token: str) -> FileResponse:
    _cleanup_expired_tokens()
    item = _tokens.get(str(token or ''))
    if not item or item.get('kind') != 'export':
        raise HTTPException(status_code=404, detail='Exportacion no encontrada o expirada.')
    zip_path = Path(item['zip_path'])
    if not zip_path.exists():
        _cleanup_token(token)
        raise HTTPException(status_code=404, detail='Archivo de exportacion no encontrado.')
    return FileResponse(
        path=zip_path,
        filename=zip_path.name,
        media_type='application/zip',
        background=BackgroundTask(_cleanup_token, token),
    )


@router.post('/import/inspect')
async def project_transfer_import_inspect(file: UploadFile = File(...)) -> dict[str, Any]:
    _cleanup_expired_tokens()
    temp_dir = _new_temp_dir()
    zip_path = temp_dir / 'import.zip'
    try:
        await _save_upload(file, zip_path)
        manifest = _validate_import_zip(zip_path)
        summary = _summarize_import(list(manifest['entries']))
    except ValueError as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    token = uuid4().hex
    _tokens[token] = {'kind': 'import', 'temp_dir': str(temp_dir), 'zip_path': str(zip_path), 'manifest': manifest, 'created_ts': time.time()}
    return {'ok': True, 'token': token, 'summary': summary}


@router.post('/import/apply')
def project_transfer_import_apply(req: ImportApplyReq) -> dict[str, Any]:
    _cleanup_expired_tokens()
    token = str(req.token or '')
    item = _tokens.get(token)
    if not item or item.get('kind') != 'import':
        raise HTTPException(status_code=404, detail='Importacion no encontrada o expirada.')
    zip_path = Path(item['zip_path'])
    try:
        manifest = _validate_import_zip(zip_path)
        result = _apply_import(zip_path, list(manifest['entries']), bool(req.overwrite))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        _cleanup_token(token)
    return {'ok': True, 'result': result}
