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
from typing import Any, Literal
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

FORMAT_VERSION = 'loco-training-project-v2'
SUPPORTED_FORMAT_VERSIONS = {'loco-training-project-v1', FORMAT_VERSION}
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


@dataclass(frozen=True)
class ExportEntry:
    path: Path | None
    relative_path: str
    data: bytes | None = None
    project_ids: tuple[str, ...] = ()


class ExportPrepareReq(BaseModel):
    categories: list[str] = Field(default_factory=list)
    project_ids: list[str] = Field(default_factory=list)
    mode: Literal['full', 'project'] = 'full'


class ImportApplyReq(BaseModel):
    token: str
    overwrite: bool = False
    project_ids: list[str] = Field(default_factory=list)


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
        'diameter_analysis': TransferCategory('diameter_analysis', 'Analisis de diametros', (TransferSource('diameter_analysis', base / 'diameter_analysis'), TransferSource('calibration', Path(__file__).resolve().parents[1] / 'data' / 'calibration'))),
    }


def _read_json_file(path: Path, fallback: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else fallback
    except Exception:
        return fallback


def _project_state() -> dict[str, Any]:
    path = scribble_persistence.OUTPUT_ROOT / 'projects' / 'projects.json'
    payload = _read_json_file(path, {'active_project_id': '', 'projects': []})
    if not isinstance(payload, dict):
        payload = {'active_project_id': '', 'projects': []}
    projects = payload.get('projects')
    if not isinstance(projects, list):
        projects = []
    return {'active_project_id': str(payload.get('active_project_id') or ''), 'projects': [dict(item or {}) for item in projects if isinstance(item, dict)]}


def _normalize_project_ids(project_ids: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    valid = {str(item.get('project_id') or '') for item in _project_state().get('projects') or []}
    for raw in project_ids or []:
        pid = str(raw or '').strip()
        if not pid or pid in seen:
            continue
        if valid and pid not in valid:
            continue
        seen.add(pid)
        out.append(pid)
    return out


def _project_rows(project_ids: list[str]) -> list[dict[str, Any]]:
    ids = set(project_ids)
    return [dict(item) for item in _project_state().get('projects') or [] if str(item.get('project_id') or '') in ids]


def _library_image_project_map() -> dict[str, set[str]]:
    root = scribble_persistence.OUTPUT_ROOT / 'library'
    out: dict[str, set[str]] = {}
    if not root.exists():
        return out
    for meta_path in root.glob('*/meta.json'):
        meta = _read_json_file(meta_path, {})
        if not isinstance(meta, dict):
            continue
        image_id = str(meta.get('image_id') or meta_path.parent.name or '').strip()
        if not image_id:
            continue
        out[image_id] = {str(pid or '').strip() for pid in (meta.get('project_ids') or []) if str(pid or '').strip()}
    return out


def _selected_image_ids(project_ids: list[str]) -> set[str]:
    selected = set(project_ids)
    if not selected:
        return set()
    return {image_id for image_id, ids in _library_image_project_map().items() if ids & selected}


def _entry_project_ids_from_image_ids(image_ids: set[str], image_project_map: dict[str, set[str]], selected_projects: set[str]) -> tuple[str, ...]:
    ids: set[str] = set()
    for image_id in image_ids:
        ids.update(image_project_map.get(image_id) or set())
    if selected_projects:
        ids &= selected_projects
    return tuple(sorted(ids))


def _text_file_contains_any(path: Path, needles: set[str]) -> bool:
    if not needles or not path.exists() or path.stat().st_size > 20 * 1024 * 1024:
        return False
    try:
        text = path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return False
    return any(needle in text for needle in needles)


def _path_image_refs(relative_path: str, selected_images: set[str]) -> set[str]:
    return {image_id for image_id in selected_images if image_id and image_id in relative_path}


def _json_image_refs(path: Path, selected_images: set[str]) -> set[str]:
    if not path.exists() or path.suffix.lower() != '.json':
        return set()
    raw = _read_json_file(path, None)
    if raw is None:
        return set()
    text = json.dumps(raw, ensure_ascii=False)
    return {image_id for image_id in selected_images if image_id in text}


def _model_dir_image_refs(meta_path: Path, selected_images: set[str]) -> set[str]:
    meta = _read_json_file(meta_path, {})
    if not isinstance(meta, dict):
        return set()
    ids = {str(item or '') for item in (meta.get('image_ids') or []) if str(item or '')}
    for row in meta.get('images') or []:
        if isinstance(row, dict) and str(row.get('image_id') or ''):
            ids.add(str(row.get('image_id')))
    return ids & selected_images


def _filtered_projects_json(project_ids: list[str]) -> bytes:
    state = _project_state()
    ids = set(project_ids)
    projects = [item for item in state.get('projects') or [] if str(item.get('project_id') or '') in ids]
    active = str(state.get('active_project_id') or '')
    if active not in ids:
        active = project_ids[0] if project_ids else ''
    return json.dumps({'active_project_id': active, 'projects': projects}, ensure_ascii=False, indent=2).encode('utf-8')


def _filtered_assist_registry(model_ids: set[str]) -> bytes:
    root = scribble_persistence.OUTPUT_ROOT / 'assist_models'
    payload = _read_json_file(root / 'registry.json', {'default_model_id': '', 'models': []})
    if not isinstance(payload, dict):
        payload = {'default_model_id': '', 'models': []}
    models = [dict(item or {}) for item in (payload.get('models') or []) if str((item or {}).get('model_id') or '') in model_ids]
    default_id = str(payload.get('default_model_id') or '')
    if default_id not in model_ids:
        default_id = str(models[0].get('model_id') or '') if models else ''
    return json.dumps({'default_model_id': default_id, 'models': models}, ensure_ascii=False, indent=2).encode('utf-8')


def _filtered_measurements_json(selected_images: set[str]) -> bytes | None:
    path = scribble_persistence.OUTPUT_ROOT / 'diameter_analysis' / 'measurements.json'
    payload = _read_json_file(path, None)
    if not isinstance(payload, dict):
        return None
    rows = [dict(item or {}) for item in (payload.get('measurements') or []) if str((item or {}).get('image_id') or '') in selected_images]
    if not rows:
        return None
    out = dict(payload)
    out['measurements'] = rows
    return json.dumps(out, ensure_ascii=False, indent=2).encode('utf-8')


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


def _export_entries_for_source(
    category_key: str,
    source: TransferSource,
    *,
    mode: str = 'full',
    project_ids: list[str] | None = None,
    selected_images: set[str] | None = None,
    image_project_map: dict[str, set[str]] | None = None,
) -> list[ExportEntry]:
    if mode != 'project' or not project_ids:
        return [ExportEntry(path=path, relative_path=relative) for path, relative in _iter_source_files(source)]

    selected = set(project_ids or [])
    images = set(selected_images or set())
    image_projects = image_project_map or {}
    rows: list[ExportEntry] = []

    if category_key == 'projects' and source.key == 'projects':
        data = _filtered_projects_json(project_ids)
        if data:
            rows.append(ExportEntry(path=None, relative_path='projects.json', data=data, project_ids=tuple(project_ids)))
        return rows

    if category_key == 'diameter_analysis' and source.key == 'diameter_analysis':
        data = _filtered_measurements_json(images)
        if data:
            rows.append(ExportEntry(path=None, relative_path='measurements.json', data=data, project_ids=tuple(project_ids)))
        return rows

    assist_model_ids: set[str] = set()
    if category_key == 'assist_models':
        model_root = source.root / 'models'
        if model_root.exists():
            for meta_path in model_root.glob('*/meta.json'):
                refs = _model_dir_image_refs(meta_path, images)
                if refs:
                    assist_model_ids.add(meta_path.parent.name)
        for path, relative in _iter_source_files(source):
            parts = PurePosixPath(relative).parts
            if relative == 'registry.json':
                data = _filtered_assist_registry(assist_model_ids)
                rows.append(ExportEntry(path=None, relative_path=relative, data=data, project_ids=tuple(project_ids)))
                continue
            if len(parts) >= 2 and parts[0] == 'models' and parts[1] in assist_model_ids:
                refs = _model_dir_image_refs(source.root / 'models' / parts[1] / 'meta.json', images)
                rows.append(ExportEntry(path=path, relative_path=relative, project_ids=_entry_project_ids_from_image_ids(refs, image_projects, selected)))
        return rows

    owner_cache: dict[str, tuple[bool, tuple[str, ...]]] = {}
    for path, relative in _iter_source_files(source):
        parts = PurePosixPath(relative).parts
        refs: set[str] = set()
        include = False

        if category_key == 'library' and parts:
            image_id = parts[0]
            include = image_id in images
            refs = {image_id} if include else set()
        elif category_key == 'drafts' and parts:
            image_id = parts[0]
            include = image_id in images
            refs = {image_id} if include else set()
        elif category_key in {'loco_training_runs', 'loco_saved_models'} and parts:
            owner = parts[0]
            if owner not in owner_cache:
                root = source.root / owner
                meta_refs = _json_image_refs(root / 'run_meta.json', images) | _json_image_refs(root / 'saved_model_meta.json', images)
                owner_cache[owner] = (bool(meta_refs), _entry_project_ids_from_image_ids(meta_refs, image_projects, selected))
            include, owner_project_ids = owner_cache[owner]
            if include:
                rows.append(ExportEntry(path=path, relative_path=relative, project_ids=owner_project_ids))
            continue
        else:
            refs = _path_image_refs(relative, images) | _json_image_refs(path, images)
            include = bool(refs)
            if not include and path.suffix.lower() in {'.csv', '.txt', '.json'}:
                include = _text_file_contains_any(path, images)
                refs = _path_image_refs(relative, images)

        if include:
            rows.append(ExportEntry(path=path, relative_path=relative, project_ids=_entry_project_ids_from_image_ids(refs, image_projects, selected)))
    return rows


def _category_entries(category: TransferCategory, mode: str = 'full', project_ids: list[str] | None = None) -> list[tuple[TransferSource, ExportEntry]]:
    ids = _normalize_project_ids(project_ids or []) if mode == 'project' else []
    image_project_map = _library_image_project_map()
    selected_images = _selected_image_ids(ids)
    rows: list[tuple[TransferSource, ExportEntry]] = []
    for source in category.sources:
        for entry in _export_entries_for_source(
            category.key,
            source,
            mode=mode,
            project_ids=ids,
            selected_images=selected_images,
            image_project_map=image_project_map,
        ):
            rows.append((source, entry))
    return rows


def _category_summary(category: TransferCategory, mode: str = 'full', project_ids: list[str] | None = None) -> dict[str, Any]:
    file_count = 0
    size_bytes = 0
    for _source, entry in _category_entries(category, mode, project_ids):
        file_count += 1
        if entry.data is not None:
            size_bytes += len(entry.data)
        elif entry.path is not None:
            size_bytes += int(entry.path.stat().st_size)
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


def _create_export_zip(categories: list[str], zip_path: Path, *, mode: str = 'full', project_ids: list[str] | None = None) -> dict[str, Any]:
    selected = _selected_categories(categories)
    export_mode = 'project' if mode == 'project' and project_ids else 'full'
    selected_project_ids = _normalize_project_ids(project_ids or []) if export_mode == 'project' else []
    entries: list[dict[str, Any]] = []
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for category in selected:
            for source in category.sources:
                source_entries = _export_entries_for_source(
                    category.key,
                    source,
                    mode=export_mode,
                    project_ids=selected_project_ids,
                    selected_images=_selected_image_ids(selected_project_ids),
                    image_project_map=_library_image_project_map(),
                )
                for entry in source_entries:
                    archive_path = _zip_archive_path(category.key, source.key, entry.relative_path)
                    if entry.data is not None:
                        archive.writestr(archive_path, entry.data)
                    elif entry.path is not None:
                        archive.write(entry.path, archive_path)
                    else:
                        continue
                    info = archive.getinfo(archive_path)
                    entries.append(
                        {
                            'category': category.key,
                            'source': source.key,
                            'path': entry.relative_path,
                            'archive_path': archive_path,
                            'size_bytes': int(info.file_size),
                            'crc32': int(info.CRC),
                            'project_ids': list(entry.project_ids or selected_project_ids),
                        }
                    )
        manifest = {
            'format_version': FORMAT_VERSION,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'export_mode': export_mode,
            'project_ids': selected_project_ids,
            'projects': _project_rows(selected_project_ids),
            'project_filter_policy': 'include_if_touches' if export_mode == 'project' else '',
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
        if str(manifest.get('format_version') or '') not in SUPPORTED_FORMAT_VERSIONS:
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
                    'project_ids': [str(pid or '') for pid in (raw.get('project_ids') or []) if str(pid or '')],
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


def _filter_entries_by_projects(entries: list[dict[str, Any]], project_ids: list[str]) -> list[dict[str, Any]]:
    selected = {str(pid or '').strip() for pid in project_ids or [] if str(pid or '').strip()}
    if not selected:
        return entries
    out: list[dict[str, Any]] = []
    for entry in entries:
        entry_projects = {str(pid or '').strip() for pid in (entry.get('project_ids') or []) if str(pid or '').strip()}
        if not entry_projects or entry_projects & selected:
            out.append(entry)
    return out


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


def _merge_projects_file(zip_path: Path, archive: zipfile.ZipFile, entry: dict[str, Any], selected_project_ids: list[str]) -> dict[str, int]:
    info = archive.getinfo(str(entry['archive_path']))
    incoming = json.loads(archive.read(info).decode('utf-8'))
    incoming_projects = [dict(item or {}) for item in (incoming.get('projects') or []) if isinstance(item, dict)]
    selected = {str(pid or '').strip() for pid in selected_project_ids or [] if str(pid or '').strip()}
    if selected:
        incoming_projects = [item for item in incoming_projects if str(item.get('project_id') or '') in selected]
    target = _safe_destination('projects', 'projects', 'projects.json')
    current = _read_json_file(target, {'active_project_id': '', 'projects': []})
    if not isinstance(current, dict):
        current = {'active_project_id': '', 'projects': []}
    existing = {str(item.get('project_id') or ''): dict(item or {}) for item in (current.get('projects') or []) if isinstance(item, dict)}
    replaced = 0
    imported = 0
    for project in incoming_projects:
        pid = str(project.get('project_id') or '')
        if not pid:
            continue
        if pid in existing:
            replaced += 1
        else:
            imported += 1
        existing[pid] = project
    active = str(current.get('active_project_id') or '')
    if not active and incoming_projects:
        active = str(incoming_projects[0].get('project_id') or '')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'active_project_id': active, 'projects': list(existing.values())}, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'imported': imported, 'replaced': replaced, 'skipped': 0}


def _merge_measurements_file(archive: zipfile.ZipFile, entry: dict[str, Any]) -> dict[str, int]:
    info = archive.getinfo(str(entry['archive_path']))
    incoming = json.loads(archive.read(info).decode('utf-8'))
    incoming_rows = [dict(item or {}) for item in (incoming.get('measurements') or []) if isinstance(item, dict)]
    target = _safe_destination('diameter_analysis', 'diameter_analysis', 'measurements.json')
    current = _read_json_file(target, {'measurements': []})
    if not isinstance(current, dict):
        current = {'measurements': []}
    existing_rows = [dict(item or {}) for item in (current.get('measurements') or []) if isinstance(item, dict)]
    def key(row: dict[str, Any]) -> str:
        return '|'.join(str(row.get(k) or '') for k in ('image_id', 'run_id', 'result_id', 'point_index', 'method_id'))
    index = {key(row): row for row in existing_rows}
    imported = 0
    replaced = 0
    for row in incoming_rows:
        k = key(row)
        if k in index:
            replaced += 1
        else:
            imported += 1
        index[k] = row
    out = dict(current)
    out['measurements'] = list(index.values())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'imported': imported, 'replaced': replaced, 'skipped': 0}


def _apply_import(zip_path: Path, entries: list[dict[str, Any]], overwrite: bool, project_ids: list[str] | None = None) -> dict[str, Any]:
    imported = 0
    skipped = 0
    replaced = 0
    per_category: dict[str, dict[str, int]] = {}
    selected_project_ids = [str(pid or '').strip() for pid in (project_ids or []) if str(pid or '').strip()]
    entries = _filter_entries_by_projects(entries, selected_project_ids)
    with zipfile.ZipFile(zip_path, 'r') as archive:
        for entry in entries:
            category = str(entry['category'])
            row = per_category.setdefault(category, {'imported': 0, 'skipped': 0, 'replaced': 0})
            target = _safe_destination(category, str(entry['source']), str(entry['path']))
            if category == 'projects' and str(entry['source']) == 'projects' and str(entry['path']) == 'projects.json':
                result = _merge_projects_file(zip_path, archive, entry, selected_project_ids)
                imported += result['imported']
                replaced += result['replaced']
                row['imported'] += result['imported']
                row['replaced'] += result['replaced']
                continue
            if category == 'diameter_analysis' and str(entry['source']) == 'diameter_analysis' and str(entry['path']) == 'measurements.json':
                result = _merge_measurements_file(archive, entry)
                imported += result['imported']
                replaced += result['replaced']
                row['imported'] += result['imported']
                row['replaced'] += result['replaced']
                continue
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
def project_transfer_catalog(project_ids: str = '', mode: str = 'full') -> dict[str, Any]:
    _cleanup_expired_tokens()
    ids = _normalize_project_ids([item.strip() for item in str(project_ids or '').split(',') if item.strip()])
    export_mode = 'project' if str(mode or '') == 'project' and ids else 'full'
    return {
        'ok': True,
        'mode': export_mode,
        'project_ids': ids,
        'projects': _project_rows(ids),
        'categories': [_category_summary(item, export_mode, ids) for item in _category_definitions().values()],
    }


@router.post('/export/prepare')
def project_transfer_export_prepare(req: ExportPrepareReq) -> dict[str, Any]:
    _cleanup_expired_tokens()
    temp_dir = _new_temp_dir()
    token = uuid4().hex
    export_mode = 'project' if req.mode == 'project' and req.project_ids else 'full'
    zip_prefix = 'loco_project' if export_mode == 'project' else 'loco_training_project'
    zip_name = f"{zip_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = temp_dir / zip_name
    try:
        manifest = _create_export_zip(req.categories, zip_path, mode=export_mode, project_ids=req.project_ids)
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
        'export_mode': manifest.get('export_mode') or 'full',
        'project_ids': list(manifest.get('project_ids') or []),
        'projects': list(manifest.get('projects') or []),
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
        summary['projects'] = list(manifest.get('projects') or [])
        summary['project_ids'] = list(manifest.get('project_ids') or [])
        summary['export_mode'] = str(manifest.get('export_mode') or 'full')
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
        result = _apply_import(zip_path, list(manifest['entries']), bool(req.overwrite), req.project_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        _cleanup_token(token)
    return {'ok': True, 'result': result}
