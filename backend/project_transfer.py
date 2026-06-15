from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import threading
import time
from typing import Any, Literal
from uuid import uuid4
import zipfile
import zlib

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
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


SCRIBBLE_RUN_REQUIRED_FILES = {'meta.json', 'scribble_map.npz', 'mask.png', 'overlay.png', 'input_image.png'}


class ExportPrepareReq(BaseModel):
    categories: list[str] = Field(default_factory=list)
    project_ids: list[str] = Field(default_factory=list)
    project_category_selection: dict[str, list[str]] = Field(default_factory=dict)
    mode: Literal['full', 'project'] = 'full'


class ImportApplyReq(BaseModel):
    token: str
    overwrite: bool = False
    project_ids: list[str] = Field(default_factory=list)


_tokens: dict[str, dict[str, Any]] = {}
_import_jobs: dict[str, dict[str, Any]] = {}
_import_jobs_lock = threading.Lock()
IMPORT_JOB_TTL_SECONDS = 6 * 60 * 60


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


def _scribble_run_meta(run_dir: Path) -> dict[str, Any]:
    meta = _read_json_file(run_dir / 'meta.json', {})
    return meta if isinstance(meta, dict) else {}


def _scribble_run_image_id(run_dir: Path) -> str:
    return str(_scribble_run_meta(run_dir).get('image_id') or '').strip()


def _scribble_run_is_complete(run_dir: Path) -> bool:
    return run_dir.is_dir() and all((run_dir / name).is_file() for name in SCRIBBLE_RUN_REQUIRED_FILES)


def _complete_scribble_run_ids_for_images(image_ids: set[str]) -> set[str]:
    runs_root = scribble_persistence.OUTPUT_ROOT / 'runs'
    out: set[str] = set()
    if not image_ids or not runs_root.exists():
        return out
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        if _scribble_run_image_id(run_dir) in image_ids and _scribble_run_is_complete(run_dir):
            out.add(run_dir.name)
    return out


def _skipped_incomplete_scribble_runs(project_ids: list[str]) -> list[dict[str, str]]:
    image_ids = _selected_image_ids(project_ids)
    runs_root = scribble_persistence.OUTPUT_ROOT / 'runs'
    if not image_ids or not runs_root.exists():
        return []
    skipped: list[dict[str, str]] = []
    for run_dir in sorted((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        image_id = _scribble_run_image_id(run_dir)
        if image_id not in image_ids or _scribble_run_is_complete(run_dir):
            continue
        missing = [name for name in sorted(SCRIBBLE_RUN_REQUIRED_FILES) if not (run_dir / name).is_file()]
        skipped.append({'run_id': run_dir.name, 'image_id': image_id, 'missing': ','.join(missing)})
    return skipped


def _filtered_scribble_index(path: Path, complete_run_ids: set[str]) -> bytes | None:
    payload = _read_json_file(path, None)
    if not isinstance(payload, dict):
        return None
    runs = [dict(item or {}) for item in (payload.get('runs') or []) if str((item or {}).get('run_id') or '') in complete_run_ids]
    if not runs:
        return None
    out = dict(payload)
    out['runs'] = runs
    return json.dumps(out, ensure_ascii=False, indent=2).encode('utf-8')


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
    with _import_jobs_lock:
        for job_id, job in list(_import_jobs.items()):
            if now - float(job.get('created_ts') or 0.0) > IMPORT_JOB_TTL_SECONDS:
                _import_jobs.pop(job_id, None)


def _category_label(category_key: str) -> str:
    definition = _category_definitions().get(str(category_key or ''))
    return definition.label if definition else str(category_key or '')


def _progress_category_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = str(entry.get('category') or '')
        row = rows.setdefault(key, {
            'key': key,
            'label': _category_label(key),
            'status': 'pending',
            'total_files': 0,
            'processed_files': 0,
            'imported': 0,
            'skipped': 0,
            'replaced': 0,
        })
        row['total_files'] += 1
    return list(rows.values())


def _set_import_job(job_id: str, **updates: Any) -> None:
    with _import_jobs_lock:
        job = _import_jobs.get(job_id)
        if not job:
            return
        job.update(updates)
        job['updated_ts'] = time.time()


def _update_import_job_category(job_id: str, category_key: str, **updates: Any) -> None:
    with _import_jobs_lock:
        job = _import_jobs.get(job_id)
        if not job:
            return
        categories = list(job.get('categories') or [])
        for row in categories:
            if str(row.get('key') or '') == str(category_key or ''):
                row.update(updates)
                break
        job['categories'] = categories
        job['updated_ts'] = time.time()


def _snapshot_import_job(job_id: str) -> dict[str, Any] | None:
    with _import_jobs_lock:
        job = _import_jobs.get(job_id)
        return json.loads(json.dumps(job, ensure_ascii=False)) if job else None


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

    if category_key == 'scribble_experiments' and source.key == 'runs':
        for run_dir in sorted((p for p in source.root.iterdir() if p.is_dir()), key=lambda p: p.name) if source.root.exists() else []:
            image_id = _scribble_run_image_id(run_dir)
            if image_id not in images or not _scribble_run_is_complete(run_dir):
                continue
            project_tuple = _entry_project_ids_from_image_ids({image_id}, image_projects, selected)
            for path, relative in _iter_source_files(TransferSource(source.key, run_dir)):
                rows.append(ExportEntry(path=path, relative_path=PurePosixPath(run_dir.name, relative).as_posix(), project_ids=project_tuple))
        return rows

    if category_key == 'scribble_experiments' and source.key == 'index':
        complete_run_ids = _complete_scribble_run_ids_for_images(images)
        for image_id in sorted(images):
            path = source.root / f'{image_id}.json'
            data = _filtered_scribble_index(path, complete_run_ids)
            if data:
                rows.append(ExportEntry(path=None, relative_path=f'{image_id}.json', data=data, project_ids=_entry_project_ids_from_image_ids({image_id}, image_projects, selected)))
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


def _normalize_project_category_selection(raw: Any) -> dict[str, list[str]]:
    if not isinstance(raw, dict):
        return {}
    definitions = _category_definitions()
    valid_projects = {str(item.get('project_id') or '') for item in _project_state().get('projects') or []}
    normalized: dict[str, list[str]] = {}
    for project_id, keys in raw.items():
        pid = str(project_id or '').strip()
        if not pid or (valid_projects and pid not in valid_projects) or not isinstance(keys, list):
            continue
        seen: set[str] = set()
        selected: list[str] = []
        for key in keys:
            category_key = str(key or '').strip()
            if category_key in definitions and category_key not in seen:
                seen.add(category_key)
                selected.append(category_key)
        if selected:
            normalized[pid] = selected
    return normalized


def _project_ids_by_category(project_category_selection: dict[str, list[str]]) -> dict[str, list[str]]:
    by_category: dict[str, list[str]] = {}
    for project_id, categories in project_category_selection.items():
        for category in categories:
            by_category.setdefault(category, [])
            if project_id not in by_category[category]:
                by_category[category].append(project_id)
    return by_category


def _zip_archive_path(category: str, source: str, relative_path: str) -> str:
    relative = _safe_relative_path(relative_path)
    return PurePosixPath('payload', category, source, *relative.parts).as_posix()


def _create_export_zip(
    categories: list[str],
    zip_path: Path,
    *,
    mode: str = 'full',
    project_ids: list[str] | None = None,
    project_category_selection: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    per_project_selection = _normalize_project_category_selection(project_category_selection or {})
    selection_by_category = _project_ids_by_category(per_project_selection)
    if per_project_selection and 'projects' not in selection_by_category:
        selection_by_category['projects'] = list(per_project_selection.keys())
    selected = _selected_categories(list(selection_by_category.keys()) if per_project_selection else categories)
    export_mode = 'project' if mode == 'project' and project_ids else 'full'
    selected_project_ids = _normalize_project_ids(project_ids or []) if export_mode == 'project' else []
    if per_project_selection:
        export_mode = 'project'
        selected_project_ids = _normalize_project_ids(list(per_project_selection.keys()))
    skipped_incomplete_runs = _skipped_incomplete_scribble_runs(selected_project_ids) if export_mode == 'project' else []
    entries: list[dict[str, Any]] = []
    written: dict[str, dict[str, Any]] = {}
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for category in selected:
            category_project_ids = _normalize_project_ids(selection_by_category.get(category.key, selected_project_ids)) if per_project_selection else selected_project_ids
            category_mode = 'project' if export_mode == 'project' and category_project_ids else 'full'
            for source in category.sources:
                source_entries = _export_entries_for_source(
                    category.key,
                    source,
                    mode=category_mode,
                    project_ids=category_project_ids,
                    selected_images=_selected_image_ids(category_project_ids),
                    image_project_map=_library_image_project_map(),
                )
                for entry in source_entries:
                    archive_path = _zip_archive_path(category.key, source.key, entry.relative_path)
                    if archive_path in written:
                        current_projects = set(written[archive_path].get('project_ids') or [])
                        current_projects.update(entry.project_ids or category_project_ids)
                        written[archive_path]['project_ids'] = sorted(current_projects)
                        continue
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
                            'project_ids': list(entry.project_ids or category_project_ids),
                        }
                    )
                    written[archive_path] = entries[-1]
        manifest = {
            'format_version': FORMAT_VERSION,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'export_mode': export_mode,
            'export_selection_mode': 'per_project' if per_project_selection else 'global',
            'project_ids': selected_project_ids,
            'projects': _project_rows(selected_project_ids),
            'project_category_selection': per_project_selection,
            'project_filter_policy': 'include_if_touches' if export_mode == 'project' else '',
            'skipped_incomplete_runs': skipped_incomplete_runs,
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


def _repair_scribble_run_indices() -> dict[str, int]:
    index_root = scribble_persistence.OUTPUT_ROOT / 'index'
    runs_root = scribble_persistence.OUTPUT_ROOT / 'runs'
    checked = 0
    repaired = 0
    removed = 0
    if not index_root.exists():
        return {'checked': checked, 'repaired': repaired, 'removed_runs': removed}
    for path in sorted(index_root.glob('*.json')):
        payload = _read_json_file(path, None)
        if not isinstance(payload, dict):
            continue
        checked += 1
        original = [dict(item or {}) for item in (payload.get('runs') or []) if isinstance(item, dict)]
        keep: list[dict[str, Any]] = []
        for row in original:
            run_id = str(row.get('run_id') or '').strip()
            if run_id and _scribble_run_is_complete(runs_root / run_id):
                keep.append(row)
        if len(keep) == len(original):
            continue
        removed += len(original) - len(keep)
        repaired += 1
        if keep:
            out = dict(payload)
            out['runs'] = keep
            path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
        else:
            path.unlink(missing_ok=True)
    return {'checked': checked, 'repaired': repaired, 'removed_runs': removed}


def _apply_import(
    zip_path: Path,
    entries: list[dict[str, Any]],
    overwrite: bool,
    project_ids: list[str] | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    imported = 0
    skipped = 0
    replaced = 0
    per_category: dict[str, dict[str, int]] = {}
    selected_project_ids = [str(pid or '').strip() for pid in (project_ids or []) if str(pid or '').strip()]
    entries = _filter_entries_by_projects(entries, selected_project_ids)
    if progress:
        progress({
            'event': 'start',
            'total_files': len(entries),
            'categories': _progress_category_rows(entries),
        })
    with zipfile.ZipFile(zip_path, 'r') as archive:
        for entry in entries:
            category = str(entry['category'])
            row = per_category.setdefault(category, {'imported': 0, 'skipped': 0, 'replaced': 0})
            target = _safe_destination(category, str(entry['source']), str(entry['path']))
            if progress:
                progress({
                    'event': 'entry_start',
                    'category': category,
                    'file': str(entry.get('path') or entry.get('archive_path') or ''),
                    'summary': {'imported': imported, 'skipped': skipped, 'replaced': replaced},
                })
            if category == 'projects' and str(entry['source']) == 'projects' and str(entry['path']) == 'projects.json':
                result = _merge_projects_file(zip_path, archive, entry, selected_project_ids)
                imported += result['imported']
                replaced += result['replaced']
                row['imported'] += result['imported']
                row['replaced'] += result['replaced']
                if progress:
                    progress({
                        'event': 'entry_done',
                        'category': category,
                        'file': str(entry.get('path') or entry.get('archive_path') or ''),
                        'imported_delta': result['imported'],
                        'skipped_delta': result.get('skipped', 0),
                        'replaced_delta': result['replaced'],
                        'summary': {'imported': imported, 'skipped': skipped, 'replaced': replaced},
                    })
                continue
            if category == 'diameter_analysis' and str(entry['source']) == 'diameter_analysis' and str(entry['path']) == 'measurements.json':
                result = _merge_measurements_file(archive, entry)
                imported += result['imported']
                replaced += result['replaced']
                row['imported'] += result['imported']
                row['replaced'] += result['replaced']
                if progress:
                    progress({
                        'event': 'entry_done',
                        'category': category,
                        'file': str(entry.get('path') or entry.get('archive_path') or ''),
                        'imported_delta': result['imported'],
                        'skipped_delta': result.get('skipped', 0),
                        'replaced_delta': result['replaced'],
                        'summary': {'imported': imported, 'skipped': skipped, 'replaced': replaced},
                    })
                continue
            exists = target.exists()
            if exists and not overwrite:
                skipped += 1
                row['skipped'] += 1
                if progress:
                    progress({
                        'event': 'entry_done',
                        'category': category,
                        'file': str(entry.get('path') or entry.get('archive_path') or ''),
                        'imported_delta': 0,
                        'skipped_delta': 1,
                        'replaced_delta': 0,
                        'summary': {'imported': imported, 'skipped': skipped, 'replaced': replaced},
                    })
                continue
            _copy_member_atomic(archive, entry, target)
            imported += 1
            row['imported'] += 1
            replaced_delta = 0
            if exists:
                replaced += 1
                row['replaced'] += 1
                replaced_delta = 1
            if progress:
                progress({
                    'event': 'entry_done',
                    'category': category,
                    'file': str(entry.get('path') or entry.get('archive_path') or ''),
                    'imported_delta': 1,
                    'skipped_delta': 0,
                    'replaced_delta': replaced_delta,
                    'summary': {'imported': imported, 'skipped': skipped, 'replaced': replaced},
                })
    if progress:
        progress({'event': 'repair_start', 'label': 'Actualizando indices'})
    repair = _repair_scribble_run_indices()
    if progress:
        progress({'event': 'repair_done', 'label': 'Actualizando indices', 'repair': repair})
    return {'imported_count': imported, 'skipped_count': skipped, 'replaced_count': replaced, 'categories': per_category, 'scribble_index_repair': repair}


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


def _run_import_job(job_id: str, token: str, overwrite: bool, project_ids: list[str]) -> None:
    def emit(event: dict[str, Any]) -> None:
        kind = str(event.get('event') or '')
        if kind == 'start':
            _set_import_job(
                job_id,
                status='running',
                total_files=int(event.get('total_files') or 0),
                processed_files=0,
                categories=list(event.get('categories') or []),
                current_category='',
                current_file='',
                imported_count=0,
                skipped_count=0,
                replaced_count=0,
            )
            return
        if kind == 'entry_start':
            category = str(event.get('category') or '')
            _set_import_job(
                job_id,
                status='running',
                current_category=_category_label(category),
                current_file=str(event.get('file') or ''),
            )
            _update_import_job_category(job_id, category, status='running')
            return
        if kind == 'entry_done':
            category = str(event.get('category') or '')
            with _import_jobs_lock:
                job = _import_jobs.get(job_id)
                if not job:
                    return
                job['processed_files'] = int(job.get('processed_files') or 0) + 1
                summary = dict(event.get('summary') or {})
                job['imported_count'] = int(summary.get('imported') or job.get('imported_count') or 0)
                job['skipped_count'] = int(summary.get('skipped') or job.get('skipped_count') or 0)
                job['replaced_count'] = int(summary.get('replaced') or job.get('replaced_count') or 0)
                categories = list(job.get('categories') or [])
                for row in categories:
                    if str(row.get('key') or '') != category:
                        continue
                    row['processed_files'] = int(row.get('processed_files') or 0) + 1
                    row['imported'] = int(row.get('imported') or 0) + int(event.get('imported_delta') or 0)
                    row['skipped'] = int(row.get('skipped') or 0) + int(event.get('skipped_delta') or 0)
                    row['replaced'] = int(row.get('replaced') or 0) + int(event.get('replaced_delta') or 0)
                    row['status'] = 'completed' if int(row.get('processed_files') or 0) >= int(row.get('total_files') or 0) else 'running'
                    break
                job['categories'] = categories
                job['updated_ts'] = time.time()
            return
        if kind == 'repair_start':
            _set_import_job(job_id, current_category=str(event.get('label') or 'Actualizando indices'), current_file='')
            return
        if kind == 'repair_done':
            _set_import_job(job_id, scribble_index_repair=dict(event.get('repair') or {}))

    _set_import_job(job_id, status='running')
    try:
        item = _tokens.get(str(token or ''))
        if not item or item.get('kind') != 'import':
            raise ValueError('Importacion no encontrada o expirada.')
        zip_path = Path(item['zip_path'])
        manifest = _validate_import_zip(zip_path)
        result = _apply_import(zip_path, list(manifest['entries']), bool(overwrite), project_ids, progress=emit)
        _set_import_job(
            job_id,
            status='completed',
            current_category='',
            current_file='',
            result=result,
            imported_count=int(result.get('imported_count') or 0),
            skipped_count=int(result.get('skipped_count') or 0),
            replaced_count=int(result.get('replaced_count') or 0),
        )
    except Exception as exc:
        _set_import_job(job_id, status='error', error=str(exc))
    finally:
        _cleanup_token(token)


@router.get('/catalog')
def project_transfer_catalog(project_ids: str = '', mode: str = 'full') -> dict[str, Any]:
    _cleanup_expired_tokens()
    ids = _normalize_project_ids([item.strip() for item in str(project_ids or '').split(',') if item.strip()])
    export_mode = 'project' if str(mode or '') == 'project' and ids else 'full'
    all_projects = _project_state().get('projects') or []
    project_catalog_ids = ids if ids else [str(item.get('project_id') or '') for item in all_projects if str(item.get('project_id') or '')]
    return {
        'ok': True,
        'mode': export_mode,
        'project_ids': ids,
        'projects': _project_rows(ids),
        'categories': [_category_summary(item, export_mode, ids) for item in _category_definitions().values()],
        'project_catalogs': [
            {
                'project_id': pid,
                'project': (_project_rows([pid]) or [{}])[0],
                'categories': [_category_summary(item, 'project', [pid]) for item in _category_definitions().values()],
            }
            for pid in project_catalog_ids
        ] if str(mode or '') == 'project' else [],
    }


@router.post('/export/prepare')
def project_transfer_export_prepare(req: ExportPrepareReq) -> dict[str, Any]:
    _cleanup_expired_tokens()
    temp_dir = _new_temp_dir()
    token = uuid4().hex
    project_category_selection = _normalize_project_category_selection(req.project_category_selection)
    selected_project_ids = list(project_category_selection.keys()) if project_category_selection else req.project_ids
    export_mode = 'project' if req.mode == 'project' and selected_project_ids else 'full'
    zip_prefix = 'loco_project' if export_mode == 'project' else 'loco_training_project'
    zip_name = f"{zip_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = temp_dir / zip_name
    try:
        manifest = _create_export_zip(
            req.categories,
            zip_path,
            mode=export_mode,
            project_ids=selected_project_ids,
            project_category_selection=project_category_selection,
        )
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
        'export_selection_mode': manifest.get('export_selection_mode') or 'global',
        'project_ids': list(manifest.get('project_ids') or []),
        'projects': list(manifest.get('projects') or []),
        'project_category_selection': dict(manifest.get('project_category_selection') or {}),
        'skipped_incomplete_runs': list(manifest.get('skipped_incomplete_runs') or []),
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
        summary['export_selection_mode'] = str(manifest.get('export_selection_mode') or 'global')
        summary['project_category_selection'] = dict(manifest.get('project_category_selection') or {})
    except ValueError as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    token = uuid4().hex
    _tokens[token] = {'kind': 'import', 'temp_dir': str(temp_dir), 'zip_path': str(zip_path), 'manifest': manifest, 'created_ts': time.time()}
    return {'ok': True, 'token': token, 'summary': summary}


@router.post('/import/start')
def project_transfer_import_start(req: ImportApplyReq, background_tasks: BackgroundTasks) -> dict[str, Any]:
    _cleanup_expired_tokens()
    token = str(req.token or '')
    item = _tokens.get(token)
    if not item or item.get('kind') != 'import':
        raise HTTPException(status_code=404, detail='Importacion no encontrada o expirada.')
    zip_path = Path(item['zip_path'])
    try:
        manifest = _validate_import_zip(zip_path)
        selected_project_ids = [str(pid or '').strip() for pid in (req.project_ids or []) if str(pid or '').strip()]
        entries = _filter_entries_by_projects(list(manifest['entries']), selected_project_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_id = uuid4().hex
    now = time.time()
    with _import_jobs_lock:
        _import_jobs[job_id] = {
            'job_id': job_id,
            'status': 'queued',
            'created_ts': now,
            'updated_ts': now,
            'total_files': len(entries),
            'processed_files': 0,
            'current_category': '',
            'current_file': '',
            'categories': _progress_category_rows(entries),
            'imported_count': 0,
            'skipped_count': 0,
            'replaced_count': 0,
            'result': None,
            'error': '',
        }
    background_tasks.add_task(_run_import_job, job_id, token, bool(req.overwrite), req.project_ids)
    return {'ok': True, 'job_id': job_id, 'progress': _snapshot_import_job(job_id)}


@router.get('/import/progress')
def project_transfer_import_progress(job_id: str) -> dict[str, Any]:
    _cleanup_expired_tokens()
    job = _snapshot_import_job(str(job_id or ''))
    if not job:
        raise HTTPException(status_code=404, detail='Trabajo de importacion no encontrado o expirado.')
    return {'ok': True, 'progress': job}


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
