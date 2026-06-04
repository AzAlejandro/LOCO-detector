from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .persistence import OUTPUT_ROOT
from .library_store import list_library_images, normalize_structured_tags, update_library_image_tags


PROJECTS_DIR = OUTPUT_ROOT / 'projects'
PROJECTS_PATH = PROJECTS_DIR / 'projects.json'
TAG_OVERRIDES_PATH = PROJECTS_DIR / 'tag_overrides.json'


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _safe_id(text: str) -> str:
    raw = str(text or '').strip().lower()
    out: list[str] = []
    for ch in raw:
        if ch.isalnum() or ch in {'_', '-'}:
            out.append(ch)
        elif ch.isspace() or ch in {'.', '/', '\\'}:
            out.append('_')
    sid = ''.join(out).strip('_')
    return sid[:80]


def _normalize_tags(items: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    raw_items = items if isinstance(items, list) else str(items or '').split(',')
    for item in raw_items:
        tag = str(item or '').strip()
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return out


def _legacy_tags_from_structured(items: Any) -> list[str]:
    out: list[str] = []
    for item in normalize_structured_tags(items):
        out.append(str(item.get('label') or '').strip())
    return _normalize_tags(out)


def _ensure() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    if not PROJECTS_PATH.exists():
        PROJECTS_PATH.write_text(json.dumps({'active_project_id': '', 'projects': []}, ensure_ascii=False, indent=2), encoding='utf-8')
    if not TAG_OVERRIDES_PATH.exists():
        TAG_OVERRIDES_PATH.write_text(json.dumps({'hidden': []}, ensure_ascii=False, indent=2), encoding='utf-8')


def _load_state() -> dict[str, Any]:
    _ensure()
    try:
        payload = dict(json.loads(PROJECTS_PATH.read_text(encoding='utf-8')) or {})
    except Exception:
        payload = {}
    projects = payload.get('projects')
    if not isinstance(projects, list):
        projects = []
    return {
        'active_project_id': str(payload.get('active_project_id') or ''),
        'projects': [dict(item or {}) for item in projects],
    }


def _save_state(payload: dict[str, Any]) -> dict[str, Any]:
    _ensure()
    projects = [dict(item or {}) for item in list(payload.get('projects') or [])]
    active_id = str(payload.get('active_project_id') or '')
    if active_id and not any(str(item.get('project_id') or '') == active_id for item in projects):
        active_id = ''
    out = {'active_project_id': active_id, 'projects': projects}
    PROJECTS_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    return out


def _tag_key(category: str, label: str) -> str:
    return f'{str(category or "other").strip().lower()}::{str(label or "").strip().lower()}'


def _load_tag_overrides() -> dict[str, Any]:
    _ensure()
    try:
        payload = dict(json.loads(TAG_OVERRIDES_PATH.read_text(encoding='utf-8')) or {})
    except Exception:
        payload = {}
    hidden = payload.get('hidden') if isinstance(payload.get('hidden'), list) else []
    return {'hidden': [dict(item or {}) for item in hidden]}


def _save_tag_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    _ensure()
    hidden = [dict(item or {}) for item in list(payload.get('hidden') or [])]
    out = {'hidden': hidden}
    TAG_OVERRIDES_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    return out


def _visible_tag_allowed(category: str, label: str, hidden: list[dict[str, Any]]) -> bool:
    key = _tag_key(category, label)
    return not any(_tag_key(item.get('category', ''), item.get('label', '')) == key for item in hidden)


def list_projects() -> dict[str, Any]:
    state = _load_state()
    projects = sorted(state['projects'], key=lambda item: str(item.get('updated_at') or ''), reverse=True)
    return {'active_project_id': state['active_project_id'], 'projects': projects}


def get_active_project() -> dict[str, Any] | None:
    state = _load_state()
    active_id = str(state.get('active_project_id') or '')
    if not active_id:
        return None
    for item in state.get('projects') or []:
        if str(item.get('project_id') or '') == active_id:
            return dict(item)
    return None


def upsert_project(
    project_id: str = '',
    name: str = '',
    tags: Any = None,
    source_dir: str = '',
    active: bool = False,
    fiber_type: str = '',
    creator: str = '',
    auto_tags_structured: Any = None,
) -> dict[str, Any]:
    state = _load_state()
    projects = list(state.get('projects') or [])
    now = _now()
    clean_name = str(name or '').strip()
    if not clean_name:
        raise ValueError('Nombre de proyecto requerido.')
    pid = str(project_id or '').strip() or _safe_id(clean_name) or f'project_{uuid4().hex[:8]}'
    existing_idx = next((idx for idx, item in enumerate(projects) if str(item.get('project_id') or '') == pid), -1)
    previous = projects[existing_idx] if existing_idx >= 0 else {}
    clean_fiber = str(fiber_type or previous.get('fiber_type') or '').strip()
    clean_creator = str(creator or previous.get('creator') or '').strip()
    structured = [
        {'category': 'project', 'label': clean_name},
        *([{'category': 'fiber_type', 'label': clean_fiber}] if clean_fiber else []),
        *([{'category': 'creator', 'label': clean_creator}] if clean_creator else []),
        *normalize_structured_tags(auto_tags_structured or tags or []),
    ]
    next_structured = normalize_structured_tags(structured)
    next_tags = _normalize_tags([clean_name, *_legacy_tags_from_structured(next_structured), *_normalize_tags(tags)])
    item = {
        'project_id': pid,
        'name': clean_name,
        'tags': next_tags,
        'structured_tags': next_structured,
        'fiber_type': clean_fiber,
        'creator': clean_creator,
        'auto_tags_structured': normalize_structured_tags(auto_tags_structured or []),
        'source_dir': str(source_dir or '').strip(),
        'created_at': str(previous.get('created_at') or now),
        'updated_at': now,
    }
    if existing_idx >= 0:
        projects[existing_idx] = item
    else:
        projects.append(item)
    state['projects'] = projects
    if active or not state.get('active_project_id'):
        state['active_project_id'] = pid
    _save_state(state)
    return item


def set_active_project(project_id: str = '') -> dict[str, Any]:
    state = _load_state()
    pid = str(project_id or '').strip()
    if pid and not any(str(item.get('project_id') or '') == pid for item in state.get('projects') or []):
        raise ValueError('Proyecto no encontrado.')
    state['active_project_id'] = pid
    return _save_state(state)


def delete_project(project_id: str) -> dict[str, Any]:
    state = _load_state()
    pid = str(project_id or '').strip()
    if not pid:
        raise ValueError('project_id requerido.')
    state['projects'] = [item for item in state.get('projects') or [] if str(item.get('project_id') or '') != pid]
    if str(state.get('active_project_id') or '') == pid:
        state['active_project_id'] = ''
    saved = _save_state(state)
    for image in list_library_images():
        image_id = str(image.get('image_id') or '')
        project_ids = [item for item in _normalize_tags(image.get('project_ids') or []) if item != pid]
        if len(project_ids) != len(_normalize_tags(image.get('project_ids') or [])):
            update_library_image_tags(
                image_id,
                image.get('tags') or [],
                project_ids,
                structured_tags=image.get('structured_tags') or image.get('tags') or [],
            )
    return saved


def project_tags_for_image(project: dict[str, Any] | None) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    if not project:
        return [], [], []
    pid = str(project.get('project_id') or '').strip()
    tags = _normalize_tags(project.get('tags') or [])
    structured = normalize_structured_tags(project.get('structured_tags') or tags)
    return tags, ([pid] if pid else []), structured


def tag_catalog() -> dict[str, list[str]]:
    values: dict[str, set[str]] = {
        'project': set(),
        'fiber_type': set(),
        'creator': set(),
        'unit': {'um', 'nm'},
        'other': set(),
    }
    state = _load_state()
    hidden = _load_tag_overrides().get('hidden') or []
    for project in state.get('projects') or []:
        if project.get('name'):
            label = str(project.get('name'))
            if _visible_tag_allowed('project', label, hidden):
                values['project'].add(label)
        if project.get('fiber_type'):
            label = str(project.get('fiber_type'))
            if _visible_tag_allowed('fiber_type', label, hidden):
                values['fiber_type'].add(label)
        if project.get('creator'):
            label = str(project.get('creator'))
            if _visible_tag_allowed('creator', label, hidden):
                values['creator'].add(label)
        for tag in normalize_structured_tags(project.get('structured_tags') or project.get('tags') or []):
            category = str(tag.get('category') or 'other')
            if category == 'size' and tag.get('unit'):
                label = str(tag.get('unit'))
                if _visible_tag_allowed('unit', label, hidden):
                    values['unit'].add(label)
            elif category in values and tag.get('label'):
                label = str(tag.get('label'))
                if _visible_tag_allowed(category, label, hidden):
                    values[category].add(label)
    for image in list_library_images():
        for tag in normalize_structured_tags(image.get('structured_tags') or image.get('tags') or []):
            category = str(tag.get('category') or 'other')
            if category == 'size' and tag.get('unit'):
                label = str(tag.get('unit'))
                if _visible_tag_allowed('unit', label, hidden):
                    values['unit'].add(label)
            elif category in values and tag.get('label'):
                label = str(tag.get('label'))
                if _visible_tag_allowed(category, label, hidden):
                    values[category].add(label)
    return {key: sorted(list(items), key=str.lower) for key, items in values.items()}


def tag_catalog_detailed() -> dict[str, Any]:
    catalog = tag_catalog()
    hidden = _load_tag_overrides().get('hidden') or []
    rows: list[dict[str, Any]] = []
    categories = ['project', 'fiber_type', 'creator', 'size', 'unit', 'other']
    for category in categories:
        for label in catalog.get(category, []):
            rows.append({'category': category, 'label': label, 'hidden': False})
    for item in hidden:
        category = str(item.get('category') or 'other')
        label = str(item.get('label') or '').strip()
        if label:
            rows.append({'category': category, 'label': label, 'hidden': True})
    rows.sort(key=lambda item: (str(item.get('category') or ''), str(item.get('label') or '').lower()))
    return {'catalog': catalog, 'items': rows, 'hidden': hidden}


def set_tag_hidden(category: str, label: str, hidden: bool = True) -> dict[str, Any]:
    payload = _load_tag_overrides()
    rows = [item for item in payload.get('hidden') or [] if _tag_key(item.get('category', ''), item.get('label', '')) != _tag_key(category, label)]
    if hidden and str(label or '').strip():
        rows.append({'category': str(category or 'other'), 'label': str(label or '').strip(), 'hidden_at': _now()})
    return _save_tag_overrides({'hidden': rows})


def _rewrite_tag_items(items: Any, category: str, old_label: str, new_label: str = '', remove: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    target_key = _tag_key(category, old_label)
    for tag in normalize_structured_tags(items):
        tag_category = str(tag.get('category') or 'other')
        label = str(tag.get('label') or '')
        if tag_category == 'size' and str(category) == 'unit':
            unit = str(tag.get('unit') or '')
            if _tag_key('unit', unit) == target_key:
                if remove:
                    tag.pop('unit', None)
                else:
                    tag['unit'] = str(new_label or '').strip()
                    if tag.get('value'):
                        tag['label'] = f"Tamaño: {tag.get('value')}{' ' + tag['unit'] if tag.get('unit') else ''}"
        elif _tag_key(tag_category, label) == target_key:
            if remove:
                continue
            tag['label'] = str(new_label or '').strip()
        out.append(tag)
    return normalize_structured_tags(out)


def rename_tag(category: str, old_label: str, new_label: str) -> dict[str, Any]:
    clean_new = str(new_label or '').strip()
    if not clean_new:
        raise ValueError('Nuevo nombre requerido.')
    state = _load_state()
    changed_projects: list[dict[str, Any]] = []
    for project in state.get('projects') or []:
        if category == 'project' and str(project.get('name') or '').strip().lower() == str(old_label or '').strip().lower():
            project['name'] = clean_new
        if category == 'fiber_type' and str(project.get('fiber_type') or '').strip().lower() == str(old_label or '').strip().lower():
            project['fiber_type'] = clean_new
        if category == 'creator' and str(project.get('creator') or '').strip().lower() == str(old_label or '').strip().lower():
            project['creator'] = clean_new
        project['structured_tags'] = _rewrite_tag_items(project.get('structured_tags') or project.get('tags') or [], category, old_label, clean_new)
        project['tags'] = _normalize_tags(_legacy_tags_from_structured(project.get('structured_tags') or []))
        project['updated_at'] = _now()
        changed_projects.append(project)
    state['projects'] = changed_projects
    _save_state(state)
    for image in list_library_images():
        image_id = str(image.get('image_id') or '')
        if not image_id:
            continue
        structured = _rewrite_tag_items(image.get('structured_tags') or image.get('tags') or [], category, old_label, clean_new)
        update_library_image_tags(image_id, _legacy_tags_from_structured(structured), image.get('project_ids') or [], structured_tags=structured)
    set_tag_hidden(category, old_label, False)
    return tag_catalog_detailed()


def delete_tag(category: str, label: str) -> dict[str, Any]:
    state = _load_state()
    next_projects: list[dict[str, Any]] = []
    for project in state.get('projects') or []:
        if category == 'fiber_type' and str(project.get('fiber_type') or '').strip().lower() == str(label or '').strip().lower():
            project['fiber_type'] = ''
        if category == 'creator' and str(project.get('creator') or '').strip().lower() == str(label or '').strip().lower():
            project['creator'] = ''
        if category != 'project':
            project['structured_tags'] = _rewrite_tag_items(project.get('structured_tags') or project.get('tags') or [], category, label, remove=True)
            project['tags'] = _normalize_tags(_legacy_tags_from_structured(project.get('structured_tags') or []))
            project['updated_at'] = _now()
        next_projects.append(project)
    state['projects'] = next_projects
    _save_state(state)
    if category != 'project':
        for image in list_library_images():
            image_id = str(image.get('image_id') or '')
            if not image_id:
                continue
            structured = _rewrite_tag_items(image.get('structured_tags') or image.get('tags') or [], category, label, remove=True)
            update_library_image_tags(image_id, _legacy_tags_from_structured(structured), image.get('project_ids') or [], structured_tags=structured)
    set_tag_hidden(category, label, True)
    return tag_catalog_detailed()
