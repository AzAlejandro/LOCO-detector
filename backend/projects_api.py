from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .library_store import list_library_images, update_library_image_tags
from .projects_store import delete_project, delete_tag, list_projects, rename_tag, set_active_project, set_tag_hidden, tag_catalog, tag_catalog_detailed, upsert_project


router = APIRouter(prefix='/api/projects', tags=['projects'])


class ProjectUpsertReq(BaseModel):
    project_id: str = ''
    name: str
    tags: list[str] = Field(default_factory=list)
    fiber_type: str = ''
    creator: str = ''
    auto_tags_structured: list[dict[str, Any]] = Field(default_factory=list)
    source_dir: str = ''
    active: bool = False


class ProjectActivateReq(BaseModel):
    project_id: str = ''


class ProjectDeleteReq(BaseModel):
    project_id: str


class ProjectImageUpdateReq(BaseModel):
    image_id: str
    tags: list[str] = Field(default_factory=list)
    structured_tags: list[dict[str, Any]] = Field(default_factory=list)
    project_ids: list[str] = Field(default_factory=list)


class ProjectTagVisibilityReq(BaseModel):
    category: str
    label: str
    hidden: bool = True


class ProjectTagRenameReq(BaseModel):
    category: str
    old_label: str
    new_label: str


class ProjectTagDeleteReq(BaseModel):
    category: str
    label: str


def _payload() -> dict[str, Any]:
    state = list_projects()
    active_id = str(state.get('active_project_id') or '')
    active = next((item for item in state.get('projects') or [] if str(item.get('project_id') or '') == active_id), None)
    return {
        'projects': list(state.get('projects') or []),
        'active_project_id': active_id,
        'active_project': active,
    }


@router.get('/list')
def api_projects_list() -> dict[str, Any]:
    return {'ok': True, 'payload': _payload()}


@router.post('/upsert')
def api_projects_upsert(req: ProjectUpsertReq) -> dict[str, Any]:
    try:
        project = upsert_project(
            project_id=req.project_id,
            name=req.name,
            tags=req.tags,
            fiber_type=req.fiber_type,
            creator=req.creator,
            auto_tags_structured=req.auto_tags_structured,
            source_dir=req.source_dir,
            active=req.active,
        )
        return {'ok': True, 'payload': {**_payload(), 'project': project}}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post('/activate')
def api_projects_activate(req: ProjectActivateReq) -> dict[str, Any]:
    try:
        state = set_active_project(req.project_id)
        active_id = str(state.get('active_project_id') or '')
        active = next((item for item in state.get('projects') or [] if str(item.get('project_id') or '') == active_id), None)
        return {'ok': True, 'payload': {'active_project_id': active_id, 'active_project': active, 'projects': state.get('projects') or []}}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post('/delete')
def api_projects_delete(req: ProjectDeleteReq) -> dict[str, Any]:
    try:
        delete_project(req.project_id)
        return {'ok': True, 'payload': _payload()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get('/images')
def api_projects_images() -> dict[str, Any]:
    return {'ok': True, 'payload': {'items': list_library_images(), **_payload()}}


@router.get('/tag-catalog')
def api_projects_tag_catalog() -> dict[str, Any]:
    return {'ok': True, 'payload': tag_catalog_detailed()}


@router.post('/tag/visibility')
def api_projects_tag_visibility(req: ProjectTagVisibilityReq) -> dict[str, Any]:
    set_tag_hidden(req.category, req.label, req.hidden)
    return {'ok': True, 'payload': tag_catalog_detailed()}


@router.post('/tag/rename')
def api_projects_tag_rename(req: ProjectTagRenameReq) -> dict[str, Any]:
    try:
        return {'ok': True, 'payload': rename_tag(req.category, req.old_label, req.new_label)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post('/tag/delete')
def api_projects_tag_delete(req: ProjectTagDeleteReq) -> dict[str, Any]:
    return {'ok': True, 'payload': delete_tag(req.category, req.label)}


@router.post('/image/update')
def api_projects_image_update(req: ProjectImageUpdateReq) -> dict[str, Any]:
    image_id = str(req.image_id or '').strip()
    if not image_id:
        raise HTTPException(status_code=400, detail='image_id requerido.')
    try:
        meta = update_library_image_tags(image_id, req.tags, req.project_ids, structured_tags=req.structured_tags)
        return {'ok': True, 'payload': {'image': meta, 'items': list_library_images(), **_payload()}}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
