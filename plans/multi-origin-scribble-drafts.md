# Multi-Origin Scribble Drafts Plan

## Problem

Currently each image has a **single** scribble draft directory (`drafts/<image_id>/`) containing:
- `scribble_map.npz`
- `scribble_preview.png`
- `meta.json` (with `scribble_origin` field)

When the user changes the origin in the table, it just renames the `scribble_origin` field in meta.json — but the actual scribble data is the same. This means:
1. Saving scribbles for one origin overwrites the other
2. Changing origin doesn't load different scribbles
3. No way to keep manual/modelo/modificado drafts separate

## Solution: Per-Origin Draft Storage

Store drafts in subdirectories keyed by origin:

```
drafts/<image_id>/
  meta.json              ← shared metadata (image_id, etc.)
  scribble_map.npz       ← CURRENT single draft (will be migrated)
  scribble_preview.png
  manual/
    scribble_map.npz
    scribble_preview.png
    meta.json
  modelo/
    scribble_map.npz
    scribble_preview.png
    meta.json
  modelo_modificado/
    scribble_map.npz
    scribble_preview.png
    meta.json
```

## Changes

### 1. Backend: [`backend/persistence.py`](backend/persistence.py)

#### 1a. New helper: `_draft_origin_dir(image_id, origin)`
Returns `drafts/<image_id>/<origin>/` for non-empty origins, or `drafts/<image_id>/` for empty/manual (backward compat).

#### 1b. Modify [`save_scribble_draft()`](backend/persistence.py:65)
- Accept `origin` parameter (already does)
- Save to `_draft_origin_dir(image_id, origin)` instead of `_draft_dir(image_id)`
- The `scribble_origin` in meta.json is implicit from the directory
- Keep the root `drafts/<image_id>/meta.json` for backward compat (list_library_images reads it)

#### 1c. Modify [`load_scribble_draft()`](backend/persistence.py:113)
- Accept optional `origin` parameter (default `''` = load from root for backward compat)
- Load from `_draft_origin_dir(image_id, origin)` if origin is provided
- If origin dir doesn't exist, fall back to root draft

#### 1d. Modify [`set_scribble_origin()`](backend/persistence.py:163)
- Rename to `move_scribble_draft()` — physically move files from old origin dir to new origin dir
- If the target origin dir already has data, do NOT overwrite (return conflict error)
- If source origin dir is empty, just create the target with a note

#### 1e. Modify [`clear_scribble_draft()`](backend/persistence.py:179)
- Accept optional `origin` parameter
- Clear only the specified origin's draft, or all if no origin specified

#### 1f. Migration script
- On startup (or first call), scan all `drafts/<image_id>/` directories
- If `scribble_map.npz` exists at root AND `meta.json` has `scribble_origin`, move files to `drafts/<image_id>/<origin>/`
- If `scribble_map.npz` exists at root but no `scribble_origin` in meta, move to `drafts/<image_id>/manual/`

### 2. Backend: [`backend/main.py`](backend/main.py)

#### 2a. Modify [`ScribbleDraftSaveReq`](backend/main.py:140)
- Already has `scribble_origin` field — no change needed

#### 2b. Modify [`api_scribble_draft_save()`](backend/main.py:776)
- Pass `origin` to `save_scribble_draft()` — already done

#### 2c. Modify [`api_scribble_draft_load()`](backend/main.py:803)
- Accept optional `origin` query parameter
- Pass it to `load_scribble_draft()`

#### 2d. Modify [`api_scribble_draft_set_origin()`](backend/main.py:842)
- Change to physically move draft files between origin directories
- Return error if target origin already has a draft (to prevent overwrite)
- Add a `force` parameter to allow overwrite if user confirms

#### 2e. Modify [`api_library_load()`](backend/main.py:578)
- Pass the image's current `scribble_origin` from library meta when loading draft

### 3. Backend: [`backend/library_store.py`](backend/library_store.py)

#### 3a. Modify [`_draft_meta_for()`](backend/library_store.py:118)
- Check all origin subdirectories and return combined info
- Return `scribble_origins_available` as a list of origins that have drafts

#### 3b. Modify [`list_library_images()`](backend/library_store.py:129)
- Add `scribble_origins_available` field (list of origins with drafts)
- Keep `scribble_origin` as the "current" one (most recently saved)

### 4. Backend: [`backend/assist_models.py`](backend/assist_models.py)

#### 4a. Modify [`_dataset_rows()`](backend/assist_models.py:180)
- When loading draft for thumbnail, try the image's `scribble_origin` first, fall back to root
- Expose `scribble_origins_available` in the row

### 5. Frontend: [`frontend/src/App.jsx`](frontend/src/App.jsx)

#### 5a. New state: `availableOrigins`
Track which origins have drafts for the current image (from `scribble_origins_available` in library data).

#### 5b. Modify [`loadScribbleDraftForImage()`](frontend/src/App.jsx:1736)
- Pass `scribbleOrigin` as query parameter to load the correct origin's draft

#### 5c. Modify [`saveScribbleDraft()`](frontend/src/App.jsx:1669)
- Already passes `scribble_origin` — no change needed

#### 5d. Modify [`updateScribbleOrigin()`](frontend/src/App.jsx:1721)
- After changing origin, call `loadScribbleDraftForImage()` with the new origin to load that draft
- Show warning if target origin already has data (conflict)

#### 5e. Modify [`loadSavedImage()`](frontend/src/App.jsx:2896)
- After loading, check `scribble_origins_available` and set `availableOrigins` state
- Load the draft for the current `scribbleOrigin`

#### 5f. Table origin dropdown handler
- When user selects a new origin in the table, call `updateScribbleOrigin()` which now moves files
- Then reload the draft for the new origin
- Show confirmation dialog if target origin already has data

### 6. Frontend: [`frontend/src/styles.css`](frontend/src/styles.css)

- Minor adjustments if needed for any new UI elements

## Data Flow

```
User changes origin in table
  → updateScribbleOrigin(imageId, newOrigin)
    → POST /api/scribble/draft/set-origin { image_id, origin: newOrigin }
      → move_scribble_draft(image_id, newOrigin)
        → moves scribble_map.npz, scribble_preview.png, meta.json
           from drafts/<image_id>/<old_origin>/
           to   drafts/<image_id>/<new_origin>/
    → loadScribbleDraftForImage(imageId)
      → GET /api/scribble/draft/load?image_id=X&origin=newOrigin
        → load_scribble_draft(image_id, origin='newOrigin')
    → refreshModelDataset()
```

## Migration

On first startup after deploy, run a one-time migration:

1. For each `drafts/<image_id>/` that has `scribble_map.npz` at root:
   - Read `meta.json` → get `scribble_origin` (default `'manual'`)
   - Create `drafts/<image_id>/<origin>/` directory
   - Move `scribble_map.npz`, `scribble_preview.png`, `meta.json` into it
   - Write a new root `meta.json` with just `{ "image_id": "...", "migrated": true }`

This preserves all existing data and makes it accessible under the correct origin.

## Migration Implementation Detail

The migration should be implemented as a function `_migrate_legacy_drafts()` called once at module import time in `persistence.py`. It should:

1. Check for a sentinel file `drafts/.migrated_v4` to avoid re-running
2. Iterate all `drafts/<image_id>/` directories
3. If `scribble_map.npz` exists at root level (not in a subdirectory):
   - Read `meta.json` → get `scribble_origin` (default `'manual'`)
   - Create subdirectory `drafts/<image_id>/<origin>/`
   - Move files into it
   - Write root `meta.json` with `{ "image_id": "...", "migrated": true }`
4. Create sentinel file

## API Changes Summary

| Endpoint | Change |
|----------|--------|
| `POST /api/scribble/draft/save` | No change (already passes origin) |
| `GET /api/scribble/draft/load` | Add optional `origin` query param |
| `POST /api/scribble/draft/set-origin` | Now moves files between origin dirs |
| `POST /api/scribble/draft/clear` | Add optional `origin` param |
| `GET /api/library/images` | Add `scribble_origins_available` field |
| `POST /api/library/load` | Pass origin when loading draft |
