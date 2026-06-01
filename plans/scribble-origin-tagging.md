# Scribble Origin Tagging Plan

## Goal

Add a "scribble origin" tag to each image's scribble metadata that tracks how the scribbles were created:
- **`manual`** — scribbles were drawn manually by the user (no model involved)
- **`modelo`** — scribbles were generated entirely by model prediction ("Aplicar")
- **`modelo_modificado`** — scribbles started from model prediction but were later modified manually

This tag will be visible in the "Modelos de Asistencia" table and filterable.

---

## Architecture

```
Frontend (App.jsx)                    Backend
┌─────────────────────┐              ┌──────────────────────────────┐
│                     │              │                              │
│  saveScribbleDraft()│── POST ──▶   │  api_scribble_draft_save()   │
│  (with origin)      │   origin    │  → save_scribble_draft()      │
│                     │              │    → meta.json {origin}      │
│  applyModelPrediction│             │                              │
│  → sets origin='modelo'           │  list_library_images()        │
│                     │              │  → exposes scribble_origin    │
│  onScribbleEdit()   │              │                              │
│  → sets origin=     │              │  _dataset_rows()             │
│    'modelo_modif'   │              │  → exposes scribble_origin    │
│    or 'manual'      │              │                              │
└─────────────────────┘              └──────────────────────────────┘
```

## Data Flow

### 1. Backend: `scribble_origin` in meta.json

**File: [`backend/persistence.py`](backend/persistence.py:65)** — `save_scribble_draft()`

Add an optional `origin` parameter. When provided, store it in the draft's `meta.json` under key `scribble_origin`. When not provided, keep existing value or default to `'manual'`.

```python
def save_scribble_draft(image_id: str, labels: np.ndarray, origin: str = '') -> dict[str, Any]:
    ...
    meta = {
        ...
        'scribble_origin': str(origin or meta.get('scribble_origin', 'manual')),
    }
```

### 2. Backend: Accept `origin` in API

**File: [`backend/main.py`](backend/main.py:139)** — `ScribbleDraftSaveReq`

Add optional field:
```python
class ScribbleDraftSaveReq(BaseModel):
    session_id: str
    image_id: str
    scribble_map_b64: str = ''
    scribble_origin: str = ''  # 'manual' | 'modelo' | 'modelo_modificado' | ''
```

**File: [`backend/main.py`](backend/main.py:768)** — `api_scribble_draft_save()`

Pass `req.scribble_origin` to `save_scribble_draft()`:
```python
meta = save_scribble_draft(image_id, labels, origin=req.scribble_origin)
```

### 3. Backend: Expose `scribble_origin` in library listing

**File: [`backend/library_store.py`](backend/library_store.py:126)** — `list_library_images()`

The `_draft_meta_for()` already reads the full meta.json. Add `scribble_origin` to the returned item:
```python
draft = _draft_meta_for(image_id)
...
items.append({
    ...
    'scribble_origin': str(draft.get('scribble_origin', '')),
})
```

### 4. Backend: Expose in `_dataset_rows()`

**File: [`backend/assist_models.py`](backend/assist_models.py:180)** — `_dataset_rows()`

The `row` already contains all fields from `list_library_images()`, so `scribble_origin` will flow through automatically. No change needed here.

### 5. Frontend: Track origin state

**File: [`frontend/src/App.jsx`](frontend/src/App.jsx)**

Add state:
```jsx
const [scribbleOrigin, setScribbleOrigin] = useState('')  // '' | 'manual' | 'modelo' | 'modelo_modificado'
```

### 6. Frontend: Set origin on save

**File: [`frontend/src/App.jsx`](frontend/src/App.jsx:1657)** — `saveScribbleDraft()`

Pass `scribbleOrigin` to the API:
```jsx
const res = await apiPost('/api/scribble/draft/save', {
    session_id: sessionId,
    image_id: imageId,
    scribble_map_b64: scribble,
    scribble_origin: scribbleOrigin,
})
```

### 7. Frontend: Set origin on model apply

**File: [`frontend/src/App.jsx`](frontend/src/App.jsx:2775)** — `applyModelPredictionAsScribbles()`

Before saving, set origin to `'modelo'`:
```jsx
async function applyModelPredictionAsScribbles() {
    ...
    setScribbleOrigin('modelo')
    await saveScribbleDraft({ silent: true })
    ...
}
```

### 8. Frontend: Detect manual modifications

**File: [`frontend/src/App.jsx`](frontend/src/App.jsx)** — scribble editing handlers

When the user draws manually (in `onEditorPointerDown` / `onEditorPointerUp` / brush events), if `scribbleOrigin` is `'modelo'`, change it to `'modelo_modificado'`. If no model was ever applied, keep as `'manual'`.

Logic:
- On any manual scribble edit (pointer down on editor with tool active):
  ```jsx
  setScribbleOrigin((prev) => {
      if (prev === 'modelo') return 'modelo_modificado'
      if (!prev) return 'manual'
      return prev
  })
  ```

### 9. Frontend: Reset origin on image load

**File: [`frontend/src/App.jsx`](frontend/src/App.jsx)** — `loadSavedImage()` / `api_image_load`

When loading a saved image, restore the `scribble_origin` from the library meta:
```jsx
setScribbleOrigin(libraryMeta?.scribble_origin || '')
```

### 10. Frontend: Filter dropdown in table

**File: [`frontend/src/App.jsx`](frontend/src/App.jsx:9288)** — Modelos de Asistencia table

Add a filter row above the table:
```jsx
<div className="inline">
  <label>
    Origen:{' '}
    <select value={originFilter} onChange={(e) => setOriginFilter(e.target.value)}>
      <option value="all">Todos</option>
      <option value="manual">Manual</option>
      <option value="modelo">Modelo</option>
      <option value="modelo_modificado">Modelo + Modificado</option>
    </select>
  </label>
</div>
```

Add state:
```jsx
const [originFilter, setOriginFilter] = useState('all')
```

Apply filter in the `withScribbles` computation:
```jsx
const withScribbles = modelDataset.filter((item) => {
    const c = item.class_counts || {}
    const hasScribbles = (c.fiber || 0) > 0 || (c.halo || 0) > 0 || (c.background || 0) > 0
    if (!hasScribbles) return false
    if (originFilter !== 'all' && item.scribble_origin !== originFilter) return false
    return true
})
```

### 11. Frontend: Show origin tag in each row

**File: [`frontend/src/App.jsx`](frontend/src/App.jsx:9379)** — table row

Add a column or tag showing the origin:
```jsx
<td>
  <span className={`origin-tag origin-${item.scribble_origin || 'manual'}`}>
    {item.scribble_origin === 'modelo' ? 'Modelo' :
     item.scribble_origin === 'modelo_modificado' ? 'Modelo+Mod' :
     'Manual'}
  </span>
</td>
```

Add CSS in [`frontend/src/styles.css`](frontend/src/styles.css):
```css
.origin-tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
}
.origin-manual { background: #e2e3e5; color: #383d41; }
.origin-modelo { background: #d4edda; color: #155724; }
.origin-modelo_modificado { background: #fff3cd; color: #856404; }
```

---

## Files to Modify

| File | Changes |
|------|---------|
| [`backend/persistence.py`](backend/persistence.py:65) | Add `origin` param to `save_scribble_draft()`, store in meta |
| [`backend/main.py`](backend/main.py:139) | Add `scribble_origin` field to `ScribbleDraftSaveReq` |
| [`backend/main.py`](backend/main.py:768) | Pass `scribble_origin` to `save_scribble_draft()` |
| [`backend/library_store.py`](backend/library_store.py:126) | Expose `scribble_origin` in `list_library_images()` |
| [`frontend/src/App.jsx`](frontend/src/App.jsx) | Add state, pass origin on save, detect edits, add filter + display |

## Test Plan

1. `vite build` — must succeed
2. `pytest tests/test_core.py tests/test_api.py tests/test_navigation.py` — must pass
3. Manual: Save scribble manually → check origin='manual' in table
4. Manual: Apply model prediction → check origin='modelo'
5. Manual: Apply model then edit → check origin='modelo_modificado'
6. Manual: Filter by each origin type
