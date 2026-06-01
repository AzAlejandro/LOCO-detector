# Scribble Grid Preview Redesign

## Problem

The current Modelos de Asistencia table shows only **one scribble thumbnail** (the auto-detected origin's scribble) alongside the real image. A dropdown lets users switch between origins (`manual`, `modelo`, `modelo_modificado`), but the thumbnail doesn't update visually when the dropdown changes due to complex caching/fallback issues.

## Proposed Solution

Replace the single scribble thumbnail + dropdown with a **2x2 grid** showing all 4 previews simultaneously:

```
┌──────────────────┬──────────────────┐
│   Imagen real    │  manual          │
│                  │  (scribble or    │
│                  │   "no disponible")│
├──────────────────┼──────────────────┤
│  modelo          │  modificado      │
│  (scribble or    │  (scribble or    │
│   "no disponible")│  "no disponible")│
└──────────────────┴──────────────────┘
```

Each cell shows:
- A **label** (the origin name)
- The **scribble thumbnail** if available, or a **"no disponible"** placeholder if not

## Changes Required

### 1. Backend: `dataset/preview` endpoint (NO CHANGES NEEDED)

The endpoint already accepts an `origin` parameter and returns the scribble for that origin. No backend changes needed.

### 2. Frontend: `refreshModelDataset()` — Pre-fetch all 3 origins

In [`frontend/src/App.jsx`](frontend/src/App.jsx:2717), change the pre-fetch loop to always fetch all 3 origins (`manual`, `modelo`, `modelo_modificado`) for every image, not just the ones in `scribble_origins_available`.

```javascript
// Before (line 2734-2740):
const origins = item.scribble_origins_available || ['manual']

// After:
const origins = ['manual', 'modelo', 'modelo_modificado']
```

This ensures all 3 scribble thumbnails are cached on page load.

### 3. Frontend: Replace thumbnail column with 2x2 grid

In [`frontend/src/App.jsx`](frontend/src/App.jsx:9666-9675), replace the current single-column thumbnail + dropdown with a 2x2 grid.

**Remove:**
- The `previewOriginByImageId` state (line ~674)
- The `setPreviewOrigin` function (line ~680)
- The `selectedOrigin` / `thumbForOrigin` / `scribbleThumbSrc` computation (lines ~9646-9654)
- The dropdown `<select>` (lines ~9684-9704)
- The `console.log` at line ~9655

**Add:**
A 2x2 grid component that renders 4 cells:

```jsx
<div className="scribble-grid">
  {/* Top-left: Real image */}
  <div className="scribble-grid-cell">
    <span className="scribble-grid-label">Imagen</span>
    {thumbs.real ? (
      <img src={thumbs.real} alt="" className="scribble-grid-thumb" />
    ) : (
      <span className="scribble-grid-empty">sin imagen</span>
    )}
  </div>

  {/* Top-right: manual */}
  <div className="scribble-grid-cell">
    <span className="scribble-grid-label">manual</span>
    <ScribblePreviewCanvas
      scribbleThumbSrc={scribbleThumbsByOrigin[item.image_id]?.['manual'] || ''}
      imageId={item.image_id}
      selectedOrigin="manual"
    />
  </div>

  {/* Bottom-left: modelo */}
  <div className="scribble-grid-cell">
    <span className="scribble-grid-label">modelo</span>
    <ScribblePreviewCanvas
      scribbleThumbSrc={scribbleThumbsByOrigin[item.image_id]?.['modelo'] || ''}
      imageId={item.image_id}
      selectedOrigin="modelo"
    />
  </div>

  {/* Bottom-right: modelo_modificado */}
  <div className="scribble-grid-cell">
    <span className="scribble-grid-label">modificado</span>
    <ScribblePreviewCanvas
      scribbleThumbSrc={scribbleThumbsByOrigin[item.image_id]?.['modelo_modificado'] || ''}
      imageId={item.image_id}
      selectedOrigin="modelo_modificado"
    />
  </div>
</div>
```

### 4. Frontend: Update `ScribblePreviewCanvas` to show "no disponible"

In [`frontend/src/App.jsx`](frontend/src/App.jsx:561-603), update the empty state to show a "no disponible" label instead of just "scribble":

```jsx
if (!scribbleThumbSrc) {
  return (
    <div className="scribble-grid-empty-cell">
      <span className="scribble-grid-empty-text">no disponible</span>
    </div>
  )
}
```

### 5. Frontend: Update `openModelImagePreview` to show all 4 origins in the modal

In [`frontend/src/App.jsx`](frontend/src/App.jsx:1172-1196), update the image preview modal to show all 4 origins (real, manual, modelo, modelo_modificado) in a 2x2 grid instead of just real + scribble.

The modal should fetch all 3 origins' previews and display them in a grid.

### 6. CSS: Add styles for the grid

In [`frontend/src/styles.css`](frontend/src/styles.css:232-235), add styles for the new grid layout:

```css
.scribble-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px;
  width: 216px;
}
.scribble-grid-cell {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}
.scribble-grid-label {
  font-size: 9px;
  color: var(--muted);
  line-height: 1.2;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}
.scribble-grid-thumb {
  width: 104px;
  height: 72px;
  object-fit: contain;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #f8fbff;
}
.scribble-grid-empty-cell {
  width: 104px;
  height: 72px;
  display: grid;
  place-items: center;
  border: 1px dashed var(--line);
  border-radius: 6px;
  background: #f8fbff;
}
.scribble-grid-empty-text {
  font-size: 10px;
  color: var(--muted);
  text-align: center;
}
```

### 7. Remove unused state

Remove these states and functions that are no longer needed:
- `previewOriginByImageId` state (line ~674)
- `setPreviewOrigin` function (line ~680)
- `originFilter` state (line ~673) — if only used for the dropdown filter

## Files to Modify

| File | Changes |
|------|---------|
| [`frontend/src/App.jsx`](frontend/src/App.jsx) | Replace thumbnail column with 2x2 grid, update `ScribblePreviewCanvas`, update `openModelImagePreview`, update `refreshModelDataset`, remove unused state |
| [`frontend/src/styles.css`](frontend/src/styles.css) | Add grid styles |

## Files NOT Modified

| File | Reason |
|------|--------|
| [`backend/assist_models.py`](backend/assist_models.py) | Endpoint already supports `origin` parameter |
| [`backend/persistence.py`](backend/persistence.py) | No changes needed |
| [`backend/library_store.py`](backend/library_store.py) | No changes needed |
| [`frontend/src/api.js`](frontend/src/api.js) | No changes needed |

## Migration Steps

1. Update `refreshModelDataset()` to pre-fetch all 3 origins
2. Update `ScribblePreviewCanvas` to show "no disponible" for empty state
3. Replace the thumbnail column with the 2x2 grid
4. Remove unused state (`previewOriginByImageId`, `setPreviewOrigin`, `originFilter`)
5. Update `openModelImagePreview` modal to show all 4 origins
6. Add CSS styles for the grid
7. Run tests to verify no regressions
