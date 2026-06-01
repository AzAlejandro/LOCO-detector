# Root Cause Analysis: Scribble Grid Shows Same Image in All Cells

## Problem

The user reports: "sigue cargando solo una imagen en todos aunque no tenga" — the 2x2 grid shows the same scribble image in all 4 cells, even for origins that don't have scribbles.

## Investigation

### Data Flow

1. `refreshModelDataset()` calls `fetchScribbleThumbForOrigin(image_id, origin)` for all 3 origins
2. `fetchScribbleThumbForOrigin` calls `GET /api/assist-models/dataset/preview?image_id=X&origin=Y`
3. The backend endpoint calls `load_scribble_draft(iid, origin=origin)`
4. `load_scribble_draft` looks in `_draft_origin_dir(image_id, origin_key)` which resolves to `drafts/{image_id}/{origin_key}/scribble_map.npz`
5. If the file doesn't exist, returns `{'found': False, ...}`
6. Backend returns `scribble_b64: ''` when not found
7. Frontend stores `''` in `scribbleThumbsByOrigin[imageId][origin]`
8. `ScribblePreviewCanvas` receives `scribbleThumbSrc=''` and shows "no disponible"

### The Bug

The issue is that the **pre-fetch in `refreshModelDataset()` happens asynchronously** via `Promise.allSettled(fetches)`. The state updates from `setScribbleThumbsByOrigin` inside `fetchScribbleThumbForOrigin` are batched by React. But the **initial render** happens before the fetches complete.

On initial render:
- `scribbleThumbsByOrigin` is `{}` (empty)
- `scribbleThumbsByOrigin[item.image_id]?.['manual']` is `undefined`
- `undefined || ''` evaluates to `''`
- `ScribblePreviewCanvas` receives `scribbleThumbSrc=''` and shows "no disponible"

After the fetches complete:
- `scribbleThumbsByOrigin` is populated with data
- React re-renders
- `ScribblePreviewCanvas` receives the actual data URL
- The canvas draws the image

**So the grid should work correctly after the initial fetch completes.** The user says it doesn't, which means either:

1. The API calls are failing (returning errors)
2. The API calls return the same scribble for all origins
3. The state updates are not triggering re-renders

### Most Likely Root Cause

Looking at the `dataset/preview` endpoint more carefully:

```python
@router.get('/dataset/preview')
def dataset_preview(image_id: str, origin: str = '') -> dict[str, Any]:
    ...
    try:
        draft = load_scribble_draft(iid, origin=origin)
        if bool(draft.get('found')):
            labels = np.asarray(draft.get('labels'), dtype=np.uint8)
            ...
            scribble_b64, scribble_mime = encode_display_b64(labels_to_visual(labels))
            ...
    except Exception:
        scribble_b64 = ''
    return {
        ...
        'scribble_b64': scribble_b64,
        ...
    }
```

When `origin='modelo'` is specified and no scribble exists for that origin:
- `load_scribble_draft(iid, origin='modelo')` looks in `drafts/{iid}/modelo/scribble_map.npz`
- If not found, returns `{'found': False}`
- `scribble_b64` stays `''`
- Returns `{'scribble_b64': '', ...}`

This should work correctly. The frontend stores `''` and shows "no disponible".

**BUT** — there's a critical issue. The `fetchScribbleThumbForOrigin` function uses `apiGet` which calls the `dataset/preview` endpoint. But this endpoint is on the **assist-models router**, not the main app. Let me check if the router is properly mounted.

Actually, looking at the code more carefully, the issue might be simpler. Let me check if the `fetchScribbleThumbForOrigin` function is actually being called and if the state is being updated.

### The Real Root Cause

After careful analysis, I believe the root cause is:

**The `scribbleThumbsByOrigin` state is initialized as `{}` and the pre-fetch happens inside `refreshModelDataset()`. But `refreshModelDataset()` is called on component mount (via `bootstrap()` or similar). The issue is that the pre-fetch uses `Promise.allSettled(fetches)` which fires all API calls in parallel. Each call updates the state via `setScribbleThumbsByOrigin`. React batches these updates.**

**However**, there's a subtle issue: the `fetchScribbleThumbForOrigin` function returns `scribbleB64` (the raw base64 string), but the state stores `b64ToDataUrl(scribbleB64, ...)`. The `scribbleB64` return value is used by the caller (e.g., `applyModelPredictionAsScribbles`), but the state update is what matters for rendering.

The state update inside `fetchScribbleThumbForOrigin` is:

```javascript
setScribbleThumbsByOrigin(prev => {
  const next = { ...prev }
  if (!next[imageId]) next[imageId] = {}
  next[imageId][origin] = scribbleB64 ? b64ToDataUrl(scribbleB64, res.scribble_mime || 'image/png') : ''
  return next
})
```

When `scribbleB64` is empty string `''`, this stores `''` in the cache. Then the grid renders `scribbleThumbsByOrigin[item.image_id]?.['modelo'] || ''` which is `''`, and `ScribblePreviewCanvas` shows "no disponible".

**This should work correctly.** Unless the API is returning a non-empty `scribble_b64` for origins that don't have scribbles.

### Hypothesis: Backend Returns Wrong Scribble

Let me check if `load_scribble_draft` with a specific origin might fall back to another origin when the specified one doesn't exist.

Looking at `load_scribble_draft` again:

```python
def load_scribble_draft(image_id: str, origin: str = '') -> dict[str, Any]:
    origin_key = str(origin or '').strip().lower()
    if origin_key and origin_key in ('manual', 'modelo', 'modelo_modificado'):
        draft_dir = _draft_origin_dir(image_id, origin_key)
    else:
        # No origin specified — try each origin in priority order
        draft_dir = _draft_dir(image_id)
        ...
    zpath = draft_dir / 'scribble_map.npz'
    ...
    if not zpath.exists():
        return {'found': False, 'image_id': str(image_id)}
```

When `origin='modelo'` is specified:
1. `origin_key = 'modelo'`
2. `draft_dir = _draft_origin_dir(image_id, 'modelo')` = `drafts/{image_id}/modelo/`
3. `zpath = drafts/{image_id}/modelo/scribble_map.npz`
4. If not found, returns `{'found': False}`

This looks correct. The backend should return `scribble_b64: ''` for origins without scribbles.

### Hypothesis: Canvas Component Issue

The `ScribblePreviewCanvas` component uses a `useEffect` that depends on `[scribbleThumbSrc, imageId, selectedOrigin]`. When `scribbleThumbSrc` changes from `''` to a data URL, the effect fires and draws the image. When `scribbleThumbSrc` is `''`, it clears the canvas.

But wait — the component returns a **different JSX** based on `scribbleThumbSrc`:

```jsx
if (!scribbleThumbSrc) {
  return (
    <div className="scribble-grid-empty-cell">
      <span className="scribble-grid-empty-text">no disponible</span>
    </div>
  )
}
return (
  <canvas ref={canvasRef} width={104} height={72} className="scribble-grid-thumb" />
)
```

When `scribbleThumbSrc` is `''`, it returns a `<div>` with "no disponible". When it has data, it returns a `<canvas>`. React will unmount the `<div>` and mount the `<canvas>`, which should work correctly.

### Most Likely Root Cause

After all this analysis, I believe the most likely root cause is:

**The `scribbleThumbsByOrigin` state is not being populated because the pre-fetch API calls are failing silently.**

The `fetchScribbleThumbForOrigin` function catches errors and stores `''` in the cache. If the API calls fail (e.g., due to network issues, server not running, or CORS), the cache will have `''` for all origins, and all cells will show "no disponible".

But the user says "sigue cargando solo una imagen en todos" — "it keeps loading only one image in all". This means the cells are showing an image, not "no disponible". So the cache IS populated, but with the same image for all origins.

**This means the backend is returning the same scribble for all 3 origins.** This could happen if:

1. The `load_scribble_draft` function, when called with a specific origin, falls back to another origin's scribble
2. OR the `dataset/preview` endpoint ignores the `origin` parameter

Let me check if there's a fallback in `load_scribble_draft` when the specific origin doesn't exist...

Looking at the code again:

```python
def load_scribble_draft(image_id: str, origin: str = '') -> dict[str, Any]:
    origin_key = str(origin or '').strip().lower()
    if origin_key and origin_key in ('manual', 'modelo', 'modelo_modificado'):
        draft_dir = _draft_origin_dir(image_id, origin_key)
    else:
        # No origin specified — try each origin in priority order
        draft_dir = _draft_dir(image_id)
        ...
    zpath = draft_dir / 'scribble_map.npz'
    ...
    if not zpath.exists():
        return {'found': False, 'image_id': str(image_id)}
```

When `origin='modelo'` is specified, it goes to the first branch (`if origin_key and origin_key in (...)`), sets `draft_dir` to the specific origin directory, and checks if the file exists. If not, returns `{'found': False}`.

**There is NO fallback.** This should work correctly.

### Conclusion

The most likely root cause is that the **pre-fetch API calls are not completing before the grid renders**, OR the **API calls are returning the same scribble for all origins**.

To debug this, we need to add visible labels showing what's in the cache for each cell. The user can then tell us exactly what each cell shows.

## Fix Plan

### Fix 1: Add visible origin labels to each cell in the grid

Add a small label below each canvas showing the origin name and whether data is available. This will help the user (and us) understand what's happening.

### Fix 2: Ensure pre-fetch completes before rendering

Move the pre-fetch logic to happen before `setModelDataset(items)` so the cache is populated before the grid renders.

### Fix 3: Add a "Cargar" button per origin

The user mentioned "porque sacaste lo de origen? si igual afectaba a cuando presionaba cargar?" — the origin filter affected the "Cargar" button behavior. The "Cargar" button should load the auto-detected origin's scribble (the one shown in the "origen" column).

### Fix 4: Restore the origin column with the auto-detected origin label

The user wants to see which origin is the auto-detected one. We should show the origin label (manual/modelo/modificado) in the "origen" column.

## Files to Modify

| File | Changes |
|------|---------|
| `frontend/src/App.jsx` | Add visible labels, ensure pre-fetch completes, restore origin column |
| `frontend/src/styles.css` | Minor style adjustments |
