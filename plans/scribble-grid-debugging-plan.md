# Scribble Grid Debugging Plan

## Problem

The 2x2 grid shows the same scribble image in all 3 scribble cells (manual, modelo, modelo_modificado), even for origins that don't have scribbles. The "no disponible" placeholder is never shown.

## Root Cause Hypotheses

After extensive code analysis, the possible root causes are:

1. **Backend receives empty `origin` parameter** — If the `origin` query parameter arrives as empty string `''`, [`load_scribble_draft()`](backend/persistence.py:171) falls back to auto-detection (lines 176-190) and returns the auto-detected origin's scribble for ALL 3 calls.

2. **Backend finds scribble files for all 3 origins** — If the user has previously used "Predecir y Aplicar" or auto-detection, scribble files may exist in all 3 origin directories with similar content.

3. **Frontend state update race condition** — The `Promise.allSettled` calls may cause React state updates to overwrite each other.

4. **Canvas component not re-rendering** — The `ScribblePreviewCanvas` `useEffect` may not fire when `scribbleThumbSrc` changes.

## Debugging: Console Logs to Add

### Server-side (backend/assist_models.py)

In `dataset_preview()` (line ~457), add logging to trace the full flow:

```python
print(f"[DBG-PREVIEW] ENTER: image_id={iid}, origin_param={origin!r}")
draft = load_scribble_draft(iid, origin=origin)
print(f"[DBG-PREVIEW] load_scribble_draft result: found={draft.get('found')}, origin_param={origin!r}")
```

This will show us:
- What `origin` value the backend actually receives
- Whether `load_scribble_draft` found a scribble for that origin

### Client-side (frontend/src/App.jsx)

#### 1. In `refreshModelDataset()` — log the pre-fetch order

```javascript
console.log('[DBG-FETCH] refreshModelDataset: starting pre-fetch for', items.length, 'items')
const ALL_ORIGINS = ['manual', 'modelo', 'modelo_modificado']
for (const item of items) {
  for (const origin of ALL_ORIGINS) {
    console.log(`[DBG-FETCH] QUEUE: image_id=${item.image_id}, origin=${origin}`)
    fetches.push(fetchScribbleThumbForOrigin(item.image_id, origin))
  }
}
await Promise.allSettled(fetches)
console.log('[DBG-FETCH] refreshModelDataset: all pre-fetches completed')
```

#### 2. In `fetchScribbleThumbForOrigin()` — log API call and result

```javascript
async function fetchScribbleThumbForOrigin(imageId, origin) {
    if (!imageId || !origin) {
      console.log(`[DBG-FETCH] SKIP: imageId=${imageId}, origin=${origin}`)
      return ''
    }
    console.log(`[DBG-FETCH] CALL: image_id=${imageId}, origin=${origin}`)
    try {
      const res = await apiGet(`/api/assist-models/dataset/preview?image_id=${encodeURIComponent(imageId)}&origin=${encodeURIComponent(origin)}`)
      const scribbleB64 = res?.scribble_b64 || ''
      console.log(`[DBG-FETCH] RESPONSE: image_id=${imageId}, origin=${origin}, scribbleB64.length=${scribbleB64.length}, found=${!!scribbleB64}`)
      setScribbleThumbsByOrigin(prev => {
        const next = { ...prev }
        if (!next[imageId]) next[imageId] = {}
        const dataUrl = scribbleB64 ? b64ToDataUrl(scribbleB64, res.scribble_mime || 'image/png') : ''
        next[imageId][origin] = dataUrl
        console.log(`[DBG-FETCH] STATE-UPDATE: image_id=${imageId}, origin=${origin}, dataUrl=${dataUrl ? 'non-empty' : 'empty'}`)
        return next
      })
      return scribbleB64
    } catch (err) {
      console.log(`[DBG-FETCH] ERROR: image_id=${imageId}, origin=${origin}, err=${errMsg(err)}`)
      setScribbleThumbsByOrigin(prev => {
        const next = { ...prev }
        if (!next[imageId]) next[imageId] = {}
        next[imageId][origin] = ''
        return next
      })
      return ''
    }
  }
```

#### 3. In the grid rendering — log what each cell receives

```jsx
{/* Top-right: manual */}
<div className="scribble-grid-cell">
  <span className="scribble-grid-label">manual</span>
  <ScribblePreviewCanvas
    scribbleThumbSrc={(() => {
      const val = scribbleThumbsByOrigin[item.image_id]?.['manual'] || ''
      console.log(`[DBG-RENDER] image_id=${item.image_id}, origin=manual, scribbleThumbSrc=${val ? 'non-empty' : 'empty'}`)
      return val
    })()}
    imageId={item.image_id}
    selectedOrigin="manual"
  />
</div>
```

(Repeat for modelo and modelo_modificado)

#### 4. In `ScribblePreviewCanvas` — log when the canvas draws

```javascript
useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    console.log(`[DBG-CANVAS] useEffect: imageId=${imageId}, origin=${selectedOrigin}, scribbleThumbSrc=${scribbleThumbSrc ? 'non-empty' : 'empty'}`)
    if (!scribbleThumbSrc) {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      return
    }
    const img = new Image()
    img.onload = () => {
      console.log(`[DBG-CANVAS] drawImage: imageId=${imageId}, origin=${selectedOrigin}`)
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
    }
    img.onerror = () => {
      console.log(`[DBG-CANVAS] error: imageId=${imageId}, origin=${selectedOrigin}`)
      ctx.clearRect(0, 0, canvas.width, canvas.height)
    }
    img.src = scribbleThumbSrc
  }, [scribbleThumbSrc, imageId, selectedOrigin])
```

## Execution Plan

### Step 1: Add all console.log statements (server + client)

### Step 2: Restart backend and frontend

### Step 3: Open Modelos de Asistencia tab

### Step 4: Check server console and browser console for logs

### Step 5: Analyze the log sequence to identify the root cause

The logs will show the EXACT sequence of events:
1. Pre-fetch queue order
2. API calls with origin parameter
3. Backend receiving origin parameter
4. Backend finding/not finding scribble
5. Frontend state updates
6. Canvas component re-renders

### Step 6: Fix the root cause

### Step 7: Restore the origin column and filter

### Step 8: Remove all debug logging

## Files to Modify

| File | Changes |
|------|---------|
| `backend/assist_models.py` | Add server-side `print()` debug logging |
| `frontend/src/App.jsx` | Add client-side `console.log()` statements throughout the data flow |
| `frontend/src/styles.css` | Minor style adjustments if needed |

## Testing

1. Run existing tests: `cd backend && python -m pytest tests/ -v`
2. Manual testing: Open Modelos de Asistencia tab, check console logs
3. Verify the log sequence reveals the root cause
