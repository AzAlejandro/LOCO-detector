# Plan: Origin Selection in Dataset Table Scribble Grid

## Problem

In the dataset table (Models tab), the scribble grid shows a 2x2 layout:

```
+----------+----------+
|  Imagen  |  manual  |
+----------+----------+
|  modelo  |modificado|
+----------+----------+
```

Currently, the "Cargar" button at line [`9707`](frontend/src/App.jsx:9707) calls:

```js
loadSavedImage(item.image_id, item.scribble_origin || '')
```

This always passes `item.scribble_origin` (the **last saved origin**), ignoring which origin the user visually selected in the grid. The grid cells for manual/modelo/modificado are display-only — they are **not clickable** for selection.

The user wants: **clicking on a specific origin cell (manual/modelo/modificado) should select that origin, and then clicking "Cargar" should load that selected origin's scribbles.**

## Current Architecture

### State
- [`scribbleThumbsByOrigin`](frontend/src/App.jsx:666) — `{ [image_id]: { manual: dataUrl, modelo: dataUrl, modelo_modificado: dataUrl } }` — caches scribble thumbnails per image per origin
- [`scribbleOrigin`](frontend/src/App.jsx:663) — current scribble origin for the active editing session (single string, not per-image)
- **No per-image selected-origin state exists** — this is what needs to be added

### Key Code Locations
- Scribble grid rendering: [`lines 9640-9695`](frontend/src/App.jsx:9640)
- "Cargar" button: [`lines 9704-9713`](frontend/src/App.jsx:9704)
- `loadSavedImage` function: [`lines 3043-3085`](frontend/src/App.jsx:3043) — already accepts `targetOrigin` parameter
- `ScribblePreviewCanvas` component: [`lines 562-603`](frontend/src/App.jsx:562) — renders scribble thumbnails onto canvas
- `refreshModelDataset` pre-fetch: [`lines 2741-2754`](frontend/src/App.jsx:2741) — pre-fetches all 3 origins

## Proposed Changes

### 1. Add `selectedGridOrigin` state (per-image)

Add a new state variable after line [`666`](frontend/src/App.jsx:666):

```js
const [selectedGridOrigin, setSelectedGridOrigin] = useState({})
```

This is a dictionary keyed by `image_id`, e.g.:
```js
{ "img123": "manual", "img456": "modelo_modificado" }
```

### 2. Make scribble grid cells clickable

Add `onClick` handlers to the three origin cells (manual, modelo, modelo_modificado) in the scribble grid at [`lines 9656-9694`](frontend/src/App.jsx:9656).

When clicked, update `selectedGridOrigin` for that `item.image_id`:

```js
onClick={() => setSelectedGridOrigin(prev => ({ ...prev, [item.image_id]: 'manual' }))}
```

The "Imagen" cell (top-left) should NOT be clickable — it's the real image, not an origin.

### 3. Add visual selection indicator

Add a CSS class (e.g., `scribble-grid-cell-selected`) to the selected cell so the user can see which origin is currently selected.

The selected cell is determined by:
```js
selectedGridOrigin[item.image_id]
```

Compare against each cell's origin value. If they match, add the selected class.

### 4. Update "Cargar" button to use selected origin

Change line [`9707`](frontend/src/App.jsx:9707) from:

```js
loadSavedImage(item.image_id, item.scribble_origin || '')
```

To:

```js
const originToLoad = selectedGridOrigin[item.image_id] || item.scribble_origin || ''
loadSavedImage(item.image_id, originToLoad)
```

This means:
- If the user has explicitly clicked an origin cell, use that origin
- Otherwise, fall back to `item.scribble_origin` (the last saved origin)
- If neither exists, pass empty string (backend default behavior)

### 5. Add CSS for selected state

In [`frontend/src/styles.css`](frontend/src/styles.css), add a style for the selected cell:

```css
.scribble-grid-cell-selected {
  outline: 2px solid #4fc3f7;
  outline-offset: -2px;
  background-color: rgba(79, 195, 247, 0.08);
}
```

## Files to Modify

| File | Changes |
|------|---------|
| [`frontend/src/App.jsx`](frontend/src/App.jsx) | Add `selectedGridOrigin` state, make cells clickable, update "Cargar" button logic |
| [`frontend/src/styles.css`](frontend/src/styles.css) | Add `.scribble-grid-cell-selected` CSS class |

## Files NOT Modified

- Backend files — no API changes needed; `loadSavedImage` already accepts `targetOrigin`
- `ScribblePreviewCanvas` — no changes needed; it already receives `selectedOrigin` prop for display
- `refreshModelDataset` — no changes needed; thumbnails are already pre-fetched for all 3 origins

## Detailed Implementation Steps

### Step 1: Add state variable

After line [`666`](frontend/src/App.jsx:666) (`const [scribbleThumbsByOrigin, setScribbleThumbsByOrigin] = useState({})`), add:

```js
const [selectedGridOrigin, setSelectedGridOrigin] = useState({})
```

### Step 2: Make origin cells clickable

For each of the three origin cells (lines 9657-9668, 9670-9681, 9683-9694), add an `onClick` handler to the outer `<div className="scribble-grid-cell">`.

The handler sets `selectedGridOrigin[item.image_id]` to the corresponding origin string.

Also add the `scribble-grid-cell-selected` class conditionally:

```jsx
className={`scribble-grid-cell ${selectedGridOrigin[item.image_id] === 'manual' ? 'scribble-grid-cell-selected' : ''}`}
```

### Step 3: Update "Cargar" button

Replace the inline `onClick` at line 9706-9708 with:

```jsx
onClick={() => {
  const originToLoad = selectedGridOrigin[item.image_id] || item.scribble_origin || ''
  loadSavedImage(item.image_id, originToLoad)
}}
```

### Step 4: Add CSS

Append to [`frontend/src/styles.css`](frontend/src/styles.css):

```css
.scribble-grid-cell-selected {
  outline: 2px solid #4fc3f7;
  outline-offset: -2px;
  background-color: rgba(79, 195, 247, 0.08);
  cursor: pointer;
}

.scribble-grid-cell {
  cursor: pointer;
}
```

## Manual Test Steps

1. Open the Models tab and navigate to the dataset table
2. Click the "manual" cell in any row — verify a blue outline appears around that cell
3. Click "Cargar" — verify manual scribbles are loaded (not the last saved origin)
4. Click "modelo" cell — verify outline moves to modelo cell
5. Click "Cargar" — verify modelo scribbles are loaded
6. Click "modificado" cell — verify outline moves
7. Click "Cargar" — verify modified scribbles are loaded
8. Without clicking any cell, click "Cargar" — verify it falls back to `item.scribble_origin`
