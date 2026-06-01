# Plan: Canvas-Based Origin Preview for Model Dataset Table

## Problem

The scribble thumbnail in the Modelos de Asistencia table does not visually update when the user changes the origin dropdown selector. Despite multiple fixes (key props, cache on error, pre-fetching), the `<img>` tag's `src` attribute changes at the React/JavaScript level but the browser does not visually update the displayed image.

## Root Cause Analysis

The debug output from extensive testing confirmed the system works correctly at the code level:
- API calls succeed and return valid scribble data
- The cache (`scribbleThumbsByOrigin`) is updated correctly
- The render shows the correct `thumbForOrigin` value (not `undefined`)
- The `selectedOrigin` changes correctly in the render

The issue is at the **browser rendering level**. When React changes the `src` attribute of an `<img>` element, the browser may not actually reload the image if:
1. The new data URL is similar in content to the previous one
2. The browser's image decoder caches the decoded image data
3. React's reconciliation algorithm reuses the DOM element despite the `key` prop

## Proposed Solution

Replace the `<img>` tag with a **`<canvas>` element** controlled by a `useEffect` hook. This gives us full control over when the image is drawn and completely bypasses any browser caching or React DOM reconciliation issues.

### Changes Required

#### 1. Add a `ScribblePreviewCanvas` component

Create a new small component (or inline it) that:
- Accepts `scribbleThumbSrc` (data URL) and `imageId` + `selectedOrigin` as props
- Uses a `useRef` for the canvas element
- Uses a `useEffect` that watches `scribbleThumbSrc` and draws the image onto the canvas whenever it changes
- Falls back to a `<span>` with "scribble" text when no image is available

```jsx
function ScribblePreviewCanvas({ scribbleThumbSrc, imageId, selectedOrigin }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !scribbleThumbSrc) return
    const ctx = canvas.getContext('2d')
    const img = new Image()
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
    }
    img.src = scribbleThumbSrc
  }, [scribbleThumbSrc, imageId, selectedOrigin])

  if (!scribbleThumbSrc) {
    return <span className="model-dataset-empty">scribble</span>
  }

  return (
    <canvas
      ref={canvasRef}
      width={104}
      height={72}
      style={{
        width: '104px',
        height: '72px',
        objectFit: 'contain',
        border: '1px solid var(--line)',
        borderRadius: '6px',
        background: '#f8fbff',
      }}
    />
  )
}
```

#### 2. Replace the `<img>` tag in the table

In the model dataset table rendering (around line 9614), replace:
```jsx
{scribbleThumbSrc ? <img key={`scribble-img-${item.image_id}-${selectedOrigin}`} src={scribbleThumbSrc} alt="" /> : <span className="model-dataset-empty">scribble</span>}
```

With:
```jsx
<ScribblePreviewCanvas scribbleThumbSrc={scribbleThumbSrc} imageId={item.image_id} selectedOrigin={selectedOrigin} />
```

#### 3. Add the component definition

Add the `ScribblePreviewCanvas` component definition near the top of the `App` component (or as a standalone function component outside `App`).

### Why This Works

1. **No browser caching**: The `Image()` object is created fresh each time `scribbleThumbSrc` changes. The `onload` callback draws the decoded image onto the canvas.

2. **No React DOM reconciliation**: Canvas elements don't have `src` attributes that React tries to manage. The `useEffect` hook explicitly controls when drawing happens.

3. **Explicit control**: The `useEffect` dependency array `[scribbleThumbSrc, imageId, selectedOrigin]` ensures the canvas is redrawn whenever any of these values change.

4. **Clean fallback**: When `scribbleThumbSrc` is falsy (no scribble for this origin), the canvas is not rendered and the "scribble" text placeholder is shown instead.

### Alternative: Simpler Approach

If the canvas approach is too complex, a simpler alternative is to use a **`background-image` CSS approach**:

```jsx
<div
  style={{
    width: '104px',
    height: '72px',
    backgroundImage: scribbleThumbSrc ? `url(${scribbleThumbSrc})` : 'none',
    backgroundSize: 'contain',
    backgroundPosition: 'center',
    backgroundRepeat: 'no-repeat',
    border: '1px solid var(--line)',
    borderRadius: '6px',
    backgroundColor: '#f8fbff',
  }}
/>
```

This might work because CSS `background-image` is handled differently by the browser's rendering engine than the `<img>` tag's `src` attribute.

### Testing

1. Restart the Python backend
2. Open the Modelos de Asistencia table
3. Change the dropdown for any image from "manual" to "modelo" to "modificado"
4. Verify the scribble thumbnail visually updates
5. Verify that selecting an origin with no scribble shows empty (no scribble overlay)
