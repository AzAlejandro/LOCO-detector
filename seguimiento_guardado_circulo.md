# Seguimiento: Guardado de circulos

## Tabla de debug

| Paso | Hipotesis | Resultado | Estado |
|------|-----------|-----------|--------|
| 1 | Backend necesita reinicio para cargar nuevo codigo | El contador 22V persiste en F5 (el endpoint `/points/counts` funciona). `_ok()` no estaba importado → fue reemplazado por dict directo. | ✅ Corregido |
| 2 | `normalize_points` descarta `circle_type` al guardar | Fix aplicado en `persistence.py:121`. `load_points` retorna `{found, points, ...}`. El guardado funciona. | ✅ Corregido |
| 3 | `apiPost` falla silenciosamente | Se agrego `.then`/`.catch` con `[DEBUG-CIRCLE]`. Aun sin confirmacion de consola. | 🔍 Pendiente |
| 4 | Endpoint `/points/save-circle` no existe en backend | Existe en `api.py:2707`. Se agregaron `print` de debug. | ✅ Corregido |
| 5 | Circulos no persisten visualmente al recargar | `locoDatasetCircles` se limpiaba en `resetImageScopedState` pero nunca se recargaba del backend. Ademas `radius_px` no se guardaba. | ✅ Corregido |
| 6 | `radius_px` no se guardaba en el backend | `normalize_points` y `SaveCircleReq` no incluyen `radius_px`. | ✅ Corregido |

## Cambios aplicados (27 Mayo - ronda 3)

### Backend
- **`api.py:120`** — `SaveCircleReq` ahora incluye `radius_px: float = 0`
- **`api.py:2718`** — `points_save_circle` almacena `radius_px` en cada punto
- **`api.py:2735-2751`** — Nuevo endpoint `GET /points/list?image_id=...` retorna puntos completos con `{x, y, circle_type, radius_px}`
- **`persistence.py:121`** — `normalize_points` ahora preserva `radius_px`

### Frontend
- **`App.jsx:4669`** — `apiPost` ahora envia `radius_px: Number(draft.radius_px)`
- **`App.jsx:5108-5124`** — Nueva funcion `loadLocoDatasetCirclesFromBackend()`: llama `/points/list`, mapea `circle_type` de vuelta a labels LOCO (`crossing`→`invalid_crossing`, `other_valid`→`invalid_other`) y puebla `locoDatasetCircles`
- **`App.jsx:3383`** — Llamada a `loadLocoDatasetCirclesFromBackend(loadedImageId)` en `loadSavedImage`, despues de `resetImageScopedState` y `loadDiameterPoints`

## Bugs encontrados (27 Mayo)

| # | Bug | Causa | Fix |
|---|-----|-------|-----|
| 1 | `_ok()` sin importar en `api.py:2704` | `_ok` definido en `main.py` pero nunca importado en `api.py`. Causa `NameError` al invocar `/points/counts`. | Reemplazado por `return {'ok': True, 'payload': {...}}` |
| 2 | `locoDatasetCircles` persiste al cambiar de imagen | `resetImageScopedState()` no limpia `locoDatasetCircles`. Circulos de imagen A se ven en imagen B. | Agregado `setLocoDatasetCircles([])` en `resetImageScopedState:3330` |
| 3 | Contador ignora circulos nuevos si hay dataset LOCO | El display prefiere `locoDatasetCircleCounts` (dataset LOCO) sobre `circleTypeCounts` (auto-save). El auto-save solo actualizaba `circleTypeCounts`. | Auto-save ahora actualiza AMBOS estados (`circleTypeCounts` y `locoDatasetCircleCounts`) |
| 4 | Delete no decrementa el contador | El handler de tecla Delete/Backspace y `deleteSelectedLocoDatasetCircle` removian el circulo del array pero no actualizaban los contadores. | Ambos handlers ahora decrementan `circleTypeCounts` y `locoDatasetCircleCounts` |
| 5 | Canvas warning: `willReadFrequently` | Advertencia de rendimiento de Canvas 2D, no afecta funcionalidad. | Sin accion requerida |

## Puntos de debug activos

### Frontend
- **[DEBUG-CIRCLE]** Al dibujar — `App.jsx:4632` → `console.log` antes de `apiPost`, `.then`/`.catch`
- **[DEBUG-COUNTS]** Al pre-fetch — `App.jsx:2795` → `console.log` tras `/points/counts`
- **[DEBUG-REFRESH]** Al clic en "Refrescar" — `App.jsx:7519` → `console.log` con estado actual

### Backend
- **[DEBUG-SAVE]** En `/points/save-circle` — `api.py:2709-2718` → `print` de `image_id`, `x`, `y`, `type`, puntos existentes, guardado
- **[DEBUG-ROUTES]** Nuevo endpoint — `api.py:2720` → `GET /points/debug-routes` lista rutas y archivos

## Verificacion

1. Abrir consola del navegador (F12 → Console)
2. **Verificar que el codigo cargo**: Debes ver `[APP-V2] LOADED — fixes: clear circles...` en amarillo al cargar la pagina. Si NO ves esto, haz **Ctrl+Shift+R** (hard refresh) o reinicia `run_local.bat`.
3. Cargar imagen desde "Imagenes con mascara" — ver `[DEBUG-LOAD] resetImageScopedState called`
4. Ir a Generate Dataset y dibujar un circulo — ver `[DEBUG-V2] onLocoDatasetPointerUp FIRED`
5. Verificar `[DEBUG-CIRCLE] Saving: {...}` + `Save OK` o `Save FAILED`
6. Ver `[DEBUG-COUNTS] Fetch: img_xxx → {valid: 1, ...}`
7. Ver `[DEBUG-REFRESH]` al clic en "Refrescar"
8. Abrir `http://127.0.0.1:8011/api/diameter-research/points/debug-routes`
