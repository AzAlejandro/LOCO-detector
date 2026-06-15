# Seguimiento de traduccion de interfaz

Fecha: 2026-06-02

## Objetivo
Traducir al español los textos visibles de la interfaz que seguían en inglés, sin tocar contratos de código ni nombres internos que se usan como claves técnicas.

## Archivos intervenidos
- `frontend/src/App.jsx`
- `frontend/src/components/Navigation.jsx`
- `frontend/src/components/TutorialHub.jsx`
- `frontend/src/tutorials.js`
- `backend/main.py`

## Cambios aplicados

### Navegación y tutoriales
- `Overview general` -> `Recorrido general`
- `Overview del proyecto` -> `Vista general del proyecto`
- `Subpestana` -> `Subpestaña`
- `subpestaÃ±as` -> `subpestañas`
- Iconos del menú reescritos en UTF-8 correcto:
  - `✏️`, `🔵`, `📏`, `⚙`, `📘`, `☰`, `✕`, `⋯`

### Scribble y revisión
- `Run selected` -> `Ejecutar seleccionado`
- `Run batch` -> `Ejecutar batch`
- `Select all` -> `Seleccionar todo`
- `Select none` -> `Limpiar seleccion`
- `Prev` -> `Anterior`
- `Next` -> `Siguiente`

### Medición de diámetros
- `Diameter Research` -> `Medicion de Diametros`
- `Linea mask` -> `Linea mascara`
- `Circle-square mask` -> `Mascara circle-square`
- `Ellipse oriented fit` -> `Ajuste eliptico orientado`
- `Undo` -> `Deshacer`
- `Redo` -> `Rehacer`
- `Reset` -> `Reiniciar`
- Toast `Diameter Research` -> `Medicion de Diametros`

### Dataset y augmentación LOCO
- `Generate Dataset` -> `Generar Dataset`
- `Preview` -> `Vista previa`
- `Preview selected` -> `Previsualizar seleccion`
- `Apply to all dataset` -> `Aplicar a todo el dataset`
- `Clear augmented` -> `Limpiar augmentados`
- `Dataset main` -> `Dataset principal`
- `Raw / Valid / Invalid` -> `Base / Validos / Invalidos`
- `other invalid` -> `otro invalido`
- Campos de augmentación:
  - `random angles` -> `angulos aleatorios`
  - `modes` -> `modos`
  - `ops` -> `operaciones`
  - `amount min/max` -> `cantidad min/max`
  - `methods` -> `metodos`
  - `size min/max` -> `tamano min/max`
  - `sizes` -> `tamanos`
- Toast `Augmentation` -> `Aumentacion`

### Entrenamiento LOCO
- `Training` -> `Entrenamiento`
- `Original only / Augmented only / Original + Augmented`
  - `Solo original / Solo augmentado / Original + Augmentado`
- `test size` -> `tamano de test`
- `random seed` -> `semilla aleatoria`
- `False Positives / False Negatives`
  - `Falsos positivos / Falsos negativos`
- `rank` -> `rango`
- `accuracy` -> `exactitud` en parte del ranking visible
- `Pred invalid / Pred valid / Real invalid / Real valid`
  - traducidos a sus equivalentes en español
- `Crossing rejection metrics` -> `Metricas de rechazo de crossing`
- `Combined decision thresholds` -> `Umbrales de decision combinada`
- `Thresholds (binario)` -> `Umbrales (binario)`
- `Performance by radius size (...)`
  - `Rendimiento por tamano de radio (...)`
- `Error Review (...)`
  - `Revision de errores (...)`
- `all` en varios filtros visibles -> `todos`
- `errors` -> `errores`
- `Train Models` -> `Entrenar modelos`

### Test de modelo y detector
- `Test circle model` -> `Probar modelo de circulos`
- `model` -> `modelo` en selectores visibles
- `threshold` -> `umbral` en selectores visibles
- `Predict circles` -> `Predecir circulos`
- `Run base detector` -> `Ejecutar detector base`
- `Apply threshold` -> `Aplicar umbral`
- `Apply threshold onward` -> `Aplicar umbral en adelante`
- `Apply NMS` -> `Aplicar NMS`
- `Apply NMS onward` -> `Aplicar NMS en adelante`
- `Apply spatial` -> `Aplicar filtro espacial`
- `Apply spatial onward` -> `Aplicar filtro espacial en adelante`
- `Apply pending filters` -> `Aplicar filtros pendientes`
- `Run base` -> `Ejecutar base`
- `Run Diameter on Accepted` -> `Ejecutar diametro en aceptados`
- Textos con mojibake en test:
  - `MÃ©tricas` -> `Metricas`
  - `cÃ­rculos` / `cÃ­rculo` -> `circulos` / `circulo`

### Backend con mensaje visible
- `Mask thumb loaded.` -> `Miniatura de mascara cargada.`

## Validaciones ejecutadas
- Búsqueda dirigida de cadenas visibles en inglés antes y después de editar.
- `npm run build`

## Pendientes para segunda pasada
- Algunos términos técnicos visibles se mantuvieron por ahora:
  - `LOCO`, `NMS`, `CV5`, nombres de modelos (`CatBoost`, `LightGBM`, `XGBoost`)
  - algunas columnas métricas tipo `F1_valid`, `P_cross`, `run`
- `No file chosen` es texto nativo del navegador para `<input type="file">`; no se corrige con una traducción simple del repo.
- Quedan posibles textos técnicos visibles en inglés dentro de selectores o parámetros avanzados del detector (`tile balanced`, `random seeded`, `row major`, etc.) si se decide hacer una segunda ronda más agresiva.
