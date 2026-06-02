# LOCO Detector — Guía completa de parámetros

## Flujo de detección

```
Imagen → Prior mask → Generación de candidatos → Evaluación (modelo binario + multiclase)
  → Filtro por threshold → NMS → Filtro espacial → Círculos aceptados finales
```

---

## 1. Generación de candidatos (sampling)

Controlan **dónde y cómo** se generan los círculos candidatos sobre la imagen.

### `grid step` (default: 10)
Distancia en píxeles entre centros de grilla. Valor bajo = más candidatos densos, valor alto = menos candidatos pero más rápido.

- Rango: 2–128
- Ejemplo: `step=10` genera un centro cada 10 px en X e Y

### `max candidatos` (default: 60000, cap: 60000)
Máximo número de candidatos a evaluar. Si la grilla genera más, se aplica el modo de muestreo para reducirlos.

### `tile px` (default: 128)
Tamaño del tile para el modo `tile_balanced`. Divide la imagen en tiles de este tamaño y selecciona equitativamente de cada tile.

### `seed` (default: 42)
Semilla aleatoria para los modos de muestreo `random_seeded` y `tile_balanced`.

### `radio min` (default: 8)
Radio mínimo de círculo a generar (píxeles). Rango: 1–512.

### `radio max` (default: 32)
Radio máximo de círculo a generar (píxeles). Rango: `radio_min`–512.

### `radio step` (default: 4)
Incremento de radio entre candidatos. Para cada centro de grilla se generan círculos con radios desde `radio_min` hasta `radio_max` en pasos de `radio_step`.

- Ejemplo: `radio_min=8, radio_max=32, radio_step=4` → radios `[8, 12, 16, 20, 24, 28, 32]`

### Modos de muestreo (`muestreo`)

Controla cómo se reduce el número de candidatos cuando la grilla genera más que `max_candidates`.

| Modo | Comportamiento |
|------|---------------|
| `row_major` | Toma los primeros N candidatos en orden de grilla (filas primero). Sesgado hacia esquina superior izquierda. |
| `random_seeded` | Selección aleatoria uniforme usando `seed`. |
| `tile_balanced` | **(default)** Divide la imagen en tiles de `tile_px` y selecciona equitativamente de cada tile. Garantiza cobertura espacial balanceada. |

---

## 2. Thresholds (filtro de score binario)

El modelo binario asigna un **`valid_score`** (0.0–1.0) a cada candidato. El threshold determina si se acepta o rechaza.

### `threshold por radio` (toggle)

- **OFF** (`use_radius_thresholds=false`): Se usa un único threshold general para todos los radios.
- **ON** (`use_radius_thresholds=true`, default): Se usa un threshold distinto según el grupo de radio.

### `general` (default: 0.89)
Threshold único usado cuando `threshold por radio` está desactivado.

### Grupos de radio

| Grupo | Rango | Parámetro | Default |
|-------|-------|-----------|---------|
| `small` | radio < `small limite` | `small th` | 0.85 |
| `medium` | `small limite` ≤ radio < `large limite` | `medium th` | 0.90 |
| `large` | radio ≥ `large limite` | `large th` | 0.95 |

### `small limite` (default: 14)
Límite superior del grupo `small` en píxeles de radio.

### `large limite` (default: 24)
Límite inferior del grupo `large` en píxeles de radio.

**Regla de decisión:**
```
valid_score >= threshold_del_grupo → accepted (pendiente de multiclase y NMS)
valid_score < threshold_del_grupo  → rejected (below_threshold)
```

---

## 3. Multiclase (filtro de cruces)

Si existe un modelo multiclase entrenado (3 clases: valid / crossing / other), se aplica un filtro adicional.

### `crossing threshold` (default: 0.5)
Probabilidad máxima de `crossing` permitida para aceptar un círculo.

**Regla:**
```
prob_crossing <= crossing_threshold → pasa el filtro
prob_crossing > crossing_threshold  → rejected (crossing_detected)
```

**Combinado con threshold binario:**
```
accepted = (valid_score >= threshold) AND (prob_crossing <= crossing_threshold)
```

Un `crossing threshold` **bajo** (ej: 0.3) es más estricto (rechaza más candidatos con posible cruce).
Un `crossing threshold` **alto** (ej: 0.8) es más permisivo.

---

## 4. Circle-NMS (Non-Maximum Suppression)

Elimina círculos duplicados o solapados después del filtro de threshold.

### `modo NMS`

| Modo | Criterio de eliminación |
|------|------------------------|
| `distance_radius` | Se elimina si: `distancia_centros < nms_distancia * radio_menor` **Y** `|r1 - r2| < radio_similar * radio_menor` |
| `circle_iou` | Se elimina si el IoU (Intersection over Union) circular ≥ `IoU th` |

### `nms distancia` (default: 0.5)
Factor de distancia para modo `distance_radius`. Multiplicado por el radio menor.

### `radio similar` (default: 0.4)
Factor de similitud de radio para modo `distance_radius`. Multiplicado por el radio menor.

### `IoU th` (default: 0.4)
Umbral de IoU circular para modo `circle_iou`. Rango: 0.0–1.0.

**Algoritmo NMS:**
1. Ordenar candidatos aceptados por `valid_score` descendente.
2. Tomar el mejor (mayor score) y mantenerlo.
3. Eliminar todos los que sean duplicados según el criterio del modo.
4. Repetir con los restantes hasta que no queden candidatos.

---

## 5. Parámetros de visualización

### `mostrar rechazados` (toggle)
- **ON**: Incluye hasta 800 candidatos rechazados en la respuesta (para depuración).
- **OFF**: Solo devuelve los aceptados (más rápido, menos datos).

### Capas de overlay
- `mascara`: Muestra la prior mask superpuesta.
- `aceptados`: Muestra círculos aceptados en verde.
- `rechazados`: Muestra círculos rechazados en gris punteado.
- `scores`: Muestra etiquetas de score sobre cada círculo.

---

## 6. Filtro espacial final (post-NMS)

Controla la **distribución espacial** de los círculos aceptados después de NMS. Sin este filtro, los círculos tienden a concentrarse en las regiones con más fibras (ej. primer tercio de la imagen), dejando otras zonas sin representación.

### `use_spatial_final_filter` (toggle, default: false)
- **ON**: Aplica el filtro espacial después de NMS.
- **OFF**: Los círculos post-NMS se devuelven directamente.

### `spatial_final_tile_px` (default: 128)
Tamaño del tile en píxeles para dividir la imagen. Cada tile se procesa independientemente.

- Rango: 16–512
- Valor bajo = más tiles = distribución más fina pero menos círculos por tile
- Valor alto = menos tiles = más círculos por tile pero distribución más gruesa

### `spatial_final_max_per_tile` (default: 3)
Máximo número de círculos aceptados por tile. Dentro de cada tile, se ordenan por `valid_score` descendente y se conservan solo los top-N.

- Rango: 1–50
- Ejemplo: `max_per_tile=3` mantiene los 3 mejores círculos de cada tile de 128×128 px

### `spatial_final_min_center_distance_factor` (default: 1.0)
Factor de distancia mínima entre centros dentro del mismo tile. Se calcula como `distancia >= (radio1 + radio2) / 2 * factor`.

- Rango: 0.0–5.0
- `0.0`: Sin restricción de distancia
- `1.0`: Los círculos no pueden solaparse (distancia mínima = radio promedio)
- `>1.0`: Separación adicional entre círculos

### Comportamiento

1. Después de NMS, se toman los círculos aceptados
2. Se dividen en tiles de `spatial_final_tile_px`
3. Dentro de cada tile, se ordenan por `valid_score` descendente
4. Se conservan solo `spatial_final_max_per_tile` por tile
5. Opcionalmente se eliminan círculos demasiado cercanos entre sí
6. Los círculos eliminados se marcan como `removed_by_spatial`

### Visualización en UI

Cuando el filtro está activo, el resumen muestra:
- `Final spatial`: Número de círculos después del filtro espacial
- `Spatial removidos`: Círculos eliminados por el filtro

Además, se genera un desglose por tile (`spatial_tiles`) con:
- Posición y tamaño de cada tile
- Candidatos en tile, mantenidos y eliminados

---

## 7. Resumen de pipeline completo

```
Imagen
  ↓
Prior mask (support region)
  ↓
Grilla de centros (grid_step)
  ↓
Para cada centro: generar radios [rmin, rmin+rstep, ..., rmax]
  ↓
Si excede max_candidates → muestrear (tile_balanced / random / row_major)
  ↓
Para cada candidato:
  - Extraer patch circular (patch_size)
  - Calcular features (cortes, simetría, continuidad, etc.)
  - Construir vector de features
  ↓
Modelo binario → valid_score (0..1)
  ↓
Modelo multiclase → prob_valid, prob_crossing, prob_other
  ↓
Filtro combinado:
  (valid_score >= threshold_grupo) AND (prob_crossing <= crossing_threshold)
  ↓
Circle-NMS (eliminar duplicados)
  ↓
[Opcional] Filtro espacial final (tiles + max/tile + distancia mínima)
  ↓
Círculos aceptados finales
```

---

## 7. Estados de cada candidato

| Estado | Significado |
|--------|-------------|
| `accepted` | Pasó threshold binario, filtro multiclase, NMS y filtro espacial |
| `rejected (below_threshold)` | No alcanzó el threshold de score |
| `rejected (crossing_detected)` | Probabilidad de crossing excedió el umbral |
| `rejected (below_threshold\|crossing_detected)` | Ambas condiciones de rechazo |
| `removed_by_nms` | Era duplicado de otro círculo con mayor score |
| `removed_by_nms (circle_iou_nms)` | Eliminado por IoU circular en modo `circle_iou` |
| `removed_by_spatial` | Eliminado por el filtro espacial final (excedió max/tile o distancia mínima) |
| `rejected (empty_mask)` | El patch circular no contenía píxeles de support |

---

## 8. Ejemplo numérico (de la UI)

### Sin filtro espacial
```
Total candidatos: 11158   (grilla completa antes de muestreo)
Muestra: 11158            (no se redujo por max_candidates)
Modelo: 11154             (4 descartados por empty_mask)
Threshold: 25             (25 pasaron el threshold binario)
Final NMS: 20             (5 eliminados por NMS)
Rechazados: 11129         (below_threshold + crossing_detected + empty_mask)
Multiclase: ✓
crossing th: 0.50
```

De 11158 candidatos iniciales, solo **20** círculos fueron aceptados finalmente. La mayoría (11129) fueron rechazados por no alcanzar el threshold de score o por detección de cruce.

### Con filtro espacial activo
```
Total candidatos: 11158
Muestra: 11158
Modelo: 11154
Threshold: 25
Final NMS: 20
Final spatial: 12          (8 eliminados por filtro espacial)
Spatial removidos: 8
Multiclase: ✓
crossing th: 0.50
```

Con `spatial_final_tile_px=128` y `spatial_final_max_per_tile=3`, de 20 círculos post-NMS solo **12** pasan el filtro espacial, distribuidos uniformemente a lo largo de toda la imagen.

---

## 9. Referencia rápida de parámetros

| Parámetro | Default | Rango | Sección |
|-----------|---------|-------|---------|
| `grid_step` | 10 | 2–128 | Generación |
| `max_candidates` | 8000 (UI: 60000) | 1–60000 | Generación |
| `tile_size_px` | 128 | 32–2048 | Generación |
| `candidate_random_seed` | 42 | cualquier int | Generación |
| `min_radius` | 8 | 1–512 | Radios |
| `max_radius` | 32 | rmin–512 | Radios |
| `radius_step` | 4 | 0.5–rmax | Radios |
| `threshold` (general) | 0.89 | 0.01–0.99 | Threshold |
| `small_threshold` | 0.85 | 0.01–0.99 | Threshold |
| `medium_threshold` | 0.90 | 0.01–0.99 | Threshold |
| `large_threshold` | 0.95 | 0.01–0.99 | Threshold |
| `small_radius_limit` | 14 | cualquier float | Threshold |
| `large_radius_limit` | 24 | cualquier float | Threshold |
| `crossing_threshold` | 0.50 | 0.0–1.0 | Multiclase |
| `nms_mode` | circle_iou | distance_radius / circle_iou | NMS |
| `nms_distance_factor` | 0.5 | 0.05–3.0 | NMS |
| `radius_similarity_factor` | 0.4 | 0.0–3.0 | NMS |
| `circle_iou_threshold` | 0.4 | 0.0–1.0 | NMS |
| `use_spatial_final_filter` | false | bool | Filtro espacial |
| `spatial_final_tile_px` | 128 | 16–512 | Filtro espacial |
| `spatial_final_max_per_tile` | 3 | 1–50 | Filtro espacial |
| `spatial_final_min_center_distance_factor` | 1.0 | 0.0–5.0 | Filtro espacial |
| `patch_size` | 64 | 16–256 | Features |
| `return_rejected` | false | bool | Visualización |
| `max_return_rejected` | 800 | 0–8000 | Visualización |

---

## 10. Configuraciones predeterminadas (presets)

Estos presets permiten cambiar rápidamente entre perfiles de detección según el tipo de fibra y el nivel de exigencia. Cada preset ajusta **thresholds, radios, NMS y filtro espacial** en conjunto.

### 10.1 Fibras pequeñas (radio ~4–16 px)

| Parámetro | Relaxed | Balanced | Strict |
|-----------|---------|----------|--------|
| `min_radius` | 4 | 4 | 4 |
| `max_radius` | 16 | 16 | 16 |
| `radius_step` | 2 | 2 | 2 |
| `grid_step` | 6 | 8 | 10 |
| `threshold` | 0.80 | 0.85 | 0.90 |
| `small_threshold` | 0.75 | 0.80 | 0.88 |
| `medium_threshold` | 0.80 | 0.85 | 0.90 |
| `large_threshold` | 0.85 | 0.90 | 0.95 |
| `small_radius_limit` | 10 | 10 | 10 |
| `large_radius_limit` | 14 | 14 | 14 |
| `crossing_threshold` | 0.60 | 0.50 | 0.40 |
| `nms_mode` | circle_iou | circle_iou | circle_iou |
| `circle_iou_threshold` | 0.50 | 0.40 | 0.30 |
| `nms_distance_factor` | 0.60 | 0.50 | 0.40 |
| `use_spatial_final_filter` | false | true | true |
| `spatial_final_tile_px` | — | 64 | 64 |
| `spatial_final_max_per_tile` | — | 4 | 2 |
| `max_candidates` | 12000 | 8000 | 6000 |
| **Uso típico** | Exploración inicial, fibras muy finas y densas | Balance calidad/cobertura | Solo las fibras pequeñas más claras |

### 10.2 Fibras medianas (radio ~8–24 px)

| Parámetro | Relaxed | Balanced | Strict |
|-----------|---------|----------|--------|
| `min_radius` | 8 | 8 | 8 |
| `max_radius` | 24 | 24 | 24 |
| `radius_step` | 3 | 3 | 3 |
| `grid_step` | 8 | 10 | 12 |
| `threshold` | 0.80 | 0.88 | 0.93 |
| `small_threshold` | 0.75 | 0.83 | 0.90 |
| `medium_threshold` | 0.80 | 0.88 | 0.93 |
| `large_threshold` | 0.85 | 0.92 | 0.96 |
| `small_radius_limit` | 12 | 12 | 12 |
| `large_radius_limit` | 20 | 20 | 20 |
| `crossing_threshold` | 0.60 | 0.50 | 0.35 |
| `nms_mode` | circle_iou | circle_iou | circle_iou |
| `circle_iou_threshold` | 0.50 | 0.40 | 0.30 |
| `nms_distance_factor` | 0.60 | 0.50 | 0.40 |
| `use_spatial_final_filter` | false | true | true |
| `spatial_final_tile_px` | — | 128 | 128 |
| `spatial_final_max_per_tile` | — | 3 | 2 |
| `max_candidates` | 10000 | 8000 | 6000 |
| **Uso típico** | Capturar todas las fibras medias posibles | Balance estándar | Solo fibras medias de alta confianza |

### 10.3 Fibras grandes (radio ~16–40 px)

| Parámetro | Relaxed | Balanced | Strict |
|-----------|---------|----------|--------|
| `min_radius` | 16 | 16 | 16 |
| `max_radius` | 40 | 40 | 40 |
| `radius_step` | 4 | 4 | 4 |
| `grid_step` | 10 | 12 | 15 |
| `threshold` | 0.82 | 0.90 | 0.95 |
| `small_threshold` | 0.78 | 0.85 | 0.92 |
| `medium_threshold` | 0.82 | 0.90 | 0.95 |
| `large_threshold` | 0.88 | 0.93 | 0.97 |
| `small_radius_limit` | 20 | 20 | 20 |
| `large_radius_limit` | 30 | 30 | 30 |
| `crossing_threshold` | 0.55 | 0.45 | 0.30 |
| `nms_mode` | circle_iou | circle_iou | circle_iou |
| `circle_iou_threshold` | 0.55 | 0.45 | 0.35 |
| `nms_distance_factor` | 0.65 | 0.55 | 0.45 |
| `use_spatial_final_filter` | false | true | true |
| `spatial_final_tile_px` | — | 192 | 192 |
| `spatial_final_max_per_tile` | — | 3 | 2 |
| `max_candidates` | 8000 | 6000 | 4000 |
| **Uso típico** | Explorar fibras gruesas | Balance para fibras grandes | Solo fibras grandes muy definidas |

### 10.4 Fibras generales (rango completo ~4–40 px)

| Parámetro | Relaxed | Balanced | Strict |
|-----------|---------|----------|--------|
| `min_radius` | 4 | 6 | 8 |
| `max_radius` | 40 | 36 | 32 |
| `radius_step` | 4 | 4 | 4 |
| `grid_step` | 8 | 10 | 12 |
| `threshold` | 0.78 | 0.85 | 0.92 |
| `small_threshold` | 0.72 | 0.80 | 0.88 |
| `medium_threshold` | 0.78 | 0.85 | 0.92 |
| `large_threshold` | 0.85 | 0.90 | 0.95 |
| `small_radius_limit` | 12 | 14 | 14 |
| `large_radius_limit` | 24 | 24 | 24 |
| `crossing_threshold` | 0.60 | 0.50 | 0.35 |
| `nms_mode` | circle_iou | circle_iou | circle_iou |
| `circle_iou_threshold` | 0.50 | 0.40 | 0.30 |
| `nms_distance_factor` | 0.60 | 0.50 | 0.40 |
| `use_spatial_final_filter` | false | true | true |
| `spatial_final_tile_px` | — | 128 | 128 |
| `spatial_final_max_per_tile` | — | 3 | 2 |
| `max_candidates` | 12000 | 8000 | 6000 |
| **Uso típico** | Capturar todo, máxima cobertura | Balance general para la mayoría de imágenes | Solo círculos de alta confianza en toda la imagen |

### 10.5 Cómo usar los presets

1. Seleccionar el perfil según el tipo de fibra (pequeña / mediana / grande / general)
2. Elegir el nivel de estrictez:
   - **Relaxed**: Máxima cobertura, más falsos positivos. Ideal para exploración.
   - **Balanced**: Compromiso entre cobertura y precisión. Recomendado como punto de partida.
   - **Strict**: Alta precisión, menos círculos. Ideal para análisis final o publicaciones.
3. Ajustar manualmente si es necesario (ej. cambiar `grid_step` si la imagen es muy grande/muy pequeña)

### 10.6 Notas

- Los presets asumen `use_radius_thresholds: true` (threshold por grupo de radio)
- El `candidate_sampling_mode` recomendado es `tile_balanced` para todos los presets
- `return_rejected` se deja en `false` por defecto; activar solo para depuración
- Los valores de `spatial_final_tile_px` están ajustados al tamaño típico de fibra:
  - Fibras pequeñas → tile más pequeño (64 px) para distribución fina
  - Fibras medianas → tile medio (128 px)
  - Fibras grandes → tile más grande (192 px) porque hay menos círculos por área
---

## 11. Pipeline completo: LOCO Detector → Medición de diámetros

A partir de la reorganización UI v2, el LOCO Detector (Grupo 3) está conectado automáticamente con el panel de Diameter Research.

### 11.1 Flujo automático

1. **Detectar círculos** en el panel LOCO Detector (Grupo 3 → Detector)
2. **Medir aceptados**: Al hacer clic en "Medir aceptados", el backend ejecuta `loco_models_measure_accepted()` que:
   - Mide cada círculo aceptado con `loco_circle_probe`
   - Convierte los círculos aceptados a puntos de diameter research
   - Navega automáticamente al panel Diameter Research (Grupo 3 → Diameter)
3. **Revisar y medir** en el panel Diameter Research con todos los métodos disponibles

### 11.2 Calibración de escala (px → nm/µm)

El panel de Diameter Research incluye un panel de **Calibración** que permite:
- Activar/desactivar calibración
- Seleccionar unidad (nm/µm)
- Ingresar valor conocido (px) y distancia en píxeles
- Calcular automáticamente el factor nm_per_px
- Guardar/cargar/eliminar calibraciones por imagen (persistidas en `data/calibration/{image_id}.json`)

### 11.3 Distribución de diámetros

El panel de Diameter Research incluye un **Histograma SVG** (sin dependencias externas) que muestra:
- Distribución de diámetros con bins configurables (5-50)
- Estadísticas: media, mediana, desviación estándar, min, max, N
- Exportación a CSV
- Unidad configurable (px, nm, µm) según calibración activa
