# Plan Tutorial

## Resumen

Este documento es la hoja de ruta del sistema de tutoriales guiados del proyecto.
Su objetivo es permitir una implementacion gradual, supervisada y validable
por pestaña, antes de escribir el recorrido definitivo con `driver.js`.

La primera version del sistema tendra estas decisiones base:

- Se usara `driver.js` con progreso visible (`showProgress`).
- La entrada principal sera una pagina propia llamada `Tutorial`.
- Los recorridos se lanzaran manualmente desde esa pagina.
- Habra tutorial general, tutoriales por macroseccion y tutoriales por
  subpestaña.
- Los tutoriales padre podran encadenar los subtutoriales de su seccion.

El sistema debe cubrir las pestañas reales que hoy existen en la app:

- `Scribble`
  - `workbench`: Scribbles y Experimentos
    - `sub_workbench`: Carga de imagen
    - `sub_scribbleDrawing`: Dibujo de Scribble
  - `review`: Revision de Resultados
  - `models`: Modelos de Asistencia
- `LOCO`
  - `locoDataset`: Generar Dataset
  - `locoAugment`: Aumentacion
  - `locoTraining`: Entrenamiento
  - `locoTest`: Test de Modelo
- `Deteccion`
  - `locoModel`: Detector LOCO
  - `diameter`: Medicion de Diametros
- `Configuracion`
  - `projectTransfer`: Exportar e importar

## Objetivo del sistema de tutoriales

El sistema debe ayudar a un usuario nuevo o poco frecuente a entender:

1. Como esta organizado el proyecto.
2. Que hace cada macroseccion.
3. En que orden se suele trabajar.
4. Cuales son las acciones principales de cada subpestaña.
5. Como pasar de una etapa a otra sin perder contexto.

La primera version no debe intentar explicar todos los parametros finos del
producto. Debe priorizar orientacion, flujo de trabajo y acciones clave.

## Arquitectura esperada del tutorial

### Entrada principal

Se agregara un grupo de navegacion nuevo llamado `Tutorial`, con una pagina
principal que actuara como hub.

Esa pagina debe mostrar:

- tutorial general del proyecto
- tutoriales por macroseccion
- tutoriales por subpestaña
- descripcion corta de cada tutorial
- estado visible: no iniciado, en progreso, completado
- acciones: iniciar, continuar, reiniciar

### Modelo de tours

Cada tutorial tendra identidad propia y sera registrable de forma central.

Campos esperados para cada tutorial:

- `id`
- `title`
- `description`
- `scope`
  - `global`
  - `macro`
  - `subtab`
- `macroGroup`
- `workspaceTab`
- `prerequisites`
- `steps`
- `children`
- `estimatedSteps`

### Contrato tecnico minimo

Antes de escribir tours concretos, la app debe soportar:

- un registro central de tours
- anchors estables con `data-tour`
- helpers para navegar a la pestaña correcta antes de mostrar un paso
- reposicionamiento del scroll al tope de la ventana antes de cada paso
- reposicionamiento del scroll interno de paneles con `overflow: auto`, no solo `window.scrollTo`
- estado local de progreso en `localStorage`
- reinicio manual del progreso
- cancelacion segura si un paso no encuentra su target
- pasos manuales bloqueantes, donde `Siguiente` no avance hasta que ocurra una señal real del usuario

## Mapa de tutoriales

### 1. Tutorial general

#### `overview_project`

- Proposito:
  Explicar la estructura general del proyecto y como se conectan entrenamiento,
  evaluacion y uso operativo.
- Audiencia:
  Usuario nuevo o usuario que vuelve despues de tiempo.
- Prerequisitos:
  Ninguno.
- Pantallas:
  Navegacion principal, pestaña `Tutorial`, macrogrupos `Scribble`, `LOCO`,
  `Deteccion`, `Configuracion`.
- El usuario debe entender al terminar:
  - que `Scribble` prepara y revisa insumos
  - que `LOCO` construye dataset, entrenamientos y pruebas
  - que `Deteccion` usa modelos y herramientas operativas
  - que `Configuracion` mueve o empaqueta trabajo entre maquinas

### 2. Macroseccion Scribble

#### `macro_scribble`

- Proposito:
  Explicar el flujo general de trabajo dentro de `Scribble`.
- Audiencia:
  Usuario que va a anotar, revisar y usar apoyo visual.
- Prerequisitos:
  Haber visto `overview_project` o conocer la navegacion.
- Pantallas:
  `workbench`, `review`, `models`.
- El usuario debe entender al terminar:
  - donde se crean scribbles y experimentos
  - donde se revisan resultados
  - para que sirven los modelos de asistencia

#### `sub_workbench`

- Proposito:
  Enseñar la carga de imagenes locales dentro de `Scribbles y Experimentos`.
- Audiencia:
  Usuario que crea o edita scribbles.
- Prerequisitos:
  `macro_scribble` recomendado.
- Pantallas:
  `workbench`.
- El usuario debe entender al terminar:
  - como cargar una imagen
  - como listar los archivos de una carpeta
  - como seleccionar la imagen correcta antes de continuar
- Nota de implementacion:
  - la carga local del tutorial debe usar una ruta dinamica basada en la instalacion actual del proyecto, no una ruta hardcodeada
  - la ruta dinamica debe venir de backend, no de estado previo del frontend ni de `ruta inicial`
  - el endpoint fuente del tutorial debe ser el que usa el frontend real segun `VITE_API_BASE`; no asumir otro puerto
  - el flujo preferido es `Copiar ruta` + `Elegir directorio` con selector nativo del sistema
  - `Copiar ruta` debe poder avanzar automaticamente al siguiente paso cuando se complete bien
  - los pasos que abren ventanas externas del sistema deben pausar el tutorial y exigir la accion del usuario antes de permitir avanzar
  - durante el tutorial, `Elegir directorio` debe priorizar la ruta dinamica del tutorial por sobre cualquier `ruta inicial` vieja guardada
  - al volver del selector del sistema, el campo visible `ruta inicial` debe actualizarse de inmediato y limpiar cualquier lista local previa
  - la ayuda de ruta no debe vivir en la UI normal del programa; debe aparecer como ventana del tutorial
  - las ventanas flotantes del tutorial necesitan excepciones de `pointer-events` frente a `driver.js`, o sus botones quedan inutilizables
  - si la ruta dinamica no carga, el tutorial debe mostrar error explicito; no debe caer en una ruta vieja por fallback silencioso
  - los botones que el usuario debe presionar durante el tutorial necesitan señales verificables; no basta con describir la accion
  - el paso `Listar imagenes` debe exigir un click real y avanzar solo cuando el backend devuelva la lista
  - un selector nativo HTML no se puede abrir de forma portable con un click sintetico; durante el tutorial debe renderizarse temporalmente como lista expandida
  - la primera seleccion guiada es `bad-example.jpg`, para mostrar una carga incorrecta de manera explicita
  - luego el usuario debe seleccionar exactamente `overview-reference.png`; el tutorial valida el nombre antes de habilitar el cierre
  - cuando `overview-reference.png` termina de cargarse correctamente, el tutorial abre el panel interno `Scribble`
  - el cierre debe señalar el lienzo de dibujo, resumir la carga realizada y anticipar el tutorial `Dibujo de Scribble`

#### `sub_scribbleDrawing`

- Proposito:
  Enseñar correccion, limpieza y dibujo de clases dentro del panel interno `Scribble`.
- Audiencia:
  Usuario que crea o edita anotaciones Scribble.
- Prerequisitos:
  `sub_workbench` recomendado.
- Pantallas:
  `workbench`, panel interno `Scribble`.
- Flujo implementado:
  - el recorrido navega a `Scribbles y Experimentos`
  - activa el panel interno `Scribble`
  - advierte que reemplazara el scribble visible y el draft guardado de la imagen actual
  - genera una semilla determinista con trazos de `Fibra`, `Halo` y `Background`
  - explica `Auto`, `Undo`, `Redo`, zoom, reinicio, limpieza de exclusion y shortcuts de rueda como controles complementarios
  - exige usar `Goma`, agrandar y achicar pincel, y luego `Limpiar`
  - exige dibujar trazos reales de las tres clases y corregir uno con `Goma`
  - muestra un checklist dentro del panel y habilita `Continuar tutorial` solo al completar los hitos
  - cierra señalando el lienzo y anticipando la generacion de mascara
- Nota de implementacion:
  - la semilla se construye solo en frontend usando el canvas de labels
  - la validacion usa diferencias reales de pixeles etiquetados; cambiar de herramienta sin dibujar no cuenta
  - `Lapiz (L)`, `Mano (M)` y `Rectangulo de exclusion (R)` se explican como modos, pero no bloquean el cierre
  - `Limpiar` queda protegido hasta practicar `Goma` y el cambio de pincel, para no borrar la semilla antes de tiempo
  - las excepciones de `pointer-events` de Driver.js deben permitir interactuar con canvas, herramientas y checklist durante el recorrido

#### `sub_review`

- Proposito:
  Enseñar como revisar salidas e inspeccionar resultados.
- Audiencia:
  Usuario que valida resultados de experimentos.
- Prerequisitos:
  `sub_workbench` recomendado.
- Pantallas:
  `review`.
- El usuario debe entender al terminar:
  - como localizar resultados
  - como compararlos
  - que acciones de revision son las principales

#### `sub_models`

- Proposito:
  Enseñar el rol de `Modelos de Asistencia`.
- Audiencia:
  Usuario que usa modelos auxiliares en la etapa Scribble.
- Prerequisitos:
  `macro_scribble` recomendado.
- Pantallas:
  `models`.
- El usuario debe entender al terminar:
  - que modelos hay disponibles
  - como refrescarlos o seleccionarlos
  - como impactan otras vistas del flujo

### 3. Macroseccion LOCO

#### `macro_loco`

- Proposito:
  Explicar el pipeline de entrenamiento LOCO.
- Audiencia:
  Usuario que construye dataset o entrena modelos.
- Prerequisitos:
  `overview_project` recomendado.
- Pantallas:
  `locoDataset`, `locoAugment`, `locoTraining`, `locoTest`.
- El usuario debe entender al terminar:
  - que el flujo va desde dataset a test
  - como se encadenan las subpestañas
  - donde revisar cada resultado

#### `sub_locoDataset`

- Proposito:
  Enseñar como se genera el dataset LOCO.
- Audiencia:
  Usuario que prepara datos de entrenamiento.
- Prerequisitos:
  `macro_loco` recomendado.
- Pantallas:
  `locoDataset`.
- El usuario debe entender al terminar:
  - de donde salen los ejemplos
  - que controles principales afectan el dataset
  - donde se ve el estado generado

#### `sub_locoAugment`

- Proposito:
  Enseñar como armar y ejecutar bloques de aumentacion.
- Audiencia:
  Usuario que amplifica el dataset.
- Prerequisitos:
  `sub_locoDataset` recomendado.
- Pantallas:
  `locoAugment`.
- El usuario debe entender al terminar:
  - como agregar bloques
  - como ordenar la pipeline
  - que resultado produce la aumentacion

#### `sub_locoTraining`

- Proposito:
  Enseñar el flujo de entrenamiento y comparacion de modelos.
- Audiencia:
  Usuario que entrena o tunea modelos LOCO.
- Prerequisitos:
  `sub_locoDataset` recomendado.
- Pantallas:
  `locoTraining`.
- El usuario debe entender al terminar:
  - como lanzar entrenamiento
  - como funciona el batch
  - como leer el ranking y las metricas
  - donde aparece el tuning

#### `sub_locoTest`

- Proposito:
  Enseñar como validar modelos entrenados.
- Audiencia:
  Usuario que evalua calidad antes de pasar a produccion.
- Prerequisitos:
  `sub_locoTraining` recomendado.
- Pantallas:
  `locoTest`.
- El usuario debe entender al terminar:
  - como elegir modelo
  - como probarlo
  - que metricas o salidas revisar

### 4. Macroseccion Produccion / Deteccion

#### `macro_detection`

- Proposito:
  Explicar el uso operativo del sistema en deteccion, medicion y movimiento
  de proyecto.
- Audiencia:
  Usuario que usa modelos entrenados en una rutina de trabajo.
- Prerequisitos:
  `overview_project` recomendado.
- Pantallas:
  `locoModel`, `diameter`, `projectTransfer`.
- El usuario debe entender al terminar:
  - como usar el detector LOCO
  - como medir en la pestaña de diametros
  - como exportar o importar el proyecto

#### `sub_locoModel`

- Proposito:
  Enseñar el flujo base del `Detector LOCO`.
- Audiencia:
  Usuario operativo.
- Prerequisitos:
  `macro_detection` recomendado.
- Pantallas:
  `locoModel`.
- El usuario debe entender al terminar:
  - como elegir modelo y soporte
  - como correr base, threshold, NMS y spatial
  - como revisar aceptados y resultados

#### `sub_diameter`

- Proposito:
  Enseñar el uso base de `Medicion de Diametros`.
- Audiencia:
  Usuario que mide puntos, lineas y circulos.
- Prerequisitos:
  `macro_detection` recomendado.
- Pantallas:
  `diameter`.
- El usuario debe entender al terminar:
  - como elegir metodo
  - como dibujar o usar puntos
  - como correr mediciones
  - donde ver resultados y revision

#### `sub_projectTransfer`

- Proposito:
  Enseñar como exportar e importar proyecto entre maquinas.
- Audiencia:
  Usuario que comparte trabajo o despliega el proyecto.
- Prerequisitos:
  `macro_detection` recomendado.
- Pantallas:
  `projectTransfer`.
- El usuario debe entender al terminar:
  - como elegir categorias para exportar
  - como generar ZIP
  - como revisar conflictos al importar
  - como decidir si sobreescribe o conserva existentes

## Etapas de desarrollo

### Etapa 1: infraestructura base

Objetivo:
Preparar la arquitectura reusable antes de escribir recorridos concretos.

Incluye:

- instalar `driver.js`
- definir modulo central de registro de tours
- definir contrato de `data-tour`
- agregar grupo y pagina `Tutorial`
- agregar helpers para navegar entre pestañas desde un tour
- persistencia local simple de estado

Entregable esperado:

- pagina `Tutorial` visible
- tours arrancables desde codigo aunque todavia sean minimos
- anchors base en navegacion y contenedores principales

### Etapa 2: tutorial general

Objetivo:
Implementar `overview_project`.

Incluye:

- orientacion general del producto
- estructura de macrosecciones
- explicacion del flujo alto nivel

Entregable esperado:

- un tutorial estable y util para usuarios nuevos

### Etapa 3: macroseccion Scribble

Objetivo:
Implementar `macro_scribble` y sus subtutoriales.

Orden:

1. `macro_scribble`
2. `sub_workbench`
3. `sub_scribbleDrawing`
4. `sub_review`
5. `sub_models`

Entregable esperado:

- recorrido padre funcional
- cuatro subtours independientes

### Etapa 4: macroseccion LOCO

Objetivo:
Implementar `macro_loco` y sus subtutoriales.

Orden:

1. `macro_loco`
2. `sub_locoDataset`
3. `sub_locoAugment`
4. `sub_locoTraining`
5. `sub_locoTest`

Entregable esperado:

- recorrido completo del pipeline de entrenamiento

### Etapa 5: Produccion / Deteccion

Objetivo:
Implementar `macro_detection` y sus subtutoriales.

Orden:

1. `macro_detection`
2. `sub_locoModel`
3. `sub_diameter`
4. `sub_projectTransfer`

Entregable esperado:

- recorrido del uso operativo final

### Etapa 6: conexion y pulido

Objetivo:
Conectar los tutoriales y cerrar la experiencia completa.

Incluye:

- encadenado entre tours padre e hijos
- continuar recorrido incompleto
- reinicio de progreso
- estados visuales en el hub
- mensajes de error claros si falta un target

Entregable esperado:

- sistema integrado, navegable y entendible

## Template por tutorial

Cada tutorial que se implemente debe documentarse y desarrollarse con esta
estructura minima:

### Identidad

- `id`
- `titulo`
- `scope`
- `macroGroup`
- `workspaceTab`

### Objetivo

- que debe lograr el tutorial

### Audiencia

- para quien esta pensado

### Prerequisitos

- que deberia haber visto antes

### Pasos UI que debe cubrir

- lista secuencial de areas o acciones clave

### Elementos que necesitan `data-tour`

- lista de anchors necesarios

### Mensajes o ideas principales del popover

- que mensaje debe dejar cada bloque

### Criterios de aceptacion

- que debe poder verificar quien prueba el tutorial

### Observaciones de UX

- duracion esperada
- casos vacios
- si requiere scroll
- si cambia de pestaña automaticamente
- si necesita forzar `scrollTop = 0` antes del paso para evitar popovers fuera de vista
- si el paso depende de una ventana del sistema, confirmar que el tutorial deja claro que la accion es manual
- si hay overlays del tutorial, confirmar que no bloquean clicks necesarios del usuario

## Checklist de validacion

Este checklist se debe ejecutar en cada etapa y al cerrar cada tutorial:

- `npm run build`
- revisar que todos los anchors `data-tour` usados por el tutorial existen
- ejecutar el tour manualmente de inicio a fin
- confirmar navegacion correcta entre pestañas
- confirmar que la ventana vuelve arriba cuando un paso podria quedar invisible por scroll
- confirmar que paneles con scroll interno tambien vuelven arriba
- confirmar que no rompe si una tabla o panel esta vacio
- confirmar que el copy es claro y no excesivo
- confirmar que cancelar el tour deja la UI en estado usable
- confirmar que reiniciar progreso funciona si aplica
- confirmar que `Copiar ruta` usa la ruta dinamica actual y no una preferencia vieja
- confirmar que `Elegir directorio` funciona dentro del tutorial igual que fuera del tutorial
- confirmar que el selector del sistema actualiza `ruta inicial` al volver

## Checklist de supervision por pestaña

Para trabajar contigo pestaña a pestaña, cada entrega debe responder estas
preguntas:

1. Que tutorial se implemento.
2. Que anchors nuevos se agregaron.
3. Que recorrido exacto cubre.
4. Que no cubre todavia.
5. Que validacion manual se hizo.
6. Que ajustes de copy o UX quedaron pendientes.

## Criterios globales de aceptacion

El sistema de tutoriales se considerara listo en su primera version cuando:

- exista una pagina `Tutorial` usable como centro de entrada
- exista un tutorial general funcional
- cada macroseccion tenga su tour padre
- cada subpestaña listada en este plan tenga su subtour
- los tours se puedan lanzar de forma independiente
- los tours padre puedan encadenar subtours
- el progreso basico se vea en la UI del hub
- el sistema no dependa de datos cargados perfectos para no romper
- el tutorial no deje popovers apuntando a elementos invisibles por scroll acumulado

## Notas de implementacion

- Este archivo es un plan operativo, no documentacion final para usuarios.
- La implementacion debe hacerse por etapas pequenas y validables.
- Cada nueva pestaña o tutorial futuro debe agregarse a este documento antes
  de implementarse.
- `projectTransfer` queda incluido como parte del bloque operativo final.
