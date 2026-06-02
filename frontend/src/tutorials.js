export const TUTORIAL_STORAGE_KEY = 'locoTutorialProgress:v1'
export const SCRIBBLE_TUTORIAL_IMAGE_NAME = 'overview-reference.png'
export const SCRIBBLE_TUTORIAL_BAD_IMAGE_NAME = 'bad-example.jpg'

export const TUTORIAL_NAV_TABS = {
  tutorialOverview: {
    title: 'General',
    description: 'Vista general del proyecto y entrada general al sistema de tutoriales.',
    featured: 'overview_project',
    macroGroup: null,
  },
  tutorialScribble: {
    title: 'Ruta Scribble',
    description: 'Tutorial padre y subtutoriales de la macroseccion Scribble.',
    featured: 'macro_scribble',
    macroGroup: 'scribble',
  },
  tutorialLoco: {
    title: 'Ruta LOCO',
    description: 'Tutorial padre y subtutoriales del pipeline LOCO.',
    featured: 'macro_loco',
    macroGroup: 'loco',
  },
  tutorialDetection: {
    title: 'Ruta Produccion / Deteccion',
    description: 'Tutorial padre y subtutoriales del uso operativo final.',
    featured: 'macro_detection',
    macroGroup: 'detection',
  },
}

const SUBTAB_TUTORIAL_META = [
  {
    id: 'sub_workbench',
    title: 'Carga de imagen',
    description: 'Carga la carpeta de tutorial, revisa sus imagenes y abre la referencia correcta.',
    macroGroup: 'scribble',
    workspaceTab: 'workbench',
    purpose: 'Ensenar como cargar una imagen local desde una carpeta seleccionada.',
    audience: 'Usuarios que incorporan imagenes al flujo Scribble.',
    prerequisites: ['macro_scribble'],
    understands: [
      'Como elegir una carpeta local',
      'Como listar las imagenes disponibles',
      'Como seleccionar y cargar una imagen correcta',
    ],
  },
  {
    id: 'sub_scribbleDrawing',
    title: 'Dibujo de Scribble',
    description: 'Abre el editor de scribbles para comenzar el recorrido de anotacion.',
    macroGroup: 'scribble',
    workspaceTab: 'workbench',
    purpose: 'Dejar preparado el editor donde se dibujan los scribbles.',
    audience: 'Usuarios que crean anotaciones Scribble.',
    prerequisites: ['sub_workbench'],
    understands: [
      'Donde se abre el editor de scribbles',
      'Como llegar desde Scribbles y Experimentos al panel Scribble',
    ],
  },
  {
    id: 'sub_review',
    title: 'Revision de Resultados',
    description: 'Revisa corridas, compara resultados y marca estados.',
    macroGroup: 'scribble',
    workspaceTab: 'review',
    purpose: 'Explicar como inspeccionar resultados y tomar decisiones.',
    audience: 'Usuarios que revisan salidas de experimentos.',
    prerequisites: ['sub_workbench'],
    understands: [
      'Como encontrar corridas',
      'Como revisar estados y notas',
      'Donde estan las acciones de limpieza y exportacion',
    ],
  },
  {
    id: 'sub_models',
    title: 'Modelos de Asistencia',
    description: 'Gestiona modelos auxiliares usados durante la etapa Scribble.',
    macroGroup: 'scribble',
    workspaceTab: 'models',
    purpose: 'Explicar el rol de los modelos auxiliares y su uso.',
    audience: 'Usuarios que entrenan o aplican modelos de asistencia.',
    prerequisites: ['macro_scribble'],
    understands: [
      'Como refrescar modelos',
      'Como usar un modelo por defecto',
      'Donde impactan estos modelos en el flujo',
    ],
  },
  {
    id: 'sub_locoDataset',
    title: 'Generar Dataset',
    description: 'Prepara el dataset LOCO a partir de ejemplos seleccionados.',
    macroGroup: 'loco',
    workspaceTab: 'locoDataset',
    purpose: 'Explicar la generacion y revision del dataset base.',
    audience: 'Usuarios que preparan datos para entrenamiento.',
    prerequisites: ['macro_loco'],
    understands: [
      'De donde salen los ejemplos',
      'Como generar features o dataset',
      'Donde revisar el estado generado',
    ],
  },
  {
    id: 'sub_locoAugment',
    title: 'Aumentacion',
    description: 'Arma y ejecuta bloques de aumentacion sobre el dataset LOCO.',
    macroGroup: 'loco',
    workspaceTab: 'locoAugment',
    purpose: 'Explicar como construir pipelines de aumentacion.',
    audience: 'Usuarios que expanden el dataset.',
    prerequisites: ['sub_locoDataset'],
    understands: [
      'Como agregar bloques',
      'Como ordenar la pipeline',
      'Como aplicar la aumentacion al dataset',
    ],
  },
  {
    id: 'sub_locoTraining',
    title: 'Entrenamiento',
    description: 'Lanza entrenamientos, compara modelos y usa tuning.',
    macroGroup: 'loco',
    workspaceTab: 'locoTraining',
    purpose: 'Ensenar el pipeline de entrenamiento y ranking de modelos.',
    audience: 'Usuarios que entrenan o tunean modelos LOCO.',
    prerequisites: ['sub_locoDataset'],
    understands: [
      'Como lanzar entrenamientos',
      'Como funciona el batch',
      'Como leer ranking, metricas y tuning',
    ],
  },
  {
    id: 'sub_locoTest',
    title: 'Test de Modelo',
    description: 'Evalua modelos entrenados sobre imagenes o circulos de prueba.',
    macroGroup: 'loco',
    workspaceTab: 'locoTest',
    purpose: 'Explicar la validacion previa a produccion.',
    audience: 'Usuarios que comparan calidad de modelos entrenados.',
    prerequisites: ['sub_locoTraining'],
    understands: [
      'Como elegir un run o modelo',
      'Como ejecutar la prediccion',
      'Que resultados mirar antes de pasar a produccion',
    ],
  },
  {
    id: 'sub_locoModel',
    title: 'Detector LOCO',
    description: 'Usa modelos entrenados para detectar candidatos y filtrarlos.',
    macroGroup: 'detection',
    workspaceTab: 'locoModel',
    purpose: 'Explicar el flujo operativo del detector LOCO.',
    audience: 'Usuarios operativos.',
    prerequisites: ['macro_detection'],
    understands: [
      'Como elegir modelo y soporte',
      'Como correr base, threshold, NMS y spatial',
      'Donde quedan aceptados y resultados',
    ],
  },
  {
    id: 'sub_diameter',
    title: 'Medicion de Diametros',
    description: 'Mide diametros usando puntos, lineas, circulos y metodos guiados.',
    macroGroup: 'detection',
    workspaceTab: 'diameter',
    purpose: 'Explicar la medicion operativa de diametros.',
    audience: 'Usuarios que miden fibras o revisan resultados.',
    prerequisites: ['macro_detection'],
    understands: [
      'Como elegir metodo',
      'Como dibujar geometria o usar puntos',
      'Donde revisar resultados y exportes',
    ],
  },
  {
    id: 'sub_projectTransfer',
    title: 'Exportar e importar',
    description: 'Transfiere el proyecto de entrenamiento entre maquinas con ZIP.',
    macroGroup: 'configuration',
    workspaceTab: 'projectTransfer',
    purpose: 'Explicar exportacion e importacion de artefactos pesados.',
    audience: 'Usuarios que migran o comparten el proyecto.',
    prerequisites: ['macro_detection'],
    understands: [
      'Como seleccionar categorias',
      'Como generar el ZIP',
      'Como revisar e importar sin sobrescribir por error',
    ],
  },
]

function buildSubtabSteps(meta) {
  if (meta.id === 'sub_workbench') {
    return [
      {
        selector: `[data-tour="sidebar-group-${meta.macroGroup}"]`,
        title: meta.title,
        description: `Este tutorial pertenece a la macroseccion ${macroLabel(meta.macroGroup)}.`,
      },
      {
        selector: `[data-tour="tab-${meta.workspaceTab}"]`,
        group: meta.macroGroup,
        tab: meta.workspaceTab,
        title: 'Subpestaña',
        description: meta.description,
      },
      {
        selector: '[data-tour="workspace-panel-workbench"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Preparacion de ruta',
        description: 'Usa la ventana del tutorial para copiar la ruta sugerida. Al copiarla, el recorrido avanzara automaticamente al selector de directorio.',
        tutorialDialog: { type: 'pathCopy' },
        autoAction: { type: 'setWorkbenchPanelTab', tab: 'image' },
        waitForSignal: 'copiedTutorialPath',
        waitForMessage: 'Primero usa el boton Copiar ruta y luego continua.',
      },
      {
        selector: '[data-tour="local-image-choose-dir"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Elegir directorio',
        description: 'Haz click aqui. En la ventana del sistema, pega la ruta copiada y presiona Aceptar. Cuando regreses y la ruta inicial aparezca cargada, pulsa Siguiente.',
        waitForSignal: 'localDirectoryChosen',
        waitForMessage: 'Selecciona la carpeta desde la ventana del sistema antes de continuar.',
      },
      {
        selector: '[data-tour="local-image-start-dir"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Ruta inicial',
        description: 'Este campo mostrara la carpeta elegida desde el selector del sistema.',
      },
      {
        selector: '[data-tour="local-image-save-dir"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Guardar ruta',
        description: 'Aqui se guarda la ruta inicial para reutilizarla en el flujo local.',
        autoAction: { type: 'saveLocalImagePrefs' },
      },
      {
        selector: '[data-tour="local-image-list"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Listar imagenes',
        description: 'Presiona Listar imagenes para buscar los archivos disponibles en esa carpeta.',
        waitForSignal: 'localImagesListed',
        waitForMessage: 'Presiona Listar imagenes antes de continuar.',
      },
      {
        selector: '[data-tour="local-image-select"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Imagenes disponibles',
        description: `La lista se abre para mostrar los archivos encontrados. Primero cargaremos ${SCRIBBLE_TUTORIAL_BAD_IMAGE_NAME} como ejemplo incorrecto.`,
        autoAction: { type: 'selectLocalImageByName', fileName: SCRIBBLE_TUTORIAL_BAD_IMAGE_NAME, expand: true },
      },
      {
        selector: '[data-tour="local-image-load"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Cargar ejemplo incorrecto',
        description: `Presiona Cargar seleccion local para abrir ${SCRIBBLE_TUTORIAL_BAD_IMAGE_NAME}. Veras por que no es la referencia que necesitamos.`,
        waitForSignal: 'badTutorialImageLoaded',
        waitForMessage: `Carga ${SCRIBBLE_TUTORIAL_BAD_IMAGE_NAME} antes de continuar.`,
      },
      {
        selector: '[data-tour="local-image-select"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Elegir imagen correcta',
        description: `Ahora selecciona ${SCRIBBLE_TUTORIAL_IMAGE_NAME} en la lista. El recorrido avanzara cuando elijas exactamente ese archivo.`,
        autoAction: { type: 'expandLocalImageSelect' },
        waitForSignal: 'correctTutorialImageSelected',
        waitForMessage: `Selecciona ${SCRIBBLE_TUTORIAL_IMAGE_NAME} antes de continuar.`,
      },
      {
        selector: '[data-tour="local-image-load"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Cargar imagen correcta',
        description: `Presiona Cargar seleccion local para abrir ${SCRIBBLE_TUTORIAL_IMAGE_NAME}.`,
        waitForSignal: 'correctTutorialImageLoaded',
        waitForMessage: `Carga ${SCRIBBLE_TUTORIAL_IMAGE_NAME} antes de continuar.`,
      },
      {
        selector: '[data-tour="scribble-drawing-area"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Carga de imagen completada',
        description: 'La imagen correcta ya esta cargada. Aqui dibujaras el scribble que se usara para obtener la mascara. En este tutorial elegiste la carpeta, listaste los archivos, revisaste un ejemplo incorrecto y cargaste la referencia correcta. El siguiente recorrido es Dibujo de Scribble.',
        autoAction: { type: 'setWorkbenchPanelTab', tab: 'editor' },
      },
    ]
  }
  if (meta.id === 'sub_scribbleDrawing') {
    return [
      {
        selector: `[data-tour="sidebar-group-${meta.macroGroup}"]`,
        title: meta.title,
        description: `Este tutorial pertenece a la macroseccion ${macroLabel(meta.macroGroup)}.`,
      },
      {
        selector: `[data-tour="tab-${meta.workspaceTab}"]`,
        group: meta.macroGroup,
        tab: meta.workspaceTab,
        title: 'Scribbles y Experimentos',
        description: 'El dibujo se realiza dentro de Scribbles y Experimentos.',
      },
      {
        selector: '[data-tour="workbench-panel-tab-editor"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Scribble',
        description: 'Esta es la pestaña interna del editor donde crearas el scribble.',
        autoAction: { type: 'setWorkbenchPanelTab', tab: 'editor' },
      },
      {
        selector: '[data-tour="scribble-drawing-area"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Preparar ejercicio',
        description: 'El ejercicio reemplazara el scribble visible y agregara trazos de ejemplo. Confirma el inicio desde la ventana del tutorial.',
        tutorialDialog: { type: 'scribbleSeedConfirm' },
        autoAction: { type: 'setWorkbenchPanelTab', tab: 'editor' },
        waitForSignal: 'scribbleSeedApplied',
        waitForMessage: 'Usa Empezar tutorial para preparar los trazos de ejemplo.',
      },
      {
        selector: '[data-tour="scribble-mode-tools"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Modos del editor',
        description: 'Lapiz (L) dibuja, Mano (M) desplaza la vista y Rectangulo de exclusion (R) marca una zona que no quieres considerar.',
      },
      {
        selector: '[data-tour="scribble-drawing-area"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Controles complementarios',
        description: 'El editor tambien incluye Auto para escalar el pincel, Undo y Redo, zoom con - y +, Reiniciar, Limpiar exclusion, Ctrl + rueda para zoom y Alt + rueda para cambiar el pincel.',
      },
      {
        selector: '[data-tour="scribble-drawing-area"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Trazos de ejemplo',
        description: 'El tutorial agrego trazos de Fibra, Halo y Background. En los pasos siguientes practicaras como corregirlos y limpiar el lienzo.',
      },
      {
        selector: '[data-tour="scribble-tool-erase"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Corregir con Goma',
        description: 'Selecciona Goma (G) y borra una parte visible de los trazos de ejemplo.',
        waitForSignal: 'scribbleSeedErased',
        waitForMessage: 'Usa Goma sobre uno de los trazos de ejemplo antes de continuar.',
      },
      {
        selector: '[data-tour="scribble-brush-control"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Tamano del pincel',
        description: 'Agranda y luego achica el pincel con el control. Tambien puedes usar Alt + rueda sobre la imagen.',
        waitForSignal: 'scribbleBrushAdjusted',
        waitForMessage: 'Agranda y luego achica el pincel antes de continuar.',
      },
      {
        selector: '[data-tour="scribble-clear"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Limpiar lienzo',
        description: 'Presiona Limpiar para retirar la semilla y comenzar tu propio scribble.',
        waitForSignal: 'scribbleSeedCleared',
        waitForMessage: 'Presiona Limpiar antes de continuar.',
      },
      {
        selector: '[data-tour="scribble-paint-tools"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Dibujar clases',
        description: 'Dibuja algunos tramos con Fibra (F), Halo (H) y Background (B). Finalmente usa Goma (G) para corregir parte de uno de tus trazos.',
      },
      {
        selector: '[data-tour="scribble-tutorial-exercise"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Completar ejercicio',
        description: 'El checklist registra trazos reales sobre el lienzo. Cuando todos los hitos esten listos, presiona Continuar tutorial.',
        waitForSignal: 'scribbleExerciseContinue',
        waitForMessage: 'Completa el checklist y presiona Continuar tutorial.',
      },
      {
        selector: '[data-tour="scribble-drawing-area"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Dibujo de Scribble completado',
        description: 'Practicaste correccion con Goma, ajuste de pincel, limpieza y trazos de Fibra, Halo y Background. Ya dejaste un scribble base utilizable. El siguiente paso sera usarlo para obtener la mascara.',
      },
    ]
  }
  return [
    {
      selector: `[data-tour="sidebar-group-${meta.macroGroup}"]`,
      title: meta.title,
      description: `Este tutorial pertenece a la macroseccion ${macroLabel(meta.macroGroup)}.`,
    },
    {
      selector: `[data-tour="tab-${meta.workspaceTab}"]`,
      group: meta.macroGroup,
      tab: meta.workspaceTab,
      title: 'Subpestaña',
      description: meta.description,
    },
    {
      selector: `[data-tour="workspace-panel-${meta.workspaceTab}"]`,
      group: meta.macroGroup,
      tab: meta.workspaceTab,
      title: 'Panel principal',
      description: meta.purpose,
    },
  ]
}

function macroLabel(group) {
  if (group === 'scribble') return 'Scribble'
  if (group === 'loco') return 'LOCO'
  if (group === 'detection') return 'Deteccion'
  if (group === 'configuration') return 'Configuracion'
  return group
}

const SUBTAB_TUTORIALS = SUBTAB_TUTORIAL_META.map((meta) => ({
  ...meta,
  scope: 'subtab',
  children: [],
  steps: buildSubtabSteps(meta),
  estimatedSteps: buildSubtabSteps(meta).length,
}))

const MACRO_TUTORIALS = [
  {
    id: 'macro_scribble',
    title: 'Ruta Scribble',
    description: 'Explica la parte de anotacion, revision y modelos de asistencia.',
    scope: 'macro',
    macroGroup: 'scribble',
    workspaceTab: 'workbench',
    purpose: 'Dar contexto del flujo general de la macroseccion Scribble.',
    audience: 'Usuarios que parten generando insumos o revisando resultados.',
    prerequisites: ['overview_project'],
    understands: [
      'Que se hace en Scribble',
      'Como se relacionan las subpestanas workbench, review y models',
      'En que orden conviene recorrerlas',
    ],
    children: ['sub_workbench', 'sub_scribbleDrawing', 'sub_review', 'sub_models'],
    estimatedSteps: 5,
    steps: [
      {
        selector: '[data-tour="sidebar-group-scribble"]',
        title: 'Macroseccion Scribble',
        description: 'Aqui parte el flujo de entrenamiento inicial basado en imagenes, scribbles y resultados revisados.',
      },
      {
        selector: '[data-tour="tab-workbench"]',
        group: 'scribble',
        tab: 'workbench',
        title: 'Scribbles y Experimentos',
        description: 'Esta subpestana es el punto de entrada para cargar imagenes, crear scribbles y ejecutar experimentos.',
      },
      {
        selector: '[data-tour="tab-review"]',
        group: 'scribble',
        tab: 'review',
        title: 'Revision de Resultados',
        description: 'Aqui revisas corridas, comparas resultados y marcas decisiones sobre las salidas del flujo.',
      },
      {
        selector: '[data-tour="tab-models"]',
        group: 'scribble',
        tab: 'models',
        title: 'Modelos de Asistencia',
        description: 'Esta subpestana concentra modelos auxiliares usados para apoyar la etapa Scribble.',
      },
      {
        selector: '[data-tour="workspace-panel-models"]',
        group: 'scribble',
        tab: 'models',
        title: 'Recorrido padre completado',
        description: 'El tutorial padre muestra la ruta general. Los subtutoriales explican cada subpestana en detalle.',
      },
    ],
  },
  {
    id: 'macro_loco',
    title: 'Ruta LOCO',
    description: 'Explica dataset, aumentacion, entrenamiento y test del pipeline LOCO.',
    scope: 'macro',
    macroGroup: 'loco',
    workspaceTab: 'locoDataset',
    purpose: 'Dar contexto del pipeline de entrenamiento LOCO de punta a punta.',
    audience: 'Usuarios que preparan datos y entrenan modelos.',
    prerequisites: ['overview_project'],
    understands: [
      'Como se ordena el pipeline dataset -> augment -> training -> test',
      'Que entregable deja cada subpestana',
      'Donde entra el tuning dentro del flujo',
    ],
    children: ['sub_locoDataset', 'sub_locoAugment', 'sub_locoTraining', 'sub_locoTest'],
    estimatedSteps: 6,
    steps: [
      {
        selector: '[data-tour="sidebar-group-loco"]',
        title: 'Macroseccion LOCO',
        description: 'Esta parte convierte insumos revisados en dataset, modelos y validaciones.',
      },
      {
        selector: '[data-tour="tab-locoDataset"]',
        group: 'loco',
        tab: 'locoDataset',
        title: 'Generar Dataset',
        description: 'Aqui parte el pipeline LOCO, preparando el dataset base para entrenamiento.',
      },
      {
        selector: '[data-tour="tab-locoAugment"]',
        group: 'loco',
        tab: 'locoAugment',
        title: 'Aumentacion',
        description: 'Esta subpestana amplifica el dataset con bloques de transformacion configurables.',
      },
      {
        selector: '[data-tour="tab-locoTraining"]',
        group: 'loco',
        tab: 'locoTraining',
        title: 'Entrenamiento',
        description: 'Aqui se entrenan modelos, se comparan metricas y se ejecuta tuning.',
      },
      {
        selector: '[data-tour="tab-locoTest"]',
        group: 'loco',
        tab: 'locoTest',
        title: 'Test de Modelo',
        description: 'Esta subpestana permite validar los modelos entrenados antes de pasar a produccion.',
      },
      {
        selector: '[data-tour="workspace-panel-locoTest"]',
        group: 'loco',
        tab: 'locoTest',
        title: 'Recorrido padre completado',
        description: 'El tutorial padre muestra el pipeline completo. Los subtutoriales desarrollan cada etapa con mayor detalle.',
      },
    ],
  },
  {
    id: 'macro_detection',
    title: 'Ruta Produccion / Deteccion',
    description: 'Explica detector, medicion y movimiento de proyecto entre maquinas.',
    scope: 'macro',
    macroGroup: 'detection',
    workspaceTab: 'locoModel',
    purpose: 'Dar contexto del uso operativo final del sistema.',
    audience: 'Usuarios que usan modelos entrenados en trabajo real.',
    prerequisites: ['overview_project'],
    understands: [
      'Como detectar candidatos con LOCO',
      'Como medir diametros',
      'Como mover el proyecto entre maquinas',
    ],
    children: ['sub_locoModel', 'sub_diameter', 'sub_projectTransfer'],
    estimatedSteps: 5,
    steps: [
      {
        selector: '[data-tour="sidebar-group-detection"]',
        title: 'Macroseccion de produccion',
        description: 'Aqui se usan modelos y herramientas operativas sobre imagenes reales.',
      },
      {
        selector: '[data-tour="tab-locoModel"]',
        group: 'detection',
        tab: 'locoModel',
        title: 'Detector LOCO',
        description: 'Esta subpestana ejecuta el flujo de deteccion y filtrado usando modelos entrenados.',
      },
      {
        selector: '[data-tour="tab-diameter"]',
        group: 'detection',
        tab: 'diameter',
        title: 'Medicion de Diametros',
        description: 'Aqui se mide y revisa el resultado geometrico y metrico sobre fibras detectadas o puntos cargados.',
      },
      {
        selector: '[data-tour="sidebar-group-configuration"]',
        title: 'Configuracion',
        description: 'La transferencia del proyecto vive en Configuracion y forma parte del recorrido operativo final.',
      },
      {
        selector: '[data-tour="workspace-panel-projectTransfer"]',
        group: 'configuration',
        tab: 'projectTransfer',
        title: 'Recorrido padre completado',
        description: 'El recorrido operativo termina en exportacion e importacion. Los subtutoriales muestran cada etapa con mas detalle.',
      },
    ],
  },
]

const OVERVIEW_TUTORIAL = {
  id: 'overview_project',
  title: 'Vista general del proyecto',
  description: 'Recorrido general por la estructura de la app y la relacion entre secciones.',
  scope: 'global',
  macroGroup: 'tutorial',
  workspaceTab: 'tutorialHub',
  purpose: 'Explicar como esta organizado el proyecto y por que existen sus macrosecciones.',
  audience: 'Usuarios nuevos o que retoman el proyecto.',
  prerequisites: [],
  understands: [
    'Que rol cumple cada macroseccion',
    'Como se conecta el flujo desde Scribble hasta Deteccion',
    'Donde lanzar los tutoriales especializados',
  ],
  children: ['macro_scribble', 'macro_loco', 'macro_detection'],
  estimatedSteps: 10,
  steps: [
    {
      selector: '[data-tour="workspace-panel-tutorialHub"]',
      group: 'tutorial',
      tab: 'tutorialOverview',
      title: 'Centro de tutoriales',
      description: 'Desde aqui se lanzan todos los recorridos del proyecto.',
    },
    {
      selector: '[data-tour="sidebar-group-scribble"]',
      title: 'Scribble',
      description: 'Primer bloque del proyecto: imagenes, scribbles, resultados y modelos de asistencia.',
    },
    {
      selector: '[data-tour="sidebar-group-loco"]',
      title: 'LOCO',
      description: 'Segundo bloque: dataset, aumentacion, entrenamiento y test.',
    },
    {
      selector: '[data-tour="sidebar-group-detection"]',
      title: 'Deteccion',
      description: 'Tercer bloque: uso de modelos entrenados y herramientas operativas.',
    },
    {
      selector: '[data-tour="sidebar-group-configuration"]',
      title: 'Configuracion',
      description: 'Aqui vive la transferencia del proyecto entre maquinas.',
    },
    {
      selector: '[data-tour="tab-tutorialOverview"]',
      group: 'tutorial',
      tab: 'tutorialOverview',
      title: 'Tutorial: General',
      description: 'Esta subpestana concentra el overview del proyecto y sirve como punto de entrada general.',
    },
    {
      selector: '[data-tour="tab-tutorialScribble"]',
      group: 'tutorial',
      tab: 'tutorialScribble',
      title: 'Tutorial: Ruta Scribble',
      description: 'Aqui se agrupan el tutorial padre de Scribble y sus subrecorridos detallados.',
    },
    {
      selector: '[data-tour="tab-tutorialLoco"]',
      group: 'tutorial',
      tab: 'tutorialLoco',
      title: 'Tutorial: Ruta LOCO',
      description: 'Aqui se concentran los recorridos del pipeline LOCO: dataset, aumentacion, entrenamiento y test.',
    },
    {
      selector: '[data-tour="tab-tutorialDetection"]',
      group: 'tutorial',
      tab: 'tutorialDetection',
      title: 'Tutorial: Ruta Produccion / Deteccion',
      description: 'Esta subpestana reune el recorrido operativo de detector, medicion y transferencia de proyecto.',
    },
    {
      selector: '[data-tour="tutorial-card-overview_project"]',
      group: 'tutorial',
      tab: 'tutorialOverview',
      title: 'Siguiente paso',
      description: 'Despues del overview puedes lanzar una macroseccion completa o entrar directo a una subpestana.',
    },
  ],
}

export const TUTORIALS = [
  OVERVIEW_TUTORIAL,
  ...MACRO_TUTORIALS,
  ...SUBTAB_TUTORIALS,
]

export function getTutorialById(id) {
  return TUTORIALS.find((item) => item.id === id) || null
}

export function getTutorialsByScope(scope) {
  return TUTORIALS.filter((item) => item.scope === scope)
}

export function getChainedTutorialIds(tutorialId, includeChildren = false) {
  const tutorial = getTutorialById(tutorialId)
  if (!tutorial) return []
  if (!includeChildren || !tutorial.children?.length) return [tutorial.id]
  return [tutorial.id, ...tutorial.children]
}
