import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react'
import Navigation, { legacyToGroup, groupToLegacy } from './components/Navigation'
import Histogram, { exportHistogramCsv } from './components/Histogram'
import CalibrationPanel from './components/CalibrationPanel'
import { apiForm, apiGet, apiPost, b64ToDataUrl } from './api'

function errMsg(err) {
  if (err instanceof Event) return 'No se pudo renderizar la imagen en el navegador.'
  if (typeof err === 'object' && err && typeof err.type === 'string') return 'No se pudo renderizar la imagen en el navegador.'
  if (err?.payload?.detail && Array.isArray(err.payload.detail)) {
    return err.payload.detail.map((d) => d.msg || String(d)).join('; ')
  }
  return String(err?.message || err || 'Error inesperado')
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v))
}

function clampOnBlur(key, lo, hi, step, defaultValue) {
  return (e) => {
    const raw = Number(e.target.value)
    if (isNaN(raw) || e.target.value === '') {
      updateLocoModelParam(key, defaultValue)
      return
    }
    let clamped = clamp(raw, lo, hi)
    if (step && step > 0) {
      clamped = Math.round(clamped / step) * step
      clamped = clamp(clamped, lo, hi)
    }
    updateLocoModelParam(key, clamped)
  }
}

function brushPxToSlider(px) {
  const v = clamp(Math.round(Number(px) || 1), 1, 120)
  if (v <= 10) return Math.round(1 + ((v - 1) * 54) / 9)
  return Math.round(55 + ((v - 10) * 65) / 110)
}

function brushSliderToPx(sliderValue) {
  const v = clamp(Math.round(Number(sliderValue) || 1), 1, 120)
  if (v <= 55) return clamp(Math.round(1 + ((v - 1) * 9) / 54), 1, 10)
  return clamp(Math.round(10 + ((v - 55) * 110) / 65), 11, 120)
}

function normalizeRectFromPoints(a, b) {
  const x0 = Math.min(Number(a?.x || 0), Number(b?.x || 0))
  const y0 = Math.min(Number(a?.y || 0), Number(b?.y || 0))
  const x1 = Math.max(Number(a?.x || 0), Number(b?.x || 0))
  const y1 = Math.max(Number(a?.y || 0), Number(b?.y || 0))
  return {
    x: Math.round(x0),
    y: Math.round(y0),
    w: Math.max(1, Math.round(x1 - x0)),
    h: Math.max(1, Math.round(y1 - y0)),
  }
}

function rectRoughEqual(a, b, tol = 1.01) {
  if (!a && !b) return true
  if (!a || !b) return false
  const ax = Number(a.x || 0); const ay = Number(a.y || 0); const aw = Number(a.w || 0); const ah = Number(a.h || 0)
  const bx = Number(b.x || 0); const by = Number(b.y || 0); const bw = Number(b.w || 0); const bh = Number(b.h || 0)
  return (
    Math.abs(ax - bx) <= tol &&
    Math.abs(ay - by) <= tol &&
    Math.abs(aw - bw) <= tol &&
    Math.abs(ah - bh) <= tol
  )
}

function emptyLoading() {
  return {
    boot: false,
    loadImage: false,
    libraryList: false,
    libraryLoad: false,
    libraryDelete: false,
    localPrefs: false,
    localImages: false,
    localLoad: false,
    openFolder: false,
    run: false,
    runBatch: false,
    listResults: false,
    clearResults: false,
    getResult: false,
    mark: false,
    export: false,
    diamPoints: false,
    diamRun: false,
    diamList: false,
    diamGet: false,
    diamExport: false,
    locoPreview: false,
    locoRun: false,
    locoLab: false,
    locoDataset: false,
    locoAugment: false,
    locoTraining: false,
    locoTest: false,
    locoModel: false,
    validationList: false,
    validationSave: false,
    validationRun: false,
    validationExport: false,
    modelsDataset: false,
    modelsList: false,
    modelsTrain: false,
    modelsPredict: false,
    modelsDelete: false,
  }
}

function ToolIcon({ name }) {
  if (name === 'hand') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M8 12V6.8a1.2 1.2 0 0 1 2.4 0V11" />
        <path d="M10.4 11V5.8a1.2 1.2 0 0 1 2.4 0V11" />
        <path d="M12.8 11V7a1.2 1.2 0 0 1 2.4 0v5" />
        <path d="M15.2 12.4v-2a1.2 1.2 0 0 1 2.4 0v4.2c0 3.2-2 5.4-5.2 5.4h-1.1c-1.7 0-2.8-.6-3.8-1.9l-2.2-2.9a1.3 1.3 0 0 1 2-1.6l1.1 1.1" />
      </svg>
    )
  }
  if (name === 'exclude') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M5 5h14v14H5z" />
        <path d="M7 17 17 7" />
      </svg>
    )
  }
  if (name === 'erase') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="m7 15 7-7a2 2 0 0 1 2.8 0l1.2 1.2a2 2 0 0 1 0 2.8l-7 7" />
        <path d="M5 17 3.8 15.8a2 2 0 0 1 0-2.8L7 9.8 14.2 17H5z" />
        <path d="M12 19h8" />
      </svg>
    )
  }
  if (name === 'circle') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="12" r="7" />
        <path d="M12 5v3M12 16v3M5 12h3M16 12h3" />
      </svg>
    )
  }
  if (name === 'center') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="12" r="2" />
        <path d="M12 3v5M12 16v5M3 12h5M16 12h5" />
      </svg>
    )
  }
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m14.5 4.5 5 5L9 20H4v-5L14.5 4.5z" />
      <path d="m13 6 5 5" />
    </svg>
  )
}

function buildRunRow(item) {
  const meta = item?.meta || {}
  const exp = meta?.experiment || {}
  const paramsEff = meta?.params_effective || {}
  return {
    run_id: item?.run_id || '',
    experiment_id: item?.experiment_id || exp?.experiment_id || '',
    created_at: item?.created_at || '',
    group: item?.group || exp?.group || '',
    display_name: item?.display_name || exp?.display_name || '',
    implementation_status: item?.implementation_status || exp?.implementation_status || '',
    profile_name: item?.batch_profile || paramsEff?.__profile_name || '',
    run_status_level: item?.run_status_level || meta?.run_status_level || 'success',
  }
}

const REVIEW_TIERS = [
  { value: 's', label: 'S excelente', short: 'S' },
  { value: 'a', label: 'A bueno', short: 'A' },
  { value: 'b', label: 'B usable', short: 'B' },
  { value: 'c', label: 'C debil', short: 'C' },
  { value: 'unusable', label: 'Inutilizable', short: 'X' },
]

const REVIEW_RANK = { s: 5, a: 4, ok: 4, b: 3, c: 2, bad: 1, unusable: 0 }

const LOCO_STEPS = [
  { key: 'image', label: 'Imagen' },
  { key: 'center', label: 'Centro' },
  { key: 'circle', label: 'Circulo' },
  { key: 'recenter', label: 'Recenter' },
  { key: 'sweep', label: 'Radios' },
  { key: 'intersections', label: 'Cortes' },
  { key: 'best', label: 'Mejor' },
  { key: 'refine', label: 'Refine' },
  { key: 'result', label: 'Resultado' },
]

const LOCO_LAB_STAGES = [
  { key: 'proposals', label: '1. Propuestas' },
  { key: 'filters', label: '2. Filtros' },
  { key: 'circles', label: '3. Circulos' },
  { key: 'measure', label: '4. Medicion' },
  { key: 'evaluate', label: '5. Evaluacion' },
]

const LOCO_PRESETS = {
  custom: { label: 'Personalizado (manual)' },
  small_relaxed: {
    label: 'Peque\u00f1as \u2014 Relaxed',
    min_radius: 4, max_radius: 16, radius_step: 2,
    grid_step: 6, threshold: 0.80,
    small_threshold: 0.75, medium_threshold: 0.80, large_threshold: 0.85,
    small_radius_limit: 10, large_radius_limit: 14,
    crossing_threshold: 0.60,
    nms_mode: 'circle_iou', circle_iou_threshold: 0.50, nms_distance_factor: 0.60,
    use_spatial_final_filter: false,
    spatial_final_tile_px: 64, spatial_final_max_per_tile: 4,
    max_candidates: 12000, candidate_max_per_tile: 0,
  },
  small_balanced: {
    label: 'Peque\u00f1as \u2014 Balanced',
    min_radius: 4, max_radius: 16, radius_step: 2,
    grid_step: 8, threshold: 0.85,
    small_threshold: 0.80, medium_threshold: 0.85, large_threshold: 0.90,
    small_radius_limit: 10, large_radius_limit: 14,
    crossing_threshold: 0.50,
    nms_mode: 'circle_iou', circle_iou_threshold: 0.40, nms_distance_factor: 0.50,
    use_spatial_final_filter: true,
    spatial_final_tile_px: 64, spatial_final_max_per_tile: 4,
    max_candidates: 8000, candidate_max_per_tile: 200,
  },
  small_strict: {
    label: 'Peque\u00f1as \u2014 Strict',
    min_radius: 4, max_radius: 16, radius_step: 2,
    grid_step: 10, threshold: 0.90,
    small_threshold: 0.88, medium_threshold: 0.90, large_threshold: 0.95,
    small_radius_limit: 10, large_radius_limit: 14,
    crossing_threshold: 0.40,
    nms_mode: 'circle_iou', circle_iou_threshold: 0.30, nms_distance_factor: 0.40,
    use_spatial_final_filter: true,
    spatial_final_tile_px: 64, spatial_final_max_per_tile: 2,
    max_candidates: 6000, candidate_max_per_tile: 150,
  },
  medium_relaxed: {
    label: 'Medianas \u2014 Relaxed',
    min_radius: 8, max_radius: 24, radius_step: 3,
    grid_step: 8, threshold: 0.80,
    small_threshold: 0.75, medium_threshold: 0.80, large_threshold: 0.85,
    small_radius_limit: 12, large_radius_limit: 20,
    crossing_threshold: 0.60,
    nms_mode: 'circle_iou', circle_iou_threshold: 0.50, nms_distance_factor: 0.60,
    use_spatial_final_filter: false,
    spatial_final_tile_px: 128, spatial_final_max_per_tile: 3,
    max_candidates: 10000, candidate_max_per_tile: 0,
  },
  medium_balanced: {
    label: 'Medianas \u2014 Balanced',
    min_radius: 8, max_radius: 24, radius_step: 3,
    grid_step: 10, threshold: 0.88,
    small_threshold: 0.83, medium_threshold: 0.88, large_threshold: 0.92,
    small_radius_limit: 12, large_radius_limit: 20,
    crossing_threshold: 0.50,
    nms_mode: 'circle_iou', circle_iou_threshold: 0.40, nms_distance_factor: 0.50,
    use_spatial_final_filter: true,
    spatial_final_tile_px: 128, spatial_final_max_per_tile: 3,
    max_candidates: 8000, candidate_max_per_tile: 150,
  },
  medium_strict: {
    label: 'Medianas \u2014 Strict',
    min_radius: 8, max_radius: 24, radius_step: 3,
    grid_step: 12, threshold: 0.93,
    small_threshold: 0.90, medium_threshold: 0.93, large_threshold: 0.96,
    small_radius_limit: 12, large_radius_limit: 20,
    crossing_threshold: 0.35,
    nms_mode: 'circle_iou', circle_iou_threshold: 0.30, nms_distance_factor: 0.40,
    use_spatial_final_filter: true,
    spatial_final_tile_px: 128, spatial_final_max_per_tile: 2,
    max_candidates: 6000, candidate_max_per_tile: 100,
  },
  large_relaxed: {
    label: 'Grandes \u2014 Relaxed',
    min_radius: 16, max_radius: 40, radius_step: 4,
    grid_step: 10, threshold: 0.82,
    small_threshold: 0.78, medium_threshold: 0.82, large_threshold: 0.88,
    small_radius_limit: 20, large_radius_limit: 30,
    crossing_threshold: 0.55,
    nms_mode: 'circle_iou', circle_iou_threshold: 0.55, nms_distance_factor: 0.65,
    use_spatial_final_filter: false,
    spatial_final_tile_px: 192, spatial_final_max_per_tile: 3,
    max_candidates: 8000, candidate_max_per_tile: 0,
  },
  large_balanced: {
    label: 'Grandes \u2014 Balanced',
    min_radius: 16, max_radius: 40, radius_step: 4,
    grid_step: 12, threshold: 0.90,
    small_threshold: 0.85, medium_threshold: 0.90, large_threshold: 0.93,
    small_radius_limit: 20, large_radius_limit: 30,
    crossing_threshold: 0.45,
    nms_mode: 'circle_iou', circle_iou_threshold: 0.45, nms_distance_factor: 0.55,
    use_spatial_final_filter: true,
    spatial_final_tile_px: 192, spatial_final_max_per_tile: 3,
    max_candidates: 6000, candidate_max_per_tile: 100,
  },
  large_strict: {
    label: 'Grandes \u2014 Strict',
    min_radius: 16, max_radius: 40, radius_step: 4,
    grid_step: 15, threshold: 0.95,
    small_threshold: 0.92, medium_threshold: 0.95, large_threshold: 0.97,
    small_radius_limit: 20, large_radius_limit: 30,
    crossing_threshold: 0.30,
    nms_mode: 'circle_iou', circle_iou_threshold: 0.35, nms_distance_factor: 0.45,
    use_spatial_final_filter: true,
    spatial_final_tile_px: 192, spatial_final_max_per_tile: 2,
    max_candidates: 4000, candidate_max_per_tile: 80,
  },
  general_relaxed: {
    label: 'Generales \u2014 Relaxed',
    min_radius: 4, max_radius: 40, radius_step: 4,
    grid_step: 8, threshold: 0.78,
    small_threshold: 0.72, medium_threshold: 0.78, large_threshold: 0.85,
    small_radius_limit: 12, large_radius_limit: 24,
    crossing_threshold: 0.60,
    nms_mode: 'circle_iou', circle_iou_threshold: 0.50, nms_distance_factor: 0.60,
    use_spatial_final_filter: false,
    spatial_final_tile_px: 128, spatial_final_max_per_tile: 3,
    max_candidates: 12000, candidate_max_per_tile: 0,
  },
  general_balanced: {
    label: 'Generales \u2014 Balanced',
    min_radius: 6, max_radius: 36, radius_step: 4,
    grid_step: 10, threshold: 0.85,
    small_threshold: 0.80, medium_threshold: 0.85, large_threshold: 0.90,
    small_radius_limit: 14, large_radius_limit: 24,
    crossing_threshold: 0.50,
    nms_mode: 'circle_iou', circle_iou_threshold: 0.40, nms_distance_factor: 0.50,
    use_spatial_final_filter: true,
    spatial_final_tile_px: 128, spatial_final_max_per_tile: 3,
    max_candidates: 8000, candidate_max_per_tile: 150,
  },
  general_strict: {
    label: 'Generales \u2014 Strict',
    min_radius: 8, max_radius: 32, radius_step: 4,
    grid_step: 12, threshold: 0.92,
    small_threshold: 0.88, medium_threshold: 0.92, large_threshold: 0.95,
    small_radius_limit: 14, large_radius_limit: 24,
    crossing_threshold: 0.35,
    nms_mode: 'circle_iou', circle_iou_threshold: 0.30, nms_distance_factor: 0.40,
    use_spatial_final_filter: true,
    spatial_final_tile_px: 128, spatial_final_max_per_tile: 2,
    max_candidates: 6000, candidate_max_per_tile: 100,
  },
}

const PARAM_DESCRIPTIONS = {
  // Generacion
  candidate_sampling_mode: 'Controla c\u00f3mo se reduce el n\u00famero de candidatos cuando la grilla genera m\u00e1s que max candidatos. tile balanced reparte equitativamente entre tiles de la imagen.',
  grid_step: 'Distancia en p\u00edxeles entre centros de grilla. Valor bajo = m\u00e1s candidatos densos, valor alto = menos candidatos pero m\u00e1s r\u00e1pido.',
  max_candidates: 'M\u00e1ximo n\u00famero de candidatos a evaluar. Si la grilla genera m\u00e1s, se aplica el modo de muestreo para reducirlos.',
  candidate_max_per_tile: 'M\u00e1ximo de candidatos por tile en modo tile balanced. 0 = sin l\u00edmite por tile (usa solo max candidatos global).',
  tile_size_px: 'Tama\u00f1o del tile en p\u00edxeles para tile balanced. Divide la imagen en tiles de este tama\u00f1o y selecciona equitativamente de cada tile.',
  candidate_random_seed: 'Semilla aleatoria para los modos random seeded y tile balanced. Cambiar este valor genera una distribuci\u00f3n diferente de candidatos.',
  min_radius: 'Radio m\u00ednimo de c\u00edrculo a generar (p\u00edxeles). C\u00edrculos m\u00e1s peque\u00f1os que esto no se generan.',
  max_radius: 'Radio m\u00e1ximo de c\u00edrculo a generar (p\u00edxeles). C\u00edrculos m\u00e1s grandes que esto no se generan.',
  radius_step: 'Incremento de radio entre candidatos. Para cada centro se generan c\u00edrculos desde radio min hasta radio max en pasos de este valor.',
  // Threshold
  use_radius_thresholds: 'Al activarlo, se usa un threshold distinto seg\u00fan el grupo de radio (small/medium/large). Desactivado usa un \u00fanico threshold general.',
  threshold: 'Threshold \u00fanico usado cuando threshold por radio est\u00e1 desactivado. El modelo asigna un valid_score (0-1) a cada candidato; si supera este valor se acepta.',
  small_threshold: 'Threshold para el grupo small (radio < small limite). V\u00e1lido solo cuando threshold por radio est\u00e1 activado.',
  medium_threshold: 'Threshold para el grupo medium (small limite \u2264 radio < large limite). V\u00e1lido solo cuando threshold por radio est\u00e1 activado.',
  large_threshold: 'Threshold para el grupo large (radio \u2265 large limite). V\u00e1lido solo cuando threshold por radio est\u00e1 activado.',
  small_radius_limit: 'L\u00edmite superior del grupo small en p\u00edxeles de radio. Radios menores a este valor se consideran small.',
  large_radius_limit: 'L\u00edmite inferior del grupo large en p\u00edxeles de radio. Radios mayores o iguales a este valor se consideran large.',
  // Multiclase
  crossing_threshold: 'Probabilidad m\u00e1xima de crossing permitida para aceptar un c\u00edrculo. Valor bajo = m\u00e1s estricto (rechaza m\u00e1s candidatos con posible cruce).',
  // NMS
  use_nms: 'Non-Maximum Suppression: elimina c\u00edrculos duplicados o solapados despu\u00e9s del filtro de threshold.',
  nms_mode: 'Criterio de eliminaci\u00f3n en NMS. circle IoU usa intersecci\u00f3n sobre uni\u00f3n circular; dist/radio usa distancia entre centros y similitud de radio.',
  circle_iou_threshold: 'Umbral de IoU circular para modo circle IoU. Si el solapamiento entre dos c\u00edrculos supera este valor, el de menor score se elimina.',
  nms_distance_factor: 'Factor de distancia para modo dist/radio. Multiplicado por el radio menor. Controla qu\u00e9 tan cerca pueden estar dos c\u00edrculos.',
  radius_similarity_factor: 'Factor de similitud de radio para modo dist/radio. Multiplicado por el radio menor. Controla qu\u00e9 tan similares deben ser los radios para considerar duplicado.',
  // Visualizacion
  return_rejected: 'Incluye candidatos rechazados en la respuesta para depuraci\u00f3n visual. Desactivar es m\u00e1s r\u00e1pido y con menos datos.',
  max_return_rejected: 'M\u00e1ximo de candidatos rechazados a devolver para visualizaci\u00f3n. 0 = mostrar todos (puede ser lento con muchos candidatos).',
  // Espacial
  use_spatial_final_filter: 'Divide la imagen en tiles y limita el n\u00famero de c\u00edrculos aceptados por tile, asegurando distribuci\u00f3n espacial uniforme post-NMS.',
  spatial_final_tile_px: 'Tama\u00f1o del tile en p\u00edxeles para el filtro espacial. Valor bajo = distribuci\u00f3n m\u00e1s fina pero menos c\u00edrculos por tile.',
  spatial_final_max_per_tile: 'M\u00e1ximo de c\u00edrculos aceptados por tile en el filtro espacial. Dentro de cada tile se conservan los top-N por valid_score.',
  spatial_final_min_center_distance_factor: 'Distancia m\u00ednima entre centros dentro del mismo tile. 0 = sin restricci\u00f3n, 1.0 = no pueden solaparse.',
}

function ParamSpan({ paramKey, children }) {
  const tip = PARAM_DESCRIPTIONS[paramKey]
  if (!tip) return <span>{children}</span>
  return <span className="param-tip" data-tip={tip}>{children}</span>
}

function reviewLabel(value) {
  const v = String(value || '').toLowerCase()
  const row = REVIEW_TIERS.find((x) => x.value === v)
  if (row) return row.label
  if (v === 'ok') return 'OK legado'
  if (v === 'bad') return 'BAD legado'
  return ''
}

function CurtainCompare({ title, subtitle, baseUrl, maskUrl, maskClassName = '', persistOnSourceChange = false }) {
  const stageRef = useRef(null)
  const panRef = useRef({ dragging: false, x: 0, y: 0 })
  const [curtainPos, setCurtainPos] = useState(50)
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [panMode, setPanMode] = useState(false)
  const [dragDivider, setDragDivider] = useState(false)
  const [dragPan, setDragPan] = useState(false)

  useEffect(() => {
    if (persistOnSourceChange) return
    setCurtainPos(50)
    setZoom(1)
    setOffset({ x: 0, y: 0 })
    setPanMode(false)
    setDragDivider(false)
    setDragPan(false)
  }, [baseUrl, maskUrl, persistOnSourceChange])

  function updateCurtain(clientX) {
    const stage = stageRef.current
    if (!stage) return
    const rect = stage.getBoundingClientRect()
    if (rect.width < 1) return
    const pct = ((clientX - rect.left) / rect.width) * 100
    setCurtainPos(clamp(pct, 0, 100))
  }

  function isNearDivider(clientX, tolPx = 28) {
    const stage = stageRef.current
    if (!stage) return false
    const rect = stage.getBoundingClientRect()
    if (rect.width < 1) return false
    const dividerX = rect.left + (clamp(curtainPos, 0, 100) / 100) * rect.width
    return Math.abs(clientX - dividerX) <= tolPx
  }

  function onPointerDown(e) {
    if (!maskUrl) return
    if (panMode) {
      panRef.current = { dragging: true, x: e.clientX, y: e.clientY }
      setDragPan(true)
      e.currentTarget.setPointerCapture?.(e.pointerId)
      e.preventDefault()
      return
    }
    if (!isNearDivider(e.clientX, 30)) return
    setDragDivider(true)
    updateCurtain(e.clientX)
    e.currentTarget.setPointerCapture?.(e.pointerId)
    e.preventDefault()
  }

  function onPointerMove(e) {
    if (dragDivider) {
      updateCurtain(e.clientX)
      e.preventDefault()
      return
    }
    if (dragPan && panRef.current.dragging) {
      const dx = e.clientX - panRef.current.x
      const dy = e.clientY - panRef.current.y
      panRef.current = { dragging: true, x: e.clientX, y: e.clientY }
      setOffset((prev) => ({ x: prev.x + dx, y: prev.y + dy }))
      e.preventDefault()
    }
  }

  function onPointerUp(e) {
    setDragDivider(false)
    setDragPan(false)
    panRef.current = { dragging: false, x: 0, y: 0 }
    e.currentTarget.releasePointerCapture?.(e.pointerId)
  }

  return (
    <div className="curtain-panel">
      <div className="curtain-head">
        <strong>{title}</strong>
        {subtitle ? <span>{subtitle}</span> : null}
      </div>
      <div className="curtain-toolbar">
        <button className={`icon-tool ${panMode ? 'toggle-active' : ''}`} onClick={() => setPanMode((v) => !v)} disabled={!maskUrl}>Mano</button>
        <button className="icon-tool" onClick={() => setZoom((z) => clamp(z * 0.84, 0.25, 8))} disabled={!maskUrl}>-</button>
        <button className="icon-tool" onClick={() => setZoom((z) => clamp(z * 1.2, 0.25, 8))} disabled={!maskUrl}>+</button>
        <button className="icon-tool" onClick={() => { setZoom(1); setOffset({ x: 0, y: 0 }); setCurtainPos(50); setPanMode(false) }} disabled={!maskUrl}>Reset</button>
        <span className="zoom-chip">{Math.round(zoom * 100)}%</span>
      </div>
      <div
        ref={stageRef}
        className={`curtain-stage segment-curtain-stage ${panMode ? 'mode-pan' : 'mode-divider'} ${dragPan ? 'is-panning' : ''}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerLeave={onPointerUp}
      >
        <div className="curtain-zoom-layer" style={{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})` }}>
          <img src={baseUrl || ''} alt="Imagen base" className="segment-curtain-img" draggable={false} />
        </div>
        {maskUrl ? (
          <>
            <div className="curtain-mask-wrap" style={{ clipPath: `inset(0 ${100 - curtainPos}% 0 0)` }}>
              <div className="curtain-zoom-layer" style={{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})` }}>
                <img src={maskUrl} alt="Capa comparada" className={`segment-curtain-img ${maskClassName}`.trim()} draggable={false} />
              </div>
            </div>
            <div className="curtain-divider" style={{ left: `${curtainPos}%` }} />
          </>
        ) : (
          <div className="placeholder">Sin capa para comparar.</div>
        )}
      </div>
    </div>
  )
}

export default function App() {
  const drawCanvasRef = useRef(null)
  const labelsCanvasRef = useRef(null)
  const editorStageRef = useRef(null)
  const diameterStageRef = useRef(null)
  const locoStageRef = useRef(null)
  const locoDatasetStageRef = useRef(null)
  const locoTestStageRef = useRef(null)
  const locoModelStageRef = useRef(null)
  const pointReviewStageRef = useRef(null)
  const drawingRef = useRef(false)
  const excludeDragRef = useRef({ dragging: false, start: null })
  const panStateRef = useRef({ dragging: false, x: 0, y: 0 })
  const diamPanRef = useRef({ dragging: false, x: 0, y: 0 })
  const diamLineDragRef = useRef({ dragging: false, start: null })
  const diamCircleDragRef = useRef({ dragging: false, center: null })
  const locoPanRef = useRef({ dragging: false, x: 0, y: 0 })
  const locoCircleDragRef = useRef({ dragging: false, center: null })
  const locoDatasetPanRef = useRef({ dragging: false, x: 0, y: 0 })
  const locoDatasetDragRef = useRef({ mode: '', id: '', start: null, circle: null })
  const locoTestPanRef = useRef({ dragging: false, x: 0, y: 0 })
  const locoTestDragRef = useRef({ mode: '', id: '', start: null, circle: null })
  const locoModelPanRef = useRef({ dragging: false, x: 0, y: 0 })
  const pointReviewPanRef = useRef({ dragging: false, x: 0, y: 0 })
  const filterFocusRef = useRef(null)
  const localObjectUrlRef = useRef('')
  const loadReqIdRef = useRef(0)
  const scribbleAutosaveDirtyRef = useRef(false)
  const scribbleAutosaveInFlightRef = useRef(false)
  const scribbleAutosaveFailCountRef = useRef(0)

  const [sessionId, setSessionId] = useState('')
  const [imageId, setImageId] = useState('')
  const [imageName, setImageName] = useState('')
  const [imageUrl, setImageUrl] = useState('')
  const [savedImages, setSavedImages] = useState([])
  const [modelDataset, setModelDataset] = useState([])
  const [assistModels, setAssistModels] = useState([])
  const [defaultAssistModelId, setDefaultAssistModelId] = useState('')
  const [selectedAssistModelId, setSelectedAssistModelId] = useState('')
  const [selectedTrainImages, setSelectedTrainImages] = useState({})
  const [trainConfig, setTrainConfig] = useState({
    model_name: '',
    class_mode: 'multiclass',
    classifier: 'extratrees',
    feature_variant: 'context',
    n_estimators: 120,
    notes: '',
  })
  const [modelMinConfidence, setModelMinConfidence] = useState(0.72)
  const [modelIncludeFiber, setModelIncludeFiber] = useState(true)
  const [modelIncludeHalo, setModelIncludeHalo] = useState(true)
  const [modelIncludeBackground, setModelIncludeBackground] = useState(false)
  const [modelPrediction, setModelPrediction] = useState(null)
  const [modelTrainSummary, setModelTrainSummary] = useState(null)
  const [modelImagePreview, setModelImagePreview] = useState(null)

  const [experiments, setExperiments] = useState([])
  const [selectedExperiment, setSelectedExperiment] = useState('')
  const [selectedBatch, setSelectedBatch] = useState({})
  const [segSaveMode, setSegSaveMode] = useState('overwrite')
  const [selectedSavedImageId, setSelectedSavedImageId] = useState('')
  const [imageStartDir, setImageStartDir] = useState('')
  const [localImageFiles, setLocalImageFiles] = useState([])
  const [selectedLocalPath, setSelectedLocalPath] = useState('')
  const [workspaceTab, setWorkspaceTab] = useState('workbench') // workbench | review | diameter | loco | locoDataset | locoAugment | locoTraining | locoTest | models
  // New hierarchical navigation state
  const [activeGroup, setActiveGroup] = useState(() => legacyToGroup('workbench').group)
  const [activeTab, setActiveTab] = useState(() => legacyToGroup('workbench').tab)
  // Sync workspaceTab <-> activeGroup/activeTab for backward compatibility
  const handleGroupChange = useCallback((group) => {
    setActiveGroup(group)
  }, [])

  const handleTabChange = useCallback((tab) => {
    setActiveTab(tab)
    const legacy = groupToLegacy(activeGroup, tab)
    if (legacy) setWorkspaceTab(legacy)
  }, [activeGroup])

  // Keep workspaceTab in sync when activeGroup changes (for tab change)
  useEffect(() => {
    const legacy = groupToLegacy(activeGroup, activeTab)
    if (legacy) setWorkspaceTab(legacy)
  }, [activeGroup, activeTab])

  const [workbenchPanelTab, setWorkbenchPanelTab] = useState('image') // image | editor | experiments
  const [batchProgress, setBatchProgress] = useState({ active: false, done: 0, total: 0, current: '' })

  const [filterGroup, setFilterGroup] = useState('all')
  const [filterExperiment, setFilterExperiment] = useState('all')
  const [filterDecision, setFilterDecision] = useState('all')
  const [reviewSort, setReviewSort] = useState('latest')

  const [tool, setTool] = useState('fiber')
  const [brushSize, setBrushSize] = useState(16)
  const [brushAutoScale, setBrushAutoScale] = useState(true)
  const [viewerMode, setViewerMode] = useState('mark')
  const [viewerZoom, setViewerZoom] = useState(1)
  const [viewerOffset, setViewerOffset] = useState({ x: 0, y: 0 })
  const [isPanning, setIsPanning] = useState(false)
  const [imageDims, setImageDims] = useState({ w: 0, h: 0 })
  const [stageSize, setStageSize] = useState({ w: 0, h: 0 })
  const [annotHistory, setAnnotHistory] = useState([])
  const [annotFuture, setAnnotFuture] = useState([])
  const [excludeRect, setExcludeRect] = useState(null)

  const [runs, setRuns] = useState([])
  const [runCache, setRunCache] = useState({})
  const [activeRunId, setActiveRunId] = useState('')

  const [reviewNote, setReviewNote] = useState('')
  const [reviews, setReviews] = useState([])
  const [reportInfo, setReportInfo] = useState(null)

  const [diamMethodPanel, setDiamMethodPanel] = useState('automatic')
  const [diamMethodId, setDiamMethodId] = useState('hybrid_profile_diameter_v3_2_auto')
  const diamSourceMode = 'prior_mask'
  const [diamPriorRunId, setDiamPriorRunId] = useState('latest')
  const [diamViewerMode, setDiamViewerMode] = useState('mark')
  const [diamViewerZoom, setDiamViewerZoom] = useState(1)
  const [diamViewerOffset, setDiamViewerOffset] = useState({ x: 0, y: 0 })
  const [diamStageSize, setDiamStageSize] = useState({ w: 0, h: 0 })
  const [diamIsPanning, setDiamIsPanning] = useState(false)
  const [manualMaskLineDraft, setManualMaskLineDraft] = useState({ start: null, end: null })
  const [manualDirectLineDraft, setManualDirectLineDraft] = useState({ start: null, end: null })
  const [manualMaskLines, setManualMaskLines] = useState([])
  const [manualDirectLines, setManualDirectLines] = useState([])
  const [manualMaskLineActiveIdx, setManualMaskLineActiveIdx] = useState(-1)
  const [manualDirectLineActiveIdx, setManualDirectLineActiveIdx] = useState(-1)
  const [manualCircleDraft, setManualCircleDraft] = useState(null)
  const [manualCircles, setManualCircles] = useState([])
  const [manualCircleActiveIdx, setManualCircleActiveIdx] = useState(-1)
  const [manualCircleSelected, setManualCircleSelected] = useState(false)
  const [manualCircleConsumed, setManualCircleConsumed] = useState(false)
  const [diamManualHistory, setDiamManualHistory] = useState([])
  const [diamManualFuture, setDiamManualFuture] = useState([])
  const [diamPoints, setDiamPoints] = useState([])
  const [diamActivePointIdx, setDiamActivePointIdx] = useState(-1)
  const [diamOverlayUrl, setDiamOverlayUrl] = useState('')
  const [diamMaskVisible, setDiamMaskVisible] = useState(false)
  const [diamMaskOpacity, setDiamMaskOpacity] = useState(0.38)
  const [diamMaskLayerUrl, setDiamMaskLayerUrl] = useState('')
  const [diamResults, setDiamResults] = useState([])
  const [diamResultsMode, setDiamResultsMode] = useState('run')
  const [diamReviewTarget, setDiamReviewTarget] = useState(null)
  const [pointReviewOpen, setPointReviewOpen] = useState(false)
  const [pointReviewZoom, setPointReviewZoom] = useState(2)
  const [pointReviewOffset, setPointReviewOffset] = useState({ x: 0, y: 0 })
  const [pointReviewStageSize, setPointReviewStageSize] = useState({ w: 0, h: 0 })
  const [pointReviewIsPanning, setPointReviewIsPanning] = useState(false)
  const [diamRuns, setDiamRuns] = useState([])
  const [diamActiveRunId, setDiamActiveRunId] = useState('')
  const [diamRunCache, setDiamRunCache] = useState({})
  const [diamReportInfo, setDiamReportInfo] = useState(null)
  const [locoViewerMode, setLocoViewerMode] = useState('pan')
  const [locoViewerZoom, setLocoViewerZoom] = useState(1)
  const [locoViewerOffset, setLocoViewerOffset] = useState({ x: 0, y: 0 })
  const [locoStageSize, setLocoStageSize] = useState({ w: 0, h: 0 })
  const [locoIsPanning, setLocoIsPanning] = useState(false)
  const [locoPoints, setLocoPoints] = useState([])
  const [locoActivePointIdx, setLocoActivePointIdx] = useState(-1)
  const [locoCircleDraft, setLocoCircleDraft] = useState(null)
  const [locoHistory, setLocoHistory] = useState([])
  const [locoFuture, setLocoFuture] = useState([])
  const [locoPreview, setLocoPreview] = useState(null)
  const [locoStep, setLocoStep] = useState(0)
  const [locoCandidateIndex, setLocoCandidateIndex] = useState(-1)
  const [locoParams, setLocoParams] = useState({
    loco_roi_radius_px: 36,
    loco_max_radius_px: 26,
    loco_radius_step_px: 1,
    loco_seed_radius_window_px: 8,
    loco_circle_samples: 128,
    loco_recenter_enabled: true,
    loco_max_recenter_shift_px: 5,
    loco_symmetry_threshold: 0.62,
    loco_reject_threshold: 0.42,
    loco_mode: 'refine',
    loco_aggregation: 'median',
  })
  const [locoLabStage, setLocoLabStage] = useState('proposals')
  const [locoProposalMethod, setLocoProposalMethod] = useState('circle_grid')
  const [locoProposals, setLocoProposals] = useState([])
  const [locoFilteredProposals, setLocoFilteredProposals] = useState([])
  const [locoMeasuredResults, setLocoMeasuredResults] = useState([])
  const [locoEvaluation, setLocoEvaluation] = useState(null)
  const [locoSelectedProposalId, setLocoSelectedProposalId] = useState('')
  const [locoLayerVisibility, setLocoLayerVisibility] = useState({
    mask: true,
    components: false,
    proposals: true,
    rejected: false,
    intersections: true,
    quadrilaterals: true,
    diameter: true,
  })
  const [locoLabParams, setLocoLabParams] = useState({
    grid_stride_px: 18,
    radius_min_px: 3,
    radius_max_px: 18,
    radius_step_px: 2,
    mask_required_ratio: 0.1,
    max_candidates: 600,
    min_score: 0.42,
    max_intersections: 12,
    circle_samples: 128,
    require_four_cuts: true,
    measure_limit: 120,
  })
  const [locoDatasetTool, setLocoDatasetTool] = useState('circle')
  const [locoDatasetCircles, setLocoDatasetCircles] = useState([])
  const [locoDatasetSelectedId, setLocoDatasetSelectedId] = useState('')
  const [locoDatasetDraftCircle, setLocoDatasetDraftCircle] = useState(null)
  const [locoDatasetFeatures, setLocoDatasetFeatures] = useState([])
  const [locoDatasetSaveInfo, setLocoDatasetSaveInfo] = useState(null)
  const [locoDatasetDefaultLabel, setLocoDatasetDefaultLabel] = useState('valid')
  const [locoDatasetZoom, setLocoDatasetZoom] = useState(1)
  const [locoDatasetOffset, setLocoDatasetOffset] = useState({ x: 0, y: 0 })
  const [locoDatasetStageSize, setLocoDatasetStageSize] = useState({ w: 0, h: 0 })
  const [locoDatasetIsPanning, setLocoDatasetIsPanning] = useState(false)
  const [locoAugItems, setLocoAugItems] = useState([])
  const [locoAugCounts, setLocoAugCounts] = useState({ total: 0, valid: 0, invalid: 0, augmented_total: 0, augmented_valid: 0, augmented_invalid: 0 })
  const [locoAugSelected, setLocoAugSelected] = useState({})
  const [locoAugLabelFilter, setLocoAugLabelFilter] = useState('all')
  const [locoAugPipeline, setLocoAugPipeline] = useState([
    { id: 'aug_rotate_default', type: 'rotate', params: { probability: 1, angles: '90,180,270' } },
  ])
  const [locoAugPreview, setLocoAugPreview] = useState([])
  const [locoAugInfo, setLocoAugInfo] = useState(null)
  const [locoAugBlockType, setLocoAugBlockType] = useState('rotate')
  const [locoAugPasses, setLocoAugPasses] = useState(4)
  const [locoTrainingDataSelection, setLocoTrainingDataSelection] = useState('all')
  const [locoTrainingTestSize, setLocoTrainingTestSize] = useState(0.2)
  const [locoTrainingSeed, setLocoTrainingSeed] = useState(42)
  const [locoTrainingResult, setLocoTrainingResult] = useState(null)
  const [locoTrainingSelectedModel, setLocoTrainingSelectedModel] = useState('catboost')
  const [locoTrainingMulticlass, setLocoTrainingMulticlass] = useState(false)
  const [locoTrainingThreshold, setLocoTrainingThreshold] = useState(0.5)
  const [locoTrainingErrorType, setLocoTrainingErrorType] = useState('False Positives')
  const [locoTrainingMcErrorClass, setLocoTrainingMcErrorClass] = useState('all')
  const [locoTrainingMcErrorType, setLocoTrainingMcErrorType] = useState('all')
  const [locoTrainingCombErrorClass, setLocoTrainingCombErrorClass] = useState('all')
  const [locoTrainingCombErrorType, setLocoTrainingCombErrorType] = useState('all')
  const [locoTrainingCombErrorSubtype, setLocoTrainingCombErrorSubtype] = useState('all')
  const [locoTrainingRuns, setLocoTrainingRuns] = useState([])
  const [locoTrainingPixelMode, setLocoTrainingPixelMode] = useState('circle_only')
  const [locoTrainingPrunePx, setLocoTrainingPrunePx] = useState(1)
  const [locoTrainingUseZoom, setLocoTrainingUseZoom] = useState(true)
  const [locoTrainingUseSourceRadius, setLocoTrainingUseSourceRadius] = useState(true)
  const [locoTestTool, setLocoTestTool] = useState('circle')
  const [locoTestCircles, setLocoTestCircles] = useState([])
  const [locoTestSelectedId, setLocoTestSelectedId] = useState('')
  const [locoTestDraftCircle, setLocoTestDraftCircle] = useState(null)
  const [locoTestDefaultLabel, setLocoTestDefaultLabel] = useState('valid')
  const [locoTestZoom, setLocoTestZoom] = useState(1)
  const [locoTestOffset, setLocoTestOffset] = useState({ x: 0, y: 0 })
  const [locoTestStageSize, setLocoTestStageSize] = useState({ w: 0, h: 0 })
  const [locoTestIsPanning, setLocoTestIsPanning] = useState(false)
  const [locoTestTrainingRunId, setLocoTestTrainingRunId] = useState('latest')
  const [locoTestModelId, setLocoTestModelId] = useState('extratrees')
  const [locoTestThreshold, setLocoTestThreshold] = useState(0.5)
  const [locoTestResult, setLocoTestResult] = useState(null)
  const [locoModelRunId, setLocoModelRunId] = useState('latest')
  const [locoModelId, setLocoModelId] = useState('extratrees')
  const [locoModelZoom, setLocoModelZoom] = useState(1)
  const [locoModelOffset, setLocoModelOffset] = useState({ x: 0, y: 0 })
  const [locoModelStageSize, setLocoModelStageSize] = useState({ w: 0, h: 0 })
  const [locoModelIsPanning, setLocoModelIsPanning] = useState(false)
  const [locoModelSelectedId, setLocoModelSelectedId] = useState('')
  const [locoModelResult, setLocoModelResult] = useState(null)
  const [locoModelMeasurement, setLocoModelMeasurement] = useState(null)
  const [locoModelLayers, setLocoModelLayers] = useState({ mask: true, accepted: true, rejected: false, scores: true, tiles: false })
  const [locoModelParams, setLocoModelParams] = useState({
    crossing_threshold: 0.5,
    grid_step: 4,
    min_radius: 4,
    max_radius: 16,
    radius_step: 4,
    threshold: 0.6,
    use_radius_thresholds: false,
    small_threshold: 0.85,
    medium_threshold: 0.9,
    large_threshold: 0.95,
    small_radius_limit: 14,
    large_radius_limit: 24,
    use_nms: true,
    nms_mode: 'circle_iou',
    nms_distance_factor: 4,
    radius_similarity_factor: 0.4,
    circle_iou_threshold: 0.1,
    candidate_sampling_mode: 'tile_balanced',
    candidate_random_seed: 42,
    tile_size_px: 6,
    candidate_max_per_tile: 8,
    return_rejected: false,
    max_return_rejected: 5000,
    max_candidates: 200000,
    use_spatial_final_filter: false,
    spatial_final_tile_px: 128,
    spatial_final_max_per_tile: 3,
    spatial_final_min_center_distance_factor: 1.0,
  })
  const [locoModelPreset, setLocoModelPreset] = useState('custom')
  const [validationCases, setValidationCases] = useState([])
  const [validationActiveCaseId, setValidationActiveCaseId] = useState('')
  const [validationExportInfo, setValidationExportInfo] = useState(null)
  const [diamCalibration, setDiamCalibration] = useState({
    enabled: false,
    known_nm: 100,
    pixel_distance: 100,
    nm_per_px: 1.0,
    unit: 'nm',
  })
  const [diamHistogramBins, setDiamHistogramBins] = useState(20)
  const [diamHistogramUnit, setDiamHistogramUnit] = useState('px')

  const [validationForm, setValidationForm] = useState({
    case_id: '',
    category: 'borde_limpio',
    manual_diameter_px: '',
    manual_left_x: '',
    manual_left_y: '',
    manual_right_x: '',
    manual_right_y: '',
    measurement_decision: 'unreviewed',
    quality_manual: 'medium',
    notes: '',
    result_comment: '',
  })
  const [diamParams, setDiamParams] = useState({
    support_high_threshold: 0.7,
    support_low_threshold: 0.35,
    support_dilation_px: 5,
    local_window_px: 41,
    profile_length_px: 80,
    profile_count: 7,
    profile_spacing_px: 2,
    edge_min_score: 0.18,
    min_valid_profiles: 3,
    max_mad_scale: 2.5,
    mask_local_radius_px: 32,
    mask_recenter_radius_px: 6,
    mask_ray_count: 36,
    auto_small_context_width_px: 14,
    circle_square_seed_mode: 'manual_circle',
    circle_square_seed_radius_px: 8,
    circle_square_max_radius_px: 26,
    circle_square_length_factor: 0.9,
    circle_square_width_factor: 0.7,
    circle_square_samples: 7,
    circle_square_aggregation: 'median',
    circle_square_recenter_seed: true,
    circle_square_max_recenter_shift_px: 5,
    ellipse_roi_radius_px: 42,
    manual_caliper_refine: true,
  })

  const [loading, setLoading] = useState(emptyLoading())
  const [notice, setNotice] = useState({ level: 'info', title: 'Listo', text: 'Inicializando...' })
  const [draftInfo, setDraftInfo] = useState('')

  const experimentsById = useMemo(() => {
    const out = {}
    experiments.forEach((x) => { out[x.experiment_id] = x })
    return out
  }, [experiments])

  const reviewByRun = useMemo(() => {
    const out = {}
    reviews.forEach((r) => {
      if (!out[r.run_id]) out[r.run_id] = r
    })
    return out
  }, [reviews])

  const filteredRuns = useMemo(() => {
    const out = runs.filter((r) => {
      const exp = experimentsById[r.experiment_id] || {}
      if (!exp.experiment_id) return false
      const group = r.group || exp.group || ''
      if (filterGroup !== 'all' && group !== filterGroup) return false
      if (filterExperiment !== 'all' && r.experiment_id !== filterExperiment) return false

      if (filterDecision !== 'all') {
        const d = String((reviewByRun[r.run_id] || {}).decision || '').toLowerCase()
        if (filterDecision === 'unreviewed' && d) return false
        if (filterDecision !== 'unreviewed' && d !== filterDecision) return false
      }
      return true
    })
    if (reviewSort === 'tier') {
      out.sort((a, b) => {
        const da = String((reviewByRun[a.run_id] || {}).decision || '').toLowerCase()
        const db = String((reviewByRun[b.run_id] || {}).decision || '').toLowerCase()
        const ra = REVIEW_RANK[da] ?? -1
        const rb = REVIEW_RANK[db] ?? -1
        if (rb !== ra) return rb - ra
        return String(b.created_at || '').localeCompare(String(a.created_at || ''))
      })
    }
    return out
  }, [runs, experimentsById, filterGroup, filterExperiment, filterDecision, reviewByRun, reviewSort])

  useEffect(() => {
    if (!filteredRuns.length) {
      setActiveRunId('')
      return
    }
    const hasActive = filteredRuns.some((r) => r.run_id === activeRunId)
    if (!hasActive) setActiveRunId(filteredRuns[0].run_id)
  }, [filteredRuns, activeRunId])

  const activeRun = runCache[activeRunId] || null
  const activeIndex = useMemo(() => filteredRuns.findIndex((r) => r.run_id === activeRunId), [filteredRuns, activeRunId])
  const activeRunExclude = useMemo(() => (activeRun?.meta?.exclude_rect || null), [activeRun])
  const activeRunExcludeMismatch = useMemo(() => {
    if (!activeRun) return false
    if (!excludeRect) return false
    if (!activeRunExclude) return false
    return !rectRoughEqual(activeRunExclude, excludeRect)
  }, [activeRun, activeRunExclude, excludeRect])
  const diamVisualMaskUrl = useMemo(() => {
    const priorRunId = diamPriorRunId === 'latest' ? (runs[0]?.run_id || '') : diamPriorRunId
    const priorRun = priorRunId ? runCache[priorRunId] : null
    if (priorRun?.mask_b64) return b64ToDataUrl(priorRun.mask_b64)
    return ''
  }, [diamPriorRunId, runs, runCache])
  const trainImageIds = useMemo(
    () => Object.keys(selectedTrainImages).filter((k) => selectedTrainImages[k]),
    [selectedTrainImages],
  )
  const activeAssistModel = useMemo(
    () => assistModels.find((m) => m.model_id === selectedAssistModelId) || assistModels.find((m) => m.model_id === defaultAssistModelId) || null,
    [assistModels, selectedAssistModelId, defaultAssistModelId],
  )

  function shortPathTail(pathText) {
    const raw = String(pathText || '').trim()
    if (!raw) return '-'
    const parts = raw.split(/[\\/]+/).filter(Boolean)
    return parts.slice(-3).join(' / ') || raw
  }

  function currentManualLineKind(methodId = diamMethodId) {
    return methodId === 'manual_line_direct_caliper' ? 'direct' : 'mask'
  }

  function lineDraftForKind(kind) {
    return kind === 'direct' ? manualDirectLineDraft : manualMaskLineDraft
  }

  function linesForKind(kind) {
    return kind === 'direct' ? manualDirectLines : manualMaskLines
  }

  function setLineDraftForKind(kind, value) {
    const setter = kind === 'direct' ? setManualDirectLineDraft : setManualMaskLineDraft
    setter((prev) => (typeof value === 'function' ? value(prev) : value))
  }

  function setLinesForKind(kind, value) {
    const setter = kind === 'direct' ? setManualDirectLines : setManualMaskLines
    setter((prev) => (typeof value === 'function' ? value(prev) : value))
  }

  function setLineActiveIndexForKind(kind, value) {
    const setter = kind === 'direct' ? setManualDirectLineActiveIdx : setManualMaskLineActiveIdx
    setter(value)
  }

  function lineAlmostEqual(a, b) {
    if (!a?.start || !a?.end || !b?.start || !b?.end) return false
    return (
      Math.hypot(Number(a.start.x) - Number(b.start.x), Number(a.start.y) - Number(b.start.y)) < 0.01 &&
      Math.hypot(Number(a.end.x) - Number(b.end.x), Number(a.end.y) - Number(b.end.y)) < 0.01
    )
  }

  function makeManualGeometryId(kind, start = null, end = null) {
    const sx = Number(start?.x)
    const sy = Number(start?.y)
    const ex = Number(end?.x)
    const ey = Number(end?.y)
    const coord = [sx, sy, ex, ey]
      .map((v) => (Number.isFinite(v) ? Math.round(v * 10) : 'na'))
      .join('_')
    const suffix = typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}_${Math.random().toString(16).slice(2)}`
    return `${kind}_${coord}_${suffix}`
  }

  function lineGeometryId(kind, line, idx = -1) {
    return String(line?.geometry_id || line?.manual_geometry_id || `${kind}_line_${idx}`)
  }

  function normalizedLinesForKind(kind) {
    const saved = linesForKind(kind).filter((line) => line?.start && line?.end)
    const draft = lineDraftForKind(kind)
    if (draft?.start && draft?.end && !saved.some((line) => lineAlmostEqual(line, draft))) {
      return [...saved, draft]
    }
    return saved
  }

  const manualLineDraft = lineDraftForKind(currentManualLineKind())

  function modelDatasetPreviewUrls(item) {
    return {
      real: item?.thumbnail_b64 ? b64ToDataUrl(item.thumbnail_b64, item.thumbnail_mime || 'image/png') : '',
      scribble: item?.scribble_thumb_b64 ? b64ToDataUrl(item.scribble_thumb_b64, item.scribble_thumb_mime || 'image/png') : '',
    }
  }

  async function openModelImagePreview(item) {
    const urls = modelDatasetPreviewUrls(item)
    setModelImagePreview({
      image_name: String(item?.image_name || item?.image_id || ''),
      source_path: String(item?.source_path || ''),
      source_mtime: String(item?.source_mtime || item?.updated_at || ''),
      real: urls.real,
      scribble: urls.scribble,
      loading: true,
    })
    const iid = String(item?.image_id || '').trim()
    if (!iid) return
    try {
      const res = await apiGet(`/api/assist-models/dataset/preview?image_id=${encodeURIComponent(iid)}`)
      setModelImagePreview({
        image_name: String(res?.image_name || item?.image_name || iid),
        source_path: String(res?.source_path || item?.source_path || ''),
        source_mtime: String(res?.source_mtime || item?.source_mtime || item?.updated_at || ''),
        real: res?.image_b64 ? b64ToDataUrl(res.image_b64, res.image_mime || 'image/png') : urls.real,
        scribble: res?.scribble_b64 ? b64ToDataUrl(res.scribble_b64, res.scribble_mime || 'image/png') : urls.scribble,
        loading: false,
      })
    } catch (err) {
      setModelImagePreview((prev) => prev ? { ...prev, loading: false, error: errMsg(err) } : prev)
      toast('warning', 'Preview dataset', errMsg(err))
    }
  }

  async function buildGreenMaskLayer(maskUrl) {
    if (!maskUrl) return ''
    const img = new Image()
    await new Promise((resolve, reject) => {
      img.onload = resolve
      img.onerror = () => reject(new Error('No se pudo renderizar la mascara.'))
      img.src = maskUrl
    })
    const w = Math.max(1, img.naturalWidth || img.width || imageDims.w || 1)
    const h = Math.max(1, img.naturalHeight || img.height || imageDims.h || 1)
    const canvas = document.createElement('canvas')
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    if (!ctx) return ''
    ctx.drawImage(img, 0, 0, w, h)
    const image = ctx.getImageData(0, 0, w, h)
    const data = image.data
    for (let i = 0; i < data.length; i += 4) {
      const lum = Math.max(data[i], data[i + 1], data[i + 2])
      data[i] = 25
      data[i + 1] = 214
      data[i + 2] = 107
      data[i + 3] = lum > 8 ? lum : 0
    }
    ctx.putImageData(image, 0, 0)
    return canvas.toDataURL('image/png')
  }

  function setLoad(k, v) {
    setLoading((prev) => ({ ...prev, [k]: v }))
  }

  async function withLoad(k, fn) {
    setLoad(k, true)
    try {
      await fn()
    } finally {
      setLoad(k, false)
    }
  }

  function toast(level, title, text) {
    setNotice({ level, title, text })
  }

  function quantizeLabelsCanvas(canvas) {
    if (!canvas) return
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    if (!ctx) return
    const { width, height } = canvas
    if (!width || !height) return
    const image = ctx.getImageData(0, 0, width, height)
    const data = image.data
    for (let i = 0; i < data.length; i += 4) {
      const a = data[i + 3]
      if (a < 16) {
        data[i] = 0
        data[i + 1] = 0
        data[i + 2] = 0
        data[i + 3] = 0
        continue
      }
      const v = data[i]
      let q = 0
      if (v >= 224) q = 255
      else if (v >= 176 && v <= 216) q = 192
      else if (v >= 96 && v <= 160) q = 128
      data[i] = q
      data[i + 1] = q
      data[i + 2] = q
      data[i + 3] = q > 0 ? 255 : 0
    }
    ctx.putImageData(image, 0, 0)
  }

  function paintLabelDisk(canvas, cx, cy, radius, labelVal) {
    if (!canvas) return
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    if (!ctx) return
    const w = canvas.width
    const h = canvas.height
    if (w < 1 || h < 1) return
    const r = Math.max(1, Number(radius) || 1)
    const x0 = Math.max(0, Math.floor(cx - r - 1))
    const y0 = Math.max(0, Math.floor(cy - r - 1))
    const x1 = Math.min(w - 1, Math.ceil(cx + r + 1))
    const y1 = Math.min(h - 1, Math.ceil(cy + r + 1))
    const patchW = Math.max(1, x1 - x0 + 1)
    const patchH = Math.max(1, y1 - y0 + 1)

    const image = ctx.getImageData(x0, y0, patchW, patchH)
    const data = image.data
    const rr = r * r
    const q = labelVal === 255 ? 255 : (labelVal === 192 ? 192 : (labelVal === 128 ? 128 : 0))
    for (let yy = y0; yy <= y1; yy += 1) {
      const dy = yy - cy
      for (let xx = x0; xx <= x1; xx += 1) {
        const dx = xx - cx
        if ((dx * dx + dy * dy) > rr) continue
        const idx = ((yy - y0) * patchW + (xx - x0)) * 4
        if (q > 0) {
          data[idx] = q
          data[idx + 1] = q
          data[idx + 2] = q
          data[idx + 3] = 255
        } else {
          data[idx] = 0
          data[idx + 1] = 0
          data[idx + 2] = 0
          data[idx + 3] = 0
        }
      }
    }
    ctx.putImageData(image, x0, y0)
  }

  function renderDrawFromLabels() {
    const draw = drawCanvasRef.current
    const labels = labelsCanvasRef.current
    if (!draw || !labels) return
    const dctx = draw.getContext('2d', { willReadFrequently: true })
    const lctx = labels.getContext('2d', { willReadFrequently: true })
    if (!dctx || !lctx) return

    const w = labels.width
    const h = labels.height
    const src = lctx.getImageData(0, 0, w, h)
    const out = dctx.createImageData(w, h)
    const s = src.data
    const o = out.data

    for (let i = 0; i < s.length; i += 4) {
      const v = s[i]
      if (v === 128) {
        o[i] = 0
        o[i + 1] = 229
        o[i + 2] = 255
        o[i + 3] = 220
      } else if (v === 192) {
        o[i] = 255
        o[i + 1] = 132
        o[i + 2] = 0
        o[i + 3] = 220
      } else if (v === 255) {
        o[i] = 136
        o[i + 1] = 75
        o[i + 2] = 220
        o[i + 3] = 220
      } else {
        o[i] = 0
        o[i + 1] = 0
        o[i + 2] = 0
        o[i + 3] = 0
      }
    }
    dctx.clearRect(0, 0, w, h)
    dctx.putImageData(out, 0, 0)
  }

  function clearScribbles() {
    const draw = drawCanvasRef.current
    const labels = labelsCanvasRef.current
    if (!draw || !labels) return
    draw.getContext('2d')?.clearRect(0, 0, draw.width, draw.height)
    labels.getContext('2d')?.clearRect(0, 0, labels.width, labels.height)
  }

  async function onClearScribbles() {
    if (!imageUrl) return
    const snap = captureAnnotSnapshot()
    if (snap) {
      setAnnotHistory((h) => [...h, snap].slice(-40))
      setAnnotFuture([])
    }
    clearScribbles()
    scribbleAutosaveDirtyRef.current = false
    scribbleAutosaveFailCountRef.current = 0
    await clearScribbleDraftRemote()
  }

  function captureAnnotSnapshot() {
    const draw = drawCanvasRef.current
    const labels = labelsCanvasRef.current
    if (!draw || !labels) return null
    const dctx = draw.getContext('2d')
    const lctx = labels.getContext('2d')
    if (!dctx || !lctx) return null
    return {
      width: draw.width,
      height: draw.height,
      draw_img: dctx.getImageData(0, 0, draw.width, draw.height),
      labels_img: lctx.getImageData(0, 0, labels.width, labels.height),
    }
  }

  function restoreAnnotSnapshot(snap) {
    if (!snap) return
    const draw = drawCanvasRef.current
    const labels = labelsCanvasRef.current
    if (!draw || !labels) return
    if (draw.width !== snap.width || draw.height !== snap.height) {
      draw.width = snap.width
      draw.height = snap.height
      labels.width = snap.width
      labels.height = snap.height
    }
    const dctx = draw.getContext('2d')
    const lctx = labels.getContext('2d')
    if (!dctx || !lctx) return
    dctx.putImageData(snap.draw_img, 0, 0)
    lctx.putImageData(snap.labels_img, 0, 0)
  }

  function resetAnnotHistory() {
    setAnnotHistory([])
    setAnnotFuture([])
  }

  function onAnnotUndo() {
    if (!annotHistory.length) return
    const prev = annotHistory[annotHistory.length - 1]
    const cur = captureAnnotSnapshot()
    restoreAnnotSnapshot(prev)
    setAnnotHistory((h) => h.slice(0, -1))
    if (cur) setAnnotFuture((f) => [cur, ...f].slice(0, 40))
    markScribbleDirty()
  }

  function onAnnotRedo() {
    if (!annotFuture.length) return
    const next = annotFuture[0]
    const cur = captureAnnotSnapshot()
    restoreAnnotSnapshot(next)
    setAnnotFuture((f) => f.slice(1))
    if (cur) setAnnotHistory((h) => [...h, cur].slice(-40))
    markScribbleDirty()
  }

  async function applyImageFromUrl(url) {
    if (!url) return false
    const img = new Image()
    await new Promise((resolve, reject) => {
      img.onload = resolve
      img.onerror = () => reject(new Error('No se pudo renderizar la imagen.'))
      img.src = url
    })
    const draw = drawCanvasRef.current
    const labels = labelsCanvasRef.current
    if (!draw || !labels) return false
    draw.width = img.naturalWidth
    draw.height = img.naturalHeight
    labels.width = img.naturalWidth
    labels.height = img.naturalHeight
    clearScribbles()
    setImageDims({ w: img.naturalWidth, h: img.naturalHeight })
    setViewerMode('mark')
    setViewerZoom(1)
    setViewerOffset({ x: 0, y: 0 })
    setExcludeRect(null)
    excludeDragRef.current = { dragging: false, start: null }
    resetAnnotHistory()
    resetScribbleDraftState()
    setImageUrl(url)
    return true
  }

  async function applyImageB64(imageB64, mime = 'image/png') {
    const url = b64ToDataUrl(imageB64, mime)
    if (!url) return false
    return applyImageFromUrl(url)
  }

  async function applyImageFilePreview(file) {
    if (!file) return false
    if (localObjectUrlRef.current) {
      try { URL.revokeObjectURL(localObjectUrlRef.current) } catch {}
      localObjectUrlRef.current = ''
    }
    const objectUrl = URL.createObjectURL(file)
    localObjectUrlRef.current = objectUrl
    try {
      return await applyImageFromUrl(objectUrl)
    } catch {
      return false
    }
  }

  function getEditorRenderMetrics() {
    const { w, h } = imageDims
    if (w < 1 || h < 1 || stageSize.w < 1 || stageSize.h < 1) return null
    const fit = Math.min(stageSize.w / w, stageSize.h / h)
    const scale = Math.max(0.0001, fit * viewerZoom)
    const contentW = w * scale
    const contentH = h * scale
    const x = (stageSize.w - contentW) * 0.5 + viewerOffset.x
    const y = (stageSize.h - contentH) * 0.5 + viewerOffset.y
    return { scale, x, y }
  }

  function pointFromClient(clientX, clientY) {
    const stage = editorStageRef.current
    if (!stage || imageDims.w < 1 || imageDims.h < 1) return null
    const metrics = getEditorRenderMetrics()
    if (!metrics) return null
    const rect = stage.getBoundingClientRect()
    if (rect.width < 1 || rect.height < 1) return null
    const localX = (clientX - rect.left - metrics.x) / metrics.scale
    const localY = (clientY - rect.top - metrics.y) / metrics.scale
    if (!Number.isFinite(localX) || !Number.isFinite(localY)) return null
    if (localX < 0 || localY < 0 || localX >= imageDims.w || localY >= imageDims.h) return null
    return { x: localX, y: localY }
  }

  function pointFromClientClamped(clientX, clientY) {
    const stage = editorStageRef.current
    if (!stage || imageDims.w < 1 || imageDims.h < 1) return null
    const metrics = getEditorRenderMetrics()
    if (!metrics) return null
    const rect = stage.getBoundingClientRect()
    if (rect.width < 1 || rect.height < 1) return null
    const rawX = (clientX - rect.left - metrics.x) / metrics.scale
    const rawY = (clientY - rect.top - metrics.y) / metrics.scale
    if (!Number.isFinite(rawX) || !Number.isFinite(rawY)) return null
    const x = clamp(rawX, 0, Math.max(0, imageDims.w - 1))
    const y = clamp(rawY, 0, Math.max(0, imageDims.h - 1))
    return { x, y }
  }

  function paintAt(clientX, clientY) {
    const p = pointFromClient(clientX, clientY)
    if (!p || viewerMode !== 'mark') return false
    const val = tool === 'fiber' ? 128 : (tool === 'halo' ? 192 : (tool === 'bg' ? 255 : 0))
    paintLabelDisk(labelsCanvasRef.current, p.x, p.y, annotBrushPx, val)
    renderDrawFromLabels()
    return true
  }

  function onEditorWheel(e) {
    if (!imageUrl) return
    if (!e.altKey) return
    e.preventDefault()
    const direction = e.deltaY < 0 ? 1 : -1
    const current = Math.max(1, Math.round(Number(brushSize) || 1))
    const step = e.shiftKey ? 5 : (current < 10 ? 1 : 2)
    setBrushSize((prev) => clamp((Number(prev) || 16) + direction * step, 1, 120))
  }

  function onEditorPointerDown(e) {
    if (!imageUrl) return
    if (viewerMode === 'pan') {
      panStateRef.current = { dragging: true, x: e.clientX, y: e.clientY }
      setIsPanning(true)
      e.currentTarget.setPointerCapture?.(e.pointerId)
      e.preventDefault()
      return
    }
    if (viewerMode === 'exclude') {
      const p = pointFromClientClamped(e.clientX, e.clientY)
      if (!p) return
      excludeDragRef.current = { dragging: true, start: p }
      setExcludeRect(normalizeRectFromPoints(p, p))
      e.currentTarget.setPointerCapture?.(e.pointerId)
      e.preventDefault()
      return
    }
    const snap = captureAnnotSnapshot()
    const didPaint = paintAt(e.clientX, e.clientY)
    if (!didPaint) return
    if (snap) {
      setAnnotHistory((h) => [...h, snap].slice(-40))
      setAnnotFuture([])
    }
    drawingRef.current = true
    e.currentTarget.setPointerCapture?.(e.pointerId)
    e.preventDefault()
  }

  function onEditorPointerMove(e) {
    if (!imageUrl) return
    if (viewerMode === 'pan') {
      if (!panStateRef.current.dragging) return
      const dx = e.clientX - panStateRef.current.x
      const dy = e.clientY - panStateRef.current.y
      panStateRef.current = { dragging: true, x: e.clientX, y: e.clientY }
      setViewerOffset((prev) => ({ x: prev.x + dx, y: prev.y + dy }))
      e.preventDefault()
      return
    }
    if (viewerMode === 'exclude') {
      if (!excludeDragRef.current.dragging || !excludeDragRef.current.start) return
      const p = pointFromClientClamped(e.clientX, e.clientY)
      if (!p) return
      setExcludeRect(normalizeRectFromPoints(excludeDragRef.current.start, p))
      e.preventDefault()
      return
    }
    if (!drawingRef.current) return
    paintAt(e.clientX, e.clientY)
    e.preventDefault()
  }

  function onEditorPointerUp(e) {
    if (excludeDragRef.current.dragging) {
      excludeDragRef.current = { dragging: false, start: null }
      e.currentTarget.releasePointerCapture?.(e.pointerId)
      return
    }
    const wasDrawing = drawingRef.current
    drawingRef.current = false
    if (panStateRef.current.dragging) {
      panStateRef.current = { dragging: false, x: 0, y: 0 }
      setIsPanning(false)
    }
    if (wasDrawing) {
      markScribbleDirty()
      void saveScribbleDraft({ silent: true })
    }
    e.currentTarget.releasePointerCapture?.(e.pointerId)
  }

  function zoomEditorBy(factor) {
    if (!imageUrl) return
    setViewerZoom((prev) => clamp(prev * factor, 0.25, 8))
  }

  function zoomEditorAt(clientX, clientY, factor) {
    const stage = editorStageRef.current
    if (!imageUrl || !stage || imageDims.w < 1 || imageDims.h < 1) return
    const metrics = getEditorRenderMetrics()
    if (!metrics) return
    const rect = stage.getBoundingClientRect()
    const localX = clientX - rect.left
    const localY = clientY - rect.top
    const imageX = (localX - metrics.x) / metrics.scale
    const imageY = (localY - metrics.y) / metrics.scale
    if (!Number.isFinite(imageX) || !Number.isFinite(imageY)) {
      zoomEditorBy(factor)
      return
    }
    const nextZoom = clamp(viewerZoom * factor, 0.25, 8)
    const fit = Math.min(stageSize.w / imageDims.w, stageSize.h / imageDims.h)
    const nextScale = Math.max(0.0001, fit * nextZoom)
    const nextContentW = imageDims.w * nextScale
    const nextContentH = imageDims.h * nextScale
    const baseX = (stageSize.w - nextContentW) * 0.5
    const baseY = (stageSize.h - nextContentH) * 0.5
    setViewerZoom(nextZoom)
    setViewerOffset({
      x: localX - imageX * nextScale - baseX,
      y: localY - imageY * nextScale - baseY,
    })
  }

  function resetEditorView() {
    setViewerMode('mark')
    setViewerZoom(1)
    setViewerOffset({ x: 0, y: 0 })
  }

  function clearExcludeRect() {
    setExcludeRect(null)
    excludeDragRef.current = { dragging: false, start: null }
  }

  function getExcludeRectPayload() {
    if (!excludeRect) return null
    const x = Number(excludeRect.x || 0)
    const y = Number(excludeRect.y || 0)
    const w = Number(excludeRect.w || 0)
    const h = Number(excludeRect.h || 0)
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(w) || !Number.isFinite(h)) return null
    if (w <= 0 || h <= 0) return null
    return { x, y, w, h }
  }

  function currentScribbleB64() {
    const labels = labelsCanvasRef.current
    if (!labels) return ''
    quantizeLabelsCanvas(labels)
    return labels.toDataURL('image/png').split(',')[1] || ''
  }

  function markScribbleDirty() {
    scribbleAutosaveDirtyRef.current = true
  }

  function resetScribbleDraftState() {
    scribbleAutosaveDirtyRef.current = false
    scribbleAutosaveInFlightRef.current = false
    scribbleAutosaveFailCountRef.current = 0
    setDraftInfo('')
  }

  async function restoreScribbleMapFromB64(scribbleB64) {
    const labels = labelsCanvasRef.current
    if (!labels || !scribbleB64) return false
    const img = new Image()
    await new Promise((resolve, reject) => {
      img.onload = resolve
      img.onerror = () => reject(new Error('No se pudo decodificar scribble guardado.'))
      img.src = b64ToDataUrl(scribbleB64)
    })
    const lctx = labels.getContext('2d', { willReadFrequently: true })
    if (!lctx) return false
    lctx.clearRect(0, 0, labels.width, labels.height)
    lctx.drawImage(img, 0, 0, labels.width, labels.height)
    quantizeLabelsCanvas(labels)
    renderDrawFromLabels()
    resetAnnotHistory()
    return true
  }

  async function mergeScribbleMapFromB64(scribbleB64) {
    const labels = labelsCanvasRef.current
    if (!labels || !scribbleB64) return 0
    const img = new Image()
    await new Promise((resolve, reject) => {
      img.onload = resolve
      img.onerror = () => reject(new Error('No se pudo decodificar la sugerencia del modelo.'))
      img.src = b64ToDataUrl(scribbleB64)
    })
    const tmp = document.createElement('canvas')
    tmp.width = labels.width
    tmp.height = labels.height
    const tctx = tmp.getContext('2d', { willReadFrequently: true })
    const lctx = labels.getContext('2d', { willReadFrequently: true })
    if (!tctx || !lctx) return 0
    tctx.drawImage(img, 0, 0, labels.width, labels.height)
    const src = tctx.getImageData(0, 0, labels.width, labels.height)
    const dst = lctx.getImageData(0, 0, labels.width, labels.height)
    const s = src.data
    const d = dst.data
    let changed = 0
    for (let i = 0; i < s.length; i += 4) {
      if (s[i + 3] < 16) continue
      const v = s[i]
      let q = 0
      if (v >= 224) q = 255
      else if (v >= 176 && v <= 216) q = 192
      else if (v >= 96 && v <= 160) q = 128
      if (!q) continue
      if (d[i] === q && d[i + 3] > 0) continue
      d[i] = q
      d[i + 1] = q
      d[i + 2] = q
      d[i + 3] = 255
      changed += 1
    }
    lctx.putImageData(dst, 0, 0)
    quantizeLabelsCanvas(labels)
    renderDrawFromLabels()
    return changed
  }

  async function saveScribbleDraft({ silent = true } = {}) {
    if (!sessionId || !imageId) return false
    const labels = labelsCanvasRef.current
    if (!labels) return false
    if (scribbleAutosaveInFlightRef.current) return false
    scribbleAutosaveInFlightRef.current = true
    try {
      const scribble = currentScribbleB64()
      const res = await apiPost('/api/scribble/draft/save', {
        session_id: sessionId,
        image_id: imageId,
        scribble_map_b64: scribble,
      })
      scribbleAutosaveDirtyRef.current = false
      scribbleAutosaveFailCountRef.current = 0
      const ts = String(res?.payload?.updated_at || '')
      if (ts) setDraftInfo(`Autoguardado: ${ts}`)
      else setDraftInfo('Autoguardado')
      if (!silent) toast('success', 'Draft', 'Scribble guardado.')
      return true
    } catch (err) {
      scribbleAutosaveFailCountRef.current += 1
      if (!silent || scribbleAutosaveFailCountRef.current >= 2) {
        toast('warning', 'Autoguardado', `No se pudo autoguardar draft: ${errMsg(err)}`)
      }
      return false
    } finally {
      scribbleAutosaveInFlightRef.current = false
    }
  }

  async function saveScribblesNow() {
    const ok = await saveScribbleDraft({ silent: false })
    if (ok) {
      await refreshSavedImages()
      await refreshModelDataset()
    }
  }

  async function clearScribbleDraftRemote() {
    if (!sessionId || !imageId) return
    try {
      await apiPost('/api/scribble/draft/clear', { session_id: sessionId, image_id: imageId })
      setDraftInfo('Draft limpiado')
      await refreshSavedImages()
      await refreshModelDataset()
    } catch (err) {
      toast('warning', 'Draft', `No se pudo limpiar draft remoto: ${errMsg(err)}`)
    }
  }

  async function loadScribbleDraftForImage(targetImageId) {
    const sid = String(targetImageId || '').trim()
    if (!sessionId || !sid) return
    try {
      const res = await apiGet(`/api/scribble/draft/load?session_id=${encodeURIComponent(sessionId)}&image_id=${encodeURIComponent(sid)}`)
      const payload = res?.payload || {}
      if (!payload?.found || !payload?.scribble_map_b64) {
        setDraftInfo('')
        return
      }
      await restoreScribbleMapFromB64(payload.scribble_map_b64)
      scribbleAutosaveDirtyRef.current = false
      const ts = String(payload?.meta?.updated_at || '')
      setDraftInfo(ts ? `Draft restaurado: ${ts}` : 'Draft restaurado')
    } catch (err) {
      toast('warning', 'Draft', `No se pudo cargar draft: ${errMsg(err)}`)
    }
  }

  async function flushDraftIfNeeded() {
    if (!scribbleAutosaveDirtyRef.current) return
    await saveScribbleDraft({ silent: true })
  }

  async function bootstrap() {
    await withLoad('boot', async () => {
      try {
        const s = await apiPost('/api/session/new', {})
        const sid = String(s?.payload?.session_id || '')
        setSessionId(sid)

        const c = await apiGet('/api/experiments/catalog')
        const exps = Array.isArray(c?.payload?.experiments) ? c.payload.experiments : []
        setExperiments(exps)
        if (exps.length > 0) {
          setSelectedExperiment(exps[0].experiment_id)
          const picked = {}
          exps.forEach((x) => { picked[x.experiment_id] = false })
          setSelectedBatch(picked)
        }
        await refreshSavedImages(sid)
        await refreshLocalImagePrefs()
        await refreshAssistModels()
        await refreshModelDataset(sid)
        toast('success', 'Scribble Research', 'Sesion lista. Carga una imagen y ejecuta experimentos A-E.')
      } catch (err) {
        toast('error', 'Error de inicio', errMsg(err))
      }
    })
  }

  async function ensureSessionReady() {
    if (sessionId) return sessionId
    try {
      const s = await apiPost('/api/session/new', {})
      const sid = String(s?.payload?.session_id || '')
      if (!sid) {
        toast('error', 'Sesion', 'No se pudo crear sesion.')
        return ''
      }
      setSessionId(sid)
      if (!experiments.length) {
        const c = await apiGet('/api/experiments/catalog')
        const exps = Array.isArray(c?.payload?.experiments) ? c.payload.experiments : []
        setExperiments(exps)
        if (exps.length > 0) {
          setSelectedExperiment(exps[0].experiment_id)
          const picked = {}
          exps.forEach((x) => { picked[x.experiment_id] = false })
          setSelectedBatch(picked)
        }
      }
      await refreshSavedImages(sid)
      await refreshLocalImagePrefs()
      await refreshAssistModels()
      await refreshModelDataset(sid)
      return sid
    } catch (err) {
      toast('error', 'Sesion', errMsg(err))
      return ''
    }
  }

  useEffect(() => {
    bootstrap()
  }, [])

  useEffect(() => {
    if (workspaceTab !== 'models') return
    refreshAssistModels()
    refreshModelDataset()
  }, [workspaceTab])

  useEffect(() => {
    if (workspaceTab !== 'locoAugment') return
    refreshLocoAugItems()
  }, [workspaceTab])

  useEffect(() => {
    if (!['locoTest', 'locoModel'].includes(workspaceTab)) return
    refreshLocoTrainingRuns()
  }, [workspaceTab])

  useEffect(() => {
    let cancelled = false
    if (!diamVisualMaskUrl) {
      setDiamMaskLayerUrl('')
      return () => { cancelled = true }
    }
    buildGreenMaskLayer(diamVisualMaskUrl)
      .then((url) => {
        if (!cancelled) setDiamMaskLayerUrl(url)
      })
      .catch(() => {
        if (!cancelled) setDiamMaskLayerUrl('')
      })
    return () => { cancelled = true }
  }, [diamVisualMaskUrl])

  useEffect(() => {
    return () => {
      if (localObjectUrlRef.current) {
        try { URL.revokeObjectURL(localObjectUrlRef.current) } catch {}
        localObjectUrlRef.current = ''
      }
    }
  }, [])

  useEffect(() => {
    const stage = editorStageRef.current
    if (!stage) return
    const update = () => {
      const r = stage.getBoundingClientRect()
      setStageSize({ w: r.width, h: r.height })
    }
    update()
    const ro = new ResizeObserver(update)
    ro.observe(stage)
    window.addEventListener('resize', update)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', update)
    }
  }, [imageUrl])

  useEffect(() => {
    const stage = editorStageRef.current
    if (!stage || workspaceTab !== 'workbench' || !imageUrl) return undefined
    const onNativeWheel = (ev) => {
      if (!ev.ctrlKey && !ev.metaKey) return
      ev.preventDefault()
      ev.stopPropagation()
      zoomEditorAt(ev.clientX, ev.clientY, ev.deltaY < 0 ? 1.12 : 1 / 1.12)
    }
    stage.addEventListener('wheel', onNativeWheel, { passive: false, capture: true })
    return () => stage.removeEventListener('wheel', onNativeWheel, { capture: true })
  }, [workspaceTab, imageUrl, viewerZoom, viewerOffset, stageSize, imageDims])

  useEffect(() => {
    const stage = diameterStageRef.current
    if (!stage) return
    const update = () => {
      const r = stage.getBoundingClientRect()
      setDiamStageSize({ w: r.width, h: r.height })
    }
    update()
    const ro = new ResizeObserver(update)
    ro.observe(stage)
    window.addEventListener('resize', update)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', update)
    }
  }, [imageUrl, workspaceTab])

  useEffect(() => {
    const stage = locoStageRef.current
    if (!stage) return
    const update = () => {
      const r = stage.getBoundingClientRect()
      setLocoStageSize({ w: r.width, h: r.height })
    }
    update()
    const ro = new ResizeObserver(update)
    ro.observe(stage)
    window.addEventListener('resize', update)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', update)
    }
  }, [imageUrl, workspaceTab])

  useEffect(() => {
    const stage = locoDatasetStageRef.current
    if (!stage) return
    const update = () => {
      const r = stage.getBoundingClientRect()
      setLocoDatasetStageSize({ w: r.width, h: r.height })
    }
    update()
    const ro = new ResizeObserver(update)
    ro.observe(stage)
    window.addEventListener('resize', update)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', update)
    }
  }, [imageUrl, workspaceTab])

  useEffect(() => {
    const stage = locoTestStageRef.current
    if (!stage) return
    const update = () => {
      const r = stage.getBoundingClientRect()
      setLocoTestStageSize({ w: r.width, h: r.height })
    }
    update()
    const ro = new ResizeObserver(update)
    ro.observe(stage)
    window.addEventListener('resize', update)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', update)
    }
  }, [imageUrl, workspaceTab])

  useEffect(() => {
    const stage = locoModelStageRef.current
    if (!stage) return
    const update = () => {
      const r = stage.getBoundingClientRect()
      setLocoModelStageSize({ w: r.width, h: r.height })
    }
    update()
    const ro = new ResizeObserver(update)
    ro.observe(stage)
    window.addEventListener('resize', update)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', update)
    }
  }, [imageUrl, workspaceTab])

  useEffect(() => {
    const stage = diameterStageRef.current
    if (!stage || workspaceTab !== 'diameter' || !imageUrl) return undefined
    const onNativeWheel = (ev) => {
      if (!ev.ctrlKey && !ev.metaKey) return
      ev.preventDefault()
      ev.stopPropagation()
      zoomDiameterAt(ev.clientX, ev.clientY, ev.deltaY < 0 ? 1.12 : 1 / 1.12)
    }
    stage.addEventListener('wheel', onNativeWheel, { passive: false, capture: true })
    return () => stage.removeEventListener('wheel', onNativeWheel, { capture: true })
  }, [workspaceTab, imageUrl, diamViewerZoom, diamViewerOffset, diamStageSize, imageDims])

  useEffect(() => {
    const stage = locoStageRef.current
    if (!stage || workspaceTab !== 'loco' || !imageUrl) return undefined
    const onNativeWheel = (ev) => {
      if (!ev.ctrlKey && !ev.metaKey) return
      ev.preventDefault()
      ev.stopPropagation()
      zoomLocoAt(ev.clientX, ev.clientY, ev.deltaY < 0 ? 1.12 : 1 / 1.12)
    }
    stage.addEventListener('wheel', onNativeWheel, { passive: false, capture: true })
    return () => stage.removeEventListener('wheel', onNativeWheel, { capture: true })
  }, [workspaceTab, imageUrl, locoViewerZoom, locoViewerOffset, locoStageSize, imageDims])

  useEffect(() => {
    const stage = locoDatasetStageRef.current
    if (!stage || workspaceTab !== 'locoDataset' || !imageUrl) return undefined
    const onNativeWheel = (ev) => {
      if (!ev.ctrlKey && !ev.metaKey) return
      ev.preventDefault()
      ev.stopPropagation()
      zoomLocoDatasetAt(ev.clientX, ev.clientY, ev.deltaY < 0 ? 1.12 : 1 / 1.12)
    }
    stage.addEventListener('wheel', onNativeWheel, { passive: false, capture: true })
    return () => stage.removeEventListener('wheel', onNativeWheel, { capture: true })
  }, [workspaceTab, imageUrl, locoDatasetZoom, locoDatasetOffset, locoDatasetStageSize, imageDims])

  useEffect(() => {
    const stage = locoTestStageRef.current
    if (!stage || workspaceTab !== 'locoTest' || !imageUrl) return undefined
    const onWheel = (e) => {
      if (!e.ctrlKey) return
      e.preventDefault()
      zoomLocoTestAt(e.clientX, e.clientY, e.deltaY < 0 ? 1.12 : 0.89)
    }
    stage.addEventListener('wheel', onWheel, { passive: false })
    return () => stage.removeEventListener('wheel', onWheel)
  }, [workspaceTab, imageUrl, locoTestZoom, locoTestOffset, locoTestStageSize, imageDims])

  useEffect(() => {
    const stage = locoModelStageRef.current
    if (!stage || workspaceTab !== 'locoModel' || !imageUrl) return undefined
    const onWheel = (e) => {
      if (!e.ctrlKey && !e.metaKey) return
      e.preventDefault()
      e.stopPropagation()
      zoomLocoModelAt(e.clientX, e.clientY, e.deltaY < 0 ? 1.12 : 0.89)
    }
    stage.addEventListener('wheel', onWheel, { passive: false, capture: true })
    return () => stage.removeEventListener('wheel', onWheel, { capture: true })
  }, [workspaceTab, imageUrl, locoModelZoom, locoModelOffset, locoModelStageSize, imageDims])

  useEffect(() => {
    if (!pointReviewOpen) return undefined
    const stage = pointReviewStageRef.current
    if (!stage) return undefined
    const update = () => {
      const r = stage.getBoundingClientRect()
      setPointReviewStageSize({ w: r.width, h: r.height })
    }
    update()
    const ro = new ResizeObserver(update)
    ro.observe(stage)
    window.addEventListener('resize', update)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', update)
    }
  }, [pointReviewOpen])

  useEffect(() => {
    if (!pointReviewOpen || !diamReviewTarget?.point || !imageDims.w || !imageDims.h) return
    if (pointReviewStageSize.w < 1 || pointReviewStageSize.h < 1) return
    const fit = Math.min(pointReviewStageSize.w / imageDims.w, pointReviewStageSize.h / imageDims.h)
    const zoom = 2
    const scale = fit * zoom
    const p = diamReviewTarget.point
    setPointReviewZoom(zoom)
    setPointReviewOffset({
      x: imageDims.w * scale * 0.5 - Number(p.x || 0) * scale,
      y: imageDims.h * scale * 0.5 - Number(p.y || 0) * scale,
    })
  }, [pointReviewOpen, diamReviewTarget?.point_index, pointReviewStageSize.w, pointReviewStageSize.h, imageDims.w, imageDims.h])

  useEffect(() => {
    if (workspaceTab !== 'review') return undefined
    const onKey = (ev) => {
      const tag = String(document.activeElement?.tagName || '').toUpperCase()
      const typing = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
      const k = String(ev.key || '')

      if (k === 'ArrowRight') {
        ev.preventDefault()
        goNext()
        return
      }
      if (k === 'ArrowLeft') {
        ev.preventDefault()
        goPrev()
        return
      }
      if (typing) return

      if (k === 's' || k === 'S') {
        ev.preventDefault()
        mark('s')
      } else if (k === 'a' || k === 'A') {
        ev.preventDefault()
        mark('a')
      } else if (k === 'b' || k === 'B') {
        ev.preventDefault()
        mark('b')
      } else if (k === 'c' || k === 'C') {
        ev.preventDefault()
        mark('c')
      } else if (k === 'u' || k === 'U') {
        ev.preventDefault()
        mark('unusable')
      } else if (k === 'r' || k === 'R') {
        ev.preventDefault()
        setFilterGroup('all')
        setFilterExperiment('all')
        setFilterDecision('all')
      } else if (k === 'f' || k === 'F') {
        ev.preventDefault()
        filterFocusRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [workspaceTab, filteredRuns, activeRunId, reviews, imageId])

  useEffect(() => {
    if (workspaceTab !== 'workbench') return undefined
    const onKey = (ev) => {
      const tag = String(document.activeElement?.tagName || '').toUpperCase()
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return
      const k = String(ev.key || '').toLowerCase()
      if ((ev.ctrlKey || ev.metaKey) && k === 'z') {
        ev.preventDefault()
        onAnnotUndo()
      } else if ((ev.ctrlKey || ev.metaKey) && k === 'y') {
        ev.preventDefault()
        onAnnotRedo()
      } else if (k === 'l') {
        ev.preventDefault()
        setViewerMode('mark')
      } else if (k === 'm') {
        ev.preventDefault()
        setViewerMode('pan')
      } else if (k === 'g') {
        ev.preventDefault()
        setViewerMode('mark')
        setTool('erase')
      } else if (k === 'r') {
        ev.preventDefault()
        setViewerMode('exclude')
      } else if (k === 'f') {
        ev.preventDefault()
        setViewerMode('mark')
        setTool('fiber')
      } else if (k === 'h') {
        ev.preventDefault()
        setViewerMode('mark')
        setTool('halo')
      } else if (k === 'b') {
        ev.preventDefault()
        setViewerMode('mark')
        setTool('bg')
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [workspaceTab, annotHistory, annotFuture])

  useEffect(() => {
    if (workspaceTab !== 'diameter') return undefined
    const onKey = (ev) => {
      const tag = String(document.activeElement?.tagName || '').toUpperCase()
      const typing = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
      const k = String(ev.key || '').toLowerCase()
      if ((ev.ctrlKey || ev.metaKey) && k === 'z') {
        ev.preventDefault()
        undoDiameterManual()
        return
      }
      if ((ev.ctrlKey || ev.metaKey) && k === 'y') {
        ev.preventDefault()
        redoDiameterManual()
        return
      }
      if (!typing && (k === 'delete' || k === 'backspace')) {
        if (manualCircleSelected && (manualCircleDraft || manualCircleActiveIdx >= 0)) {
          ev.preventDefault()
          clearDiameterManualCircle()
          return
        }
        if (diamViewerMode === 'manual' && manualLineDraft?.start) {
          ev.preventDefault()
          setDiameterManualState({ start: null, end: null }, manualCircleDraft)
          return
        }
        if (diamActivePointIdx >= 0) {
          ev.preventDefault()
          void updateDiameterPoints('remove_active')
          return
        }
      }
      if (typing) return
      if (k === 'l') {
        ev.preventDefault()
        if (diamMethodId === 'circle_square_mask_diameter') {
          setDiamViewerMode('circle')
          toast('warning', 'Circle-square', 'Este metodo usa solo circulo manual.')
          return
        }
        setDiamViewerMode('mark')
      } else if (k === 'm') {
        ev.preventDefault()
        setDiamViewerMode('pan')
      } else if (k === 'c') {
        ev.preventDefault()
        setDiamMethodPanel('manual')
        setDiamMethodId('circle_square_mask_diameter')
        setDiamViewerMode('circle')
        updateDiamRawParam('circle_square_seed_mode', 'manual_circle')
      } else if (k === 'd') {
        ev.preventDefault()
        setDiamMethodPanel('manual')
        setDiamMethodId('manual_dual_side_caliper')
        setDiamViewerMode('manual')
      } else if (k === 'x') {
        ev.preventDefault()
        setDiamMethodPanel('manual')
        setDiamMethodId('manual_line_direct_caliper')
        setDiamViewerMode('manual')
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [workspaceTab, diamMethodId, diamViewerMode, diamManualHistory, diamManualFuture, manualMaskLineDraft, manualDirectLineDraft, manualCircleDraft, manualCircles, manualCircleActiveIdx, manualCircleSelected, diamPoints, diamActivePointIdx])

  useEffect(() => {
    if (workspaceTab !== 'loco') return undefined
    const onKey = (ev) => {
      const tag = String(document.activeElement?.tagName || '').toUpperCase()
      const typing = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
      const k = String(ev.key || '').toLowerCase()
      if (typing) return
      if (k === 'm') {
        ev.preventDefault()
        setLocoViewerMode('pan')
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [workspaceTab])

  useEffect(() => {
    setLocoProposals([])
    setLocoFilteredProposals([])
    setLocoMeasuredResults([])
    setLocoEvaluation(null)
    setLocoSelectedProposalId('')
    setLocoLabStage('proposals')
  }, [imageId])

  useEffect(() => {
    const row = reviewByRun[activeRunId]
    setReviewNote(row ? String(row.note || '') : '')
  }, [activeRunId, reviewByRun])

  useEffect(() => {
    if (!sessionId) return undefined
    const timer = window.setInterval(() => {
      if (!imageUrl || !imageId) return
      if (!scribbleAutosaveDirtyRef.current) return
      if (scribbleAutosaveInFlightRef.current) return
      void saveScribbleDraft({ silent: true })
    }, 30000)
    return () => window.clearInterval(timer)
  }, [sessionId, imageUrl, imageId])

  async function onLoadImage(e) {
    const file = e.target.files?.[0]
    if (!file) return
    const sid = await ensureSessionReady()
    if (!sid) {
      e.target.value = ''
      return
    }
    const gt = document.getElementById('gt-file-input')?.files?.[0]
    const reqId = ++loadReqIdRef.current
    await withLoad('loadImage', async () => {
      try {
        // Same UX principle as nano4: immediate local preview on choose-file.
        const previewOk = await applyImageFilePreview(file)
        setImageName(String(file.name || ''))
        if (!previewOk) {
          toast('warning', 'Preview local', 'No se pudo previsualizar localmente. Se cargara desde backend.')
        }

        const form = new FormData()
        form.append('session_id', sid)
        form.append('file', file)
        form.append('scale_percent', '100')
        if (gt) form.append('gt_file', gt)
        const res = await apiForm('/api/image/load', form)
        if (reqId !== loadReqIdRef.current) return
        const p = res.payload || {}
        const backendApplied = await applyImageB64(p.image_b64, String(p.image_mime || 'image/png'))
        if (!backendApplied) {
          toast('warning', 'Carga parcial', 'Vista local activa. El backend no devolvio imagen renderizable.')
        }
        const loadedImageId = String(p.image_id || '')
        setImageId(loadedImageId)
        setSelectedSavedImageId(loadedImageId)
        setImageName(String(p.image_name || file.name || ''))
        resetImageScopedState()
        await refreshSavedImages(sid)
        await refreshModelDataset(sid)
        await refreshResults(loadedImageId)
        await loadScribbleDraftForImage(loadedImageId)
        await loadDiameterPoints(sid, loadedImageId, { silent: true })
        await refreshDiameterRuns(loadedImageId, { silent: true })
        await refreshValidationCases(loadedImageId, { silent: true })
        toast('success', 'Imagen cargada', `Activa: ${p.image_id || '-'} (${p.image_shape?.[1] || '?'}x${p.image_shape?.[0] || '?'})`)
      } catch (err) {
        if (reqId !== loadReqIdRef.current) return
        toast('error', 'Carga de imagen', errMsg(err))
      } finally {
        e.target.value = ''
      }
    })
  }

  async function runOne() {
    if (!sessionId || !imageUrl || !selectedExperiment) {
      toast('warning', 'Ejecucion', 'Falta imagen o experimento seleccionado.')
      return
    }
    await flushDraftIfNeeded()
    const scribble = currentScribbleB64()
    await withLoad('run', async () => {
      try {
        const res = await apiPost('/api/experiments/run', {
          session_id: sessionId,
          experiment_id: selectedExperiment,
          params: { __profile_name: 'single' },
          scribble_map_b64: scribble,
          exclude_rect: getExcludeRectPayload(),
          save_mode: segSaveMode,
        })
        const item = res.payload || {}
        setRunCache((prev) => ({ ...prev, [item.run_id]: item }))
        setActiveRunId(item.run_id)
        setImageId(String(item.image_id || imageId))
        if (segSaveMode === 'overwrite') {
          await refreshResults(String(item.image_id || imageId))
        } else {
          setRuns((prev) => {
            const row = buildRunRow(item)
            if (!experimentsById[row.experiment_id]) return prev
            return [row, ...prev]
          })
        }
        toast(res.status_level || 'success', 'Experimento', `${item.experiment_id} ejecutado (${item.run_status_level || 'success'}).`)
      } catch (err) {
        toast('error', 'Ejecucion', errMsg(err))
      }
    })
  }

  async function runBatch() {
    if (!sessionId || !imageUrl) {
      toast('warning', 'Batch', 'Carga una imagen primero.')
      return
    }
    const picked = Object.keys(selectedBatch).filter((k) => selectedBatch[k])
    if (!picked.length) {
      toast('warning', 'Batch', 'Selecciona al menos un experimento.')
      return
    }
    await flushDraftIfNeeded()
    const scribble = currentScribbleB64()
    await withLoad('runBatch', async () => {
      const perExperimentPlanned = 1
      const totalPlanned = picked.length * perExperimentPlanned
      let done = 0
      let allItems = []
      let allWarnings = []
      let allFailures = []

      setBatchProgress({ active: true, done: 0, total: totalPlanned, current: 'Inicializando...' })
      try {
        for (const eid of picked) {
          setBatchProgress((prev) => ({ ...prev, current: eid, done }))
          try {
            const res = await apiPost('/api/experiments/run-batch', {
              session_id: sessionId,
              experiment_ids: [eid],
              params: {},
              params_by_experiment: {},
              param_sweep: 'high',
              scribble_map_b64: scribble,
              exclude_rect: getExcludeRectPayload(),
              save_mode: segSaveMode,
            })
            const items = Array.isArray(res?.payload?.items) ? res.payload.items : []
            const warnings = Array.isArray(res?.payload?.warnings) ? res.payload.warnings : []
            const failures = Array.isArray(res?.payload?.failures) ? res.payload.failures : []
            allItems = allItems.concat(items)
            allWarnings = allWarnings.concat(warnings)
            allFailures = allFailures.concat(failures)
            items.forEach((it) => {
              setRunCache((prev) => ({ ...prev, [it.run_id]: it }))
              if (segSaveMode !== 'overwrite') {
                setRuns((prev) => {
                  const row = buildRunRow(it)
                  if (!experimentsById[row.experiment_id]) return prev
                  return [row, ...prev]
                })
              }
            })
          } catch (err) {
            allFailures.push({ experiment_id: eid, profile: '*', error: errMsg(err) })
          }
          done = Math.min(totalPlanned, done + perExperimentPlanned)
          setBatchProgress((prev) => ({ ...prev, done }))
        }

        if (!allItems.length) {
          toast('error', 'Batch', 'No se ejecutaron corridas validas.')
          return
        }
        setActiveRunId(allItems[0].run_id)
        setImageId(String(allItems[0].image_id || imageId))
        if (segSaveMode === 'overwrite') {
          await refreshResults(String(allItems[0].image_id || imageId))
        }
        let msg = `${allItems.length} corridas ejecutadas.`
        if (allWarnings.length || allFailures.length) {
          msg = `${allItems.length} OK | ${allWarnings.length} warnings | ${allFailures.length} fallas.`
        }
        toast(allFailures.length ? 'warning' : 'success', 'Batch finalizado', msg)
      } finally {
        setBatchProgress((prev) => ({ ...prev, active: false, current: '', done: prev.total }))
      }
    })
  }

  function selectAllBatch() {
    if (!experiments.length) return
    const next = {}
    experiments.forEach((x) => { next[x.experiment_id] = true })
    setSelectedBatch(next)
  }

  function clearBatchSelection() {
    setSelectedBatch({})
  }

  async function refreshSavedImages(customSessionId = '') {
    const sid = String(customSessionId || sessionId || '').trim()
    await withLoad('libraryList', async () => {
      try {
        const path = sid ? `/api/library/images?session_id=${encodeURIComponent(sid)}` : '/api/library/images'
        const res = await apiGet(path)
        const items = Array.isArray(res?.payload?.items) ? res.payload.items : []
        setSavedImages(items)
        setSelectedSavedImageId((prev) => {
          if (items.some((it) => it.image_id === prev)) return prev
          if (items.some((it) => it.image_id === imageId)) return imageId
          return ''
        })
      } catch (err) {
        toast('warning', 'Imagenes guardadas', `No se pudo listar biblioteca: ${errMsg(err)}`)
      }
    })
  }

  async function refreshLocalImagePrefs() {
    await withLoad('localPrefs', async () => {
      try {
        const res = await apiGet('/api/local-images/prefs')
        setImageStartDir(String(res?.payload?.start_dir || ''))
      } catch (err) {
        toast('warning', 'Ruta inicial', `No se pudo cargar ruta inicial: ${errMsg(err)}`)
      }
    })
  }

  async function saveLocalImagePrefs() {
    const path = String(imageStartDir || '').trim()
    if (!path) {
      toast('warning', 'Ruta inicial', 'Escribe una ruta de carpeta.')
      return
    }
    await withLoad('localPrefs', async () => {
      try {
        const res = await apiPost('/api/local-images/prefs', { start_dir: path })
        setImageStartDir(String(res?.payload?.start_dir || path))
        toast('success', 'Ruta inicial', 'Ruta guardada.')
      } catch (err) {
        toast('warning', 'Ruta inicial', errMsg(err))
      }
    })
  }

  async function listLocalImages() {
    const path = String(imageStartDir || '').trim()
    if (!path) {
      toast('warning', 'Imagenes locales', 'Guarda o escribe una ruta inicial primero.')
      return
    }
    await withLoad('localImages', async () => {
      try {
        const res = await apiGet(`/api/local-images/list?start_dir=${encodeURIComponent(path)}&recursive=true&limit=600`)
        const items = Array.isArray(res?.payload?.items) ? res.payload.items : []
        setLocalImageFiles(items)
        setSelectedLocalPath((prev) => (items.some((it) => it.path === prev) ? prev : String(items[0]?.path || '')))
        toast('success', 'Imagenes locales', `${items.length} imagenes encontradas.`)
      } catch (err) {
        toast('warning', 'Imagenes locales', errMsg(err))
      }
    })
  }

  async function openFolder(kind = 'outputs', customPath = '') {
    await withLoad('openFolder', async () => {
      try {
        const res = await apiPost('/api/system/open-folder', { kind, path: customPath })
        toast('success', 'Carpeta', String(res?.payload?.path || 'Carpeta abierta.'))
      } catch (err) {
        toast('warning', 'Carpeta', errMsg(err))
      }
    })
  }

  async function loadLocalImage(pathToLoad = '') {
    const sid = await ensureSessionReady()
    const target = String(pathToLoad || selectedLocalPath || '').trim()
    if (!sid || !target) return
    await withLoad('localLoad', async () => {
      try {
        const res = await apiPost('/api/local-images/load', {
          session_id: sid,
          path: target,
          scale_percent: 100,
        })
        const p = res.payload || {}
        const ok = await applyImageB64(p.image_b64, String(p.image_mime || 'image/png'))
        if (!ok) throw new Error('No se pudo renderizar imagen local.')
        const loadedImageId = String(p.image_id || '')
        setImageId(loadedImageId)
        setSelectedSavedImageId(loadedImageId)
        setImageName(String(p.image_name || 'image'))
        setImageStartDir(String(target).replace(/[\\/][^\\/]*$/, ''))
        resetImageScopedState()
        await refreshSavedImages(sid)
        await refreshModelDataset(sid)
        await refreshResults(loadedImageId)
        await loadScribbleDraftForImage(loadedImageId)
        await loadDiameterPoints(sid, loadedImageId, { silent: true })
        await refreshDiameterRuns(loadedImageId, { silent: true })
        await refreshValidationCases(loadedImageId, { silent: true })
        toast('success', 'Imagen local', `Activa: ${loadedImageId}`)
      } catch (err) {
        toast('error', 'Imagen local', errMsg(err))
      }
    })
  }

  async function deleteSelectedSavedImage() {
    const sid = await ensureSessionReady()
    const target = String(selectedSavedImageId || imageId || '').trim()
    if (!sid || !target) return
    await withLoad('libraryDelete', async () => {
      try {
        await apiPost('/api/library/delete', { session_id: sid, image_id: target })
        if (target === imageId) {
          setImageId('')
          setImageName('')
          setImageUrl('')
          setSelectedSavedImageId('')
          resetImageScopedState()
          resetScribbleDraftState()
          resetAnnotHistory()
        }
        await refreshSavedImages(sid)
        await refreshModelDataset(sid)
        toast('success', 'Imagenes guardadas', 'Imagen seleccionada eliminada.')
      } catch (err) {
        toast('error', 'Imagenes guardadas', errMsg(err))
      }
    })
  }

  async function refreshModelDataset(customSessionId = '') {
    const sid = String(customSessionId || sessionId || '').trim()
    await withLoad('modelsDataset', async () => {
      try {
        const path = sid ? `/api/assist-models/dataset/images?session_id=${encodeURIComponent(sid)}` : '/api/assist-models/dataset/images'
        const res = await apiGet(path)
        const items = Array.isArray(res?.items) ? res.items : []
        setModelDataset(items)
        setSelectedTrainImages((prev) => {
          const valid = {}
          items.forEach((item) => {
            const id = String(item?.image_id || '')
            if (id && prev[id]) valid[id] = true
          })
          return valid
        })
      } catch (err) {
        toast('warning', 'Modelos', `No se pudo listar dataset: ${errMsg(err)}`)
      }
    })
  }

  async function refreshAssistModels() {
    await withLoad('modelsList', async () => {
      try {
        const res = await apiGet('/api/assist-models/list')
        const items = Array.isArray(res?.models) ? res.models : []
        const defaultId = String(res?.default_model_id || '')
        setAssistModels(items)
        setDefaultAssistModelId(defaultId)
        setSelectedAssistModelId((prev) => {
          if (items.some((m) => m.model_id === prev)) return prev
          return defaultId || String(items[0]?.model_id || '')
        })
      } catch (err) {
        toast('warning', 'Modelos', `No se pudo listar modelos: ${errMsg(err)}`)
      }
    })
  }

  function toggleTrainImage(imageIdToToggle) {
    const id = String(imageIdToToggle || '')
    if (!id) return
    setSelectedTrainImages((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  function selectTrainableImages() {
    const next = {}
    modelDataset.forEach((item) => {
      if (item.trainable_multiclass || item.trainable_binary) next[item.image_id] = true
    })
    setSelectedTrainImages(next)
  }

  function clearTrainImages() {
    setSelectedTrainImages({})
  }

  function updateTrainConfig(key, value) {
    setTrainConfig((prev) => ({ ...prev, [key]: value }))
  }

  async function trainAssistModel() {
    const sid = await ensureSessionReady()
    if (!sid) return
    if (!trainImageIds.length) {
      toast('warning', 'Modelos', 'Selecciona imagenes con scribbles para entrenar.')
      return
    }
    await withLoad('modelsTrain', async () => {
      try {
        const res = await apiPost('/api/assist-models/train', {
          session_id: sid,
          model_name: trainConfig.model_name || '',
          image_ids: trainImageIds,
          class_mode: trainConfig.class_mode || 'multiclass',
          classifier: trainConfig.classifier || 'extratrees',
          feature_variant: trainConfig.feature_variant || 'context',
          n_estimators: Number(trainConfig.n_estimators || 120),
          notes: trainConfig.notes || '',
        })
        const model = res?.model || {}
        setModelTrainSummary(res?.meta || model)
        await refreshAssistModels()
        setSelectedAssistModelId(String(model.model_id || res?.default_model_id || ''))
        toast('success', 'Modelo entrenado', `${model.model_name || model.model_id || 'modelo'} listo para usar.`)
      } catch (err) {
        toast('error', 'Entrenamiento', errMsg(err))
      }
    })
  }

  async function setDefaultAssistModel(modelIdToSet) {
    const mid = String(modelIdToSet || '').trim()
    if (!mid) return
    await withLoad('modelsList', async () => {
      try {
        const res = await apiPost('/api/assist-models/set-default', { model_id: mid })
        const items = Array.isArray(res?.models) ? res.models : []
        setAssistModels(items)
        setDefaultAssistModelId(String(res?.default_model_id || mid))
        setSelectedAssistModelId(mid)
        toast('success', 'Modelo activo', 'Modelo marcado como predeterminado.')
      } catch (err) {
        toast('warning', 'Modelos', errMsg(err))
      }
    })
  }

  async function deleteAssistModel(modelIdToDelete) {
    const mid = String(modelIdToDelete || '').trim()
    if (!mid) return
    await withLoad('modelsDelete', async () => {
      try {
        const res = await apiPost('/api/assist-models/delete', { model_id: mid })
        const items = Array.isArray(res?.models) ? res.models : []
        setAssistModels(items)
        setDefaultAssistModelId(String(res?.default_model_id || ''))
        setSelectedAssistModelId(String(res?.default_model_id || items[0]?.model_id || ''))
        setModelPrediction(null)
        toast('success', 'Modelo eliminado', mid)
      } catch (err) {
        toast('warning', 'Modelos', errMsg(err))
      }
    })
  }

  async function predictWithAssistModel() {
    const sid = await ensureSessionReady()
    if (!sid || !imageId) return
    const mid = String(selectedAssistModelId || defaultAssistModelId || '').trim()
    if (!mid) {
      toast('warning', 'Modelo guardado', 'Entrena o selecciona un modelo primero.')
      return
    }
    await withLoad('modelsPredict', async () => {
      try {
        const res = await apiPost('/api/assist-models/predict', {
          session_id: sid,
          image_id: imageId,
          model_id: mid,
          min_confidence: Number(modelMinConfidence || 0.72),
          include_fiber: !!modelIncludeFiber,
          include_halo: !!modelIncludeHalo,
          include_background: !!modelIncludeBackground,
        })
        const previewUrl = res?.preview_b64 ? b64ToDataUrl(res.preview_b64, res.preview_mime || 'image/png') : ''
        setModelPrediction({ ...res, previewUrl })
        toast('success', 'Prediccion lista', `F${res?.counts?.fiber || 0} H${res?.counts?.halo || 0} B${res?.counts?.background || 0}`)
      } catch (err) {
        toast('warning', 'Modelo guardado', errMsg(err))
      }
    })
  }

  async function applyModelPredictionAsScribbles() {
    if (!modelPrediction?.suggestion_b64 || !imageUrl) return
    const snap = captureAnnotSnapshot()
    try {
      const changed = await mergeScribbleMapFromB64(modelPrediction.suggestion_b64)
      if (snap && changed > 0) {
        setAnnotHistory((h) => [...h, snap].slice(-40))
        setAnnotFuture([])
      }
      markScribbleDirty()
      await saveScribbleDraft({ silent: true })
      toast('success', 'Scribbles agregados', `${changed} px agregados desde el modelo.`)
    } catch (err) {
      toast('warning', 'Modelo guardado', errMsg(err))
    }
  }

  function resetImageScopedState() {
    setRuns([])
    setRunCache({})
    setActiveRunId('')
    setReviews([])
    setReportInfo(null)
    setDiamPoints([])
    setDiamActivePointIdx(-1)
    setDiamOverlayUrl('')
    setDiamResults([])
    setDiamResultsMode('run')
    setDiamReviewTarget(null)
    setDiamRuns([])
    setDiamActiveRunId('')
    setDiamRunCache({})
    setDiamReportInfo(null)
    setDiamPriorRunId('latest')
    setDiamViewerMode('mark')
    setDiamViewerZoom(1)
    setDiamViewerOffset({ x: 0, y: 0 })
    setManualMaskLineDraft({ start: null, end: null })
    setManualDirectLineDraft({ start: null, end: null })
    setManualMaskLines([])
    setManualDirectLines([])
    setManualMaskLineActiveIdx(-1)
    setManualDirectLineActiveIdx(-1)
    setManualCircleDraft(null)
    setManualCircles([])
    setManualCircleActiveIdx(-1)
    setManualCircleSelected(false)
    setManualCircleConsumed(false)
    setValidationCases([])
    setValidationActiveCaseId('')
    setValidationExportInfo(null)
    setValidationForm({
      case_id: '',
      category: 'borde_limpio',
      manual_diameter_px: '',
      manual_left_x: '',
      manual_left_y: '',
      manual_right_x: '',
      manual_right_y: '',
      measurement_decision: 'unreviewed',
      quality_manual: 'medium',
      notes: '',
      result_comment: '',
    })
    setFilterGroup('all')
    setFilterExperiment('all')
    setFilterDecision('all')
    setReviewSort('latest')
    setModelPrediction(null)
    setLocoModelResult(null)
    setLocoModelMeasurement(null)
    setLocoModelSelectedId('')
  }

  async function loadSavedImage(imageIdToLoad) {
    const sid = await ensureSessionReady()
    const target = String(imageIdToLoad || '').trim()
    if (!sid || !target) return
    await flushDraftIfNeeded()
    await withLoad('libraryLoad', async () => {
      try {
        const res = await apiPost('/api/library/load', {
          session_id: sid,
          image_id: target,
          restore_scribbles: true,
        })
        const p = res.payload || {}
        const ok = await applyImageB64(p.image_b64, String(p.image_mime || 'image/png'))
        if (!ok) throw new Error('No se pudo renderizar imagen guardada.')
        const loadedImageId = String(p.image_id || target)
        setImageId(loadedImageId)
        setSelectedSavedImageId(loadedImageId)
        setImageName(String(p.image_name || target))
        resetImageScopedState()
        const draft = p.scribble_draft || {}
        if (draft?.found && draft?.scribble_map_b64) {
          await restoreScribbleMapFromB64(draft.scribble_map_b64)
          scribbleAutosaveDirtyRef.current = false
          const ts = String(draft?.meta?.updated_at || '')
          setDraftInfo(ts ? `Draft restaurado: ${ts}` : 'Draft restaurado')
        }
        await refreshSavedImages(sid)
        await refreshResults(loadedImageId)
        await loadDiameterPoints(sid, loadedImageId, { silent: true })
        await refreshDiameterRuns(loadedImageId, { silent: true })
        await refreshValidationCases(loadedImageId, { silent: true })
        toast('success', 'Imagen guardada', `Cargada: ${loadedImageId}`)
      } catch (err) {
        toast('error', 'Imagen guardada', errMsg(err))
      }
    })
  }

  async function refreshResults(customImageId = '') {
    const iid = String(customImageId || imageId || '').trim()
    if (!iid) return
    await withLoad('listResults', async () => {
      try {
        const res = await apiGet(`/api/results/list?image_id=${encodeURIComponent(iid)}`)
        const items = (Array.isArray(res?.payload?.items) ? res.payload.items : []).filter((x) => Boolean(experimentsById[String(x?.experiment_id || '')]))
        setRuns(items)
        setActiveRunId((prev) => (items.some((r) => r.run_id === prev) ? prev : (items[0]?.run_id || '')))
      } catch (err) {
        toast('error', 'Resultados', errMsg(err))
      }
    })

    try {
      const rev = await apiGet(`/api/review/list?image_id=${encodeURIComponent(iid)}`)
      setReviews(Array.isArray(rev?.payload?.items) ? rev.payload.items : [])
    } catch {}
  }

  async function loadRunIfNeeded(runId) {
    if (!runId || runCache[runId]) return
    await withLoad('getResult', async () => {
      try {
        const res = await apiGet(`/api/results/get?run_id=${encodeURIComponent(runId)}`)
        const item = res.payload || {}
        setRunCache((prev) => ({ ...prev, [runId]: item }))
      } catch (err) {
        toast('error', 'Resultado', errMsg(err))
      }
    })
  }

  useEffect(() => {
    if (activeRunId) loadRunIfNeeded(activeRunId)
  }, [activeRunId])

  useEffect(() => {
    const priorRunId = diamPriorRunId === 'latest' ? (runs[0]?.run_id || '') : diamPriorRunId
    if (priorRunId) loadRunIfNeeded(priorRunId)
  }, [diamPriorRunId, runs])

  useEffect(() => {
    if (diamActiveRunId && diamResultsMode === 'run') loadDiameterRun(diamActiveRunId)
  }, [diamActiveRunId, diamResultsMode])

  function goPrev() {
    if (!filteredRuns.length) return
    const idx = filteredRuns.findIndex((r) => r.run_id === activeRunId)
    if (idx < 0) {
      setActiveRunId(filteredRuns[0].run_id)
      return
    }
    const next = filteredRuns[Math.max(0, idx - 1)]
    setActiveRunId(next.run_id)
  }

  function goNext() {
    if (!filteredRuns.length) return
    const idx = filteredRuns.findIndex((r) => r.run_id === activeRunId)
    if (idx < 0) {
      setActiveRunId(filteredRuns[0].run_id)
      return
    }
    const next = filteredRuns[Math.min(filteredRuns.length - 1, idx + 1)]
    setActiveRunId(next.run_id)
  }

  async function mark(decision) {
    if (!activeRunId || !imageId) return
    await withLoad('mark', async () => {
      try {
        await apiPost('/api/review/mark', {
          run_id: activeRunId,
          image_id: imageId,
          decision,
          note: reviewNote,
        })
        const rev = await apiGet(`/api/review/list?image_id=${encodeURIComponent(imageId)}`)
        setReviews(Array.isArray(rev?.payload?.items) ? rev.payload.items : [])
        toast('success', 'Review', `Marcado como ${reviewLabel(decision) || decision.toUpperCase()}`)
      } catch (err) {
        toast('error', 'Review', errMsg(err))
      }
    })
  }

  async function exportReport() {
    if (!imageId) return
    await withLoad('export', async () => {
      try {
        const res = await apiGet(`/api/reports/export?image_id=${encodeURIComponent(imageId)}`)
        setReportInfo(res?.payload || null)
        toast('success', 'Reporte', 'CSV + JSON + galeria generados.')
      } catch (err) {
        toast('error', 'Reporte', errMsg(err))
      }
    })
  }

  async function clearReviewResults() {
    if (!sessionId || !imageId) return
    const ok = window.confirm('Borrar todos los resultados de segmentacion y revisiones de esta imagen? No borra la imagen ni los scribbles.')
    if (!ok) return
    await withLoad('clearResults', async () => {
      try {
        const res = await apiPost('/api/results/clear', {
          session_id: sessionId,
          image_id: imageId,
        })
        setRuns([])
        setRunCache({})
        setActiveRunId('')
        setReviews([])
        setReportInfo(null)
        setDiamPriorRunId('latest')
        await refreshSavedImages()
        toast('success', 'Resultados', `${res?.payload?.deleted_count || 0} runs borrados.`)
      } catch (err) {
        toast('error', 'Resultados', errMsg(err))
      }
    })
  }

  function getDiameterRenderMetrics() {
    const { w, h } = imageDims
    if (w < 1 || h < 1 || diamStageSize.w < 1 || diamStageSize.h < 1) return null
    const fit = Math.min(diamStageSize.w / w, diamStageSize.h / h)
    const scale = Math.max(0.0001, fit * diamViewerZoom)
    const contentW = w * scale
    const contentH = h * scale
    const x = (diamStageSize.w - contentW) * 0.5 + diamViewerOffset.x
    const y = (diamStageSize.h - contentH) * 0.5 + diamViewerOffset.y
    return { scale, x, y }
  }

  function getLocoRenderMetrics() {
    const { w, h } = imageDims
    if (w < 1 || h < 1 || locoStageSize.w < 1 || locoStageSize.h < 1) return null
    const fit = Math.min(locoStageSize.w / w, locoStageSize.h / h)
    const scale = Math.max(0.0001, fit * locoViewerZoom)
    const contentW = w * scale
    const contentH = h * scale
    const x = (locoStageSize.w - contentW) * 0.5 + locoViewerOffset.x
    const y = (locoStageSize.h - contentH) * 0.5 + locoViewerOffset.y
    return { scale, x, y }
  }

  function getLocoDatasetRenderMetrics() {
    const { w, h } = imageDims
    if (w < 1 || h < 1 || locoDatasetStageSize.w < 1 || locoDatasetStageSize.h < 1) return null
    const fit = Math.min(locoDatasetStageSize.w / w, locoDatasetStageSize.h / h)
    const scale = Math.max(0.0001, fit * locoDatasetZoom)
    const contentW = w * scale
    const contentH = h * scale
    const x = (locoDatasetStageSize.w - contentW) * 0.5 + locoDatasetOffset.x
    const y = (locoDatasetStageSize.h - contentH) * 0.5 + locoDatasetOffset.y
    return { scale, x, y }
  }

  function getLocoTestRenderMetrics() {
    const { w, h } = imageDims
    if (w < 1 || h < 1 || locoTestStageSize.w < 1 || locoTestStageSize.h < 1) return null
    const fit = Math.min(locoTestStageSize.w / w, locoTestStageSize.h / h)
    const scale = Math.max(0.0001, fit * locoTestZoom)
    const contentW = w * scale
    const contentH = h * scale
    const x = (locoTestStageSize.w - contentW) * 0.5 + locoTestOffset.x
    const y = (locoTestStageSize.h - contentH) * 0.5 + locoTestOffset.y
    return { scale, x, y }
  }

  function getLocoModelRenderMetrics() {
    const { w, h } = imageDims
    if (w < 1 || h < 1 || locoModelStageSize.w < 1 || locoModelStageSize.h < 1) return null
    const fit = Math.min(locoModelStageSize.w / w, locoModelStageSize.h / h)
    const scale = Math.max(0.0001, fit * locoModelZoom)
    const contentW = w * scale
    const contentH = h * scale
    const x = (locoModelStageSize.w - contentW) * 0.5 + locoModelOffset.x
    const y = (locoModelStageSize.h - contentH) * 0.5 + locoModelOffset.y
    return { scale, x, y }
  }

  function diameterPointFromClient(clientX, clientY) {
    const stage = diameterStageRef.current
    if (!stage || !imageDims.w || !imageDims.h) return null
    const metrics = getDiameterRenderMetrics()
    if (!metrics) return null
    const rect = stage.getBoundingClientRect()
    if (rect.width < 1 || rect.height < 1) return null
    const x = (clientX - rect.left - metrics.x) / metrics.scale
    const y = (clientY - rect.top - metrics.y) / metrics.scale
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null
    if (x < 0 || y < 0 || x >= imageDims.w || y >= imageDims.h) return null
    return {
      x: clamp(x, 0, Math.max(0, imageDims.w - 1)),
      y: clamp(y, 0, Math.max(0, imageDims.h - 1)),
    }
  }

  function locoPointFromClient(clientX, clientY) {
    const stage = locoStageRef.current
    if (!stage || !imageDims.w || !imageDims.h) return null
    const metrics = getLocoRenderMetrics()
    if (!metrics) return null
    const rect = stage.getBoundingClientRect()
    if (rect.width < 1 || rect.height < 1) return null
    const x = (clientX - rect.left - metrics.x) / metrics.scale
    const y = (clientY - rect.top - metrics.y) / metrics.scale
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null
    if (x < 0 || y < 0 || x >= imageDims.w || y >= imageDims.h) return null
    return {
      x: clamp(x, 0, Math.max(0, imageDims.w - 1)),
      y: clamp(y, 0, Math.max(0, imageDims.h - 1)),
    }
  }

  function locoDatasetPointFromClient(clientX, clientY) {
    const stage = locoDatasetStageRef.current
    if (!stage || !imageDims.w || !imageDims.h) return null
    const metrics = getLocoDatasetRenderMetrics()
    if (!metrics) return null
    const rect = stage.getBoundingClientRect()
    if (rect.width < 1 || rect.height < 1) return null
    const x = (clientX - rect.left - metrics.x) / metrics.scale
    const y = (clientY - rect.top - metrics.y) / metrics.scale
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null
    if (x < 0 || y < 0 || x >= imageDims.w || y >= imageDims.h) return null
    return {
      x: clamp(x, 0, Math.max(0, imageDims.w - 1)),
      y: clamp(y, 0, Math.max(0, imageDims.h - 1)),
    }
  }

  function locoTestPointFromClient(clientX, clientY) {
    const stage = locoTestStageRef.current
    if (!stage || !imageDims.w || !imageDims.h) return null
    const metrics = getLocoTestRenderMetrics()
    if (!metrics) return null
    const rect = stage.getBoundingClientRect()
    if (rect.width < 1 || rect.height < 1) return null
    const x = (clientX - rect.left - metrics.x) / metrics.scale
    const y = (clientY - rect.top - metrics.y) / metrics.scale
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null
    if (x < 0 || y < 0 || x >= imageDims.w || y >= imageDims.h) return null
    return {
      x: clamp(x, 0, Math.max(0, imageDims.w - 1)),
      y: clamp(y, 0, Math.max(0, imageDims.h - 1)),
    }
  }

  function zoomDiameterBy(factor) {
    if (!imageUrl) return
    setDiamViewerZoom((prev) => clamp(prev * factor, 0.25, 12))
  }

  function zoomDiameterAt(clientX, clientY, factor) {
    if (!imageUrl) return
    const stage = diameterStageRef.current
    const metrics = getDiameterRenderMetrics()
    if (!stage || !metrics) {
      zoomDiameterBy(factor)
      return
    }
    const rect = stage.getBoundingClientRect()
    const imgX = (clientX - rect.left - metrics.x) / metrics.scale
    const imgY = (clientY - rect.top - metrics.y) / metrics.scale
    const nextZoom = clamp(diamViewerZoom * factor, 0.25, 12)
    const fit = Math.min(diamStageSize.w / Math.max(1, imageDims.w), diamStageSize.h / Math.max(1, imageDims.h))
    const nextScale = Math.max(0.0001, fit * nextZoom)
    setDiamViewerZoom(nextZoom)
    setDiamViewerOffset({
      x: clientX - rect.left - imgX * nextScale - (diamStageSize.w - imageDims.w * nextScale) * 0.5,
      y: clientY - rect.top - imgY * nextScale - (diamStageSize.h - imageDims.h * nextScale) * 0.5,
    })
  }

  function zoomLocoBy(factor) {
    if (!imageUrl) return
    setLocoViewerZoom((prev) => clamp(prev * factor, 0.25, 12))
  }

  function zoomLocoAt(clientX, clientY, factor) {
    if (!imageUrl) return
    const stage = locoStageRef.current
    const metrics = getLocoRenderMetrics()
    if (!stage || !metrics) {
      zoomLocoBy(factor)
      return
    }
    const rect = stage.getBoundingClientRect()
    const imgX = (clientX - rect.left - metrics.x) / metrics.scale
    const imgY = (clientY - rect.top - metrics.y) / metrics.scale
    const nextZoom = clamp(locoViewerZoom * factor, 0.25, 12)
    const fit = Math.min(locoStageSize.w / Math.max(1, imageDims.w), locoStageSize.h / Math.max(1, imageDims.h))
    const nextScale = Math.max(0.0001, fit * nextZoom)
    setLocoViewerZoom(nextZoom)
    setLocoViewerOffset({
      x: clientX - rect.left - imgX * nextScale - (locoStageSize.w - imageDims.w * nextScale) * 0.5,
      y: clientY - rect.top - imgY * nextScale - (locoStageSize.h - imageDims.h * nextScale) * 0.5,
    })
  }

  function zoomLocoDatasetBy(factor) {
    if (!imageUrl) return
    setLocoDatasetZoom((prev) => clamp(prev * factor, 0.25, 12))
  }

  function zoomLocoDatasetAt(clientX, clientY, factor) {
    if (!imageUrl) return
    const stage = locoDatasetStageRef.current
    const metrics = getLocoDatasetRenderMetrics()
    if (!stage || !metrics) {
      zoomLocoDatasetBy(factor)
      return
    }
    const rect = stage.getBoundingClientRect()
    const imgX = (clientX - rect.left - metrics.x) / metrics.scale
    const imgY = (clientY - rect.top - metrics.y) / metrics.scale
    const nextZoom = clamp(locoDatasetZoom * factor, 0.25, 12)
    const fit = Math.min(locoDatasetStageSize.w / Math.max(1, imageDims.w), locoDatasetStageSize.h / Math.max(1, imageDims.h))
    const nextScale = Math.max(0.0001, fit * nextZoom)
    setLocoDatasetZoom(nextZoom)
    setLocoDatasetOffset({
      x: clientX - rect.left - imgX * nextScale - (locoDatasetStageSize.w - imageDims.w * nextScale) * 0.5,
      y: clientY - rect.top - imgY * nextScale - (locoDatasetStageSize.h - imageDims.h * nextScale) * 0.5,
    })
  }

  function zoomLocoTestBy(factor) {
    if (!imageUrl) return
    setLocoTestZoom((prev) => clamp(prev * factor, 0.25, 12))
  }

  function zoomLocoTestAt(clientX, clientY, factor) {
    if (!imageUrl) return
    const stage = locoTestStageRef.current
    const metrics = getLocoTestRenderMetrics()
    if (!stage || !metrics) {
      zoomLocoTestBy(factor)
      return
    }
    const rect = stage.getBoundingClientRect()
    const imgX = (clientX - rect.left - metrics.x) / metrics.scale
    const imgY = (clientY - rect.top - metrics.y) / metrics.scale
    const nextZoom = clamp(locoTestZoom * factor, 0.25, 12)
    const fit = Math.min(locoTestStageSize.w / Math.max(1, imageDims.w), locoTestStageSize.h / Math.max(1, imageDims.h))
    const nextScale = Math.max(0.0001, fit * nextZoom)
    setLocoTestZoom(nextZoom)
    setLocoTestOffset({
      x: clientX - rect.left - imgX * nextScale - (locoTestStageSize.w - imageDims.w * nextScale) * 0.5,
      y: clientY - rect.top - imgY * nextScale - (locoTestStageSize.h - imageDims.h * nextScale) * 0.5,
    })
  }

  function zoomLocoModelBy(factor) {
    if (!imageUrl) return
    setLocoModelZoom((prev) => clamp(prev * factor, 0.25, 12))
  }

  function zoomLocoModelAt(clientX, clientY, factor) {
    if (!imageUrl) return
    const stage = locoModelStageRef.current
    const metrics = getLocoModelRenderMetrics()
    if (!stage || !metrics) {
      zoomLocoModelBy(factor)
      return
    }
    const rect = stage.getBoundingClientRect()
    const imgX = (clientX - rect.left - metrics.x) / metrics.scale
    const imgY = (clientY - rect.top - metrics.y) / metrics.scale
    const nextZoom = clamp(locoModelZoom * factor, 0.25, 12)
    const fit = Math.min(locoModelStageSize.w / Math.max(1, imageDims.w), locoModelStageSize.h / Math.max(1, imageDims.h))
    const nextScale = Math.max(0.0001, fit * nextZoom)
    setLocoModelZoom(nextZoom)
    setLocoModelOffset({
      x: clientX - rect.left - imgX * nextScale - (locoModelStageSize.w - imageDims.w * nextScale) * 0.5,
      y: clientY - rect.top - imgY * nextScale - (locoModelStageSize.h - imageDims.h * nextScale) * 0.5,
    })
  }

  function resetDiameterView() {
    if (diamMethodId === 'circle_square_mask_diameter') setDiamViewerMode('circle')
    else if (['manual_dual_side_caliper', 'manual_line_direct_caliper'].includes(diamMethodId)) setDiamViewerMode('manual')
    else setDiamViewerMode('mark')
    setDiamViewerZoom(1)
    setDiamViewerOffset({ x: 0, y: 0 })
  }

  function resetLocoView() {
    setLocoViewerMode('center')
    setLocoViewerZoom(1)
    setLocoViewerOffset({ x: 0, y: 0 })
    setLocoIsPanning(false)
  }

  function resetLocoDatasetView() {
    setLocoDatasetZoom(1)
    setLocoDatasetOffset({ x: 0, y: 0 })
    setLocoDatasetIsPanning(false)
  }

  function resetLocoTestView() {
    setLocoTestZoom(1)
    setLocoTestOffset({ x: 0, y: 0 })
    setLocoTestIsPanning(false)
  }

  function resetLocoModelView() {
    setLocoModelZoom(1)
    setLocoModelOffset({ x: 0, y: 0 })
    setLocoModelIsPanning(false)
  }

  function diameterSnapshot() {
    return {
      line: manualLineDraft,
      maskLine: manualMaskLineDraft,
      directLine: manualDirectLineDraft,
      maskLines: manualMaskLines,
      directLines: manualDirectLines,
      maskLineActiveIdx: manualMaskLineActiveIdx,
      directLineActiveIdx: manualDirectLineActiveIdx,
      circle: manualCircleDraft,
      circles: manualCircles,
      circleActiveIdx: manualCircleActiveIdx,
      circleSelected: manualCircleSelected,
      points: Array.isArray(diamPoints) ? diamPoints.map((p) => ({ x: Number(p.x), y: Number(p.y) })) : [],
      active: Number.isFinite(Number(diamActivePointIdx)) ? Number(diamActivePointIdx) : -1,
    }
  }

  function rememberDiameterManualState() {
    setDiamManualHistory((prev) => [...prev.slice(-24), diameterSnapshot()])
    setDiamManualFuture([])
  }

  function setDiameterManualState(nextLine, nextCircle, { remember = true } = {}) {
    if (remember) rememberDiameterManualState()
    setLineDraftForKind(currentManualLineKind(), nextLine)
    setManualCircleDraft(nextCircle)
    setManualCircleConsumed(false)
  }

  function undoDiameterManual() {
    setDiamManualHistory((prev) => {
      if (!prev.length) return prev
      const snapshot = prev[prev.length - 1]
      setDiamManualFuture((future) => [diameterSnapshot(), ...future.slice(0, 24)])
      setManualMaskLineDraft(snapshot.maskLine || snapshot.line || { start: null, end: null })
      setManualDirectLineDraft(snapshot.directLine || { start: null, end: null })
      setManualMaskLines(Array.isArray(snapshot.maskLines) ? snapshot.maskLines : [])
      setManualDirectLines(Array.isArray(snapshot.directLines) ? snapshot.directLines : [])
      setManualMaskLineActiveIdx(Number.isFinite(Number(snapshot.maskLineActiveIdx)) ? Number(snapshot.maskLineActiveIdx) : -1)
      setManualDirectLineActiveIdx(Number.isFinite(Number(snapshot.directLineActiveIdx)) ? Number(snapshot.directLineActiveIdx) : -1)
      setManualCircleDraft(snapshot.circle || null)
      setManualCircles(Array.isArray(snapshot.circles) ? snapshot.circles : [])
      setManualCircleActiveIdx(Number.isFinite(Number(snapshot.circleActiveIdx)) ? Number(snapshot.circleActiveIdx) : -1)
      setManualCircleSelected(Boolean(snapshot.circleSelected) && !!snapshot.circle)
      setDiamPoints(Array.isArray(snapshot.points) ? snapshot.points : [])
      setDiamActivePointIdx(Number.isFinite(Number(snapshot.active)) ? Number(snapshot.active) : -1)
      void syncDiameterPointsToServer(snapshot)
      return prev.slice(0, -1)
    })
  }

  function redoDiameterManual() {
    setDiamManualFuture((prev) => {
      if (!prev.length) return prev
      const snapshot = prev[0]
      setDiamManualHistory((history) => [...history.slice(-24), diameterSnapshot()])
      setManualMaskLineDraft(snapshot.maskLine || snapshot.line || { start: null, end: null })
      setManualDirectLineDraft(snapshot.directLine || { start: null, end: null })
      setManualMaskLines(Array.isArray(snapshot.maskLines) ? snapshot.maskLines : [])
      setManualDirectLines(Array.isArray(snapshot.directLines) ? snapshot.directLines : [])
      setManualMaskLineActiveIdx(Number.isFinite(Number(snapshot.maskLineActiveIdx)) ? Number(snapshot.maskLineActiveIdx) : -1)
      setManualDirectLineActiveIdx(Number.isFinite(Number(snapshot.directLineActiveIdx)) ? Number(snapshot.directLineActiveIdx) : -1)
      setManualCircleDraft(snapshot.circle || null)
      setManualCircles(Array.isArray(snapshot.circles) ? snapshot.circles : [])
      setManualCircleActiveIdx(Number.isFinite(Number(snapshot.circleActiveIdx)) ? Number(snapshot.circleActiveIdx) : -1)
      setManualCircleSelected(Boolean(snapshot.circleSelected) && !!snapshot.circle)
      setDiamPoints(Array.isArray(snapshot.points) ? snapshot.points : [])
      setDiamActivePointIdx(Number.isFinite(Number(snapshot.active)) ? Number(snapshot.active) : -1)
      void syncDiameterPointsToServer(snapshot)
      return prev.slice(1)
    })
  }

  function clearDiameterManualCircle({ remember = true } = {}) {
    if (remember) rememberDiameterManualState()
    setManualCircleDraft(null)
    setManualCircles((prev) => (
      manualCircleActiveIdx >= 0 && manualCircleActiveIdx < prev.length
        ? prev.filter((_, idx) => idx !== manualCircleActiveIdx)
        : []
    ))
    setManualCircleActiveIdx(-1)
    setManualCircleSelected(false)
    setManualCircleConsumed(true)
    setDiamParams((prev) => ({ ...prev, circle_square_seed_mode: 'manual_circle' }))
  }

  function consumeDiameterManualCircle() {
    diamCircleDragRef.current = { dragging: false, center: null }
    setManualCircleDraft(null)
    setManualCircles((prev) => prev.map((circle) => ({ ...circle, consumed: true })))
    setManualCircleActiveIdx(-1)
    setManualCircleSelected(false)
    setManualCircleConsumed(true)
    setDiamViewerMode('circle')
    setDiamParams((prev) => ({ ...prev, circle_square_seed_mode: 'manual_circle' }))
  }

  function rememberLocoState() {
    setLocoHistory((prev) => [...prev.slice(-24), { points: locoPoints, active: locoActivePointIdx, circle: locoCircleDraft }])
    setLocoFuture([])
  }

  function setLocoGeometry(nextPoints, nextActive, nextCircle, { remember = true } = {}) {
    if (remember) rememberLocoState()
    setLocoPoints(nextPoints)
    setLocoActivePointIdx(nextActive)
    setLocoCircleDraft(nextCircle)
    setLocoPreview(null)
    setLocoCandidateIndex(-1)
  }

  function undoLoco() {
    setLocoHistory((prev) => {
      if (!prev.length) return prev
      const snapshot = prev[prev.length - 1]
      setLocoFuture((future) => [{ points: locoPoints, active: locoActivePointIdx, circle: locoCircleDraft }, ...future.slice(0, 24)])
      setLocoPoints(Array.isArray(snapshot.points) ? snapshot.points : [])
      setLocoActivePointIdx(Number.isFinite(Number(snapshot.active)) ? Number(snapshot.active) : -1)
      setLocoCircleDraft(snapshot.circle || null)
      setLocoPreview(null)
      setLocoCandidateIndex(-1)
      return prev.slice(0, -1)
    })
  }

  function redoLoco() {
    setLocoFuture((prev) => {
      if (!prev.length) return prev
      const snapshot = prev[0]
      setLocoHistory((history) => [...history.slice(-24), { points: locoPoints, active: locoActivePointIdx, circle: locoCircleDraft }])
      setLocoPoints(Array.isArray(snapshot.points) ? snapshot.points : [])
      setLocoActivePointIdx(Number.isFinite(Number(snapshot.active)) ? Number(snapshot.active) : -1)
      setLocoCircleDraft(snapshot.circle || null)
      setLocoPreview(null)
      setLocoCandidateIndex(-1)
      return prev.slice(1)
    })
  }

  function getPointReviewRenderMetrics() {
    const { w, h } = imageDims
    if (w < 1 || h < 1 || pointReviewStageSize.w < 1 || pointReviewStageSize.h < 1) return null
    const fit = Math.min(pointReviewStageSize.w / w, pointReviewStageSize.h / h)
    const scale = Math.max(0.0001, fit * pointReviewZoom)
    const contentW = w * scale
    const contentH = h * scale
    const x = (pointReviewStageSize.w - contentW) * 0.5 + pointReviewOffset.x
    const y = (pointReviewStageSize.h - contentH) * 0.5 + pointReviewOffset.y
    return { scale, x, y }
  }

  function zoomPointReviewBy(factor) {
    if (!imageUrl) return
    setPointReviewZoom((prev) => clamp(prev * factor, 0.5, 12))
  }

  function centerPointReview() {
    if (!diamReviewTarget?.point || pointReviewStageSize.w < 1 || pointReviewStageSize.h < 1 || imageDims.w < 1 || imageDims.h < 1) return
    const fit = Math.min(pointReviewStageSize.w / imageDims.w, pointReviewStageSize.h / imageDims.h)
    const zoom = 2
    const scale = fit * zoom
    const p = diamReviewTarget.point
    setPointReviewZoom(zoom)
    setPointReviewOffset({
      x: imageDims.w * scale * 0.5 - Number(p.x || 0) * scale,
      y: imageDims.h * scale * 0.5 - Number(p.y || 0) * scale,
    })
  }

  function commitDiameterManualLine(start, end, kind = currentManualLineKind(), lineIndex = -1) {
    if (!start || !end) return
    const dx = Number(end.x) - Number(start.x)
    const dy = Number(end.y) - Number(start.y)
    const dist = Math.sqrt(dx * dx + dy * dy)
    const existingLine = Number.isInteger(lineIndex) && lineIndex >= 0 ? linesForKind(kind)[lineIndex] : null
    const draftLine = lineDraftForKind(kind)
    const line = {
      start,
      end,
      method_id: kind === 'direct' ? 'manual_line_direct_caliper' : 'manual_dual_side_caliper',
      geometry_id: String(existingLine?.geometry_id || draftLine?.geometry_id || makeManualGeometryId(kind, start, end)),
    }
    setLineDraftForKind(kind, line)
    if (Number.isInteger(lineIndex) && lineIndex >= 0) {
      setLinesForKind(kind, (prev) => prev.map((item, idx) => (idx === lineIndex ? line : item)))
      setLineActiveIndexForKind(kind, lineIndex)
    } else {
      setLinesForKind(kind, (prev) => {
        if (prev.some((item) => lineAlmostEqual(item, line))) return prev
        const next = [...prev, line]
        setLineActiveIndexForKind(kind, next.length - 1)
        return next
      })
    }
    setValidationForm((prev) => ({
      ...prev,
      manual_diameter_px: Number.isFinite(dist) ? dist.toFixed(2) : prev.manual_diameter_px,
      manual_left_x: Number(start.x).toFixed(2),
      manual_left_y: Number(start.y).toFixed(2),
      manual_right_x: Number(end.x).toFixed(2),
      manual_right_y: Number(end.y).toFixed(2),
      measurement_decision: prev.measurement_decision === 'unreviewed' ? 'rejected' : prev.measurement_decision,
    }))
    if (sessionId && imageId && Number.isFinite(dist) && dist >= 1) {
      const center = { x: (Number(start.x) + Number(end.x)) * 0.5, y: (Number(start.y) + Number(end.y)) * 0.5 }
      const existingIdx = diamPoints.findIndex((p) => Math.hypot(Number(p.x) - Number(center.x), Number(p.y) - Number(center.y)) <= 2)
      if (existingIdx >= 0) {
        void updateDiameterPoints('set_active', { active_index: existingIdx })
      } else {
        void updateDiameterPoints('add', { x: center.x, y: center.y }, { remember: false })
      }
    }
  }

  function startDiameterManualLineEndpointDrag(which, kind, lineIndex, e) {
    const draft = Number.isInteger(lineIndex) && lineIndex >= 0 ? linesForKind(kind)[lineIndex] : lineDraftForKind(kind)
    if (!draft?.start || !draft?.end) return
    e.stopPropagation()
    e.preventDefault()
    rememberDiameterManualState()
    diamLineDragRef.current = { dragging: true, mode: which, kind, lineIndex, start: draft.start, geometry_id: draft.geometry_id || lineGeometryId(kind, draft, lineIndex) }
    setLineDraftForKind(kind, draft)
    setLineActiveIndexForKind(kind, lineIndex)
    setDiamViewerMode('manual')
    try {
      diameterStageRef.current?.setPointerCapture?.(e.pointerId)
    } catch {
      e.currentTarget.setPointerCapture?.(e.pointerId)
    }
  }

  function onDiameterPointerDown(e) {
    if (!imageUrl) return
    if (diamViewerMode === 'manual') {
      const p = diameterPointFromClient(e.clientX, e.clientY)
      if (!p) return
      rememberDiameterManualState()
      const kind = currentManualLineKind()
      const geometryId = makeManualGeometryId(kind, p, p)
      diamLineDragRef.current = { dragging: true, mode: 'new', kind, lineIndex: -1, start: p, geometry_id: geometryId }
      setLineActiveIndexForKind(kind, -1)
      setLineDraftForKind(kind, { start: p, end: p, geometry_id: geometryId })
      e.currentTarget.setPointerCapture?.(e.pointerId)
      e.preventDefault()
      return
    }
    if (diamViewerMode === 'circle') {
      const p = diameterPointFromClient(e.clientX, e.clientY)
      if (!p) return
      rememberDiameterManualState()
      const geometryId = makeManualGeometryId('circle', p, p)
      diamCircleDragRef.current = { dragging: true, center: p, geometry_id: geometryId }
      setManualCircleDraft({ center: p, radius: 1, geometry_id: geometryId })
      setManualCircleSelected(false)
      setManualCircleConsumed(false)
      e.currentTarget.setPointerCapture?.(e.pointerId)
      e.preventDefault()
      return
    }
    if (diamViewerMode !== 'pan') return
    diamPanRef.current = { dragging: true, x: e.clientX, y: e.clientY }
    setDiamIsPanning(true)
    e.currentTarget.setPointerCapture?.(e.pointerId)
    e.preventDefault()
  }

  function onDiameterPointerMove(e) {
    if (diamLineDragRef.current.dragging) {
      const p = diameterPointFromClient(e.clientX, e.clientY)
      const drag = diamLineDragRef.current
      if (p) {
        if (drag.mode === 'start') {
          const draft = lineDraftForKind(drag.kind || currentManualLineKind())
          const nextLine = { ...draft, start: p, end: (draft?.end || drag.start || p), geometry_id: draft?.geometry_id || drag.geometry_id }
          setLineDraftForKind(drag.kind || currentManualLineKind(), nextLine)
          if (Number.isInteger(drag.lineIndex) && drag.lineIndex >= 0) setLinesForKind(drag.kind || currentManualLineKind(), (prev) => prev.map((item, idx) => (idx === drag.lineIndex ? nextLine : item)))
        } else if (drag.mode === 'end') {
          const draft = lineDraftForKind(drag.kind || currentManualLineKind())
          const nextLine = { ...draft, start: (draft?.start || drag.start || p), end: p, geometry_id: draft?.geometry_id || drag.geometry_id }
          setLineDraftForKind(drag.kind || currentManualLineKind(), nextLine)
          if (Number.isInteger(drag.lineIndex) && drag.lineIndex >= 0) setLinesForKind(drag.kind || currentManualLineKind(), (prev) => prev.map((item, idx) => (idx === drag.lineIndex ? nextLine : item)))
        } else {
          const start = drag.start
          if (start) setLineDraftForKind(drag.kind || currentManualLineKind(), { start, end: p, geometry_id: drag.geometry_id })
        }
      }
      e.preventDefault()
      return
    }
    if (diamCircleDragRef.current.dragging) {
      const p = diameterPointFromClient(e.clientX, e.clientY)
      const center = diamCircleDragRef.current.center
      if (p && center) {
        const dx = Number(p.x) - Number(center.x)
        const dy = Number(p.y) - Number(center.y)
        const radius = clamp(Math.sqrt(dx * dx + dy * dy), 1, Math.max(2, Number(diamParams.circle_square_max_radius_px) || 26))
        setManualCircleDraft({ center, radius, geometry_id: diamCircleDragRef.current.geometry_id })
        setManualCircleSelected(false)
        setManualCircleConsumed(false)
        setDiamParams((prev) => ({ ...prev, circle_square_seed_mode: 'manual_circle', circle_square_seed_radius_px: Number(radius.toFixed(1)) }))
      }
      e.preventDefault()
      return
    }
    if (!diamPanRef.current.dragging) return
    const dx = e.clientX - diamPanRef.current.x
    const dy = e.clientY - diamPanRef.current.y
    diamPanRef.current = { dragging: true, x: e.clientX, y: e.clientY }
    setDiamViewerOffset((prev) => ({ x: prev.x + dx, y: prev.y + dy }))
    e.preventDefault()
  }

  function onDiameterPointerUp(e) {
    if (diamLineDragRef.current.dragging) {
      const drag = diamLineDragRef.current
      const p = diameterPointFromClient(e.clientX, e.clientY)
      const draft = lineDraftForKind(drag.kind || currentManualLineKind())
      let start = draft.start || drag.start
      let end = draft.end
      if (p && drag.mode === 'start') start = p
      else if (p && drag.mode === 'end') end = p
      else if (p && drag.mode === 'new') {
        start = drag.start
        end = p
      }
      const geometryId = drag.geometry_id
      diamLineDragRef.current = { dragging: false, mode: '', kind: '', lineIndex: -1, start: null, geometry_id: '' }
      try {
        e.currentTarget.releasePointerCapture?.(e.pointerId)
      } catch {
        // Some endpoint drags start from the endpoint hit target and bubble here.
      }
      if (start && end) {
        const kind = drag.kind || currentManualLineKind()
        if (geometryId) setLineDraftForKind(kind, (prev) => ({ ...(prev || {}), start, end, geometry_id: geometryId }))
        commitDiameterManualLine(start, end, kind, Number.isInteger(drag.lineIndex) ? drag.lineIndex : -1)
      }
      e.preventDefault()
      return
    }
    if (diamCircleDragRef.current.dragging) {
      const center = diamCircleDragRef.current.center
      const geometryId = diamCircleDragRef.current.geometry_id
      const draft = manualCircleDraft
      diamCircleDragRef.current = { dragging: false, center: null, geometry_id: '' }
      e.currentTarget.releasePointerCapture?.(e.pointerId)
      if (center && draft?.center && Number.isFinite(Number(draft.radius)) && Number(draft.radius) >= 1) {
        const circle = {
          center: draft.center,
          radius: Number(draft.radius),
          geometry_id: String(draft.geometry_id || geometryId || makeManualGeometryId('circle', draft.center, draft.center)),
          consumed: false,
        }
        setManualCircleDraft(circle)
        setManualCircleConsumed(false)
        setManualCircles((prev) => {
          const existing = prev.findIndex((item) => String(item.geometry_id || '') === circle.geometry_id)
          const next = existing >= 0 ? prev.map((item, idx) => (idx === existing ? circle : item)) : [...prev, circle]
          setManualCircleActiveIdx(existing >= 0 ? existing : next.length - 1)
          return next
        })
      }
      if (center && sessionId && imageId) {
        const existingIdx = diamPoints.findIndex((p) => Math.hypot(Number(p.x) - Number(center.x), Number(p.y) - Number(center.y)) <= 2)
        if (existingIdx >= 0) {
          void updateDiameterPoints('set_active', { active_index: existingIdx })
        } else {
          void updateDiameterPoints('add', { x: center.x, y: center.y }, { remember: false })
        }
        setManualCircleSelected(true)
      }
      e.preventDefault()
      return
    }
    if (!diamPanRef.current.dragging) return
    diamPanRef.current = { dragging: false, x: 0, y: 0 }
    setDiamIsPanning(false)
    e.currentTarget.releasePointerCapture?.(e.pointerId)
  }

  function onPointReviewPointerDown(e) {
    if (!imageUrl) return
    pointReviewPanRef.current = { dragging: true, x: e.clientX, y: e.clientY }
    setPointReviewIsPanning(true)
    e.currentTarget.setPointerCapture?.(e.pointerId)
    e.preventDefault()
  }

  function onPointReviewPointerMove(e) {
    if (!pointReviewPanRef.current.dragging) return
    const dx = e.clientX - pointReviewPanRef.current.x
    const dy = e.clientY - pointReviewPanRef.current.y
    pointReviewPanRef.current = { dragging: true, x: e.clientX, y: e.clientY }
    setPointReviewOffset((prev) => ({ x: prev.x + dx, y: prev.y + dy }))
    e.preventDefault()
  }

  function onPointReviewPointerUp(e) {
    if (!pointReviewPanRef.current.dragging) return
    pointReviewPanRef.current = { dragging: false, x: 0, y: 0 }
    setPointReviewIsPanning(false)
    e.currentTarget.releasePointerCapture?.(e.pointerId)
  }

  function syncDiameterPointPayload(payload) {
    const points = Array.isArray(payload?.points) ? payload.points : []
    setDiamPoints(points)
    setDiamActivePointIdx(Number.isFinite(Number(payload?.active_point_idx)) ? Number(payload.active_point_idx) : -1)
  }

  async function syncDiameterPointsToServer(snapshot) {
    if (!sessionId || !imageId) return
    try {
      await apiPost('/api/diameter-research/points/update', {
        session_id: sessionId,
        action: 'replace',
        points: Array.isArray(snapshot?.points) ? snapshot.points : [],
        active_index: Number.isFinite(Number(snapshot?.active)) ? Number(snapshot.active) : -1,
      })
    } catch (err) {
      toast('warning', 'Diameter undo', errMsg(err))
    }
  }

  function clearDiameterLocalState() {
    setDiamOverlayUrl('')
    setDiamResults([])
    setDiamResultsMode('run')
    setDiamReviewTarget(null)
    setPointReviewOpen(false)
    setDiamActiveRunId('')
    setDiamReportInfo(null)
    setManualMaskLineDraft({ start: null, end: null })
    setManualDirectLineDraft({ start: null, end: null })
    setManualMaskLines([])
    setManualDirectLines([])
    setManualMaskLineActiveIdx(-1)
    setManualDirectLineActiveIdx(-1)
    setManualCircleDraft(null)
    setManualCircles([])
    setManualCircleActiveIdx(-1)
    setManualCircleSelected(false)
    setManualCircleConsumed(true)
    setDiamManualHistory([])
    setDiamManualFuture([])
    setValidationActiveCaseId('')
    setValidationForm((prev) => ({
      ...prev,
      case_id: '',
      manual_diameter_px: '',
      manual_left_x: '',
      manual_left_y: '',
      manual_right_x: '',
      manual_right_y: '',
      measurement_decision: 'unreviewed',
      notes: '',
      result_comment: '',
    }))
  }

  async function clearDiameterPanel() {
    clearDiameterLocalState()
    if (!sessionId || !imageId) {
      setDiamPoints([])
      setDiamActivePointIdx(-1)
      return
    }
    await withLoad('diamPoints', async () => {
      try {
        const res = await apiPost('/api/diameter-research/points/update', {
          session_id: sessionId,
          action: 'clear',
        })
        syncDiameterPointPayload(res)
        toast('success', 'Diameter Research', 'Puntos y marcas visuales limpiados.')
      } catch (err) {
        setDiamPoints([])
        setDiamActivePointIdx(-1)
        toast('error', 'Diameter points', errMsg(err))
      }
    })
  }

  async function updateDiameterPoints(action, extra = {}, options = {}) {
    if (!sessionId || !imageId) return
    if (options.remember !== false && ['add', 'remove_last', 'remove_active', 'clear', 'replace'].includes(String(action))) {
      rememberDiameterManualState()
    }
    await withLoad('diamPoints', async () => {
      try {
        const res = await apiPost('/api/diameter-research/points/update', {
          session_id: sessionId,
          action,
          ...extra,
        })
        syncDiameterPointPayload(res)
      } catch (err) {
        toast('error', 'Diameter points', errMsg(err))
      }
    })
  }

  async function loadDiameterPoints(customSessionId = '', customImageId = '', { silent = false } = {}) {
    const sid = String(customSessionId || sessionId || '').trim()
    const iid = String(customImageId || imageId || '').trim()
    if (!sid || !iid) return
    await withLoad('diamPoints', async () => {
      try {
        const res = await apiGet(`/api/diameter-research/points/load?session_id=${encodeURIComponent(sid)}&image_id=${encodeURIComponent(iid)}`)
        syncDiameterPointPayload(res)
      } catch (err) {
        if (!silent) toast('warning', 'Diameter points', errMsg(err))
      }
    })
  }

  async function saveDiameterPoints() {
    if (!sessionId || !imageId) return
    await withLoad('diamPoints', async () => {
      try {
        const res = await apiPost('/api/diameter-research/points/save', {
          session_id: sessionId,
          image_id: imageId,
        })
        syncDiameterPointPayload(res)
        toast('success', 'Diameter points', 'Puntos guardados.')
      } catch (err) {
        toast('error', 'Diameter points', errMsg(err))
      }
    })
  }

  function onDiameterStageClick(e) {
    if (!imageUrl || loading.diamPoints) return
    if (diamViewerMode === 'pan') return
    if (diamViewerMode === 'circle') return
    if (diamViewerMode === 'manual') return
    const p = diameterPointFromClient(e.clientX, e.clientY)
    if (!p) return
    void updateDiameterPoints('add', { x: p.x, y: p.y })
  }

  function updateDiamParam(key, value, integer = false) {
    const raw = integer ? Math.round(Number(value)) : Number(value)
    if (!Number.isFinite(raw)) return
    setDiamParams((prev) => ({ ...prev, [key]: raw }))
    if (key === 'circle_square_seed_radius_px') {
      setManualCircleDraft((prev) => (prev ? { ...prev, radius: Math.max(1, raw) } : prev))
    }
    if (key === 'circle_square_max_radius_px') {
      setManualCircleDraft((prev) => {
        if (!prev) return prev
        return { ...prev, radius: Math.min(Math.max(1, raw), Math.max(1, Number(prev.radius) || 1)) }
      })
    }
  }

  function updateDiamRawParam(key, value) {
    setDiamParams((prev) => ({ ...prev, [key]: value }))
  }

  function diameterParamsPayload() {
    const payload = {
      support_high_threshold: clamp(Number(diamParams.support_high_threshold), 0, 1),
      support_low_threshold: clamp(Number(diamParams.support_low_threshold), 0, 1),
      support_dilation_px: Math.max(0, Math.round(Number(diamParams.support_dilation_px))),
      local_window_px: Math.max(9, Math.round(Number(diamParams.local_window_px))),
      profile_length_px: Math.max(8, Number(diamParams.profile_length_px)),
      profile_count: Math.max(1, Math.round(Number(diamParams.profile_count))),
      profile_spacing_px: Math.max(0, Number(diamParams.profile_spacing_px)),
      edge_min_score: Math.max(0, Number(diamParams.edge_min_score)),
      min_valid_profiles: Math.max(1, Math.round(Number(diamParams.min_valid_profiles))),
      max_mad_scale: Math.max(0.5, Number(diamParams.max_mad_scale)),
      mask_local_radius_px: Math.max(8, Number(diamParams.mask_local_radius_px)),
      mask_recenter_radius_px: Math.max(1, Number(diamParams.mask_recenter_radius_px)),
      mask_ray_count: Math.max(8, Math.round(Number(diamParams.mask_ray_count))),
      auto_small_context_width_px: Math.max(2, Number(diamParams.auto_small_context_width_px)),
      circle_square_seed_mode: String(diamParams.circle_square_seed_mode || 'manual_circle'),
      circle_square_seed_radius_px: Math.max(1, Number(diamParams.circle_square_seed_radius_px)),
      circle_square_max_radius_px: Math.max(4, Number(diamParams.circle_square_max_radius_px)),
      circle_square_length_factor: Math.max(0.25, Number(diamParams.circle_square_length_factor)),
      circle_square_width_factor: Math.max(0.25, Number(diamParams.circle_square_width_factor)),
      circle_square_samples: Math.max(3, Math.round(Number(diamParams.circle_square_samples))),
      circle_square_aggregation: String(diamParams.circle_square_aggregation || 'median'),
      circle_square_recenter_seed: Boolean(diamParams.circle_square_recenter_seed),
      circle_square_max_recenter_shift_px: Math.max(0, Number(diamParams.circle_square_max_recenter_shift_px)),
      ellipse_roi_radius_px: Math.max(5, Number(diamParams.ellipse_roi_radius_px)),
      manual_caliper_refine: Boolean(diamParams.manual_caliper_refine),
    }
    if (manualLineDraft.start && manualLineDraft.end) {
      payload.manual_left_x = Number(manualLineDraft.start.x)
      payload.manual_left_y = Number(manualLineDraft.start.y)
      payload.manual_right_x = Number(manualLineDraft.end.x)
      payload.manual_right_y = Number(manualLineDraft.end.y)
    }
    if (!manualCircleConsumed && manualCircleDraft?.center && Number.isFinite(Number(manualCircleDraft.radius))) {
      payload.circle_square_seed_mode = 'manual_circle'
      payload.circle_square_seed_radius_px = Math.max(1, Number(manualCircleDraft.radius))
    }
    return payload
  }

  function updateLocoParam(key, value, integer = false) {
    const raw = integer ? Math.round(Number(value)) : Number(value)
    if (!Number.isFinite(raw)) return
    setLocoParams((prev) => ({ ...prev, [key]: raw }))
  }

  function updateLocoRawParam(key, value) {
    setLocoParams((prev) => ({ ...prev, [key]: value }))
  }

  function updateLocoLabParam(key, value, integer = false) {
    const raw = integer ? Math.round(Number(value)) : Number(value)
    if (!Number.isFinite(raw)) return
    setLocoLabParams((prev) => ({ ...prev, [key]: raw }))
  }

  function updateLocoLabRawParam(key, value) {
    setLocoLabParams((prev) => ({ ...prev, [key]: value }))
  }

  function toggleLocoLayer(key) {
    setLocoLayerVisibility((prev) => ({ ...prev, [key]: !prev[key] }))
  }

  function locoLabPayloadParams() {
    return {
      grid_stride_px: Math.max(3, Math.round(Number(locoLabParams.grid_stride_px) || 18)),
      radius_min_px: Math.max(1, Number(locoLabParams.radius_min_px) || 3),
      radius_max_px: Math.max(1, Number(locoLabParams.radius_max_px) || 18),
      radius_step_px: Math.max(0.5, Number(locoLabParams.radius_step_px) || 2),
      mask_required_ratio: clamp(Number(locoLabParams.mask_required_ratio), 0, 1),
      max_candidates: Math.max(1, Math.round(Number(locoLabParams.max_candidates) || 600)),
      min_score: clamp(Number(locoLabParams.min_score), 0, 1),
      max_intersections: Math.max(1, Math.round(Number(locoLabParams.max_intersections) || 12)),
      circle_samples: Math.max(32, Math.round(Number(locoLabParams.circle_samples) || 128)),
      require_four_cuts: Boolean(locoLabParams.require_four_cuts),
      measure_limit: Math.max(1, Math.round(Number(locoLabParams.measure_limit) || 120)),
    }
  }

  function activeLocoPoint() {
    if (locoActivePointIdx >= 0 && locoActivePointIdx < locoPoints.length) return locoPoints[locoActivePointIdx]
    return locoPoints[0] || null
  }

  function locoParamsPayload(points = locoPoints) {
    const seedMap = {}
    ;(Array.isArray(points) ? points : []).forEach((point, idx) => {
      if (Number.isFinite(Number(point?.seed_radius_px)) && Number(point.seed_radius_px) > 0) {
        seedMap[String(idx)] = Number(point.seed_radius_px)
      }
    })
    return {
      ...diameterParamsPayload(),
      loco_roi_radius_px: Math.max(8, Number(locoParams.loco_roi_radius_px)),
      loco_max_radius_px: Math.max(2, Number(locoParams.loco_max_radius_px)),
      loco_radius_step_px: Math.max(0.5, Number(locoParams.loco_radius_step_px)),
      loco_seed_radius_window_px: Math.max(1, Number(locoParams.loco_seed_radius_window_px)),
      loco_circle_samples: Math.max(32, Math.round(Number(locoParams.loco_circle_samples))),
      loco_recenter_enabled: Boolean(locoParams.loco_recenter_enabled),
      loco_max_recenter_shift_px: Math.max(0, Number(locoParams.loco_max_recenter_shift_px)),
      loco_symmetry_threshold: clamp(Number(locoParams.loco_symmetry_threshold), 0, 1),
      loco_reject_threshold: clamp(Number(locoParams.loco_reject_threshold), 0, 1),
      loco_mode: String(locoParams.loco_mode || 'refine'),
      loco_aggregation: String(locoParams.loco_aggregation || 'median'),
      loco_seed_radii_by_point: seedMap,
    }
  }

  function onLocoStageClick(e) {
    if (!imageUrl || locoViewerMode === 'pan' || locoViewerMode === 'circle') return
    const p = locoPointFromClient(e.clientX, e.clientY)
    if (!p) return
    const next = [...locoPoints, { x: p.x, y: p.y }]
    setLocoGeometry(next, next.length - 1, null)
    setLocoStep(1)
  }

  function onLocoPointerDown(e) {
    if (!imageUrl) return
    if (locoViewerMode === 'circle') {
      const p = locoPointFromClient(e.clientX, e.clientY)
      if (!p) return
      rememberLocoState()
      locoCircleDragRef.current = { dragging: true, center: p }
      setLocoCircleDraft({ center: p, radius: 1 })
      e.currentTarget.setPointerCapture?.(e.pointerId)
      e.preventDefault()
      return
    }
    if (locoViewerMode !== 'pan') return
    locoPanRef.current = { dragging: true, x: e.clientX, y: e.clientY }
    setLocoIsPanning(true)
    e.currentTarget.setPointerCapture?.(e.pointerId)
    e.preventDefault()
  }

  function onLocoPointerMove(e) {
    if (locoCircleDragRef.current.dragging) {
      const p = locoPointFromClient(e.clientX, e.clientY)
      const center = locoCircleDragRef.current.center
      if (!p || !center) return
      const dx = Number(p.x) - Number(center.x)
      const dy = Number(p.y) - Number(center.y)
      const radius = clamp(Math.sqrt(dx * dx + dy * dy), 1, Math.max(2, Number(locoParams.loco_max_radius_px) || 26))
      setLocoCircleDraft({ center, radius: Number(radius.toFixed(1)) })
      e.preventDefault()
      return
    }
    if (!locoPanRef.current.dragging) return
    const dx = e.clientX - locoPanRef.current.x
    const dy = e.clientY - locoPanRef.current.y
    locoPanRef.current = { dragging: true, x: e.clientX, y: e.clientY }
    setLocoViewerOffset((prev) => ({ x: prev.x + dx, y: prev.y + dy }))
    e.preventDefault()
  }

  function onLocoPointerUp(e) {
    if (locoCircleDragRef.current.dragging) {
      const draft = locoCircleDraft
      locoCircleDragRef.current = { dragging: false, center: null }
      e.currentTarget.releasePointerCapture?.(e.pointerId)
      if (draft?.center) {
        const nextPoint = { x: Number(draft.center.x), y: Number(draft.center.y), seed_radius_px: Math.max(1, Number(draft.radius) || 1) }
        setLocoPoints((prev) => {
          const existingIdx = prev.findIndex((p) => Math.hypot(Number(p.x) - nextPoint.x, Number(p.y) - nextPoint.y) <= 2)
          if (existingIdx >= 0) {
            const next = prev.map((p, idx) => (idx === existingIdx ? { ...p, seed_radius_px: nextPoint.seed_radius_px } : p))
            setLocoActivePointIdx(existingIdx)
            return next
          }
          setLocoActivePointIdx(prev.length)
          return [...prev, nextPoint]
        })
        setLocoCircleDraft({ center: draft.center, radius: nextPoint.seed_radius_px })
        setLocoPreview(null)
        setLocoCandidateIndex(-1)
        setLocoStep(2)
      }
      e.preventDefault()
      return
    }
    if (!locoPanRef.current.dragging) return
    locoPanRef.current = { dragging: false, x: 0, y: 0 }
    setLocoIsPanning(false)
    e.currentTarget.releasePointerCapture?.(e.pointerId)
  }

  function onLocoDatasetPointerDown(e) {
    if (!imageUrl) return
    if (locoDatasetTool === 'pan') {
      locoDatasetPanRef.current = { dragging: true, x: e.clientX, y: e.clientY }
      setLocoDatasetIsPanning(true)
      e.currentTarget.setPointerCapture?.(e.pointerId)
      e.preventDefault()
      return
    }
    if (locoDatasetTool !== 'circle') return
    const p = locoDatasetPointFromClient(e.clientX, e.clientY)
    if (!p) return
    locoDatasetDragRef.current = { mode: 'create', id: '', start: p, circle: null }
    setLocoDatasetDraftCircle({ center_x: p.x, center_y: p.y, radius_px: 1 })
    e.currentTarget.setPointerCapture?.(e.pointerId)
    e.preventDefault()
  }

  function onLocoDatasetPointerMove(e) {
    if (locoDatasetPanRef.current.dragging) {
      const dx = e.clientX - locoDatasetPanRef.current.x
      const dy = e.clientY - locoDatasetPanRef.current.y
      locoDatasetPanRef.current = { dragging: true, x: e.clientX, y: e.clientY }
      setLocoDatasetOffset((prev) => ({ x: prev.x + dx, y: prev.y + dy }))
      e.preventDefault()
      return
    }
    const drag = locoDatasetDragRef.current
    if (drag.mode === 'create') {
      const p = locoDatasetPointFromClient(e.clientX, e.clientY)
      if (!p || !drag.start) return
      const dx = Number(p.x) - Number(drag.start.x)
      const dy = Number(p.y) - Number(drag.start.y)
      const radius = clamp(Math.sqrt(dx * dx + dy * dy), 1, 256)
      setLocoDatasetDraftCircle({ center_x: Number(drag.start.x), center_y: Number(drag.start.y), radius_px: Number(radius.toFixed(1)) })
      e.preventDefault()
      return
    }
    if (drag.mode === 'move') {
      const p = locoDatasetPointFromClient(e.clientX, e.clientY)
      if (!p || !drag.start || !drag.circle) return
      const dx = Number(p.x) - Number(drag.start.x)
      const dy = Number(p.y) - Number(drag.start.y)
      updateLocoDatasetCircle(drag.id, {
        center_x: clamp(Number(drag.circle.center_x) + dx, 0, Math.max(0, imageDims.w - 1)),
        center_y: clamp(Number(drag.circle.center_y) + dy, 0, Math.max(0, imageDims.h - 1)),
      })
      e.preventDefault()
    }
  }

  function onLocoDatasetPointerUp(e) {
    if (locoDatasetPanRef.current.dragging) {
      locoDatasetPanRef.current = { dragging: false, x: 0, y: 0 }
      setLocoDatasetIsPanning(false)
      e.currentTarget.releasePointerCapture?.(e.pointerId)
      return
    }
    const drag = locoDatasetDragRef.current
    if (drag.mode === 'create') {
      const draft = locoDatasetDraftCircle
      locoDatasetDragRef.current = { mode: '', id: '', start: null, circle: null }
      e.currentTarget.releasePointerCapture?.(e.pointerId)
      if (draft?.center_x != null && Number(draft.radius_px) >= 2) {
        const candidate_id = `cand_${Date.now().toString(36)}_${locoDatasetCircles.length + 1}`
        const next = {
          candidate_id,
          center_x: Number(draft.center_x),
          center_y: Number(draft.center_y),
          radius_px: Number(draft.radius_px),
          label: locoDatasetDefaultLabel,
        }
        setLocoDatasetCircles((prev) => [...prev, next])
        setLocoDatasetSelectedId(candidate_id)
        setLocoDatasetFeatures([])
        setLocoDatasetSaveInfo(null)
      }
      setLocoDatasetDraftCircle(null)
      e.preventDefault()
      return
    }
    if (drag.mode === 'move') {
      locoDatasetDragRef.current = { mode: '', id: '', start: null, circle: null }
      e.currentTarget.releasePointerCapture?.(e.pointerId)
      e.preventDefault()
    }
  }

  function beginMoveLocoDatasetCircle(e, circle) {
    e.stopPropagation()
    e.preventDefault()
    setLocoDatasetSelectedId(String(circle.candidate_id))
    if (locoDatasetTool !== 'select') {
      // In circle mode, just select the circle without changing tool
      return
    }
    const p = locoDatasetPointFromClient(e.clientX, e.clientY)
    if (!p) return
    locoDatasetDragRef.current = { mode: 'move', id: String(circle.candidate_id), start: p, circle: { ...circle } }
    e.currentTarget.setPointerCapture?.(e.pointerId)
  }

  function onLocoTestPointerDown(e) {
    if (!imageUrl) return
    if (locoTestTool === 'pan') {
      locoTestPanRef.current = { dragging: true, x: e.clientX, y: e.clientY }
      setLocoTestIsPanning(true)
      e.currentTarget.setPointerCapture?.(e.pointerId)
      e.preventDefault()
      return
    }
    if (locoTestTool !== 'circle') return
    const p = locoTestPointFromClient(e.clientX, e.clientY)
    if (!p) return
    locoTestDragRef.current = { mode: 'create', id: '', start: p, circle: null }
    setLocoTestDraftCircle({ center_x: p.x, center_y: p.y, radius_px: 1 })
    e.currentTarget.setPointerCapture?.(e.pointerId)
    e.preventDefault()
  }

  function onLocoTestPointerMove(e) {
    if (locoTestPanRef.current.dragging) {
      const dx = e.clientX - locoTestPanRef.current.x
      const dy = e.clientY - locoTestPanRef.current.y
      locoTestPanRef.current = { dragging: true, x: e.clientX, y: e.clientY }
      setLocoTestOffset((prev) => ({ x: prev.x + dx, y: prev.y + dy }))
      e.preventDefault()
      return
    }
    const drag = locoTestDragRef.current
    if (drag.mode === 'create') {
      const p = locoTestPointFromClient(e.clientX, e.clientY)
      if (!p || !drag.start) return
      const dx = Number(p.x) - Number(drag.start.x)
      const dy = Number(p.y) - Number(drag.start.y)
      const radius = clamp(Math.sqrt(dx * dx + dy * dy), 1, 256)
      setLocoTestDraftCircle({ center_x: Number(drag.start.x), center_y: Number(drag.start.y), radius_px: Number(radius.toFixed(1)) })
      e.preventDefault()
      return
    }
    if (drag.mode === 'move') {
      const p = locoTestPointFromClient(e.clientX, e.clientY)
      if (!p || !drag.start || !drag.circle) return
      const dx = Number(p.x) - Number(drag.start.x)
      const dy = Number(p.y) - Number(drag.start.y)
      updateLocoTestCircle(drag.id, {
        center_x: clamp(Number(drag.circle.center_x) + dx, 0, Math.max(0, imageDims.w - 1)),
        center_y: clamp(Number(drag.circle.center_y) + dy, 0, Math.max(0, imageDims.h - 1)),
      })
      e.preventDefault()
    }
  }

  function onLocoTestPointerUp(e) {
    if (locoTestPanRef.current.dragging) {
      locoTestPanRef.current = { dragging: false, x: 0, y: 0 }
      setLocoTestIsPanning(false)
      e.currentTarget.releasePointerCapture?.(e.pointerId)
      return
    }
    const drag = locoTestDragRef.current
    if (drag.mode === 'create') {
      const draft = locoTestDraftCircle
      locoTestDragRef.current = { mode: '', id: '', start: null, circle: null }
      e.currentTarget.releasePointerCapture?.(e.pointerId)
      if (draft?.center_x != null && Number(draft.radius_px) >= 2) {
        const candidate_id = `test_${Date.now().toString(36)}_${locoTestCircles.length + 1}`
        const next = {
          candidate_id,
          center_x: Number(draft.center_x),
          center_y: Number(draft.center_y),
          radius_px: Number(draft.radius_px),
          label: locoTestDefaultLabel || 'valid',
        }
        setLocoTestCircles((prev) => [...prev, next])
        setLocoTestSelectedId(candidate_id)
        setLocoTestResult(null)
      }
      setLocoTestDraftCircle(null)
      e.preventDefault()
      return
    }
    if (drag.mode === 'move') {
      locoTestDragRef.current = { mode: '', id: '', start: null, circle: null }
      e.currentTarget.releasePointerCapture?.(e.pointerId)
      e.preventDefault()
    }
  }

  function beginMoveLocoTestCircle(e, circle) {
    e.stopPropagation()
    e.preventDefault()
    setLocoTestSelectedId(String(circle.candidate_id))
    if (locoTestTool !== 'select') {
      // In circle mode, just select the circle without changing tool
      return
    }
    const p = locoTestPointFromClient(e.clientX, e.clientY)
    if (!p) return
    locoTestDragRef.current = { mode: 'move', id: String(circle.candidate_id), start: p, circle: { ...circle } }
    e.currentTarget.setPointerCapture?.(e.pointerId)
  }

  function onLocoModelPointerDown(e) {
    if (!imageUrl) return
    locoModelPanRef.current = { dragging: true, x: e.clientX, y: e.clientY }
    setLocoModelIsPanning(true)
    e.currentTarget.setPointerCapture?.(e.pointerId)
    e.preventDefault()
  }

  function onLocoModelPointerMove(e) {
    if (!locoModelPanRef.current.dragging) return
    const dx = e.clientX - locoModelPanRef.current.x
    const dy = e.clientY - locoModelPanRef.current.y
    locoModelPanRef.current = { dragging: true, x: e.clientX, y: e.clientY }
    setLocoModelOffset((prev) => ({ x: prev.x + dx, y: prev.y + dy }))
    e.preventDefault()
  }

  function onLocoModelPointerUp(e) {
    if (!locoModelPanRef.current.dragging) return
    locoModelPanRef.current = { dragging: false, x: 0, y: 0 }
    setLocoModelIsPanning(false)
    e.currentTarget.releasePointerCapture?.(e.pointerId)
  }

  async function refreshLocoPreview({ candidateIndex = -1, step = locoStep } = {}) {
    const point = activeLocoPoint()
    if (!sessionId || !imageId || !imageUrl || !point) {
      toast('warning', 'LOCO', 'Marca un centro o dibuja un circulo primero.')
      return
    }
    await flushDraftIfNeeded()
    const scribble = currentScribbleB64()
    const pointParams = locoParamsPayload([point])
    if (Number.isFinite(Number(point.seed_radius_px))) {
      pointParams.loco_seed_radius_px = Number(point.seed_radius_px)
    }
    await withLoad('locoPreview', async () => {
      try {
        const res = await apiPost('/api/diameter-research/loco/preview', {
          session_id: sessionId,
          image_id: imageId,
          source_mode: diamSourceMode,
          prior_run_id: diamPriorRunId,
          point: { x: Number(point.x), y: Number(point.y) },
          params: pointParams,
          scribble_map_b64: scribble,
          step,
          candidate_index: candidateIndex,
        })
        setLocoPreview(res || null)
        setLocoCandidateIndex(Number.isFinite(Number(res?.candidate_index)) ? Number(res.candidate_index) : -1)
        toast(res?.result?.status === 'ok' ? 'success' : 'warning', 'LOCO preview', res?.result?.status === 'ok' ? 'Geometria estable.' : (res?.result?.reason || 'Revision requerida.'))
      } catch (err) {
        toast('error', 'LOCO preview', errMsg(err))
      }
    })
  }

  async function runLoco(activeOnly) {
    const points = activeOnly ? [activeLocoPoint()].filter(Boolean) : locoPoints
    if (!sessionId || !imageId || !imageUrl || !points.length) {
      toast('warning', 'LOCO', 'Marca al menos un centro o circulo.')
      return
    }
    await flushDraftIfNeeded()
    const scribble = currentScribbleB64()
    await withLoad('locoRun', async () => {
      try {
        const res = await apiPost('/api/diameter-research/run', {
          session_id: sessionId,
          image_id: imageId,
          method_id: 'loco_circle_probe',
          source_mode: diamSourceMode,
          prior_run_id: diamPriorRunId,
          points,
          active_only: false,
          params: locoParamsPayload(points),
          scribble_map_b64: scribble,
        })
        setDiamRunCache((prev) => ({ ...prev, [res.run_id]: res }))
        setDiamActiveRunId(res.run_id)
        const nextResults = Array.isArray(res?.results) ? res.results : []
        setDiamResults(nextResults)
        setDiamResultsMode('run')
        if (res?.overlay_b64) setDiamOverlayUrl(b64ToDataUrl(res.overlay_b64, res.overlay_mime || 'image/png'))
        if (diamMethodId === 'circle_square_mask_diameter') clearDiameterManualCircle({ remember: false })
        await refreshDiameterRuns(imageId, { silent: true })
        const okCount = nextResults.filter((r) => r.status === 'ok').length
        toast(okCount ? 'success' : 'warning', 'LOCO', `${okCount}/${nextResults.length} puntos medidos.`)
      } catch (err) {
        toast('error', 'LOCO', errMsg(err))
      }
    })
  }

  function selectLocoCandidate(nextIdx) {
    const list = Array.isArray(locoPreview?.radius_candidates) ? locoPreview.radius_candidates : []
    if (!list.length) return
    setLocoCandidateIndex(clamp(nextIdx, 0, list.length - 1))
  }

  function clearLocoPanel() {
    setLocoGeometry([], -1, null)
    setLocoPreview(null)
    setLocoCandidateIndex(-1)
    setLocoStep(0)
    setLocoViewerMode('center')
  }

  function clearLocoLab() {
    setLocoProposals([])
    setLocoFilteredProposals([])
    setLocoMeasuredResults([])
    setLocoEvaluation(null)
    setLocoSelectedProposalId('')
    setLocoLabStage('proposals')
  }

  async function generateLocoLabProposals() {
    if (!sessionId || !imageId || !imageUrl) {
      toast('warning', 'LOCO Lab', 'Carga una imagen primero.')
      return
    }
    await flushDraftIfNeeded()
    const scribble = currentScribbleB64()
    const params = locoLabPayloadParams()
    await withLoad('locoLab', async () => {
      try {
        const res = await apiPost('/api/diameter-research/loco/proposals', {
          session_id: sessionId,
          image_id: imageId,
          source_mode: 'prior_mask',
          prior_run_id: diamPriorRunId,
          method: locoProposalMethod,
          params,
          scribble_map_b64: scribble,
        })
        const proposals = Array.isArray(res?.proposals) ? res.proposals : []
        setLocoProposals(proposals)
        setLocoFilteredProposals(proposals)
        setLocoMeasuredResults([])
        setLocoEvaluation(res?.summary || null)
        setLocoSelectedProposalId(proposals[0]?.proposal_id || '')
        setLocoLabStage('filters')
        toast(proposals.length ? 'success' : 'warning', 'LOCO Lab', `${proposals.length} propuestas generadas.`)
      } catch (err) {
        toast('error', 'LOCO Lab', errMsg(err))
      }
    })
  }

  async function filterLocoLabProposals() {
    const proposals = locoFilteredProposals.length ? locoFilteredProposals : locoProposals
    if (!sessionId || !imageId || !proposals.length) {
      toast('warning', 'LOCO Lab', 'Primero genera propuestas.')
      return
    }
    await withLoad('locoLab', async () => {
      try {
        const res = await apiPost('/api/diameter-research/loco/filter', {
          session_id: sessionId,
          image_id: imageId,
          proposals,
          params: locoLabPayloadParams(),
        })
        const next = Array.isArray(res?.proposals) ? res.proposals : []
        setLocoFilteredProposals(next)
        setLocoEvaluation(res?.summary || null)
        setLocoSelectedProposalId((prev) => prev || next[0]?.proposal_id || '')
        setLocoLabStage('circles')
        const accepted = next.filter((p) => p.status === 'accepted').length
        toast(accepted ? 'success' : 'warning', 'LOCO Lab', `${accepted}/${next.length} propuestas aceptadas.`)
      } catch (err) {
        toast('error', 'LOCO Lab', errMsg(err))
      }
    })
  }

  async function measureLocoLabProposals() {
    const proposals = locoFilteredProposals.length ? locoFilteredProposals : locoProposals
    if (!sessionId || !imageId || !proposals.length) {
      toast('warning', 'LOCO Lab', 'No hay circulos para medir.')
      return
    }
    await flushDraftIfNeeded()
    const scribble = currentScribbleB64()
    await withLoad('locoLab', async () => {
      try {
        const res = await apiPost('/api/diameter-research/loco/measure', {
          session_id: sessionId,
          image_id: imageId,
          prior_run_id: diamPriorRunId,
          proposals,
          params: locoLabPayloadParams(),
          scribble_map_b64: scribble,
        })
        const measurements = Array.isArray(res?.measurements) ? res.measurements : []
        setLocoMeasuredResults(measurements)
        setLocoEvaluation(res?.summary || null)
        setLocoLabStage('measure')
        const ok = measurements.filter((m) => m.status === 'ok').length
        toast(ok ? 'success' : 'warning', 'LOCO Lab', `${ok}/${measurements.length} mediciones validas.`)
      } catch (err) {
        toast('error', 'LOCO Lab', errMsg(err))
      }
    })
  }

  async function evaluateLocoLab() {
    if (!sessionId || !imageId) return
    await withLoad('locoLab', async () => {
      try {
        const res = await apiPost('/api/diameter-research/loco/evaluate', {
          session_id: sessionId,
          image_id: imageId,
          proposals: locoFilteredProposals.length ? locoFilteredProposals : locoProposals,
          measurements: locoMeasuredResults,
          params: locoLabPayloadParams(),
        })
        setLocoEvaluation(res?.summary || null)
        setLocoLabStage('evaluate')
      } catch (err) {
        toast('error', 'LOCO Lab', errMsg(err))
      }
    })
  }

  async function saveLocoLabRun() {
    const proposals = locoFilteredProposals.length ? locoFilteredProposals : locoProposals
    if (!sessionId || !imageId || (!proposals.length && !locoMeasuredResults.length)) {
      toast('warning', 'LOCO Lab', 'No hay resultados LOCO para guardar.')
      return
    }
    await flushDraftIfNeeded()
    const scribble = currentScribbleB64()
    await withLoad('locoLab', async () => {
      try {
        const res = await apiPost('/api/diameter-research/loco/run', {
          session_id: sessionId,
          image_id: imageId,
          prior_run_id: diamPriorRunId,
          proposals,
          measurements: locoMeasuredResults,
          params: locoLabPayloadParams(),
          scribble_map_b64: scribble,
        })
        setDiamRunCache((prev) => ({ ...prev, [res.run_id]: res }))
        setDiamActiveRunId(res.run_id)
        setDiamResults(Array.isArray(res?.results) ? res.results : [])
        setDiamResultsMode('run')
        if (res?.overlay_b64) setDiamOverlayUrl(b64ToDataUrl(res.overlay_b64, res.overlay_mime || 'image/png'))
        await refreshDiameterRuns(imageId, { silent: true })
        toast('success', 'LOCO Lab', 'Run LOCO guardado.')
      } catch (err) {
        toast('error', 'LOCO Lab', errMsg(err))
      }
    })
  }

  function locoDatasetPayloadCandidates() {
    return (Array.isArray(locoDatasetCircles) ? locoDatasetCircles : []).map((c) => ({
      candidate_id: String(c.candidate_id || ''),
      center_x: Number(c.center_x),
      center_y: Number(c.center_y),
      radius_px: Number(c.radius_px),
      label: c.label || 'invalid_other',
    }))
  }

  function selectedLocoDatasetCircle() {
    return locoDatasetCircles.find((c) => String(c.candidate_id) === String(locoDatasetSelectedId)) || null
  }

  function updateLocoDatasetCircle(id, patch) {
    setLocoDatasetCircles((prev) => prev.map((c) => (String(c.candidate_id) === String(id) ? { ...c, ...patch } : c)))
    if (['valid', 'invalid_crossing', 'invalid_other'].includes(patch?.label)) setLocoDatasetDefaultLabel(patch.label)
    setLocoDatasetFeatures([])
    setLocoDatasetSaveInfo(null)
  }

  function deleteSelectedLocoDatasetCircle() {
    if (!locoDatasetSelectedId) return
    setLocoDatasetCircles((prev) => prev.filter((c) => String(c.candidate_id) !== String(locoDatasetSelectedId)))
    setLocoDatasetSelectedId('')
    setLocoDatasetFeatures([])
    setLocoDatasetSaveInfo(null)
  }

  function clearLocoDatasetCanvas() {
    setLocoDatasetCircles([])
    setLocoDatasetSelectedId('')
    setLocoDatasetDraftCircle(null)
    setLocoDatasetFeatures([])
  }

  async function previewLocoDatasetFeatures() {
    const candidates = locoDatasetPayloadCandidates()
    if (!sessionId || !imageId || !candidates.length) {
      toast('warning', 'Generate Dataset', 'Dibuja al menos un circulo.')
      return
    }
    await flushDraftIfNeeded()
    await withLoad('locoDataset', async () => {
      try {
        const res = await apiPost('/api/diameter-research/loco-dataset/features', {
          session_id: sessionId,
          image_id: imageId,
          source_mode: 'prior_mask',
          prior_run_id: diamPriorRunId,
          candidates,
          params: { patch_size: 64, circle_samples: 128, require_four_cuts: false },
          scribble_map_b64: '',
        })
        setLocoDatasetFeatures(Array.isArray(res?.items) ? res.items : [])
        toast('success', 'Generate Dataset', 'Features calculadas.')
      } catch (err) {
        toast('error', 'Generate Dataset', errMsg(err))
      }
    })
  }

  async function saveLocoDataset() {
    const candidates = locoDatasetPayloadCandidates()
    if (!sessionId || !imageId || !candidates.length) {
      toast('warning', 'Generate Dataset', 'No hay circulos para guardar.')
      return
    }
    await flushDraftIfNeeded()
    await withLoad('locoDataset', async () => {
      try {
        const res = await apiPost('/api/diameter-research/loco-dataset/save', {
          session_id: sessionId,
          image_id: imageId,
          source_mode: 'prior_mask',
          prior_run_id: diamPriorRunId,
          candidates,
          params: { patch_size: 64, circle_samples: 128, require_four_cuts: false },
          scribble_map_b64: '',
        })
        setLocoDatasetSaveInfo(res || null)
        const warn = Array.isArray(res?.warnings) && res.warnings.length ? ` (${res.warnings.join(', ')})` : ''
        toast('success', 'Generate Dataset', `Dataset main actualizado: ${res?.candidate_count || candidates.length} ejemplos totales${warn}.`)
      } catch (err) {
        toast('error', 'Generate Dataset', errMsg(err))
      }
    })
  }

  function defaultLocoAugBlock(type = 'rotate') {
    const id = `aug_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`
    if (type === 'flip') return { id, type, params: { probability: 0.5, modes: 'h,v,hv' } }
    if (type === 'morphology') return { id, type, params: { probability: 0.35, ops: 'erode1,dilate1,open1,close1' } }
    if (type === 'perturb') return { id, type, params: { probability: 0.35, amount_min: 0.005, amount_max: 0.02 } }
    if (type === 'resize_method') return { id, type, params: { probability: 0.45, methods: 'nearest,bilinear_threshold,area_threshold', target_size_min: 40, target_size_max: 56 } }
    if (type === 'resolution') return { id, type, params: { probability: 0.45, sizes: '48,40' } }
    return { id, type: 'rotate', params: { probability: 1, angles: '90,180,270' } }
  }

  function locoAugBlockLabel(type) {
    const labels = {
      rotate: 'Rotate',
      flip: 'Flip',
      morphology: 'Morphology',
      perturb: 'Binary perturbation',
      resize_method: 'Resize method',
      resolution: 'Resolution simulation',
    }
    return labels[type] || type
  }

  function locoAugPayload(overrides = {}) {
    return {
      items: Object.keys(locoAugSelected).filter((id) => locoAugSelected[id]),
      label_filter: locoAugLabelFilter,
      pipeline: locoAugPipeline.map((b) => ({ type: b.type, params: b.params || {} })),
      max_items: 12,
      max_variants_per_source: Math.max(1, Number(locoAugPasses) || 1),
      passes_per_source: Math.max(1, Number(locoAugPasses) || 1),
      ...overrides,
    }
  }

  async function refreshLocoAugItems() {
    await withLoad('locoAugment', async () => {
      try {
        const res = await apiGet('/api/diameter-research/loco-dataset/augment/items')
        const items = Array.isArray(res?.items) ? res.items : []
        setLocoAugItems(items)
        setLocoAugCounts(res?.counts || { total: items.length, valid: 0, invalid: 0, augmented_total: 0, augmented_valid: 0, augmented_invalid: 0 })
        setLocoAugSelected((prev) => {
          const validIds = new Set(items.map((x) => String(x.item_id)))
          const next = {}
          Object.entries(prev || {}).forEach(([id, checked]) => {
            if (checked && validIds.has(id)) next[id] = true
          })
          return next
        })
      } catch (err) {
        toast('error', 'Augmentation', errMsg(err))
      }
    })
  }

  function updateLocoAugBlock(id, patch) {
    setLocoAugPipeline((prev) => prev.map((b) => (b.id === id ? { ...b, ...patch, params: patch.params || b.params || {} } : b)))
    setLocoAugPreview([])
    setLocoAugInfo(null)
  }

  function moveLocoAugBlock(id, dir) {
    setLocoAugPipeline((prev) => {
      const idx = prev.findIndex((b) => b.id === id)
      const nextIdx = idx + dir
      if (idx < 0 || nextIdx < 0 || nextIdx >= prev.length) return prev
      const copy = [...prev]
      const [item] = copy.splice(idx, 1)
      copy.splice(nextIdx, 0, item)
      return copy
    })
    setLocoAugPreview([])
  }

  function removeLocoAugBlock(id) {
    setLocoAugPipeline((prev) => prev.filter((b) => b.id !== id))
    setLocoAugPreview([])
  }

  function selectMixedLocoAugSample() {
    const valid = locoAugItems.filter((x) => x.label === 'valid').slice(0, 3)
    const invalid = locoAugItems.filter((x) => x.label === 'invalid_other' || x.label === 'invalid').slice(0, 2)
    const picked = [...valid, ...invalid]
    const next = {}
    picked.forEach((item) => { next[item.item_id] = true })
    setLocoAugSelected(next)
  }

  async function previewLocoAugmentation() {
    if (!locoAugPipeline.length) {
      toast('warning', 'Augmentation', 'Agrega al menos un bloque.')
      return
    }
    if (!locoAugSelectedCount) {
      toast('warning', 'Augmentation', 'Selecciona ejemplos para previsualizar.')
      return
    }
    await withLoad('locoAugment', async () => {
      try {
        const res = await apiPost('/api/diameter-research/loco-dataset/augment/preview', locoAugPayload())
        setLocoAugPreview(Array.isArray(res?.items) ? res.items : [])
        setLocoAugInfo(res || null)
        toast('success', 'Augmentation', `Preview: ${res?.variant_count || 0} variantes.`)
      } catch (err) {
        toast('error', 'Augmentation', errMsg(err))
      }
    })
  }

  async function applyLocoAugmentation() {
    if (!locoAugPipeline.length) {
      toast('warning', 'Augmentation', 'Agrega al menos un bloque.')
      return
    }
    await withLoad('locoAugment', async () => {
      try {
        const res = await apiPost('/api/diameter-research/loco-dataset/augment/apply', locoAugPayload({ items: [], label_filter: 'all', max_items: 999999, max_variants_per_source: Math.max(1, Number(locoAugPasses) || 1), passes_per_source: Math.max(1, Number(locoAugPasses) || 1) }))
        setLocoAugInfo(res || null)
        await refreshLocoAugItems()
        toast('success', 'Augmentation', `Augmented actualizado: ${res?.augmented_count || 0} imagenes.`)
      } catch (err) {
        toast('error', 'Augmentation', errMsg(err))
      }
    })
  }

  async function clearLocoAugmented() {
    if (!window.confirm('Eliminar por completo la carpeta augmented?')) return
    await withLoad('locoAugment', async () => {
      try {
        const res = await apiPost('/api/diameter-research/loco-dataset/augment/clear', {})
        setLocoAugInfo(res || null)
        setLocoAugPreview([])
        await refreshLocoAugItems()
        toast('success', 'Augmentation', 'Carpeta augmented eliminada.')
      } catch (err) {
        toast('error', 'Augmentation', errMsg(err))
      }
    })
  }

  async function cleanLegacyInvalid() {
    if (!window.confirm('Eliminar todos los ejemplos antiguos "invalid" (label_text=invalid) del dataset? Los ejemplos invalid_crossing e invalid_other se conservan.')) return
    await withLoad('locoDataset', async () => {
      try {
        const res = await apiPost('/api/diameter-research/loco-dataset/clean-legacy-invalid', {})
        toast('success', 'Dataset', `Eliminados ${res.removed_count} ejemplos legacy invalid. Quedan: ${res.valid_count} valid, ${res.crossing_count} crossing, ${res.other_count} other.`)
        await refreshLocoAugItems()
      } catch (err) {
        toast('error', 'Dataset', errMsg(err))
      }
    })
  }

  async function trainLocoModels() {
    await withLoad('locoTraining', async () => {
      try {
        const res = await apiPost('/api/diameter-research/loco-training/train', {
          data_selection: locoTrainingDataSelection,
          test_size: Number(locoTrainingTestSize || 0.2),
          random_seed: Number(locoTrainingSeed || 42),
          pixel_mode: locoTrainingPixelMode,
          circle_prune_px: Number(locoTrainingPrunePx || 0),
          patch_size: 64,
          uses_patch_zoom_factor: !!locoTrainingUseZoom,
          uses_source_radius_px: !!locoTrainingUseSourceRadius,
          models: ['catboost', 'lightgbm', 'xgboost', 'extratrees'],
          multiclass_model: !!locoTrainingMulticlass,
        })
        setLocoTrainingResult(res || null)
        const firstOk = (res?.metrics_summary || []).find((row) => row.status === 'ok')
        if (firstOk?.model_id) setLocoTrainingSelectedModel(firstOk.model_id)
        toast('success', 'Training', `Run ${res?.run_id || ''} completado.`)
      } catch (err) {
        toast('error', 'Training', errMsg(err))
      }
    })
  }

  async function refreshLocoTrainingRuns() {
    await withLoad('locoTraining', async () => {
      try {
        const res = await apiGet('/api/diameter-research/loco-training/runs')
        const items = Array.isArray(res?.items) ? res.items : []
        setLocoTrainingRuns(items)
        setLocoTestTrainingRunId((prev) => (prev && prev !== 'latest' && items.some((r) => r.run_id === prev) ? prev : (items[0]?.run_id || 'latest')))
        setLocoModelRunId((prev) => (prev && prev !== 'latest' && items.some((r) => r.run_id === prev) ? prev : (items[0]?.run_id || 'latest')))
      } catch (err) {
        toast('error', 'Training runs', errMsg(err))
      }
    })
  }

  function locoTestPayloadCandidates() {
    return (Array.isArray(locoTestCircles) ? locoTestCircles : []).map((c) => ({
      candidate_id: String(c.candidate_id || ''),
      center_x: Number(c.center_x),
      center_y: Number(c.center_y),
      radius_px: Number(c.radius_px),
      label: c.label === 'invalid' ? 'invalid_other' : (c.label || 'invalid_other'),
    }))
  }

  function updateLocoTestCircle(id, patch) {
    setLocoTestCircles((prev) => prev.map((c) => (String(c.candidate_id) === String(id) ? { ...c, ...patch } : c)))
    if (patch?.label) setLocoTestDefaultLabel(patch.label)
    setLocoTestResult(null)
  }

  function deleteSelectedLocoTestCircle() {
    if (!locoTestSelectedId) return
    setLocoTestCircles((prev) => prev.filter((c) => String(c.candidate_id) !== String(locoTestSelectedId)))
    setLocoTestSelectedId('')
    setLocoTestResult(null)
  }

  function clearLocoTestCanvas() {
    setLocoTestCircles([])
    setLocoTestSelectedId('')
    setLocoTestDraftCircle(null)
    setLocoTestResult(null)
  }

  async function predictLocoTestCircles() {
    const candidates = locoTestPayloadCandidates()
    if (!sessionId || !imageId || !candidates.length) {
      toast('warning', 'Test circle model', 'Dibuja al menos un circulo.')
      return
    }
    await flushDraftIfNeeded()
    await withLoad('locoTest', async () => {
      try {
        const res = await apiPost('/api/diameter-research/loco-training/test-circles', {
          session_id: sessionId,
          image_id: imageId,
          source_mode: 'prior_mask',
          prior_run_id: diamPriorRunId,
          training_run_id: locoTestTrainingRunId || 'latest',
          model_id: locoTestModelId,
          threshold: Number(locoTestThreshold || 0.5),
          candidates,
          params: {},
          scribble_map_b64: '',
        })
        setLocoTestResult(res || null)
        toast('success', 'Test circle model', `${res?.candidate_count || 0} circulos evaluados.`)
      } catch (err) {
        toast('error', 'Test circle model', errMsg(err))
      }
    })
  }

  function updateLocoModelParam(key, value) {
    setLocoModelParams((prev) => ({ ...prev, [key]: value }))
    if (['grid_step', 'min_radius', 'max_radius', 'radius_step'].includes(key)) {
      setLocoModelResult(null)
      setLocoModelSelectedId('')
    }
    if (key !== 'locoModelPreset') {
      setLocoModelPreset('custom')
    }
  }

  function applyLocoModelPreset(presetKey) {
    const preset = LOCO_PRESETS[presetKey]
    if (!preset || presetKey === 'custom') {
      setLocoModelPreset('custom')
      return
    }
    setLocoModelPreset(presetKey)
    setLocoModelParams((prev) => ({ ...prev, ...preset }))
    setLocoModelResult(null)
    setLocoModelSelectedId('')
  }

  async function detectLocoModelCircles() {
    if (!sessionId || !imageId || !imageUrl) {
      toast('warning', 'LOCO Detector', 'Carga una imagen primero.')
      return
    }
    await flushDraftIfNeeded()
    await withLoad('locoModel', async () => {
      try {
        const res = await apiPost('/api/diameter-research/loco-models/detect-circles', {
          session_id: sessionId,
          image_id: imageId,
          source_mode: 'prior_mask',
          prior_run_id: diamPriorRunId,
          model_run_id: locoModelRunId || 'latest',
          model_id: locoModelId,
          patch_size: 64,
          grid_step: Number(locoModelParams.grid_step || 6),
          min_radius: Number(locoModelParams.min_radius || 8),
          max_radius: Number(locoModelParams.max_radius || 32),
          radius_step: Number(locoModelParams.radius_step || 2),
          threshold: Number(locoModelParams.threshold || 0.8),
          use_radius_thresholds: !!locoModelParams.use_radius_thresholds,
          small_threshold: Number(locoModelParams.small_threshold || 0.8),
          medium_threshold: Number(locoModelParams.medium_threshold || 0.8),
          large_threshold: Number(locoModelParams.large_threshold || 0.9),
          small_radius_limit: Number(locoModelParams.small_radius_limit || 14),
          large_radius_limit: Number(locoModelParams.large_radius_limit || 24),
          use_nms: !!locoModelParams.use_nms,
          nms_mode: locoModelParams.nms_mode || 'distance_radius',
          nms_distance_factor: Number(locoModelParams.nms_distance_factor || 0.5),
          radius_similarity_factor: Number(locoModelParams.radius_similarity_factor || 0.4),
          circle_iou_threshold: Number(locoModelParams.circle_iou_threshold || 0.4),
          candidate_sampling_mode: locoModelParams.candidate_sampling_mode || 'row_major',
          candidate_random_seed: Number(locoModelParams.candidate_random_seed || 42),
          tile_size_px: Number(locoModelParams.tile_size_px || 128),
          candidate_max_per_tile: Number(locoModelParams.candidate_max_per_tile || 0),
          return_rejected: !!locoModelParams.return_rejected,
          max_candidates: Number(locoModelParams.max_candidates || 8000),
          max_return_rejected: Number(locoModelParams.max_return_rejected || 5000),
          crossing_threshold: Number(locoModelParams.crossing_threshold || 0.5),
          use_spatial_final_filter: !!locoModelParams.use_spatial_final_filter,
          spatial_final_tile_px: Number(locoModelParams.spatial_final_tile_px || 128),
          spatial_final_max_per_tile: Number(locoModelParams.spatial_final_max_per_tile || 3),
          spatial_final_min_center_distance_factor: Number(locoModelParams.spatial_final_min_center_distance_factor || 1.0),
          scribble_map_b64: '',
        })
        setLocoModelResult(res || null)
        setLocoModelMeasurement(null)
        const first = Array.isArray(res?.accepted) ? res.accepted[0] : null
        setLocoModelSelectedId(String(first?.candidate_id || ''))
        toast('success', 'LOCO Detector', `${res?.summary?.accepted_after_nms || 0} circulos aceptados.`)
      } catch (err) {
        toast('error', 'LOCO Detector', errMsg(err))
      }
    })
  }

  async function measureLocoModelAccepted() {
    const accepted = Array.isArray(locoModelResult?.accepted) ? locoModelResult.accepted : []
    if (!sessionId || !imageId || !accepted.length) {
      toast('warning', 'LOCO Detector', 'Primero detecta circulos aceptados.')
      return
    }
    await flushDraftIfNeeded()
    await withLoad('locoModel', async () => {
      try {
        const res = await apiPost('/api/diameter-research/loco-models/measure-accepted', {
          session_id: sessionId,
          image_id: imageId,
          source_mode: 'prior_mask',
          prior_run_id: diamPriorRunId,
          candidates: accepted,
          params: { measure_limit: accepted.length, circle_samples: 128 },
          scribble_map_b64: '',
        })
        setLocoModelMeasurement(res || null)
        if (res?.run_id) {
          setDiamRunCache((prev) => ({ ...prev, [res.run_id]: res }))
          setDiamActiveRunId(res.run_id)
          setDiamResults(Array.isArray(res.results) ? res.results : [])
          setDiamResultsMode('run')
          if (res.overlay_b64) setDiamOverlayUrl(b64ToDataUrl(res.overlay_b64, res.overlay_mime || 'image/png'))
          await refreshDiameterRuns(imageId, { silent: true })
        }
        // Populate diamPoints with accepted circle centers
        const points = accepted.map((c) => ({
          x: c.center_x ?? c.x ?? 0,
          y: c.center_y ?? c.y ?? 0,
          kind: 'circle',
          radius: c.radius ?? c.r ?? 0,
          score: c.valid_score ?? c.score ?? 0,
        }))
        await updateDiameterPoints('replace', { points })
        // Navigate to Diameter Measurement tab
        handleGroupChange('detection')
        handleTabChange('diameter')
        toast('success', 'LOCO Detector', `${res?.meta?.points_ok ?? 0} mediciones guardadas.`)
      } catch (err) {
        toast('error', 'LOCO Detector', errMsg(err))
      }
    })
  }

  // --- Calibration functions ---
  async function saveCalibration() {
    if (!imageId) return
    await withLoad('calibration', async () => {
      try {
        await apiPost('/api/diameter-research/calibration/save', {
          image_id: imageId,
          ...diamCalibration,
        })
        toast('success', 'Calibracion', 'Calibracion guardada.')
      } catch (err) {
        toast('error', 'Calibracion', errMsg(err))
      }
    })
  }

  async function loadCalibration() {
    if (!imageId) return
    await withLoad('calibration', async () => {
      try {
        const res = await apiPost('/api/diameter-research/calibration/load', {
          image_id: imageId,
        })
        if (res) {
          setDiamCalibration({
            enabled: res.enabled ?? false,
            known_nm: res.known_nm ?? 100,
            pixel_distance: res.pixel_distance ?? 100,
            nm_per_px: res.nm_per_px ?? 1.0,
            unit: res.unit ?? 'nm',
          })
          toast('success', 'Calibracion', 'Calibracion cargada.')
        }
      } catch (err) {
        toast('error', 'Calibracion', errMsg(err))
      }
    })
  }

  async function deleteCalibration() {
    if (!imageId) return
    await withLoad('calibration', async () => {
      try {
        await apiPost('/api/diameter-research/calibration/delete', {
          image_id: imageId,
        })
        setDiamCalibration({
          enabled: false,
          known_nm: 100,
          pixel_distance: 100,
          nm_per_px: 1.0,
          unit: 'nm',
        })
        toast('success', 'Calibracion', 'Calibracion eliminada.')
      } catch (err) {
        toast('error', 'Calibracion', errMsg(err))
      }
    })
  }

  function clearLocoModelDetector() {
    setLocoModelResult(null)
    setLocoModelMeasurement(null)
    setLocoModelSelectedId('')
  }

  async function refreshDiameterRuns(customImageId = '', { silent = false } = {}) {
    const iid = String(customImageId || imageId || '').trim()
    if (!iid) return
    await withLoad('diamList', async () => {
      try {
        const res = await apiGet(`/api/diameter-research/results/list?image_id=${encodeURIComponent(iid)}`)
        const items = Array.isArray(res?.items) ? res.items : []
        setDiamRuns(items)
        setDiamActiveRunId((prev) => (items.some((r) => r.run_id === prev) ? prev : (items[0]?.run_id || '')))
      } catch (err) {
        if (!silent) toast('error', 'Diameter runs', errMsg(err))
      }
    })
  }

  async function loadDiameterRun(runId) {
    const rid = String(runId || '').trim()
    if (!rid) return
    if (diamRunCache[rid]) {
      const item = diamRunCache[rid]
      setDiamResults(Array.isArray(item.results) ? item.results : [])
      setDiamResultsMode('run')
      setDiamReviewTarget(null)
      if (item.overlay_b64) setDiamOverlayUrl(b64ToDataUrl(item.overlay_b64, item.overlay_mime || 'image/png'))
      setDiamActiveRunId(rid)
      return
    }
    await withLoad('diamGet', async () => {
      try {
        const res = await apiGet(`/api/diameter-research/results/get?run_id=${encodeURIComponent(rid)}`)
        setDiamRunCache((prev) => ({ ...prev, [rid]: res }))
        setDiamActiveRunId(rid)
        setDiamResults(Array.isArray(res?.results) ? res.results : [])
        setDiamResultsMode('run')
        setDiamReviewTarget(null)
        if (res?.overlay_b64) setDiamOverlayUrl(b64ToDataUrl(res.overlay_b64, res.overlay_mime || 'image/png'))
      } catch (err) {
        toast('error', 'Diameter run', errMsg(err))
      }
    })
  }

  const MANUAL_DIAMETER_METHODS = ['circle_square_mask_diameter', 'manual_dual_side_caliper', 'manual_line_direct_caliper']

  function lineToRunPoint(line) {
    return {
      x: (Number(line.start.x) + Number(line.end.x)) * 0.5,
      y: (Number(line.start.y) + Number(line.end.y)) * 0.5,
    }
  }

  function lineToParams(line, kind = 'mask', idx = -1) {
    return {
      manual_left_x: Number(line.start.x),
      manual_left_y: Number(line.start.y),
      manual_right_x: Number(line.end.x),
      manual_right_y: Number(line.end.y),
      manual_geometry_id: lineGeometryId(kind, line, idx),
    }
  }

  function buildLineBatch(kind, lines, activeOnly = false) {
    const methodId = kind === 'direct' ? 'manual_line_direct_caliper' : 'manual_dual_side_caliper'
    const activeIdx = kind === 'direct' ? manualDirectLineActiveIdx : manualMaskLineActiveIdx
    const selected = activeOnly
      ? [lines[Number.isInteger(activeIdx) && activeIdx >= 0 && activeIdx < lines.length ? activeIdx : lines.length - 1]].filter(Boolean)
      : lines
    if (!selected.length) return null
    const params = diameterParamsPayload()
    params.manual_lines_by_point = selected.map((line, idx) => lineToParams(line, kind, idx))
    Object.assign(params, lineToParams(selected[0], kind, 0))
    return {
      method_id: methodId,
      points: selected.map(lineToRunPoint),
      params,
      geometry_centers: selected.map(lineToRunPoint),
      geometry_ids: selected.map((line, idx) => lineGeometryId(kind, line, idx)),
    }
  }

  function buildCircleBatch(circles, activeOnly = false) {
    const selected = (Array.isArray(circles) ? circles : [])
      .filter((circle) => circle?.center && Number.isFinite(Number(circle.radius)))
    if (!selected.length) return null
    if (activeOnly && diamMethodId !== 'circle_square_mask_diameter') return null
    const params = diameterParamsPayload()
    params.circle_square_seed_mode = 'manual_circle'
    params.circle_square_circles_by_point = selected.map((circle, idx) => ({
      circle_square_seed_radius_px: Math.max(1, Number(circle.radius)),
      circle_square_center_x: Number(circle.center.x),
      circle_square_center_y: Number(circle.center.y),
      circle_square_geometry_id: String(circle.geometry_id || `circle_square_${idx}`),
    }))
    const first = selected[0]
    params.circle_square_seed_radius_px = Math.max(1, Number(first.radius))
    params.circle_square_center_x = Number(first.center.x)
    params.circle_square_center_y = Number(first.center.y)
    params.circle_square_geometry_id = String(first.geometry_id || 'circle_square_0')
    return {
      method_id: 'circle_square_mask_diameter',
      points: selected.map((circle) => ({ x: Number(circle.center.x), y: Number(circle.center.y) })),
      params,
      geometry_centers: selected.map((circle) => ({ x: Number(circle.center.x), y: Number(circle.center.y) })),
      geometry_ids: selected.map((circle, idx) => String(circle.geometry_id || `circle_square_${idx}`)),
    }
  }

  function pointNearAnyGeometry(point, centers) {
    return centers.some((center) => Math.hypot(Number(point.x) - Number(center.x), Number(point.y) - Number(center.y)) <= 2.5)
  }

  function resultIdentity(result) {
    const r = result || {}
    const method = String(r.method_id || r.method || r.route || '')
    const geometryId = String(r.interactive_geometry_id || r.manual_geometry_id || r.circle_square_geometry_id || '')
    const original = Array.isArray(r.original_xy) ? r.original_xy : null
    const x = original ? Number(original[0]) : Number(r.x)
    const y = original ? Number(original[1]) : Number(r.y)
    const left = Array.isArray(r.left_edge_xy) ? r.left_edge_xy : (Array.isArray(r.debug_left_edge_xy) ? r.debug_left_edge_xy : null)
    const right = Array.isArray(r.right_edge_xy) ? r.right_edge_xy : (Array.isArray(r.debug_right_edge_xy) ? r.debug_right_edge_xy : null)
    const lineKey = left && right
      ? [left[0], left[1], right[0], right[1]].map((v) => Math.round(Number(v) * 10)).join('_')
      : ''
    return { method, geometryId, x, y, lineKey }
  }

  function sameResultIdentity(a, b) {
    const aa = resultIdentity(a)
    const bb = resultIdentity(b)
    if (!aa.method || !bb.method || aa.method !== bb.method) return false
    if (aa.geometryId && bb.geometryId) return aa.geometryId === bb.geometryId
    if (aa.lineKey || bb.lineKey) return Boolean(aa.lineKey && bb.lineKey && aa.lineKey === bb.lineKey)
    if (![aa.x, aa.y, bb.x, bb.y].every(Number.isFinite)) return false
    return Math.hypot(aa.x - bb.x, aa.y - bb.y) <= 2.5
  }

  function mergeDiameterResults(previous, incoming) {
    const prev = Array.isArray(previous) ? previous : []
    const next = Array.isArray(incoming) ? incoming : []
    if (!next.length) return prev
    // Keep prior manual measurements visible. Only replace an existing result
    // when the new result can be matched by an explicit geometry id or exact
    // endpoint line; never collapse different manual geometries by center.
    return [
      ...prev.filter((oldResult) => !next.some((newResult) => sameResultIdentity(oldResult, newResult))),
      ...next,
    ]
  }

  function buildAutomaticPointBatch(excludedCenters, activeOnly = false) {
    const manualActive = MANUAL_DIAMETER_METHODS.includes(diamMethodId)
    const methodId = manualActive ? 'hybrid_profile_diameter_v3_2_auto' : diamMethodId
    const sourcePoints = activeOnly
      ? [diamPoints[diamActivePointIdx >= 0 ? diamActivePointIdx : 0]].filter(Boolean)
      : diamPoints
    const points = sourcePoints
      .filter((point) => point?.x != null && point?.y != null)
      .filter((point) => !pointNearAnyGeometry(point, excludedCenters))
    if (!points.length) return null
    return {
      method_id: methodId,
      points: points.map((point) => ({ x: Number(point.x), y: Number(point.y) })),
      params: diameterParamsPayload(),
      geometry_centers: [],
    }
  }

  function buildDiameterRunBatches(activeOnly) {
    if (activeOnly) {
      if (diamMethodId === 'manual_dual_side_caliper') {
        return [buildLineBatch('mask', normalizedLinesForKind('mask'), true)].filter(Boolean)
      }
      if (diamMethodId === 'manual_line_direct_caliper') {
        return [buildLineBatch('direct', normalizedLinesForKind('direct'), true)].filter(Boolean)
      }
      if (diamMethodId === 'circle_square_mask_diameter') {
        const activeCircle = manualCircleActiveIdx >= 0 && manualCircleActiveIdx < manualCircles.length
          ? manualCircles[manualCircleActiveIdx]
          : manualCircles[manualCircles.length - 1]
        return [buildCircleBatch(activeCircle ? [activeCircle] : [], true)].filter(Boolean)
      }
      return [buildAutomaticPointBatch([], true)].filter(Boolean)
    }

    const batches = []
    const excludedCenters = []
    const maskBatch = buildLineBatch('mask', normalizedLinesForKind('mask'), false)
    if (maskBatch) {
      batches.push(maskBatch)
      excludedCenters.push(...maskBatch.geometry_centers)
    }
    const directBatch = buildLineBatch('direct', normalizedLinesForKind('direct'), false)
    if (directBatch) {
      batches.push(directBatch)
      excludedCenters.push(...directBatch.geometry_centers)
    }
    const circleBatch = buildCircleBatch(manualCircles, false)
    if (circleBatch) {
      batches.push(circleBatch)
      excludedCenters.push(...circleBatch.geometry_centers)
    }
    const pointBatch = buildAutomaticPointBatch(excludedCenters, false)
    if (pointBatch) batches.push(pointBatch)
    return batches
  }

  async function runDiameterResearch(activeOnly) {
    if (!sessionId || !imageId || !imageUrl) {
      toast('warning', 'Diameter Research', 'Carga una imagen primero.')
      return
    }
    await flushDraftIfNeeded()
    const scribble = currentScribbleB64()
    const batches = buildDiameterRunBatches(Boolean(activeOnly))
    if (!batches.length) {
      if (diamMethodId === 'circle_square_mask_diameter') {
        setDiamMethodPanel('manual')
        setDiamViewerMode('circle')
        setDiamParams((prev) => ({ ...prev, circle_square_seed_mode: 'manual_circle' }))
        toast('warning', 'Circle-square', 'Dibuja un circulo manual antes de ejecutar este metodo.')
      } else if (['manual_dual_side_caliper', 'manual_line_direct_caliper'].includes(diamMethodId)) {
        setDiamMethodPanel('manual')
        setDiamViewerMode('manual')
        toast('warning', 'Linea manual', 'Dibuja la linea manteniendo click y soltando al final.')
      } else {
        toast('warning', 'Diameter Research', 'Agrega al menos un punto o geometria manual.')
      }
      return
    }
    const hideManualCircleAfterRun = batches.some((batch) => batch.method_id === 'circle_square_mask_diameter')
    if (hideManualCircleAfterRun) consumeDiameterManualCircle()
    await withLoad('diamRun', async () => {
      try {
        const responses = []
        for (const batch of batches) {
          const res = await apiPost('/api/diameter-research/run', {
            session_id: sessionId,
            image_id: imageId,
            method_id: batch.method_id,
            source_mode: diamSourceMode,
            prior_run_id: diamPriorRunId,
            points: batch.points,
            active_only: false,
            params: batch.params,
            scribble_map_b64: scribble,
          })
          responses.push(res)
        }
        setDiamRunCache((prev) => {
          const next = { ...prev }
          responses.forEach((res) => {
            if (res?.run_id) next[res.run_id] = res
          })
          return next
        })
        const lastRes = responses[responses.length - 1] || null
        setDiamResultsMode('composite')
        if (lastRes?.run_id) setDiamActiveRunId(lastRes.run_id)
        const nextResults = responses.flatMap((res) => (Array.isArray(res?.results) ? res.results : []))
        setDiamResults((prev) => mergeDiameterResults(prev, nextResults))
        if (nextResults.length === 1) selectDiameterResultForReview(nextResults[0], { silent: true, runId: lastRes?.run_id || '' })
        if (lastRes?.overlay_b64) setDiamOverlayUrl(b64ToDataUrl(lastRes.overlay_b64, lastRes.overlay_mime || 'image/png'))
        if (hideManualCircleAfterRun) {
          consumeDiameterManualCircle()
        }
        await refreshDiameterRuns(imageId, { silent: true })
        const okCount = nextResults.filter((r) => r.status === 'ok').length
        const methodCount = new Set(batches.map((batch) => batch.method_id)).size
        toast(okCount ? 'success' : 'warning', 'Diameter Research', `${okCount}/${nextResults.length} puntos medidos en ${methodCount} metodo(s).`)
      } catch (err) {
        toast('error', 'Diameter Research', errMsg(err))
      }
    })
  }

  async function exportDiameterReport() {
    if (!imageId) return
    await withLoad('diamExport', async () => {
      try {
        const res = await apiGet(`/api/diameter-research/reports/export?image_id=${encodeURIComponent(imageId)}`)
        setDiamReportInfo(res || null)
        toast('success', 'Diameter report', 'CSV + JSON + galeria generados.')
      } catch (err) {
        toast('error', 'Diameter report', errMsg(err))
      }
    })
  }

  function updateValidationForm(key, value) {
    setValidationForm((prev) => ({ ...prev, [key]: value }))
  }

  function activeValidationPoint() {
    if (diamReviewTarget?.point) return diamReviewTarget.point
    if (diamActivePointIdx >= 0 && diamActivePointIdx < diamPoints.length) return diamPoints[diamActivePointIdx]
    return diamPoints[0] || null
  }

  function resultPointForReview(result) {
    const r = result || {}
    const idx = Number(r.point_index)
    if (Number.isInteger(idx) && idx >= 0 && idx < diamPoints.length) {
      const p = diamPoints[idx]
      return { x: Number(p.x), y: Number(p.y), point_index: idx }
    }
    const original = Array.isArray(r.original_xy) ? r.original_xy : null
    const x = original ? Number(original[0]) : Number(r.x)
    const y = original ? Number(original[1]) : Number(r.y)
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null
    return { x, y, point_index: Number.isInteger(idx) ? idx : 0 }
  }

  function samePoint(a, b, tol = 1.25) {
    if (!a || !b) return false
    return Math.abs(Number(a.x) - Number(b.x)) <= tol && Math.abs(Number(a.y) - Number(b.y)) <= tol
  }

  function currentManualGeometryIds() {
    const ids = new Set()
    manualMaskLines.forEach((line, idx) => ids.add(lineGeometryId('mask', line, idx)))
    manualDirectLines.forEach((line, idx) => ids.add(lineGeometryId('direct', line, idx)))
    if (manualMaskLineDraft?.start && manualMaskLineDraft?.end) ids.add(lineGeometryId('mask', manualMaskLineDraft, -1))
    if (manualDirectLineDraft?.start && manualDirectLineDraft?.end) ids.add(lineGeometryId('direct', manualDirectLineDraft, -1))
    if (manualCircleDraft?.center) ids.add(String(manualCircleDraft.geometry_id || 'circle_square_current'))
    return ids
  }

  function resultHasCurrentPoint(result) {
    const r = result || {}
    const geometryId = String(r.interactive_geometry_id || r.manual_geometry_id || r.circle_square_geometry_id || '')
    if (geometryId) return true
    const original = Array.isArray(r.original_xy) ? r.original_xy : null
    const x = original ? Number(original[0]) : Number(r.x)
    const y = original ? Number(original[1]) : Number(r.y)
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      const idx = Number(r.point_index)
      return Number.isInteger(idx) && idx >= 0 && idx < diamPoints.length
    }
    return diamPoints.some((p) => samePoint({ x, y }, p, 2.0))
  }

  function selectDiameterResultForReview(result, { silent = false, runId = '' } = {}) {
    const r = result || {}
    const point = resultPointForReview(r)
    if (!point) {
      if (!silent) toast('warning', 'Revision por punto', 'No se pudo resolver el punto del resultado.')
      return
    }
    const pointIdx = Number(point.point_index)
    const targetRunId = String(runId || diamActiveRunId || '')
    setDiamReviewTarget({
      point,
      point_index: pointIdx,
      run_id: targetRunId,
      method_id: r.method_id || diamRunCache[targetRunId]?.method_id || '',
      status: String(r.status || ''),
      reason: String(r.reason || ''),
      diameter_px: r.diameter_px == null ? null : Number(r.diameter_px),
      quality_label: String(r.quality_label || ''),
      confidence: r.confidence == null ? null : Number(r.confidence),
      measurement_mode: String(r.measurement_mode || ''),
      geometry_status: String(r.geometry_status || ''),
      support_status: String(r.support_status || ''),
      geometry_control_status: String(r.geometry_control_status || ''),
      profile_length_effective_px: r.profile_length_effective_px == null ? null : Number(r.profile_length_effective_px),
      methodology_id: String(r.methodology_id || ''),
      local_context_label: String(r.local_context_label || ''),
      selected_edge_policy: String(r.selected_edge_policy || ''),
      size_route: String(r.size_route || ''),
      fiber_size_mode: String(r.fiber_size_mode || ''),
      auto_size_reason: String(r.auto_size_reason || ''),
      diameter_route: String(r.diameter_route || r.size_route || ''),
      mask_method: String(r.mask_method || ''),
      mask_confidence: r.mask_confidence == null ? null : Number(r.mask_confidence),
      mask_center_shift_px: r.mask_center_shift_px == null ? null : Number(r.mask_center_shift_px),
      mask_caliper_diameter_px: r.mask_caliper_diameter_px == null ? null : Number(r.mask_caliper_diameter_px),
      mask_raycast_diameter_px: r.mask_raycast_diameter_px == null ? null : Number(r.mask_raycast_diameter_px),
      circle_radius_px: r.circle_radius_px == null ? null : Number(r.circle_radius_px),
      square_samples_valid: r.square_samples_valid == null ? null : Number(r.square_samples_valid),
      square_samples_total: r.square_samples_total == null ? null : Number(r.square_samples_total),
      manual_input_diameter_px: r.manual_input_diameter_px == null ? null : Number(r.manual_input_diameter_px),
      ellipse_minor_px: r.ellipse_minor_px == null ? null : Number(r.ellipse_minor_px),
      ellipse_major_px: r.ellipse_major_px == null ? null : Number(r.ellipse_major_px),
      halo_status: String(r.halo_status || ''),
      ridge_anchor_status: String(r.ridge_anchor_status || ''),
      flux_status: String(r.flux_status || ''),
      contour_refine_status: String(r.contour_refine_status || ''),
      curvelet_status: String(r.curvelet_status || ''),
      orientation_coherence: r.orientation_coherence == null ? null : Number(r.orientation_coherence),
      used_upscale: Boolean(r.used_upscale),
      scale_factor: r.scale_factor == null ? null : Number(r.scale_factor),
      small_diameter_suspect: Boolean(r.small_diameter_suspect),
    })
    if (Number.isInteger(pointIdx) && pointIdx >= 0 && pointIdx < diamPoints.length) {
      void updateDiameterPoints('set_active', { active_index: pointIdx })
    }
    const existing = validationCases.find((c) => samePoint(c?.point, point))
    if (existing) {
      applyValidationCaseToForm(existing)
      if (!silent) toast('success', 'Revision por punto', `Caso existente cargado para punto ${pointIdx + 1}.`)
      return
    }
    setValidationActiveCaseId('')
    setValidationForm((prev) => ({
      ...prev,
      case_id: '',
      manual_diameter_px: '',
      manual_left_x: '',
      manual_left_y: '',
      manual_right_x: '',
      manual_right_y: '',
      measurement_decision: 'unreviewed',
      notes: '',
      result_comment: r.status === 'ok'
        ? `Resultado automatico: ${Number(r.diameter_px || 0).toFixed(2)} px (${r.quality_label || 'sin etiqueta'}).`
        : `Resultado rechazado: ${r.reason || r.quality_label || 'sin razon'}.`,
    }))
    setLineDraftForKind(currentManualLineKind(), { start: null, end: null })
    if (!silent) toast('success', 'Revision por punto', `Punto ${pointIdx + 1} listo para revisar.`)
  }

  function openDiameterPointReview(result) {
    selectDiameterResultForReview(result, { silent: true })
    setPointReviewOpen(true)
  }

  function manualLineFromForm(form = validationForm) {
    const raw = [form.manual_left_x, form.manual_left_y, form.manual_right_x, form.manual_right_y]
    if (raw.some((v) => String(v ?? '').trim() === '')) return { start: null, end: null }
    const lx = Number(form.manual_left_x)
    const ly = Number(form.manual_left_y)
    const rx = Number(form.manual_right_x)
    const ry = Number(form.manual_right_y)
    if (![lx, ly, rx, ry].every(Number.isFinite)) return { start: null, end: null }
    return { start: { x: lx, y: ly }, end: { x: rx, y: ry } }
  }

  function applyValidationCaseToForm(item) {
    const c = item || {}
    setValidationActiveCaseId(String(c.case_id || ''))
    const nextForm = {
      case_id: String(c.case_id || ''),
      category: String(c.category || 'borde_limpio'),
      manual_diameter_px: c.manual_diameter_px == null ? '' : String(c.manual_diameter_px),
      manual_left_x: c.manual_left_x == null ? '' : String(c.manual_left_x),
      manual_left_y: c.manual_left_y == null ? '' : String(c.manual_left_y),
      manual_right_x: c.manual_right_x == null ? '' : String(c.manual_right_x),
      manual_right_y: c.manual_right_y == null ? '' : String(c.manual_right_y),
      measurement_decision: String(c.measurement_decision || 'unreviewed'),
      quality_manual: String(c.quality_manual || 'medium'),
      notes: String(c.notes || ''),
      result_comment: String(c.result_comment || ''),
    }
    setValidationForm(nextForm)
    setLineDraftForKind(currentManualLineKind(), manualLineFromForm(nextForm))
  }

  async function refreshValidationCases(customImageId = '', { silent = false } = {}) {
    const iid = String(customImageId || imageId || '').trim()
    if (!iid) return
    await withLoad('validationList', async () => {
      try {
        const res = await apiGet(`/api/diameter-research/validation/cases?image_id=${encodeURIComponent(iid)}`)
        const items = Array.isArray(res?.items) ? res.items : []
        setValidationCases(items)
        setValidationActiveCaseId((prev) => (items.some((c) => c.case_id === prev) ? prev : (items[0]?.case_id || '')))
      } catch (err) {
        if (!silent) toast('error', 'Validacion', errMsg(err))
      }
    })
  }

  async function saveValidationCase({ silent = false, overrides = {} } = {}) {
    if (!sessionId || !imageId) {
      toast('warning', 'Validacion', 'Carga una imagen primero.')
      return null
    }
    const form = { ...validationForm, ...overrides }
    setValidationForm(form)
    const existingCase = validationCases.find((c) => String(c.case_id || '') === String(form.case_id || ''))
    const point = existingCase?.point || activeValidationPoint()
    if (!point) {
      toast('warning', 'Validacion', 'Agrega o selecciona un punto de medicion.')
      return null
    }
    let saved = null
    await withLoad('validationSave', async () => {
      try {
        const manual = String(form.manual_diameter_px || '').trim()
        const res = await apiPost('/api/diameter-research/validation/case/upsert', {
          session_id: sessionId,
          image_id: imageId,
          case_id: form.case_id,
          point: { x: Number(point.x), y: Number(point.y) },
          category: form.category,
          quality_manual: form.quality_manual,
          manual_diameter_px: manual ? Number(manual) : null,
          manual_left_x: String(form.manual_left_x || '').trim() ? Number(form.manual_left_x) : null,
          manual_left_y: String(form.manual_left_y || '').trim() ? Number(form.manual_left_y) : null,
          manual_right_x: String(form.manual_right_x || '').trim() ? Number(form.manual_right_x) : null,
          manual_right_y: String(form.manual_right_y || '').trim() ? Number(form.manual_right_y) : null,
          measurement_decision: form.measurement_decision,
          notes: form.notes,
          result_comment: form.result_comment,
          source_mode: diamSourceMode,
          prior_run_id: diamPriorRunId,
          params: diameterParamsPayload(),
        })
        saved = res?.case || null
        if (saved) {
          applyValidationCaseToForm(saved)
          await refreshValidationCases(imageId, { silent: true })
        }
        if (!silent) toast('success', 'Validacion', 'Caso guardado.')
      } catch (err) {
        toast('error', 'Validacion', errMsg(err))
      }
    })
    return saved
  }

  async function runValidationCase() {
    if (!sessionId || !imageId) return
    const saved = validationActiveCaseId ? await saveValidationCase({ silent: true }) : await saveValidationCase({ silent: true })
    const caseId = String(saved?.case_id || validationActiveCaseId || '').trim()
    if (!caseId) return
    await flushDraftIfNeeded()
    const scribble = currentScribbleB64()
    await withLoad('validationRun', async () => {
      try {
        const res = await apiPost('/api/diameter-research/validation/run-case', {
          session_id: sessionId,
          image_id: imageId,
          case_id: caseId,
          source_mode: diamSourceMode,
          prior_run_id: diamPriorRunId,
          params: diameterParamsPayload(),
          scribble_map_b64: scribble,
          methods: [
            'hybrid_profile_diameter_v1',
            'hybrid_profile_diameter_v2',
            'hybrid_profile_diameter_v3_1',
            'hybrid_profile_diameter_v3_2_auto',
            'hybrid_profile_diameter_v3_2_small_mask',
            'hybrid_profile_diameter_v3_2_large_image',
            'circle_square_mask_diameter',
            'manual_dual_side_caliper',
            'manual_line_direct_caliper',
            'ellipse_oriented_fit',
          ],
        })
        const runs = res?.runs || {}
        setDiamRunCache((prev) => {
          const next = { ...prev }
          Object.values(runs).forEach((runItem) => {
            if (runItem?.run_id) next[runItem.run_id] = runItem
          })
          return next
        })
        if (res?.case) applyValidationCaseToForm(res.case)
        await refreshValidationCases(imageId, { silent: true })
        await refreshDiameterRuns(imageId, { silent: true })
        toast('success', 'Validacion', 'Caso ejecutado con baselines y geometria interactiva.')
      } catch (err) {
        toast('error', 'Validacion', errMsg(err))
      }
    })
  }

  async function exportValidationAutofill() {
    if (!imageId) return
    await withLoad('validationExport', async () => {
      try {
        const res = await apiGet(`/api/diameter-research/validation/export?image_id=${encodeURIComponent(imageId)}`)
        setValidationExportInfo(res || null)
        toast('success', 'Validacion', 'MD + CSV + JSON generados.')
      } catch (err) {
        toast('error', 'Validacion', errMsg(err))
      }
    })
  }

  const latestDecision = useMemo(() => {
    if (!activeRunId) return ''
    const row = reviewByRun[activeRunId]
    return row ? (reviewLabel(row.decision) || String(row.decision || '').toUpperCase()) : ''
  }, [reviewByRun, activeRunId])

  const groups = useMemo(() => {
    const s = new Set()
    experiments.forEach((e) => s.add(e.group))
    return Array.from(s).sort()
  }, [experiments])

  const selectedPriorRun = useMemo(() => {
    if (diamPriorRunId === 'latest') return runs[0] || null
    return runs.find((r) => r.run_id === diamPriorRunId) || null
  }, [runs, diamPriorRunId])

  const annotBrushPx = useMemo(() => {
    const base = Math.max(1, Number(brushSize) || 1)
    if (!brushAutoScale || !imageDims.w || !imageDims.h) return base
    if (base <= 10) return Math.round(base)
    const longest = Math.max(imageDims.w, imageDims.h)
    const factor = clamp(longest / 900, 0.65, 2.6)
    return Math.max(1, Math.round(base * factor))
  }, [brushSize, brushAutoScale, imageDims])

  const brushSliderValue = useMemo(() => brushPxToSlider(brushSize), [brushSize])

  const editorRenderMetrics = useMemo(() => getEditorRenderMetrics(), [imageDims, stageSize, viewerZoom, viewerOffset])
  const diameterRenderMetrics = useMemo(
    () => getDiameterRenderMetrics(),
    [imageDims, diamStageSize, diamViewerZoom, diamViewerOffset],
  )
  const locoRenderMetrics = useMemo(
    () => getLocoRenderMetrics(),
    [imageDims, locoStageSize, locoViewerZoom, locoViewerOffset],
  )
  const locoDatasetRenderMetrics = useMemo(
    () => getLocoDatasetRenderMetrics(),
    [imageDims, locoDatasetStageSize, locoDatasetZoom, locoDatasetOffset],
  )
  const pointReviewRenderMetrics = useMemo(
    () => getPointReviewRenderMetrics(),
    [imageDims, pointReviewStageSize, pointReviewZoom, pointReviewOffset],
  )
  const diameterResultLines = useMemo(() => {
    return (Array.isArray(diamResults) ? diamResults : [])
      .map((r, idx) => {
        const ok = String(r?.status || '') === 'ok'
        const left = Array.isArray(r?.left_edge_xy) ? r.left_edge_xy : (Array.isArray(r?.debug_left_edge_xy) ? r.debug_left_edge_xy : null)
        const right = Array.isArray(r?.right_edge_xy) ? r.right_edge_xy : (Array.isArray(r?.debug_right_edge_xy) ? r.debug_right_edge_xy : null)
        if (!left || !right) return null
        const x1 = Number(left[0])
        const y1 = Number(left[1])
        const x2 = Number(right[0])
        const y2 = Number(right[1])
        if (![x1, y1, x2, y2].every(Number.isFinite)) return null
        return {
          key: `${idx}-${x1}-${y1}-${x2}-${y2}`,
          x1,
          y1,
          x2,
          y2,
          ok,
        }
      })
      .filter(Boolean)
  }, [diamResults])
  const diameterResultQuads = useMemo(() => {
    return (Array.isArray(diamResults) ? diamResults : [])
      .map((r, idx) => {
        const vertices = Array.isArray(r?.square_vertices_xy) ? r.square_vertices_xy : []
        if (vertices.length !== 4) return null
        const points = vertices
          .map((xy) => `${Number(xy?.[0])},${Number(xy?.[1])}`)
          .join(' ')
        if (!points || points.includes('NaN')) return null
        return { key: `quad-${idx}-${points}`, points, ok: String(r?.status || '') === 'ok' }
      })
      .filter(Boolean)
  }, [diamResults])
  const diameterManualCirclePreviews = useMemo(() => {
    if (loading.diamRun) return []
    const list = (Array.isArray(manualCircles) ? manualCircles : [])
      .filter((circle) => !circle?.consumed)
      .filter((circle) => circle?.center && Number.isFinite(Number(circle.radius)))
      .map((circle, idx) => ({ ...circle, idx, radius: Math.max(1, Number(circle.radius) || 1) }))
    if (
      manualCircleDraft?.center &&
      Number.isFinite(Number(manualCircleDraft.radius)) &&
      !list.some((circle) => String(circle.geometry_id || '') === String(manualCircleDraft.geometry_id || ''))
    ) {
      list.push({ ...manualCircleDraft, idx: -1, radius: Math.max(1, Number(manualCircleDraft.radius) || 1) })
    }
    return list
  }, [loading.diamRun, manualCircles, manualCircleDraft])
  const activeLocoCircle = useMemo(() => {
    if (locoCircleDraft?.center) return locoCircleDraft
    const point = activeLocoPoint()
    if (!point || !Number.isFinite(Number(point.seed_radius_px))) return null
    return { center: { x: Number(point.x), y: Number(point.y) }, radius: Number(point.seed_radius_px) }
  }, [locoCircleDraft, locoPoints, locoActivePointIdx])
  const locoCandidate = useMemo(() => {
    const list = Array.isArray(locoPreview?.radius_candidates) ? locoPreview.radius_candidates : []
    if (!list.length) return null
    const idx = locoCandidateIndex >= 0 ? locoCandidateIndex : Number(locoPreview?.best_candidate_index ?? -1)
    return list[clamp(idx, 0, list.length - 1)] || null
  }, [locoPreview, locoCandidateIndex])
  const locoPreviewResult = locoPreview?.result || {}
  const locoRecenteredPoint = useMemo(() => {
    const xy = locoPreviewResult?.recentered_xy
    if (!Array.isArray(xy) || xy.length < 2) return null
    const x = Number(xy[0])
    const y = Number(xy[1])
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null
    return { x, y }
  }, [locoPreviewResult])
  const locoResultLine = useMemo(() => {
    const left = locoPreviewResult?.left_edge_xy
    const right = locoPreviewResult?.right_edge_xy
    if (!Array.isArray(left) || !Array.isArray(right) || left.length < 2 || right.length < 2) return null
    const x1 = Number(left[0])
    const y1 = Number(left[1])
    const x2 = Number(right[0])
    const y2 = Number(right[1])
    if (![x1, y1, x2, y2].every(Number.isFinite)) return null
    return { x1, y1, x2, y2 }
  }, [locoPreviewResult])
  const locoIntersections = useMemo(() => {
    const fromCandidate = Array.isArray(locoCandidate?.intersections_xy)
      ? locoCandidate.intersections_xy
      : (Array.isArray(locoCandidate?.intersections) ? locoCandidate.intersections.map((item) => item?.xy) : [])
    const fromResult = Array.isArray(locoPreviewResult?.loco_intersections_xy) ? locoPreviewResult.loco_intersections_xy : []
    return (fromCandidate.length ? fromCandidate : fromResult)
      .map((xy) => ({ x: Number(xy?.[0]), y: Number(xy?.[1]) }))
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
  }, [locoCandidate, locoPreviewResult])
  const locoLabActiveProposals = useMemo(
    () => (locoFilteredProposals.length ? locoFilteredProposals : locoProposals),
    [locoFilteredProposals, locoProposals],
  )
  const locoLabSelectedProposal = useMemo(() => {
    if (!locoLabActiveProposals.length) return null
    return locoLabActiveProposals.find((p) => String(p.proposal_id || '') === String(locoSelectedProposalId || '')) || locoLabActiveProposals[0]
  }, [locoLabActiveProposals, locoSelectedProposalId])
  const locoLabMeasurementsByProposal = useMemo(() => {
    const out = {}
    ;(Array.isArray(locoMeasuredResults) ? locoMeasuredResults : []).forEach((m) => {
      const pid = String(m?.proposal_id || '')
      if (pid) out[pid] = m
    })
    return out
  }, [locoMeasuredResults])
  const locoLabVisibleProposals = useMemo(() => {
    const list = Array.isArray(locoLabActiveProposals) ? locoLabActiveProposals : []
    return list
      .filter((p) => locoLayerVisibility.rejected || String(p?.status || '') === 'accepted' || String(p?.proposal_id || '') === String(locoLabSelectedProposal?.proposal_id || ''))
      .slice(0, 650)
  }, [locoLabActiveProposals, locoLayerVisibility.rejected, locoLabSelectedProposal])
  const locoLabComponentBoxes = useMemo(() => {
    const seen = new Set()
    const out = []
    ;(Array.isArray(locoLabActiveProposals) ? locoLabActiveProposals : []).forEach((p) => {
      const bbox = Array.isArray(p?.component_bbox) ? p.component_bbox : null
      if (!bbox || bbox.length < 4) return
      const key = `${p.component_id || ''}-${bbox.join(',')}`
      if (seen.has(key)) return
      seen.add(key)
      out.push({ key, x: Number(bbox[0]), y: Number(bbox[1]), w: Number(bbox[2]), h: Number(bbox[3]) })
    })
    return out
  }, [locoLabActiveProposals])
  const locoLabOkMeasurements = useMemo(
    () => (Array.isArray(locoMeasuredResults) ? locoMeasuredResults : []).filter((m) => String(m?.status || '') === 'ok'),
    [locoMeasuredResults],
  )
  const locoDatasetSelectedCircle = useMemo(
    () => locoDatasetCircles.find((c) => String(c.candidate_id) === String(locoDatasetSelectedId)) || null,
    [locoDatasetCircles, locoDatasetSelectedId],
  )
  const locoDatasetCounts = useMemo(() => {
    const valid = locoDatasetCircles.filter((c) => c.label === 'valid').length
    const invalidCrossing = locoDatasetCircles.filter((c) => c.label === 'invalid_crossing').length
    const invalidOther = locoDatasetCircles.filter((c) => c.label === 'invalid_other').length
    return { total: locoDatasetCircles.length, valid, invalidCrossing, invalidOther }
  }, [locoDatasetCircles])
  const locoAugFilteredItems = useMemo(() => {
    if (locoAugLabelFilter === 'valid' || locoAugLabelFilter === 'invalid' || locoAugLabelFilter === 'invalid_crossing' || locoAugLabelFilter === 'invalid_other') {
      const normFilter = locoAugLabelFilter === 'invalid' ? 'invalid_other' : locoAugLabelFilter
      return locoAugItems.filter((item) => item.label === normFilter)
    }
    return locoAugItems
  }, [locoAugItems, locoAugLabelFilter])
  const locoAugSelectedCount = useMemo(
    () => Object.values(locoAugSelected || {}).filter(Boolean).length,
    [locoAugSelected],
  )
  const locoAugEstimatedVariants = useMemo(() => {
    return Math.max(1, Number(locoAugPasses) || 1)
  }, [locoAugPasses])
  const locoTrainingMetrics = useMemo(
    () => Array.isArray(locoTrainingResult?.metrics_summary) ? locoTrainingResult.metrics_summary : [],
    [locoTrainingResult],
  )
  const locoTrainingOkMetrics = useMemo(
    () => locoTrainingMetrics.filter((row) => row.status === 'ok'),
    [locoTrainingMetrics],
  )
  const locoTrainingBest = useMemo(() => {
    const best = (field) => {
      const rows = locoTrainingOkMetrics.filter((row) => row[field] != null)
      if (!rows.length) return ''
      return rows.slice().sort((a, b) => Number(b[field] || 0) - Number(a[field] || 0))[0]?.model_id || ''
    }
    return {
      precision_valid: best('precision_valid'),
      f1_valid: best('f1_valid'),
      pr_auc: best('pr_auc'),
    }
  }, [locoTrainingOkMetrics])
  const locoTrainingThresholdRows = useMemo(
    () => (Array.isArray(locoTrainingResult?.threshold_metrics) ? locoTrainingResult.threshold_metrics : []).filter((row) => row.model_id === locoTrainingSelectedModel),
    [locoTrainingResult, locoTrainingSelectedModel],
  )
  const locoTrainingRadiusRows = useMemo(
    () => (Array.isArray(locoTrainingResult?.radius_group_metrics) ? locoTrainingResult.radius_group_metrics : []),
    [locoTrainingResult],
  )
  const locoTrainingErrors = useMemo(
    () => (Array.isArray(locoTrainingResult?.error_review) ? locoTrainingResult.error_review : [])
      .filter((row) => row.model_id === locoTrainingSelectedModel && row.error_type === locoTrainingErrorType && Number(row.probability_valid) >= Number(locoTrainingThreshold || 0)),
    [locoTrainingResult, locoTrainingSelectedModel, locoTrainingErrorType, locoTrainingThreshold],
  )
  const locoTrainingMcErrors = useMemo(
    () => (Array.isArray(locoTrainingResult?.error_review_multiclass) ? locoTrainingResult.error_review_multiclass : [])
      .filter((row) => {
        if (locoTrainingMcErrorClass !== 'all' && row.label_real_name !== locoTrainingMcErrorClass && row.prediction_name !== locoTrainingMcErrorClass) return false
        if (locoTrainingMcErrorType !== 'all' && row.error_type !== locoTrainingMcErrorType) return false
        return true
      }),
    [locoTrainingResult, locoTrainingMcErrorClass, locoTrainingMcErrorType],
  )
  const locoTrainingCombErrors = useMemo(
    () => (Array.isArray(locoTrainingResult?.error_review_combined) ? locoTrainingResult.error_review_combined : [])
      .filter((row) => {
        if (locoTrainingCombErrorClass !== 'all' && row.label_real_multiclass_name !== locoTrainingCombErrorClass) return false
        if (locoTrainingCombErrorType !== 'all' && row.error_type !== locoTrainingCombErrorType) return false
        if (locoTrainingCombErrorSubtype !== 'all' && row.error_subtype !== locoTrainingCombErrorSubtype) return false
        return true
      }),
    [locoTrainingResult, locoTrainingCombErrorClass, locoTrainingCombErrorType, locoTrainingCombErrorSubtype],
  )
  const locoTestSelectedCircle = useMemo(
    () => locoTestCircles.find((c) => String(c.candidate_id) === String(locoTestSelectedId)) || null,
    [locoTestCircles, locoTestSelectedId],
  )
  const locoTestCounts = useMemo(() => {
    const valid = locoTestCircles.filter((c) => c.label === 'valid').length
    const invalidCrossing = locoTestCircles.filter((c) => c.label === 'invalid_crossing').length
    const invalidOther = locoTestCircles.filter((c) => c.label === 'invalid_other' || c.label === 'invalid').length
    return { total: locoTestCircles.length, valid, invalidCrossing, invalidOther }
  }, [locoTestCircles])
  const locoTestPredById = useMemo(() => {
    const out = {}
    ;(locoTestResult?.predictions || []).forEach((p) => { out[p.candidate_id] = p })
    return out
  }, [locoTestResult])
  const locoTestErrors = useMemo(
    () => (locoTestResult?.predictions || []).filter((p) => !p.correct),
    [locoTestResult],
  )
  const locoModelAccepted = useMemo(
    () => Array.isArray(locoModelResult?.accepted) ? locoModelResult.accepted : [],
    [locoModelResult],
  )
  const locoModelRejected = useMemo(
    () => Array.isArray(locoModelResult?.rejected) ? locoModelResult.rejected : [],
    [locoModelResult],
  )
  const locoModelSelectedCandidate = useMemo(() => {
    const rows = [...locoModelAccepted, ...locoModelRejected]
    return rows.find((c) => String(c.candidate_id) === String(locoModelSelectedId)) || null
  }, [locoModelAccepted, locoModelRejected, locoModelSelectedId])
  const locoModelMeasureByProposal = useMemo(() => {
    const out = {}
    ;(locoModelMeasurement?.results || []).forEach((row) => {
      const id = String(row.proposal_id || '')
      if (id) out[id] = row
    })
    return out
  }, [locoModelMeasurement])
  const batchProgressPct = useMemo(() => {
    if (!batchProgress.total) return 0
    return Math.round((100 * batchProgress.done) / batchProgress.total)
  }, [batchProgress])

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>Scribble Research</h1>
          <p>Comparacion secuencial rapida A-E con descarte OK/BAD</p>
        </div>
        <div className="session">Sesion: <strong>{sessionId || '-'}</strong></div>
      </header>

      <div className={`notice ${notice.level}`}>
        <strong>{notice.title}</strong>
        <span>{notice.text}</span>
      </div>

      <Navigation
        activeGroup={activeGroup}
        activeTab={activeTab}
        onGroupChange={handleGroupChange}
        onTabChange={handleTabChange}
      />

      <div className="layout">
        <aside className="left">
          {workspaceTab === 'workbench' ? (
            <>
              <div className="side-tabs">
                <button className={workbenchPanelTab === 'image' ? 'active' : ''} onClick={() => setWorkbenchPanelTab('image')}>Imagen</button>
                <button className={workbenchPanelTab === 'editor' ? 'active' : ''} onClick={() => setWorkbenchPanelTab('editor')}>Scribble</button>
                <button className={workbenchPanelTab === 'experiments' ? 'active' : ''} onClick={() => setWorkbenchPanelTab('experiments')}>Experimentos</button>
              </div>
              {workbenchPanelTab === 'image' ? (
                <>
              <section className="card">
                <h2>Imagen activa</h2>
                <input type="file" accept="image/*,.tif,.tiff" onChange={onLoadImage} disabled={loading.loadImage || loading.boot} />
                <label className="field">
                  <span>ruta inicial</span>
                  <input
                    value={imageStartDir}
                    onChange={(e) => setImageStartDir(e.target.value)}
                    placeholder="C:\\ruta\\a\\carpeta\\de\\imagenes"
                    disabled={loading.localPrefs}
                  />
                </label>
                <div className="inline">
                  <button onClick={saveLocalImagePrefs} disabled={loading.localPrefs || !imageStartDir.trim()}>Guardar ruta</button>
                  <button onClick={listLocalImages} disabled={loading.localImages || !imageStartDir.trim()}>Listar imagenes</button>
                  <button onClick={() => openFolder('custom', imageStartDir)} disabled={loading.openFolder || !imageStartDir.trim()}>Abrir ruta</button>
                </div>
                {localImageFiles.length ? (
                  <label className="field">
                    <span>imagenes desde ruta guardada</span>
                    <select value={selectedLocalPath} onChange={(e) => setSelectedLocalPath(e.target.value)} disabled={loading.localLoad}>
                      {localImageFiles.map((item) => (
                        <option key={item.path} value={item.path}>{item.relative_path || item.name}</option>
                      ))}
                    </select>
                  </label>
                ) : null}
                {localImageFiles.length ? (
                  <button onClick={() => loadLocalImage()} disabled={!selectedLocalPath || loading.localLoad}>Cargar seleccion local</button>
                ) : null}
                <label className="field">
                  <span>GT opcional (binaria)</span>
                  <input id="gt-file-input" type="file" accept="image/*,.png,.tif,.tiff" />
                </label>
                <div className="kpi">ID: <strong>{imageId || '-'}</strong></div>
                <div className="kpi">Archivo: <strong>{imageName || '-'}</strong></div>
                <div className="kpi">Exclusion: <strong>{excludeRect ? `${excludeRect.w}x${excludeRect.h}px` : 'sin rectangulo'}</strong></div>
                {draftInfo ? <div className="kpi">Draft: <strong>{draftInfo}</strong></div> : null}
                <div className="inline">
                  <button onClick={saveScribblesNow} disabled={!imageId || loading.libraryList}>Guardar scribbles</button>
                  <button onClick={() => refreshSavedImages()} disabled={loading.libraryList}>Refrescar guardadas</button>
                </div>
                <button onClick={refreshResults} disabled={!imageId || loading.listResults}>Refrescar resultados</button>
              </section>

              <section className="card">
                <h2>Imagenes guardadas</h2>
                <div className="inline">
                  <button onClick={() => openFolder('outputs')} disabled={loading.openFolder}>Abrir carpeta experimentos</button>
                  <button onClick={() => openFolder('library')} disabled={loading.openFolder}>Abrir biblioteca</button>
                </div>
                <button className="bad" onClick={deleteSelectedSavedImage} disabled={!selectedSavedImageId || loading.libraryDelete}>Eliminar seleccionada</button>
                {!savedImages.length ? (
                  <div className="placeholder small">Sin imagenes guardadas.</div>
                ) : (
                  <div className="saved-image-list">
                    {savedImages.map((img) => (
                      <button
                        key={img.image_id}
                        className={`saved-image-row ${img.image_id === selectedSavedImageId || img.image_id === imageId ? 'selected' : ''}`}
                        onClick={() => loadSavedImage(img.image_id)}
                        disabled={loading.libraryLoad}
                      >
                        {img.thumbnail_b64 ? (
                          <img src={b64ToDataUrl(img.thumbnail_b64, img.thumbnail_mime || 'image/png')} alt={img.image_name || img.image_id} />
                        ) : <span className="saved-image-empty" />}
                        <span>
                          <strong>{img.image_name || img.image_id}</strong>
                          <em>{img.has_scribble_draft ? `scribbles F${img.draft_n_fg || 0} H${img.draft_n_halo || 0} B${img.draft_n_bg || 0}` : 'sin scribbles'}</em>
                          <em>{img.has_prior_cache ? `prior ${img.latest_prior_experiment_id || img.latest_prior_run_id || 'ok'}` : 'sin prior'}</em>
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </section>
                </>
              ) : workbenchPanelTab === 'editor' ? (
                <section className="card scribble-controls-card">
                  <h2>Editor de scribbles</h2>
                  <div className="tool-row">
                    <button className={`icon-tool tool-square ${viewerMode === 'mark' ? 'toggle-active' : ''}`} onClick={() => setViewerMode('mark')} disabled={!imageUrl} title="Lapiz (L)"><ToolIcon name="pencil" /><kbd>L</kbd></button>
                    <button className={`icon-tool tool-square ${viewerMode === 'pan' ? 'toggle-active' : ''}`} onClick={() => setViewerMode('pan')} disabled={!imageUrl} title="Mano (M)"><ToolIcon name="hand" /><kbd>M</kbd></button>
                    <button className={`icon-tool tool-square ${viewerMode === 'exclude' ? 'toggle-active' : ''}`} onClick={() => setViewerMode('exclude')} disabled={!imageUrl} title="Rectangulo de exclusion (R)"><ToolIcon name="exclude" /><kbd>R</kbd></button>
                  </div>
                  <div className="paint-grid compact">
                    <button className={`paint-tool ${tool === 'fiber' ? 'toggle-active' : ''}`} onClick={() => { setViewerMode('mark'); setTool('fiber') }} disabled={!imageUrl}><span className="color-dot fiber" />Fibra <kbd>F</kbd></button>
                    <button className={`paint-tool ${tool === 'halo' ? 'toggle-active' : ''}`} onClick={() => { setViewerMode('mark'); setTool('halo') }} disabled={!imageUrl}><span className="color-dot halo" />Halo <kbd>H</kbd></button>
                    <button className={`paint-tool ${tool === 'bg' ? 'toggle-active' : ''}`} onClick={() => { setViewerMode('mark'); setTool('bg') }} disabled={!imageUrl}><span className="color-dot bg" />Background <kbd>B</kbd></button>
                    <button className={`paint-tool ${tool === 'erase' ? 'toggle-active' : ''}`} onClick={() => { setViewerMode('mark'); setTool('erase') }} disabled={!imageUrl}><ToolIcon name="erase" />Goma <kbd>G</kbd></button>
                  </div>
                  <div className="brush-panel">
                    <label className="brush-toolbar-control wide">
                      <span>Pincel</span>
                      <input type="range" min="1" max="120" value={brushSliderValue} onChange={(e) => setBrushSize(brushSliderToPx(e.target.value))} disabled={!imageUrl} />
                      <strong>{annotBrushPx}px</strong>
                    </label>
                    <button className={`icon-tool ${brushAutoScale ? 'toggle-active' : ''}`} onClick={() => setBrushAutoScale((v) => !v)} disabled={!imageUrl} title="Escalado automatico por resolucion">Auto</button>
                  </div>
                  <div className="shortcut-note">Ctrl + rueda hace zoom en la imagen. Alt + rueda cambia el pincel.</div>
                  <div className="tool-row">
                    <button className="icon-tool" onClick={onAnnotUndo} disabled={!annotHistory.length || !imageUrl} title="Deshacer (Ctrl+Z)">Undo</button>
                    <button className="icon-tool" onClick={onAnnotRedo} disabled={!annotFuture.length || !imageUrl} title="Rehacer (Ctrl+Y)">Redo</button>
                    <button className="icon-tool" onClick={() => zoomEditorBy(0.84)} disabled={!imageUrl} title="Zoom menos">-</button>
                    <button className="icon-tool" onClick={() => zoomEditorBy(1.2)} disabled={!imageUrl} title="Zoom mas">+</button>
                    <button className="icon-tool" onClick={resetEditorView} disabled={!imageUrl} title="Reset vista">Reset</button>
                    <span className="zoom-chip">{Math.round(viewerZoom * 100)}%</span>
                  </div>
                  <div className="inline">
                    <button onClick={onClearScribbles} disabled={!imageUrl}>Limpiar</button>
                    <button onClick={clearExcludeRect} disabled={!imageUrl || !excludeRect}>Limpiar exclusion</button>
                  </div>
                  <div className="model-assist-panel compact-model-panel">
                    <label className="field">
                      <span>modelo</span>
                      <select value={selectedAssistModelId} onChange={(e) => setSelectedAssistModelId(e.target.value)} disabled={!assistModels.length}>
                        <option value="">sin modelo</option>
                        {assistModels.map((model) => (
                          <option key={model.model_id} value={model.model_id}>
                            {model.model_name || model.model_id}{model.model_id === defaultAssistModelId ? ' (default)' : ''}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="brush-toolbar-control wide">
                      <span>confianza</span>
                      <input type="range" min="0.3" max="0.95" step="0.01" value={modelMinConfidence} onChange={(e) => setModelMinConfidence(Number(e.target.value || 0.72))} disabled={!imageUrl} />
                      <strong>{Number(modelMinConfidence).toFixed(2)}</strong>
                    </label>
                    <div className="model-class-row">
                      <label><input type="checkbox" checked={modelIncludeFiber} onChange={(e) => setModelIncludeFiber(e.target.checked)} disabled={!imageUrl} /> fibra</label>
                      <label><input type="checkbox" checked={modelIncludeHalo} onChange={(e) => setModelIncludeHalo(e.target.checked)} disabled={!imageUrl} /> halo</label>
                      <label><input type="checkbox" checked={modelIncludeBackground} onChange={(e) => setModelIncludeBackground(e.target.checked)} disabled={!imageUrl} /> background</label>
                    </div>
                    <div className="inline">
                      <button className="primary" onClick={predictWithAssistModel} disabled={!imageUrl || !selectedAssistModelId || loading.modelsPredict}>Predecir</button>
                      <button onClick={applyModelPredictionAsScribbles} disabled={!modelPrediction?.suggestion_b64}>Aplicar</button>
                    </div>
                    {modelPrediction ? (
                      <div className="model-prediction-strip">
                        <strong>{modelPrediction.model_name || modelPrediction.model_id}</strong>
                        <span>F{modelPrediction.counts?.fiber || 0} H{modelPrediction.counts?.halo || 0} B{modelPrediction.counts?.background || 0}</span>
                      </div>
                    ) : null}
                  </div>
                </section>
              ) : (

              <section className="card">
                <h2>Experimentos A-E</h2>
                <label className="field">
                  <span>Seleccion principal</span>
                  <select value={selectedExperiment} onChange={(e) => setSelectedExperiment(e.target.value)}>
                    <option value="">-- seleccionar --</option>
                    {experiments.map((x) => (
                      <option key={x.experiment_id} value={x.experiment_id}>
                        {x.group} | {x.experiment_id} | {x.implementation_status}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span>Guardado de resultados</span>
                  <select value={segSaveMode} onChange={(e) => setSegSaveMode(e.target.value)}>
                    <option value="overwrite">sobrescribir mismo experimento/perfil</option>
                    <option value="append">acumular nuevos runs</option>
                  </select>
                </label>
                <div className="inline">
                  <button className="primary" onClick={runOne} disabled={!selectedExperiment || loading.run}>Run selected</button>
                  <button className="primary" onClick={runBatch} disabled={loading.runBatch}>Run batch</button>
                </div>
                {loading.runBatch ? (
                  <div className="batch-progress-box">
                    <div className="batch-progress-head">
                      <strong>Ejecutando batch</strong>
                      <span>{batchProgress.done}/{batchProgress.total}</span>
                    </div>
                    <div className="batch-progress-track">
                      <div className="batch-progress-fill" style={{ width: `${batchProgressPct}%` }} />
                    </div>
                    <div className="batch-progress-meta">
                      <span>{batchProgressPct}%</span>
                      <span>{batchProgress.current ? `Actual: ${batchProgress.current}` : 'Preparando...'}</span>
                    </div>
                  </div>
                ) : null}
                <p className="small">Batch usa solo perfil <strong>high</strong>.</p>
                <div className="inline">
                  <button onClick={selectAllBatch} disabled={!experiments.length}>Select all</button>
                  <button onClick={clearBatchSelection} disabled={!Object.values(selectedBatch).some(Boolean)}>Select none</button>
                </div>
                <div className="batch-list">
                  {experiments.map((x) => (
                    <label key={x.experiment_id} className="batch-item">
                      <input
                        type="checkbox"
                        checked={!!selectedBatch[x.experiment_id]}
                        onChange={(e) => setSelectedBatch((prev) => ({ ...prev, [x.experiment_id]: e.target.checked }))}
                      />
                      <span>{x.group} - {x.experiment_id}</span>
                    </label>
                  ))}
                </div>
              </section>
              )}

            </>
          ) : workspaceTab === 'review' ? (
            <>
              <section className="card">
                <h2>Revision de resultados</h2>
                <div className="kpi">ID imagen: <strong>{imageId || '-'}</strong></div>
                <div className="kpi">Archivo: <strong>{imageName || '-'}</strong></div>
                <div className="inline">
                  <button onClick={refreshResults} disabled={!imageId || loading.listResults}>Refrescar resultados</button>
                  <button className="bad" onClick={clearReviewResults} disabled={!imageId || loading.clearResults}>Borrar todo</button>
                </div>
              </section>

              <section className="card">
                <h2>Visor secuencial</h2>
                <div className="inline">
                  <button onClick={goPrev} disabled={!filteredRuns.length}>{'<-'} Prev</button>
                  <button onClick={goNext} disabled={!filteredRuns.length}>Next {'->'}</button>
                </div>

                <label className="field">
                  <span>Filtro grupo</span>
                  <select ref={filterFocusRef} value={filterGroup} onChange={(e) => setFilterGroup(e.target.value)}>
                    <option value="all">all</option>
                    {groups.map((g) => <option key={g} value={g}>{g}</option>)}
                  </select>
                </label>
                <label className="field">
                  <span>Filtro experimento</span>
                  <select value={filterExperiment} onChange={(e) => setFilterExperiment(e.target.value)}>
                    <option value="all">all</option>
                    {experiments.map((x) => <option key={x.experiment_id} value={x.experiment_id}>{x.experiment_id}</option>)}
                  </select>
                </label>
                <label className="field">
                  <span>Filtro decision</span>
                  <select value={filterDecision} onChange={(e) => setFilterDecision(e.target.value)}>
                    <option value="all">all</option>
                    {REVIEW_TIERS.map((tier) => <option key={tier.value} value={tier.value}>{tier.label}</option>)}
                    <option value="unreviewed">unreviewed</option>
                  </select>
                </label>
                <label className="field">
                  <span>Orden</span>
                  <select value={reviewSort} onChange={(e) => setReviewSort(e.target.value)}>
                    <option value="latest">recientes</option>
                    <option value="tier">tier mejor a peor</option>
                  </select>
                </label>

                <div className="kpi">Run activo: <strong>{activeRunId || '-'}</strong></div>
                <div className="kpi">Indice filtrado: <strong>{filteredRuns.length ? `${Math.max(1, activeIndex + 1)} / ${filteredRuns.length}` : '-'}</strong></div>
                <div className="kpi">Total runs: <strong>{runs.length}</strong></div>
                <div className="kpi">Decision actual: <strong>{latestDecision || '-'}</strong></div>

                <label className="field">
                  <span>Salto rapido</span>
                  <select value={activeRunId} onChange={(e) => setActiveRunId(e.target.value)}>
                    <option value="">-- seleccionar --</option>
                    {filteredRuns.map((r, idx) => (
                      <option key={r.run_id} value={r.run_id}>
                        {idx + 1}. {r.experiment_id}{r.profile_name ? `(${r.profile_name})` : ''} | {r.run_status_level}
                        {reviewByRun[r.run_id]?.decision ? ` | ${reviewLabel(reviewByRun[r.run_id].decision)}` : ''}
                      </option>
                    ))}
                  </select>
                </label>

                <textarea value={reviewNote} onChange={(e) => setReviewNote(e.target.value)} placeholder="Nota opcional" rows={3} />
                <div className="tier-buttons">
                  {REVIEW_TIERS.map((tier) => (
                    <button
                      key={tier.value}
                      className={`tier tier-${tier.value}`}
                      onClick={() => mark(tier.value)}
                      disabled={!activeRunId || loading.mark}
                    >
                      {tier.short}
                    </button>
                  ))}
                </div>
                <button onClick={exportReport} disabled={!imageId || loading.export}>Exportar CSV+JSON+Galeria</button>
                {reportInfo ? <p className="small">Reporte: {reportInfo.report_dir}</p> : null}
                <p className="small">Atajos: &larr;/&rarr; navegar, S/A/B/C/U=tier, R=reset filtros, F=focus filtro grupo.</p>
              </section>
            </>
          ) : workspaceTab === 'diameter' ? (
            <>
              <section className="card">
                <h2>Diameter Research</h2>
                <div className="kpi">ID imagen: <strong>{imageId || '-'}</strong></div>
                <div className="side-tabs compact-diam-tabs">
                  <button
                    className={diamMethodPanel === 'automatic' ? 'active' : ''}
                    onClick={() => {
                      setDiamMethodPanel('automatic')
                      if (!['hybrid_profile_diameter_v3_2_auto', 'hybrid_profile_diameter_v3_2_small_mask', 'hybrid_profile_diameter_v3_2_large_image', 'hybrid_profile_diameter_v3_1', 'hybrid_profile_diameter_v2', 'hybrid_profile_diameter_v1'].includes(diamMethodId)) {
                        setDiamMethodId('hybrid_profile_diameter_v3_2_auto')
                      }
                    }}
                  >
                    Automatico
                  </button>
                  <button
                    className={diamMethodPanel === 'manual' ? 'active' : ''}
                    onClick={() => {
                      setDiamMethodPanel('manual')
                      if (!['circle_square_mask_diameter', 'manual_dual_side_caliper', 'manual_line_direct_caliper', 'ellipse_oriented_fit'].includes(diamMethodId)) {
                        setDiamMethodId('circle_square_mask_diameter')
                        setDiamViewerMode('circle')
                      } else if (diamMethodId === 'circle_square_mask_diameter') {
                        setDiamViewerMode('circle')
                      } else {
                        setDiamViewerMode('manual')
                      }
                    }}
                  >
                    Manual geometrico
                  </button>
                </div>
                <label className="field">
                  <span>Metodo</span>
                  <select
                    value={diamMethodId}
                    onChange={(e) => {
                      const nextMethod = e.target.value
                      setDiamMethodId(nextMethod)
                      if (nextMethod === 'circle_square_mask_diameter') {
                        setDiamViewerMode('circle')
                        updateDiamRawParam('circle_square_seed_mode', 'manual_circle')
                      } else if (['manual_dual_side_caliper', 'manual_line_direct_caliper'].includes(nextMethod)) {
                        setDiamViewerMode('manual')
                      }
                    }}
                    disabled={!imageId}
                  >
                    {diamMethodPanel === 'automatic' ? (
                      <>
                        <option value="hybrid_profile_diameter_v3_2_auto">Auto small/large</option>
                        <option value="hybrid_profile_diameter_v3_2_small_mask">Small mask-driven</option>
                        <option value="hybrid_profile_diameter_v3_2_large_image">Large image-driven</option>
                        <option value="hybrid_profile_diameter_v3_1">v3.1 baseline local</option>
                        <option value="hybrid_profile_diameter_v2">v2 robust</option>
                        <option value="hybrid_profile_diameter_v1">v1 baseline</option>
                      </>
                    ) : (
                      <>
                        <option value="circle_square_mask_diameter">Circle-square mask</option>
                        <option value="manual_dual_side_caliper">Linea mask</option>
                        <option value="manual_line_direct_caliper">Linea manual</option>
                        <option value="ellipse_oriented_fit">Ellipse oriented fit</option>
                      </>
                    )}
                  </select>
                </label>
                <p className="small">
                  `Auto` decide entre ruta image-driven y mask-driven. `Small` mide dentro de la mascara local. `Circle-square` queda bloqueado a circulo manual. `Linea mask` corta tu linea con la mascara; `Linea manual` usa exactamente el trazo dibujado.
                </p>
                <label className="field">
                  <span>Fuente de soporte</span>
                  <input value="prior_mask" disabled readOnly />
                </label>
                <p className="small">
                  Diameter usa siempre la mascara binaria del run de soporte seleccionado. La mascara verde mostrada en el visor corresponde a esa fuente.
                </p>
                <label className="field">
                  <span>Run de soporte</span>
                  <select value={diamPriorRunId} onChange={(e) => setDiamPriorRunId(e.target.value)} disabled={!imageId}>
                    <option value="latest">ultimo run disponible</option>
                    {runs.map((r, idx) => (
                      <option key={r.run_id} value={r.run_id}>
                        {idx + 1}. {r.experiment_id}{r.profile_name ? ` (${r.profile_name})` : ''} | {r.run_status_level} | {r.created_at || r.run_id}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="kpi">
                  Soporte elegido: <strong>{selectedPriorRun ? `${selectedPriorRun.experiment_id}${selectedPriorRun.profile_name ? ` (${selectedPriorRun.profile_name})` : ''}` : 'sin runs'}</strong>
                </div>
                <div className="inline">
                  <button onClick={() => refreshResults()} disabled={!imageId || loading.listResults}>Refrescar modalidades</button>
                  <button onClick={() => setDiamPriorRunId('latest')} disabled={!imageId || diamPriorRunId === 'latest'}>Usar ultimo</button>
                </div>
                <div className="inline">
                  <button className="primary" onClick={() => runDiameterResearch(true)} disabled={!imageId || !diamPoints.length || loading.diamRun}>Run active point</button>
                  <button className="primary" onClick={() => runDiameterResearch(false)} disabled={!imageId || !diamPoints.length || loading.diamRun}>Run all points</button>
                </div>
                <div className="inline">
                  <button onClick={() => refreshDiameterRuns()} disabled={!imageId || loading.diamList}>Refresh runs</button>
                  <button onClick={exportDiameterReport} disabled={!imageId || loading.diamExport}>Export report</button>
                </div>
                {diamReportInfo ? <p className="small">Reporte: {diamReportInfo.report_dir}</p> : null}
              </section>

              <section className="card">
                <h2>Puntos</h2>
                <div className="inline">
                  <button onClick={() => updateDiameterPoints('remove_active')} disabled={diamActivePointIdx < 0 || !diamPoints.length || loading.diamPoints}>Borrar punto</button>
                  <button
                    onClick={clearDiameterPanel}
                    disabled={!imageId || loading.diamPoints}
                    title="Borra puntos, overlay, soporte, resultados visibles y marcas manuales del panel. No borra runs ni scribbles guardados."
                  >
                    Limpiar
                  </button>
                </div>
                <button onClick={saveDiameterPoints} disabled={!imageId || loading.diamPoints}>Guardar puntos</button>
                {!diamPoints.length ? (
                  <div className="placeholder small">Sin puntos.</div>
                ) : (
                  <div className="diam-point-list">
                    {diamPoints.map((p, idx) => (
                      <button
                        key={`${idx}-${p.x}-${p.y}`}
                        className={`diam-point-row ${idx === diamActivePointIdx ? 'selected' : ''}`}
                        onClick={() => updateDiameterPoints('set_active', { active_index: idx })}
                        disabled={loading.diamPoints}
                      >
                        <span>{idx + 1}</span>
                        <strong>{Number(p.x).toFixed(1)}, {Number(p.y).toFixed(1)}</strong>
                      </button>
                    ))}
                  </div>
                )}
              </section>

              <section className="card">
                <h2>Parametros</h2>
                <div className="diam-param-grid">
                  <label className="field">
                    <span>support high</span>
                    <input type="number" step="0.05" min="0" max="1" value={diamParams.support_high_threshold} onChange={(e) => updateDiamParam('support_high_threshold', e.target.value)} />
                  </label>
                  <label className="field">
                    <span>support low</span>
                    <input type="number" step="0.05" min="0" max="1" value={diamParams.support_low_threshold} onChange={(e) => updateDiamParam('support_low_threshold', e.target.value)} />
                  </label>
                  <label className="field">
                    <span>dilation px</span>
                    <input type="number" step="1" min="0" value={diamParams.support_dilation_px} onChange={(e) => updateDiamParam('support_dilation_px', e.target.value, true)} />
                  </label>
                  <label className="field">
                    <span>window px</span>
                    <input type="number" step="2" min="9" value={diamParams.local_window_px} onChange={(e) => updateDiamParam('local_window_px', e.target.value, true)} />
                  </label>
                  <label className="field">
                    <span>profile length</span>
                    <input type="number" step="2" min="8" value={diamParams.profile_length_px} onChange={(e) => updateDiamParam('profile_length_px', e.target.value)} />
                  </label>
                  <label className="field">
                    <span>profile count</span>
                    <input type="number" step="1" min="1" value={diamParams.profile_count} onChange={(e) => updateDiamParam('profile_count', e.target.value, true)} />
                  </label>
                  <label className="field">
                    <span>spacing px</span>
                    <input type="number" step="0.5" min="0" value={diamParams.profile_spacing_px} onChange={(e) => updateDiamParam('profile_spacing_px', e.target.value)} />
                  </label>
                  <label className="field">
                    <span>edge score</span>
                    <input type="number" step="0.02" min="0" value={diamParams.edge_min_score} onChange={(e) => updateDiamParam('edge_min_score', e.target.value)} />
                  </label>
                  <label className="field">
                    <span>min profiles</span>
                    <input type="number" step="1" min="1" value={diamParams.min_valid_profiles} onChange={(e) => updateDiamParam('min_valid_profiles', e.target.value, true)} />
                  </label>
                  <label className="field">
                    <span>MAD scale</span>
                    <input type="number" step="0.25" min="0.5" value={diamParams.max_mad_scale} onChange={(e) => updateDiamParam('max_mad_scale', e.target.value)} />
                  </label>
                  <label className="field">
                    <span>mask radius</span>
                    <input type="number" step="2" min="8" value={diamParams.mask_local_radius_px} onChange={(e) => updateDiamParam('mask_local_radius_px', e.target.value)} />
                  </label>
                  <label className="field">
                    <span>mask recenter</span>
                    <input type="number" step="1" min="1" value={diamParams.mask_recenter_radius_px} onChange={(e) => updateDiamParam('mask_recenter_radius_px', e.target.value)} />
                  </label>
                  <label className="field">
                    <span>mask rays</span>
                    <input type="number" step="2" min="8" value={diamParams.mask_ray_count} onChange={(e) => updateDiamParam('mask_ray_count', e.target.value, true)} />
                  </label>
                  <label className="field">
                    <span>auto small px</span>
                    <input type="number" step="1" min="2" value={diamParams.auto_small_context_width_px} onChange={(e) => updateDiamParam('auto_small_context_width_px', e.target.value)} />
                  </label>
                </div>
                {diamMethodId === 'circle_square_mask_diameter' ? (
                  <>
                    <h3>Circle-square</h3>
                    <div className="diam-param-grid">
                      <label className="field">
                        <span>seed mode</span>
                        <input value="manual circle" disabled readOnly />
                      </label>
                      <label className="field">
                        <span>seed radius</span>
                        <input type="number" step="1" min="1" value={diamParams.circle_square_seed_radius_px} onChange={(e) => updateDiamParam('circle_square_seed_radius_px', e.target.value)} />
                      </label>
                      <label className="field">
                        <span>max circle</span>
                        <input type="number" step="2" min="4" value={diamParams.circle_square_max_radius_px} onChange={(e) => updateDiamParam('circle_square_max_radius_px', e.target.value)} />
                      </label>
                      <label className="field">
                        <span>square length</span>
                        <input type="number" step="0.05" min="0.25" value={diamParams.circle_square_length_factor} onChange={(e) => updateDiamParam('circle_square_length_factor', e.target.value)} />
                      </label>
                      <label className="field">
                        <span>square width</span>
                        <input type="number" step="0.05" min="0.25" value={diamParams.circle_square_width_factor} onChange={(e) => updateDiamParam('circle_square_width_factor', e.target.value)} />
                      </label>
                      <label className="field">
                        <span>samples</span>
                        <input type="number" step="2" min="3" value={diamParams.circle_square_samples} onChange={(e) => updateDiamParam('circle_square_samples', e.target.value, true)} />
                      </label>
                      <label className="field">
                        <span>max recenter</span>
                        <input type="number" step="1" min="0" value={diamParams.circle_square_max_recenter_shift_px} onChange={(e) => updateDiamParam('circle_square_max_recenter_shift_px', e.target.value)} />
                      </label>
                      <label className="field">
                        <span>aggregation</span>
                        <select value={diamParams.circle_square_aggregation} onChange={(e) => updateDiamRawParam('circle_square_aggregation', e.target.value)}>
                          <option value="median">median</option>
                          <option value="trimmed_mean">trimmed mean</option>
                        </select>
                      </label>
                      <label className="check-field">
                        <input type="checkbox" checked={Boolean(diamParams.circle_square_recenter_seed)} onChange={(e) => updateDiamRawParam('circle_square_recenter_seed', e.target.checked)} />
                        <span>recenter seed</span>
                      </label>
                    </div>
                  </>
                ) : null}
                {['manual_dual_side_caliper', 'manual_line_direct_caliper'].includes(diamMethodId) ? (
                  <>
                    <h3>{diamMethodId === 'manual_dual_side_caliper' ? 'Linea mask' : 'Linea manual'}</h3>
                    <p className="small">
                      {diamMethodId === 'manual_dual_side_caliper'
                        ? 'MantÃ©n click, arrastra y suelta. El diÃ¡metro se calcula donde esa direcciÃ³n corta la mÃ¡scara.'
                        : 'MantÃ©n click, arrastra y suelta. El diÃ¡metro serÃ¡ exactamente la longitud de la lÃ­nea.'}
                    </p>
                  </>
                ) : null}
                {diamMethodId === 'ellipse_oriented_fit' ? (
                  <>
                    <h3>Ellipse fit</h3>
                    <div className="diam-param-grid">
                      <label className="field">
                        <span>ellipse ROI</span>
                        <input type="number" step="2" min="5" value={diamParams.ellipse_roi_radius_px} onChange={(e) => updateDiamParam('ellipse_roi_radius_px', e.target.value)} />
                      </label>
                    </div>
                  </>
                ) : null}
              </section>

              <section className="card">
                <h2>Historial</h2>
                {!diamRuns.length ? (
                  <div className="placeholder small">Sin runs.</div>
                ) : (
                  <div className="diam-run-list">
                    {diamRuns.map((r) => (
                      <button
                        key={r.run_id}
                        className={`diam-run-row ${r.run_id === diamActiveRunId ? 'selected' : ''}`}
                        onClick={() => loadDiameterRun(r.run_id)}
                        disabled={loading.diamGet}
                      >
                        <strong>{r.points_ok}/{r.point_count} OK</strong>
                        <em>{r.method_id || r.experiment_id || 'diameter'}</em>
                        <span>{r.created_at || r.run_id}</span>
                      </button>
                    ))}
                  </div>
                )}
              </section>

              <section className="card">
                <h2>Analisis por punto</h2>
                {diamReviewTarget ? (
                  <div className="review-target-box">
                    <strong>Punto en revision: #{Number(diamReviewTarget.point_index) + 1}</strong>
                    <span>
                      {Number(diamReviewTarget.point?.x || 0).toFixed(1)}, {Number(diamReviewTarget.point?.y || 0).toFixed(1)}
                      {' | '}
                      {diamReviewTarget.status || 'sin estado'}
                      {diamReviewTarget.diameter_px == null ? '' : ` | ${Number(diamReviewTarget.diameter_px).toFixed(2)} px`}
                    </span>
                  </div>
                ) : (
                  <p className="small">Selecciona `Revisar` en la tabla de resultados por punto para abrir la inspeccion ampliada y escribir tu conclusion.</p>
                )}
                <div className="inline">
                  <label className="field">
                    <span>categoria</span>
                    <select value={validationForm.category} onChange={(e) => updateValidationForm('category', e.target.value)}>
                      <option value="borde_limpio">borde_limpio</option>
                      <option value="halo_moderado">halo_moderado</option>
                      <option value="halo_fuerte">halo_fuerte</option>
                      <option value="interseccion">interseccion</option>
                      <option value="fibra_curva">fibra_curva</option>
                      <option value="bajo_contraste">bajo_contraste</option>
                      <option value="mala_segmentacion">mala_segmentacion</option>
                      <option value="otro">otro</option>
                    </select>
                  </label>
                  <label className="field">
                    <span>calidad manual</span>
                    <select value={validationForm.quality_manual} onChange={(e) => updateValidationForm('quality_manual', e.target.value)}>
                      <option value="high">high</option>
                      <option value="medium">medium</option>
                      <option value="low">low</option>
                    </select>
                  </label>
                </div>
                <label className="field">
                  <span>revision medicion</span>
                  <select value={validationForm.measurement_decision} onChange={(e) => updateValidationForm('measurement_decision', e.target.value)}>
                    <option value="unreviewed">sin revisar</option>
                    <option value="validated">validada</option>
                    <option value="rejected">negada</option>
                    <option value="uncertain">dudosa</option>
                  </select>
                </label>
                <div className="tier-buttons validation-decision-buttons">
                  <button className="tier tier-a" onClick={() => saveValidationCase({ overrides: { measurement_decision: 'validated' } })} disabled={!imageId || loading.validationSave}>Validar punto</button>
                  <button className="tier tier-c" onClick={() => saveValidationCase({ overrides: { measurement_decision: 'uncertain' } })} disabled={!imageId || loading.validationSave}>Dudoso</button>
                  <button className="tier tier-unusable" onClick={() => saveValidationCase({ overrides: { measurement_decision: 'rejected' } })} disabled={!imageId || loading.validationSave}>Negar punto</button>
                </div>
                <label className="field">
                  <span>diametro manual px (opcional)</span>
                  <input type="number" step="0.01" min="0" value={validationForm.manual_diameter_px} onChange={(e) => updateValidationForm('manual_diameter_px', e.target.value)} />
                </label>
                <p className="small">Si el resultado automatico es malo, activa `Manual` sobre la imagen y marca dos extremos del diametro. El valor en px se rellena solo.</p>
                <label className="field">
                  <span>nota tecnica</span>
                  <textarea rows={2} value={validationForm.notes} onChange={(e) => updateValidationForm('notes', e.target.value)} />
                </label>
                <label className="field">
                  <span>conclusion del punto</span>
                  <textarea rows={3} value={validationForm.result_comment} onChange={(e) => updateValidationForm('result_comment', e.target.value)} />
                </label>
                <div className="inline">
                  <button onClick={() => saveValidationCase()} disabled={!imageId || !diamPoints.length || loading.validationSave}>Guardar caso</button>
                  <button className="primary" onClick={runValidationCase} disabled={!imageId || !diamPoints.length || loading.validationRun}>Run Step 5 comparativo</button>
                </div>
                <div className="inline">
                  <button onClick={() => refreshValidationCases()} disabled={!imageId || loading.validationList}>Refrescar casos</button>
                  <button onClick={exportValidationAutofill} disabled={!imageId || loading.validationExport}>Export analisis</button>
                </div>
                {validationExportInfo ? <p className="small">Autofill: {validationExportInfo.autofill_md}</p> : null}
                {!validationCases.length ? (
                  <div className="placeholder small">Sin casos.</div>
                ) : (
                  <div className="validation-case-list">
                    {validationCases.map((c) => (
                      <button
                        key={c.case_id}
                        className={`validation-case-row ${c.case_id === validationActiveCaseId ? 'selected' : ''}`}
                        onClick={() => applyValidationCaseToForm(c)}
                        disabled={loading.validationList}
                      >
                        <strong>{c.case_id}</strong>
                        <span>{c.category || 'otro'} | {c.measurement_decision || 'sin revisar'} | manual {c.manual_diameter_px ?? '-'}</span>
                      </button>
                    ))}
                  </div>
                )}
              </section>

              {/* Calibration Panel */}
              <section className="card">
                <CalibrationPanel
                  calibration={diamCalibration}
                  onChange={setDiamCalibration}
                  onSave={saveCalibration}
                  onLoad={loadCalibration}
                  onDelete={deleteCalibration}
                  loading={loading.calibration}
                  imageId={imageId}
                />
              </section>

              {/* Diameter Distribution */}
              <section className="card">
                <h3>Distribucion de Diametros</h3>
                {diamResults.length > 0 ? (
                  <>
                    <label className="field">
                      <span>Unidad</span>
                      <select value={diamHistogramUnit} onChange={(e) => setDiamHistogramUnit(e.target.value)}>
                        <option value="px">px</option>
                        <option value="nm">nm</option>
                        <option value="um">um</option>
                      </select>
                    </label>
                    <label className="field">
                      <span>Bins</span>
                      <input
                        type="number"
                        min="5"
                        max="50"
                        value={diamHistogramBins}
                        onChange={(e) => setDiamHistogramBins(Math.max(5, Math.min(50, Number(e.target.value) || 20)))}
                      />
                    </label>
                    <Histogram
                      values={diamResults
                        .filter((r) => r.status === 'ok' && r.diameter_px != null)
                        .map((r) => {
                          const d = Number(r.diameter_px)
                          if (diamCalibration.enabled && diamHistogramUnit !== 'px') {
                            const factor = diamCalibration.nm_per_px || (diamCalibration.known_nm / Math.max(1, diamCalibration.pixel_distance))
                            return diamHistogramUnit === 'um' ? (d * factor) / 1000 : d * factor
                          }
                          return d
                        })}
                      unit={diamHistogramUnit}
                      bins={diamHistogramBins}
                    />
                    <button
                      onClick={() => {
                        const csv = exportHistogramCsv(
                          diamResults.filter((r) => r.status === 'ok' && r.diameter_px != null).map((r) => Number(r.diameter_px)),
                          diamHistogramUnit,
                          diamHistogramBins,
                        )
                        const blob = new Blob([csv], { type: 'text/csv' })
                        const url = URL.createObjectURL(blob)
                        const a = document.createElement('a')
                        a.href = url
                        a.download = `diametros_distribucion_${imageId || 'unknown'}.csv`
                        a.click()
                        URL.revokeObjectURL(url)
                      }}
                    >
                      Exportar CSV
                    </button>
                  </>
                ) : (
                  <p className="small">Sin resultados de diametro para mostrar distribucion.</p>
                )}
              </section>
            </>
          ) : workspaceTab === 'loco' ? (
            <>
              <section className="card">
                <h2>LOCO Lab</h2>
                <div className="kpi">ID imagen: <strong>{imageId || '-'}</strong></div>
                <p className="small">Laboratorio experimental para propuestas circulares, filtros, cortes de mascara y medicion tipo circle-square.</p>
                <label className="field">
                  <span>Fuente de soporte</span>
                  <input value="prior_mask" disabled readOnly />
                </label>
                <label className="field">
                  <span>Run de soporte</span>
                  <select value={diamPriorRunId} onChange={(e) => setDiamPriorRunId(e.target.value)} disabled={!imageId}>
                    <option value="latest">ultimo run disponible</option>
                    {runs.map((r, idx) => (
                      <option key={r.run_id} value={r.run_id}>
                        {idx + 1}. {r.experiment_id}{r.profile_name ? ` (${r.profile_name})` : ''} | {r.run_status_level} | {r.created_at || r.run_id}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="inline">
                  <button onClick={() => refreshResults()} disabled={!imageId || loading.listResults}>Refrescar soporte</button>
                  <button onClick={() => setDiamPriorRunId('latest')} disabled={!imageId || diamPriorRunId === 'latest'}>Usar ultimo</button>
                </div>
                <div className="inline">
                  <button className="primary" onClick={generateLocoLabProposals} disabled={!imageId || loading.locoLab}>Generar propuestas</button>
                  <button onClick={filterLocoLabProposals} disabled={!locoProposals.length || loading.locoLab}>Filtrar</button>
                  <button onClick={measureLocoLabProposals} disabled={!locoLabActiveProposals.length || loading.locoLab}>Medir</button>
                </div>
              </section>

              <section className="card">
                <h2>Etapas</h2>
                <div className="loco-step-list">
                  {LOCO_LAB_STAGES.map((step) => (
                    <button
                      key={step.key}
                      className={step.key === locoLabStage ? 'selected' : ''}
                      onClick={() => setLocoLabStage(step.key)}
                      disabled={!imageId}
                    >
                      <strong>{step.label}</strong>
                    </button>
                  ))}
                </div>
                <div className="kpi">Propuestas: <strong>{locoLabActiveProposals.length}</strong></div>
                <div className="kpi">Aceptadas: <strong>{locoLabActiveProposals.filter((p) => p.status === 'accepted').length}</strong></div>
                <div className="kpi">Mediciones OK: <strong>{locoLabOkMeasurements.length}</strong></div>
              </section>

              <section className="card">
                <h2>Parametros LOCO Lab</h2>
                <label className="field">
                  <span>ruta propuesta</span>
                  <select value={locoProposalMethod} onChange={(e) => setLocoProposalMethod(e.target.value)}>
                    <option value="circle_grid">Circle Proposal Grid</option>
                  </select>
                </label>
                <div className="diam-param-grid loco-param-grid">
                  <label className="field">
                    <span>grid stride</span>
                    <input type="number" min="3" step="1" value={locoLabParams.grid_stride_px} onChange={(e) => updateLocoLabParam('grid_stride_px', e.target.value, true)} />
                  </label>
                  <label className="field">
                    <span>radio min</span>
                    <input type="number" min="1" step="1" value={locoLabParams.radius_min_px} onChange={(e) => updateLocoLabParam('radius_min_px', e.target.value)} />
                  </label>
                  <label className="field">
                    <span>radio max</span>
                    <input type="number" min="1" step="1" value={locoLabParams.radius_max_px} onChange={(e) => updateLocoLabParam('radius_max_px', e.target.value)} />
                  </label>
                  <label className="field">
                    <span>radio step</span>
                    <input type="number" min="0.5" step="0.5" value={locoLabParams.radius_step_px} onChange={(e) => updateLocoLabParam('radius_step_px', e.target.value)} />
                  </label>
                  <label className="field">
                    <span>score min</span>
                    <input type="number" min="0" max="1" step="0.02" value={locoLabParams.min_score} onChange={(e) => updateLocoLabParam('min_score', e.target.value)} />
                  </label>
                  <label className="field">
                    <span>ratio mascara</span>
                    <input type="number" min="0" max="1" step="0.02" value={locoLabParams.mask_required_ratio} onChange={(e) => updateLocoLabParam('mask_required_ratio', e.target.value)} />
                  </label>
                  <label className="field">
                    <span>max candidatos</span>
                    <input type="number" min="1" step="50" value={locoLabParams.max_candidates} onChange={(e) => updateLocoLabParam('max_candidates', e.target.value, true)} />
                  </label>
                  <label className="field">
                    <span>measure limit</span>
                    <input type="number" min="1" step="10" value={locoLabParams.measure_limit} onChange={(e) => updateLocoLabParam('measure_limit', e.target.value, true)} />
                  </label>
                  <label className="check-field">
                    <input type="checkbox" checked={Boolean(locoLabParams.require_four_cuts)} onChange={(e) => updateLocoLabRawParam('require_four_cuts', e.target.checked)} />
                    <span>exigir 4 cortes</span>
                  </label>
                </div>
                <div className="inline">
                  <button onClick={clearLocoLab} disabled={!locoProposals.length && !locoMeasuredResults.length}>Limpiar LOCO</button>
                  <button onClick={saveLocoLabRun} disabled={(!locoLabActiveProposals.length && !locoMeasuredResults.length) || loading.locoLab}>Guardar run LOCO</button>
                </div>
              </section>
            </>
          ) : workspaceTab === 'locoDataset' ? (
            <>
              <section className="card">
                <h2>Generate Dataset</h2>
                <label className="field">
                  <span>Run de soporte</span>
                  <select value={diamPriorRunId} onChange={(e) => setDiamPriorRunId(e.target.value)} disabled={!imageId}>
                    <option value="latest">ultimo run disponible</option>
                    {runs.map((r, idx) => (
                      <option key={r.run_id} value={r.run_id}>
                        {idx + 1}. {r.experiment_id}{r.profile_name ? ` (${r.profile_name})` : ''} | {r.run_status_level} | {r.created_at || r.run_id}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="inline">
                  <button onClick={previewLocoDatasetFeatures} disabled={!locoDatasetCircles.length || loading.locoDataset}>Calcular features</button>
                  <button className="primary" onClick={saveLocoDataset} disabled={!locoDatasetCircles.length || loading.locoDataset}>Generate Dataset</button>
                  <button onClick={clearLocoDatasetCanvas} disabled={!locoDatasetCircles.length && !locoDatasetDraftCircle}>Limpiar imagen</button>
                  <button className="danger" onClick={cleanLegacyInvalid} disabled={loading.locoDataset}>Borrar invalid legacy</button>
                </div>
                <p className="small">Dataset: main</p>
              </section>

              <section className="card">
                <h2>Preview</h2>
                {locoDatasetSelectedCircle ? (
                  <>
                    <div className={`loco-dataset-preview ${locoDatasetSelectedCircle.label || 'invalid_other'}`}>
                      <svg
                        viewBox={`${Math.max(0, Number(locoDatasetSelectedCircle.center_x) - Number(locoDatasetSelectedCircle.radius_px))} ${Math.max(0, Number(locoDatasetSelectedCircle.center_y) - Number(locoDatasetSelectedCircle.radius_px))} ${Math.max(1, Number(locoDatasetSelectedCircle.radius_px) * 2)} ${Math.max(1, Number(locoDatasetSelectedCircle.radius_px) * 2)}`}
                        aria-label="Recorte real del candidato"
                      >
                        <defs>
                          <clipPath id="locoDatasetPreviewClip">
                            <circle
                              cx={Number(locoDatasetSelectedCircle.center_x)}
                              cy={Number(locoDatasetSelectedCircle.center_y)}
                              r={Math.max(1, Number(locoDatasetSelectedCircle.radius_px))}
                            />
                          </clipPath>
                        </defs>
                        <g clipPath="url(#locoDatasetPreviewClip)">
                          <image href={imageUrl} x="0" y="0" width={Math.max(1, imageDims.w)} height={Math.max(1, imageDims.h)} preserveAspectRatio="none" />
                          {diamVisualMaskUrl ? (
                            <image href={diamVisualMaskUrl} x="0" y="0" width={Math.max(1, imageDims.w)} height={Math.max(1, imageDims.h)} preserveAspectRatio="none" opacity={diamMaskOpacity} />
                          ) : null}
                        </g>
                        <circle
                          cx={Number(locoDatasetSelectedCircle.center_x)}
                          cy={Number(locoDatasetSelectedCircle.center_y)}
                          r={Math.max(1, Number(locoDatasetSelectedCircle.radius_px))}
                        />
                        <line
                          x1={Number(locoDatasetSelectedCircle.center_x) - Number(locoDatasetSelectedCircle.radius_px)}
                          y1={Number(locoDatasetSelectedCircle.center_y)}
                          x2={Number(locoDatasetSelectedCircle.center_x) + Number(locoDatasetSelectedCircle.radius_px)}
                          y2={Number(locoDatasetSelectedCircle.center_y)}
                        />
                        <line
                          x1={Number(locoDatasetSelectedCircle.center_x)}
                          y1={Number(locoDatasetSelectedCircle.center_y) - Number(locoDatasetSelectedCircle.radius_px)}
                          x2={Number(locoDatasetSelectedCircle.center_x)}
                          y2={Number(locoDatasetSelectedCircle.center_y) + Number(locoDatasetSelectedCircle.radius_px)}
                        />
                      </svg>
                      <strong>{locoDatasetSelectedCircle.label === 'valid' ? 'Valid' : locoDatasetSelectedCircle.label === 'invalid_crossing' ? 'Crossing' : 'Other invalid'}</strong>
                      <span>r {Number(locoDatasetSelectedCircle.radius_px).toFixed(1)} px</span>
                    </div>
                    <p className="small">{locoDatasetSelectedCircle.candidate_id}<br />{Number(locoDatasetSelectedCircle.center_x).toFixed(1)}, {Number(locoDatasetSelectedCircle.center_y).toFixed(1)}</p>
                  </>
                ) : (
                  <div className="placeholder small">Selecciona o dibuja un circulo.</div>
                )}
                <div className="kpi">Total: <strong>{locoDatasetCounts.total}</strong></div>
                <div className="kpi">Valid: <strong>{locoDatasetCounts.valid}</strong></div>
                <div className="kpi">Crossing: <strong>{locoDatasetCounts.invalidCrossing}</strong></div>
                <div className="kpi">Other: <strong>{locoDatasetCounts.invalidOther}</strong></div>
                <div className="kpi">Siguiente: <strong>{locoDatasetDefaultLabel === 'valid' ? 'Valid' : locoDatasetDefaultLabel === 'invalid_crossing' ? 'Crossing' : 'Other invalid'}</strong></div>
              </section>

              <section className="card">
                <h2>Dataset</h2>
                {locoDatasetSaveInfo ? (
                  <p className="small">Guardado en: {locoDatasetSaveInfo.dataset_dir}<br />Total {locoDatasetSaveInfo.candidate_count}</p>
                ) : null}
                <div className="loco-candidate-list">
                  {locoDatasetCircles.length ? locoDatasetCircles.map((c, idx) => (
                    <button
                      key={c.candidate_id}
                      className={String(c.candidate_id) === String(locoDatasetSelectedId) ? 'selected' : ''}
                      onClick={() => { setLocoDatasetSelectedId(String(c.candidate_id)); if (locoDatasetTool !== 'circle') setLocoDatasetTool('select') }}
                    >
                      <strong>{idx + 1}</strong>
                      <span>{c.label} | r {Number(c.radius_px).toFixed(1)}</span>
                      <em>{Number(c.center_x).toFixed(0)}, {Number(c.center_y).toFixed(0)}</em>
                    </button>
                  )) : <div className="placeholder small">Sin circulos anotados.</div>}
                </div>
              </section>
            </>
          ) : workspaceTab === 'locoAugment' ? (
            <>
              <section className="card">
                <h2>Augmentation</h2>
                <div className="kpi">Dataset: <strong>main</strong></div>
                <div className="kpi">Raw: <strong>{locoAugCounts.total || 0}</strong></div>
                <div className="kpi">Valid: <strong>{locoAugCounts.valid || 0}</strong></div>
                <div className="kpi">Invalid: <strong>{locoAugCounts.invalid || 0}</strong></div>
                <div className="kpi">Augmented: <strong>{locoAugCounts.augmented_total || 0}</strong></div>
                <label className="field">
                  <span>pasadas por imagen</span>
                  <input type="number" min="1" max="128" value={locoAugPasses} onChange={(e) => setLocoAugPasses(clamp(Number(e.target.value || 1), 1, 128))} />
                </label>
                <div className="inline">
                  <button onClick={refreshLocoAugItems} disabled={loading.locoAugment}>Refrescar</button>
                  <button onClick={selectMixedLocoAugSample} disabled={!locoAugItems.length}>Seleccionar muestra mixta</button>
                  <button onClick={() => setLocoAugSelected({})} disabled={!locoAugSelectedCount}>Limpiar selecciÃ³n</button>
                </div>
              </section>

              <section className="card">
                <h2>Fuente</h2>
                <label className="field">
                  <span>clase</span>
                  <select value={locoAugLabelFilter} onChange={(e) => setLocoAugLabelFilter(e.target.value)}>
                    <option value="all">todas</option>
                    <option value="valid">valid</option>
                    <option value="invalid_crossing">crossing</option>
                    <option value="invalid_other">other invalid</option>
                  </select>
                </label>
                <div className="kpi">Visibles: <strong>{locoAugFilteredItems.length}</strong></div>
                <div className="augment-source-grid">
                  {locoAugFilteredItems.map((item) => (
                    <button
                      key={item.item_id}
                      className={`augment-source-thumb ${item.label} ${locoAugSelected[item.item_id] ? 'selected' : ''}`}
                      onClick={() => setLocoAugSelected((prev) => ({ ...prev, [item.item_id]: !prev[item.item_id] }))}
                      title={`${item.label} | ${item.candidate_id}`}
                    >
                      {item.source_b64 ? <img src={b64ToDataUrl(item.source_b64, 'image/png')} alt={item.candidate_id} /> : <span className="thumb-placeholder" />}
                      <strong>{item.label}</strong>
                      <em>{item.candidate_id}</em>
                    </button>
                  ))}
                  {!locoAugFilteredItems.length ? <div className="placeholder small">No hay ejemplos en main.</div> : null}
                </div>
              </section>

              <section className="card">
                <h2>Acciones</h2>
                <div className="kpi">Seleccionados: <strong>{locoAugSelectedCount}</strong></div>
                <div className="kpi">Pasadas por ejemplo: <strong>{locoAugEstimatedVariants}</strong></div>
                <div className="inline">
                  <button onClick={previewLocoAugmentation} disabled={loading.locoAugment || !locoAugPipeline.length}>Preview selected</button>
                  <button className="primary" onClick={applyLocoAugmentation} disabled={loading.locoAugment || !locoAugPipeline.length || !locoAugItems.length}>Apply to all dataset</button>
                  <button onClick={clearLocoAugmented} disabled={loading.locoAugment || !(locoAugCounts.augmented_total > 0)}>Clear augmented</button>
                </div>
                {locoAugInfo ? (
                  <p className="small">
                    {locoAugInfo.augmented_dir || `pipeline ${locoAugInfo.pipeline_hash || '-'}`}<br />
                    Total augmented: {locoAugInfo.augmented_count ?? locoAugInfo.variant_count ?? '-'}
                  </p>
                ) : null}
              </section>
            </>
          ) : workspaceTab === 'locoTraining' ? (
            <>
              <section className="card">
                <h2>Training</h2>
                <label className="field">
                  <span>datos</span>
                  <select value={locoTrainingDataSelection} onChange={(e) => setLocoTrainingDataSelection(e.target.value)}>
                    <option value="original">Original only</option>
                    <option value="augmented">Augmented only</option>
                    <option value="all">Original + Augmented</option>
                  </select>
                </label>
                <label className="field">
                  <span>test size</span>
                  <input type="number" min="0.05" max="0.5" step="0.05" value={locoTrainingTestSize} onChange={(e) => setLocoTrainingTestSize(clamp(Number(e.target.value || 0.2), 0.05, 0.5))} />
                </label>
                <label className="field">
                  <span>random seed</span>
                  <input type="number" value={locoTrainingSeed} onChange={(e) => setLocoTrainingSeed(Number(e.target.value || 42))} />
                </label>
                <label className="field">
                  <span>pixeles</span>
                  <select value={locoTrainingPixelMode} onChange={(e) => setLocoTrainingPixelMode(e.target.value)}>
                    <option value="square_64">square_64 legacy</option>
                    <option value="circle_only">circle_only</option>
                  </select>
                </label>
                <label className="field">
                  <span>poda borde</span>
                  <input type="number" min="0" max="2" step="1" value={locoTrainingPrunePx} disabled={locoTrainingPixelMode !== 'circle_only'} onChange={(e) => setLocoTrainingPrunePx(clamp(Number(e.target.value || 0), 0, 2))} />
                </label>
                <label className="check-field"><input type="checkbox" checked={!!locoTrainingUseZoom} onChange={(e) => setLocoTrainingUseZoom(e.target.checked)} /><span>patch_zoom_factor</span></label>
                <label className="check-field"><input type="checkbox" checked={!!locoTrainingUseSourceRadius} onChange={(e) => setLocoTrainingUseSourceRadius(e.target.checked)} /><span>radio real en aumentados</span></label>
                <label className="check-field"><input type="checkbox" checked={!!locoTrainingMulticlass} onChange={(e) => setLocoTrainingMulticlass(e.target.checked)} /><span>modelo multiclase (valid/crossing/other)</span></label>
                <button className="primary" onClick={trainLocoModels} disabled={loading.locoTraining}>Train Models</button>
                {locoTrainingResult ? (
                  <p className="small">{locoTrainingResult.run_id}<br />{locoTrainingResult.meta?.pixel_mode || '-'} prune {locoTrainingResult.meta?.circle_prune_px ?? 0} | {locoTrainingResult.meta?.feature_count ?? '-'} cols<br />{locoTrainingResult.run_dir}</p>
                ) : null}
              </section>

              <section className="card">
                <h2>Modelo</h2>
                <label className="field">
                  <span>modelo</span>
                  <select value={locoTrainingSelectedModel} onChange={(e) => setLocoTrainingSelectedModel(e.target.value)} disabled={!locoTrainingMetrics.length}>
                    {locoTrainingMetrics.map((row) => <option key={row.model_id} value={row.model_id}>{row.model}</option>)}
                  </select>
                </label>
                <label className="field">
                  <span>threshold</span>
                  <input type="number" min="0.05" max="0.95" step="0.05" value={locoTrainingThreshold} onChange={(e) => setLocoTrainingThreshold(clamp(Number(e.target.value || 0.5), 0.05, 0.95))} />
                </label>
                <label className="field">
                  <span>error</span>
                  <select value={locoTrainingErrorType} onChange={(e) => setLocoTrainingErrorType(e.target.value)}>
                    <option value="False Positives">False Positives</option>
                    <option value="False Negatives">False Negatives</option>
                  </select>
                </label>
                {locoTrainingResult?.model_recommendations?.[locoTrainingSelectedModel] ? (
                  <div className="kpi-row">
                    <span>precision th <strong>{locoTrainingResult.model_recommendations[locoTrainingSelectedModel].recommended_threshold_precision}</strong></span>
                    <span>f1 th <strong>{locoTrainingResult.model_recommendations[locoTrainingSelectedModel].recommended_threshold_f1}</strong></span>
                  </div>
                ) : null}
              </section>
            </>
          ) : workspaceTab === 'locoTest' ? (
            <>
              <section className="card">
                <h2>Test circle model</h2>
                <label className="field">
                  <span>training run</span>
                  <select value={locoTestTrainingRunId} onChange={(e) => setLocoTestTrainingRunId(e.target.value)}>
                    <option value="latest">latest</option>
                    {locoTrainingRuns.map((r) => (
                      <option key={r.run_id} value={r.run_id}>{r.created_at || r.run_id} | {r.sample_count} samples</option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span>model</span>
                  <select value={locoTestModelId} onChange={(e) => setLocoTestModelId(e.target.value)}>
                    <option value="catboost">CatBoost</option>
                    <option value="lightgbm">LightGBM</option>
                    <option value="xgboost">XGBoost</option>
                    <option value="extratrees">ExtraTrees</option>
                  </select>
                </label>
                <label className="field">
                  <span>threshold</span>
                  <input type="number" min="0.05" max="0.95" step="0.05" value={locoTestThreshold} onChange={(e) => setLocoTestThreshold(clamp(Number(e.target.value || 0.5), 0.05, 0.95))} />
                </label>
                <label className="field">
                  <span>Run de soporte</span>
                  <select value={diamPriorRunId} onChange={(e) => setDiamPriorRunId(e.target.value)} disabled={!imageId}>
                    <option value="latest">ultimo run disponible</option>
                    {runs.map((r, idx) => (
                      <option key={r.run_id} value={r.run_id}>
                        {idx + 1}. {r.experiment_id} | {r.run_status_level}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="inline">
                  <button onClick={refreshLocoTrainingRuns} disabled={loading.locoTraining}>Refrescar modelos</button>
                  <button className="primary" onClick={predictLocoTestCircles} disabled={!locoTestCircles.length || loading.locoTest}>Predict circles</button>
                  <button onClick={clearLocoTestCanvas} disabled={!locoTestCircles.length && !locoTestDraftCircle}>Limpiar imagen</button>
                </div>
              </section>

              <section className="card">
                <h2>Etiquetas</h2>
                <div className="kpi">Total: <strong>{locoTestCounts.total}</strong></div>
                <div className="kpi">Valid: <strong>{locoTestCounts.valid}</strong></div>
                <div className="kpi">Crossing: <strong>{locoTestCounts.invalidCrossing}</strong></div>
                <div className="kpi">Other: <strong>{locoTestCounts.invalidOther}</strong></div>
                <div className="kpi">Siguiente: <strong>{locoTestDefaultLabel}</strong></div>
                {locoTestSelectedCircle ? (
                  <>
                    <div className={`inline dataset-label-actions ${locoTestSelectedCircle.label === 'invalid' ? 'invalid_other' : (locoTestSelectedCircle.label || 'invalid_other')}`}>
                      <button className={`dataset-valid ${locoTestSelectedCircle.label === 'valid' ? 'active' : ''}`} onClick={() => updateLocoTestCircle(locoTestSelectedCircle.candidate_id, { label: 'valid' })}>Valid</button>
                      <button className={`dataset-invalid-crossing ${locoTestSelectedCircle.label === 'invalid_crossing' ? 'active' : ''}`} onClick={() => updateLocoTestCircle(locoTestSelectedCircle.candidate_id, { label: 'invalid_crossing' })}>Crossing</button>
                      <button className={`dataset-invalid-other ${locoTestSelectedCircle.label === 'invalid_other' || locoTestSelectedCircle.label === 'invalid' ? 'active' : ''}`} onClick={() => updateLocoTestCircle(locoTestSelectedCircle.candidate_id, { label: 'invalid_other' })}>Other</button>
                      <button className="dataset-delete" onClick={deleteSelectedLocoTestCircle}>Eliminar</button>
                    </div>
                    <label className="field">
                      <span>radio px</span>
                      <input type="number" min="1" step="0.5" value={Number(locoTestSelectedCircle.radius_px).toFixed(1)} onChange={(e) => updateLocoTestCircle(locoTestSelectedCircle.candidate_id, { radius_px: Math.max(1, Number(e.target.value || 1)) })} />
                    </label>
                  </>
                ) : <p className="small">Selecciona o dibuja un circulo.</p>}
              </section>

              {locoTestResult ? (
                <section className="card">
                  <h2>MÃ©tricas</h2>
                  <div className="kpi">Precision valid: <strong>{locoTestResult.metrics?.precision_valid == null ? '-' : Number(locoTestResult.metrics.precision_valid).toFixed(3)}</strong></div>
                  <div className="kpi">Recall valid: <strong>{locoTestResult.metrics?.recall_valid == null ? '-' : Number(locoTestResult.metrics.recall_valid).toFixed(3)}</strong></div>
                  <div className="kpi">F1 valid: <strong>{locoTestResult.metrics?.f1_valid == null ? '-' : Number(locoTestResult.metrics.f1_valid).toFixed(3)}</strong></div>
                  <div className="kpi">FP: <strong>{locoTestResult.metrics?.fp ?? '-'}</strong></div>
                  <div className="kpi">FN: <strong>{locoTestResult.metrics?.fn ?? '-'}</strong></div>
                  {locoTestResult.has_multiclass ? <div className="kpi">Multiclase: <strong>âœ“</strong></div> : null}
                </section>
              ) : null}
            </>
          ) : workspaceTab === 'locoModel' ? (
            <>
              <section className="card">
                <h2>LOCO Detector</h2>
                <label className="field">
                  <span>training run</span>
                  <select value={locoModelRunId} onChange={(e) => setLocoModelRunId(e.target.value)}>
                    <option value="latest">latest</option>
                    {locoTrainingRuns.map((r) => (
                      <option key={r.run_id} value={r.run_id}>{r.created_at || r.run_id} | {r.sample_count} samples</option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span>model</span>
                  <select value={locoModelId} onChange={(e) => setLocoModelId(e.target.value)}>
                    <option value="catboost">CatBoost</option>
                    <option value="lightgbm">LightGBM</option>
                    <option value="xgboost">XGBoost</option>
                    <option value="extratrees">ExtraTrees</option>
                  </select>
                </label>
                <label className="field">
                  <span>Run de soporte</span>
                  <select value={diamPriorRunId} onChange={(e) => setDiamPriorRunId(e.target.value)} disabled={!imageId}>
                    <option value="latest">ultimo run disponible</option>
                    {runs.map((r, idx) => (
                      <option key={r.run_id} value={r.run_id}>{idx + 1}. {r.experiment_id} | {r.run_status_level}</option>
                    ))}
                  </select>
                </label>
                <div className="inline">
                  <button onClick={refreshLocoTrainingRuns} disabled={loading.locoTraining}>Refrescar modelos</button>
                  <button className="primary" onClick={detectLocoModelCircles} disabled={!imageUrl || loading.locoModel}>Detect circles</button>
                  <button onClick={clearLocoModelDetector} disabled={!locoModelResult}>Limpiar detector</button>
                </div>
                <button onClick={measureLocoModelAccepted} disabled={!locoModelAccepted.length || loading.locoModel}>Run Diameter on Accepted</button>
              </section>

              <section className="card">
                <h2>Preset</h2>
                <label className="field">
                  <span>configuraci\u00f3n predefinida</span>
                  <select value={locoModelPreset} onChange={(e) => applyLocoModelPreset(e.target.value)}>
                    {Object.entries(LOCO_PRESETS).map(([key, p]) => (
                      <option key={key} value={key}>{p.label}</option>
                    ))}
                  </select>
                </label>
                <p className="small">Selecciona un perfil seg\u00fan el tipo de fibra y nivel de estrictez. Los par\u00e1metros se ajustan autom\u00e1ticamente. Al modificar cualquier par\u00e1metro manualmente, vuelve a "Personalizado".</p>
              </section>

              <section className="card">
                <h2>Generacion</h2>
                <div className="diam-param-grid">
                  <label className="field"><ParamSpan paramKey="candidate_sampling_mode">muestreo</ParamSpan><select value={locoModelParams.candidate_sampling_mode} onChange={(e) => updateLocoModelParam('candidate_sampling_mode', e.target.value)}><option value="tile_balanced">tile balanced</option><option value="random_seeded">random seeded</option><option value="row_major">row major</option></select></label>
                  <label className="field"><ParamSpan paramKey="grid_step">grid step</ParamSpan><input type="number" min="2" step="1" value={locoModelParams.grid_step} onChange={(e) => updateLocoModelParam('grid_step', Number(e.target.value || 10))} onBlur={clampOnBlur('grid_step', 2, 128, 1, 10)} /></label>
                  <label className="field"><ParamSpan paramKey="max_candidates">max candidatos</ParamSpan><input type="number" min="100" step="500" value={locoModelParams.max_candidates} onChange={(e) => updateLocoModelParam('max_candidates', Number(e.target.value || 8000))} onBlur={clampOnBlur('max_candidates', 100, 60000, 500, 8000)} /></label>
                  {locoModelParams.candidate_sampling_mode === 'tile_balanced' ? (
                    <label className="field"><ParamSpan paramKey="candidate_max_per_tile">max/tile</ParamSpan><input type="number" min="0" step="50" value={locoModelParams.candidate_max_per_tile} onChange={(e) => updateLocoModelParam('candidate_max_per_tile', Number(e.target.value || 0))} onBlur={clampOnBlur('candidate_max_per_tile', 0, 60000, 50, 0)} /></label>
                  ) : null}
                  {locoModelParams.candidate_sampling_mode === 'tile_balanced' ? (
                    <label className="field"><ParamSpan paramKey="tile_size_px">tile px</ParamSpan><input type="number" min="32" step="32" value={locoModelParams.tile_size_px} onChange={(e) => updateLocoModelParam('tile_size_px', Number(e.target.value || 128))} onBlur={clampOnBlur('tile_size_px', 32, 2048, 32, 128)} /></label>
                  ) : null}
                  {locoModelParams.candidate_sampling_mode !== 'row_major' ? (
                    <label className="field"><ParamSpan paramKey="candidate_random_seed">seed</ParamSpan><input type="number" value={locoModelParams.candidate_random_seed} onChange={(e) => updateLocoModelParam('candidate_random_seed', Number(e.target.value || 42))} /></label>
                  ) : null}
                  <label className="field"><ParamSpan paramKey="min_radius">radio min</ParamSpan><input type="number" min="1" step="1" value={locoModelParams.min_radius} onChange={(e) => updateLocoModelParam('min_radius', Number(e.target.value || 1))} onBlur={clampOnBlur('min_radius', 1, 9999, 1, 1)} /></label>
                  <label className="field"><ParamSpan paramKey="max_radius">radio max</ParamSpan><input type="number" min="1" step="1" value={locoModelParams.max_radius} onChange={(e) => updateLocoModelParam('max_radius', Number(e.target.value || 1))} onBlur={clampOnBlur('max_radius', 1, 9999, 1, 1)} /></label>
                  <label className="field"><ParamSpan paramKey="radius_step">radio step</ParamSpan><input type="number" min="0.5" step="0.5" value={locoModelParams.radius_step} onChange={(e) => updateLocoModelParam('radius_step', Number(e.target.value || 1))} onBlur={clampOnBlur('radius_step', 0.5, 9999, 0.5, 1)} /></label>
                </div>
              </section>

              <section className="card">
                <h2>Threshold</h2>
                <label className="check-field"><input type="checkbox" checked={!!locoModelParams.use_radius_thresholds} onChange={(e) => updateLocoModelParam('use_radius_thresholds', e.target.checked)} /><ParamSpan paramKey="use_radius_thresholds">threshold por radio</ParamSpan></label>
                <div className="diam-param-grid">
                  {!locoModelParams.use_radius_thresholds ? (
                    <label className="field"><ParamSpan paramKey="threshold">general</ParamSpan><input type="number" min="0.01" max="0.99" step="0.01" value={locoModelParams.threshold} onChange={(e) => updateLocoModelParam('threshold', Number(e.target.value || 0.8))} onBlur={clampOnBlur('threshold', 0.01, 0.99, 0.01, 0.8)} /></label>
                  ) : null}
                  {locoModelParams.use_radius_thresholds ? (
                    <label className="field"><ParamSpan paramKey="small_threshold">small th</ParamSpan><input type="number" min="0.01" max="0.99" step="0.01" value={locoModelParams.small_threshold} onChange={(e) => updateLocoModelParam('small_threshold', Number(e.target.value || 0.8))} onBlur={clampOnBlur('small_threshold', 0.01, 0.99, 0.01, 0.8)} /></label>
                  ) : null}
                  {locoModelParams.use_radius_thresholds ? (
                    <label className="field"><ParamSpan paramKey="medium_threshold">medium th</ParamSpan><input type="number" min="0.01" max="0.99" step="0.01" value={locoModelParams.medium_threshold} onChange={(e) => updateLocoModelParam('medium_threshold', Number(e.target.value || 0.8))} onBlur={clampOnBlur('medium_threshold', 0.01, 0.99, 0.01, 0.8)} /></label>
                  ) : null}
                  {locoModelParams.use_radius_thresholds ? (
                    <label className="field"><ParamSpan paramKey="large_threshold">large th</ParamSpan><input type="number" min="0.01" max="0.99" step="0.01" value={locoModelParams.large_threshold} onChange={(e) => updateLocoModelParam('large_threshold', Number(e.target.value || 0.9))} onBlur={clampOnBlur('large_threshold', 0.01, 0.99, 0.01, 0.9)} /></label>
                  ) : null}
                  {locoModelParams.use_radius_thresholds ? (
                    <label className="field"><ParamSpan paramKey="small_radius_limit">small limite</ParamSpan><input type="number" min="1" step="1" value={locoModelParams.small_radius_limit} onChange={(e) => updateLocoModelParam('small_radius_limit', Number(e.target.value || 14))} onBlur={clampOnBlur('small_radius_limit', 1, 9999, 1, 14)} /></label>
                  ) : null}
                  {locoModelParams.use_radius_thresholds ? (
                    <label className="field"><ParamSpan paramKey="large_radius_limit">large limite</ParamSpan><input type="number" min="1" step="1" value={locoModelParams.large_radius_limit} onChange={(e) => updateLocoModelParam('large_radius_limit', Number(e.target.value || 24))} onBlur={clampOnBlur('large_radius_limit', 1, 9999, 1, 24)} /></label>
                  ) : null}
                </div>
              </section>

              <section className="card">
                <h2>Multiclase</h2>
                <label className="field"><ParamSpan paramKey="crossing_threshold">crossing threshold</ParamSpan><input type="number" min="0.01" max="0.99" step="0.01" value={locoModelParams.crossing_threshold} onChange={(e) => updateLocoModelParam('crossing_threshold', Number(e.target.value || 0.5))} onBlur={clampOnBlur('crossing_threshold', 0.01, 0.99, 0.01, 0.5)} /></label>
                <p className="small">Controla qu\u00e9 tan estricto es el filtro de cruces. Valor bajo = menos falsos positivos por cruce.</p>
              </section>

              <section className="card">
                <h2>NMS y capas</h2>
                <label className="check-field"><input type="checkbox" checked={!!locoModelParams.use_nms} onChange={(e) => updateLocoModelParam('use_nms', e.target.checked)} /><ParamSpan paramKey="use_nms">Circle-NMS</ParamSpan></label>
                <label className="check-field"><input type="checkbox" checked={!!locoModelParams.return_rejected} onChange={(e) => { updateLocoModelParam('return_rejected', e.target.checked); setLocoModelLayers((prev) => ({ ...prev, rejected: e.target.checked })) }} /><ParamSpan paramKey="return_rejected">mostrar rechazados</ParamSpan></label>
                {locoModelParams.return_rejected ? (
                  <label className="field" style={{ marginTop: 4 }}><ParamSpan paramKey="max_return_rejected">max rechazados</ParamSpan><input type="number" min="0" max="50000" step="100" value={locoModelParams.max_return_rejected} onChange={(e) => updateLocoModelParam('max_return_rejected', Number(e.target.value || 5000))} onBlur={clampOnBlur('max_return_rejected', 0, 50000, 100, 5000)} /><span className="small" style={{ marginLeft: 4 }}>0=todos</span></label>
                ) : null}
                <div className="diam-param-grid">
                  <label className="field"><ParamSpan paramKey="nms_mode">modo NMS</ParamSpan><select value={locoModelParams.nms_mode} onChange={(e) => updateLocoModelParam('nms_mode', e.target.value)}><option value="circle_iou">circle IoU</option><option value="distance_radius">dist/radio</option></select></label>
                  <label className="field"><ParamSpan paramKey="circle_iou_threshold">IoU th</ParamSpan><input type="number" min="0.05" max="0.95" step="0.05" value={locoModelParams.circle_iou_threshold} onChange={(e) => updateLocoModelParam('circle_iou_threshold', Number(e.target.value || 0.4))} onBlur={clampOnBlur('circle_iou_threshold', 0.05, 0.95, 0.05, 0.4)} /></label>
                  <label className="field"><ParamSpan paramKey="nms_distance_factor">nms distancia</ParamSpan><input type="number" min="0.05" step="0.05" value={locoModelParams.nms_distance_factor} onChange={(e) => updateLocoModelParam('nms_distance_factor', Number(e.target.value || 0.5))} onBlur={clampOnBlur('nms_distance_factor', 0.05, 3, 0.05, 0.5)} /></label>
                  <label className="field"><ParamSpan paramKey="radius_similarity_factor">radio similar</ParamSpan><input type="number" min="0" step="0.05" value={locoModelParams.radius_similarity_factor} onChange={(e) => updateLocoModelParam('radius_similarity_factor', Number(e.target.value || 0.4))} onBlur={clampOnBlur('radius_similarity_factor', 0, 3, 0.05, 0.4)} /></label>
                </div>
                <div className="loco-layer-grid">
                  <label><input type="checkbox" checked={!!locoModelLayers.mask} onChange={(e) => setLocoModelLayers((prev) => ({ ...prev, mask: e.target.checked }))} /> mascara</label>
                  <label><input type="checkbox" checked={!!locoModelLayers.accepted} onChange={(e) => setLocoModelLayers((prev) => ({ ...prev, accepted: e.target.checked }))} /> aceptados</label>
                  <label><input type="checkbox" checked={!!locoModelLayers.rejected} onChange={(e) => setLocoModelLayers((prev) => ({ ...prev, rejected: e.target.checked }))} /> rechazados</label>
                  <label><input type="checkbox" checked={!!locoModelLayers.scores} onChange={(e) => setLocoModelLayers((prev) => ({ ...prev, scores: e.target.checked }))} /> scores</label>
                  <label><input type="checkbox" checked={!!locoModelLayers.tiles} onChange={(e) => setLocoModelLayers((prev) => ({ ...prev, tiles: e.target.checked }))} /> tiles</label>
                </div>
              </section>

              <section className="card">
                <h2>Filtro espacial</h2>
                <label className="check-field"><input type="checkbox" checked={!!locoModelParams.use_spatial_final_filter} onChange={(e) => updateLocoModelParam('use_spatial_final_filter', e.target.checked)} /><ParamSpan paramKey="use_spatial_final_filter">Filtro espacial final (post-NMS)</ParamSpan></label>
                <p className="small">Divide la imagen en tiles y limita el n\u00famero de c\u00edrculos aceptados por tile, asegurando distribuci\u00f3n espacial uniforme.</p>
                <div className="diam-param-grid">
                  <label className="field"><ParamSpan paramKey="spatial_final_tile_px">tile px</ParamSpan><input type="number" min="16" max="512" step="8" value={locoModelParams.spatial_final_tile_px} onChange={(e) => updateLocoModelParam('spatial_final_tile_px', Number(e.target.value || 128))} onBlur={clampOnBlur('spatial_final_tile_px', 16, 512, 8, 128)} /></label>
                  <label className="field"><ParamSpan paramKey="spatial_final_max_per_tile">max/tile</ParamSpan><input type="number" min="1" max="50" step="1" value={locoModelParams.spatial_final_max_per_tile} onChange={(e) => updateLocoModelParam('spatial_final_max_per_tile', Number(e.target.value || 3))} onBlur={clampOnBlur('spatial_final_max_per_tile', 1, 50, 1, 3)} /></label>
                  <label className="field"><ParamSpan paramKey="spatial_final_min_center_distance_factor">distancia min</ParamSpan><input type="number" min="0" max="5" step="0.1" value={locoModelParams.spatial_final_min_center_distance_factor} onChange={(e) => updateLocoModelParam('spatial_final_min_center_distance_factor', Number(e.target.value || 1.0))} onBlur={clampOnBlur('spatial_final_min_center_distance_factor', 0, 5, 0.1, 1.0)} /></label>
                </div>
              </section>

              {locoModelResult ? (
                <section className="card">
                  <h2>Resumen</h2>
                  <div className="kpi">Total candidatos: <strong>{locoModelResult.summary?.total_candidates ?? 0}</strong></div>
                  <div className="kpi">Muestra: <strong>{locoModelResult.summary?.sampled_candidates ?? locoModelResult.summary?.total_candidates ?? 0}</strong></div>
                  <div className="kpi">Modelo: <strong>{locoModelResult.summary?.evaluated_candidates ?? 0}</strong></div>
                  <div className="kpi">Threshold: <strong>{locoModelResult.summary?.accepted_before_nms ?? 0}</strong></div>
                  <div className="kpi">Final NMS: <strong>{locoModelResult.summary?.accepted_after_nms ?? 0}</strong></div>
                  <div className="kpi">NMS removidos: <strong>{locoModelResult.summary?.removed_by_nms ?? 0}</strong></div>
                  {locoModelResult.summary?.accepted_after_spatial != null ? <div className="kpi">Final spatial: <strong>{locoModelResult.summary.accepted_after_spatial}</strong></div> : null}
                  {locoModelResult.summary?.removed_by_spatial ? <div className="kpi">Spatial removidos: <strong>{locoModelResult.summary.removed_by_spatial}</strong></div> : null}
                  <div className="kpi">Multiclase: <strong>{locoModelResult.has_multiclass ? 'âœ“' : 'âœ—'}</strong></div>
                  {locoModelResult.has_multiclass ? <div className="kpi">crossing th: <strong>{Number(locoModelResult.crossing_threshold || 0.5).toFixed(2)}</strong></div> : null}
                  <div className="kpi">Run: <strong>{locoModelResult.run_id}</strong></div>
                </section>
              ) : null}
            </>
          ) : (
            <>
              <section className="card">
                <h2>Entrenamiento</h2>
                <div className="kpi">Seleccionadas: <strong>{trainImageIds.length}</strong></div>
                <label className="field">
                  <span>nombre modelo</span>
                  <input value={trainConfig.model_name} onChange={(e) => updateTrainConfig('model_name', e.target.value)} placeholder="assist_model_v001" />
                </label>
                <div className="inline">
                  <label className="field">
                    <span>clases</span>
                    <select value={trainConfig.class_mode} onChange={(e) => updateTrainConfig('class_mode', e.target.value)}>
                      <option value="multiclass">fibra / halo / background</option>
                      <option value="binary">fibra / no-fibra</option>
                    </select>
                  </label>
                  <label className="field">
                    <span>modelo</span>
                    <select value={trainConfig.classifier} onChange={(e) => updateTrainConfig('classifier', e.target.value)}>
                      <option value="extratrees">ExtraTrees</option>
                      <option value="rf">RandomForest</option>
                    </select>
                  </label>
                </div>
                <div className="inline">
                  <label className="field">
                    <span>features</span>
                    <select value={trainConfig.feature_variant} onChange={(e) => updateTrainConfig('feature_variant', e.target.value)}>
                      <option value="context">context</option>
                      <option value="base">base</option>
                    </select>
                  </label>
                  <label className="field">
                    <span>arboles</span>
                    <input type="number" min="20" max="500" step="20" value={trainConfig.n_estimators} onChange={(e) => updateTrainConfig('n_estimators', Number(e.target.value || 120))} />
                  </label>
                </div>
                <label className="field">
                  <span>notas</span>
                  <textarea rows={3} value={trainConfig.notes} onChange={(e) => updateTrainConfig('notes', e.target.value)} />
                </label>
                <button className="primary" onClick={trainAssistModel} disabled={!trainImageIds.length || loading.modelsTrain}>
                  Entrenar modelo
                </button>
                {modelTrainSummary ? (
                  <div className="model-summary">
                    <strong>{modelTrainSummary.model_name || modelTrainSummary.model_id}</strong>
                    <span>{modelTrainSummary.train_samples || 0} muestras | {modelTrainSummary.image_count || 0} imagenes</span>
                  </div>
                ) : null}
                <div className="inline">
                  <button onClick={refreshAssistModels} disabled={loading.modelsList}>Refrescar modelos</button>
                  <button onClick={() => setDefaultAssistModel(selectedAssistModelId)} disabled={!selectedAssistModelId || loading.modelsList}>Usar por defecto</button>
                </div>
                {!assistModels.length ? (
                  <div className="placeholder small">Sin modelos entrenados.</div>
                ) : (
                  <div className="model-list">
                    {assistModels.map((model) => (
                      <button
                        key={model.model_id}
                        className={`model-row ${selectedAssistModelId === model.model_id ? 'selected' : ''}`}
                        onClick={() => setSelectedAssistModelId(model.model_id)}
                        disabled={loading.modelsList}
                      >
                        <strong>{model.model_name || model.model_id}</strong>
                        <span>
                          {model.model_id === defaultAssistModelId ? 'default | ' : ''}
                          {model.class_mode || '-'} | {model.image_count || 0} imagenes | {model.train_samples || 0} muestras
                        </span>
                      </button>
                    ))}
                  </div>
                )}
                <button className="bad" onClick={() => deleteAssistModel(selectedAssistModelId)} disabled={!selectedAssistModelId || loading.modelsDelete}>Eliminar seleccionado</button>
              </section>
            </>
          )}
        </aside>

        <section className="main">
          <article className={`card viewer ${workspaceTab === 'workbench' ? '' : 'hidden-panel'}`}>
              <div
                ref={editorStageRef}
                className={`stage editor-stage annot-stage mode-${viewerMode} ${isPanning ? 'is-panning' : ''}`}
                onPointerDown={onEditorPointerDown}
                onPointerMove={onEditorPointerMove}
                onPointerUp={onEditorPointerUp}
                onPointerLeave={onEditorPointerUp}
                onPointerCancel={onEditorPointerUp}
                onWheel={onEditorWheel}
              >
                <div
                  className={`editor-content ${imageUrl ? '' : 'hidden-layer'}`}
                  style={{
                    width: `${Math.max(1, imageDims.w)}px`,
                    height: `${Math.max(1, imageDims.h)}px`,
                    transform: editorRenderMetrics
                      ? `translate(${editorRenderMetrics.x}px, ${editorRenderMetrics.y}px) scale(${editorRenderMetrics.scale})`
                      : 'translate(-99999px, -99999px) scale(1)',
                  }}
                >
                  <img src={imageUrl || ''} alt="base" className="base" draggable={false} />
                  {modelPrediction?.previewUrl ? <img src={modelPrediction.previewUrl} alt="preview modelo" className="model-preview-overlay" draggable={false} /> : null}
                  <canvas ref={drawCanvasRef} className="draw" />
                  {excludeRect ? (
                    <div
                      className="exclude-rect-overlay"
                      style={{
                        left: `${excludeRect.x}px`,
                        top: `${excludeRect.y}px`,
                        width: `${excludeRect.w}px`,
                        height: `${excludeRect.h}px`,
                      }}
                    />
                  ) : null}
                </div>
                {!imageUrl ? <div className="placeholder">Carga una imagen para empezar.</div> : null}
              </div>
              <canvas ref={labelsCanvasRef} className="hidden" />
          </article>
          <article className={`card result ${workspaceTab === 'review' ? '' : 'hidden-panel'}`}>
              <h2>Comparador de resultado</h2>
              {activeRun ? (
                <>
                  <div className="run-head">
                    <strong>{activeRun.experiment_id}</strong>
                    <span>{activeRun.run_id}</span>
                  </div>
                  <div className="kpi-row">
                    <span>Group: <strong>{activeRun?.meta?.experiment?.group || '-'}</strong></span>
                    <span>Impl: <strong>{activeRun?.meta?.experiment?.implementation_status || '-'}</strong></span>
                    <span>Perfil: <strong>{activeRun?.batch_profile || activeRun?.meta?.params_effective?.__profile_name || '-'}</strong></span>
                    <span>Status: <strong>{activeRun?.meta?.run_status_level || 'success'}</strong></span>
                    <span>Score aux: <strong>{Number(activeRun?.meta?.aux_score ?? 0).toFixed(3)}</strong></span>
                  </div>
                  {activeRunExcludeMismatch ? (
                    <div className="warn">
                      El run activo no coincide con el rectangulo de exclusion actual. Ejecuta de nuevo para actualizar resultados.
                    </div>
                  ) : null}
                  {activeRunExclude ? (
                    <div className="kpi">
                      Exclusion usada en este run:
                      <strong>{` x=${activeRunExclude.x}, y=${activeRunExclude.y}, w=${activeRunExclude.w}, h=${activeRunExclude.h}`}</strong>
                      {Number(activeRunExclude.mask_nonzero_after_px || 0) > 0 ? (
                        <span>{` (warning: ${activeRunExclude.mask_nonzero_after_px}px dentro de exclusion)`}</span>
                      ) : (
                        <span>{' (mask dentro de exclusion: 0px)'}</span>
                      )}
                    </div>
                  ) : null}
                  {activeRun?.meta?.blocker_reason ? <div className="warn">{String(activeRun.meta.blocker_reason)}</div> : null}
                  <div className="img-grid">
                    <figure><figcaption>Original</figcaption><img src={b64ToDataUrl(activeRun.input_image_b64, activeRun.input_image_mime || 'image/png')} alt="orig" /></figure>
                    <figure><figcaption>Overlay</figcaption><img src={b64ToDataUrl(activeRun.overlay_b64, activeRun.overlay_mime || 'image/png')} alt="ov" /></figure>
                    <figure><figcaption>Mascara</figcaption><img src={b64ToDataUrl(activeRun.mask_b64)} alt="mask" /></figure>
                  </div>
                  <div className="result-curtain-grid">
                    <CurtainCompare
                      title="Original vs Overlay"
                      subtitle="Comparador tipo persiana"
                      baseUrl={b64ToDataUrl(activeRun.input_image_b64, activeRun.input_image_mime || 'image/png')}
                      maskUrl={b64ToDataUrl(activeRun.overlay_b64, activeRun.overlay_mime || 'image/png')}
                      maskClassName="curtain-overlay-image"
                      persistOnSourceChange={true}
                    />
                    <CurtainCompare
                      title="Original vs Mascara"
                      subtitle="Comparador tipo persiana"
                      baseUrl={b64ToDataUrl(activeRun.input_image_b64, activeRun.input_image_mime || 'image/png')}
                      maskUrl={b64ToDataUrl(activeRun.mask_b64)}
                      maskClassName="curtain-mask-binary-image"
                      persistOnSourceChange={true}
                    />
                  </div>
                  <pre className="meta">{JSON.stringify(activeRun.meta || {}, null, 2)}</pre>
                </>
              ) : <div className="placeholder">Ejecuta un experimento para ver resultados.</div>}
          </article>
          <article className={`card viewer diameter-viewer ${workspaceTab === 'diameter' ? '' : 'hidden-panel'}`}>
            <h2>Diameter Research</h2>
            <div className="viewer-toolbar">
              <button
                className={`icon-tool ${diamViewerMode === 'mark' ? 'toggle-active' : ''}`}
                onClick={() => {
                  if (diamMethodId === 'circle_square_mask_diameter') {
                    setDiamViewerMode('circle')
                    toast('warning', 'Circle-square', 'Este metodo usa solo circulo manual.')
                    return
                  }
                  setDiamViewerMode('mark')
                }}
                disabled={!imageUrl || diamMethodId === 'circle_square_mask_diameter'}
                title={diamMethodId === 'circle_square_mask_diameter' ? 'Circle-square bloquea puntos: usa Circulo' : 'Agregar o seleccionar puntos'}
              >
                Punto
              </button>
              <button className={`icon-tool ${diamViewerMode === 'pan' ? 'toggle-active' : ''}`} onClick={() => setDiamViewerMode('pan')} disabled={!imageUrl} title="Mover vista">Mano</button>
              <button className={`icon-tool ${diamViewerMode === 'manual' && diamMethodId === 'manual_dual_side_caliper' ? 'toggle-active' : ''}`} onClick={() => { setDiamMethodPanel('manual'); setDiamMethodId('manual_dual_side_caliper'); setDiamViewerMode('manual') }} disabled={!imageUrl} title="Linea que se corta con la mascara (D)">Linea mask</button>
              <button className={`icon-tool ${diamViewerMode === 'manual' && diamMethodId === 'manual_line_direct_caliper' ? 'toggle-active' : ''}`} onClick={() => { setDiamMethodPanel('manual'); setDiamMethodId('manual_line_direct_caliper'); setDiamViewerMode('manual') }} disabled={!imageUrl} title="Linea 100% manual (X)">Linea manual</button>
              <button className={`icon-tool ${diamViewerMode === 'circle' ? 'toggle-active' : ''}`} onClick={() => { setDiamMethodPanel('manual'); setDiamMethodId('circle_square_mask_diameter'); setDiamViewerMode('circle'); updateDiamRawParam('circle_square_seed_mode', 'manual_circle') }} disabled={!imageUrl} title="Dibujar circulo semilla (C)">Circulo</button>
              <span className="toolbar-sep" />
              <button className="icon-tool" onClick={undoDiameterManual} disabled={!diamManualHistory.length} title="Deshacer geometria manual (Ctrl+Z)">Undo</button>
              <button className="icon-tool" onClick={redoDiameterManual} disabled={!diamManualFuture.length} title="Rehacer geometria manual (Ctrl+Y)">Redo</button>
              <span className="toolbar-sep" />
              <button className="icon-tool" onClick={() => zoomDiameterBy(0.84)} disabled={!imageUrl} title="Zoom menos">-</button>
              <button className="icon-tool" onClick={() => zoomDiameterBy(1.2)} disabled={!imageUrl} title="Zoom mas">+</button>
              <button className="icon-tool" onClick={resetDiameterView} disabled={!imageUrl} title="Reset vista">Reset</button>
              <span className="zoom-chip">{Math.round(diamViewerZoom * 100)}%</span>
              <span className="toolbar-sep" />
              <label className="mask-toggle">
                <input
                  type="radio"
                  checked={diamMaskVisible}
                  onChange={() => setDiamMaskVisible(true)}
                  disabled={!diamVisualMaskUrl}
                />
                mascara
              </label>
              <label className="mask-toggle">
                <input
                  type="radio"
                  checked={!diamMaskVisible}
                  onChange={() => setDiamMaskVisible(false)}
                />
                sin mascara
              </label>
              <label className="compact-slider mask-opacity-control">
                transparencia
                <input
                  type="range"
                  min="0.05"
                  max="0.9"
                  step="0.05"
                  value={diamMaskOpacity}
                  onChange={(e) => setDiamMaskOpacity(Number(e.target.value || 0.38))}
                  disabled={!diamMaskVisible || !diamVisualMaskUrl}
                />
                <span>{Math.round(diamMaskOpacity * 100)}%</span>
              </label>
            </div>
            <div className="diameter-workspace">
              <div
                ref={diameterStageRef}
                className={`stage diameter-stage mode-${diamViewerMode} ${diamIsPanning ? 'is-panning' : ''} ${imageUrl ? 'has-image' : ''}`}
                onClick={onDiameterStageClick}
                onPointerDown={onDiameterPointerDown}
                onPointerMove={onDiameterPointerMove}
                onPointerUp={onDiameterPointerUp}
                onPointerCancel={onDiameterPointerUp}
                onPointerLeave={onDiameterPointerUp}
              >
                {imageUrl ? (
                  <div
                    className="diameter-content"
                    style={{
                      width: `${Math.max(1, imageDims.w)}px`,
                      height: `${Math.max(1, imageDims.h)}px`,
                      transform: diameterRenderMetrics
                        ? `translate(${diameterRenderMetrics.x}px, ${diameterRenderMetrics.y}px) scale(${diameterRenderMetrics.scale})`
                        : 'translate(-99999px, -99999px) scale(1)',
                    }}
                  >
                    <div className="diameter-raster-layer">
                      <img src={imageUrl} alt="diameter-base" draggable={false} />
                      {diamMaskVisible && diamMaskLayerUrl ? (
                        <img
                          src={diamMaskLayerUrl}
                          alt="mascara visual"
                          className="diam-mask-layer"
                          style={{ opacity: diamMaskOpacity }}
                          draggable={false}
                        />
                      ) : null}
                    </div>
                    <div className="diameter-annotation-layer">
                      {[
                        ...manualMaskLines.map((draft, lineIndex) => ({ kind: 'mask', draft, lineIndex })),
                        ...manualDirectLines.map((draft, lineIndex) => ({ kind: 'direct', draft, lineIndex })),
                        ...(manualMaskLineDraft?.start && !manualMaskLines.some((line) => lineAlmostEqual(line, manualMaskLineDraft)) ? [{ kind: 'mask', draft: manualMaskLineDraft, lineIndex: -1 }] : []),
                        ...(manualDirectLineDraft?.start && !manualDirectLines.some((line) => lineAlmostEqual(line, manualDirectLineDraft)) ? [{ kind: 'direct', draft: manualDirectLineDraft, lineIndex: -1 }] : []),
                      ].map(({ kind, draft, lineIndex }) => draft?.start ? (
                        <React.Fragment key={`manual-line-${kind}-${lineIndex}-${Number(draft.start.x).toFixed(1)}-${Number(draft.start.y).toFixed(1)}`}>
                          <svg className={`manual-line-overlay ${kind}`} viewBox={`0 0 ${Math.max(1, imageDims.w)} ${Math.max(1, imageDims.h)}`}>
                            {draft.end ? (
                              <line
                                x1={Number(draft.start.x)}
                                y1={Number(draft.start.y)}
                                x2={Number(draft.end.x)}
                                y2={Number(draft.end.y)}
                              />
                            ) : null}
                            <circle cx={Number(draft.start.x)} cy={Number(draft.start.y)} r="0.85" />
                            {draft.end ? <circle cx={Number(draft.end.x)} cy={Number(draft.end.y)} r="0.85" /> : null}
                          </svg>
                          {draft.start && draft.end ? (
                            <>
                              <button
                                className={`manual-line-hit start ${kind}`}
                                style={{
                                  left: `${Number(draft.start.x)}px`,
                                  top: `${Number(draft.start.y)}px`,
                                }}
                                onPointerDown={(e) => startDiameterManualLineEndpointDrag('start', kind, lineIndex, e)}
                                title={`Arrastrar inicio de linea ${kind === 'direct' ? 'manual' : 'mask'}`}
                                aria-label={`Arrastrar inicio de linea ${kind === 'direct' ? 'manual' : 'mask'}`}
                              />
                              <button
                                className={`manual-line-hit end ${kind}`}
                                style={{
                                  left: `${Number(draft.end.x)}px`,
                                  top: `${Number(draft.end.y)}px`,
                                }}
                                onPointerDown={(e) => startDiameterManualLineEndpointDrag('end', kind, lineIndex, e)}
                                title={`Arrastrar fin de linea ${kind === 'direct' ? 'manual' : 'mask'}`}
                                aria-label={`Arrastrar fin de linea ${kind === 'direct' ? 'manual' : 'mask'}`}
                              />
                            </>
                          ) : null}
                        </React.Fragment>
                      ) : null)}
                      {diameterManualCirclePreviews.length ? (
                        <svg className="manual-circle-overlay" viewBox={`0 0 ${Math.max(1, imageDims.w)} ${Math.max(1, imageDims.h)}`}>
                          {diameterManualCirclePreviews.map((circle) => (
                            <circle
                              key={`manual-circle-svg-${circle.geometry_id || circle.idx}`}
                              className={circle.idx === manualCircleActiveIdx ? 'selected' : ''}
                              cx={Number(circle.center.x)}
                              cy={Number(circle.center.y)}
                              r={circle.radius}
                            />
                          ))}
                        </svg>
                      ) : null}
                      {diameterManualCirclePreviews.map((circle) => (
                        <button
                          key={`manual-circle-hit-${circle.geometry_id || circle.idx}`}
                          className={`manual-circle-hit ${circle.idx === manualCircleActiveIdx ? 'selected' : ''}`}
                          style={{
                            left: `${Number(circle.center.x) - Number(circle.radius)}px`,
                            top: `${Number(circle.center.y) - Number(circle.radius)}px`,
                            width: `${Number(circle.radius) * 2}px`,
                            height: `${Number(circle.radius) * 2}px`,
                          }}
                          onPointerDown={(e) => {
                            e.stopPropagation()
                            setManualCircleSelected(true)
                            setManualCircleActiveIdx(Number(circle.idx))
                            if (Number(circle.idx) >= 0) setManualCircleDraft(manualCircles[Number(circle.idx)] || null)
                          }}
                          onPointerUp={(e) => {
                            e.stopPropagation()
                          }}
                          onClick={(e) => {
                            e.stopPropagation()
                            setManualCircleSelected(true)
                            setManualCircleActiveIdx(Number(circle.idx))
                            if (Number(circle.idx) >= 0) setManualCircleDraft(manualCircles[Number(circle.idx)] || null)
                          }}
                          title="Circulo manual seleccionado. Suprimir/Delete lo borra."
                          aria-label="Seleccionar circulo manual"
                        />
                      ))}
                      {diamPoints.map((p, idx) => (
                        <button
                          key={`${idx}-${p.x}-${p.y}-marker`}
                          className={`diam-marker ${idx === diamActivePointIdx ? 'active' : ''} ${(['circle_square_mask_diameter', 'manual_dual_side_caliper', 'manual_line_direct_caliper'].includes(diamMethodId) && (diamResults.length || diamMethodId !== 'circle_square_mask_diameter')) ? 'hide-dot' : ''}`}
                          style={{
                            left: `${(Number(p.x) / Math.max(1, imageDims.w)) * 100}%`,
                            top: `${(Number(p.y) / Math.max(1, imageDims.h)) * 100}%`,
                          }}
                          onPointerDown={(e) => {
                            e.stopPropagation()
                            if (idx !== diamActivePointIdx) void updateDiameterPoints('set_active', { active_index: idx })
                          }}
                          onPointerUp={(e) => {
                            e.stopPropagation()
                          }}
                          onClick={(e) => {
                            e.stopPropagation()
                          }}
                          title={`Punto ${idx + 1}`}
                        >
                          <span className="diam-marker-dot" />
                          <span className="diam-marker-label">{idx + 1}</span>
                        </button>
                      ))}
                      {diameterResultLines.length || diameterResultQuads.length ? (
                        <svg className="diam-result-line-overlay" viewBox={`0 0 ${Math.max(1, imageDims.w)} ${Math.max(1, imageDims.h)}`}>
                          {diameterResultQuads.map((quad) => (
                            <polygon key={quad.key} className={quad.ok ? '' : 'rejected'} points={quad.points} />
                          ))}
                          {diameterResultLines.map((line) => (
                            <line
                              key={line.key}
                              className={line.ok ? '' : 'rejected'}
                              x1={line.x1}
                              y1={line.y1}
                              x2={line.x2}
                              y2={line.y2}
                            />
                          ))}
                        </svg>
                      ) : null}
                    </div>
                  </div>
                ) : (
                  <div className="placeholder">Carga una imagen para medir diametros.</div>
                )}
              </div>

              <div className="diam-side">
                <div className="kpi-row">
                  <span>Puntos: <strong>{diamPoints.length}</strong></span>
                  <span>Vista: <strong>{diamResultsMode === 'composite' ? 'manual compuesta' : 'run'}</strong></span>
                  <span>Run: <strong>{diamActiveRunId || '-'}</strong></span>
                  <span>Metodo: <strong>{diamRunCache[diamActiveRunId]?.method_id || diamMethodId}</strong></span>
                  <span>Fuente: <strong>{diamSourceMode}</strong></span>
                  <span>Soporte: <strong>{diamRunCache[diamActiveRunId]?.meta?.prior_run_id || diamPriorRunId}</strong></span>
                  <span>OK: <strong>{diamResults.filter((r) => r.status === 'ok').length}/{diamResults.length}</strong></span>
                </div>
              </div>
            </div>

            <div className="diam-results">
              <h2>Resultados por punto</h2>
              {!diamResults.length ? (
                <div className="placeholder small">Sin resultados.</div>
              ) : (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>revision</th>
                        <th>#</th>
                        <th>x</th>
                        <th>y</th>
                        <th>method_id</th>
                        <th>quality</th>
                        <th>diameter_px</th>
                        <th>confidence</th>
                        <th>stability</th>
                        <th>mode</th>
                        <th>route</th>
                        <th>fiber</th>
                        <th>mask</th>
                        <th>caliper</th>
                        <th>ray</th>
                          <th>circle</th>
                          <th>mask px</th>
                          <th>loco r</th>
                        <th>loco sym</th>
                        <th>loco cuts</th>
                        <th>square</th>
                        <th>ellipse</th>
                        <th>methodology</th>
                        <th>context</th>
                        <th>halo</th>
                        <th>ridge</th>
                        <th>flux</th>
                        <th>control</th>
                        <th>geometry</th>
                        <th>support</th>
                        <th>coherence</th>
                        <th>length</th>
                        <th>upscale</th>
                        <th>small</th>
                        <th>recenter_px</th>
                        <th>theta_delta</th>
                        <th>profiles</th>
                        <th>orientation</th>
                        <th>status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {diamResults.map((r) => (
                        <tr
                          key={`${r.point_index}-${r.x}-${r.y}`}
                          className={`diam-result-row ${diamReviewTarget?.run_id === diamActiveRunId && Number(diamReviewTarget?.point_index) === Number(r.point_index) ? 'selected' : ''}`}
                          onClick={() => selectDiameterResultForReview(r)}
                        >
                          <td>
                            <button
                              className="small-action"
                              onClick={(e) => {
                                e.stopPropagation()
                                openDiameterPointReview(r)
                              }}
                            >
                              Revisar
                            </button>
                          </td>
                          <td>{Number(r.point_index) + 1}</td>
                          <td>{Number(r.x).toFixed(1)}</td>
                          <td>{Number(r.y).toFixed(1)}</td>
                          <td>{r.method_id || diamRunCache[diamActiveRunId]?.method_id || '-'}</td>
                          <td>{r.quality_label || '-'}</td>
                          <td>{r.diameter_px == null ? '-' : Number(r.diameter_px).toFixed(2)}</td>
                          <td>{Number(r.confidence || 0).toFixed(3)}</td>
                          <td>{r.stability_score == null ? '-' : Number(r.stability_score).toFixed(3)}</td>
                          <td>{r.measurement_mode || '-'}</td>
                          <td>{r.diameter_route || r.size_route || '-'}</td>
                          <td>{r.fiber_size_mode || '-'}</td>
                          <td>{r.mask_method ? `${r.mask_method}${r.mask_confidence == null ? '' : ` ${Number(r.mask_confidence).toFixed(2)}`}` : '-'}</td>
                          <td>{r.mask_caliper_diameter_px == null ? '-' : Number(r.mask_caliper_diameter_px).toFixed(2)}</td>
                          <td>{r.mask_raycast_diameter_px == null ? '-' : Number(r.mask_raycast_diameter_px).toFixed(2)}</td>
                          <td>{r.circle_radius_px == null ? '-' : Number(r.circle_radius_px).toFixed(1)}</td>
                          <td>{r.circle_square_measurement_mask_pixels == null ? '-' : `${r.circle_square_measurement_mask_pixels}/${r.circle_square_mask_pixels || 0}`}</td>
                          <td>{r.loco_best_radius_px == null ? '-' : Number(r.loco_best_radius_px).toFixed(1)}</td>
                          <td>{r.loco_symmetry_score == null ? '-' : Number(r.loco_symmetry_score).toFixed(3)}</td>
                          <td>{r.loco_intersection_count == null ? '-' : r.loco_intersection_count}</td>
                          <td>{r.square_samples_valid == null ? '-' : `${r.square_samples_valid}/${r.square_samples_total || 0}`}</td>
                          <td>{r.ellipse_minor_px == null ? '-' : `${Number(r.ellipse_minor_px).toFixed(2)} / ${Number(r.ellipse_major_px || 0).toFixed(2)}`}</td>
                          <td>{r.methodology_id || '-'}</td>
                          <td>{r.local_context_label || '-'}</td>
                          <td>{r.halo_status || '-'}</td>
                          <td>{r.ridge_anchor_status || '-'}</td>
                          <td>{r.flux_status || '-'}</td>
                          <td>{r.geometry_control_status || '-'}</td>
                          <td>{r.geometry_status || '-'}</td>
                          <td>{r.support_status || '-'}</td>
                          <td>{r.orientation_coherence == null ? '-' : Number(r.orientation_coherence).toFixed(3)}</td>
                          <td>{r.profile_length_effective_px == null ? '-' : Number(r.profile_length_effective_px).toFixed(1)}</td>
                          <td>{r.used_upscale ? `${r.scale_factor || 1}x` : '-'}</td>
                          <td>{r.small_diameter_suspect ? 'si' : '-'}</td>
                          <td>{r.recenter_shift_px == null ? '-' : Number(r.recenter_shift_px).toFixed(2)}</td>
                          <td>{r.orientation_delta_deg == null ? '-' : Number(r.orientation_delta_deg).toFixed(1)}</td>
                          <td>{r.valid_profiles}/{r.total_profiles}</td>
                          <td>{r?.orientation?.source || '-'}</td>
                          <td>{r.status}{r.reason ? `: ${r.reason}` : ''}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {validationCases.length ? (
                <div className="diam-results">
                  <h2>Validacion trazable</h2>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>case_id</th>
                          <th>categoria</th>
                          <th>decision</th>
                          <th>manual_px</th>
                          <th>v1_px</th>
                          <th>v1_err</th>
                          <th>v2_status</th>
                          <th>v2_px</th>
                          <th>v2_err</th>
                          <th>v2_quality</th>
                          <th>v3.1</th>
                          <th>auto</th>
                          <th>small mask</th>
                          <th>large image</th>
                          <th>circle-square</th>
                          <th>LOCO</th>
                          <th>manual</th>
                          <th>ellipse</th>
                          <th>comentario</th>
                        </tr>
                      </thead>
                      <tbody>
                        {validationCases.map((c) => {
                          const v1 = c?.runs?.hybrid_profile_diameter_v1 || {}
                          const v2 = c?.runs?.hybrid_profile_diameter_v2 || {}
                          const v31 = c?.runs?.hybrid_profile_diameter_v3_1 || {}
                          const auto = c?.runs?.hybrid_profile_diameter_v3_2_auto || {}
                          const small = c?.runs?.hybrid_profile_diameter_v3_2_small_mask || {}
                          const large = c?.runs?.hybrid_profile_diameter_v3_2_large_image || {}
                          const circle = c?.runs?.circle_square_mask_diameter || {}
                          const loco = c?.runs?.loco_circle_probe || {}
                          const manual = c?.runs?.manual_dual_side_caliper || {}
                          const ellipse = c?.runs?.ellipse_oriented_fit || {}
                          const fmtRun = (r) => {
                            const px = r?.diameter_px == null ? '-' : `${Number(r.diameter_px).toFixed(2)} px`
                            const mode = r?.measurement_mode || r?.status || '-'
                            const q = r?.quality_label ? ` | ${r.quality_label}` : ''
                            const route = r?.diameter_route || r?.size_route ? ` | ${r.diameter_route || r.size_route}` : ''
                            return `${mode}${route} | ${px}${q}`
                          }
                          return (
                            <tr key={`validation-${c.case_id}`}>
                              <td>{c.case_id}</td>
                              <td>{c.category || '-'}</td>
                              <td>{c.measurement_decision || '-'}</td>
                              <td>{c.manual_diameter_px == null ? '-' : Number(c.manual_diameter_px).toFixed(2)}</td>
                              <td>{v1.diameter_px == null ? '-' : Number(v1.diameter_px).toFixed(2)}</td>
                              <td>{v1.absolute_error_px == null ? '-' : Number(v1.absolute_error_px).toFixed(2)}</td>
                              <td>{v2.status || '-'}</td>
                              <td>{v2.diameter_px == null ? '-' : Number(v2.diameter_px).toFixed(2)}</td>
                              <td>{v2.absolute_error_px == null ? '-' : Number(v2.absolute_error_px).toFixed(2)}</td>
                              <td>{v2.quality_label || '-'}</td>
                              <td>{fmtRun(v31)}</td>
                              <td>{fmtRun(auto)}</td>
                              <td>{fmtRun(small)}</td>
                              <td>{fmtRun(large)}</td>
                              <td>{fmtRun(circle)}</td>
                              <td>{fmtRun(loco)}</td>
                              <td>{fmtRun(manual)}</td>
                              <td>{fmtRun(ellipse)}</td>
                              <td>{c.result_comment || '-'}</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : null}
              {diamActiveRunId && diamRunCache[diamActiveRunId]?.meta ? (
                <pre className="meta">{JSON.stringify(diamRunCache[diamActiveRunId].meta || {}, null, 2)}</pre>
              ) : null}
            </div>
          </article>
          <article className={`card viewer loco-viewer ${workspaceTab === 'loco' ? '' : 'hidden-panel'}`}>
            <h2>LOCO Lab</h2>
            <div className="viewer-toolbar loco-toolbar">
              <button className={`icon-tool ${locoViewerMode === 'pan' ? 'toggle-active' : ''}`} onClick={() => setLocoViewerMode('pan')} disabled={!imageUrl} title="Mover vista (M)"><ToolIcon name="hand" />Mano</button>
              <span className="toolbar-sep" />
              <button className="icon-tool" onClick={() => zoomLocoBy(0.84)} disabled={!imageUrl}>-</button>
              <button className="icon-tool" onClick={() => zoomLocoBy(1.2)} disabled={!imageUrl}>+</button>
              <button className="icon-tool" onClick={resetLocoView} disabled={!imageUrl}>Reset</button>
              <span className="zoom-chip">{Math.round(locoViewerZoom * 100)}%</span>
              <span className="toolbar-sep" />
              <select value={diamPriorRunId} onChange={(e) => setDiamPriorRunId(e.target.value)} disabled={!imageId}>
                <option value="latest">prior mask: ultimo</option>
                {runs.map((r) => (
                  <option key={`loco-prior-${r.run_id}`} value={r.run_id}>{r.run_id}</option>
                ))}
              </select>
              <button onClick={() => refreshResults()} disabled={!imageId || loading.listResults}>Refrescar soporte</button>
              <span className="toolbar-sep" />
              {LOCO_LAB_STAGES.map((stage) => (
                <button
                  key={`loco-stage-${stage.key}`}
                  className={`icon-tool ${locoLabStage === stage.key ? 'toggle-active' : ''}`}
                  onClick={() => setLocoLabStage(stage.key)}
                >
                  {stage.label}
                </button>
              ))}
              <span className="toolbar-sep" />
              <button className="primary" onClick={generateLocoLabProposals} disabled={!imageId || loading.locoLab}>Generar propuestas</button>
              <button onClick={clearLocoLab} disabled={!locoProposals.length && !locoMeasuredResults.length}>Limpiar LOCO</button>
            </div>

            <div className="loco-workspace loco-lab-workspace">
              <div
                ref={locoStageRef}
                className={`stage loco-stage mode-pan ${locoIsPanning ? 'is-panning' : ''} ${imageUrl ? 'has-image' : ''}`}
                onPointerDown={onLocoPointerDown}
                onPointerMove={onLocoPointerMove}
                onPointerUp={onLocoPointerUp}
                onPointerCancel={onLocoPointerUp}
                onPointerLeave={onLocoPointerUp}
              >
                {imageUrl ? (
                  <div
                    className="loco-content"
                    style={{
                      width: `${Math.max(1, imageDims.w)}px`,
                      height: `${Math.max(1, imageDims.h)}px`,
                      transform: locoRenderMetrics
                        ? `translate(${locoRenderMetrics.x}px, ${locoRenderMetrics.y}px) scale(${locoRenderMetrics.scale})`
                        : 'translate(-99999px, -99999px) scale(1)',
                    }}
                  >
                    <div className="diameter-raster-layer">
                      <img src={imageUrl} alt="loco-base" draggable={false} />
                      {locoLayerVisibility.mask && diamVisualMaskUrl ? (
                        <img
                          src={diamVisualMaskUrl}
                          alt="prior mask loco"
                          className="diam-mask-layer"
                          style={{ opacity: diamMaskOpacity }}
                          draggable={false}
                        />
                      ) : null}
                    </div>
                    <div className="loco-annotation-layer">
                      <svg className="loco-overlay loco-lab-overlay" viewBox={`0 0 ${Math.max(1, imageDims.w)} ${Math.max(1, imageDims.h)}`}>
                        {locoLayerVisibility.proposals && locoLabVisibleProposals.map((p) => {
                          const c = p.center_xy || [0, 0]
                          const selected = String(p.proposal_id || '') === String(locoLabSelectedProposal?.proposal_id || '')
                          const status = String(p.status || 'rejected')
                          return (
                            <circle
                              key={`loco-lab-circle-${p.proposal_id}`}
                              className={`loco-lab-circle ${status} ${selected ? 'selected' : ''}`}
                              cx={Number(c[0])}
                              cy={Number(c[1])}
                              r={Math.max(0.5, Number(p.radius_px) || 0.5)}
                            />
                          )
                        })}
                        {locoLayerVisibility.components && locoLabComponentBoxes.map((box) => (
                          <rect key={`loco-lab-component-${box.key}`} className="loco-lab-component" x={box.x} y={box.y} width={box.w} height={box.h} />
                        ))}
                        {locoLayerVisibility.intersections && locoLabSelectedProposal?.intersection_points?.map((pt, idx) => (
                          <circle key={`loco-lab-cut-${idx}`} className="loco-lab-cut" cx={Number(pt.x)} cy={Number(pt.y)} r="2" />
                        ))}
                        {locoLayerVisibility.quadrilaterals && locoLabOkMeasurements.map((m) => {
                          const verts = Array.isArray(m.quadrilateral_vertices) ? m.quadrilateral_vertices : []
                          if (verts.length < 4) return null
                          const points = verts.map((v) => `${Number(v.x)},${Number(v.y)}`).join(' ')
                          return <polygon key={`loco-lab-quad-${m.proposal_id}`} className="loco-lab-quad" points={points} />
                        })}
                        {locoLayerVisibility.diameter && locoLabOkMeasurements.map((m) => {
                          const left = m.left_edge_xy || []
                          const right = m.right_edge_xy || []
                          if (left.length < 2 || right.length < 2) return null
                          return <line key={`loco-lab-dia-${m.proposal_id}`} className="loco-lab-diameter" x1={Number(left[0])} y1={Number(left[1])} x2={Number(right[0])} y2={Number(right[1])} />
                        })}
                      </svg>
                    </div>
                  </div>
                ) : (
                  <div className="placeholder">Carga una imagen para usar LOCO Lab.</div>
                )}
              </div>

              <aside className="loco-debug-panel loco-lab-panel">
                <div className="loco-step-head">
                  <strong>{LOCO_LAB_STAGES.find((s) => s.key === locoLabStage)?.label || 'LOCO Lab'}</strong>
                  <span>{loading.locoLab ? 'procesando' : 'listo'}</span>
                </div>
                <div className="kpi-row">
                  <span>Propuestas: <strong>{locoLabActiveProposals.length}</strong></span>
                  <span>Aceptadas: <strong>{locoLabActiveProposals.filter((p) => p.status === 'accepted').length}</strong></span>
                  <span>Medidas: <strong>{locoLabOkMeasurements.length}/{locoMeasuredResults.length}</strong></span>
                  <span>Soporte: <strong>{diamPriorRunId || 'latest'}</strong></span>
                </div>

                <div className="loco-layer-grid">
                  {Object.entries({
                    mask: 'mascara',
                    components: 'componentes',
                    proposals: 'propuestas',
                    rejected: 'rechazadas',
                    intersections: 'cortes',
                    quadrilaterals: 'cuadrilateros',
                    diameter: 'diametro',
                  }).map(([key, label]) => (
                    <label key={`loco-layer-${key}`} className="mask-toggle">
                      <input type="checkbox" checked={!!locoLayerVisibility[key]} onChange={() => toggleLocoLayer(key)} />
                      {label}
                    </label>
                  ))}
                  <label className="compact-slider mask-opacity-control">
                    transparencia
                    <input type="range" min="0.05" max="0.9" step="0.05" value={diamMaskOpacity} onChange={(e) => setDiamMaskOpacity(Number(e.target.value || 0.38))} disabled={!locoLayerVisibility.mask || !diamVisualMaskUrl} />
                    <span>{Math.round(diamMaskOpacity * 100)}%</span>
                  </label>
                </div>

                {locoLabStage === 'proposals' ? (
                  <div className="loco-lab-section">
                    <h3>Propuestas</h3>
                    <label>
                      ruta
                      <select value={locoProposalMethod} onChange={(e) => setLocoProposalMethod(e.target.value)}>
                        <option value="circle_grid">Circle Proposal Grid</option>
                      </select>
                    </label>
                    <div className="diam-param-grid loco-param-grid">
                      <label>grid stride<input type="number" min="3" step="1" value={locoLabParams.grid_stride_px} onChange={(e) => updateLocoLabParam('grid_stride_px', e.target.value, true)} /></label>
                      <label>radio min<input type="number" min="1" step="1" value={locoLabParams.radius_min_px} onChange={(e) => updateLocoLabParam('radius_min_px', e.target.value)} /></label>
                      <label>radio max<input type="number" min="1" step="1" value={locoLabParams.radius_max_px} onChange={(e) => updateLocoLabParam('radius_max_px', e.target.value)} /></label>
                      <label>radio step<input type="number" min="0.5" step="0.5" value={locoLabParams.radius_step_px} onChange={(e) => updateLocoLabParam('radius_step_px', e.target.value)} /></label>
                      <label>max candidatos<input type="number" min="1" step="50" value={locoLabParams.max_candidates} onChange={(e) => updateLocoLabParam('max_candidates', e.target.value, true)} /></label>
                    </div>
                    <button className="primary" onClick={generateLocoLabProposals} disabled={!imageId || loading.locoLab}>Generar propuestas</button>
                  </div>
                ) : null}

                {locoLabStage === 'filters' ? (
                  <div className="loco-lab-section">
                    <h3>Filtros</h3>
                    <div className="diam-param-grid loco-param-grid">
                      <label>score min<input type="number" min="0" max="1" step="0.02" value={locoLabParams.min_score} onChange={(e) => updateLocoLabParam('min_score', e.target.value)} /></label>
                      <label>ratio mascara<input type="number" min="0" max="1" step="0.02" value={locoLabParams.mask_required_ratio} onChange={(e) => updateLocoLabParam('mask_required_ratio', e.target.value)} /></label>
                      <label>max cortes<input type="number" min="1" step="1" value={locoLabParams.max_intersections} onChange={(e) => updateLocoLabParam('max_intersections', e.target.value, true)} /></label>
                      <label className="check-row"><input type="checkbox" checked={Boolean(locoLabParams.require_four_cuts)} onChange={(e) => updateLocoLabRawParam('require_four_cuts', e.target.checked)} /> exigir 4 cortes</label>
                    </div>
                    <button className="primary" onClick={filterLocoLabProposals} disabled={!locoProposals.length || loading.locoLab}>Aplicar filtros</button>
                  </div>
                ) : null}

                {locoLabStage === 'circles' ? (
                  <div className="loco-lab-section">
                    <h3>Circulos</h3>
                    {!locoLabActiveProposals.length ? (
                      <div className="placeholder small">Sin propuestas.</div>
                    ) : (
                      <div className="loco-candidate-list">
                        {locoLabActiveProposals.slice(0, 90).map((p) => (
                          <button
                            key={`loco-lab-row-${p.proposal_id}`}
                            className={String(p.proposal_id) === String(locoLabSelectedProposal?.proposal_id) ? 'selected' : ''}
                            onClick={() => setLocoSelectedProposalId(p.proposal_id)}
                          >
                            <span>{p.method}</span>
                            <strong>{Number(p.score || 0).toFixed(3)}</strong>
                            <em>{p.intersection_count ?? 0} cortes</em>
                          </button>
                        ))}
                      </div>
                    )}
                    {locoLabSelectedProposal ? (
                      <div className="loco-result-card">
                        <h3>Seleccionado</h3>
                        <div className="kpi">ID: <strong>{locoLabSelectedProposal.proposal_id}</strong></div>
                        <div className="kpi">Radio: <strong>{Number(locoLabSelectedProposal.radius_px || 0).toFixed(1)} px</strong></div>
                        <div className="kpi">Ratio mascara: <strong>{Number(locoLabSelectedProposal.mask_ratio || 0).toFixed(3)}</strong></div>
                        <div className="kpi">Simetria: <strong>{Number(locoLabSelectedProposal.symmetry_score || 0).toFixed(3)}</strong></div>
                        <div className="kpi">Razon: <strong>{locoLabSelectedProposal.reason || '-'}</strong></div>
                      </div>
                    ) : null}
                  </div>
                ) : null}

                {locoLabStage === 'measure' ? (
                  <div className="loco-lab-section">
                    <h3>Medicion</h3>
                    <div className="diam-param-grid loco-param-grid">
                      <label>limite medir<input type="number" min="1" step="10" value={locoLabParams.measure_limit} onChange={(e) => updateLocoLabParam('measure_limit', e.target.value, true)} /></label>
                    </div>
                    <div className="inline">
                      <button className="primary" onClick={measureLocoLabProposals} disabled={!locoLabActiveProposals.length || loading.locoLab}>Medir aceptadas</button>
                      <button onClick={saveLocoLabRun} disabled={(!locoLabActiveProposals.length && !locoMeasuredResults.length) || loading.locoLab}>Guardar run LOCO</button>
                    </div>
                    {!locoMeasuredResults.length ? (
                      <div className="placeholder small">Sin mediciones.</div>
                    ) : (
                      <div className="table-wrap loco-lab-table">
                        <table>
                          <thead><tr><th>id</th><th>diam</th><th>score</th><th>estado</th></tr></thead>
                          <tbody>
                            {locoMeasuredResults.slice(0, 80).map((m) => (
                              <tr key={`loco-measure-${m.proposal_id}`} className={m.status === 'ok' ? '' : 'rejected-row'}>
                                <td>{m.proposal_id}</td>
                                <td>{m.diameter_px == null ? '-' : Number(m.diameter_px).toFixed(2)}</td>
                                <td>{Number(m.score || m.confidence || 0).toFixed(3)}</td>
                                <td>{m.status}{m.reason ? `: ${m.reason}` : ''}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                ) : null}

                {locoLabStage === 'evaluate' ? (
                  <div className="loco-lab-section">
                    <h3>Evaluacion</h3>
                    <div className="inline">
                      <button className="primary" onClick={evaluateLocoLab} disabled={!imageId || loading.locoLab}>Evaluar</button>
                      <button onClick={saveLocoLabRun} disabled={(!locoLabActiveProposals.length && !locoMeasuredResults.length) || loading.locoLab}>Guardar run LOCO</button>
                    </div>
                    <div className="loco-result-card">
                      <div className="kpi">Propuestas: <strong>{locoEvaluation?.proposal_count ?? locoLabActiveProposals.length}</strong></div>
                      <div className="kpi">Aceptadas: <strong>{locoEvaluation?.accepted_count ?? locoLabActiveProposals.filter((p) => p.status === 'accepted').length}</strong></div>
                      <div className="kpi">Mediciones OK: <strong>{locoEvaluation?.measurement_ok_count ?? locoLabOkMeasurements.length}</strong></div>
                      <div className="kpi">Mediana diam: <strong>{locoEvaluation?.diameter_median_px == null ? '-' : `${Number(locoEvaluation.diameter_median_px).toFixed(2)} px`}</strong></div>
                    </div>
                    {locoEvaluation?.by_method ? <pre className="meta">{JSON.stringify(locoEvaluation.by_method, null, 2)}</pre> : null}
                  </div>
                ) : null}
              </aside>
            </div>
          </article>
          <article className={`card viewer loco-viewer ${workspaceTab === 'locoDataset' ? '' : 'hidden-panel'}`}>
            <h2>Generate Dataset</h2>
            <div className="viewer-toolbar loco-toolbar">
              <button className={`icon-tool ${locoDatasetTool === 'circle' ? 'toggle-active' : ''}`} onClick={() => setLocoDatasetTool('circle')} disabled={!imageUrl} title="Dibujar circulo"><ToolIcon name="circle" />Circulo</button>
              <button className={`icon-tool ${locoDatasetTool === 'select' ? 'toggle-active' : ''}`} onClick={() => setLocoDatasetTool('select')} disabled={!imageUrl} title="Seleccionar circulo"><ToolIcon name="center" />Seleccionar</button>
              <button className={`icon-tool ${locoDatasetTool === 'pan' ? 'toggle-active' : ''}`} onClick={() => setLocoDatasetTool('pan')} disabled={!imageUrl} title="Mover vista"><ToolIcon name="hand" />Mano</button>
              <span className="toolbar-sep" />
              <button className="icon-tool" onClick={() => zoomLocoDatasetBy(0.84)} disabled={!imageUrl}>-</button>
              <button className="icon-tool" onClick={() => zoomLocoDatasetBy(1.2)} disabled={!imageUrl}>+</button>
              <button className="icon-tool" onClick={resetLocoDatasetView} disabled={!imageUrl}>Reset</button>
              <span className="zoom-chip">{Math.round(locoDatasetZoom * 100)}%</span>
              <span className="toolbar-sep" />
              <label className="compact-slider mask-opacity-control">
                mascara
                <input type="range" min="0.05" max="0.9" step="0.05" value={diamMaskOpacity} onChange={(e) => setDiamMaskOpacity(Number(e.target.value || 0.38))} disabled={!diamVisualMaskUrl} />
                <span>{Math.round(diamMaskOpacity * 100)}%</span>
              </label>
              <span className="toolbar-sep" />
              <button onClick={previewLocoDatasetFeatures} disabled={!locoDatasetCircles.length || loading.locoDataset}>Features</button>
              <button className="primary" onClick={saveLocoDataset} disabled={!locoDatasetCircles.length || loading.locoDataset}>Generate Dataset</button>
              <button onClick={clearLocoDatasetCanvas} disabled={!locoDatasetCircles.length && !locoDatasetDraftCircle}>Limpiar imagen</button>
              <button className="danger" onClick={cleanLegacyInvalid} disabled={loading.locoDataset}>Borrar invalid legacy</button>
            </div>

            <div className="loco-workspace loco-lab-workspace">
              <div
                ref={locoDatasetStageRef}
                className={`stage loco-stage mode-${locoDatasetTool === 'pan' ? 'pan' : 'circle'} ${locoDatasetIsPanning ? 'is-panning' : ''} ${imageUrl ? 'has-image' : ''}`}
                onPointerDown={onLocoDatasetPointerDown}
                onPointerMove={onLocoDatasetPointerMove}
                onPointerUp={onLocoDatasetPointerUp}
                onPointerCancel={onLocoDatasetPointerUp}
                onPointerLeave={onLocoDatasetPointerUp}
              >
                {imageUrl ? (
                  <div
                    className="loco-content"
                    style={{
                      width: `${Math.max(1, imageDims.w)}px`,
                      height: `${Math.max(1, imageDims.h)}px`,
                      transform: locoDatasetRenderMetrics
                        ? `translate(${locoDatasetRenderMetrics.x}px, ${locoDatasetRenderMetrics.y}px) scale(${locoDatasetRenderMetrics.scale})`
                        : 'translate(-99999px, -99999px) scale(1)',
                    }}
                  >
                    <div className="diameter-raster-layer">
                      <img src={imageUrl} alt="dataset-base" draggable={false} />
                      {diamVisualMaskUrl ? (
                        <img
                          src={diamVisualMaskUrl}
                          alt="prior mask dataset"
                          className="diam-mask-layer"
                          style={{ opacity: diamMaskOpacity }}
                          draggable={false}
                        />
                      ) : null}
                    </div>
                    <div className="loco-annotation-layer">
                      <svg className="loco-overlay loco-dataset-overlay" viewBox={`0 0 ${Math.max(1, imageDims.w)} ${Math.max(1, imageDims.h)}`}>
                        {locoDatasetCircles.map((c) => {
                          const selected = String(c.candidate_id) === String(locoDatasetSelectedId)
                          return (
                            <g key={`dataset-circle-${c.candidate_id}`}>
                              <circle
                                className={`loco-dataset-circle ${c.label || 'invalid_other'} ${selected ? 'selected' : ''}`}
                                cx={Number(c.center_x)}
                                cy={Number(c.center_y)}
                                r={Math.max(1, Number(c.radius_px) || 1)}
                              />
                              <circle
                                className="loco-dataset-hit"
                                cx={Number(c.center_x)}
                                cy={Number(c.center_y)}
                                r={Math.max(6, Number(c.radius_px) || 6)}
                                onPointerDown={(e) => beginMoveLocoDatasetCircle(e, c)}
                                onClick={(e) => {
                                  e.stopPropagation()
                                  setLocoDatasetSelectedId(String(c.candidate_id))
                                  if (locoDatasetTool !== 'circle') setLocoDatasetTool('select')
                                }}
                              />
                            </g>
                          )
                        })}
                        {locoDatasetDraftCircle ? (
                          <circle
                            className="loco-dataset-draft"
                            cx={Number(locoDatasetDraftCircle.center_x)}
                            cy={Number(locoDatasetDraftCircle.center_y)}
                            r={Math.max(1, Number(locoDatasetDraftCircle.radius_px) || 1)}
                          />
                        ) : null}
                      </svg>
                    </div>
                  </div>
                ) : (
                  <div className="placeholder">Carga una imagen para crear dataset LOCO.</div>
                )}
              </div>

              <aside className="loco-debug-panel loco-lab-panel">
                <div className="loco-step-head">
                  <strong>Dataset bruto</strong>
                  <span>{loading.locoDataset ? 'guardando' : 'listo'}</span>
                </div>
                <div className="kpi-row">
                  <span>Total: <strong>{locoDatasetCounts.total}</strong></span>
                  <span>Valid: <strong>{locoDatasetCounts.valid}</strong></span>
                  <span>Crossing: <strong>{locoDatasetCounts.invalidCrossing}</strong></span>
                  <span>Other: <strong>{locoDatasetCounts.invalidOther}</strong></span>
                  <span>Soporte: <strong>{diamPriorRunId || 'latest'}</strong></span>
                  <span>Dataset: <strong>main</strong></span>
                  <span>Siguiente: <strong>{locoDatasetDefaultLabel === 'valid' ? 'Valid' : locoDatasetDefaultLabel === 'invalid_crossing' ? 'Crossing' : 'Other invalid'}</strong></span>
                </div>
                {locoDatasetSelectedCircle ? (
                  <div className="loco-lab-section">
                    <h3>Seleccionado</h3>
                    <div className={`inline dataset-label-actions ${locoDatasetSelectedCircle.label || 'invalid_other'}`}>
                      <button className={`dataset-valid ${locoDatasetSelectedCircle.label === 'valid' ? 'active' : ''}`} onClick={() => updateLocoDatasetCircle(locoDatasetSelectedCircle.candidate_id, { label: 'valid' })}>Valid</button>
                      <button className={`dataset-invalid-crossing ${locoDatasetSelectedCircle.label === 'invalid_crossing' ? 'active' : ''}`} onClick={() => updateLocoDatasetCircle(locoDatasetSelectedCircle.candidate_id, { label: 'invalid_crossing' })}>Crossing</button>
                      <button className={`dataset-invalid-other ${locoDatasetSelectedCircle.label === 'invalid_other' ? 'active' : ''}`} onClick={() => updateLocoDatasetCircle(locoDatasetSelectedCircle.candidate_id, { label: 'invalid_other' })}>Other invalid</button>
                      <button className="dataset-delete" onClick={deleteSelectedLocoDatasetCircle}>Eliminar</button>
                    </div>
                    <label className="field">
                      <span>radio px</span>
                      <input type="number" min="1" step="0.5" value={Number(locoDatasetSelectedCircle.radius_px).toFixed(1)} onChange={(e) => updateLocoDatasetCircle(locoDatasetSelectedCircle.candidate_id, { radius_px: Math.max(1, Number(e.target.value || 1)) })} />
                    </label>
                  </div>
                ) : null}
                <div className="loco-lab-section">
                  <h3>Circulos anotados</h3>
                  <div className="loco-candidate-list">
                    {locoDatasetCircles.length ? locoDatasetCircles.map((c, idx) => (
                      <button
                        key={`dataset-row-${c.candidate_id}`}
                        className={String(c.candidate_id) === String(locoDatasetSelectedId) ? 'selected' : ''}
                        onClick={() => { setLocoDatasetSelectedId(String(c.candidate_id)); if (locoDatasetTool !== 'circle') setLocoDatasetTool('select') }}
                      >
                        <span>{idx + 1}</span>
                        <strong>{c.label}</strong>
                        <em>r {Number(c.radius_px).toFixed(1)}</em>
                      </button>
                    )) : <div className="placeholder small">Sin circulos.</div>}
                  </div>
                </div>
                {locoDatasetFeatures.length ? (
                  <div className="loco-lab-section">
                    <h3>Features preview</h3>
                    <div className="table-wrap loco-lab-table">
                      <table>
                        <thead><tr><th>id</th><th>cortes</th><th>area</th><th>sim</th></tr></thead>
                        <tbody>
                          {locoDatasetFeatures.slice(0, 80).map((item) => (
                            <tr key={`dataset-feature-${item.candidate_id}`}>
                              <td>{item.candidate_id}</td>
                              <td>{item.features?.n_cortes ?? '-'}</td>
                              <td>{item.features?.area_mask_ratio == null ? '-' : Number(item.features.area_mask_ratio).toFixed(3)}</td>
                              <td>{item.features?.simetria_cuadrilatero == null ? '-' : Number(item.features.simetria_cuadrilatero).toFixed(3)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : null}
                {locoDatasetSaveInfo ? (
                  <div className="loco-result-card">
                    <strong>{locoDatasetSaveInfo.dataset_id}</strong>
                    <span>{locoDatasetSaveInfo.candidate_count} ejemplos</span>
                    <p className="small">{locoDatasetSaveInfo.dataset_dir}</p>
                  </div>
                ) : null}
              </aside>
            </div>
          </article>
          <article className={`card viewer loco-viewer ${workspaceTab === 'locoAugment' ? '' : 'hidden-panel'}`}>
            <h2>Augmentation</h2>
            <div className="viewer-toolbar loco-toolbar">
              <label className="field inline-field">
                <span>bloque</span>
                <select value={locoAugBlockType} onChange={(e) => setLocoAugBlockType(e.target.value)}>
                  <option value="rotate">Rotate</option>
                  <option value="flip">Flip</option>
                  <option value="morphology">Morphology</option>
                  <option value="perturb">Binary perturbation</option>
                  <option value="resize_method">Resize method</option>
                  <option value="resolution">Resolution simulation</option>
                </select>
              </label>
              <button onClick={() => setLocoAugPipeline((prev) => [...prev, defaultLocoAugBlock(locoAugBlockType)])}>Agregar bloque</button>
              <span className="toolbar-sep" />
              <button onClick={previewLocoAugmentation} disabled={loading.locoAugment || !locoAugPipeline.length}>Preview selected</button>
              <button className="primary" onClick={applyLocoAugmentation} disabled={loading.locoAugment || !locoAugPipeline.length || !locoAugItems.length}>Apply to all dataset</button>
              <button onClick={clearLocoAugmented} disabled={loading.locoAugment || !(locoAugCounts.augmented_total > 0)}>Clear augmented</button>
              <span className="zoom-chip">{Math.max(1, Number(locoAugPasses) || 1)} pasadas/img</span>
            </div>

            <div className="augment-workspace">
              <aside className="augment-panel">
                <section className="loco-lab-section">
                  <h3>Pipeline</h3>
                  <div className="augment-block-list">
                    {locoAugPipeline.map((block, idx) => (
                      <div className="augment-block" key={block.id}>
                        <div className="augment-block-head">
                          <strong>{idx + 1}. {locoAugBlockLabel(block.type)}</strong>
                          <span>
                            <button onClick={() => moveLocoAugBlock(block.id, -1)} disabled={idx === 0}>â†‘</button>
                            <button onClick={() => moveLocoAugBlock(block.id, 1)} disabled={idx === locoAugPipeline.length - 1}>â†“</button>
                            <button onClick={() => removeLocoAugBlock(block.id)}>Eliminar</button>
                          </span>
                        </div>
                        <label className="field">
                          <span>probabilidad</span>
                          <input
                            type="number"
                            min="0"
                            max="1"
                            step="0.05"
                            value={block.params?.probability ?? 1}
                            onChange={(e) => updateLocoAugBlock(block.id, { params: { ...(block.params || {}), probability: clamp(Number(e.target.value || 0), 0, 1) } })}
                          />
                        </label>
                        {block.type === 'rotate' ? (
                          <label className="field"><span>random angles</span><input value={block.params?.angles || ''} onChange={(e) => updateLocoAugBlock(block.id, { params: { ...(block.params || {}), angles: e.target.value } })} /></label>
                        ) : null}
                        {block.type === 'flip' ? (
                          <label className="field"><span>modes</span><input value={block.params?.modes || ''} onChange={(e) => updateLocoAugBlock(block.id, { params: { ...(block.params || {}), modes: e.target.value } })} /></label>
                        ) : null}
                        {block.type === 'morphology' ? (
                          <label className="field"><span>ops</span><input value={block.params?.ops || ''} onChange={(e) => updateLocoAugBlock(block.id, { params: { ...(block.params || {}), ops: e.target.value } })} /></label>
                        ) : null}
                        {block.type === 'perturb' ? (
                          <>
                            <label className="field"><span>amount min</span><input type="number" min="0" max="0.08" step="0.005" value={block.params?.amount_min ?? 0.005} onChange={(e) => updateLocoAugBlock(block.id, { params: { ...(block.params || {}), amount_min: Number(e.target.value || 0.005) } })} /></label>
                            <label className="field"><span>amount max</span><input type="number" min="0" max="0.08" step="0.005" value={block.params?.amount_max ?? 0.02} onChange={(e) => updateLocoAugBlock(block.id, { params: { ...(block.params || {}), amount_max: Number(e.target.value || 0.02) } })} /></label>
                          </>
                        ) : null}
                        {block.type === 'resize_method' ? (
                          <>
                            <label className="field"><span>methods</span><input value={block.params?.methods || ''} onChange={(e) => updateLocoAugBlock(block.id, { params: { ...(block.params || {}), methods: e.target.value } })} /></label>
                            <label className="field"><span>size min</span><input type="number" min="24" max="63" value={block.params?.target_size_min ?? 40} onChange={(e) => updateLocoAugBlock(block.id, { params: { ...(block.params || {}), target_size_min: Number(e.target.value || 40) } })} /></label>
                            <label className="field"><span>size max</span><input type="number" min="24" max="63" value={block.params?.target_size_max ?? 56} onChange={(e) => updateLocoAugBlock(block.id, { params: { ...(block.params || {}), target_size_max: Number(e.target.value || 56) } })} /></label>
                          </>
                        ) : null}
                        {block.type === 'resolution' ? (
                          <label className="field"><span>sizes</span><input value={block.params?.sizes || ''} onChange={(e) => updateLocoAugBlock(block.id, { params: { ...(block.params || {}), sizes: e.target.value } })} /></label>
                        ) : null}
                      </div>
                    ))}
                    {!locoAugPipeline.length ? <div className="placeholder small">Sin bloques.</div> : null}
                  </div>
                </section>

                <section className="loco-lab-section">
                  <h3>Dataset main</h3>
                  <div className="kpi-row">
                    <span>Raw <strong>{locoAugCounts.total || 0}</strong></span>
                    <span>Valid <strong>{locoAugCounts.valid || 0}</strong></span>
                    <span>Invalid <strong>{locoAugCounts.invalid || 0}</strong></span>
                    <span>Aug <strong>{locoAugCounts.augmented_total || 0}</strong></span>
                    <span>Pasadas <strong>{locoAugPasses}</strong></span>
                  </div>
                  <p className="small">Preview usa la selecciÃ³n. Apply procesa todo `main/valid` y `main/invalid`.</p>
                </section>
                <section className="loco-lab-section">
                  <h3>Fuente</h3>
                  <label className="field">
                    <span>clase</span>
                    <select value={locoAugLabelFilter} onChange={(e) => setLocoAugLabelFilter(e.target.value)}>
                      <option value="all">todas</option>
                      <option value="valid">valid</option>
                      <option value="invalid_crossing">crossing</option>
                      <option value="invalid_other">other invalid</option>
                    </select>
                  </label>
                  <div className="kpi-row">
                    <span>Visibles <strong>{locoAugFilteredItems.length}</strong></span>
                    <span>Seleccionados <strong>{locoAugSelectedCount}</strong></span>
                  </div>
                  <div className="augment-source-grid compact">
                    {locoAugFilteredItems.map((item) => (
                      <button
                        key={`aug-thumb-${item.item_id}`}
                        className={`augment-source-thumb ${item.label} ${locoAugSelected[item.item_id] ? 'selected' : ''}`}
                        onClick={() => setLocoAugSelected((prev) => ({ ...prev, [item.item_id]: !prev[item.item_id] }))}
                        title={`${item.label} | ${item.candidate_id}`}
                      >
                        {item.source_b64 ? <img src={b64ToDataUrl(item.source_b64, 'image/png')} alt={item.candidate_id} /> : <span className="thumb-placeholder" />}
                        <strong>{item.label}</strong>
                        <em>{item.candidate_id}</em>
                      </button>
                    ))}
                    {!locoAugFilteredItems.length ? <div className="placeholder small">No hay ejemplos en main.</div> : null}
                  </div>
                </section>
              </aside>

              <section className="augment-preview">
                {locoAugPreview.length ? (
                  locoAugPreview.map((entry) => (
                    <div className="augment-preview-group" key={entry.item?.item_id}>
                      <div className="augment-source-card">
                        <img src={b64ToDataUrl(entry.source_b64, 'image/png')} alt="source" />
                        <strong>{entry.item?.label}</strong>
                        <span>{entry.item?.candidate_id}</span>
                      </div>
                      {(entry.variants || []).map((variant, idx) => (
                        <figure key={`${entry.item?.item_id}-${idx}`} className="augment-card">
                          <img src={b64ToDataUrl(variant.image_b64, 'image/png')} alt="augmented" />
                          <figcaption>{(variant.chain || []).filter((x) => x !== 'source').join(' â†’ ') || 'source'}</figcaption>
                        </figure>
                      ))}
                    </div>
                  ))
                ) : (
                  <div className="placeholder">Selecciona una muestra y toca Preview selected.</div>
                )}
              </section>
            </div>
          </article>
          <article className={`card viewer models-viewer ${workspaceTab === 'locoTraining' ? '' : 'hidden-panel'}`}>
            <h2>Training</h2>
            {locoTrainingResult ? (
              <div className="training-workspace">
                {/* â”€â”€ Section 1: Binary model comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
                <section className="training-section">
                  <h3>Comparador (binario)</h3>
                  <div className="table-wrap model-table">
                    <table>
                      <thead>
                        <tr>
                          <th>model</th>
                          <th>precision_valid</th>
                          <th>recall_valid</th>
                          <th>f1_valid</th>
                          <th>PR-AUC</th>
                          <th>accuracy</th>
                          <th>balanced_accuracy</th>
                          <th>status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {locoTrainingMetrics.map((row) => (
                          <tr key={`train-metric-${row.model_id}`}>
                            <td>{row.model}</td>
                            <td className={locoTrainingBest.precision_valid === row.model_id ? 'best-cell' : ''}>{row.precision_valid == null ? '-' : Number(row.precision_valid).toFixed(3)}</td>
                            <td>{row.recall_valid == null ? '-' : Number(row.recall_valid).toFixed(3)}</td>
                            <td className={locoTrainingBest.f1_valid === row.model_id ? 'best-cell' : ''}>{row.f1_valid == null ? '-' : Number(row.f1_valid).toFixed(3)}</td>
                            <td className={locoTrainingBest.pr_auc === row.model_id ? 'best-cell' : ''}>{row.pr_auc == null ? '-' : Number(row.pr_auc).toFixed(3)}</td>
                            <td>{row.accuracy == null ? '-' : Number(row.accuracy).toFixed(3)}</td>
                            <td>{row.balanced_accuracy == null ? '-' : Number(row.balanced_accuracy).toFixed(3)}</td>
                            <td>{row.status}{row.reason ? `: ${row.reason}` : ''}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>

                {/* â”€â”€ Section 2: Binary confusion matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
                <section className="training-section">
                  <h3>Matriz de confusiÃ³n (binario)</h3>
                  {locoTrainingResult.confusion_matrices?.[locoTrainingSelectedModel]?.status === 'unavailable' ? (
                    <p className="small">{locoTrainingResult.confusion_matrices[locoTrainingSelectedModel].reason}</p>
                  ) : (
                    <div className="confusion-grid">
                      <span></span><strong>Pred invalid</strong><strong>Pred valid</strong>
                      <strong>Real invalid</strong><span>{locoTrainingResult.confusion_matrices?.[locoTrainingSelectedModel]?.tn ?? '-'}</span><span className="bad-cell">FP {locoTrainingResult.confusion_matrices?.[locoTrainingSelectedModel]?.fp ?? '-'}</span>
                      <strong>Real valid</strong><span className="bad-cell">FN {locoTrainingResult.confusion_matrices?.[locoTrainingSelectedModel]?.fn ?? '-'}</span><span>{locoTrainingResult.confusion_matrices?.[locoTrainingSelectedModel]?.tp ?? '-'}</span>
                    </div>
                  )}
                </section>

                {/* â”€â”€ Section 3: Multiclass metrics summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
                {Array.isArray(locoTrainingResult?.multiclass_metrics_summary) && locoTrainingResult.multiclass_metrics_summary.length > 0 ? (
                <section className="training-section">
                  <h3>MÃ©tricas multiclase (valid / crossing / other)</h3>
                  <div className="table-wrap model-table">
                    <table>
                      <thead><tr><th>modelo</th><th>accuracy</th><th>P_valid</th><th>R_valid</th><th>F1_valid</th><th>P_cross</th><th>R_cross</th><th>F1_cross</th><th>P_other</th><th>R_other</th><th>F1_other</th><th>test</th><th>train</th></tr></thead>
                      <tbody>
                        {(Array.isArray(locoTrainingResult.multiclass_metrics_summary) ? locoTrainingResult.multiclass_metrics_summary : []).map((row) => (
                          <tr key={`mc-${row.model_id}`}>
                            <td>{row.model}</td>
                            <td>{row.status === 'ok' ? Number(row.accuracy).toFixed(3) : row.reason || row.status}</td>
                            <td>{row.status === 'ok' ? Number(row.precision_valid).toFixed(3) : '-'}</td>
                            <td>{row.status === 'ok' ? Number(row.recall_valid).toFixed(3) : '-'}</td>
                            <td>{row.status === 'ok' ? Number(row.f1_valid).toFixed(3) : '-'}</td>
                            <td>{row.status === 'ok' ? Number(row.precision_crossing).toFixed(3) : '-'}</td>
                            <td>{row.status === 'ok' ? Number(row.recall_crossing).toFixed(3) : '-'}</td>
                            <td>{row.status === 'ok' ? Number(row.f1_crossing).toFixed(3) : '-'}</td>
                            <td>{row.status === 'ok' ? Number(row.precision_other).toFixed(3) : '-'}</td>
                            <td>{row.status === 'ok' ? Number(row.recall_other).toFixed(3) : '-'}</td>
                            <td>{row.status === 'ok' ? Number(row.f1_other).toFixed(3) : '-'}</td>
                            <td>{row.test_samples ?? '-'}</td>
                            <td>{row.train_samples ?? '-'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
                ) : null}

                {/* â”€â”€ Section 4: Multiclass confusion matrix 3x3 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
                {Array.isArray(locoTrainingResult?.multiclass_confusion_matrices) === false && locoTrainingResult?.multiclass_confusion_matrices?.[locoTrainingSelectedModel] && typeof locoTrainingResult.multiclass_confusion_matrices[locoTrainingSelectedModel] !== 'string' ? (
                <section className="training-section">
                  <h3>Matriz de confusiÃ³n multiclase 3Ã—3</h3>
                  {(() => {
                    const cm = locoTrainingResult.multiclass_confusion_matrices[locoTrainingSelectedModel]
                    if (!cm || cm.status === 'unavailable') return <p className="small">{cm?.reason || 'N/A'}</p>
                    const labels = ['valid', 'crossing', 'other']
                    return (
                      <div className="confusion-grid confusion-3x3">
                        <span></span><strong>Pred valid</strong><strong>Pred crossing</strong><strong>Pred other</strong>
                        {[0,1,2].map((r) => (
                          <React.Fragment key={`cm3-r${r}`}>
                            <strong>Real {labels[r]}</strong>
                            {[0,1,2].map((c) => {
                              const val = Array.isArray(cm) ? cm[r]?.[c] : (r === c ? cm?.tp : 0)
                              const isCorrect = r === c
                              return <span key={`cm3-${r}-${c}`} className={isCorrect ? '' : 'bad-cell'}>{val ?? '-'}</span>
                            })}
                          </React.Fragment>
                        ))}
                      </div>
                    )
                  })()}
                </section>
                ) : null}

                {/* â”€â”€ Section 5: Per-class metrics (long format) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
                {Array.isArray(locoTrainingResult?.multiclass_class_metrics) && locoTrainingResult.multiclass_class_metrics.length > 0 ? (
                <section className="training-section">
                  <h3>MÃ©tricas por clase (formato largo)</h3>
                  <div className="table-wrap model-table">
                    <table>
                      <thead><tr><th>model</th><th>class</th><th>precision</th><th>recall</th><th>f1</th><th>support</th></tr></thead>
                      <tbody>
                        {locoTrainingResult.multiclass_class_metrics.map((row, i) => (
                          <tr key={`mc-class-${i}`}>
                            <td>{row.model}</td>
                            <td>{row.class}</td>
                            <td>{row.precision == null ? '-' : Number(row.precision).toFixed(3)}</td>
                            <td>{row.recall == null ? '-' : Number(row.recall).toFixed(3)}</td>
                            <td>{row.f1 == null ? '-' : Number(row.f1).toFixed(3)}</td>
                            <td>{row.support ?? '-'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
                ) : null}

                {/* â”€â”€ Section 6: Crossing rejection metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
                {Array.isArray(locoTrainingResult?.crossing_metrics) && locoTrainingResult.crossing_metrics.length > 0 ? (
                <section className="training-section">
                  <h3>Crossing rejection metrics</h3>
                  <div className="table-wrap model-table">
                    <table>
                      <thead><tr><th>model</th><th>crossing_total</th><th>crossing_rejected</th><th>accepted_as_valid</th><th>false_accept_rate</th><th>rejection_rate</th></tr></thead>
                      <tbody>
                        {locoTrainingResult.crossing_metrics.map((row, i) => (
                          <tr key={`cross-metrics-${i}`}>
                            <td>{row.model}</td>
                            <td>{row.crossing_total}</td>
                            <td>{row.crossing_rejected}</td>
                            <td className="bad-cell">{row.crossing_accepted_as_valid}</td>
                            <td className={Number(row.crossing_false_accept_rate || 1) > 0.1 ? 'bad-cell' : ''}>{row.crossing_false_accept_rate == null ? '-' : Number(row.crossing_false_accept_rate).toFixed(4)}</td>
                            <td>{row.crossing_rejection_rate == null ? '-' : Number(row.crossing_rejection_rate).toFixed(4)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
                ) : null}

                {/* â”€â”€ Section 7: Combined decision thresholds grid â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
                {Array.isArray(locoTrainingResult?.combined_decision_thresholds) && locoTrainingResult.combined_decision_thresholds.length > 0 ? (
                <section className="training-section">
                  <h3>Combined decision thresholds</h3>
                  <p className="small">valid_threshold (binary) &times; crossing_threshold (multiclass) &mdash; combined decision: binary_score {'>='} vt AND crossing_prob {'<='} ct</p>
                  <div className="table-wrap model-table">
                    <table>
                      <thead><tr><th>model</th><th>vt</th><th>ct</th><th>P_valid</th><th>R_valid</th><th>F1_valid</th><th>accuracy</th><th>crossing_accepted</th><th>crossing_total</th><th>crossing_accept_rate</th></tr></thead>
                      <tbody>
                        {locoTrainingResult.combined_decision_thresholds
                          .filter((r) => r.model_id === locoTrainingSelectedModel || !locoTrainingSelectedModel)
                          .map((row, i) => (
                          <tr key={`combined-th-${i}`}>
                            <td>{row.model}</td>
                            <td>{Number(row.valid_threshold).toFixed(2)}</td>
                            <td>{Number(row.crossing_threshold).toFixed(2)}</td>
                            <td>{row.precision_valid == null ? '-' : Number(row.precision_valid).toFixed(3)}</td>
                            <td>{row.recall_valid == null ? '-' : Number(row.recall_valid).toFixed(3)}</td>
                            <td>{row.f1_valid == null ? '-' : Number(row.f1_valid).toFixed(3)}</td>
                            <td>{row.accuracy == null ? '-' : Number(row.accuracy).toFixed(3)}</td>
                            <td className={Number(row.crossing_accepted_as_valid || 0) > 0 ? 'bad-cell' : ''}>{row.crossing_accepted_as_valid}</td>
                            <td>{row.crossing_total}</td>
                            <td>{row.crossing_accept_rate == null ? '-' : Number(row.crossing_accept_rate).toFixed(4)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
                ) : null}

                {/* â”€â”€ Section 8: Best combined thresholds â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
                {Array.isArray(locoTrainingResult?.best_combined_thresholds) && locoTrainingResult.best_combined_thresholds.length > 0 ? (
                <section className="training-section">
                  <h3>Mejores reglas combinadas</h3>
                  <div className="table-wrap model-table">
                    <table>
                      <thead><tr><th>model</th><th>criterio</th><th>vt</th><th>ct</th><th>P_valid</th><th>R_valid</th><th>F1_valid</th><th>crossing_accept_rate</th></tr></thead>
                      <tbody>
                        {locoTrainingResult.best_combined_thresholds.map((row, i) => (
                          <tr key={`best-combined-${i}`}>
                            <td>{row.model}</td>
                            <td>{row.criterion}</td>
                            <td>{Number(row.valid_threshold).toFixed(2)}</td>
                            <td>{Number(row.crossing_threshold).toFixed(2)}</td>
                            <td className="best-cell">{row.precision_valid == null ? '-' : Number(row.precision_valid).toFixed(3)}</td>
                            <td>{row.recall_valid == null ? '-' : Number(row.recall_valid).toFixed(3)}</td>
                            <td>{row.f1_valid == null ? '-' : Number(row.f1_valid).toFixed(3)}</td>
                            <td>{row.crossing_accept_rate == null ? '-' : Number(row.crossing_accept_rate).toFixed(4)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
                ) : null}

                {/* â”€â”€ Section 9: Binary thresholds â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
                <section className="training-section">
                  <h3>Thresholds (binario)</h3>
                  <div className="table-wrap model-table">
                    <table>
                      <thead><tr><th>threshold</th><th>precision_valid</th><th>recall_valid</th><th>f1_valid</th><th>FP</th><th>FN</th></tr></thead>
                      <tbody>
                        {locoTrainingThresholdRows.map((row) => (
                          <tr key={`th-${row.model_id}-${row.threshold}`}>
                            <td>{Number(row.threshold).toFixed(2)}</td>
                            <td>{Number(row.precision_valid).toFixed(3)}</td>
                            <td>{Number(row.recall_valid).toFixed(3)}</td>
                            <td>{Number(row.f1_valid).toFixed(3)}</td>
                            <td>{row.fp}</td>
                            <td>{row.fn}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>

                {/* â”€â”€ Section 10: Binary performance by radius â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
                <section className="training-section">
                  <h3>Performance by radius size (binario)</h3>
                  <div className="table-wrap model-table">
                    <table>
                      <thead><tr><th>model</th><th>radius_group</th><th>n</th><th>precision_valid</th><th>recall_valid</th><th>f1_valid</th><th>FP</th><th>FN</th></tr></thead>
                      <tbody>
                        {locoTrainingRadiusRows.map((row) => (
                          <tr key={`radius-${row.model_id}-${row.radius_group}`}>
                            <td>{row.model}</td>
                            <td>{row.radius_group}</td>
                            <td>{row.n_samples}</td>
                            <td>{Number(row.precision_valid).toFixed(3)}</td>
                            <td>{Number(row.recall_valid).toFixed(3)}</td>
                            <td>{Number(row.f1_valid).toFixed(3)}</td>
                            <td>{row.fp}</td>
                            <td>{row.fn}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>

                {/* â”€â”€ Section 11: Multiclass performance by radius â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
                {Array.isArray(locoTrainingResult?.multiclass_radius_group_metrics) && locoTrainingResult.multiclass_radius_group_metrics.length > 0 ? (
                <section className="training-section">
                  <h3>Performance by radius size (multiclase)</h3>
                  <div className="table-wrap model-table">
                    <table>
                      <thead><tr><th>model</th><th>radius_group</th><th>class</th><th>n</th><th>precision</th><th>recall</th><th>f1</th></tr></thead>
                      <tbody>
                        {locoTrainingResult.multiclass_radius_group_metrics.map((row, i) => (
                          <tr key={`mc-radius-${i}`}>
                            <td>{row.model}</td>
                            <td>{row.radius_group}</td>
                            <td>{row.class}</td>
                            <td>{row.n_samples}</td>
                            <td>{row.precision == null ? '-' : Number(row.precision).toFixed(3)}</td>
                            <td>{row.recall == null ? '-' : Number(row.recall).toFixed(3)}</td>
                            <td>{row.f1 == null ? '-' : Number(row.f1).toFixed(3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
                ) : null}

                {/* â”€â”€ Section 12: Binary error review â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
                <section className="training-section">
                  <h3>Error Review (binario)</h3>
                  <div className="training-error-grid">
                    {locoTrainingErrors.slice(0, 80).map((err) => (
                      <figure key={`${err.model_id}-${err.item_id}-${err.error_type}`} className="augment-card">
                        {err.patch_b64 ? <img src={b64ToDataUrl(err.patch_b64, 'image/png')} alt={err.candidate_id} /> : null}
                        <figcaption>{err.candidate_id}<br />real {err.label_real} pred {err.prediction}<br />p(valid) {Number(err.probability_valid).toFixed(3)} | r {err.radius_px == null ? '-' : Number(err.radius_px).toFixed(1)}</figcaption>
                      </figure>
                    ))}
                    {!locoTrainingErrors.length ? <div className="placeholder">Sin errores para este filtro.</div> : null}
                  </div>
                </section>

                {/* â”€â”€ Section 13: Multiclass error review â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
                {Array.isArray(locoTrainingResult?.error_review_multiclass) && locoTrainingResult.error_review_multiclass.length > 0 ? (
                <section className="training-section">
                  <h3>Error Review (multiclase)</h3>
                  <div className="filter-row">
                    <label className="field">
                      <span>class</span>
                      <select value={locoTrainingMcErrorClass} onChange={(e) => setLocoTrainingMcErrorClass(e.target.value)}>
                        <option value="all">all</option>
                        <option value="valid">valid</option>
                        <option value="invalid_crossing">invalid_crossing</option>
                        <option value="invalid_other">invalid_other</option>
                      </select>
                    </label>
                    <label className="field">
                      <span>error type</span>
                      <select value={locoTrainingMcErrorType} onChange={(e) => setLocoTrainingMcErrorType(e.target.value)}>
                        <option value="all">all</option>
                        {[...new Set(locoTrainingResult.error_review_multiclass.map((r) => r.error_type))].map((t) => (
                          <option key={t} value={t}>{t}</option>
                        ))}
                      </select>
                    </label>
                    <span className="chip">{locoTrainingMcErrors.length} errors</span>
                  </div>
                  <div className="training-error-grid">
                    {locoTrainingMcErrors.slice(0, 80).map((err, i) => (
                      <figure key={`mc-err-${i}`} className="augment-card">
                        {err.patch_b64 ? <img src={b64ToDataUrl(err.patch_b64, 'image/png')} alt={err.candidate_id} /> : null}
                        <figcaption>
                          {err.candidate_id}<br />
                          {err.label_real_name} &rarr; {err.prediction_name}<br />
                          v{Number(err.prob_valid).toFixed(2)} c{Number(err.prob_crossing).toFixed(2)} o{Number(err.prob_other).toFixed(2)}<br />
                          r {err.radius_px == null ? '-' : Number(err.radius_px).toFixed(1)}
                        </figcaption>
                      </figure>
                    ))}
                    {!locoTrainingMcErrors.length ? <div className="placeholder">Sin errores para este filtro.</div> : null}
                  </div>
                </section>
                ) : null}

                {/* â”€â”€ Section 14: Combined decision error review â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
                {Array.isArray(locoTrainingResult?.error_review_combined) && locoTrainingResult.error_review_combined.length > 0 ? (
                <section className="training-section">
                  <h3>Error Review (decisiÃ³n combinada)</h3>
                  <div className="filter-row">
                    <label className="field">
                      <span>class</span>
                      <select value={locoTrainingCombErrorClass} onChange={(e) => setLocoTrainingCombErrorClass(e.target.value)}>
                        <option value="all">all</option>
                        <option value="valid">valid</option>
                        <option value="invalid_crossing">invalid_crossing</option>
                        <option value="invalid_other">invalid_other</option>
                      </select>
                    </label>
                    <label className="field">
                      <span>error type</span>
                      <select value={locoTrainingCombErrorType} onChange={(e) => setLocoTrainingCombErrorType(e.target.value)}>
                        <option value="all">all</option>
                        {[...new Set(locoTrainingResult.error_review_combined.map((r) => r.error_type))].map((t) => (
                          <option key={t} value={t}>{t}</option>
                        ))}
                      </select>
                    </label>
                    <label className="field">
                      <span>subtype</span>
                      <select value={locoTrainingCombErrorSubtype} onChange={(e) => setLocoTrainingCombErrorSubtype(e.target.value)}>
                        <option value="all">all</option>
                        {[...new Set(locoTrainingResult.error_review_combined.map((r) => r.error_subtype).filter(Boolean))].map((t) => (
                          <option key={t} value={t}>{t}</option>
                        ))}
                      </select>
                    </label>
                    <span className="chip">{locoTrainingCombErrors.length} errors</span>
                  </div>
                  <div className="training-error-grid">
                    {locoTrainingCombErrors.slice(0, 80).map((err, i) => (
                      <figure key={`comb-err-${i}`} className="augment-card">
                        {err.patch_b64 ? <img src={b64ToDataUrl(err.patch_b64, 'image/png')} alt={err.candidate_id} /> : null}
                        <figcaption>
                          {err.candidate_id}<br />
                          real={err.label_real_multiclass_name} bin_pred={err.prediction_binary}<br />
                          bin={Number(err.binary_valid_score).toFixed(2)} v{Number(err.multiclass_prob_valid).toFixed(2)} c{Number(err.multiclass_prob_crossing).toFixed(2)} o{Number(err.multiclass_prob_other).toFixed(2)}<br />
                          {err.error_subtype ? <span className="bad-cell">{err.error_subtype}</span> : null}
                        </figcaption>
                      </figure>
                    ))}
                    {!locoTrainingCombErrors.length ? <div className="placeholder">Sin errores para este filtro.</div> : null}
                  </div>
                </section>
                ) : null}
              </div>
            ) : (
              <div className="placeholder">Configura Training en el panel izquierdo y toca Train Models.</div>
            )}
          </article>
          <article className={`card viewer loco-viewer ${workspaceTab === 'locoTest' ? '' : 'hidden-panel'}`}>
            <h2>Test circle model</h2>
            <div className="viewer-toolbar loco-toolbar">
              <button className={`icon-tool ${locoTestTool === 'circle' ? 'toggle-active' : ''}`} onClick={() => setLocoTestTool('circle')} disabled={!imageUrl} title="Dibujar circulo"><ToolIcon name="circle" />Circulo</button>
              <button className={`icon-tool ${locoTestTool === 'select' ? 'toggle-active' : ''}`} onClick={() => setLocoTestTool('select')} disabled={!imageUrl} title="Seleccionar circulo"><ToolIcon name="center" />Seleccionar</button>
              <button className={`icon-tool ${locoTestTool === 'pan' ? 'toggle-active' : ''}`} onClick={() => setLocoTestTool('pan')} disabled={!imageUrl} title="Mover vista"><ToolIcon name="hand" />Mano</button>
              <span className="toolbar-sep" />
              <button className="icon-tool" onClick={() => zoomLocoTestBy(0.84)} disabled={!imageUrl}>-</button>
              <button className="icon-tool" onClick={() => zoomLocoTestBy(1.2)} disabled={!imageUrl}>+</button>
              <button className="icon-tool" onClick={resetLocoTestView} disabled={!imageUrl}>Reset</button>
              <span className="zoom-chip">{Math.round(locoTestZoom * 100)}%</span>
              <span className="toolbar-sep" />
              <label className="compact-slider mask-opacity-control">
                mascara
                <input type="range" min="0.05" max="0.9" step="0.05" value={diamMaskOpacity} onChange={(e) => setDiamMaskOpacity(Number(e.target.value || 0.38))} disabled={!diamVisualMaskUrl} />
                <span>{Math.round(diamMaskOpacity * 100)}%</span>
              </label>
              <button className="primary" onClick={predictLocoTestCircles} disabled={!locoTestCircles.length || loading.locoTest}>Predict circles</button>
            </div>

            <div className="loco-workspace loco-lab-workspace">
              <div
                ref={locoTestStageRef}
                className={`stage loco-stage mode-${locoTestTool === 'pan' ? 'pan' : 'circle'} ${locoTestIsPanning ? 'is-panning' : ''} ${imageUrl ? 'has-image' : ''}`}
                onPointerDown={onLocoTestPointerDown}
                onPointerMove={onLocoTestPointerMove}
                onPointerUp={onLocoTestPointerUp}
                onPointerCancel={onLocoTestPointerUp}
                onPointerLeave={onLocoTestPointerUp}
              >
                {imageUrl ? (
                  <div
                    className="loco-content"
                    style={{
                      width: `${Math.max(1, imageDims.w)}px`,
                      height: `${Math.max(1, imageDims.h)}px`,
                      transform: getLocoTestRenderMetrics()
                        ? `translate(${getLocoTestRenderMetrics().x}px, ${getLocoTestRenderMetrics().y}px) scale(${getLocoTestRenderMetrics().scale})`
                        : 'translate(-99999px, -99999px) scale(1)',
                    }}
                  >
                    <div className="diameter-raster-layer">
                      <img src={imageUrl} alt="test-base" draggable={false} />
                      {diamVisualMaskUrl ? (
                        <img src={diamVisualMaskUrl} alt="prior mask test" className="diam-mask-layer" style={{ opacity: diamMaskOpacity }} draggable={false} />
                      ) : null}
                    </div>
                    <div className="loco-annotation-layer">
                      <svg className="loco-overlay loco-dataset-overlay" viewBox={`0 0 ${Math.max(1, imageDims.w)} ${Math.max(1, imageDims.h)}`}>
                        {locoTestCircles.map((c) => {
                          const selected = String(c.candidate_id) === String(locoTestSelectedId)
                          const pred = locoTestPredById[c.candidate_id]
                          const rawLabel = c.label || 'invalid_other'
                          const labelClass = rawLabel === 'invalid' ? 'invalid_other' : rawLabel
                          const cls = pred ? (pred.correct ? 'test-correct' : 'test-error') : labelClass
                          return (
                            <g key={`test-circle-${c.candidate_id}`}>
                              <circle className={`loco-dataset-circle ${cls} ${selected ? 'selected' : ''}`} cx={Number(c.center_x)} cy={Number(c.center_y)} r={Math.max(1, Number(c.radius_px) || 1)} />
                              {pred ? (
                                <text className="loco-test-label" x={Number(c.center_x)} y={Number(c.center_y)}>
                                  {pred.prediction} {Number(pred.probability_valid).toFixed(2)}
                                  {pred.multiclass ? ` | v${Number(pred.multiclass.prob_valid).toFixed(2)} c${Number(pred.multiclass.prob_crossing).toFixed(2)} o${Number(pred.multiclass.prob_other).toFixed(2)}` : ''}
                                </text>
                              ) : null}
                              <circle
                                className="loco-dataset-hit"
                                cx={Number(c.center_x)}
                                cy={Number(c.center_y)}
                                r={Math.max(6, Number(c.radius_px) || 6)}
                                onPointerDown={(e) => beginMoveLocoTestCircle(e, c)}
                                onClick={(e) => {
                                  e.stopPropagation()
                                  setLocoTestSelectedId(String(c.candidate_id))
                                  if (locoTestTool !== 'circle') setLocoTestTool('select')
                                }}
                              />
                            </g>
                          )
                        })}
                        {locoTestDraftCircle ? <circle className="loco-dataset-draft" cx={Number(locoTestDraftCircle.center_x)} cy={Number(locoTestDraftCircle.center_y)} r={Math.max(1, Number(locoTestDraftCircle.radius_px) || 1)} /> : null}
                      </svg>
                    </div>
                  </div>
                ) : <div className="placeholder">Carga una imagen para testear cÃ­rculos.</div>}
              </div>

              <aside className="loco-debug-panel loco-lab-panel">
                <div className="loco-step-head">
                  <strong>Resultado por cÃ­rculo</strong>
                  <span>{loading.locoTest ? 'prediciendo' : 'listo'}</span>
                </div>
                {locoTestResult ? (
                  <div className="loco-result-card">
                    <strong>{locoTestResult.model_id}</strong>
                    <span>{locoTestResult.training_run_id}</span>
                    <p className="small">threshold {locoTestResult.threshold} | errores {locoTestErrors.length}{locoTestResult.has_multiclass ? ' | multiclase âœ“' : ''}</p>
                  </div>
                ) : null}
                <div className="loco-candidate-list">
                  {locoTestCircles.length ? locoTestCircles.map((c, idx) => {
                    const pred = locoTestPredById[c.candidate_id]
                    return (
                      <button
                        key={`test-row-${c.candidate_id}`}
                        className={String(c.candidate_id) === String(locoTestSelectedId) ? 'selected' : ''}
                        onClick={() => { setLocoTestSelectedId(String(c.candidate_id)); if (locoTestTool !== 'circle') setLocoTestTool('select') }}
                      >
                        <span>{idx + 1}</span>
                        <strong>{c.label}{pred ? ` -> ${pred.prediction}` : ''}</strong>
                        <em>{pred ? `${pred.correct ? 'OK' : pred.error_type} ${Number(pred.probability_valid).toFixed(2)}${pred.multiclass ? ` v${Number(pred.multiclass.prob_valid).toFixed(2)}` : ''}` : `r ${Number(c.radius_px).toFixed(1)}`}</em>
                      </button>
                    )
                  }) : <div className="placeholder small">Sin circulos.</div>}
                </div>
              </aside>
            </div>
          </article>
          <article className={`card viewer loco-viewer ${workspaceTab === 'locoModel' ? '' : 'hidden-panel'}`}>
            <h2>LOCO Detector</h2>
            <div className="viewer-toolbar loco-toolbar">
              <button className="icon-tool toggle-active" disabled={!imageUrl} title="Mover vista"><ToolIcon name="hand" />Mano</button>
              <span className="toolbar-sep" />
              <button className="icon-tool" onClick={() => zoomLocoModelBy(0.84)} disabled={!imageUrl}>-</button>
              <button className="icon-tool" onClick={() => zoomLocoModelBy(1.2)} disabled={!imageUrl}>+</button>
              <button className="icon-tool" onClick={resetLocoModelView} disabled={!imageUrl}>Reset</button>
              <span className="zoom-chip">{Math.round(locoModelZoom * 100)}%</span>
              <span className="toolbar-sep" />
              <label className="compact-slider mask-opacity-control">
                mascara
                <input type="range" min="0.05" max="0.9" step="0.05" value={diamMaskOpacity} onChange={(e) => setDiamMaskOpacity(Number(e.target.value || 0.38))} disabled={!diamVisualMaskUrl} />
                <span>{Math.round(diamMaskOpacity * 100)}%</span>
              </label>
              <button className="primary" onClick={detectLocoModelCircles} disabled={!imageUrl || loading.locoModel}>Detect circles</button>
              <button onClick={measureLocoModelAccepted} disabled={!locoModelAccepted.length || loading.locoModel}>Run Diameter on Accepted</button>
            </div>

            <div className="loco-workspace loco-lab-workspace">
              <div
                ref={locoModelStageRef}
                className={`stage loco-stage mode-pan ${locoModelIsPanning ? 'is-panning' : ''} ${imageUrl ? 'has-image' : ''}`}
                onPointerDown={onLocoModelPointerDown}
                onPointerMove={onLocoModelPointerMove}
                onPointerUp={onLocoModelPointerUp}
                onPointerCancel={onLocoModelPointerUp}
                onPointerLeave={onLocoModelPointerUp}
              >
                {imageUrl ? (
                  <div
                    className="loco-content"
                    style={{
                      width: `${Math.max(1, imageDims.w)}px`,
                      height: `${Math.max(1, imageDims.h)}px`,
                      transform: getLocoModelRenderMetrics()
                        ? `translate(${getLocoModelRenderMetrics().x}px, ${getLocoModelRenderMetrics().y}px) scale(${getLocoModelRenderMetrics().scale})`
                        : 'translate(-99999px, -99999px) scale(1)',
                    }}
                  >
                    <div className="diameter-raster-layer">
                      <img src={imageUrl} alt="detector-base" draggable={false} />
                      {diamVisualMaskUrl && locoModelLayers.mask ? (
                        <img src={diamVisualMaskUrl} alt="prior mask detector" className="diam-mask-layer" style={{ opacity: diamMaskOpacity }} draggable={false} />
                      ) : null}
                    </div>
                    <div className="loco-annotation-layer">
                      <svg className="loco-overlay loco-model-overlay" viewBox={`0 0 ${Math.max(1, imageDims.w)} ${Math.max(1, imageDims.h)}`}>
                        {locoModelLayers.tiles && locoModelParams.candidate_sampling_mode === 'tile_balanced' ? (() => {
                          const tw = Number(locoModelParams.tile_size_px || 128)
                          const iw = Math.max(1, imageDims.w)
                          const ih = Math.max(1, imageDims.h)
                          const lines = []
                          for (let x = tw; x < iw; x += tw) {
                            lines.push(<line key={`vt-${x}`} x1={x} y1={0} x2={x} y2={ih} className="loco-tile-grid" />)
                          }
                          for (let y = tw; y < ih; y += tw) {
                            lines.push(<line key={`ht-${y}`} x1={0} y1={y} x2={iw} y2={y} className="loco-tile-grid" />)
                          }
                          return lines
                        })() : null}
                        {locoModelLayers.rejected ? locoModelRejected.map((c) => {
                          const selected = String(c.candidate_id) === String(locoModelSelectedId)
                          const reason = String(c.reason || 'rejected')
                          return (
                            <g key={`det-rej-${c.candidate_id}`}>
                              <circle
                                className={`loco-model-circle rejected ${reason} ${selected ? 'selected' : ''}`}
                                cx={Number(c.center_x)}
                                cy={Number(c.center_y)}
                                r={Math.max(1, Number(c.radius_px) || 1)}
                              />
                              <circle
                                className="loco-dataset-hit"
                                cx={Number(c.center_x)}
                                cy={Number(c.center_y)}
                                r={Math.max(6, Number(c.radius_px) || 6)}
                                onPointerDown={(e) => {
                                  e.stopPropagation()
                                  setLocoModelSelectedId(String(c.candidate_id))
                                }}
                              />
                            </g>
                          )
                        }) : null}
                        {locoModelLayers.accepted ? locoModelAccepted.map((c) => {
                          const selected = String(c.candidate_id) === String(locoModelSelectedId)
                          const measured = !!locoModelMeasureByProposal[String(c.candidate_id)]
                          return (
                            <g key={`det-acc-${c.candidate_id}`}>
                              <circle
                                className={`loco-model-circle accepted ${measured ? 'measured' : ''} ${selected ? 'selected' : ''}`}
                                cx={Number(c.center_x)}
                                cy={Number(c.center_y)}
                                r={Math.max(1, Number(c.radius_px) || 1)}
                              />
                              {locoModelLayers.scores ? (
                                <text className="loco-model-label" x={Number(c.center_x)} y={Number(c.center_y)}>
                                  {Number(c.valid_score || 0).toFixed(2)}
                                </text>
                              ) : null}
                              <circle
                                className="loco-dataset-hit"
                                cx={Number(c.center_x)}
                                cy={Number(c.center_y)}
                                r={Math.max(6, Number(c.radius_px) || 6)}
                                onPointerDown={(e) => {
                                  e.stopPropagation()
                                  setLocoModelSelectedId(String(c.candidate_id))
                                }}
                              />
                            </g>
                          )
                        }) : null}
                      </svg>
                    </div>
                  </div>
                ) : <div className="placeholder">Carga una imagen para detectar circulos con el modelo.</div>}
              </div>

              <aside className="loco-debug-panel loco-lab-panel">
                <div className="loco-step-head">
                  <strong>Detector</strong>
                  <span>{loading.locoModel ? 'procesando' : 'listo'}</span>
                </div>
                {locoModelResult ? (
                  <div className="loco-result-card">
                    <strong>{locoModelResult.model_id}</strong>
                    <span>{locoModelResult.model_run_id}</span>
                    <p className="small">{locoModelResult.run_id}<br />{locoModelResult.run_dir}</p>
                  </div>
                ) : null}
                <div className="kpi-row">
                  <span>Total <strong>{locoModelResult?.summary?.total_candidates ?? 0}</strong></span>
                  <span>Muestra <strong>{locoModelResult?.summary?.sampled_candidates ?? locoModelResult?.summary?.total_candidates ?? 0}</strong></span>
                  <span>Evaluados <strong>{locoModelResult?.summary?.evaluated_candidates ?? 0}</strong></span>
                  <span>Threshold <strong>{locoModelResult?.summary?.accepted_before_nms ?? 0}</strong></span>
                  <span>Final <strong>{locoModelResult?.summary?.accepted_after_nms ?? 0}</strong></span>
                  <span>Rechazados <strong>{locoModelResult?.summary?.rejected_by_threshold ?? 0}</strong></span>
                  <span>NMS <strong>{locoModelResult?.summary?.removed_by_nms ?? 0}</strong></span>
                  <span>Multiclase <strong>{locoModelResult?.has_multiclass ? 'âœ“' : 'âœ—'}</strong></span>
                  {locoModelResult?.has_multiclass ? <span>cross th <strong>{Number(locoModelResult?.crossing_threshold || 0.5).toFixed(2)}</strong></span> : null}
                </div>

                {locoModelSelectedCandidate ? (
                  <div className="loco-lab-section">
                    <h3>Seleccionado</h3>
                    <div className="kpi">id: <strong>{locoModelSelectedCandidate.candidate_id}</strong></div>
                    <div className="kpi">score: <strong>{Number(locoModelSelectedCandidate.valid_score || 0).toFixed(4)}</strong></div>
                    <div className="kpi">radio: <strong>{Number(locoModelSelectedCandidate.radius_px || 0).toFixed(1)}</strong></div>
                    <div className="kpi">grupo: <strong>{locoModelSelectedCandidate.radius_group || '-'}</strong></div>
                    <div className="kpi">estado: <strong>{locoModelSelectedCandidate.status}</strong></div>
                    <div className="kpi">razon: <strong>{locoModelSelectedCandidate.reason}</strong></div>
                    <div className="kpi">cortes: <strong>{locoModelSelectedCandidate.features?.n_cortes ?? '-'}</strong></div>
                    <div className="kpi">area: <strong>{locoModelSelectedCandidate.features?.area_mask_ratio == null ? '-' : Number(locoModelSelectedCandidate.features.area_mask_ratio).toFixed(3)}</strong></div>
                    <div className="kpi">sim: <strong>{locoModelSelectedCandidate.features?.simetria_cuadrilatero == null ? '-' : Number(locoModelSelectedCandidate.features.simetria_cuadrilatero).toFixed(3)}</strong></div>
                    <div className="kpi">bridge: <strong>{locoModelSelectedCandidate.diagnostics?.component_bridge_score == null ? '-' : Number(locoModelSelectedCandidate.diagnostics.component_bridge_score).toFixed(3)}</strong></div>
                    {locoModelSelectedCandidate.multiclass ? (
                      <>
                        <div className="kpi">p(valid): <strong>{Number(locoModelSelectedCandidate.multiclass.prob_valid).toFixed(3)}</strong></div>
                        <div className="kpi">p(crossing): <strong>{Number(locoModelSelectedCandidate.multiclass.prob_crossing).toFixed(3)}</strong></div>
                        <div className="kpi">p(other): <strong>{Number(locoModelSelectedCandidate.multiclass.prob_other).toFixed(3)}</strong></div>
                        <div className="kpi">clase pred: <strong>{['valid', 'crossing', 'other'][locoModelSelectedCandidate.multiclass.predicted_class] ?? locoModelSelectedCandidate.multiclass.predicted_class}</strong></div>
                      </>
                    ) : null}
                    {locoModelMeasureByProposal[String(locoModelSelectedCandidate.candidate_id)] ? (
                      <div className="kpi">diametro: <strong>{Number(locoModelMeasureByProposal[String(locoModelSelectedCandidate.candidate_id)].diameter_px || 0).toFixed(2)} px</strong></div>
                    ) : null}
                  </div>
                ) : null}

                <div className="loco-lab-section">
                  <h3>Aceptados</h3>
                  <div className="loco-candidate-list">
                    {locoModelAccepted.length ? locoModelAccepted.slice(0, 400).map((c, idx) => (
                      <button
                        key={`model-accepted-${c.candidate_id}`}
                        className={String(c.candidate_id) === String(locoModelSelectedId) ? 'selected' : ''}
                        onClick={() => setLocoModelSelectedId(String(c.candidate_id))}
                      >
                        <span>{idx + 1}</span>
                        <strong>{Number(c.valid_score || 0).toFixed(3)}</strong>
                        <em>r {Number(c.radius_px || 0).toFixed(1)} | {c.radius_group}</em>
                      </button>
                    )) : <div className="placeholder small">Sin aceptados.</div>}
                  </div>
                </div>

                {locoModelLayers.rejected ? (
                  <div className="loco-lab-section">
                    <h3>Rechazados debug</h3>
                    <div className="loco-candidate-list">
                      {locoModelRejected.length ? locoModelRejected.slice(0, 250).map((c, idx) => (
                        <button
                          key={`model-rejected-${c.candidate_id}`}
                          className={String(c.candidate_id) === String(locoModelSelectedId) ? 'selected' : ''}
                          onClick={() => setLocoModelSelectedId(String(c.candidate_id))}
                        >
                          <span>{idx + 1}</span>
                          <strong>{Number(c.valid_score || 0).toFixed(3)}</strong>
                          <em>{c.reason || c.status}{c.multiclass && c.reason === 'crossing_detected' ? ` v${Number(c.multiclass.prob_valid).toFixed(2)} c${Number(c.multiclass.prob_crossing).toFixed(2)}` : ''}</em>
                        </button>
                      )) : <div className="placeholder small">Activa return rejected y detecta de nuevo.</div>}
                    </div>
                  </div>
                ) : null}
              </aside>
            </div>
          </article>
          <article className={`card viewer models-viewer ${workspaceTab === 'models' ? '' : 'hidden-panel'}`}>
            <div className="model-workspace">
              <section>
                <h2>Dataset y modelos de asistencia</h2>
                <div className="kpi-row">
                  <span>Imagenes: <strong>{trainImageIds.length}</strong></span>
                  <span>Trainables: <strong>{modelDataset.filter((x) => x.trainable_multiclass || x.trainable_binary).length}</strong></span>
                  <span>Modelos: <strong>{assistModels.length}</strong></span>
                  <span>Activo: <strong>{activeAssistModel?.model_name || activeAssistModel?.model_id || '-'}</strong></span>
                </div>
                <div className="inline model-dataset-actions">
                  <button onClick={() => refreshModelDataset()} disabled={loading.modelsDataset}>Refrescar dataset</button>
                  <button onClick={selectTrainableImages} disabled={!modelDataset.length}>Seleccionar validas</button>
                  <button onClick={clearTrainImages} disabled={!trainImageIds.length}>Limpiar seleccion</button>
                </div>
                <div className="table-wrap model-table">
                  <table>
                    <thead>
                      <tr>
                        <th>usar</th>
                        <th>imagen</th>
                        <th>path</th>
                        <th>modificacion</th>
                        <th>fibra</th>
                        <th>halo</th>
                        <th>fondo</th>
                        <th>estado</th>
                      </tr>
                    </thead>
                    <tbody>
                      {modelDataset.map((item) => {
                        const counts = item.class_counts || {}
                        const trainable = !!(item.trainable_multiclass || item.trainable_binary)
                        const thumbs = modelDatasetPreviewUrls(item)
                        return (
                          <tr key={`model-dataset-${item.image_id}`}>
                            <td>
                              <input
                                type="checkbox"
                                checked={!!selectedTrainImages[item.image_id]}
                                onChange={() => toggleTrainImage(item.image_id)}
                                disabled={!trainable}
                              />
                            </td>
                            <td>
                              <div
                                className="model-dataset-table-image"
                                title={`${item.image_name || item.image_id}\nDoble click para ampliar`}
                                onDoubleClick={() => openModelImagePreview(item)}
                              >
                                {thumbs.real ? <img src={thumbs.real} alt="" /> : <span className="model-dataset-empty">imagen</span>}
                                {thumbs.scribble ? <img src={thumbs.scribble} alt="" /> : <span className="model-dataset-empty">scribble</span>}
                              </div>
                            </td>
                            <td title={item.source_path || ''}>{shortPathTail(item.source_path)}</td>
                            <td>{item.source_mtime || item.updated_at || '-'}</td>
                            <td>{counts.fiber || 0}</td>
                            <td>{counts.halo || 0}</td>
                            <td>{counts.background || 0}</td>
                            <td>{trainable ? 'usable' : 'faltan clases'}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </section>

              <section>
                <h2>Modelos guardados</h2>
                {!assistModels.length ? (
                  <div className="placeholder small">Entrena un modelo desde el panel lateral.</div>
                ) : (
                  <div className="table-wrap model-table">
                    <table>
                      <thead>
                        <tr>
                          <th>modelo</th>
                          <th>modo</th>
                          <th>imagenes</th>
                          <th>muestras</th>
                          <th>accuracy</th>
                          <th>fecha</th>
                        </tr>
                      </thead>
                      <tbody>
                        {assistModels.map((model) => (
                          <tr key={`registry-${model.model_id}`} className={model.model_id === selectedAssistModelId ? 'selected-row' : ''} onClick={() => setSelectedAssistModelId(model.model_id)}>
                            <td>{model.model_name || model.model_id}{model.model_id === defaultAssistModelId ? ' *' : ''}</td>
                            <td>{model.class_mode || '-'}</td>
                            <td>{model.image_count || 0}</td>
                            <td>{model.train_samples || 0}</td>
                            <td>{model.metrics?.holdout_accuracy == null ? (model.metrics?.train_accuracy == null ? '-' : Number(model.metrics.train_accuracy).toFixed(3)) : Number(model.metrics.holdout_accuracy).toFixed(3)}</td>
                            <td>{model.created_at || '-'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>
            </div>
          </article>
        </section>
      </div>
      {modelImagePreview ? (
        <div className="image-preview-modal" role="dialog" aria-modal="true" onDoubleClick={() => setModelImagePreview(null)}>
          <div className="image-preview-dialog">
            <div className="point-review-head">
              <div>
                <h2>{modelImagePreview.image_name || 'Imagen'}</h2>
                <p>
                  {shortPathTail(modelImagePreview.source_path)} | {modelImagePreview.source_mtime || '-'}
                  {modelImagePreview.loading ? ' | cargando imagen real...' : ''}
                  {modelImagePreview.error ? ` | ${modelImagePreview.error}` : ''}
                </p>
              </div>
              <button className="icon-tool" onClick={() => setModelImagePreview(null)}>Cerrar</button>
            </div>
            <div className="image-preview-grid">
              <figure>
                <figcaption>Imagen real</figcaption>
                {modelImagePreview.real ? <img src={modelImagePreview.real} alt="imagen real" /> : <div className="placeholder">Sin imagen.</div>}
              </figure>
              <figure>
                <figcaption>Scribble</figcaption>
                {modelImagePreview.scribble ? <img src={modelImagePreview.scribble} alt="scribble" /> : <div className="placeholder">Sin scribble.</div>}
              </figure>
            </div>
          </div>
        </div>
      ) : null}
      {pointReviewOpen ? (
        <div className="point-review-modal" role="dialog" aria-modal="true">
          <div className="point-review-dialog">
            <div className="point-review-head">
              <div>
                <h2>Analisis del punto</h2>
                <p>
                  #{Number(diamReviewTarget?.point_index ?? -1) + 1}
                  {' | '}
                  {diamReviewTarget?.status || 'sin estado'}
                  {diamReviewTarget?.diameter_px == null ? '' : ` | ${Number(diamReviewTarget.diameter_px).toFixed(2)} px`}
                </p>
              </div>
              <button className="icon-tool" onClick={() => setPointReviewOpen(false)}>Cerrar</button>
            </div>
            <div className="point-review-body">
              <div className="point-review-view">
                <div className="viewer-toolbar">
                  <button className="icon-tool" onClick={() => zoomPointReviewBy(0.84)} disabled={!imageUrl}>-</button>
                  <button className="icon-tool" onClick={() => zoomPointReviewBy(1.2)} disabled={!imageUrl}>+</button>
                  <button className="icon-tool" onClick={centerPointReview} disabled={!imageUrl || !diamReviewTarget}>Centrar 200%</button>
                  <span className="zoom-chip">{Math.round(pointReviewZoom * 100)}%</span>
                </div>
                <div
                  ref={pointReviewStageRef}
                  className={`stage point-review-stage ${pointReviewIsPanning ? 'is-panning' : ''}`}
                  onPointerDown={onPointReviewPointerDown}
                  onPointerMove={onPointReviewPointerMove}
                  onPointerUp={onPointReviewPointerUp}
                  onPointerCancel={onPointReviewPointerUp}
                  onPointerLeave={onPointReviewPointerUp}
                >
                  {imageUrl ? (
                    <div
                      className="diameter-content"
                      style={{
                        width: `${Math.max(1, imageDims.w)}px`,
                        height: `${Math.max(1, imageDims.h)}px`,
                        transform: pointReviewRenderMetrics
                          ? `translate(${pointReviewRenderMetrics.x}px, ${pointReviewRenderMetrics.y}px) scale(${pointReviewRenderMetrics.scale})`
                          : 'translate(-99999px, -99999px) scale(1)',
                      }}
                    >
                      <img src={diamOverlayUrl || imageUrl} alt="point-review" draggable={false} />
                      {diamReviewTarget?.point ? (
                        <button
                          className="diam-marker active point-review-marker"
                          style={{
                            left: `${(Number(diamReviewTarget.point.x) / Math.max(1, imageDims.w)) * 100}%`,
                            top: `${(Number(diamReviewTarget.point.y) / Math.max(1, imageDims.h)) * 100}%`,
                          }}
                          title={`Punto ${Number(diamReviewTarget.point_index) + 1}`}
                        >
                          <span className="diam-marker-dot" />
                          <span className="diam-marker-label">{Number(diamReviewTarget.point_index) + 1}</span>
                        </button>
                      ) : null}
                      {diameterResultLines.length || diameterResultQuads.length ? (
                        <svg className="diam-result-line-overlay" viewBox={`0 0 ${Math.max(1, imageDims.w)} ${Math.max(1, imageDims.h)}`}>
                          {diameterResultQuads.map((quad) => (
                            <polygon key={`review-${quad.key}`} className={quad.ok ? '' : 'rejected'} points={quad.points} />
                          ))}
                          {diameterResultLines.map((line) => (
                            <line
                              key={`review-${line.key}`}
                              className={line.ok ? '' : 'rejected'}
                              x1={line.x1}
                              y1={line.y1}
                              x2={line.x2}
                              y2={line.y2}
                            />
                          ))}
                        </svg>
                      ) : null}
                    </div>
                  ) : (
                    <div className="placeholder">Carga una imagen para revisar puntos.</div>
                  )}
                </div>
              </div>
              <div className="point-review-form">
                <div className="review-target-box">
                  <strong>{diamReviewTarget?.quality_label || 'sin etiqueta'}</strong>
                  <span>
                    confianza {diamReviewTarget?.confidence == null ? '-' : Number(diamReviewTarget.confidence).toFixed(3)}
                    {diamReviewTarget?.reason ? ` | ${diamReviewTarget.reason}` : ''}
                  </span>
                  {diamReviewTarget?.method_id === 'hybrid_profile_diameter_v3' || diamReviewTarget?.measurement_mode ? (
                    <span>
                      modo {diamReviewTarget?.measurement_mode || '-'}
                      {' | '}geometria {diamReviewTarget?.geometry_status || '-'}
                      {' | '}soporte {diamReviewTarget?.support_status || '-'}
                      {' | '}control {diamReviewTarget?.geometry_control_status || '-'}
                      {' | '}perfil {diamReviewTarget?.profile_length_effective_px == null ? '-' : Number(diamReviewTarget.profile_length_effective_px).toFixed(1)}
                      {' | '}coherencia {diamReviewTarget?.orientation_coherence == null ? '-' : Number(diamReviewTarget.orientation_coherence).toFixed(3)}
                      {' | '}upscale {diamReviewTarget?.used_upscale ? `${diamReviewTarget?.scale_factor || 1}x` : 'no'}
                    </span>
                  ) : null}
                  {diamReviewTarget?.methodology_id ? (
                    <span>
                      metodologia {diamReviewTarget.methodology_id}
                      {' | '}contexto {diamReviewTarget.local_context_label || '-'}
                      {' | '}politica {diamReviewTarget.selected_edge_policy || '-'}
                      {' | '}halo {diamReviewTarget.halo_status || '-'}
                      {' | '}ruta {diamReviewTarget.size_route || '-'}
                      {' | '}ridge {diamReviewTarget.ridge_anchor_status || '-'}
                      {' | '}flux {diamReviewTarget.flux_status || '-'}
                    </span>
                  ) : null}
                  {diamReviewTarget?.diameter_route || diamReviewTarget?.mask_method ? (
                    <span>
                      ruta {diamReviewTarget?.diameter_route || diamReviewTarget?.size_route || '-'}
                      {' | '}fibra {diamReviewTarget?.fiber_size_mode || '-'}
                      {' | '}auto {diamReviewTarget?.auto_size_reason || '-'}
                      {' | '}mask {diamReviewTarget?.mask_method || '-'}
                      {' | '}conf mask {diamReviewTarget?.mask_confidence == null ? '-' : Number(diamReviewTarget.mask_confidence).toFixed(3)}
                      {' | '}shift mask {diamReviewTarget?.mask_center_shift_px == null ? '-' : Number(diamReviewTarget.mask_center_shift_px).toFixed(2)}
                      {' | '}caliper {diamReviewTarget?.mask_caliper_diameter_px == null ? '-' : Number(diamReviewTarget.mask_caliper_diameter_px).toFixed(2)}
                      {' | '}ray {diamReviewTarget?.mask_raycast_diameter_px == null ? '-' : Number(diamReviewTarget.mask_raycast_diameter_px).toFixed(2)}
                      {' | '}circulo {diamReviewTarget?.circle_radius_px == null ? '-' : Number(diamReviewTarget.circle_radius_px).toFixed(1)}
                      {' | '}cuadrado {diamReviewTarget?.square_samples_valid == null ? '-' : `${diamReviewTarget.square_samples_valid}/${diamReviewTarget.square_samples_total || 0}`}
                      {' | '}manual {diamReviewTarget?.manual_input_diameter_px == null ? '-' : Number(diamReviewTarget.manual_input_diameter_px).toFixed(2)}
                      {' | '}elipse {diamReviewTarget?.ellipse_minor_px == null ? '-' : `${Number(diamReviewTarget.ellipse_minor_px).toFixed(2)}/${Number(diamReviewTarget.ellipse_major_px || 0).toFixed(2)}`}
                    </span>
                  ) : null}
                </div>
                <label className="field">
                  <span>revision medicion</span>
                  <select value={validationForm.measurement_decision} onChange={(e) => updateValidationForm('measurement_decision', e.target.value)}>
                    <option value="unreviewed">sin revisar</option>
                    <option value="validated">validada</option>
                    <option value="rejected">negada</option>
                    <option value="uncertain">dudosa</option>
                  </select>
                </label>
                <label className="field">
                  <span>conclusion del punto</span>
                  <textarea
                    rows={10}
                    value={validationForm.result_comment}
                    onChange={(e) => updateValidationForm('result_comment', e.target.value)}
                    placeholder="Escribe tu interpretacion del punto, por que la medicion sirve o falla, y que deberia ajustarse despues."
                  />
                </label>
                <label className="field">
                  <span>nota tecnica opcional</span>
                  <textarea rows={4} value={validationForm.notes} onChange={(e) => updateValidationForm('notes', e.target.value)} />
                </label>
                <div className="tier-buttons validation-decision-buttons">
                  <button className="tier tier-a" onClick={() => saveValidationCase({ overrides: { measurement_decision: 'validated' } })} disabled={!imageId || loading.validationSave}>Validar</button>
                  <button className="tier tier-c" onClick={() => saveValidationCase({ overrides: { measurement_decision: 'uncertain' } })} disabled={!imageId || loading.validationSave}>Dudosa</button>
                  <button className="tier tier-unusable" onClick={() => saveValidationCase({ overrides: { measurement_decision: 'rejected' } })} disabled={!imageId || loading.validationSave}>Negar</button>
                </div>
                <button className="primary" onClick={() => saveValidationCase()} disabled={!imageId || loading.validationSave}>Guardar conclusion</button>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
