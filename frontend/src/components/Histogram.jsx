import React, { useMemo } from 'react'

/**
 * Histogram component renders an SVG bar chart of diameter values.
 *
 * Props:
 *   values: number[] — array of diameter values (already converted to target unit)
 *   unit: string — 'px' | 'nm' | 'um'
 *   bins: number — number of bins (5-50)
 *   width: number — SVG width
 *   height: number — SVG height
 */

const MARGIN = { top: 22, right: 24, bottom: 78, left: 82 }

function computeStats(values) {
  if (!values || values.length === 0) {
    return { mean: 0, median: 0, std: 0, min: 0, max: 0, n: 0 }
  }
  const n = values.length
  const sorted = [...values].sort((a, b) => a - b)
  const min = sorted[0]
  const max = sorted[n - 1]
  const mean = sorted.reduce((s, v) => s + v, 0) / n
  const median = n % 2 === 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[Math.floor(n / 2)]
  const variance = sorted.reduce((s, v) => s + (v - mean) ** 2, 0) / n
  const std = Math.sqrt(variance)
  return { mean, median, std, min, max, n }
}

function computeBins(values, binCount) {
  if (!values || values.length === 0) return []
  const min = Math.min(...values)
  const max = Math.max(...values)
  if (max === min) {
    // All values identical: single bin
    return [{ binStart: min, binEnd: min + 1, count: values.length }]
  }
  const binWidth = (max - min) / binCount
  const bins = Array.from({ length: binCount }, (_, i) => ({
    binStart: min + i * binWidth,
    binEnd: min + (i + 1) * binWidth,
    count: 0,
  }))
  for (const v of values) {
    const idx = Math.min(Math.floor((v - min) / binWidth), binCount - 1)
    bins[idx].count += 1
  }
  return bins
}

function formatTick(value) {
  const abs = Math.abs(value)
  if (abs >= 100) return value.toFixed(0)
  if (abs >= 10) return value.toFixed(1)
  return value.toFixed(2)
}

function buildIntegerTicks(maxCount, maxTicks = 6) {
  if (maxCount <= 0) return [0]
  if (maxCount <= maxTicks - 1) {
    return Array.from({ length: maxCount + 1 }, (_, i) => i)
  }
  const step = Math.ceil(maxCount / (maxTicks - 1))
  const ticks = []
  for (let value = 0; value < maxCount; value += step) {
    ticks.push(value)
  }
  if (ticks[ticks.length - 1] !== maxCount) ticks.push(maxCount)
  return ticks
}

function buildValueTicks(binned, maxTicks = 6) {
  if (!binned.length) return []
  const min = binned[0].binStart
  const max = binned[binned.length - 1].binEnd
  if (max === min) return [min]
  return Array.from({ length: maxTicks }, (_, i) => min + ((max - min) * i) / (maxTicks - 1))
}

function labelWithUnit(label, unit) {
  const clean = String(label || '').trim() || 'Diametro'
  if (clean.includes('{unit}')) return clean.replace('{unit}', unit)
  return `${clean} (${unit})`
}

export default function Histogram({
  values,
  unit = 'px',
  bins: binCount = 20,
  width = 500,
  height = 250,
  xPaddingEnabled = true,
  xPaddingPercent = 4,
  grid = 'both',
  gridStyle = 'solid',
  barColor = '#45464d',
  strokeColor = '#191c1e',
  backgroundColor = 'transparent',
  backgroundAltColor = '#f7f9fb',
  backgroundMode = 'solid',
  fontFamily = 'JetBrains Mono',
  fontSize = 10,
  tickFontSize = fontSize,
  labelFontSize = fontSize + 1,
  fontColor = '#45464d',
  axisColor = '#76777d',
  gridColor = '#eceef0',
  xLabel = 'Diametro',
  yLabel = 'Frecuencia',
  xLabelOffset = 16,
  yLabelOffset = 18,
  borderTop = false,
  borderRight = false,
  borderBottom = true,
  borderLeft = true,
}) {
  const stats = useMemo(() => computeStats(values), [values])
  const binned = useMemo(() => computeBins(values, binCount), [values, binCount])

  const plotW = Math.max(10, width - MARGIN.left - MARGIN.right)
  const plotH = Math.max(10, height - MARGIN.top - MARGIN.bottom)

  const maxCount = Math.max(1, ...binned.map((b) => b.count))
  const xPad = xPaddingEnabled ? Math.min(plotW * 0.25, Math.max(0, plotW * (Number(xPaddingPercent || 0) / 100))) : 0
  const innerPlotW = Math.max(10, plotW - xPad * 2)
  const barW = binned.length > 0 ? Math.max(1, innerPlotW / binned.length - 1) : 0
  const yTicks = buildIntegerTicks(maxCount)
  const xTicks = buildValueTicks(binned)
  const minX = binned[0]?.binStart ?? 0
  const maxX = binned[binned.length - 1]?.binEnd ?? 1
  const xRange = maxX - minX || 1
  const xAxisLabel = labelWithUnit(xLabel, unit)
  const tickSize = Number(tickFontSize || fontSize || 10)
  const labelSize = Number(labelFontSize || fontSize + 1 || 11)
  const safeXLabelOffset = Number(xLabelOffset ?? 16)
  const safeYLabelOffset = Number(yLabelOffset ?? 18)
  const backgroundFill = backgroundMode === 'transparent'
    ? 'transparent'
    : backgroundMode === 'soft'
      ? backgroundAltColor
      : backgroundMode === 'gradient'
        ? 'url(#histogram-bg-gradient)'
        : backgroundColor
  const showGrid = gridStyle !== 'none' && grid !== 'none'
  const gridDash = gridStyle === 'dotted'
    ? '1 4'
    : gridStyle === 'dashed'
      ? '5 4'
      : undefined

  if (!values || values.length === 0) {
    return (
      <div className="histogram-container">
        <p className="histogram-empty">Sin datos para mostrar distribucion.</p>
      </div>
    )
  }

  return (
    <div className="histogram-container">
      <div className="histogram-stats">
        <table className="stats-table">
          <tbody>
            <tr>
              <td className="stat-label">N</td>
              <td className="stat-value">{stats.n}</td>
              <td className="stat-label">Media</td>
              <td className="stat-value">{stats.mean.toFixed(2)} {unit}</td>
            </tr>
            <tr>
              <td className="stat-label">Mediana</td>
              <td className="stat-value">{stats.median.toFixed(2)} {unit}</td>
              <td className="stat-label">Desv. Est.</td>
              <td className="stat-value">{stats.std.toFixed(2)} {unit}</td>
            </tr>
            <tr>
              <td className="stat-label">Min</td>
              <td className="stat-value">{stats.min.toFixed(2)} {unit}</td>
              <td className="stat-label">Max</td>
              <td className="stat-value">{stats.max.toFixed(2)} {unit}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <svg width={width} height={height} className="histogram-svg" style={{ background: 'transparent', fontFamily }}>
        <defs>
          <linearGradient id="histogram-bg-gradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={backgroundColor} />
            <stop offset="100%" stopColor={backgroundAltColor} />
          </linearGradient>
        </defs>
        <rect x="0" y="0" width={width} height={height} fill={backgroundFill} />

        {/* Grid */}
        {yTicks.map((tick) => {
          const y = MARGIN.top + plotH - (tick / maxCount) * plotH
          return showGrid && ['y', 'both'].includes(grid) ? (
            <line key={`ygrid-${tick}`} x1={MARGIN.left} y1={y} x2={MARGIN.left + plotW} y2={y} stroke={gridColor} strokeWidth="0.5" strokeDasharray={gridDash} />
          ) : null
        })}
        {showGrid && ['x', 'both'].includes(grid) ? xTicks.map((tick) => {
          const x = MARGIN.left + xPad + ((tick - minX) / xRange) * innerPlotW
          return <line key={`xgrid-${tick}`} x1={x} y1={MARGIN.top} x2={x} y2={MARGIN.top + plotH} stroke={gridColor} strokeWidth="0.35" strokeDasharray={gridDash} />
        }) : null}

        {/* Plot borders */}
        {borderTop ? <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left + plotW} y2={MARGIN.top} stroke={axisColor} /> : null}
        {borderRight ? <line x1={MARGIN.left + plotW} y1={MARGIN.top} x2={MARGIN.left + plotW} y2={MARGIN.top + plotH} stroke={axisColor} /> : null}
        {borderBottom ? <line x1={MARGIN.left} y1={MARGIN.top + plotH} x2={MARGIN.left + plotW} y2={MARGIN.top + plotH} stroke={axisColor} /> : null}
        {borderLeft ? <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={MARGIN.top + plotH} stroke={axisColor} /> : null}

        {/* Y-axis labels */}
        {yTicks.map((tick) => {
          const y = MARGIN.top + plotH - (tick / maxCount) * plotH
          return (
            <g key={tick}>
              <text x={MARGIN.left - 10} y={y + 4} textAnchor="end" fontSize={tickSize} fill={fontColor}>
                {tick}
              </text>
            </g>
          )
        })}

        {/* Bars */}
        {binned.map((b, i) => {
          const x = MARGIN.left + xPad + i * (barW + 1)
          const barH = (b.count / maxCount) * plotH
          const y = MARGIN.top + plotH - barH
          return (
            <rect
              key={i}
              x={x}
              y={y}
              width={barW}
              height={barH}
              fill={barColor}
              stroke={strokeColor}
              strokeWidth="0.5"
              rx="1"
            >
              <title>{`${b.binStart.toFixed(1)}-${b.binEnd.toFixed(1)} ${unit}: ${b.count}`}</title>
            </rect>
          )
        })}

        {/* X-axis ticks and labels */}
        {xTicks.map((tick) => {
          const x = MARGIN.left + xPad + ((tick - minX) / xRange) * innerPlotW
          return (
            <g key={`xtick-${tick}`}>
              {borderBottom ? <line x1={x} y1={MARGIN.top + plotH} x2={x} y2={MARGIN.top + plotH + 4} stroke={axisColor} /> : null}
              <text x={x} y={MARGIN.top + plotH + tickSize + 11} textAnchor="middle" fontSize={tickSize} fill={fontColor}>
                {formatTick(tick)}
              </text>
            </g>
          )
        })}

        <text x={MARGIN.left + plotW / 2} y={height - safeXLabelOffset} textAnchor="middle" fontSize={labelSize} fill={fontColor}>
          {xAxisLabel}
        </text>
        <text
          x={safeYLabelOffset}
          y={MARGIN.top + plotH / 2}
          textAnchor="middle"
          fontSize={labelSize}
          fill={fontColor}
          transform={`rotate(-90 ${safeYLabelOffset} ${MARGIN.top + plotH / 2})`}
        >
          {yLabel || 'Frecuencia'}
        </text>
      </svg>
    </div>
  )
}

/**
 * Export histogram data as CSV string.
 */
export function exportHistogramCsv(values, unit = 'px', bins = 20) {
  const stats = computeStats(values)
  const binned = computeBins(values, bins)
  let csv = `Estadisticas de Diametros (${unit})\n`
  csv += `N,Media,Mediana,Desv.Est.,Min,Max\n`
  csv += `${stats.n},${stats.mean.toFixed(4)},${stats.median.toFixed(4)},${stats.std.toFixed(4)},${stats.min.toFixed(4)},${stats.max.toFixed(4)}\n\n`
  csv += `Distribucion (${bins} bins)\n`
  csv += `BinInicio,BinFin,Conteo\n`
  for (const b of binned) {
    csv += `${b.binStart.toFixed(4)},${b.binEnd.toFixed(4)},${b.count}\n`
  }
  return csv
}
