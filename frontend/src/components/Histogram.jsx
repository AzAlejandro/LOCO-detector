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

const MARGIN = { top: 20, right: 20, bottom: 50, left: 60 }

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

export default function Histogram({ values, unit = 'px', bins: binCount = 20, width = 500, height = 250 }) {
  const stats = useMemo(() => computeStats(values), [values])
  const binned = useMemo(() => computeBins(values, binCount), [values, binCount])

  const plotW = width - MARGIN.left - MARGIN.right
  const plotH = height - MARGIN.top - MARGIN.bottom

  const maxCount = Math.max(1, ...binned.map((b) => b.count))
  const barW = binned.length > 0 ? Math.max(1, plotW / binned.length - 1) : 0

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
      <svg width={width} height={height} className="histogram-svg">
        {/* Axes */}
        <line x1={MARGIN.left} y1={MARGIN.top + plotH} x2={MARGIN.left + plotW} y2={MARGIN.top + plotH} stroke="#888" />
        <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={MARGIN.top + plotH} stroke="#888" />

        {/* Y-axis labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
          const y = MARGIN.top + plotH - frac * plotH
          const val = Math.round(frac * maxCount)
          return (
            <g key={frac}>
              <text x={MARGIN.left - 8} y={y + 4} textAnchor="end" fontSize="10" fill="#888">
                {val}
              </text>
              <line x1={MARGIN.left} y1={y} x2={MARGIN.left + plotW} y2={y} stroke="#eee" strokeWidth="0.5" />
            </g>
          )
        })}

        {/* Bars */}
        {binned.map((b, i) => {
          const x = MARGIN.left + i * (barW + 1)
          const barH = (b.count / maxCount) * plotH
          const y = MARGIN.top + plotH - barH
          return (
            <rect
              key={i}
              x={x}
              y={y}
              width={barW}
              height={barH}
              fill="#4a90d9"
              stroke="#2a6cb8"
              strokeWidth="0.5"
              rx="1"
            >
              <title>{`${b.binStart.toFixed(1)}-${b.binEnd.toFixed(1)} ${unit}: ${b.count}`}</title>
            </rect>
          )
        })}

        {/* X-axis label */}
        <text x={MARGIN.left + plotW / 2} y={height - 5} textAnchor="middle" fontSize="11" fill="#888">
          Diametro ({unit})
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
