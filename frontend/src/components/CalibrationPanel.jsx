import React from 'react'

/**
 * CalibrationPanel component for scale calibration (px -> nm/um).
 *
 * Props:
 *   calibration: { enabled, known_nm, pixel_distance, nm_per_px, unit }
 *   onChange: (updatedCalibration) => void
 *   onSave: () => void
 *   onLoad: () => void
 *   onDelete: () => void
 *   loading: boolean
 *   imageId: string
 */

const UNITS = [
  { value: 'nm', label: 'nm' },
  { value: 'um', label: 'um' },
]

export default function CalibrationPanel({ calibration, onChange, onSave, onLoad, onDelete, loading, imageId }) {
  if (!calibration) return null

  const handleToggle = () => {
    onChange({ ...calibration, enabled: !calibration.enabled })
  }

  const handleChange = (field, value) => {
    const updated = { ...calibration, [field]: value }
    // Auto-calculate nm_per_px when known_nm or pixel_distance changes
    if (field === 'known_nm' || field === 'pixel_distance') {
      const px = field === 'pixel_distance' ? parseFloat(value) : parseFloat(calibration.pixel_distance)
      const nm = field === 'known_nm' ? parseFloat(value) : parseFloat(calibration.known_nm)
      if (px > 0 && nm > 0) {
        updated.nm_per_px = nm / px
      }
    }
    onChange(updated)
  }

  const factor = calibration.pixel_distance > 0
    ? (calibration.known_nm / calibration.pixel_distance).toFixed(4)
    : '—'

  return (
    <div className="calibration-panel">
      <h3 className="calibration-title">
        Calibracion de Escala
        {imageId && <span className="calibration-image-id">({imageId})</span>}
      </h3>

      <label className="calibration-toggle">
        <input type="checkbox" checked={calibration.enabled} onChange={handleToggle} />
        <span>Activar calibracion</span>
      </label>

      <div className="calibration-fields">
        <label className="field">
          <span>Unidad</span>
          <div className="calibration-unit-group">
            {UNITS.map((u) => (
              <button
                key={u.value}
                className={`btn-small ${calibration.unit === u.value ? 'active' : ''}`}
                onClick={() => handleChange('unit', u.value)}
              >
                {u.label}
              </button>
            ))}
          </div>
        </label>

        <label className="field">
          <span>Valor conocido ({calibration.unit})</span>
          <input
            type="number"
            min="0.001"
            step="any"
            value={calibration.known_nm}
            onChange={(e) => handleChange('known_nm', e.target.value)}
            disabled={loading}
          />
        </label>

        <label className="field">
          <span>Distancia en pixeles (px)</span>
          <input
            type="number"
            min="1"
            step="any"
            value={calibration.pixel_distance}
            onChange={(e) => handleChange('pixel_distance', e.target.value)}
            disabled={loading}
          />
        </label>
      </div>

      <div className="calibration-result">
        <strong>Factor:</strong> {factor} {calibration.unit}/px
      </div>

      <div className="calibration-actions">
        <button onClick={onSave} disabled={loading || !imageId} className="btn-small">
          Guardar
        </button>
        <button onClick={onLoad} disabled={loading || !imageId} className="btn-small">
          Cargar
        </button>
        <button onClick={onDelete} disabled={loading || !imageId} className="btn-small btn-danger">
          Eliminar
        </button>
      </div>
    </div>
  )
}
