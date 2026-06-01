import React from 'react'

const UNITS = [
  { value: 'nm', label: 'nm' },
  { value: 'um', label: 'um' },
]

export default function CalibrationPanel({
  calibration,
  onChange,
  onSave,
  onLoad,
  onDelete,
  onStartDraw,
  onClearLine,
  drawing,
  loading,
  imageId,
}) {
  if (!calibration) return null

  const handleToggle = () => {
    onChange({ ...calibration, enabled: !calibration.enabled })
  }

  const handleChange = (field, value) => {
    onChange({ ...calibration, [field]: value })
  }

  const hasLine = ['line_x1', 'line_y1', 'line_x2', 'line_y2'].every((key) => Number.isFinite(Number(calibration[key])))
  const pixelDistance = Number(calibration.pixel_distance || 0)
  const factor = pixelDistance > 0 && Number(calibration.known_value || 0) > 0
    ? Number(calibration.unit_per_px || (calibration.known_value / pixelDistance)).toFixed(4)
    : '0.0000'

  return (
    <div className="calibration-panel">
      <h3 className="calibration-title">
        Calibracion de Escala
        {imageId ? <span className="calibration-image-id">({imageId})</span> : null}
      </h3>

      <label className="calibration-toggle">
        <input type="checkbox" checked={!!calibration.enabled} onChange={handleToggle} />
        <span>Activar calibracion</span>
      </label>

      <div className="calibration-fields">
        <label className="field">
          <span>Unidad</span>
          <div className="calibration-unit-group">
            {UNITS.map((u) => (
              <button
                key={u.value}
                type="button"
                className={`btn-small ${calibration.unit === u.value ? 'active' : ''}`}
                onClick={() => handleChange('unit', u.value)}
                disabled={loading}
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
            value={calibration.known_value}
            onChange={(e) => handleChange('known_value', e.target.value)}
            disabled={loading}
          />
        </label>

        <div className="calibration-readout">
          <strong>Distancia en pixeles:</strong> {pixelDistance > 0 ? pixelDistance.toFixed(2) : 'sin linea'}
        </div>

        <div className="calibration-actions">
          <button type="button" onClick={onStartDraw} disabled={loading || !imageId} className={`btn-small ${drawing ? 'active' : ''}`}>
            {drawing ? 'Dibujando...' : 'Dibujar linea'}
          </button>
          <button type="button" onClick={onClearLine} disabled={loading || !hasLine} className="btn-small">
            Limpiar linea
          </button>
        </div>
      </div>

      <div className="calibration-result">
        <strong>Factor:</strong> {factor} {calibration.unit}/px
      </div>

      <p className="small calibration-help">
        Dibuja una linea horizontal sobre la barra de escala en la imagen TEM. La longitud en px se calcula sola.
      </p>

      <div className="calibration-actions">
        <button onClick={onSave} disabled={loading || !imageId || !hasLine} className="btn-small">
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
