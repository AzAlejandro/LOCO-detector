import React, { useState, useCallback, useRef, useEffect } from 'react'

const ORIGIN_LABELS = {
  manual: 'Manual',
  modelo: 'Modelo',
  modelo_modificado: 'Modificado',
}

const ORIGIN_COLORS = {
  manual: '#198754',
  modelo: '#0d6efd',
  modelo_modificado: '#f59f00',
}

let _showFn = null

export function showOriginToast(origin) {
  if (_showFn) _showFn(origin)
}

export default function OriginToast() {
  const [visible, setVisible] = useState(false)
  const [origin, setOrigin] = useState('')
  const [fading, setFading] = useState(false)
  const timerRef = useRef(null)
  const fadeTimerRef = useRef(null)

  const show = useCallback((newOrigin) => {
    if (timerRef.current) clearTimeout(timerRef.current)
    if (fadeTimerRef.current) clearTimeout(fadeTimerRef.current)
    setOrigin(newOrigin)
    setFading(false)
    setVisible(true)
    timerRef.current = setTimeout(() => {
      setFading(true)
      fadeTimerRef.current = setTimeout(() => {
        setVisible(false)
        setFading(false)
      }, 400)
    }, 1500)
  }, [])

  useEffect(() => {
    _showFn = show
    return () => {
      _showFn = null
      if (timerRef.current) clearTimeout(timerRef.current)
      if (fadeTimerRef.current) clearTimeout(fadeTimerRef.current)
    }
  }, [show])

  if (!visible) return null

  const label = ORIGIN_LABELS[origin] || origin
  const color = ORIGIN_COLORS[origin] || '#6c757d'

  return (
    <div
      className={`origin-toast ${fading ? 'origin-toast--fading' : ''}`}
      style={{ '--toast-color': color }}
    >
      Cambio de etiqueta a: <strong>{label}</strong>
    </div>
  )
}
