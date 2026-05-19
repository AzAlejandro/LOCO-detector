const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8011'

async function parseJsonBody(res) {
  const text = await res.text()
  if (!text || !text.trim()) return {}
  try {
    return JSON.parse(text)
  } catch (err) {
    const e = new Error('Respuesta invalida del servidor (JSON malformado).')
    e.cause = err
    e.raw = text.slice(0, 240)
    throw e
  }
}

async function handle(res) {
  const data = await parseJsonBody(res)
  if (!res.ok) {
    const msg = data?.detail || data?.status || `HTTP ${res.status}`
    const err = new Error(String(msg))
    err.status = res.status
    err.payload = data
    throw err
  }
  return data
}

export function b64ToDataUrl(b64, mime = 'image/png') {
  if (!b64) return ''
  return `data:${mime};base64,${b64}`
}

export async function apiGet(path) {
  return handle(await fetch(`${API_BASE}${path}`))
}

export async function apiPost(path, payload) {
  return handle(await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }))
}

export async function apiForm(path, formData) {
  return handle(await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    body: formData,
  }))
}

export { API_BASE }
