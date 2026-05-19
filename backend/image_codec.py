from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def normalize_any_to_uint8(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return np.ascontiguousarray(a)
    if a.size == 0:
        return np.zeros_like(a, dtype=np.uint8)
    f = np.asarray(a, dtype=np.float32)
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(np.min(f))
    mx = float(np.max(f))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(f, dtype=np.uint8)
    if mn >= 0.0 and mx <= 1.0:
        out = np.clip(f * 255.0, 0.0, 255.0)
    elif mn >= 0.0 and mx <= 255.0:
        out = np.clip(f, 0.0, 255.0)
    else:
        out = (f - mn) / (mx - mn)
        out = np.clip(out * 255.0, 0.0, 255.0)
    return np.ascontiguousarray(out.astype(np.uint8))


def to_uint8_rgb(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = normalize_any_to_uint8(arr)
        arr = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim != 3 or arr.shape[2] != 3:
        return None
    if arr.dtype != np.uint8:
        arr = normalize_any_to_uint8(arr)
    return np.ascontiguousarray(arr)


def to_gray_u8(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    rgb = to_uint8_rgb(image)
    if rgb is None:
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def encode_png_b64(image: np.ndarray | None) -> str:
    rgb = to_uint8_rgb(image)
    if rgb is None:
        return ''
    pil = Image.fromarray(rgb)
    bio = BytesIO()
    pil.save(bio, format='PNG')
    return base64.b64encode(bio.getvalue()).decode('ascii')


def encode_gray_png_b64(image: np.ndarray | None) -> str:
    if image is None:
        return ''
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = to_gray_u8(arr)
    arr = normalize_any_to_uint8(np.asarray(arr))
    pil = Image.fromarray(arr, mode='L')
    bio = BytesIO()
    pil.save(bio, format='PNG')
    return base64.b64encode(bio.getvalue()).decode('ascii')


def encode_display_b64(
    image: np.ndarray | None,
    *,
    png_max_chars: int = 8_000_000,
    jpeg_quality: int = 90,
) -> tuple[str, str]:
    """
    Encode for browser display.
    Uses PNG by default; falls back to JPEG when PNG payload is too large for
    reliable data-url rendering in browser UIs.
    Returns: (base64_data, mime_type)
    """
    rgb = to_uint8_rgb(image)
    if rgb is None:
        return '', 'image/png'

    png_b64 = encode_png_b64(rgb)
    if png_b64 and len(png_b64) <= int(max(1, png_max_chars)):
        return png_b64, 'image/png'

    ok, buf = cv2.imencode(
        '.jpg',
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(jpeg_quality, 30, 100))],
    )
    if ok:
        return base64.b64encode(bytes(buf)).decode('ascii'), 'image/jpeg'
    return png_b64, 'image/png'


def decode_png_b64(data: str) -> np.ndarray | None:
    if not data:
        return None
    raw = base64.b64decode(data)
    img = Image.open(BytesIO(raw))
    return np.array(img)


def apply_mask_overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 220, 80), alpha: float = 0.35) -> np.ndarray:
    rgb = to_uint8_rgb(image)
    if rgb is None:
        raise ValueError('Imagen invalida para overlay.')
    m = np.asarray(mask)
    if m.ndim == 3:
        g = to_gray_u8(m)
        if g is None:
            raise ValueError('Mascara invalida.')
        m = g
    if m.shape != rgb.shape[:2]:
        m = cv2.resize(normalize_any_to_uint8(m), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = m > 0
    out = rgb.astype(np.float32)
    overlay = np.zeros_like(out)
    overlay[:, :, 0] = color[0]
    overlay[:, :, 1] = color[1]
    overlay[:, :, 2] = color[2]
    out[m] = out[m] * (1.0 - alpha) + overlay[m] * alpha
    return np.clip(out, 0, 255).astype(np.uint8)
