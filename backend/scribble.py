from __future__ import annotations

import cv2
import numpy as np

from .image_codec import decode_png_b64, to_uint8_rgb

LABEL_UNLABELED = 0
LABEL_FIBER = 1
LABEL_HALO = 2
LABEL_BACKGROUND = 3

VIS_FIBER = 128
VIS_HALO = 192
VIS_BACKGROUND = 255


def clean_scribble_speckles(labels: np.ndarray) -> np.ndarray:
    out = np.asarray(labels, dtype=np.uint8).copy()
    if out.ndim != 2 or out.size == 0:
        return out
    h, w = out.shape
    valid_values = (LABEL_FIBER, LABEL_HALO, LABEL_BACKGROUND)
    for value in valid_values:
        mask = out == value
        if not np.any(mask):
            continue
        ncc, cc = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        for cid in range(1, int(ncc)):
            comp = cc == cid
            area = int(np.sum(comp))
            if area > 2:
                continue
            ys, xs = np.where(comp)
            y0 = max(0, int(np.min(ys)) - 1)
            y1 = min(h, int(np.max(ys)) + 2)
            x0 = max(0, int(np.min(xs)) - 1)
            x1 = min(w, int(np.max(xs)) + 2)
            neigh = out[y0:y1, x0:x1]
            other = (neigh > 0) & (neigh != value)
            if int(np.sum(other)) >= 2:
                out[comp] = 0
    return out


def normalize_scribble_labels(scribble_map: np.ndarray | None, target_shape: tuple[int, int] | None = None) -> np.ndarray | None:
    if scribble_map is None:
        return None
    raw = np.asarray(scribble_map)
    alpha = None
    if raw.ndim == 3 and raw.shape[2] >= 4:
        rgba = np.asarray(raw[:, :, :4])
        if rgba.dtype != np.uint8:
            if float(np.max(rgba)) <= 1.0:
                rgba = np.clip(rgba * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                rgba = np.clip(rgba, 0.0, 255.0).astype(np.uint8)
        alpha = np.asarray(rgba[:, :, 3], dtype=np.uint8)
        arr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    elif raw.ndim == 3:
        rgb = to_uint8_rgb(raw)
        if rgb is None:
            return None
        arr = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        arr = np.asarray(raw)

    arr = np.asarray(arr, dtype=np.uint8)
    if target_shape is not None and arr.shape != target_shape:
        arr = cv2.resize(arr, (int(target_shape[1]), int(target_shape[0])), interpolation=cv2.INTER_NEAREST)
        if isinstance(alpha, np.ndarray):
            alpha = cv2.resize(alpha, (int(target_shape[1]), int(target_shape[0])), interpolation=cv2.INTER_NEAREST)

    labels = np.zeros_like(arr, dtype=np.uint8)
    labels[arr == LABEL_FIBER] = LABEL_FIBER
    labels[arr == LABEL_HALO] = LABEL_HALO
    labels[arr == LABEL_BACKGROUND] = LABEL_BACKGROUND
    maxv = int(np.max(arr)) if arr.size else 0

    if maxv > LABEL_BACKGROUND:
        fiber = (arr >= 96) & (arr <= 160) & (labels == 0)
        halo = (arr >= 176) & (arr <= 216) & (labels == 0)
        bg = (arr >= 224) & (labels == 0)
        labels[fiber] = LABEL_FIBER
        labels[halo] = LABEL_HALO
        labels[bg] = LABEL_BACKGROUND

    if isinstance(alpha, np.ndarray):
        labels[alpha < 16] = LABEL_UNLABELED

    return clean_scribble_speckles(labels)


def decode_scribble_b64(scribble_map_b64: str, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    arr = decode_png_b64(str(scribble_map_b64 or ''))
    labels = normalize_scribble_labels(arr, target_shape=target_shape)
    if labels is None:
        raise ValueError('Scribble invalido.')
    return labels


def labels_to_visual(labels: np.ndarray) -> np.ndarray:
    vis = np.zeros_like(np.asarray(labels, dtype=np.uint8), dtype=np.uint8)
    vis[labels == LABEL_FIBER] = VIS_FIBER
    vis[labels == LABEL_HALO] = VIS_HALO
    vis[labels == LABEL_BACKGROUND] = VIS_BACKGROUND
    return vis


def has_fg_bg(labels: np.ndarray) -> tuple[int, int, bool]:
    arr = np.asarray(labels, dtype=np.uint8)
    n_fg = int(np.sum(arr == LABEL_FIBER))
    n_halo = int(np.sum(arr == LABEL_HALO))
    n_bg = int(np.sum(arr == LABEL_BACKGROUND))
    # Compatibility for old runs/drafts that encoded background as label 2.
    n_bg_effective = n_bg if n_bg > 0 else n_halo
    return n_fg, n_bg_effective, bool(n_fg > 0 and n_bg_effective > 0)


def scribble_label_counts(labels: np.ndarray) -> dict[str, int]:
    arr = np.asarray(labels, dtype=np.uint8)
    return {
        'fiber': int(np.sum(arr == LABEL_FIBER)),
        'halo': int(np.sum(arr == LABEL_HALO)),
        'background': int(np.sum(arr == LABEL_BACKGROUND)),
        'unlabeled': int(np.sum(arr == LABEL_UNLABELED)),
    }
