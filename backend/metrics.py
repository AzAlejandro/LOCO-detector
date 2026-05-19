from __future__ import annotations

import cv2
import numpy as np


def to_binary(mask_like: np.ndarray) -> np.ndarray:
    m = np.asarray(mask_like)
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8)
    return (m > 0).astype(np.uint8)


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    p = to_binary(pred)
    g = to_binary(gt)
    inter = int(np.sum((p == 1) & (g == 1)))
    area_p = int(np.sum(p == 1))
    area_g = int(np.sum(g == 1))
    union = area_p + area_g - inter
    dice = (2.0 * inter) / max(1, area_p + area_g)
    iou = inter / max(1, union)
    return float(dice), float(iou)


def compactness(mask: np.ndarray) -> float:
    m = (to_binary(mask) * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    per = float(cv2.arcLength(c, True))
    if area <= 0.0 or per <= 0.0:
        return 0.0
    return float((4.0 * np.pi * area) / (per * per))


def border_touch_ratio(mask: np.ndarray) -> float:
    m = to_binary(mask)
    area = int(np.sum(m))
    if area == 0:
        return 0.0
    border = np.zeros_like(m, dtype=np.uint8)
    border[0, :] = 1
    border[-1, :] = 1
    border[:, 0] = 1
    border[:, -1] = 1
    touch = int(np.sum((m == 1) & (border == 1)))
    return float(touch / area)


def components_count(mask: np.ndarray) -> int:
    m = to_binary(mask)
    ncc, _ = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    return max(0, int(ncc) - 1)


def fragmentation_index(mask: np.ndarray) -> float:
    m = to_binary(mask)
    area = int(np.sum(m))
    if area <= 0:
        return 0.0
    comps = components_count(m)
    return float(comps / max(1.0, area / 1000.0))


def leakage_to_bg(mask: np.ndarray, labels: np.ndarray, radius: int = 3) -> float:
    m = to_binary(mask)
    labs = np.asarray(labels, dtype=np.uint8)
    bg = ((labs == 2) | (labs == 3)).astype(np.uint8)
    if int(np.sum(bg)) == 0:
        return 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    zone = cv2.dilate(bg, kernel, iterations=1) > 0
    if int(np.sum(zone)) == 0:
        return 0.0
    leak = int(np.sum((m == 1) & zone))
    return float(leak / max(1, int(np.sum(zone))))


def operational_metrics(mask: np.ndarray, labels: np.ndarray, runtime_ms: float) -> dict[str, float]:
    m = to_binary(mask)
    area = int(np.sum(m))
    return {
        'runtime_ms': float(runtime_ms),
        'mask_area_px': float(area),
        'compactness': compactness(m),
        'border_touch_ratio': border_touch_ratio(m),
        'components_count': float(components_count(m)),
        'fragmentation_index': fragmentation_index(m),
        'leakage_to_bg': leakage_to_bg(m, labels),
    }
