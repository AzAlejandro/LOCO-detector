from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def ensure_gray_float01(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[2] in (3, 4):
        rgb = img[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 2:
        gray = np.clip(img, 0, 255).astype(np.uint8)
    else:
        raise ValueError('Formato de imagen no soportado.')
    gray_f = gray.astype(np.float32) / 255.0
    return np.clip(gray_f, 0.0, 1.0)


def _local_stats(gray_f: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    mu = cv2.blur(gray_f, (k, k))
    mu2 = cv2.blur(gray_f * gray_f, (k, k))
    var = np.clip(mu2 - mu * mu, 0.0, None)
    std = np.sqrt(var)
    return mu, std


def build_features(image: np.ndarray, variant: str = 'base') -> np.ndarray:
    gray_f = ensure_gray_float01(image)

    g1 = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=1.0)
    g2 = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=2.0)
    g4 = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=4.0)

    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    lap = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)

    mu7, std7 = _local_stats(gray_f, 7)
    lbp = local_binary_pattern((gray_f * 255.0).astype(np.uint8), P=8, R=1, method='uniform').astype(np.float32)
    lbp /= max(1.0, float(np.max(lbp)))

    feats = [gray_f, g1, g2, g4, grad, lap, mu7, std7, lbp]

    if variant in {'context', 'extended'}:
        mu15, std15 = _local_stats(gray_f, 15)
        med = cv2.medianBlur((gray_f * 255.0).astype(np.uint8), 5).astype(np.float32) / 255.0
        minf = cv2.erode(gray_f, np.ones((5, 5), dtype=np.uint8), iterations=1)
        maxf = cv2.dilate(gray_f, np.ones((5, 5), dtype=np.uint8), iterations=1)
        feats.extend([mu15, std15, med, minf, maxf])

    stack = np.stack([np.asarray(f, dtype=np.float32) for f in feats], axis=-1)
    return np.ascontiguousarray(stack)
