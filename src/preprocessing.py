"""Webcam hand crop -> ASL-MNIST-compatible 28x28 grayscale input."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

import config


def _center_on_square(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=gray.dtype)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = gray
    return canvas


def preprocess_hand_crop(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    debug: bool = False,
) -> tuple[np.ndarray, Optional[dict]]:
    """
    Crop hand region and transform into float32 (28, 28) in [0, 1].

    Inspected ASL-MNIST samples: bright grayscale hands (mean ~150–170),
    soft shading, relatively plain backgrounds — NOT pure binary silhouettes.

    Strategy:
      1. Crop with padding (bbox already padded by detector)
      2. Grayscale + CLAHE for lighting robustness
      3. Mild blur
      4. Soft background suppression via Otsu without destroying shading
      5. Center on square, resize 28x28, match dataset brightness range
    """
    x1, y1, x2, y2 = bbox
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        blank = np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)
        return blank, None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Keep the denser foreground as hand; ASL-MNIST hands are light
    if mask.mean() < 127:
        mask = 255 - mask

    # Soft blend: keep grayscale texture; darken background like dataset samples
    bg = np.full_like(gray, 40)
    masked = np.where(mask > 0, gray, bg).astype(np.uint8)
    if not np.any(mask > 0) or float(np.mean(gray[mask > 0])) < 5:
        masked = gray

    square = _center_on_square(masked)
    resized = cv2.resize(square, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
    resized = cv2.normalize(resized, None, 30, 220, cv2.NORM_MINMAX)

    out = resized.astype(np.float32) / 255.0

    debug_info = None
    if debug:
        debug_info = {
            "crop_gray": gray,
            "mask": mask,
            "masked": masked,
            "model_input_u8": (out * 255).astype(np.uint8),
        }
    return out, debug_info


def preprocess_for_display(image_28: np.ndarray) -> np.ndarray:
    """Upscale 28x28 float image for debug UI."""
    u8 = (np.clip(image_28, 0, 1) * 255).astype(np.uint8)
    return cv2.resize(u8, (112, 112), interpolation=cv2.INTER_NEAREST)
