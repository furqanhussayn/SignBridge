"""Reject face-like / low-quality MediaPipe detections before letter recognition."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import config

# MediaPipe landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_TIP = 1, 2, 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_MCP, RING_TIP = 13, 16
PINKY_MCP, PINKY_TIP = 17, 20
FINGER_MCPS = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
FINGER_TIPS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)


@dataclass
class HandQualityResult:
    ok: bool
    reason: str
    score: float  # 0..1 composite quality


def _span(landmarks: np.ndarray, axis: int) -> float:
    return float(landmarks[:, axis].max() - landmarks[:, axis].min())


def assess_hand_quality(
    landmarks: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    handedness_score: float = 1.0,
) -> HandQualityResult:
    """
    Gate MediaPipe results that are likely faces, partial blobs, or tiny/noisy tracks.

    Returns ok=False with a short reason when the crop should not feed the CNN.
    """
    if landmarks is None or len(landmarks) < 21:
        return HandQualityResult(False, "no landmarks", 0.0)

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    area_frac = (bw * bh) / max(1, w * h)

    # Too small / too large to be a signing hand crop
    if area_frac < config.MIN_HAND_AREA_FRAC:
        return HandQualityResult(False, "hand too small", area_frac)
    if area_frac > config.MAX_HAND_AREA_FRAC:
        return HandQualityResult(False, "region too large", area_frac)

    aspect = bw / float(bh)
    if aspect < config.MIN_BBOX_ASPECT or aspect > config.MAX_BBOX_ASPECT:
        return HandQualityResult(False, "odd bbox shape", min(1.0, aspect / 2))

    # Landmark geometric extent (normalized image coords)
    x_span = _span(landmarks, 0)
    y_span = _span(landmarks, 1)
    if x_span < config.MIN_LANDMARK_SPAN or y_span < config.MIN_LANDMARK_SPAN:
        return HandQualityResult(False, "landmarks collapsed", min(x_span, y_span))

    # Faces tend to produce a tight blob with weak finger extension structure.
    # Real hands usually have fingertips meaningfully away from the wrist / palm.
    wrist = landmarks[WRIST, :2]
    tip_dists = [float(np.linalg.norm(landmarks[t, :2] - wrist)) for t in FINGER_TIPS]
    mean_tip = float(np.mean(tip_dists))
    max_tip = float(np.max(tip_dists))
    if max_tip < config.MIN_FINGERTIP_EXTENT:
        return HandQualityResult(False, "no finger extent", max_tip)

    # Palm width vs finger length — faces lack a clear MCP row span
    mcp_pts = landmarks[list(FINGER_MCPS), :2]
    palm_width = float(np.linalg.norm(mcp_pts[0] - mcp_pts[-1]))
    if palm_width < config.MIN_PALM_WIDTH:
        return HandQualityResult(False, "palm too narrow", palm_width)

    # Finger tips should not all cluster near the same point (face blob)
    tip_pts = landmarks[list(FINGER_TIPS), :2]
    tip_spread = float(
        np.mean([np.linalg.norm(tip_pts[i] - tip_pts.mean(axis=0)) for i in range(4)])
    )
    if tip_spread < config.MIN_TIP_SPREAD:
        return HandQualityResult(False, "tips clustered", tip_spread)

    # Handedness confidence from MediaPipe
    if handedness_score < config.MIN_HANDEDNESS_SCORE:
        return HandQualityResult(False, "weak hand score", handedness_score)

    # Prefer detections in the lower/central frame for signing; face is often upper-center.
    cy = (y1 + y2) / (2.0 * h)
    face_zone_penalty = 0.0
    if cy < config.FACE_ZONE_Y_MAX and area_frac > config.FACE_ZONE_MIN_AREA:
        # Upper-frame large blob is suspicious — require stronger finger structure
        if mean_tip < config.FACE_ZONE_MIN_TIP_EXTENT:
            return HandQualityResult(False, "face-like upper region", mean_tip)
        face_zone_penalty = 0.15

    quality = float(
        np.clip(
            0.25 * min(1.0, area_frac / 0.12)
            + 0.25 * min(1.0, max_tip / 0.25)
            + 0.20 * min(1.0, palm_width / 0.15)
            + 0.15 * min(1.0, tip_spread / 0.08)
            + 0.15 * handedness_score
            - face_zone_penalty,
            0.0,
            1.0,
        )
    )
    if quality < config.MIN_HAND_QUALITY:
        return HandQualityResult(False, "low quality", quality)

    return HandQualityResult(True, "ok", quality)


def prediction_margin_ok(probs: Optional[np.ndarray]) -> bool:
    """Require the top class to clearly beat the runner-up."""
    if probs is None or len(probs) < 2:
        return True
    ranked = np.sort(probs)[::-1]
    return float(ranked[0] - ranked[1]) >= config.MIN_PREDICTION_MARGIN
