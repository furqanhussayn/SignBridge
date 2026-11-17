"""Thumbs-up confirmation via MediaPipe hand geometry (separate from alphabet CNN)."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

import config

# MediaPipe landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20


def _finger_folded(landmarks: np.ndarray, tip: int, pip: int, mcp: int) -> bool:
    """True if fingertip is closer to wrist than the PIP joint (folded toward palm)."""
    tip_d = np.linalg.norm(landmarks[tip, :2] - landmarks[WRIST, :2])
    pip_d = np.linalg.norm(landmarks[pip, :2] - landmarks[WRIST, :2])
    return tip_d < pip_d * 1.05


def _thumb_extended_up(landmarks: np.ndarray) -> bool:
    """Thumb tip clearly above thumb MCP and reasonably extended."""
    tip = landmarks[THUMB_TIP]
    mcp = landmarks[THUMB_MCP]
    wrist = landmarks[WRIST]
    # In image coords, y decreases upward
    pointing_up = tip[1] < mcp[1] - 0.03
    extended = np.linalg.norm(tip[:2] - mcp[:2]) > 0.06
    # Thumb should not be curled toward palm center
    away_from_palm = abs(tip[0] - wrist[0]) > 0.02 or pointing_up
    return bool(pointing_up and extended and away_from_palm)


def is_thumbs_up(landmarks: np.ndarray) -> bool:
    """
    Detect a thumbs-up: thumb extended upward, other four fingers folded.

    Intentionally geometric — never routed through the alphabet CNN.
    """
    if landmarks is None or len(landmarks) < 21:
        return False
    fingers_folded = all(
        [
            _finger_folded(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP),
            _finger_folded(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            _finger_folded(landmarks, RING_TIP, RING_PIP, RING_MCP),
            _finger_folded(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP),
        ]
    )
    return fingers_folded and _thumb_extended_up(landmarks)


class ConfirmationDetector:
    """Stable thumbs-up with cooldown so one gesture confirms once."""

    def __init__(self, stable_frames: int = 3):
        self.stable_frames = stable_frames
        self._streak = 0
        self._last_confirm_time = 0.0
        self.last_confirmed = False

    def update(self, landmarks: Optional[np.ndarray]) -> bool:
        self.last_confirmed = False
        now = time.monotonic()
        if now - self._last_confirm_time < config.CONFIRM_COOLDOWN_SEC:
            self._streak = 0
            return False
        if landmarks is None or not is_thumbs_up(landmarks):
            self._streak = 0
            return False
        self._streak += 1
        if self._streak >= self.stable_frames:
            self._streak = 0
            self._last_confirm_time = now
            self.last_confirmed = True
            return True
        return False

    def reset(self) -> None:
        self._streak = 0
