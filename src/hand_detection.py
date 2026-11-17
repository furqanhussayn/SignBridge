"""MediaPipe Hands wrapper — detection, landmarks, and hand crop bounding box."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

import config


@dataclass
class HandResult:
    landmarks: np.ndarray  # (21, 3) normalized x,y,z
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    handedness: str
    score: float


class HandDetector:
    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.75,
        tracking_confidence: float = 0.6,
    ):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=0,
        )
        self._drawer = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles

    def process(self, frame_bgr: np.ndarray) -> Optional[HandResult]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None

        hand_lms = result.multi_hand_landmarks[0]
        handedness = "Unknown"
        score = 0.0
        if result.multi_handedness:
            handedness = result.multi_handedness[0].classification[0].label
            score = float(result.multi_handedness[0].classification[0].score)

        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
        xs = pts[:, 0] * w
        ys = pts[:, 1] * h
        pad = config.HAND_PADDING
        x1 = int(max(0, xs.min() - pad * (xs.max() - xs.min() + 1)))
        y1 = int(max(0, ys.min() - pad * (ys.max() - ys.min() + 1)))
        x2 = int(min(w - 1, xs.max() + pad * (xs.max() - xs.min() + 1)))
        y2 = int(min(h - 1, ys.max() + pad * (ys.max() - ys.min() + 1)))

        # Ensure minimum crop size
        if x2 - x1 < 40 or y2 - y1 < 40:
            cx, cy = int(xs.mean()), int(ys.mean())
            half = 60
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(w - 1, cx + half), min(h - 1, cy + half)

        return HandResult(
            landmarks=pts, bbox=(x1, y1, x2, y2), handedness=handedness, score=score
        )

    def draw(self, frame_bgr: np.ndarray, hand: Optional[HandResult]) -> np.ndarray:
        out = frame_bgr.copy()
        if hand is None:
            return out
        x1, y1, x2, y2 = hand.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (61, 220, 151), 2)
        # Draw landmark dots
        h, w = out.shape[:2]
        for x, y, _ in hand.landmarks:
            cv2.circle(out, (int(x * w), int(y * h)), 2, (61, 220, 151), -1)
        return out

    def close(self) -> None:
        self._hands.close()
