"""Rolling prediction smoothing, confidence gates, and letter debounce."""
from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

import config


@dataclass
class SmoothedPrediction:
    letter: str
    confidence: float
    stability: float  # 0..1 fraction agreeing in window
    accepted: bool  # True when a new letter was committed this frame
    state: str  # "tracking" | "stable" | "unknown" | "cooldown"


class LetterSmoother:
    """
    Smooth noisy frame-level CNN outputs into stable letter commits.

    - Rolling majority vote over SMOOTHING_WINDOW frames
    - Confidence threshold
    - Require STABLE_FRAMES agreeing predictions
    - Cooldown after accept to avoid duplicates while holding a pose
    - Movement/neutral reset allows repeating the same letter (e.g. HELLO)
    """

    def __init__(self):
        self.window: deque[tuple[str, float]] = deque(maxlen=config.SMOOTHING_WINDOW)
        self._last_accepted: Optional[str] = None
        self._last_accept_time: float = 0.0
        self._last_landmarks: Optional[np.ndarray] = None
        self._pose_reset = True  # True after movement/neutral — allows same letter again

    def reset_word(self) -> None:
        self.window.clear()
        self._last_accepted = None
        self._pose_reset = True

    def notify_hand_lost(self) -> None:
        """Losing the hand counts as a neutral reset between repeated letters."""
        self.window.clear()
        self._pose_reset = True
        self._last_landmarks = None
        # Allow an immediate re-accept of the same letter after a neutral break
        self._last_accept_time = 0.0

    def _movement_reset(self, landmarks: Optional[np.ndarray]) -> None:
        if landmarks is None or self._last_landmarks is None:
            self._last_landmarks = landmarks
            return
        # Use wrist + middle fingertip displacement
        idxs = [0, 12]
        prev = self._last_landmarks[idxs, :2]
        curr = landmarks[idxs, :2]
        dist = float(np.linalg.norm(curr - prev, axis=1).mean())
        if dist >= config.MIN_HAND_MOVEMENT_RESET:
            self._pose_reset = True
            self.window.clear()
            self._last_accept_time = 0.0
        self._last_landmarks = landmarks.copy()

    def update(
        self,
        letter: Optional[str],
        confidence: float,
        landmarks: Optional[np.ndarray] = None,
    ) -> SmoothedPrediction:
        now = time.monotonic()
        self._movement_reset(landmarks)

        if letter is None or confidence < config.CONFIDENCE_THRESHOLD:
            self.window.append((config.UNKNOWN_LABEL, confidence))
            return SmoothedPrediction(
                letter=config.UNKNOWN_LABEL,
                confidence=confidence,
                stability=0.0,
                accepted=False,
                state="unknown",
            )

        self.window.append((letter, confidence))
        counts = Counter(l for l, _ in self.window)
        top_letter, top_count = counts.most_common(1)[0]
        stability = top_count / max(1, len(self.window))
        avg_conf = float(
            np.mean([c for l, c in self.window if l == top_letter] or [confidence])
        )

        if top_letter == config.UNKNOWN_LABEL or top_count < config.STABLE_FRAMES:
            return SmoothedPrediction(
                letter=top_letter if top_letter != config.UNKNOWN_LABEL else letter,
                confidence=avg_conf,
                stability=stability,
                accepted=False,
                state="tracking",
            )

        # Cooldown after recent accept
        if now - self._last_accept_time < config.LETTER_COOLDOWN_SEC:
            return SmoothedPrediction(
                letter=top_letter,
                confidence=avg_conf,
                stability=stability,
                accepted=False,
                state="cooldown",
            )

        # Suppress duplicate while still holding same pose without reset
        if top_letter == self._last_accepted and not self._pose_reset:
            return SmoothedPrediction(
                letter=top_letter,
                confidence=avg_conf,
                stability=stability,
                accepted=False,
                state="stable",
            )

        # Accept letter
        self._last_accepted = top_letter
        self._last_accept_time = now
        self._pose_reset = False
        self.window.clear()
        return SmoothedPrediction(
            letter=top_letter,
            confidence=avg_conf,
            stability=1.0,
            accepted=True,
            state="stable",
        )
