"""Threaded webcam capture so the UI stays responsive."""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

import config


class CameraWorker:
    def __init__(self, index: int = config.CAMERA_INDEX):
        self.index = index
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._running = False
        self.error: str = ""

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> bool:
        if self._running:
            return True
        self.error = ""
        cap = cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        if not cap.isOpened():
            self.error = f"Could not open camera index {self.index}"
            return False
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self) -> None:
        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                self.error = "Camera read failed"
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)  # mirror for natural UX
            with self._lock:
                self._frame = frame
            time.sleep(0.001)

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._frame = None
