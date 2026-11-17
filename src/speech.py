"""Offline English TTS via pyttsx3."""
from __future__ import annotations

import threading
from typing import Optional

import config


class SpeechEngine:
    def __init__(self, rate: int = config.DEFAULT_VOICE_RATE):
        self.rate = rate
        self._lock = threading.Lock()
        self._engine = None
        self._thread: Optional[threading.Thread] = None
        self.status = "idle"  # idle | speaking | error
        self.last_error = ""
        self._stop_flag = threading.Event()

    def _ensure_engine(self):
        if self._engine is None:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)

    def speak(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.stop()
        self._stop_flag.clear()

        def _run():
            with self._lock:
                try:
                    self.status = "speaking"
                    self._ensure_engine()
                    self._engine.say(text)
                    self._engine.runAndWait()
                    self.status = "idle"
                except Exception as exc:  # noqa: BLE001
                    self.status = "error"
                    self.last_error = str(exc)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        with self._lock:
            if self._engine is not None:
                try:
                    self._engine.stop()
                except Exception:  # noqa: BLE001
                    pass
            self.status = "idle"

    def close(self) -> None:
        self.stop()
        self._engine = None
