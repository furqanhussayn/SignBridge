#!/usr/bin/env python3
"""SignBridge — ASL fingerspelling-to-speech assistant entry point."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from src.camera import CameraWorker
from src.classifier import LetterClassifier
from src.confirmation_gesture import ConfirmationDetector
from src.hand_detection import HandDetector
from src.hand_quality import assess_hand_quality, prediction_margin_ok
from src.prediction_smoothing import LetterSmoother
from src.preprocessing import preprocess_for_display, preprocess_hand_crop
from src.sentence_engine import SentenceEngine
from src.speech import SpeechEngine
from src.ui import SignBridgeUI
from src.word_builder import WordBuilder


class SignBridgeApp:
    def __init__(self):
        self.camera = CameraWorker()
        self.hands = HandDetector()
        self.smoother = LetterSmoother()
        self.confirmer = ConfirmationDetector()
        self.words = WordBuilder()
        self.sentences = SentenceEngine()
        self.speech = SpeechEngine()
        self.pending_letter = ""
        self.classifier = None
        self.model_error = ""
        try:
            self.classifier = LetterClassifier()
        except Exception as exc:  # noqa: BLE001
            self.model_error = str(exc)

        self.ui = SignBridgeUI(self)
        self._last_debug = None
        if self.model_error:
            self.ui.update_hand_status(f"Model: {self.model_error}")

    # ---- controls ----
    def start_camera(self) -> None:
        if not self.camera.start():
            self.ui.update_hand_status(self.camera.error or "Camera failed")
        else:
            self.ui.update_hand_status("Hand: searching…")

    def stop_camera(self) -> None:
        self.camera.stop()
        self.ui.update_hand_status("Camera stopped")

    def undo_letter(self) -> None:
        self.words.undo_letter()
        self.pending_letter = ""
        self._refresh_pending()
        self._refresh_language()

    def undo_word(self) -> None:
        self.words.undo_word()
        self.pending_letter = ""
        self._refresh_pending()
        self._refresh_language()

    def clear_word(self) -> None:
        self.words.clear_word()
        self.smoother.reset_word()
        self.pending_letter = ""
        self._refresh_pending()
        self._refresh_language()

    def clear_sentence(self) -> None:
        self.words.clear_sentence()
        self.smoother.reset_word()
        self.pending_letter = ""
        self._refresh_pending()
        self._refresh_language()

    def accept_pending_letter(self) -> None:
        if not self.pending_letter:
            return
        self.words.add_letter(self.pending_letter)
        self.pending_letter = ""
        self.smoother.reset_word()
        self._refresh_pending()
        self._refresh_language()

    def reject_pending_letter(self) -> None:
        self.pending_letter = ""
        self.smoother.reset_word()
        self._refresh_pending()

    def confirm_best(self) -> None:
        self.words.confirm()
        self.smoother.reset_word()
        self.confirmer.reset()
        self.pending_letter = ""
        self._refresh_pending()
        self._refresh_language()

    def confirm_word(self, word: str) -> None:
        self.words.confirm(word)
        self.smoother.reset_word()
        self.confirmer.reset()
        self.pending_letter = ""
        self._refresh_pending()
        self._refresh_language()

    def speak(self) -> None:
        text = self.ui.get_sentence_text()
        self.speech.speak(text)
        self.ui.update_speech_status(self.speech.status)

    def stop_speech(self) -> None:
        self.speech.stop()
        self.ui.update_speech_status("idle")

    def show_diagnostics(self) -> None:
        metrics_path = config.RESULTS_DIR / "metrics.json"
        hist_path = config.RESULTS_DIR / "train_history.json"
        lines = ["SignBridge Diagnostics", "=" * 40, ""]
        if self.model_error:
            lines.append(f"Model load error: {self.model_error}")
        lines.append(f"Model path: {config.MODEL_PATH} (exists={config.MODEL_PATH.exists()})")
        if metrics_path.exists():
            lines.append("Test metrics:")
            lines.append(json.dumps(json.loads(metrics_path.read_text()), indent=2))
        else:
            lines.append("No metrics.json — run: python -m training.evaluate")
        if hist_path.exists():
            hist = json.loads(hist_path.read_text())
            if hist.get("val_acc"):
                lines.append(f"Best recorded val_acc during train history max: {max(hist['val_acc']):.4f}")
        lines.append("")
        lines.append("Limitations: static alphabet only (no J/Z). Rule-based sentences.")
        lines.append("Webcam domain gap may reduce live accuracy vs test-set metrics.")
        self.ui.show_message("Diagnostics", "\n".join(lines))

    # ---- frame loop ----
    def on_frame(self) -> None:
        self.ui.update_speech_status(self.speech.status)
        if not self.camera.running:
            return
        frame = self.camera.read()
        if frame is None:
            return

        hand = self.hands.process(frame)
        display = self.hands.draw(frame, hand)

        if hand is None:
            self.smoother.notify_hand_lost()
            self.ui.update_hand_status("Hand: not detected")
            self.ui.update_recognition(config.UNKNOWN_LABEL, 0.0, 0.0, False)
            self.ui.update_video(display)
            return

        quality = assess_hand_quality(
            hand.landmarks, hand.bbox, frame.shape, handedness_score=hand.score
        )
        if not quality.ok:
            # Do not feed face/junk detections into the alphabet CNN
            self.smoother.notify_hand_lost()
            self.confirmer.reset()
            self.ui.update_hand_status(f"Hand: rejected ({quality.reason})")
            self.ui.update_recognition(config.UNKNOWN_LABEL, 0.0, 0.0, False)
            self.ui.update_video(display)
            return

        self.ui.update_hand_status(
            f"Hand: {hand.handedness} ({hand.score * 100:.0f}%)  Q:{quality.score * 100:.0f}%"
        )

        # Confirmation gesture checked BEFORE alphabet accept path
        if self.confirmer.update(hand.landmarks):
            if self.pending_letter:
                self.accept_pending_letter()
                cv2.putText(
                    display,
                    "LETTER CONFIRMED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (61, 220, 151),
                    2,
                )
            elif self.words.state.partial or self.words.state.suggestions:
                import cv2

                self.confirm_best()
                cv2.putText(
                    display,
                    "CONFIRMED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (61, 220, 151),
                    2,
                )
                self.ui.update_video(display)
                return

        letter, conf, stability, accepted = config.UNKNOWN_LABEL, 0.0, 0.0, False
        if self.classifier is not None:
            model_in, dbg = preprocess_hand_crop(frame, hand.bbox, debug=True)
            self._last_debug = preprocess_for_display(model_in)
            self.ui.update_debug(self._last_debug)
            pred_letter, pred_conf, probs = self.classifier.predict(model_in)

            # Skip alphabet updates while thumbs-up geometry is active
            from src.confirmation_gesture import is_thumbs_up

            if is_thumbs_up(hand.landmarks):
                letter, conf, stability, accepted = "👍", pred_conf, 0.0, False
            elif not prediction_margin_ok(probs):
                # Ambiguous softmax — treat as unknown, do not commit
                smooth = self.smoother.update(None, pred_conf, hand.landmarks)
                letter, conf, stability, accepted = (
                    config.UNKNOWN_LABEL,
                    pred_conf,
                    smooth.stability,
                    False,
                )
            else:
                smooth = self.smoother.update(pred_letter, pred_conf, hand.landmarks)
                letter, conf, stability, accepted = (
                    smooth.letter,
                    smooth.confidence,
                    smooth.stability,
                    smooth.accepted,
                )
                if accepted:
                    if config.MANUAL_LETTER_CONFIRMATION:
                        self.pending_letter = letter
                        self._refresh_pending()
                    else:
                        self.words.add_letter(letter)
                        self._refresh_language()

        self.ui.update_recognition(letter, conf, stability, accepted)
        self.ui.update_video(display)

    def _refresh_language(self) -> None:
        st = self.words.state
        result = self.sentences.generate(st.confirmed_words)
        self.ui.update_word_state(st.partial, st.suggestions, st.confirmed_words, result.sentence)

    def _refresh_pending(self) -> None:
        self.ui.update_pending_letter(self.pending_letter)

    def shutdown(self) -> None:
        self.camera.stop()
        self.speech.close()
        self.hands.close()

    def run(self) -> None:
        self._refresh_language()
        self._refresh_pending()
        self.start_camera()
        self.ui.mainloop()


def main() -> None:
    try:
        SignBridgeApp().run()
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
