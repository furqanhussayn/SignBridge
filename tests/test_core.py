"""Unit tests for SignBridge core components (no webcam required)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.confirmation_gesture import ConfirmationDetector, is_thumbs_up
from src.prediction_smoothing import LetterSmoother
from src.sentence_engine import SentenceEngine
from src.word_builder import WordBuilder
from src.word_suggestions import WordSuggester


def test_label_map_has_24_letters():
    assert len(config.SUPPORTED_LETTERS) == 24
    assert "J" not in config.SUPPORTED_LETTERS
    assert "Z" not in config.SUPPORTED_LETTERS
    assert "A" in config.SUPPORTED_LETTERS
    assert "Y" in config.SUPPORTED_LETTERS


def test_word_suggestions_prefix():
    s = WordSuggester()
    suggestions = s.suggest("nee")
    assert suggestions, "Expected suggestions for prefix NEE"
    joined = " ".join(suggestions)
    assert "NEED" in joined or suggestions[0].startswith("NEE")


def test_word_builder_confirm_and_undo():
    wb = WordBuilder()
    for ch in "NEE":
        wb.add_letter(ch)
    assert wb.state.partial == "NEE"
    wb.confirm("NEED")
    assert wb.state.confirmed_words == ["NEED"]
    assert wb.state.partial == ""
    wb.undo_word()
    assert wb.state.confirmed_words == []


def test_repeated_letter_requires_reset():
    sm = LetterSmoother()
    # Feed stable L many times
    accepted = []
    for _ in range(20):
        out = sm.update("L", 0.95, landmarks=np.zeros((21, 3), dtype=np.float32))
        if out.accepted:
            accepted.append(out.letter)
    assert accepted.count("L") == 1
    # Simulate hand lost (neutral) then same letter again
    sm.notify_hand_lost()
    for _ in range(20):
        out = sm.update("L", 0.95, landmarks=np.zeros((21, 3), dtype=np.float32))
        if out.accepted:
            accepted.append(out.letter)
    assert accepted.count("L") == 2


def test_low_confidence_unknown():
    sm = LetterSmoother()
    out = sm.update("A", 0.1)
    assert out.state == "unknown"
    assert out.accepted is False

def test_strict_live_confidence_gate():
    sm = LetterSmoother()
    accepted = []
    for _ in range(20):
        out = sm.update("P", config.CONFIDENCE_THRESHOLD - 0.01)
        if out.accepted:
            accepted.append(out.letter)
    assert accepted == []


def _thumbs_up_landmarks() -> np.ndarray:
    lm = np.zeros((21, 3), dtype=np.float32)
    # Wrist
    lm[0] = [0.5, 0.7, 0]
    # Thumb extended up
    lm[2] = [0.55, 0.55, 0]
    lm[4] = [0.55, 0.35, 0]
    # Other fingers folded (tips closer to wrist than PIPs)
    for tip, pip, mcp, x in [
        (8, 6, 5, 0.45),
        (12, 10, 9, 0.50),
        (16, 14, 13, 0.55),
        (20, 18, 17, 0.60),
    ]:
        lm[mcp] = [x, 0.55, 0]
        lm[pip] = [x, 0.58, 0]
        lm[tip] = [x, 0.62, 0]
    return lm


def test_thumbs_up_geometry():
    assert is_thumbs_up(_thumbs_up_landmarks())
    flat = np.zeros((21, 3), dtype=np.float32)
    # All tips extended — not thumbs up
    for i in range(21):
        flat[i] = [0.5, 0.5 - i * 0.01, 0]
    assert not is_thumbs_up(flat)


def test_confirmation_cooldown():
    det = ConfirmationDetector(stable_frames=2)
    lm = _thumbs_up_landmarks()
    assert det.update(lm) is False  # streak 1
    assert det.update(lm) is True  # streak 2 -> confirm
    assert det.update(lm) is False  # cooldown


def test_sentence_need_water():
    eng = SentenceEngine()
    r = eng.generate(["NEED", "WATER"])
    assert r.sentence.lower() == "i need water."
    assert r.intent == "need"


def test_sentence_want_food():
    eng = SentenceEngine()
    r = eng.generate(["WANT", "FOOD"])
    assert r.sentence.lower() == "i want food."


def test_sentence_help():
    eng = SentenceEngine()
    r = eng.generate(["HELP"])
    assert "help" in r.sentence.lower()


def test_model_files_if_trained():
    if not config.MODEL_PATH.exists():
        pytest.skip("Model not trained yet")
    import torch
    from src.classifier import LetterClassifier

    clf = LetterClassifier()
    dummy = np.random.rand(28, 28).astype(np.float32)
    letter, conf, probs = clf.predict(dummy)
    assert letter in config.SUPPORTED_LETTERS or letter == config.UNKNOWN_LABEL
    assert 0.0 <= conf <= 1.0
    assert probs.shape == (24,)
    assert config.LABEL_MAP_PATH.exists()
    data = json.loads(config.LABEL_MAP_PATH.read_text())
    assert len(data["supported_letters"]) == 24


def test_cnn_forward_shape():
    import torch
    from src.classifier import ASLCNN

    m = ASLCNN()
    x = torch.randn(2, 1, 28, 28)
    y = m(x)
    assert y.shape == (2, 24)


def test_preprocess_output_shape():
    from src.preprocessing import preprocess_hand_crop

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out, _ = preprocess_hand_crop(frame, (100, 100, 300, 300))
    assert out.shape == (28, 28)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_speech_engine_init():
    from src.speech import SpeechEngine

    eng = SpeechEngine()
    assert eng.status == "idle"
    eng.close()


def _plausible_hand_landmarks() -> np.ndarray:
    """Rough open-hand landmark layout in normalized coords."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.50, 0.75, 0]  # wrist
    # thumb
    lm[1], lm[2], lm[3], lm[4] = [0.42, 0.65, 0], [0.38, 0.55, 0], [0.35, 0.45, 0], [0.32, 0.35, 0]
    # index → pinky MCP / tips
    xs = [0.45, 0.50, 0.55, 0.60]
    mcps = [5, 9, 13, 17]
    tips = [8, 12, 16, 20]
    for x, mcp, tip in zip(xs, mcps, tips):
        lm[mcp] = [x, 0.55, 0]
        lm[mcp + 1] = [x, 0.45, 0]
        lm[mcp + 2] = [x, 0.35, 0]
        lm[tip] = [x, 0.22, 0]
    return lm


def test_hand_quality_accepts_plausible_hand():
    from src.hand_quality import assess_hand_quality

    lm = _plausible_hand_landmarks()
    # bbox roughly covering the hand in a 480x640 frame
    bbox = (200, 80, 420, 400)
    q = assess_hand_quality(lm, bbox, (480, 640, 3), handedness_score=0.95)
    assert q.ok, q.reason


def test_hand_quality_rejects_tiny_region():
    from src.hand_quality import assess_hand_quality

    lm = _plausible_hand_landmarks()
    bbox = (300, 200, 320, 220)  # tiny
    q = assess_hand_quality(lm, bbox, (480, 640, 3), handedness_score=0.95)
    assert not q.ok
    assert "small" in q.reason


def test_hand_quality_rejects_collapsed_landmarks():
    from src.hand_quality import assess_hand_quality

    lm = np.full((21, 3), 0.5, dtype=np.float32)
    bbox = (150, 100, 400, 380)
    q = assess_hand_quality(lm, bbox, (480, 640, 3), handedness_score=0.95)
    assert not q.ok


def test_prediction_margin_gate():
    from src.hand_quality import prediction_margin_ok

    sharp = np.zeros(24, dtype=np.float32)
    sharp[0] = 0.90
    sharp[1] = 0.05
    assert prediction_margin_ok(sharp)
    flat = np.full(24, 1.0 / 24, dtype=np.float32)
    assert not prediction_margin_ok(flat)
