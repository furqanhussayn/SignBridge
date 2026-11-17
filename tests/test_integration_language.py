"""Integration-style language path without webcam."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sentence_engine import SentenceEngine
from src.word_builder import WordBuilder


def test_need_water_flow():
    wb = WordBuilder()
    for ch in "NEE":
        wb.add_letter(ch)
    assert any(s.startswith("NEED") for s in wb.state.suggestions)
    wb.confirm("NEED")
    for ch in "WATER":
        wb.add_letter(ch)
    wb.confirm("WATER")
    result = SentenceEngine().generate(wb.state.confirmed_words)
    assert result.sentence == "I need water."


def test_undo_operations():
    wb = WordBuilder()
    wb.add_letter("H")
    wb.add_letter("I")
    wb.undo_letter()
    assert wb.state.partial == "H"
    wb.confirm("HI")
    wb.undo_word()
    assert wb.state.confirmed_words == []
