"""Partial-word buffer, suggestion confirmation, and undo operations."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.word_suggestions import WordSuggester


@dataclass
class WordState:
    partial: str = ""
    suggestions: list[str] = field(default_factory=list)
    confirmed_words: list[str] = field(default_factory=list)
    last_accepted_letter: str = ""


class WordBuilder:
    def __init__(self, suggester: WordSuggester | None = None):
        self.suggester = suggester or WordSuggester()
        self.state = WordState()

    def add_letter(self, letter: str) -> WordState:
        letter = letter.upper()
        if not letter.isalpha() or len(letter) != 1:
            return self.state
        self.state.partial += letter
        self.state.last_accepted_letter = letter
        self._refresh()
        return self.state

    def undo_letter(self) -> WordState:
        self.state.partial = self.state.partial[:-1]
        self.state.last_accepted_letter = self.state.partial[-1:] if self.state.partial else ""
        self._refresh()
        return self.state

    def clear_word(self) -> WordState:
        self.state.partial = ""
        self.state.suggestions = []
        self.state.last_accepted_letter = ""
        return self.state

    def confirm(self, word: str | None = None) -> WordState:
        """Confirm best suggestion or provided word; clear partial."""
        choice = (word or (self.state.suggestions[0] if self.state.suggestions else self.state.partial)).strip()
        if not choice and self.state.partial:
            choice = self.state.partial
        if choice:
            self.state.confirmed_words.append(choice.upper())
        self.clear_word()
        self.suggester.set_context(self.state.confirmed_words)
        return self.state

    def undo_word(self) -> WordState:
        if self.state.confirmed_words:
            self.state.confirmed_words.pop()
            self.suggester.set_context(self.state.confirmed_words)
        return self.state

    def clear_sentence(self) -> WordState:
        self.state.confirmed_words = []
        self.clear_word()
        self.suggester.set_context([])
        return self.state

    def _refresh(self) -> None:
        self.suggester.set_context(self.state.confirmed_words)
        self.state.suggestions = self.suggester.suggest(self.state.partial)
