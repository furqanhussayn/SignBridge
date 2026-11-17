"""Offline predictive word suggestions using SymSpell + wordfreq."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from symspellpy import SymSpell, Verbosity

import config

try:
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover
    def zipf_frequency(word: str, lang: str = "en") -> float:
        return 0.0


# Compact high-value fallback lexicon if dictionary file is missing
_FALLBACK_WORDS = """
the be to of and a in that have i it for not on with he as you do at this
but his by from they we say her she or an will my one all would there their
what so up out if about who get which go me when make can like time no just
him know take people into year your good some could them see other than then
now look only come its over think also back after use two how our work first
well way even new want because any these give day most us need water food
help please thank hello yes no stop start more less open close please
hungry thirsty tired happy sad pain bathroom doctor nurse medicine home school
friend family mother father brother sister love hate cold hot today tomorrow
yes okay sorry thanks need help water food want please
needle needs needed needing needles hello world name is are was were been
""".split()


class WordSuggester:
    def __init__(self, dictionary_path: Optional[Path] = None):
        self.sym = SymSpell(max_dictionary_edit_distance=config.MAX_EDIT_DISTANCE, prefix_length=7)
        path = Path(dictionary_path or config.DICTIONARY_PATH)
        loaded = False
        if path.exists():
            loaded = self.sym.load_dictionary(str(path), term_index=0, count_index=1)
        if not loaded:
            self._build_fallback()
        self._context_boost: dict[str, float] = {}

    def _build_fallback(self) -> None:
        config.DICTIONARY_DIR.mkdir(parents=True, exist_ok=True)
        # Build a small frequency dictionary from fallback + wordfreq when available
        lines = []
        seen = set()
        for w in _FALLBACK_WORDS:
            w = w.lower().strip()
            if not w or w in seen:
                continue
            seen.add(w)
            # Approximate counts from zipf: 10^(zipf-3)
            z = zipf_frequency(w, "en")
            count = max(1, int(10 ** max(z - 3, 0)))
            self.sym.create_dictionary_entry(w, count)
            lines.append(f"{w} {count}")
        # Add many common words via a seed list expansion from wordfreq top words if present
        try:
            from wordfreq import top_n_list

            for w in top_n_list("en", 20000):
                if not re.fullmatch(r"[a-z]+", w):
                    continue
                if w in seen:
                    continue
                seen.add(w)
                z = zipf_frequency(w, "en")
                count = max(1, int(10 ** max(z - 3, 0)))
                self.sym.create_dictionary_entry(w, count)
                lines.append(f"{w} {count}")
        except Exception:  # noqa: BLE001
            pass
        config.DICTIONARY_PATH.write_text("\n".join(lines) + "\n")

    def set_context(self, confirmed_words: list[str]) -> None:
        """Boost words that commonly follow previous confirmed content."""
        self._context_boost.clear()
        for w in confirmed_words[-3:]:
            self._context_boost[w.lower()] = 1.0

    def suggest(self, prefix: str, limit: int = config.MAX_SUGGESTIONS) -> list[str]:
        prefix = re.sub(r"[^A-Za-z]", "", prefix).lower()
        if len(prefix) < config.MIN_PREFIX_LEN:
            return []

        candidates: dict[str, float] = {}

        # Prefix matches via lookup with large edit distance capped by remaining letters
        # SymSpell compound / lookup for close completions
        results = self.sym.lookup(
            prefix,
            Verbosity.CLOSEST,
            max_edit_distance=min(config.MAX_EDIT_DISTANCE, max(0, len(prefix) // 2)),
            include_unknown=False,
        )
        for r in results:
            word = r.term
            if word.startswith(prefix) or self._near_prefix(word, prefix):
                candidates[word] = self._score(word, prefix, r.count)

        # Explicit prefix scan of dictionary for completions
        # Use lookup_compound lightly + transfer_casing
        # Brute prefix from term dictionary keys if available
        if hasattr(self.sym, "_words"):
            for word, count in self.sym._words.items():
                if word.startswith(prefix):
                    candidates[word] = self._score(word, prefix, count)

        ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
        out = []
        for w, _ in ranked:
            if w not in out:
                out.append(w.upper())
            if len(out) >= limit:
                break

        # If nothing found, return the prefix itself as a custom word option
        if not out and prefix:
            out = [prefix.upper()]
        return out

    def _near_prefix(self, word: str, prefix: str) -> bool:
        if abs(len(word) - len(prefix)) > config.MAX_EDIT_DISTANCE:
            return False
        # Allow one early typo in prefix region
        n = min(len(word), len(prefix))
        mismatches = sum(1 for i in range(n) if word[i] != prefix[i])
        return mismatches <= 1 and word.startswith(prefix[: max(0, len(prefix) - 1)])

    def _score(self, word: str, prefix: str, count: int) -> float:
        z = zipf_frequency(word, "en")
        prefix_bonus = 3.0 if word.startswith(prefix) else 0.0
        length_penalty = 0.05 * max(0, len(word) - len(prefix))
        ctx = 0.5 if word in self._context_boost else 0.0
        return z + prefix_bonus - length_penalty + ctx + min(2.0, np_log_count(count))


def np_log_count(count: int) -> float:
    import math

    return math.log10(max(1, count))
