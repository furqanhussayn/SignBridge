"""Prepare offline SymSpell frequency dictionary using wordfreq top words."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config


def main() -> None:
    from wordfreq import top_n_list, zipf_frequency

    config.DICTIONARY_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for w in top_n_list("en", 50000):
        if not re.fullmatch(r"[a-z]+", w):
            continue
        if len(w) < 2 or len(w) > 20:
            continue
        z = zipf_frequency(w, "en")
        count = max(1, int(10 ** max(z - 3, 0)))
        lines.append(f"{w} {count}")
    config.DICTIONARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} words -> {config.DICTIONARY_PATH}")


if __name__ == "__main__":
    main()
