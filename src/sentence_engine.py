"""Lightweight offline rule-based sentence generation from confirmed content words."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SentenceResult:
    sentence: str
    intent: str
    note: str


# Content words that imply intents
NEED_WORDS = {"NEED", "NEEDS", "NEEDED", "WANT", "WANTS", "WANTED", "REQUIRE"}
HELP_WORDS = {"HELP", "SOS", "EMERGENCY", "ASSIST"}
FEELING_WORDS = {
    "HAPPY",
    "SAD",
    "TIRED",
    "ANGRY",
    "SICK",
    "HUNGRY",
    "THIRSTY",
    "COLD",
    "HOT",
    "PAIN",
    "HURT",
}
RESPONSE_WORDS = {"YES", "NO", "OK", "OKAY", "THANKS", "THANK", "SORRY", "PLEASE"}
GREETING_WORDS = {"HELLO", "HI", "HEY", "BYE", "GOODBYE"}


class SentenceEngine:
    """
    Rule / template engine — NOT full ASL-to-English translation.

    Users fingerspell English content words; this fills light grammar
    (pronouns, articles, punctuation) for speech output.
    """

    def generate(self, words: list[str]) -> SentenceResult:
        tokens = [w.strip().upper() for w in words if w and w.strip()]
        if not tokens:
            return SentenceResult("", "empty", "No confirmed words yet.")

        intent = self._detect_intent(tokens)
        sentence = self._render(tokens, intent)
        return SentenceResult(
            sentence=sentence,
            intent=intent,
            note=(
                "Rule-based English expansion from fingerspelled content words. "
                "Not ASL grammar translation."
            ),
        )

    def _detect_intent(self, tokens: list[str]) -> str:
        s = set(tokens)
        if s & GREETING_WORDS and len(tokens) <= 2:
            return "greeting"
        if s & HELP_WORDS:
            return "help"
        if s & NEED_WORDS:
            return "need"
        if s & FEELING_WORDS and not (s & NEED_WORDS):
            return "feeling"
        if s <= RESPONSE_WORDS or (len(tokens) == 1 and tokens[0] in RESPONSE_WORDS):
            return "response"
        if tokens[0] in {"WHERE", "WHAT", "WHEN", "WHO", "WHY", "HOW"}:
            return "question"
        return "statement"

    def _article(self, word: str) -> str:
        if word[:1].lower() in "aeiou":
            return "an"
        return "a"

    def _render(self, tokens: list[str], intent: str) -> str:
        lower = [t.lower() for t in tokens]

        if intent == "greeting":
            g = lower[0].capitalize()
            if len(lower) == 1:
                return f"{g}!"
            return f"{g}, {' '.join(lower[1:])}."

        if intent == "response":
            mapping = {
                "yes": "Yes.",
                "no": "No.",
                "ok": "Okay.",
                "okay": "Okay.",
                "thanks": "Thank you.",
                "thank": "Thank you.",
                "sorry": "I am sorry.",
                "please": "Please.",
            }
            if len(lower) == 1 and lower[0] in mapping:
                return mapping[lower[0]]
            return self._capitalize(" ".join(lower) + ".")

        if intent == "help":
            rest = [t for t in lower if t not in {"help", "sos", "assist", "emergency"}]
            if rest:
                return f"Please help me with {' '.join(rest)}."
            return "Please help me."

        if intent == "need":
            # NEED WATER -> I need water.
            verb = "need"
            objs = []
            skip = {"need", "needs", "needed", "want", "wants", "wanted", "require"}
            for t in lower:
                if t in {"want", "wants", "wanted"}:
                    verb = "want"
                elif t in skip:
                    continue
                else:
                    objs.append(t)
            if objs:
                obj = " ".join(objs)
                # Light article for single noun object
                if len(objs) == 1 and objs[0] not in {"water", "food", "help", "home"}:
                    obj = f"{self._article(objs[0])} {objs[0]}"
                return f"I {verb} {obj}."
            return f"I {verb} help."

        if intent == "feeling":
            feelings = [t for t in lower if t.upper() in FEELING_WORDS]
            other = [t for t in lower if t.upper() not in FEELING_WORDS]
            if feelings:
                feel = " and ".join(feelings)
                extra = f" {' '.join(other)}" if other else ""
                return f"I am {feel}{extra}."
            return self._statement(lower)

        if intent == "question":
            return self._capitalize(" ".join(lower) + "?")

        return self._statement(lower)

    def _statement(self, lower: list[str]) -> str:
        # Prefer first-person if missing subject pronouns
        subjects = {"i", "we", "you", "he", "she", "they", "it"}
        if lower and lower[0] not in subjects:
            # Simple declarative with implied I for short content phrases
            if len(lower) <= 4:
                return f"I {' '.join(lower)}."
        return self._capitalize(" ".join(lower) + ".")

    @staticmethod
    def _capitalize(text: str) -> str:
        text = text.strip()
        if not text:
            return text
        return text[0].upper() + text[1:]
