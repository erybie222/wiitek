"""Telling whether a question is written in neither Polish nor English.

The agent answers Polish questions in Polish and everything else in English. A rule in the system prompt did not hold
for other languages and made English answers worse, so the decision is made here, in code, before the agent runs.

The check is deliberately cautious: wrongly answering a Polish student in English is worse than leaving a question in
another language to the agent. Short messages ("a w 2024?") carry too little text to judge and are never flagged.
"""

from __future__ import annotations

import re
from functools import lru_cache

from py3langid.langid import MODEL_FILE, LanguageIdentifier

# A text with this share of letters outside the Latin script is neither Polish nor English, whatever names it quotes.
MIN_NON_LATIN_LETTER_SHARE = 0.3
LAST_LATIN_CODE_POINT = 0x24F
# Fewer words than this are too little for the detector, which then often names an unrelated language.
MIN_WORDS_TO_DETECT = 4
# Polish course and major names inside a foreign question pull the detector towards Polish, so a question is flagged
# only when Polish and English together stay this unlikely.
MAX_POLISH_OR_ENGLISH_PROBABILITY = 0.2
WORD = re.compile(r"[^\W\d_]{2,}")


@lru_cache(maxsize=1)
def get_language_identifier() -> LanguageIdentifier:
    """Load the language model once per process.

    Returns:
        LanguageIdentifier: The detector, reporting probabilities that sum to 1.
    """
    return LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)


def is_mostly_non_latin(text: str) -> bool:
    """Check whether a substantial share of the text's letters are outside the Latin script.

    Args:
        text: The text to check.

    Returns:
        bool: True for e.g. a Ukrainian question quoting a Polish major name; False when there are no letters.
    """
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    non_latin_letters = sum(ord(letter) > LAST_LATIN_CODE_POINT for letter in letters)
    return non_latin_letters / len(letters) > MIN_NON_LATIN_LETTER_SHARE


def is_in_third_language(question: str) -> bool:
    """Check whether a question is confidently in a language other than Polish and English.

    Args:
        question: The user's message.

    Returns:
        bool: True when the question should be answered in English although it is not written in English; False for
            Polish, English, and anything too short or too uncertain to tell.
    """
    if is_mostly_non_latin(question):
        return True
    if len(WORD.findall(question)) < MIN_WORDS_TO_DETECT:
        return False
    probabilities = dict(get_language_identifier().rank(question))
    most_likely_language = max(probabilities, key=probabilities.__getitem__)
    polish_or_english_probability = probabilities.get("pl", 0.0) + probabilities.get("en", 0.0)
    return (
        most_likely_language not in {"pl", "en"} and polish_or_english_probability <= MAX_POLISH_OR_ENGLISH_PROBABILITY
    )
