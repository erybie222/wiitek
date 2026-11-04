"""Check that no text a model sees contains a value or question from the eval sets (data leakage).

A prompt holding the benchmark's answers turns the evaluation into a memory
test: the agent "answers correctly" from the prompt, and no judge can tell.
The same holds for the judges and the calibration set. This scans every text
that reaches a model —

- ``SYSTEM_PROMPT`` and both judge prompts (``src/templates.py``),
- every tool's description and argument descriptions (``database.repository``),
- the judges' structured-output schemas (``eval/schemas.py``)

— for the values and questions of ``eval/benchmark.yaml`` and
``eval/judge_calibration.yaml``. Matching ignores case, Polish diacritics and
line breaks, since a phrase wrapped across two prompt lines still leaks.

Exact domain vocabulary the agent must pass to a tool is not leakage — it
reveals a spelling, not an answer — and is listed in ``ALLOWED_VOCABULARY``.

Usage:
    uv run python -m eval.check_leakage      # exit code 1 when anything leaks
"""

import json
import re
import sys
import unicodedata
from typing import Iterator

import yaml

# Must come before the app imports below: it puts src/ on the import path.
from ._src_path import ROOT

from database.repository import TOOLS
from templates import CORRECTNESS_PROMPT, FAITHFULNESS_PROMPT, SYSTEM_PROMPT

from .benchmark_loader import load_cases
from .schemas import Claim, Correctness, FactCheck, Faithfulness

BENCHMARK_PATH = ROOT / "eval" / "benchmark.yaml"
CALIBRATION_PATH = ROOT / "eval" / "judge_calibration.yaml"

# Values that also occur in the eval sets but are vocabulary, not answers: the
# exact spelling the agent must pass as a tool argument, or a generic title.
ALLOWED_VOCABULARY = {
    "dyrektor instytutu",   # a function_name example in SYSTEM_PROMPT and get_staff_function
    "profesor",             # a `position` value
    "prof. dr hab. inz",    # an `academic_title` value
    "poniedzialek",         # weekday names are the day_of_week vocabulary
    "wtorek", "sroda", "czwartek", "piatek",
    "egzamin", "zaliczenie z ocena",  # assessment_type values
    "full-time", "part-time", "bachelor-of-engineering",  # fixed codes
    "wiit",
}

# Matches what is worth looking for inside a reference answer: decimals,
# numbers of 3+ digits, and room codes like "A1 503" or "A23a 1.6.5".
SPECIFICS_IN_TEXT = re.compile(r"\d+[.,]\d+|\b\d{3,}\b|\b[A-Z]\d+[a-z]?\s?[\d.]+\b")
BARE_YEAR = re.compile(r"(19|20)\d\d")


def normalize(text: str) -> str:
    """Lower-case, strip Polish diacritics and collapse whitespace, so wording differences don't hide a match.

    Args:
        text: Any text.

    Returns:
        str: E.g. ``"Wydział\\nInformatyki"`` → ``"wydzial informatyki"``.
    """
    decomposed = unicodedata.normalize("NFKD", str(text).lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", stripped).replace("ł", "l")


def model_facing_texts() -> dict[str, str]:
    """Collect every text the agent or a judge receives.

    Returns:
        dict[str, str]: A label such as ``"tool get_staff"`` → the text.
    """
    texts = {
        "SYSTEM_PROMPT": SYSTEM_PROMPT,
        "FAITHFULNESS_PROMPT": FAITHFULNESS_PROMPT,
        "CORRECTNESS_PROMPT": CORRECTNESS_PROMPT,
    }
    for schema in (Faithfulness, Correctness, Claim, FactCheck):
        texts[f"judge schema {schema.__name__}"] = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    for tool in TOOLS:
        texts[f"tool {tool.name}"] = tool.description + "\n" + json.dumps(tool.args, ensure_ascii=False)
    return texts


def _strings_in(value) -> Iterator[str]:
    """Yield every scalar inside nested dicts and lists, as text (booleans skipped)."""
    if isinstance(value, dict):
        for item in value.values():
            yield from _strings_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings_in(item)
    elif value is not None and not isinstance(value, bool):
        yield str(value)


def leak_needles() -> dict[str, str]:
    """Collect what must not appear in a model-facing text, each with where it came from.

    From the benchmark: every question (earlier turns included), the
    ``entities``, ``expected_data`` and the arguments (or, for a syllabus
    search, the ``expected_sections``) of every expected or alternative
    call, plus numbers and room codes from the reference answers.
    From the calibration set: every question and answer.

    Returns:
        dict[str, str]: Normalized needle → origin, e.g.
        ``"759.00"`` → ``"benchmark:prog-informatyka-2025"``. Short words,
        bare years and ``ALLOWED_VOCABULARY`` are left out.
    """
    needles: dict[str, str] = {}

    def add(raw: str, origin: str) -> None:
        needle = normalize(raw).strip(" .,;:!?\"'()")
        too_short = len(needle) < 4 if not needle.isdigit() else len(needle) < 3
        if too_short or BARE_YEAR.fullmatch(needle) or needle in ALLOWED_VOCABULARY:
            return
        needles.setdefault(needle, origin)

    for case in load_cases(BENCHMARK_PATH):
        origin = f"benchmark:{case['id']}"
        for question in [case["question"], *(case.get("earlier_questions") or [])]:
            add(question, origin)
        for field in ("entities", "expected_data"):
            for value in _strings_in(case.get(field)):
                add(value, origin)
        routes = [case.get("expected_calls") or []] + [
            route.get("calls") or [] for route in case.get("alternative_routes") or []
        ]
        for call in (call for route in routes for call in route):
            for value in _strings_in([call.get("args"), call.get("expected_sections")]):
                add(value, origin)
        for field in ("expected_answer", "expected_behavior"):
            for specific in SPECIFICS_IN_TEXT.findall(str(case.get(field) or "")):
                add(specific, origin)
                add(specific.replace(",", "."), origin)

    calibration = yaml.safe_load(CALIBRATION_PATH.read_text(encoding="utf-8"))
    for item in calibration["items"]:
        origin = f"calibration:{item['id']}"
        add(item["question"], origin)
        add(item["answer"], origin)
        for turn in item.get("earlier_turns") or []:
            add(turn["question"], origin)
            add(turn["answer"], origin)
    return needles


def find_leaks(texts: dict[str, str], needles: dict[str, str]) -> list[dict]:
    """Find every needle occurring as a whole phrase in a model-facing text.

    Args:
        texts: Label → text, from ``model_facing_texts``.
        needles: Normalized needle → origin, from ``leak_needles``.

    Returns:
        list[dict]: One per occurrence: ``text`` (label), ``needle``,
        ``origin`` and ``context`` (the surrounding normalized text). Empty
        when nothing leaks.
    """
    leaks = []
    for label, text in texts.items():
        haystack = normalize(text)
        for needle, origin in needles.items():
            pattern = r"(?<![\w.])" + re.escape(needle) + r"(?!\w)"
            for match in re.finditer(pattern, haystack):
                context = haystack[max(0, match.start() - 60): match.end() + 60]
                leaks.append({"text": label, "needle": needle, "origin": origin, "context": context})
    return leaks


def main() -> None:
    """Scan, print every leak with its context, and exit with 1 if there is any.

    An eval file that does not load (a case ``load_cases`` rejects, or broken
    YAML) also exits with 1, with one line saying why instead of a traceback.
    """
    texts = model_facing_texts()
    try:
        needles = leak_needles()
    except (ValueError, yaml.YAMLError) as error:
        print(f"PROBLEM  {BENCHMARK_PATH.name} or {CALIBRATION_PATH.name} could not be read: "
              f"{type(error).__name__}: {error}")
        sys.exit(1)
    leaks = find_leaks(texts, needles)
    print(f"Checked {len(needles)} eval values against {len(texts)} model-facing texts.")
    if not leaks:
        print("No leakage found.")
        return
    print(f"LEAKAGE: {len(leaks)} occurrence(s)\n")
    for leak in leaks:
        print(f"[{leak['text']}] \"{leak['needle']}\"  <- {leak['origin']}")
        print(f"    ...{leak['context']}...\n")
    print("Replace them with invented values, or add true tool vocabulary to ALLOWED_VOCABULARY.")
    sys.exit(1)


if __name__ == "__main__":
    main()
