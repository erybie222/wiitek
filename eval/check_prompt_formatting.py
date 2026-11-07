"""Check that every prompt fills in correctly, without calling any model.

The two judge prompts (``FAITHFULNESS_PROMPT``, ``CORRECTNESS_PROMPT``) go
through ``str.format()``, so every literal brace in them must be doubled. A
brace left single breaks the judge call only when an eval run reaches it, and
a placeholder doubled by mistake fails silently: the judge gets the text
``{answer}`` instead of the answer. ``SYSTEM_PROMPT`` is the opposite — it
reaches the agent verbatim, so a doubled brace there reaches the model as two.

The judge prompts are built by the real ``judge_faithfulness`` and
``judge_correctness``, with the judge replaced by a function that returns the
prompt it is given:

- once with a marker in every input, to check that each input lands in the
  prompt;
- for every benchmark case and calibration item, to check that real data
  builds a prompt too — for the calibration set this also checks its YAML,
  which nothing else reads before a paid run.

Usage:
    uv run python -m eval.check_prompt_formatting   # exit code 1 when a prompt breaks
"""

from __future__ import annotations

import sys

from langchain_core.runnables import RunnableLambda

from ._src_path import ROOT

# isort: split
from templates import SYSTEM_PROMPT

# Must come before the app imports below: it puts src/ on the import path.
from .answer_judges import judge_correctness, judge_faithfulness
from .benchmark_loader import load_cases
from .calibrate_judges import CALIBRATION_PATH, judge_item, load_calibration_items
from .schemas import EarlierTurn, ToolCall

BENCHMARK_PATH = ROOT / "eval" / "benchmark.yaml"

# Returns the prompt it is given instead of asking a model.
ECHO_JUDGE = RunnableLambda(lambda prompt: prompt)

# One marker per input of the judge prompts; each must reappear in the
# prompt that takes that input.
MARKERS = {
    "question": "<<question under test>>",
    "answer": "<<answer under test>>",
    "earlier turns": "<<earlier question under test>>",
    "tool output": "<<tool under test>>",
    "key facts": "<<key fact under test>>",
    "reference": "<<reference under test>>",
}
FAITHFULNESS_INPUTS = ("question", "answer", "earlier turns", "tool output")
CORRECTNESS_INPUTS = ("question", "answer", "earlier turns", "key facts", "reference")

# What str.format() raises on a broken template: an undoubled brace around a
# word (KeyError), a lone brace (ValueError), a bare "{}" (IndexError).
FORMAT_ERRORS = (KeyError, IndexError, ValueError)


def check_system_prompt() -> list[str]:
    """Check that SYSTEM_PROMPT has no doubled braces.

    Returns:
        list[str]: One problem description, or an empty list.
    """
    if "{{" in SYSTEM_PROMPT or "}}" in SYSTEM_PROMPT:
        return ["SYSTEM_PROMPT contains a doubled brace — it reaches the agent verbatim, so write single braces"]
    return []


def check_judge_inputs_land() -> list[str]:
    """Build both judge prompts from marker inputs and check every marker is in its prompt.

    Returns:
        list[str]: One description per template error or missing input,
        e.g. "CORRECTNESS_PROMPT: the reference is missing — its placeholder
        is probably doubled". Empty when both prompts are fine.
    """
    case = {
        "question": MARKERS["question"],
        "expected_data": {"fact": MARKERS["key facts"]},
        "expected_answer": MARKERS["reference"],
    }
    earlier_turns = [EarlierTurn(question=MARKERS["earlier turns"], answer="earlier answer")]
    tool_calls = [ToolCall(name=MARKERS["tool output"], args={})]

    builders = {
        "FAITHFULNESS_PROMPT": (
            lambda: judge_faithfulness(ECHO_JUDGE, case, MARKERS["answer"], tool_calls, earlier_turns),
            FAITHFULNESS_INPUTS,
        ),
        "CORRECTNESS_PROMPT": (
            lambda: judge_correctness(ECHO_JUDGE, case, MARKERS["answer"], earlier_turns),
            CORRECTNESS_INPUTS,
        ),
    }
    problems = []
    for name, (build, inputs) in builders.items():
        try:
            prompt = build()
        except FORMAT_ERRORS as error:
            problems.append(f"{name}: {type(error).__name__}: {error} — a literal brace is probably not doubled")
            continue
        for input_name in inputs:
            if MARKERS[input_name] not in prompt:
                problems.append(f"{name}: the {input_name} is missing — its placeholder is probably doubled")
    return problems


def check_real_data_builds() -> list[str]:
    """Build both judge prompts for every benchmark case and calibration item.

    Benchmark cases have no answer, so a placeholder stands in; their earlier questions get placeholder answers.
    Calibration items are built exactly as ``calibrate_judges`` builds them.

    Returns:
        list[str]: One description per file that would not load, or per
        case or item that failed, e.g. "benchmark.yaml: ValueError: ..." or
        "calibration: <id>: ValidationError: ..."; empty when all built.
    """
    problems = []
    try:
        cases = load_cases(BENCHMARK_PATH)
    except Exception as error:
        cases = []
        problems.append(f"{BENCHMARK_PATH.name}: {type(error).__name__}: {error}")
    for case in cases:
        earlier_turns = [
            EarlierTurn(question=question, answer="earlier answer") for question in case.get("earlier_questions") or []
        ]
        try:
            judge_faithfulness(ECHO_JUDGE, case, "answer", [], earlier_turns)
            judge_correctness(ECHO_JUDGE, case, "answer", earlier_turns)
        except Exception as error:
            problems.append(f"benchmark:{case['id']}: {type(error).__name__}: {error}")

    try:
        items = load_calibration_items(CALIBRATION_PATH)
    except Exception as error:
        return [*problems, f"{CALIBRATION_PATH.name}: {type(error).__name__}: {error}"]
    for item in items:
        try:
            judge_item(item, ECHO_JUDGE, ECHO_JUDGE)
        except Exception as error:
            problems.append(f"calibration:{item.get('id')}: {type(error).__name__}: {error}")
    return problems


def main() -> None:
    """Run every check, print what broke, and exit with code 1 if anything did.

    The real-data pass runs only on templates that passed: a broken template fails every case in the same way, and
    listing them all would bury the one line that says what to fix.
    """
    problems = check_system_prompt() + check_judge_inputs_land()
    if not problems:
        problems = check_real_data_builds()
    for problem in problems:
        print(f"PROBLEM  {problem}")
    if problems:
        print(f"\n{len(problems)} problem(s) found.")
        sys.exit(1)
    print("All prompts fill in correctly.")


if __name__ == "__main__":
    main()
