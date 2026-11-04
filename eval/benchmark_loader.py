"""Reading the benchmark (``eval/benchmark.yaml``) and deriving values from its cases.

A case is one question with what a correct answer must contain and which
tool calls should produce it. The YAML file itself explains its format.
"""

from pathlib import Path

import yaml


def load_cases(path: Path) -> list[dict]:
    """Read benchmark.yaml into one flat list of cases.

    Every top-level list except ``meta`` is a category; its name is copied
    into each of its cases.

    Args:
        path: Path to benchmark.yaml.

    Returns:
        list[dict]: Cases in file order. Each is its YAML entry as written,
        plus one added key. The keys the evaluation reads:
            ``id`` (str) — unique case name, used in output and ``--only``.
            ``category`` (str) — added here: the top-level list it came from.
            ``question`` (str) — what the agent is asked; the scored question.
            ``earlier_questions`` (list[str] | None) — asked first, in the
                same conversation, so ``question`` can be a follow-up. Their
                answers are not scored; ``expected_calls`` covers the calls of
                the whole conversation.
            ``expected_calls`` (list[dict] | None) — ``{"tool", "args"}`` per
                needed call; empty or missing means no tools are needed. A
                call may add ``only_resolves_prev_id: true``: it is run to
                give the next ``$prev.id`` its row, but not required. A
                syllabus search has ``expected_sections`` (a list of
                ``{"course_code", "section"}``) instead of ``args``: it is
                not run, its results must include one of them.
            ``alternative_routes`` (list[dict] | None) — other valid call
                sequences, each ``{"why", "calls"}``; ``calls`` has the
                format of ``expected_calls``.
            ``expected_answer`` (str | None) — reference answer for the judge.
            ``expected_data`` (dict | None) — key facts for the judge.
            ``expected_behavior`` (str | None) — what the agent should do
                instead of answering, e.g. ask back.

    Raises:
        ValueError: An alternative route lacks ``why`` or ``calls``,
            ``earlier_questions`` is not a list of non-empty strings, or an
            ``expected_sections`` entry is malformed.
    """
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = []
    for category, entries in document.items():
        if category == "meta" or not isinstance(entries, list):
            continue
        for entry in entries:
            check_alternative_routes(entry)
            check_earlier_questions(entry)
            check_expected_sections(entry)
            cases.append({**entry, "category": category})
    return cases


def check_expected_sections(case: dict) -> None:
    """Make sure every syllabus search a case expects names the sections properly.

    A misspelt key would otherwise never match a search result, and the
    agent would be scored as having missed a section the benchmark never
    really named.

    Args:
        case: One benchmark case as written in the YAML.

    Raises:
        ValueError: A call in ``expected_calls`` or an alternative route has
            ``expected_sections`` that is not a non-empty list of
            ``{"course_code", "section"}`` with non-empty strings, or has
            ``args`` as well (a search with sections is never run).
    """
    routes = [case.get("expected_calls") or []] + [
        alternative.get("calls") or [] for alternative in case.get("alternative_routes") or []
    ]
    for call in (call for route in routes for call in route):
        if "expected_sections" not in call:
            continue
        sections = call["expected_sections"]
        well_formed = (
            isinstance(sections, list) and sections and "args" not in call
            and all(
                isinstance(entry, dict) and set(entry) == {"course_code", "section"}
                and all(isinstance(value, str) and value.strip() for value in entry.values())
                for entry in sections
            )
        )
        if not well_formed:
            raise ValueError(
                f"{case.get('id')}: {call.get('tool')} needs expected_sections as a non-empty list "
                "of {course_code, section} and no args"
            )


def check_earlier_questions(case: dict) -> None:
    """Make sure a case's ``earlier_questions``, if present, is a list of questions.

    A single string instead of a list would otherwise be asked letter by letter.

    Args:
        case: One benchmark case as written in the YAML.

    Raises:
        ValueError: ``earlier_questions`` is not a list, or holds an empty or
            non-string item.
    """
    earlier = case.get("earlier_questions")
    if earlier is None:
        return
    if not isinstance(earlier, list) or not all(isinstance(q, str) and q.strip() for q in earlier):
        raise ValueError(f"{case.get('id')}: earlier_questions must be a list of non-empty strings")


def check_alternative_routes(case: dict) -> None:
    """Make sure every alternative route of a case says why it is valid and what to call.

    ``why`` is required so an alternative cannot be added just to make a
    failing case pass without someone arguing it is an equally honest route.

    Args:
        case: One benchmark case as written in the YAML.

    Raises:
        ValueError: An alternative is not a mapping, has an empty ``why``,
            or has no ``calls`` key. ``calls: []`` is allowed (no tools needed).
    """
    for position, alternative in enumerate(case.get("alternative_routes") or []):
        if not isinstance(alternative, dict) or not alternative.get("why") or "calls" not in alternative:
            raise ValueError(
                f"{case.get('id')}: alternative_routes[{position}] needs a non-empty 'why' and a 'calls' list"
            )


def case_kind(case: dict) -> str:
    """Say what kind of response the case expects; shown in the console and the report.

    Args:
        case: One benchmark case.

    Returns:
        str: ``"clarify"`` if it has ``expected_behavior`` (the agent should
        ask the user something back), ``"lookup"`` if it has ``expected_data``
        (the agent should answer from the database), otherwise ``"refusal"``
        (the agent should say it does not know or cannot help).
    """
    if case.get("expected_behavior"):
        return "clarify"
    if case.get("expected_data"):
        return "lookup"
    return "refusal"


def reference_text(case: dict) -> str:
    """Build the reference text the correctness judge compares the answer with.

    Args:
        case: One benchmark case.

    Returns:
        str: ``expected_answer`` and/or ``expected_behavior``, one per line;
        for a case with neither, a note that the right answer is "I don't know".
    """
    parts = []
    if case.get("expected_answer"):
        parts.append(str(case["expected_answer"]))
    if case.get("expected_behavior"):
        parts.append(f"Oczekiwane zachowanie: {case['expected_behavior']}")
    if not parts:
        parts.append('Poprawna reakcja to "nie wiem" - brak danych w bazie.')
    return "\n".join(parts)
