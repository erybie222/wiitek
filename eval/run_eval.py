"""Measure the WIiTek agent's answer quality on the benchmark (``eval/benchmark.yaml``).

Every benchmark question is put to the agent, and the result is scored in
two layers that fail for different reasons, so a bad score shows which part
to fix:

**Layer 1 — tool calls, checked by code** (``tool_call_scoring.py``):
``called_right_tools`` — the agent called every tool the case needs;
``got_right_data`` — those calls returned the same data as the correct calls
(for a syllabus search: one of the case's expected sections);
``called_no_tools`` — on cases that need no tools, the agent called none;
``extra_calls`` — how many calls it made beyond what was needed.
A case may list ``alternative_routes``; they are scored only when the agent
did not follow ``expected_calls``, and the console marks a pass through one
with ``via alternative_routes[N]``.

A case may also have ``earlier_questions``: they are asked first in the same
conversation, so the scored ``question`` can be a follow-up. Both layers
score only the last answer, but against the tool calls of the whole
conversation, and the judges see the earlier turns as context.

**Layer 2 — the answer, checked by LLM judges** (``answer_judges.py``):
``faithfulness`` — every claim is backed by the data the tools returned (the
share of 0 scores is the hallucination rate); ``correctness`` — the answer
matches the verified reference.

The console shows a summary (``report.py``); the full report with every
answer and tool call is saved to ``eval/results/``, e.g.
``2026-10-31_09-42-00_category-dopytania.json`` (local time, then what was run).

Usage:
    uv run python -m eval.run_eval                  # whole benchmark
    uv run python -m eval.run_eval --category progi_punktowe
    uv run python -m eval.run_eval --only prog-informatyka-2025 --verbose
    uv run python -m eval.run_eval --no-judge       # layer 1 only

Run it from the repository root: ``eval`` is a package, so it is started
with ``-m`` rather than as a file path.
"""

from __future__ import annotations

import argparse

# The application (src/)
from ._src_path import ROOT

# isort: split
from agent import ask, build_agent, build_judge_model
from config import AgentSettings, JudgeSettings
from database.connection import close_pool

# Must come before the app imports below: it puts src/ on the import path.
# This package
from .answer_judges import judge_correctness, judge_faithfulness
from .benchmark_loader import case_kind, load_cases, reference_text
from .report import aggregate, summarize, write_report
from .schemas import CaseRecord, Correctness, EarlierTurn, Faithfulness
from .tool_call_scoring import agent_tool_calls, score_tool_calls_with_alternatives

BENCHMARK_PATH = ROOT / "eval" / "benchmark.yaml"
RESULTS_DIR = ROOT / "eval" / "results"


def run(cases: list[dict], use_judge: bool, verbose: bool) -> list[CaseRecord]:
    """Ask the agent every case's question, score the result, and print one line per case.

    The line shows the layer-1 result and, with judges, both judge scores; below it come details worth reading
    (unsupported claims, wrong data).

    Args:
        cases: Benchmark cases to run.
        use_judge: Also run the layer-2 judges; False scores layer 1 only.
        verbose: Also print each case's tools and full answer.

    Returns:
        list[CaseRecord]: One per case, in run order. If asking the agent raised an exception, the record has ``error``
            set and no answer or
        scores. What each field means is described on the ``CaseRecord`` model.
    """
    agent = build_agent(AgentSettings())
    judge_model = build_judge_model(JudgeSettings()) if use_judge else None
    faith_judge = judge_model.with_structured_output(Faithfulness) if judge_model else None
    correct_judge = judge_model.with_structured_output(Correctness) if judge_model else None

    records = []
    for index, case in enumerate(cases, start=1):
        kind = case_kind(case)
        earlier_questions = case.get("earlier_questions") or []
        turns_note = f", {len(earlier_questions) + 1} turns" if earlier_questions else ""
        print(f"[{index}/{len(cases)}] {case['id']} ({kind}{turns_note}) ... ", end="", flush=True)

        try:
            # Its own conversation per case, so no case sees another's questions.
            # Earlier questions go first in the same conversation; only the last
            # answer is scored, but tool calls count from every turn — a follow-up
            # may rightly reuse what an earlier turn fetched (e.g. a staff id).
            earlier_turns = []
            tool_calls = []
            for earlier_question in earlier_questions:
                turn = ask(agent, earlier_question, thread_id=case["id"])
                earlier_turns.append(EarlierTurn(question=earlier_question, answer=turn[-1].content))
                tool_calls += agent_tool_calls(turn)
            messages = ask(agent, case["question"], thread_id=case["id"])
            answer = messages[-1].content
            tool_calls += agent_tool_calls(messages)
        except Exception as error:
            print(f"ERROR: {error}")
            records.append(
                CaseRecord(
                    id=case["id"],
                    category=case["category"],
                    kind=kind,
                    question=case["question"],
                    error=str(error),
                )
            )
            continue

        tool_scores = score_tool_calls_with_alternatives(case, tool_calls)
        faithfulness = judge_faithfulness(faith_judge, case, answer, tool_calls, earlier_turns) if faith_judge else None
        correctness = judge_correctness(correct_judge, case, answer, earlier_turns) if correct_judge else None

        marks = ["tools OK" if tool_scores.called_right_tools else "tools FAIL"]
        if tool_scores.scored_against_route != "expected_calls":
            marks.append(f"via {tool_scores.scored_against_route}")
        if tool_scores.got_right_data is False:
            marks.append("data FAIL")
        if tool_scores.extra_calls:
            marks.append(f"+{tool_scores.extra_calls} extra")
        if faithfulness:
            marks.append(f"faith {faithfulness.score}")
        if correctness:
            marks.append(f"corr {correctness.score}")
        print("  ".join(marks))

        if faithfulness and faithfulness.score == 0:
            for claim in [c for c in faithfulness.claims if c.status != "supported"][:3]:
                print(f"      [{claim.status}] {claim.text}")
        if tool_scores.wrong_data_tools:
            print(f"      wrong data from: {', '.join(tool_scores.wrong_data_tools)}")
        if tool_scores.expected_calls_error:
            print(f"      {tool_scores.expected_calls_error}")
        if verbose:
            print(f"      tools: {tool_scores.called_tools}")
            print(f"      answer: {answer}\n")

        records.append(
            CaseRecord(
                id=case["id"],
                category=case["category"],
                kind=kind,
                earlier_turns=earlier_turns,
                question=case["question"],
                reference=reference_text(case),
                expected_data=case.get("expected_data"),
                answer=answer,
                tool_calls=tool_calls,
                tool_scores=tool_scores,
                faithfulness=faithfulness,
                correctness=correctness,
            )
        )

    return records


def main() -> None:
    """Run the benchmark from the command line.

    Parses the filters, runs the selected cases, prints the summary and writes the full report to ``eval/results/``
    (named by ``write_report``).
    """
    parser = argparse.ArgumentParser(prog="python -m eval.run_eval", description="Run the WIiTek benchmark.")
    parser.add_argument("--category", help="Run only one benchmark category.")
    parser.add_argument("--only", help="Run only the case with this id.")
    parser.add_argument("--limit", type=int, help="Run at most N cases.")
    parser.add_argument("--no-judge", action="store_true", help="Layer 1 only; skip the LLM judges.")
    parser.add_argument("--verbose", action="store_true", help="Print every answer in full.")
    args = parser.parse_args()

    cases = load_cases(BENCHMARK_PATH)

    # A missing expected_calls is scored as "call nothing" — flag it in case it was just forgotten.
    unknown = [case["id"] for case in cases if "expected_calls" not in case]
    if unknown:
        print(f"WARNING: no expected_calls in {len(unknown)} cases: {', '.join(unknown)}")
        print("         Layer 1 will score them as needing no tools.\n")

    if args.category:
        cases = [case for case in cases if case["category"] == args.category]
    if args.only:
        cases = [case for case in cases if case["id"] == args.only]
    if args.limit is not None:  # 0 means run nothing, not "no limit"
        cases = cases[: args.limit]

    if not cases:
        print("No cases selected - check --category / --only.")
        return

    try:
        records = run(cases, use_judge=not args.no_judge, verbose=args.verbose)
    finally:
        close_pool()

    totals = aggregate(records)
    summarize(records, totals)

    run_options = {"category": args.category, "only": args.only, "limit": args.limit, "judges": not args.no_judge}
    report_path = write_report(records, totals, RESULTS_DIR, run_options)
    print(f"\nReport: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
