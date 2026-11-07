"""Summing up an evaluation run: totals over all cases, the console summary and the JSON report file."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .schemas import CaseRecord


def aggregate(records: list[CaseRecord]) -> dict:
    """Add up the scores of all cases into run-wide totals.

    Args:
        records: Per-case records from ``run``.

    Returns:
        dict: Always the same keys: ``cases`` (int) — cases run. ``errors`` (int) — cases where the agent itself raised
            exception. ``called_right_tools`` / ``got_right_data`` / ``called_no_tools`` (dict) — how often each layer-1
            metric passed, as
        ``{"hits": passed, "total": scored, "rate": hits / total}``.
        ``total`` counts only cases the metric applies to (not None): ``got_right_data`` skips cases that need no tools,
            ``called_no_tools`` counts only those. E.g. 1 of 2
                no-tool cases passed → ``{"hits": 1, "total": 2, "rate": 0.5}``.
            ``passed_via_alternative_route`` (int) — cases that did not
                follow ``expected_calls`` but followed one of their
                ``alternative_routes``. Many of them suggest the prompt and
                the expected route disagree.
            ``extra_calls_total`` (int) — extra calls summed over all cases.
            ``extra_calls_cases`` (int) — cases with at least one extra call.
            ``per_tool`` (dict[str, dict]) — tool name → three case counts,
                each tool judged on its own (not on whether the whole case
        passed). Counts are cases, not calls: calling a tool twice in one case still counts once. ``needed`` — cases
            whose ``expected_calls`` include it. ``called_when_needed`` — of those, cases where the agent did call it.
            Lower than ``needed`` = the agent misses this tool. ``called_when_not_needed`` — cases that did not need it
            but the agent called it anyway. High = the agent overuses this tool. E.g. ``get_course`` needed in 3 cases,
            called in 2 of them and
                in 2 other cases → ``{"needed": 3, "called_when_needed": 2,
        "called_when_not_needed": 2}``. ``faithfulness`` (dict) — ``distribution`` (score "0"/"1"/"2" → count),
            ``zero_rate`` (share of 0s, None if nothing scored; the headline hallucination metric), ``scored`` (cases
            judged). ``correctness`` (dict) — ``distribution`` and ``scored``, as above. ``cross_table`` (dict[str,
            int]) — label like ``"tools=OK faith=2 corr=1"`` → number of cases with that combination; only cases scored
            on all three.
    """
    scored = [r for r in records if r.tool_scores]

    # Per tool, per case: was this tool needed, and did the agent call it?
    # Sets, because only presence matters — two calls in one case count once.
    per_tool: dict[str, dict[str, int]] = {}
    for record in scored:
        needed = set(record.tool_scores.expected_tools)
        called = set(record.tool_scores.called_tools)
        for name in needed | called:
            bucket = per_tool.setdefault(name, {"needed": 0, "called_when_needed": 0, "called_when_not_needed": 0})
            if name in needed:
                bucket["needed"] += 1
                if name in called:
                    bucket["called_when_needed"] += 1
            else:
                bucket["called_when_not_needed"] += 1

    def rate(key: str) -> dict:
        """Count how many cases passed one layer-1 metric, ignoring cases it does not apply to.

        Args:
            key: Field name on ``ToolScores``, e.g. ``"got_right_data"``.

        Returns:
            dict: Always the same keys: ``hits`` (int) — cases where the metric is True. ``total`` (int) — cases where
                the metric is not None. ``rate`` (float | None) — ``hits / total``; None if ``total`` is 0.
        """
        rows = [r for r in scored if getattr(r.tool_scores, key) is not None]
        hits = [r for r in rows if getattr(r.tool_scores, key)]
        return {"hits": len(hits), "total": len(rows), "rate": (len(hits) / len(rows)) if rows else None}

    faith = [r.faithfulness.score for r in records if r.faithfulness]
    corr = [r.correctness.score for r in records if r.correctness]

    cross: dict[str, int] = {}
    for record in records:
        if not (record.tool_scores and record.faithfulness and record.correctness):
            continue
        key = (
            f"tools={'OK' if record.tool_scores.called_right_tools else 'FAIL'}"
            f"  faith={record.faithfulness.score}"
            f"  corr={record.correctness.score}"
        )
        cross[key] = cross.get(key, 0) + 1

    return {
        "cases": len(records),
        "errors": len([r for r in records if r.error]),
        "called_right_tools": rate("called_right_tools"),
        "got_right_data": rate("got_right_data"),
        "called_no_tools": rate("called_no_tools"),
        "passed_via_alternative_route": len(
            [r for r in scored if r.tool_scores.scored_against_route != "expected_calls"]
        ),
        "extra_calls_total": sum(r.tool_scores.extra_calls for r in scored),
        "extra_calls_cases": len([r for r in scored if r.tool_scores.extra_calls]),
        "per_tool": per_tool,
        "faithfulness": {
            "distribution": {str(s): faith.count(s) for s in (0, 1, 2)},
            "zero_rate": (faith.count(0) / len(faith)) if faith else None,
            "scored": len(faith),
        },
        "correctness": {
            "distribution": {str(s): corr.count(s) for s in (0, 1, 2)},
            "scored": len(corr),
        },
        "cross_table": cross,
    }


def summarize(records: list[CaseRecord], totals: dict) -> None:
    """Print the run summary to the console.

    Shows the layer-1 pass rates, how the judges' scores are spread, the diagnostic table (which combinations of tool
    result and judge scores occurred), every answer the faithfulness judge scored 0 — the hallucinations worth reading
    first — and every case where the agent failed with an error.

    Args:
        records: Per-case records from ``run``.
        totals: Totals from ``aggregate``.
    """

    def pct(entry: dict | None) -> str:
        """Format a pass rate for printing, e.g. ``"6/8 (75%)"``.

        Args:
            entry: One ``{"hits", "total", "rate"}`` dict, or None.

        Returns:
            str: Formatted rate, or "no data" when there is nothing to show.
        """
        if not entry or entry["rate"] is None:
            return "no data"
        return f"{entry['hits']}/{entry['total']}  ({entry['rate']:.0%})"

    print("\n" + "=" * 66)
    print("LAYER 1 - tool calls (deterministic)")
    print("=" * 66)
    print(f"called_right_tools: {pct(totals['called_right_tools'])}")
    print(f"got_right_data:     {pct(totals['got_right_data'])}")
    print(f"called_no_tools:    {pct(totals['called_no_tools'])}")
    print(f"via alternative:    {totals['passed_via_alternative_route']} cases")
    print(f"extra_calls:        {totals['extra_calls_total']} calls in {totals['extra_calls_cases']} cases")
    if totals["per_tool"]:
        print("\nper tool (cases): called when needed / needed, + called when not needed")
        for name, bucket in sorted(totals["per_tool"].items()):
            print(
                f"  {name:24} {bucket['called_when_needed']}/{bucket['needed']}"
                f"  +{bucket['called_when_not_needed']} unneeded"
            )

    faith = totals["faithfulness"]
    corr = totals["correctness"]
    if faith["scored"]:
        print("\n" + "=" * 66)
        print("LAYER 2 - answer (LLM judges)")
        print("=" * 66)
        print(f"faithfulness zero rate: {faith['zero_rate']:.0%}   <- headline hallucination metric")
        dist = faith["distribution"]
        print(f"  faithfulness scores: 0={dist['0']}  1={dist['1']}  2={dist['2']}")
        dist = corr["distribution"]
        print(f"  correctness scores:  0={dist['0']}  1={dist['1']}  2={dist['2']}")

    if totals["cross_table"]:
        print("\nDiagnostic table:")
        for key, count in sorted(totals["cross_table"].items(), key=lambda kv: -kv[1]):
            print(f"  {key:40} {count}")

    zeros = [r for r in records if r.faithfulness and r.faithfulness.score == 0]
    if zeros:
        print(f"\nfaithfulness = 0 ({len(zeros)}) - hallucinations to read:")
        for record in zeros:
            bad = [c for c in record.faithfulness.claims if c.status != "supported"]
            if bad:
                print(f"  - {record.id}: [{bad[0].status}] {bad[0].text}")
            else:
                print(f"  - {record.id}: (judge flagged no claim)")

    errors = [r for r in records if r.error]
    if errors:
        print(f"\nAgent errors ({len(errors)}):")
        for record in errors:
            print(f"  - {record.id}: {record.error}")


def describe_run(run_options: dict) -> str:
    """Say in a few words which cases a run covered, for the report's file name.

    Args:
        run_options: ``category``, ``only``, ``limit`` (each None when not
            used) and ``judges`` (bool), as given on the command line.

    Returns:
        str: E.g. ``"all"``, ``"category-dopytania"``, ``"only-prog-informatyka-2025"`` or ``"all_limit-5_no-judge"``.
    """
    parts = []
    if run_options.get("category"):
        parts.append(f"category-{run_options['category']}")
    if run_options.get("only"):
        parts.append(f"only-{run_options['only']}")
    if not parts:
        parts.append("all")
    if run_options.get("limit") is not None:
        parts.append(f"limit-{run_options['limit']}")
    if not run_options.get("judges", True):
        parts.append("no-judge")
    return "_".join(parts)


def save_report_json(content: dict, results_dir: Path, label: str) -> Path:
    """Write a report as ``<results_dir>/<local date and time>_<label>.json``.

    The name starts with the local time, so files sort by date, e.g. ``2026-10-31_09-42-00_category-dopytania.json``. A
    second report in the same second gets ``_2`` (``_3``, ...) instead of overwriting the first.

    Args:
        content: The report; ``generated_at`` (local time with its UTC offset,
            e.g. ``"2026-10-31T09:42:00+01:00"``) is put in front of it.
        results_dir: Folder for reports; created if missing.
        label: What the report is about, e.g. ``describe_run(...)``.

    Returns:
        Path: The written file.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone()
    base_name = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{label}"
    report_path = results_dir / f"{base_name}.json"
    copy_number = 2
    while report_path.exists():
        report_path = results_dir / f"{base_name}_{copy_number}.json"
        copy_number += 1
    report_path.write_text(
        json.dumps({"generated_at": now.isoformat(timespec="seconds"), **content}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def write_report(records: list[CaseRecord], totals: dict, results_dir: Path, run_options: dict) -> Path:
    """Save a benchmark run, named by local time and ``describe_run``.

    Args:
        records: Per-case records from ``run``.
        totals: Totals from ``aggregate``.
        results_dir: Folder for reports; created if missing.
        run_options: What the run covered — see ``describe_run``.

    Returns:
        Path: The written file, e.g. ``2026-10-31_09-42-00_category-dopytania.json``. Its JSON has ``generated_at`` (see
            ``save_report_json``), ``run_options``, ``totals`` (as from ``aggregate``) and ``cases`` (every record).
    """
    return save_report_json(
        {"run_options": run_options, "totals": totals, "cases": [record.model_dump(mode="json") for record in records]},
        results_dir,
        describe_run(run_options),
    )
