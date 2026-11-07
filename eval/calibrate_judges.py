"""Check the two judges against answers whose right score is known (``eval/judge_calibration.yaml``).

The benchmark measures the bot; this measures the judges. Most calibration
answers are broken on purpose — an invented number, a swapped hour, a value
from another row — and must score 0; the rest are correct answers that must
not be punished. Two failures matter, and they are counted apart:

- **missed zero** — a broken answer scored above 0: the judge is blind to that
  kind of error, so the benchmark's zero rate is lower than the truth;
- **false zero** — a correct answer scored 0: the judge punishes something
  harmless, so the zero rate is higher than the truth.

No database is needed: the tool output is stored in the YAML.

Usage:
    uv run python -m eval.calibrate_judges                 # every item once
    uv run python -m eval.calibrate_judges --repeat 3      # judges are not deterministic
    uv run python -m eval.calibrate_judges --only prog-zmyslona-liczba
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

from ._src_path import ROOT

# isort: split
from agent import build_judge_model
from config import JudgeSettings

# Must come before the app imports below: it puts src/ on the import path.
from .answer_judges import judge_correctness, judge_faithfulness
from .report import save_report_json
from .schemas import Correctness, EarlierTurn, Faithfulness, ToolCall

CALIBRATION_PATH = ROOT / "eval" / "judge_calibration.yaml"
RESULTS_DIR = ROOT / "eval" / "results"
JUDGES = ("faithfulness", "correctness")
SHORT_NAMES = {"faithfulness": "faith", "correctness": "corr"}


def load_calibration_items(path: Path) -> list[dict]:
    """Read and check the calibration items.

    Args:
        path: Path to judge_calibration.yaml.

    Returns:
        list[dict]: The items in file order, as written (see the file's header for the keys).

    Raises:
        ValueError: An id repeats, an item has no ``expected_faithfulness`` / ``expected_correctness``, or expects a
            correctness score without ``expected_answer`` or ``expected_behavior`` to compare with.
    """
    items = yaml.safe_load(path.read_text(encoding="utf-8"))["items"]
    seen = set()
    for item in items:
        item_id = item.get("id")
        if item_id in seen:
            raise ValueError(f"{item_id}: id used twice")
        seen.add(item_id)
        if not any(f"expected_{judge}" in item for judge in JUDGES):
            raise ValueError(f"{item_id}: needs expected_faithfulness and/or expected_correctness")
        if "expected_correctness" in item and not (item.get("expected_answer") or item.get("expected_behavior")):
            raise ValueError(f"{item_id}: expected_correctness needs expected_answer or expected_behavior")
    return items


def judge_item(item: dict, faith_judge, correct_judge) -> dict:
    """Ask each judge the item expects a score from.

    Args:
        item: One calibration item.
        faith_judge: The judge model replying in the ``Faithfulness`` format.
        correct_judge: The judge model replying in the ``Correctness`` format.

    Returns:
        dict: ``"faithfulness"`` / ``"correctness"`` → the judge's reply, only for the judges the item has an
            ``expected_*`` score for.
    """
    tool_calls = [ToolCall(**call) for call in item.get("tool_calls") or []]
    earlier_turns = [EarlierTurn(**turn) for turn in item.get("earlier_turns") or []]
    replies = {}
    if "expected_faithfulness" in item:
        replies["faithfulness"] = judge_faithfulness(faith_judge, item, item["answer"], tool_calls, earlier_turns)
    if "expected_correctness" in item:
        replies["correctness"] = judge_correctness(correct_judge, item, item["answer"], earlier_turns)
    return replies


def calibration_totals(results: list[dict]) -> dict:
    """Count, per judge, how often it gave the expected score and how it went wrong.

    Args:
        results: One entry per judged score: ``judge``, ``expected``, ``got``.

    Returns:
        dict: Judge name → always the same keys: ``exact_match`` (``{"hits", "total"}``) — got == expected.
            ``zeros_caught`` (``{"hits", "total"}``) — of the answers that should score 0, how many did.
        ``missed_zeros`` (int) — should be 0, got more: errors the judge does not see.
        ``false_zeros`` (int) — should be above 0, got 0: harmless answers the judge punishes. ``confusion`` (dict[str,
            int]) — ``"expected 0 -> got 1"`` → count, only for mismatches.
    """
    totals = {}
    for judge in JUDGES:
        rows = [r for r in results if r["judge"] == judge]
        should_be_zero = [r for r in rows if r["expected"] == 0]
        totals[judge] = {
            "exact_match": {"hits": sum(r["got"] == r["expected"] for r in rows), "total": len(rows)},
            "zeros_caught": {"hits": sum(r["got"] == 0 for r in should_be_zero), "total": len(should_be_zero)},
            "missed_zeros": sum(r["got"] > 0 for r in should_be_zero),
            "false_zeros": sum(r["got"] == 0 and r["expected"] > 0 for r in rows),
            "confusion": dict(
                Counter(f"expected {r['expected']} -> got {r['got']}" for r in rows if r["got"] != r["expected"])
            ),
        }
    return totals


def main() -> None:
    """Run the calibration from the command line, print the result and save a report."""
    parser = argparse.ArgumentParser(
        prog="python -m eval.calibrate_judges", description="Check the judges against answers with known scores."
    )
    parser.add_argument("--only", help="Judge only the item with this id.")
    parser.add_argument("--repeat", type=int, default=1, help="Judge every item N times (default 1).")
    args = parser.parse_args()

    items = load_calibration_items(CALIBRATION_PATH)
    if args.only:
        items = [item for item in items if item["id"] == args.only]
    if not items:
        print("No items selected - check --only.")
        return

    judge_model = build_judge_model(JudgeSettings())
    faith_judge = judge_model.with_structured_output(Faithfulness)
    correct_judge = judge_model.with_structured_output(Correctness)

    results = []
    for repeat in range(1, args.repeat + 1):
        for index, item in enumerate(items, start=1):
            prefix = f"[{index}/{len(items)}]" + (f" run {repeat}" if args.repeat > 1 else "")
            print(f"{prefix} {item['id']} ... ", end="", flush=True)
            replies = judge_item(item, faith_judge, correct_judge)
            marks, details = [], []
            for judge, reply in replies.items():
                expected = item[f"expected_{judge}"]
                ok = reply.score == expected
                marks.append(f"{SHORT_NAMES[judge]} {reply.score}/{expected} {'OK' if ok else 'MISS'}")
                evidence = reply.claims if judge == "faithfulness" else reply.facts
                if not ok:
                    flagged = [e for e in evidence if e.status not in ("supported", "present")]
                    details += [
                        f"      {judge}: [{e.status}] {getattr(e, 'text', None) or e.fact}" for e in flagged[:2]
                    ] or [f"      {judge}: (nothing flagged)"]
                results.append(
                    {
                        "id": item["id"],
                        "run": repeat,
                        "judge": judge,
                        "expected": expected,
                        "got": reply.score,
                        "evidence": [e.model_dump() for e in evidence],
                    }
                )
            print("  ".join(marks))
            for line in details:
                print(line)

    totals = calibration_totals(results)
    print("\n" + "=" * 66)
    print("JUDGE CALIBRATION  (score got/expected)")
    print("=" * 66)
    for judge, t in totals.items():
        if not t["exact_match"]["total"]:
            continue
        print(f"{judge}:")
        print(f"  exact match:   {t['exact_match']['hits']}/{t['exact_match']['total']}")
        print(
            f"  zeros caught:  {t['zeros_caught']['hits']}/{t['zeros_caught']['total']}"
            f"   missed zeros: {t['missed_zeros']}   false zeros: {t['false_zeros']}"
        )
        for key, count in t["confusion"].items():
            print(f"    {key}: {count}")

    label = (
        "judge-calibration"
        + (f"_only-{args.only}" if args.only else "")
        + (f"_repeat-{args.repeat}" if args.repeat > 1 else "")
    )
    report_path = save_report_json(
        {"run_options": {"only": args.only, "repeat": args.repeat}, "totals": totals, "results": results},
        RESULTS_DIR,
        label,
    )
    print(f"\nReport: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
