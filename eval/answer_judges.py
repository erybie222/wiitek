"""Layer 2 of the evaluation: two LLM judges that score the agent's final answer.

- **Faithfulness** — is every claim in the answer backed by the data the
  agent's tools returned? The judge never sees the reference answer, so this
  measures made-up content (hallucination) on its own.
- **Correctness** — does the answer match the benchmark's verified reference?

They are two separate calls on purpose: one judge seeing both the tool data
and the reference would mix the two questions up.

In a multi-turn case both judges also see the earlier turns, so they can read
a follow-up like "a w 2024?"; they still score only the last answer.
"""

from __future__ import annotations

import json

from langchain_core.runnables import Runnable

import eval._src_path  # noqa: F401  (puts src/ on sys.path)

# isort: split
from templates import CORRECTNESS_PROMPT, FAITHFULNESS_PROMPT

from .benchmark_loader import reference_text
from .schemas import Correctness, EarlierTurn, Faithfulness, ToolCall


def format_earlier_turns(earlier_turns: list[EarlierTurn]) -> str:
    r"""Write the turns before the scored question as plain text for a judge prompt.

    Args:
        earlier_turns: The case's earlier turns, oldest first; empty for a single-question case.

    Returns:
        str: One "User: ..." and one "Bot: ..." line per turn, e.g.
        "User: Jaki był próg na Informatykę w 2025?\\nBot: 759 pkt ..."; a note that there were none when the list is
            empty.
    """
    if not earlier_turns:
        return "(none — this is the first question of the conversation)"
    return "\n\n".join(f"User: {turn.question}\nBot: {turn.answer}" for turn in earlier_turns)


def judge_faithfulness(
    judge: Runnable,
    case: dict,
    answer: str,
    tool_calls: list[ToolCall],
    earlier_turns: list[EarlierTurn],
) -> Faithfulness:
    """Ask the judge whether every claim in the answer is backed by the tool output.

    The judge sees the earlier turns, the question, the agent's tool calls with results, and the answer — never the
    reference, so it measures hallucination, not correctness. The earlier turns are context for reading the question
    only; the bot's earlier answers are not evidence.

    Args:
        judge: The judge model, set up to reply in the ``Faithfulness`` format.
        case: One benchmark case; only its question is used.
        answer: The agent's final answer — what gets checked.
        tool_calls: The agent's calls with results, from the whole conversation — the evidence every claim must be
            backed by.
        earlier_turns: Turns before the question; empty for a single question.

    Returns:
        Faithfulness: ``claims`` — every claim in the answer, each marked supported / contradicted / unsupported by the
            tool data — and a
        ``score``: 2 all supported, 1 nothing contradicted but something vague, 0 a contradiction or an invented detail.
    """
    tool_output = json.dumps([call.model_dump(mode="json") for call in tool_calls], ensure_ascii=False, indent=2)
    prompt = FAITHFULNESS_PROMPT.format(
        earlier_turns=format_earlier_turns(earlier_turns),
        question=case["question"],
        tool_output=tool_output if tool_calls else "(nie wywolano zadnego narzedzia)",
        answer=answer,
    )
    return judge.invoke(prompt)


def judge_correctness(judge: Runnable, case: dict, answer: str, earlier_turns: list[EarlierTurn]) -> Correctness:
    """Ask the judge whether the answer is right against the verified reference.

    The judge sees the earlier turns, the question, the case's ``expected_data`` as key facts, ``reference_text`` and
    the answer — not the tool output.

    Args:
        judge: The judge model, set up to reply in the ``Correctness`` format.
        case: One benchmark case — its question, key facts and reference.
        answer: The agent's final answer — what gets checked.
        earlier_turns: Turns before the question; empty for a single question.

    Returns:
        Correctness: ``facts`` — each key fact of the reference, marked
        present / missing / contradicted in the answer — and a ``score``:
        2 all present, 1 some missing but nothing wrong, 0 something wrong
        or a critical fact missing.
    """
    prompt = CORRECTNESS_PROMPT.format(
        earlier_turns=format_earlier_turns(earlier_turns),
        question=case["question"],
        key_facts=json.dumps(case.get("expected_data"), ensure_ascii=False, indent=2),
        reference=reference_text(case),
        answer=answer,
    )
    return judge.invoke(prompt)
