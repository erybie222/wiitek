"""Data shapes used by the evaluation.

Two groups:

- **Judge answers** (``Claim``, ``Faithfulness``, ``FactCheck``,
  ``Correctness``): the format each LLM judge must reply in.
- **Report records** (``ToolCall``, ``ToolScores``, ``EarlierTurn``, ``CaseRecord``): what is
  saved for every benchmark case in the JSON report.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Judge output schemas
# --------------------------------------------------------------------------
# These classes are sent to the judge model as the format it must answer in,
# so their docstrings and Field descriptions are part of the judge's
# instructions — edit them like a prompt.
# Field order matters too: the model fills fields in order, so putting the
# evidence (claims / facts) before the score makes it decide the score after
# the analysis instead of justifying a score it already picked.

class Claim(BaseModel):
    """One factual assertion pulled out of the answer."""

    text: str = Field(description="The claim, quoted or paraphrased from the answer, in Polish.")
    status: Literal["supported", "contradicted", "unsupported"] = Field(
        description="Whether the tool data supports it, contradicts it, or is silent."
    )


class Faithfulness(BaseModel):
    """Whether every claim in the answer is backed by the tool output."""

    claims: list[Claim] = Field(description="Every factual claim in the answer.")
    score: Literal[0, 1, 2] = Field(
        description="2 = all supported; 1 = nothing contradicted, one vague aside; 0 = a contradiction or an invented specific."
    )


class FactCheck(BaseModel):
    """One key fact from the reference, looked for in the answer."""

    fact: str = Field(description="A key fact from the reference data, in Polish.")
    status: Literal["present", "missing", "contradicted"] = Field(
        description="Whether the answer contains it, omits it, or contradicts it."
    )


class Correctness(BaseModel):
    """Whether the answer is actually right against the verified reference."""

    facts: list[FactCheck] = Field(description="The key facts and their status in the answer.")
    score: Literal[0, 1, 2] = Field(
        description="2 = every key fact present and correct; 1 = nothing contradicted, some missing; 0 = a contradiction or a missing critical fact."
    )


# --------------------------------------------------------------------------
# Report records
# --------------------------------------------------------------------------
# One CaseRecord per benchmark case is saved in the JSON report. The Field
# descriptions below document what each value in the report means.

class ToolCall(BaseModel):
    """One tool call with what it returned — made by the agent, or an expected call run by us."""

    name: str = Field(description="Tool name, e.g. get_course.")
    args: dict = Field(description="Arguments the call was made with.")
    result: Any = Field(
        default=None,
        description=(
            "What the tool returned: a list of rows, or for get_department one unit "
            "(dict) or None if the unit does not exist. For an agent call that failed, "
            "the error text; None if no result message was found."
        ),
    )


class ToolScores(BaseModel):
    """Layer-1 result for one case: did the agent call the right tools and get the right data."""

    called_right_tools: bool = Field(
        description=(
            "The agent called every tool the case needs (extra calls don't fail this; "
            "they go to extra_calls). On a case that needs no tools: nothing was called."
        )
    )
    got_right_data: bool | None = Field(
        description=(
            "Every expected call has an agent call to the same tool that returned the "
            "same data — for a syllabus search, one whose results include one of the "
            "case's expected_sections. Arguments are not compared, only results. None "
            "when the case needs no tools, or when the expected calls could not be run."
        )
    )
    called_no_tools: bool | None = Field(
        description=(
            "The agent called nothing. Scored only on cases that need no tools "
            "(out of scope, greetings); None on every other case."
        )
    )
    extra_calls: int = Field(
        description="How many more calls the agent made than the case needs; at least 0."
    )
    expected_tools: list[str] = Field(
        description=(
            "Tools the case needs, from expected_calls — without the ones marked "
            "only_resolves_prev_id, which are not required from the agent."
        )
    )
    called_tools: list[str] = Field(description="Tools the agent called, in call order.")
    wrong_data_tools: list[str] = Field(
        description=(
            "Needed tools for which no agent call returned the right data — called with "
            "wrong arguments, a search that returned none of the expected sections, or "
            "not called at all. Empty when got_right_data is True."
        )
    )
    expected_calls_error: str | None = Field(
        description=(
            "Why the expected calls could not be run — a benchmark or database problem, "
            "not the agent's. Prefixed with alternative_routes[N] when it was an "
            "alternative's calls that failed. None when they ran fine."
        )
    )
    scored_against_route: str = Field(
        default="expected_calls",
        description=(
            "Which route these scores compare the agent with: \"expected_calls\", or "
            "\"alternative_routes[N]\" when the agent did not follow expected_calls but "
            "followed that alternative. A case that followed no route is scored against "
            "expected_calls."
        ),
    )


class EarlierTurn(BaseModel):
    """One turn before the scored question in a multi-turn case — context, not scored itself."""

    question: str = Field(description="The question asked in this turn, from earlier_questions.")
    answer: str = Field(description="The agent's answer in this turn.")


class CaseRecord(BaseModel):
    """Everything about one case: the question, what the agent did, and both layers' scores."""

    id: str = Field(description="Case id from the benchmark.")
    category: str = Field(description="Benchmark category (the top-level list the case is in).")
    kind: Literal["lookup", "clarify", "refusal"] = Field(
        description="What the agent is expected to do; see case_kind."
    )
    earlier_turns: list[EarlierTurn] = Field(
        default_factory=list,
        description=(
            "Turns asked before the scored question in the same conversation, in order; "
            "empty for a single-question case."
        ),
    )
    question: str = Field(description="The question put to the agent — the scored one, asked last.")
    reference: str | None = Field(
        default=None,
        description="What the correctness judge compared the answer with; see reference_text.",
    )
    expected_data: Any = Field(default=None, description="The case's key facts, as in the benchmark.")
    answer: str | None = Field(default=None, description="The agent's final answer.")
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description=(
            "The agent's tool calls with results, in call order — from the whole "
            "conversation, earlier turns included, since that is what is scored."
        ),
    )
    tool_scores: ToolScores | None = Field(
        default=None, description="Layer 1. None only when asking the agent failed (see error)."
    )
    faithfulness: Faithfulness | None = Field(
        default=None, description="Layer-2 faithfulness judge. None with --no-judge or on error."
    )
    correctness: Correctness | None = Field(
        default=None, description="Layer-2 correctness judge. None with --no-judge or on error."
    )
    error: str | None = Field(
        default=None,
        description="Exception text if asking the agent failed; then answer and all scores are empty.",
    )
