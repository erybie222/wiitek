"""Layer 1 of the evaluation: checking the agent's tool calls, without any LLM.

For each case the benchmark lists the tool calls a correct answer needs
(``expected_calls``). This module answers two questions about what the agent
actually did:

- did it call every tool the case needs? (``called_right_tools``)
- did those calls return the right data? (``got_right_data``) — checked by
  running the expected calls on the database and comparing results, so the
  exact wording of the agent's arguments does not matter. A syllabus search
  is the exception: it is not run, because its ranking depends on how the
  query is worded; it lists the sections a correct answer can come from
  (``expected_sections``), and the agent's search results must include one.

A case may also list ``alternative_routes``: other call sequences that are
equally valid. ``score_tool_calls_with_alternatives`` scores the agent against
``expected_calls`` first and tries the alternatives only when that fails.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage

from . import _src_path  # noqa: F401  (puts src/ on sys.path)

# isort: split
from database.connection import get_db_connection
from database.queries.course_card_sections import course_card_section_exists
from database.repository import TOOLS

from .schemas import ToolCall, ToolScores

TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}

# Placeholder in benchmark arguments meaning "the id returned by the previous
# expected call". Staff ids change on every ingestion, so the benchmark cannot
# contain real ones.
PREV_ID = "$prev.id"


def fill_prev_id(args: dict, previous_rows: list | None) -> dict:
    """Replace ``$prev.id`` in an expected call's arguments with a real id.

    Staff ids change on every ingestion, so the benchmark chains calls with ``$prev.id`` instead of hard-coding an id.

    Args:
        args: Arguments of one expected call, as written in the benchmark.
        previous_rows: Rows returned by the preceding expected call, or None.

    Returns:
        dict: Copy of ``args`` with each ``$prev.id`` replaced by the ``id`` of the single row in ``previous_rows`` — or
            None when there are zero or several rows, since picking one would hide the ambiguity. Other values
            are unchanged.
    """
    resolved = {}
    for key, value in (args or {}).items():
        if value != PREV_ID:
            resolved[key] = value
        elif previous_rows and len(previous_rows) == 1:
            resolved[key] = previous_rows[0].get("id")
        else:
            resolved[key] = None
    return resolved


def run_expected_calls(calls: list[dict]) -> list[ToolCall]:
    """Run a case's expected calls on the live DB to get the correct data.

    This is the answer key for ``got_right_data``: ``score_tool_calls`` compares the agent's results with these. It is
    computed on every run, not stored in the benchmark, because ids and data change between ingestions. Calls run in
    order; ``$prev.id`` resolves from the last call that returned a list.

    Args:
        calls: Expected calls to run, each ``{"tool", "args"}`` — never a search call with ``expected_sections``, which
            has no arguments to run (see ``returned_expected_section``).

    Returns:
        list[ToolCall]: One per expected call, in order. ``args`` are the arguments actually used (after
            ``fill_prev_id``), ``result`` is what the tool returned.

    Raises:
        Exception: Whatever the tool raises (a DB error, or ``KeyError`` for an unknown tool name); ``score_tool_calls``
            catches it.
    """
    results = []
    previous_rows = None
    for call in calls:
        name = call["tool"]
        args = fill_prev_id(call.get("args"), previous_rows)
        # Bare args would return only the text; a tool_call returns the ToolMessage with the artifact.
        message = TOOLS_BY_NAME[name].invoke({"type": "tool_call", "name": name, "args": args, "id": "expected"})
        result = message.artifact
        results.append(ToolCall(name=name, args=args, result=result))
        if isinstance(result, list):
            previous_rows = result
    return results


def agent_tool_calls(messages: list[BaseMessage]) -> list[ToolCall]:
    """Extract the agent's tool calls from its messages, each with its result.

    Calls and results are joined by ``tool_call_id``, not position: a batched turn interleaves them. A failure is
    detected by ``status``, not ``artifact is None``, since ``get_department`` returns None legitimately.

    Args:
        messages: The message list returned by ``agent.ask``.

    Returns:
        list[ToolCall]: One per agent tool call, in call order. ``result`` is what the tool returned, the error text if
            the call failed, or None if no result message was found.
    """
    results = {
        message.tool_call_id: message.content if message.status == "error" else message.artifact
        for message in messages
        if message.type == "tool"
    }
    return [
        ToolCall(name=call["name"], args=call["args"], result=results.get(call["id"]))
        for message in messages
        if message.type == "ai"
        for call in message.tool_calls
    ]


def returned_expected_section(expected_sections: list[dict], agent_calls: list[ToolCall], tool: str) -> bool:
    """Say whether any of the agent's searches returned one of the sections a case accepts.

    A search is scored this way instead of being run like other expected calls: its ranking depends on how the query is
    worded, so the benchmark names the sections a correct answer can come from, not a query.

    Args:
        expected_sections: The accepted sections, each ``{"course_code", "section"}``,
            e.g. ``{"course_code": "S1Inf1>ALIN", "section": "Literatura"}``.
        agent_calls: The agent's calls with results, from ``agent_tool_calls``.
        tool: Name of the search tool; only the agent's calls to it are read.

    Returns:
        bool: True when some search result row has the section of an accepted entry and lists that entry's course code
            among its ``courses`` (one row per syllabus fragment; a fragment shared by several courses, like the
            faculty-wide PE syllabus, lists them all). False when no search was made, or none returned such a row.
    """
    rows = [
        row
        for call in agent_calls
        if call.name == tool and isinstance(call.result, list)
        for row in call.result
        if isinstance(row, dict)
    ]
    return any(
        row.get("section") == wanted["section"]
        and any(course.get("course_code") == wanted["course_code"] for course in row.get("courses") or [])
        for wanted in expected_sections
        for row in rows
    )


def expected_sections_not_stored(search_calls: list[dict]) -> list[str]:
    """List the syllabus sections a case expects that the database does not hold at all.

    Such a section can never be returned by a search, so a miss on it is a benchmark typo or a section the parser did
    not store — not the agent's fault.

    Args:
        search_calls: The case's expected searches, each ``{"tool", "expected_sections"}``.

    Returns:
        list[str]: One "course_code / section" per missing section, e.g. ``["S1Inf1>ALIN / Literatura"]``; empty when
            all are stored (or there are no searches, in which case the database is not queried).
    """
    wanted_sections = [wanted for call in search_calls for wanted in call["expected_sections"]]
    if not wanted_sections:
        return []
    with get_db_connection() as connection, connection.cursor() as cursor:
        return [
            f"{wanted['course_code']} / {wanted['section']}"
            for wanted in wanted_sections
            if not course_card_section_exists(cursor, wanted["course_code"], wanted["section"])
        ]


def score_tool_calls(expected: list[dict], agent_calls: list[ToolCall]) -> ToolScores:
    """Score the agent's tool calls on one case against one route (no LLM).

    A route is ``expected_calls`` or one of the case's ``alternative_routes``; ``score_tool_calls_with_alternatives``
    decides which to score against. Two questions, answered separately so a failure says which one went wrong:

    - ``called_right_tools`` — did the agent call every tool the case needs?
    Only tool names count. Extra calls do not fail it; they are counted once, in ``extra_calls``, instead of being
    punished twice.
    - ``got_right_data`` — did those calls return the right data? The
    expected calls are run on the DB (``run_expected_calls``) and each result must equal the result of some agent call
    to the same tool. Arguments are never compared, so "Informatyka" vs "informatyka" costs nothing as long as the data
    is the same. Caveat: if the right result is empty, a wrong call that also finds nothing passes too. An expected
    search (a call with ``expected_sections`` instead of ``args``) is not run: some agent search must have returned one
    of its sections (``returned_expected_section``).

    An expected call marked ``only_resolves_prev_id: true`` is run but never required from the agent: it exists only so
    the next call's ``$prev.id`` has one row to read. E.g. turn 1 finds two people with one name, turn 2 says which one
    — the agent rightly takes that person's id from the conversation, while the benchmark needs a filtered ``get_staff``
    to know the id. Such a call counts in no score; an agent that makes it anyway gets one ``extra_calls``.

    Args:
        expected: The route's calls, each ``{"tool", "args"}`` — or, for a search, ``{"tool", "expected_sections"}``;
            empty means no tools are needed.
        agent_calls: The agent's calls with results, from ``agent_tool_calls``.

    Returns:
        ToolScores: The case's layer-1 result. What each field means is described on the ``ToolScores`` model.
    """
    called_tools = [call.name for call in agent_calls]
    # Calls the agent must make; only_resolves_prev_id calls are run below but not required.
    required = [not call.get("only_resolves_prev_id") for call in expected]
    expected_tools = [call["tool"] for call, needed in zip(expected, required) if needed]
    extra_calls = max(0, len(called_tools) - len(expected_tools))

    # Case needs no tools (out of scope, greeting, ...): the only right move is
    # to call nothing, so both called_right_tools and called_no_tools mean
    # "nothing was called", and every call counts as extra. There are no
    # expected calls to run, so got_right_data is not scored.
    if not expected:
        return ToolScores(
            called_right_tools=not called_tools,
            got_right_data=None,
            called_no_tools=not called_tools,
            extra_calls=len(called_tools),
            expected_tools=[],
            called_tools=called_tools,
            wrong_data_tools=[],
            expected_calls_error=None,
        )

    # Right tools: tick off each needed tool against the agent's calls; one
    # call can only cover one needed tool.
    missing_tools = []
    unclaimed = list(called_tools)
    for name in expected_tools:
        if name in unclaimed:
            unclaimed.remove(name)
        else:
            missing_tools.append(name)

    # Right data, step 1: run the expected calls now to get the correct rows.
    # Searches are not run; their sections are checked in step 3, but each
    # must exist in the database, or no agent could ever find it.
    run_calls = [call for call in expected if "expected_sections" not in call]
    search_calls = [call for call in expected if "expected_sections" in call]
    run_calls_required = [not call.get("only_resolves_prev_id") for call in run_calls]
    expected_calls_error = None
    try:
        correct_results = run_expected_calls(run_calls)
        sections_not_stored = expected_sections_not_stored(search_calls)
    except Exception as error:
        expected_calls_error = f"could not run an expected call: {error}"
    else:
        if sections_not_stored:
            expected_calls_error = (
                "expected course card sections not in the database (a typo in the benchmark, or the parser did not "
                "store them): " + "; ".join(sections_not_stored)
            )
    if expected_calls_error:
        return ToolScores(
            called_right_tools=not missing_tools,
            got_right_data=None,
            called_no_tools=None,
            extra_calls=extra_calls,
            expected_tools=expected_tools,
            called_tools=called_tools,
            wrong_data_tools=[],
            expected_calls_error=expected_calls_error,
        )

    # Right data, step 2: each correct result needs an agent call to the same
    # tool with an equal result. A matched agent call is used up so it cannot
    # cover two expected calls.
    wrong_data_tools = []
    unused_agent_calls = list(agent_calls)
    for correct, needed in zip(correct_results, run_calls_required):
        if not needed:
            continue
        agent_call = next(
            (call for call in unused_agent_calls if call.name == correct.name and call.result == correct.result),
            None,
        )
        if agent_call is None:
            wrong_data_tools.append(correct.name)
        else:
            unused_agent_calls.remove(agent_call)

    # Right data, step 3: each expected search needs an agent search that
    # returned one of its sections. Searches are not used up: one search may
    # return the sections of two expected calls.
    for call in search_calls:
        if not returned_expected_section(call["expected_sections"], agent_calls, call["tool"]):
            wrong_data_tools.append(call["tool"])

    return ToolScores(
        called_right_tools=not missing_tools,
        got_right_data=not wrong_data_tools,
        called_no_tools=None,
        extra_calls=extra_calls,
        expected_tools=expected_tools,
        called_tools=called_tools,
        wrong_data_tools=wrong_data_tools,
        expected_calls_error=None,
    )


def agent_followed_route(scores: ToolScores) -> bool:
    """Say whether the agent's calls pass against one route.

    Args:
        scores: Result of ``score_tool_calls`` against that route.

    Returns:
        bool: True when the agent called every tool of the route and got the same data. ``got_right_data`` None counts
            as a pass only for a route with no calls — never when the route's calls could not be run.
    """
    return scores.called_right_tools and scores.got_right_data is not False and scores.expected_calls_error is None


def score_tool_calls_with_alternatives(case: dict, agent_calls: list[ToolCall]) -> ToolScores:
    """Score the agent's tool calls on one case, accepting any listed route.

    ``expected_calls`` is scored first. Only when the agent did not follow it are the case's ``alternative_routes``
    tried, in order, so their expected calls hit the database only when needed. The first alternative the agent followed
    is returned. A case that follows no route is scored against ``expected_calls`` — the route the prompt prescribes —
    so its failure says what was missing from that route.

    When ``expected_calls`` cannot be run, alternatives are not tried: that is a benchmark or database problem, and an
    alternative passing would hide it.

    Args:
        case: One benchmark case, with ``expected_calls`` and optionally ``alternative_routes`` (each ``{"why",
            "calls"}``).
        agent_calls: The agent's calls with results, from ``agent_tool_calls``.

    Returns:
        ToolScores: The scores of the route the agent followed, with ``scored_against_route`` saying which one
            ("expected_calls" or "alternative_routes[N]"); the ``expected_calls`` scores if it followed none. If an
            alternative's calls could not be run, its error is put in ``expected_calls_error`` of the returned scores.
    """
    expected_route_scores = score_tool_calls(case.get("expected_calls") or [], agent_calls)
    if agent_followed_route(expected_route_scores) or expected_route_scores.expected_calls_error:
        return expected_route_scores

    for position, alternative in enumerate(case.get("alternative_routes") or []):
        route_name = f"alternative_routes[{position}]"
        alternative_scores = score_tool_calls(alternative["calls"] or [], agent_calls)
        if alternative_scores.expected_calls_error:
            return expected_route_scores.model_copy(
                update={
                    "expected_calls_error": f"{route_name}: {alternative_scores.expected_calls_error}",
                }
            )
        if agent_followed_route(alternative_scores):
            return alternative_scores.model_copy(update={"scored_against_route": route_name})

    return expected_route_scores
