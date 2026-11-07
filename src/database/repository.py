"""The database lookups the agent can call, one per table.

Each function here wraps one ``get_*`` query from ``database/queries`` as a
LangChain tool. **Their docstrings are written for the model, not for
developers**: LangChain sends the docstring, including the ``Args:``
section, to the model as the tool's description, and the model decides from
it when and how to call the tool. Editing a docstring changes how the agent
behaves, so re-run the evaluation afterwards.

Each tool returns its result twice: as a JSON string, which is what the
model reads, and as the original Python rows, which LangChain keeps in
``ToolMessage.artifact`` for the evaluation. The JSON is produced here rather
than by LangChain because LangChain turns an empty list into an empty message
instead of ``"[]"``, and the model would not see that nothing was found.

Every call borrows its own database connection and returns it at the end.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from database.connection import get_db_connection
from database.queries.admission_thresholds import (
    get_admission_threshold as _get_admission_threshold,
)
from database.queries.courses import get_course as _get_course
from database.queries.departments import get_department as _get_department
from database.queries.faculties import get_faculties as _get_faculties
from database.queries.majors import get_major as _get_major
from database.queries.office_hours import get_office_hour as _get_office_hour
from database.queries.staff import get_staff as _get_staff
from database.queries.staff_functions import get_staff_function as _get_staff_function


@tool(parse_docstring=True, response_format="content_and_artifact")
def get_faculties() -> tuple[str, list[dict]]:
    """List the faculties (wydziały) the database has data for: symbol and full name.

    Call this only when the question is about the faculties themselves, e.g. which faculties you have data for, or
    faculty's full name. It is not a scope check: a greeting, a question about what you can do, and a question about
    another faculty, another university or a topic no tool covers are all answered without calling any tool.
    """
    with get_db_connection() as conn, conn.cursor() as cursor:
        rows = _get_faculties(cursor)
    return json.dumps(rows, ensure_ascii=False), rows


@tool(parse_docstring=True, response_format="content_and_artifact")
def get_major(
    major_name: str | None = None,
    degree: str | None = None,
    study_mode: str | None = None,
) -> tuple[str, list[dict]]:
    """Look up majors (kierunki) by name, or list all of them.

    Call with no arguments to list every major the system covers — do this before claiming any major does or does not
    exist. Returns every match instead of one best guess: if more than one row comes back for a name, it doesn't
    uniquely identify a major, so ask which degree, study mode or teaching language is meant. A major with an
    `english_name` is taught in English; a null `english_name` means it is taught in Polish.

    Args:
        major_name: Polish name, or the English name of an English-taught major (case- and diacritic-insensitive). Omit
            to list all majors.
        degree: Optional filter: "bachelor-of-engineering" or "masters".
        study_mode: Optional filter: "full-time" or "part-time".
    """
    with get_db_connection() as conn, conn.cursor() as cursor:
        rows = _get_major(cursor, major_name, degree, study_mode)
    return json.dumps(rows, ensure_ascii=False), rows


@tool(parse_docstring=True, response_format="content_and_artifact")
def get_admission_threshold(
    major_name: str,
    year: int,
    round: int | None = None,
    degree: str | None = None,
    study_mode: str | None = None,
) -> tuple[str, list[dict]]:
    """Look up the admission points threshold (próg punktowy) for a major in a recruitment year.

    An empty result means there is no stored threshold for that major/year — it may not have been offered that year, or
    the year may be outside the data the system has. Say you don't know rather than estimate a number. Rows for a
    Polish- and an English-taught version of one major differ in `major_english_name` (null for the Polish-taught one) —
    never merge them into one figure.

    Args:
        major_name: Polish name, or the English name of an English-taught major (case- and diacritic-insensitive).
        year: Recruitment year.
        round: Optional filter on recruitment round (1 or 2).
        degree: Optional filter: "bachelor-of-engineering" or "masters".
        study_mode: Optional filter: "full-time" or "part-time".
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            rows = _get_admission_threshold(cursor, major_name, year, round, degree, study_mode)
    return json.dumps(rows, ensure_ascii=False), rows


@tool(parse_docstring=True, response_format="content_and_artifact")
def get_course(
    major_name: str,
    course_name: str | None = None,
    semester: int | None = None,
    is_obligatory: bool | None = None,
    is_english: bool | None = None,
) -> tuple[str, list[dict]]:
    """Look up study-plan courses (przedmioty) for a major.

    An empty result means this major's study plan has no course matching those filters — say so plainly instead of
    offering a similarly named course from another semester or major. The same course name can carry different ECTS and
    hours on different majors, so never present one major's figures as universal.

    Args:
        major_name: Polish name, or the English name of an English-taught major (case- and diacritic-insensitive).
        course_name: Optional substring filter on course name (case- and diacritic-insensitive).
        semester: Optional filter on semester (1..8).
        is_obligatory: Optional filter: true for obligatory, false for elective. Leave unset when asked whether a course
            is obligatory — read it from the result.
        is_english: Optional filter for English-taught courses. Leave unset when asked whether a course is
            English-taught — read it from the result.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            rows = _get_course(cursor, major_name, course_name, semester, is_obligatory, is_english)
    return json.dumps(rows, ensure_ascii=False), rows


@tool(parse_docstring=True, response_format="content_and_artifact")
def get_department(name: str) -> tuple[str, dict | None]:
    """Look up one organizational unit (jednostka: instytut, zakład, dziekanat) by name.

    The result includes "parent" (the unit directly above it, or null when it sits directly under the faculty) and
    "children" (its direct sub-units). A null result means the unit doesn't exist; an empty "children" list means it
    exists but has no sub-units — those are different and must not be confused.

    Args:
        name: Unit name to look up (case- and diacritic-insensitive).
    """
    with get_db_connection() as conn, conn.cursor() as cursor:
        rows = _get_department(cursor, name)
    return json.dumps(rows, ensure_ascii=False), rows


@tool(parse_docstring=True, response_format="content_and_artifact")
def get_staff(
    last_name: str,
    first_name: str | None = None,
    department: str | None = None,
) -> tuple[str, list[dict]]:
    """Look up staff members (pracownicy, kadra) by last name.

    Returns every match instead of one best guess. This faculty has staff members who share the same first and last
    name, so if more than one row comes back, ask the user to disambiguate (e.g. by department or academic
    title) before answering anything about that person — never pick one.

    Args:
        last_name: Family name to look up (case- and diacritic-insensitive).
        first_name: Optional exact filter on given name (case- and diacritic-insensitive).
        department: Optional exact filter on unit name (case- and diacritic-insensitive).
    """
    with get_db_connection() as conn, conn.cursor() as cursor:
        rows = _get_staff(cursor, last_name, first_name, department)
    return json.dumps(rows, ensure_ascii=False), rows


@tool(parse_docstring=True, response_format="content_and_artifact")
def get_staff_function(
    staff_id: int | None = None,
    function_name: str | None = None,
    department_name: str | None = None,
) -> tuple[str, list[dict]]:
    """Look up administrative functions (funkcje, np. dyrektor instytutu), with the holder's staff details.

    Use function_name/department_name for "who holds function X in unit Y" (e.g. function_name="Dyrektor instytutu").
    Use staff_id (resolved via get_staff) for "what functions does this person hold". A function is an appointed role,
    distinct from the person's regular employment position.

    Args:
        staff_id: Optional exact filter on the holder's staff id.
        function_name: Optional exact filter on function name (case- and diacritic-insensitive).
        department_name: Optional exact filter on the unit the function is held in (case- and diacritic-insensitive).
    """
    with get_db_connection() as conn, conn.cursor() as cursor:
        rows = _get_staff_function(cursor, staff_id, function_name, department_name)
    return json.dumps(rows, ensure_ascii=False), rows


@tool(parse_docstring=True, response_format="content_and_artifact")
def get_office_hour(staff_id: int, day_of_week: str | None = None) -> tuple[str, list[dict]]:
    """Look up weekly office hours (konsultacje) for one staff member.

    Call get_staff first to resolve staff_id, and if it returned more than one match, ask the user to disambiguate
    before calling this. An empty result means the person has no currently published office hours — that is different
    from the person not existing, so don't say you have no record of them.

    Args:
        staff_id: The staff member's id, from get_staff.
        day_of_week: Optional filter on Polish weekday name (e.g. "poniedziałek").
    """
    with get_db_connection() as conn, conn.cursor() as cursor:
        rows = _get_office_hour(cursor, staff_id, day_of_week)
    return json.dumps(rows, ensure_ascii=False), rows


TOOLS = [
    get_faculties,
    get_major,
    get_admission_threshold,
    get_course,
    get_department,
    get_staff,
    get_staff_function,
    get_office_hour,
]
