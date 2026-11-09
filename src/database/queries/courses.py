"""SQL for the ``courses`` table: courses in each major's study plan, from the ECTS catalogs."""

from __future__ import annotations

from psycopg import Cursor as DBCursor

from database.serialization import row_to_dict


def delete_courses_not_seen_in_run(cursor: DBCursor, major_id: int) -> int:
    """Delete the courses of one major that the current ingestion run has not written.

    Ingestion calls this right after storing the major's catalog, so a course dropped from the catalog disappears here
    too, while a course still in it keeps its row (and whatever references that row). A row written in this run has
    ``last_updated`` equal to ``CURRENT_TIMESTAMP``, which Postgres fixes at the start of the transaction, so this only
    works while the whole run is one transaction.

    Args:
        cursor: Database cursor of the ingestion run.
        major_id: Id of the major whose catalog was just stored.

    Returns:
        int: Number of courses deleted, e.g. 2 when two courses left the catalog since the last run.
    """
    cursor.execute(
        """
        DELETE FROM courses
        WHERE major_id = %s
          AND last_updated IS DISTINCT FROM CURRENT_TIMESTAMP;
    """,
        (major_id,),
    )
    return cursor.rowcount


def upsert_course(
    cursor: DBCursor,
    code: str,
    major_id: int,
    semester: int,
    name: str,
    is_obligatory: bool,
    is_english: bool,
    ects: int,
    hours_lecture: int,
    hours_classes: int,
    hours_lab: int,
    hours_project: int,
    assessment_type: str,
    source_url: str | None,
) -> None:
    """Insert a course, or update it if the major already has this code in this semester.

    A course is one row of a major's catalog: the same code can be listed on several majors' pages (e.g. PE electives)
    and in more than one semester, and each listing is stored as its own row.

    Args:
        cursor: Database cursor to run the SQL on.
        code: Course code from the catalog.
        major_id: Id of the major whose study plan the course belongs to.
        semester: Semester number, 1 to 8.
        name: Course name.
        is_obligatory: True for an obligatory course, False for an elective.
        is_english: True if taught in English.
        ects: ECTS points.
        hours_lecture: Lecture hours (W).
        hours_classes: Classes hours (C).
        hours_lab: Lab hours (L).
        hours_project: Project hours (P).
        assessment_type: "Egzamin" (exam) or "Zaliczenie z ocena" (graded credit).
        source_url: Link to the course card PDF, or None if the catalog has none.
    """
    cursor.execute(
        """
        INSERT INTO courses
        (course_code, major_id, semester, name, is_obligatory, is_english, ects,
         hours_lecture, hours_classes, hours_lab, hours_project, assessment_type, source_url)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (course_code, major_id, semester)
        DO UPDATE SET
            name = EXCLUDED.name,
            is_obligatory = EXCLUDED.is_obligatory,
            is_english = EXCLUDED.is_english,
            ects = EXCLUDED.ects,
            hours_lecture = EXCLUDED.hours_lecture,
            hours_classes = EXCLUDED.hours_classes,
            hours_lab = EXCLUDED.hours_lab,
            hours_project = EXCLUDED.hours_project,
            assessment_type = EXCLUDED.assessment_type,
            source_url = EXCLUDED.source_url,
            last_updated = CURRENT_TIMESTAMP;
    """,
        (
            code,
            major_id,
            semester,
            name,
            is_obligatory,
            is_english,
            ects,
            hours_lecture,
            hours_classes,
            hours_lab,
            hours_project,
            assessment_type,
            source_url,
        ),
    )


def get_course(
    cursor: DBCursor,
    major_name: str,
    course_name: str | None = None,
    semester: int | None = None,
    is_obligatory: bool | None = None,
    is_english: bool | None = None,
) -> list[dict]:
    """Find the courses of one major, optionally narrowed down.

    Args:
        cursor: Database cursor to run the SQL on.
        major_name: Polish name, or the English name of an English-taught major; letter case and Polish diacritics are
            ignored.
        course_name: Only courses whose name contains this text (case is ignored); None for all.
        semester: Only this semester; None for all.
        is_obligatory: True for obligatory only, False for electives only, None for both.
        is_english: True for English-taught only, False for Polish only, None for both.

    Returns:
        list[dict]: Matching courses ordered by semester and name; empty if
        none match. Sums such as total ECTS are left to the caller. Each dict:
            ``course_code`` (str) — catalog code.
            ``name`` (str) — course name.
            ``semester`` (int) — semester number.
            ``is_obligatory`` / ``is_english`` (bool) — as in the arguments.
            ``ects`` (int) — ECTS points.
            ``hours_lecture`` / ``hours_classes`` / ``hours_lab`` /
                ``hours_project`` (int) — hours per semester of each form.
            ``assessment_type`` (str) — "Egzamin" or "Zaliczenie z ocena".
            ``source_url`` (str | None) — course card link.
            ``last_updated`` (str) — when the row was last written (ISO date-time).
    """
    # Substring match, so "Analiza" also finds "Analiza matematyczna 1";
    # diacritics are ignored on both sides, as in every name lookup here.
    course_pattern = f"%{course_name}%" if course_name is not None else None

    # Optional filters: each value is passed twice — once for the IS NULL
    # check (None = no filter), once for the comparison.
    cursor.execute(
        """
        SELECT c.course_code, c.name, c.semester, c.is_obligatory, c.is_english,
               c.ects, c.hours_lecture, c.hours_classes, c.hours_lab, c.hours_project,
               c.assessment_type, c.source_url, c.last_updated
        FROM courses c
        JOIN majors m ON m.id = c.major_id
        WHERE (LOWER(unaccent(m.polish_name)) = LOWER(unaccent(%s::text))
               OR LOWER(unaccent(m.english_name)) = LOWER(unaccent(%s::text)))
          AND (%s::text IS NULL OR unaccent(c.name) ILIKE unaccent(%s::text))
          AND (%s::int IS NULL OR c.semester = %s::int)
          AND (%s::boolean IS NULL OR c.is_obligatory = %s::boolean)
          AND (%s::boolean IS NULL OR c.is_english = %s::boolean)
        ORDER BY c.semester, c.name;
    """,
        (
            major_name,
            major_name,
            course_pattern,
            course_pattern,
            semester,
            semester,
            is_obligatory,
            is_obligatory,
            is_english,
            is_english,
        ),
    )

    columns = (
        "course_code",
        "name",
        "semester",
        "is_obligatory",
        "is_english",
        "ects",
        "hours_lecture",
        "hours_classes",
        "hours_lab",
        "hours_project",
        "assessment_type",
        "source_url",
        "last_updated",
    )
    return [row_to_dict(columns, row) for row in cursor.fetchall()]


def get_courses_with_card_links(cursor: DBCursor) -> list[dict]:
    """List every stored course that links to a course card PDF.

    Args:
        cursor: Database cursor to run the SQL on.

    Returns:
        list[dict]: One dict per course row, ordered by link; courses without a link are left out. Each dict:
            ``course_code`` (str) — catalog code, e.g. "S1Inf1>ALIN". ``major_id`` (int) — id of the major whose catalog
            lists it. ``semester`` (int) — semester it is listed in. ``source_url`` (str) — link to its course card
            PDF, e.g.
        "https: //put.poznan.pl/cards/2026_2027/Informatyka/.../Algebra liniowa (ALIN).pdf".
    """
    cursor.execute("""
        SELECT course_code, major_id, semester, source_url
        FROM courses
        WHERE source_url IS NOT NULL
        ORDER BY source_url, course_code, major_id, semester;
    """)
    columns = ("course_code", "major_id", "semester", "source_url")
    return [row_to_dict(columns, row) for row in cursor.fetchall()]


def delete_courses(cursor: DBCursor) -> None:
    """Delete every course (used by ``clean_db.py``).

    Args:
        cursor: Database cursor to run the SQL on.
    """
    cursor.execute("""
        DELETE FROM courses;
    """)
