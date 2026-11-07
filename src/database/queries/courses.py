"""SQL for the ``courses`` table: courses in each major's study plan, from the ECTS catalogs."""

from __future__ import annotations

from psycopg import Cursor as DBCursor

from database.serialization import row_to_dict


def delete_courses_of_major(cursor: DBCursor, major_id: int) -> None:
    """Delete all courses of one major.

    Ingestion calls this before storing the major's current catalog, so courses dropped from the catalog also disappear
    here.

    Args:
        cursor: Database cursor to run the SQL on.
        major_id: Id of the major.
    """
    cursor.execute(
        """
        DELETE FROM courses
        WHERE major_id = %s;
    """,
        (major_id,),
    )


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


def delete_courses(cursor: DBCursor) -> None:
    """Delete every course (used by ``clean_db.py``).

    Args:
        cursor: Database cursor to run the SQL on.
    """
    cursor.execute("""
        DELETE FROM courses;
    """)
