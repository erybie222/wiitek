"""SQL for the ``course_card_sections`` table: sections of the course card PDFs.

Each row belongs to one course row (``course_code``, ``major_id``, ``semester``) and is deleted with it
(``ON DELETE CASCADE``), so nothing here has to remove sections of courses that left the catalog.
"""

from psycopg import Cursor as DBCursor


def upsert_course_card_section(
    cursor: DBCursor,
    course_code: str,
    major_id: int,
    semester: int,
    source_url: str,
    section: str,
    chunk_index: int,
    content: str,
) -> None:
    """Insert one piece of a course's syllabus section, or update it if it is already stored.

    Args:
        cursor: Database cursor to run the SQL on.
        course_code: Catalog code of the course, e.g. "S1Inf1>ALIN".
        major_id: Id of the major whose catalog lists the course.
        semester: Semester the catalog lists the course in.
        source_url: Link to the syllabus PDF the text was read from.
        section: Section heading as printed in the PDF, e.g. "Literatura".
        chunk_index: Position of the piece within its section; 0 while sections are stored whole.
        content: Text of the piece.
    """
    cursor.execute(
        """
        INSERT INTO course_card_sections
        (course_code, major_id, semester, source_url, section, chunk_index, content)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (course_code, major_id, semester, section, chunk_index)
        DO UPDATE SET
            source_url = EXCLUDED.source_url,
            content = EXCLUDED.content,
            last_updated = CURRENT_TIMESTAMP;
    """,
        (course_code, major_id, semester, source_url, section, chunk_index, content),
    )


def delete_sections_not_seen_in_run(cursor: DBCursor, course_code: str, major_id: int, semester: int) -> int:
    """Delete the syllabus pieces of one course that the current ingestion run has not written.

    Call it right after storing the course's syllabus: what is left unwritten is a section the PDF no longer has. A row
    written in this run has ``last_updated`` equal to ``CURRENT_TIMESTAMP``, which Postgres fixes at the start of the
    transaction, so this only works while the whole run is one transaction.

    Args:
        cursor: Database cursor of the ingestion run.
        course_code: Catalog code of the course whose syllabus was just stored.
        major_id: Id of the course's major.
        semester: The course's semester.

    Returns:
        int: Number of pieces deleted, e.g. 1 when the new PDF dropped a section the old one had.
    """
    cursor.execute(
        """
        DELETE FROM course_card_sections
        WHERE course_code = %s AND major_id = %s AND semester = %s
          AND last_updated IS DISTINCT FROM CURRENT_TIMESTAMP;
    """,
        (course_code, major_id, semester),
    )
    return cursor.rowcount


def course_card_section_exists(cursor: DBCursor, course_code: str, section: str) -> bool:
    """Say whether the syllabus of a course has this section stored.

    Used by the evaluation to tell a benchmark typo (or a section the parser did not store) from an agent that searched
    badly.

    Args:
        cursor: Database cursor to run the SQL on.
        course_code: Catalog code of the course, e.g. "S1Inf1>ALIN".
        section: Section name, compared exactly, e.g. "Literatura".

    Returns:
        bool: True when a piece of this section is stored for some course with this code (on any major, in any
            semester).
    """
    cursor.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM course_card_sections
            WHERE course_code = %s AND section = %s
        );
    """,
        (course_code, section),
    )
    return cursor.fetchone()[0]


def delete_course_card_sections(cursor: DBCursor) -> None:
    """Delete every syllabus piece (used by ``clean_db.py``).

    Args:
        cursor: Database cursor to run the SQL on.
    """
    cursor.execute("""
        DELETE FROM course_card_sections;
    """)
