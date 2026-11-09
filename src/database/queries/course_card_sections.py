"""SQL for the ``course_card_sections`` table: sections of the course card PDFs.

Each row belongs to one course row (``course_code``, ``major_id``, ``semester``) and is deleted with it
(``ON DELETE CASCADE``), so nothing here has to remove sections of courses that left the catalog.
"""

from __future__ import annotations

from psycopg import Cursor as DBCursor

from database.serialization import row_to_dict


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

    When the stored text changes, its vector is cleared with it, so a vector of the old text can never be searched; the
    embedding step computes a new one. An unchanged text keeps its vector.

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
            -- Every expression here reads the row as it was before this update.
            content_hash = CASE WHEN course_card_sections.content = EXCLUDED.content
                                THEN course_card_sections.content_hash END,
            embedding = CASE WHEN course_card_sections.content = EXCLUDED.content
                             THEN course_card_sections.embedding END,
            embedding_model = CASE WHEN course_card_sections.content = EXCLUDED.content
                                   THEN course_card_sections.embedding_model END,
            last_updated = CURRENT_TIMESTAMP;
    """,
        (course_code, major_id, semester, source_url, section, chunk_index, content),
    )


def get_sections_for_embedding(cursor: DBCursor) -> list[dict]:
    """List every stored syllabus piece with what the embedding step needs to decide whether its vector is current.

    Args:
        cursor: Database cursor of the ingestion run.

    Returns:
        list[dict]: One dict per piece, ordered by id. Each dict: ``id`` (int). ``course_name`` (str) — the course's
            current name in the catalog, e.g. "Algebra liniowa". ``section`` (str), e.g. "Literatura". ``content``
            (str). ``content_hash`` (str | None) — hash of the text its stored vector was computed from, None when it
            has none. ``embedding_model`` (str | None), e.g. "text-embedding-3-large". ``has_embedding`` (bool).
    """
    cursor.execute("""
        SELECT s.id, c.name, s.section, s.content, s.content_hash, s.embedding_model, s.embedding IS NOT NULL
        FROM course_card_sections s
        JOIN courses c USING (course_code, major_id, semester)
        ORDER BY s.id;
    """)
    columns = ("id", "course_name", "section", "content", "content_hash", "embedding_model", "has_embedding")
    return [row_to_dict(columns, row) for row in cursor.fetchall()]


def copy_embedding_by_hash(cursor: DBCursor, ids: list[int], content_hash: str, embedding_model: str) -> bool:
    """Give pieces the vector another piece already has for the same text and model, without asking the API.

    This is how the copies of a faculty-wide syllabus (PE, listed on every major) share one computed vector.

    Args:
        cursor: Database cursor of the ingestion run.
        ids: Ids of the pieces to update.
        content_hash: Hash of the text they would be embedded from.
        embedding_model: The model the vector must come from.

    Returns:
        bool: True when a stored vector was found and copied; False when none exists yet, so it must be computed.
    """
    cursor.execute(
        """
        UPDATE course_card_sections AS target
        SET embedding = source.embedding, content_hash = %s, embedding_model = %s
        FROM (
            SELECT embedding FROM course_card_sections
            WHERE content_hash = %s AND embedding_model = %s AND embedding IS NOT NULL
            LIMIT 1
        ) AS source
        WHERE target.id = ANY(%s);
    """,
        (content_hash, embedding_model, content_hash, embedding_model, ids),
    )
    return cursor.rowcount > 0


def set_embedding(cursor: DBCursor, ids: list[int], content_hash: str, embedding_model: str, embedding: str) -> None:
    """Store a computed vector on every piece embedded from the same text.

    ``last_updated`` is left alone: it says when the text was read from the PDF, not when it was embedded.

    Args:
        cursor: Database cursor of the ingestion run.
        ids: Ids of the pieces whose text has this hash.
        content_hash: Hash of the text the vector was computed from.
        embedding_model: The model that computed it, e.g. "text-embedding-3-large".
        embedding: The vector as a pgvector literal, e.g. "[0.012,-0.034,...]".
    """
    cursor.execute(
        """
        UPDATE course_card_sections
        SET embedding = %s::vector, content_hash = %s, embedding_model = %s
        WHERE id = ANY(%s);
    """,
        (embedding, content_hash, embedding_model, ids),
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
