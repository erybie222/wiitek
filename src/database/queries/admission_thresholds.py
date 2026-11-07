"""SQL for the ``admission_thresholds`` table: minimum points and places per major and year."""

from __future__ import annotations

from psycopg import Cursor as DBCursor

from database.serialization import row_to_dict


def upsert_threshold(
    cursor: DBCursor,
    major_id: int,
    year: int,
    points: float,
    capacity: int,
    source_url: str,
) -> None:
    """Insert one major's first-round threshold for one year, or update it if already stored.

    Args:
        cursor: Database cursor to run the SQL on.
        major_id: Id of the major; the major must already be in the database.
        year: Recruitment year.
        points: Minimum number of points that got a candidate admitted.
        capacity: Number of places offered.
        source_url: Where the numbers come from, as ``"PDF: <path>"``.
    """
    # round is always 1: the thresholds PDF only lists the first recruitment round.
    cursor.execute(
        """
        INSERT INTO admission_thresholds
        (major_id, year, round, min_points, capacity, source_url)
        VALUES (%s, %s, 1, %s, %s, %s)
        ON CONFLICT (major_id, year, round)
        DO UPDATE SET
            min_points = EXCLUDED.min_points,
            capacity = EXCLUDED.capacity,
            last_updated = CURRENT_TIMESTAMP;
    """,
        (major_id, year, points, capacity, source_url),
    )


def get_admission_threshold(
    cursor: DBCursor,
    major_name: str,
    year: int,
    round: int | None = None,
    degree: str | None = None,
    study_mode: str | None = None,
) -> list[dict]:
    """Find a major's admission thresholds for one year.

    Args:
        cursor: Database cursor to run the SQL on.
        major_name: Polish name, or the English name of an English-taught major; letter case and Polish diacritics are
            ignored.
        year: Recruitment year.
        round: Only this recruitment round (1 or 2); None for all.
        degree: Only this degree; None for all.
        study_mode: Only this study mode; None for all.

    Returns:
        list[dict]: Matching thresholds, ordered by round; empty if none are
        stored. Each dict:
            ``major_name`` / ``degree`` / ``study_mode`` (str) — which major.
            ``major_english_name`` (str | None) — set for the English-taught
                version of the major, None for the Polish-taught one.
            ``year`` (int) — recruitment year.
            ``round`` (int) — recruitment round.
            ``min_points`` (float) — minimum points to be admitted.
            ``capacity`` (int | None) — number of places.
            ``source_url`` (str) — where the numbers come from.
            ``last_updated`` (str) — when the row was last written (ISO date-time).
    """
    # Optional filters: each value is passed twice — once for the IS NULL
    # check (None = no filter), once for the comparison.
    cursor.execute(
        """
        SELECT m.polish_name, m.degree, m.study_mode, m.english_name, t.year, t.round,
               t.min_points, t.capacity, t.source_url, t.last_updated
        FROM admission_thresholds t
        JOIN majors m ON m.id = t.major_id
        WHERE (LOWER(unaccent(m.polish_name)) = LOWER(unaccent(%s::text))
               OR LOWER(unaccent(m.english_name)) = LOWER(unaccent(%s::text)))
          AND t.year = %s
          AND (%s::int IS NULL OR t.round = %s::int)
          AND (%s::text IS NULL OR m.degree = %s::text)
          AND (%s::text IS NULL OR m.study_mode = %s::text)
        ORDER BY m.english_name NULLS FIRST, t.round;
    """,
        (major_name, major_name, year, round, round, degree, degree, study_mode, study_mode),
    )

    columns = (
        "major_name",
        "degree",
        "study_mode",
        "major_english_name",
        "year",
        "round",
        "min_points",
        "capacity",
        "source_url",
        "last_updated",
    )
    return [row_to_dict(columns, row) for row in cursor.fetchall()]


def delete_admission_thresholds(cursor: DBCursor) -> None:
    """Delete every threshold (used by ``clean_db.py``).

    Args:
        cursor: Database cursor to run the SQL on.
    """
    cursor.execute("""
        DELETE FROM admission_thresholds;
                   """)
