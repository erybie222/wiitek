"""SQL for the ``majors`` table: fields of study (kierunki) offered by the faculty."""

from __future__ import annotations

from psycopg import Cursor as DBCursor

from database.serialization import row_to_dict


def upsert_major(
    cursor: DBCursor,
    major_name: str,
    degree: str,
    study_mode: str,
    profile: str,
    faculty: str,
    english_name: str | None = None,
) -> int:
    """Insert a major, or update it if it already exists, and return its id.

    A major is identified by Polish name + degree + study mode + English name, so "Informatyka" full-time and part-time
    are two separate rows, and so are a Polish- and an English-taught major sharing one Polish name.

    Args:
        cursor: Database cursor to run the SQL on.
        major_name: Polish name with a capital first letter, e.g. "Informatyka". Case matters for identity, so always
            spell a major the same way.
        degree: "bachelor-of-engineering" or "masters".
        study_mode: "full-time" or "part-time".
        profile: "general-academic" or "practical".
        faculty: Faculty symbol, e.g. "WIiT".
        english_name: Official English name of an English-taught major, e.g. "Artificial Intelligence"; None for a
            Polish-taught one. Setting it is what marks the major as taught in English.

    Returns:
        int: The major's id, needed to attach its courses.
    """
    cursor.execute(
        """
        INSERT INTO majors (polish_name, degree, study_mode, profile, faculty, english_name)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (polish_name, degree, study_mode, english_name)
        DO UPDATE SET
            profile = EXCLUDED.profile,
            faculty = EXCLUDED.faculty
        RETURNING id;
    """,
        (major_name, degree, study_mode, profile, faculty, english_name),
    )
    return cursor.fetchone()[0]


def get_major_id(
    cursor: DBCursor,
    major_name: str,
    degree: str,
    study_mode: str,
    english_name: str | None,
) -> int | None:
    """Find one major's id; used during ingestion to link thresholds to majors.

    Not used by the chatbot — the agent uses ``get_major``, which returns every matching row instead of a single id.

    Args:
        cursor: Database cursor to run the SQL on.
        major_name: Polish name; letter case and Polish diacritics are ignored.
        degree: Degree, spelled exactly as stored.
        study_mode: Study mode, spelled exactly as stored.
        english_name: English name of the English-taught major, compared the same way as the Polish one; None for the
            Polish-taught major.

    Returns:
        int | None: The id, or None if there is no such major.
    """
    # IS NOT DISTINCT FROM: NULL matches NULL, so None finds the Polish-taught row.
    cursor.execute(
        """
        SELECT id FROM majors
        WHERE LOWER(unaccent(polish_name)) = LOWER(unaccent(%s::text))
          AND LOWER(unaccent(english_name)) IS NOT DISTINCT FROM LOWER(unaccent(%s::text))
          AND degree = %s AND study_mode = %s;
    """,
        (major_name, english_name, degree, study_mode),
    )

    result = cursor.fetchone()
    return result[0] if result else None


def get_major(
    cursor: DBCursor,
    major_name: str | None = None,
    degree: str | None = None,
    study_mode: str | None = None,
) -> list[dict]:
    """Find majors by name, or list every major when no name is given.

    The full list is what lets the bot say which majors exist — and refuse to answer about one that is not in the
    database.

    Args:
        cursor: Database cursor to run the SQL on.
        major_name: Polish name, or the English name of an English-taught major; letter case and Polish diacritics
            ignored. None lists all majors.
        degree: Only this degree; None for all.
        study_mode: Only this study mode; None for all.

    Returns:
        list[dict]: Every match, ordered by name. One name can match several rows (e.g. full-time and part-time, or a
            Polish- and an English-taught version); the caller must then ask which one is meant instead of
        picking one. Each dict:
            ``id`` (int) — database id.
            ``polish_name`` (str) — Polish major name.
            ``degree`` (str) — "bachelor-of-engineering" or "masters".
            ``study_mode`` (str) — "full-time" or "part-time".
            ``profile`` (str) — "general-academic" or "practical".
            ``faculty`` (str) — faculty symbol, e.g. "WIiT".
            ``english_name`` (str | None) — English name of an English-taught
                major; None means the major is taught in Polish.
    """
    # Optional filters: each value is passed twice — once for the IS NULL
    # check (None = no filter), once for the comparison.
    cursor.execute(
        """
        SELECT id, polish_name, degree, study_mode, profile, faculty, english_name
        FROM majors
        WHERE (%s::text IS NULL
               OR LOWER(unaccent(polish_name)) = LOWER(unaccent(%s::text))
               OR LOWER(unaccent(english_name)) = LOWER(unaccent(%s::text)))
          AND (%s::text IS NULL OR degree = %s::text)
          AND (%s::text IS NULL OR study_mode = %s::text)
        ORDER BY polish_name, degree, study_mode, english_name NULLS FIRST;
    """,
        (major_name, major_name, major_name, degree, degree, study_mode, study_mode),
    )

    columns = ("id", "polish_name", "degree", "study_mode", "profile", "faculty", "english_name")
    return [row_to_dict(columns, row) for row in cursor.fetchall()]


def delete_majors(cursor: DBCursor) -> None:
    """Delete every major (used by ``clean_db.py``).

    Args:
        cursor: Database cursor to run the SQL on.
    """
    cursor.execute("""
        DELETE FROM majors;
                   """)
