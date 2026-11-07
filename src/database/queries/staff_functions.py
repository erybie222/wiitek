"""SQL for the ``staff_functions`` table: administrative roles held by staff.

A function is a role such as "Dyrektor instytutu" or "Prodziekan". It is
separate from a person's job title (``staff.position``) and from the unit
they work in (``staff.department``).
"""

from __future__ import annotations

from psycopg import Cursor as DBCursor

from database.serialization import row_to_dict


def delete_staff_functions_of_staff(cursor: DBCursor, staff_id: int) -> None:
    """Delete all functions of one person.

    Ingestion calls this before inserting the person's current functions, so the database always holds exactly what
    Informator shows now.

    Args:
        cursor: Database cursor to run the SQL on.
        staff_id: Id of the person.
    """
    cursor.execute(
        """
        DELETE FROM staff_functions
        WHERE staff_id = %s;
    """,
        (staff_id,),
    )


def insert_staff_function(
    cursor: DBCursor,
    staff_id: int,
    function_name: str,
    department_name: str | None,
    source_url: str,
) -> None:
    """Store one function held by one person; does nothing if it is already stored.

    Args:
        cursor: Database cursor to run the SQL on.
        staff_id: Id of the person.
        function_name: Name of the role, e.g. "Dyrektor instytutu".
        department_name: Unit the role is in, as plain text, or None. Not linked to ``departments`` because it may be
            outside WIiT.
        source_url: Link to the person's Informator card.
    """
    # DO NOTHING skips a function listed twice on the same card, also one
    # with no department_name (the constraint is NULLS NOT DISTINCT).
    cursor.execute(
        """
        INSERT INTO staff_functions
        (staff_id, function_name, department_name, source_url)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (staff_id, function_name, department_name) DO NOTHING;
    """,
        (staff_id, function_name, department_name, source_url),
    )


def get_staff_function(
    cursor: DBCursor,
    staff_id: int | None = None,
    function_name: str | None = None,
    department_name: str | None = None,
) -> list[dict]:
    """Find administrative functions together with the people who hold them.

    Args:
        cursor: Database cursor to run the SQL on.
        staff_id: Only this person's functions; None for everyone.
        function_name: Only this exact function (case and diacritics ignored); None for all.
        department_name: Only functions in this exact unit (case and diacritics ignored); None for all.

    Returns:
        list[dict]: Matching functions ordered by the holder's name; empty if
        none. Each dict:
            ``function_name`` (str) — the role.
            ``department_name`` (str | None) — unit the role is in.
            ``source_url`` (str) — Informator card.
            ``last_updated`` (str) — when the row was last written (ISO date-time).
            ``staff_id`` (int) — the holder's id.
            ``first_name`` / ``last_name`` / ``academic_title`` /
                ``position`` / ``email`` — the holder, as in ``get_staff``.
    """
    # Optional filters: each value is passed twice — once for the IS NULL
    # check (None = no filter), once for the comparison.
    cursor.execute(
        """
        SELECT sf.function_name, sf.department_name, sf.source_url, sf.last_updated,
               s.id, s.first_name, s.last_name, s.academic_title, s.position, s.email
        FROM staff_functions sf
        JOIN staff s ON s.id = sf.staff_id
        WHERE (%s::int IS NULL OR sf.staff_id = %s::int)
          AND (%s::text IS NULL OR LOWER(unaccent(sf.function_name)) = LOWER(unaccent(%s::text)))
          AND (%s::text IS NULL OR LOWER(unaccent(sf.department_name)) = LOWER(unaccent(%s::text)))
        ORDER BY s.last_name, s.first_name;
    """,
        (
            staff_id,
            staff_id,
            function_name,
            function_name,
            department_name,
            department_name,
        ),
    )

    columns = (
        "function_name",
        "department_name",
        "source_url",
        "last_updated",
        "staff_id",
        "first_name",
        "last_name",
        "academic_title",
        "position",
        "email",
    )
    return [row_to_dict(columns, row) for row in cursor.fetchall()]


def delete_staff_functions(cursor: DBCursor) -> None:
    """Delete every function (used by ``clean_db.py``).

    Args:
        cursor: Database cursor to run the SQL on.
    """
    cursor.execute("""
        DELETE FROM staff_functions;
    """)
