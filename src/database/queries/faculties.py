"""SQL for the ``faculties`` table: the faculties the system has data for (today only WIiT)."""

from __future__ import annotations

from psycopg import Cursor as DBCursor

from database.serialization import row_to_dict


def insert_wiit_faculty(cursor: DBCursor) -> None:
    """Insert the WIiT faculty row; does nothing if it already exists.

    Majors and units refer to the faculty by its symbol, so this must run before anything else is ingested.

    Args:
        cursor: Database cursor to run the SQL on.
    """
    cursor.execute("""
        INSERT INTO faculties (symbol, name)
        VALUES ('WIiT', 'Wydział Informatyki i Telekomunikacji')
        ON CONFLICT (symbol) DO NOTHING;
    """)


def get_faculties(cursor: DBCursor) -> list[dict]:
    """List every faculty in the database, ordered by name.

    Args:
        cursor: Database cursor to run the SQL on.

    Returns:
        list[dict]: One dict per faculty: ``symbol`` (str) — short code, e.g. "WIiT". ``name`` (str) — full Polish name.
    """
    cursor.execute("""
        SELECT symbol, name FROM faculties ORDER BY name;
    """)

    columns = ("symbol", "name")
    return [row_to_dict(columns, row) for row in cursor.fetchall()]
