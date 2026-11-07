"""SQL for the ``departments`` table: units inside the faculty (institutes, zakłady, dean's office).

The faculty itself is not a row here — it is in ``faculties``. Units form a
tree through ``parent``; units directly under the faculty have no parent.
"""

from __future__ import annotations

from psycopg import Cursor as DBCursor

from database.serialization import row_to_dict


def upsert_department(
    cursor: DBCursor,
    name: str,
    symbol: str | None,
    parent: str | None,
    unit_type: str | None,
    faculty: str,
    address: str | None,
    phone: str | None,
    email: str | None,
    source_url: str | None,
) -> None:
    """Insert a unit, or update it if a unit with this name is already stored.

    Units are identified by their Polish name. If the university renames a unit, the next ingestion adds it as a new row
    and the old one remains.

    Args:
        cursor: Database cursor to run the SQL on.
        name: Polish unit name; identifies the unit.
        symbol: Short symbol from Informator. Not unique: a zakład often has the same symbol as its institute.
        parent: Name of the unit this one belongs to; None if it sits directly under the faculty. The parent must
            already be stored.
        unit_type: Kind of unit, e.g. "instytut", "zakład".
        faculty: Faculty symbol, e.g. "WIiT".
        address: Postal address, or None.
        phone: Phone number(s), or None.
        email: Office e-mail, or None.
        source_url: Link to the unit's page in Informator.
    """
    cursor.execute(
        """
        INSERT INTO departments (name, symbol, parent, unit_type, faculty, address, phone, email, source_url)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (name)
        DO UPDATE SET
            symbol = EXCLUDED.symbol,
            parent = EXCLUDED.parent,
            unit_type = EXCLUDED.unit_type,
            faculty = EXCLUDED.faculty,
            address = EXCLUDED.address,
            phone = EXCLUDED.phone,
            email = EXCLUDED.email,
            source_url = EXCLUDED.source_url,
            last_updated = CURRENT_TIMESTAMP;
    """,
        (name, symbol, parent, unit_type, faculty, address, phone, email, source_url),
    )


def get_department(cursor: DBCursor, name: str) -> dict | None:
    """Find one unit by name, together with the names of its direct sub-units.

    Args:
        cursor: Database cursor to run the SQL on.
        name: Unit name; letter case and Polish diacritics are ignored.

    Returns:
        dict | None: None if no unit has this name. Otherwise:
            ``name`` / ``symbol`` / ``parent`` / ``unit_type`` / ``faculty`` —
                as described in ``upsert_department``.
            ``address`` / ``phone`` / ``email`` (str | None) — contact details.
            ``source_url`` (str | None) — Informator page.
            ``last_updated`` (str) — when the row was last written (ISO date-time).
            ``children`` (list[str]) — names of the units directly below it.
        Note the difference: None means the unit does not exist, while
        ``children == []`` means it exists but has no sub-units.
    """
    cursor.execute(
        """
        SELECT name, symbol, parent, unit_type, faculty, address, phone, email,
               source_url, last_updated
        FROM departments
        WHERE LOWER(unaccent(name)) = LOWER(unaccent(%s::text));
    """,
        (name,),
    )

    row = cursor.fetchone()
    if row is None:
        return None

    columns = (
        "name",
        "symbol",
        "parent",
        "unit_type",
        "faculty",
        "address",
        "phone",
        "email",
        "source_url",
        "last_updated",
    )
    department = row_to_dict(columns, row)

    # Direct sub-units only (one level down), not the whole subtree.
    cursor.execute(
        """
        SELECT name FROM departments WHERE LOWER(parent) = LOWER(%s) ORDER BY name;
    """,
        (department["name"],),
    )
    department["children"] = [child[0] for child in cursor.fetchall()]

    return department


def delete_departments_not_seen_in_run(cursor: DBCursor) -> int:
    """Delete the units the current ingestion run has not written, e.g. renamed or closed ones.

    A row written in this run has ``last_updated`` equal to ``CURRENT_TIMESTAMP``, which Postgres fixes at the start of
    the transaction, so this only works while the whole run is one transaction. Call it after
    ``delete_staff_not_seen_in_run``: staff who left may still point at a removed unit (``staff.department`` is a
    foreign key).

    Args:
        cursor: Database cursor of the ingestion run.

    Returns:
        int: Number of units deleted.
    """
    cursor.execute("""
        DELETE FROM departments
        WHERE last_updated IS DISTINCT FROM CURRENT_TIMESTAMP;
    """)
    return cursor.rowcount


def delete_departments(cursor: DBCursor) -> None:
    """Delete every unit (used by ``clean_db.py``).

    Args:
        cursor: Database cursor to run the SQL on.
    """
    cursor.execute("""
        DELETE FROM departments;
    """)
