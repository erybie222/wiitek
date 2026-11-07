"""SQL for the ``staff`` table: faculty employees, from Informator."""

from __future__ import annotations

from psycopg import Cursor as DBCursor

from database.serialization import row_to_dict


def upsert_staff(
    cursor: DBCursor,
    first_name: str,
    last_name: str,
    academic_title: str | None,
    position: str | None,
    user_type: str | None,
    department: str | None,
    email: str,
    phone: str | None,
    source_url: str,
) -> int:
    """Insert a staff member, or update them if someone with this e-mail is already stored.

    The e-mail identifies a person, because names are not unique.

    Args:
        cursor: Database cursor to run the SQL on.
        first_name: First name.
        last_name: Last name.
        academic_title: Academic title, e.g. "dr hab. inż.", or None.
        position: Job title, e.g. "Profesor uczelni", or None.
        user_type: Kind of employee as Informator labels it, or None.
        department: Name of the unit the person works in (must be stored in ``departments``), or None.
        email: E-mail address; identifies the person.
        phone: Phone number(s), comma-separated, or None.
        source_url: Link to the person's Informator card.

    Returns:
        int: The person's id, needed to attach their office hours and functions.
    """
    # On conflict RETURNING gives the id of the existing row, so the id is
    # stable across re-runs for the same e-mail.
    cursor.execute(
        """
        INSERT INTO staff
        (first_name, last_name, academic_title, position, user_type, department, email, phone, source_url)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (email)
        DO UPDATE SET
            first_name = EXCLUDED.first_name,
            last_name = EXCLUDED.last_name,
            academic_title = EXCLUDED.academic_title,
            position = EXCLUDED.position,
            user_type = EXCLUDED.user_type,
            department = EXCLUDED.department,
            phone = EXCLUDED.phone,
            source_url = EXCLUDED.source_url,
            last_updated = CURRENT_TIMESTAMP
        RETURNING id;
    """,
        (first_name, last_name, academic_title, position, user_type, department, email, phone, source_url),
    )
    return cursor.fetchone()[0]


def get_staff(
    cursor: DBCursor,
    last_name: str,
    first_name: str | None = None,
    department: str | None = None,
) -> list[dict]:
    """Find staff members by last name, optionally narrowed by first name or unit.

    Args:
        cursor: Database cursor to run the SQL on.
        last_name: Last name; letter case and Polish diacritics are ignored.
        first_name: Only people with exactly this first name (case and diacritics ignored); None for any.
        department: Only people in exactly this unit (case and diacritics ignored); None for any.

    Returns:
        list[dict]: Every match. WIiT has different people with the same name, so several rows are possible; the caller
            must then ask which
        person is meant instead of picking one. Each dict:
            ``id`` (int) — pass this to the office-hours and functions lookups.
            ``first_name`` / ``last_name`` (str) — name.
            ``academic_title`` / ``position`` / ``user_type`` (str | None) —
                as described in ``upsert_staff``.
            ``department`` (str | None) — unit the person works in.
            ``email`` (str) — e-mail address.
            ``phone`` (str | None) — phone number(s).
            ``source_url`` (str) — Informator card.
    """
    # Optional filters: each value is passed twice — once for the IS NULL
    # check (None = no filter), once for the comparison.
    cursor.execute(
        """
        SELECT id, first_name, last_name, academic_title, position, user_type,
               department, email, phone, source_url
        FROM staff
        WHERE LOWER(unaccent(last_name)) = LOWER(unaccent(%s::text))
          AND (%s::text IS NULL OR LOWER(unaccent(first_name)) = LOWER(unaccent(%s::text)))
          AND (%s::text IS NULL OR LOWER(unaccent(department)) = LOWER(unaccent(%s::text)))
        ORDER BY first_name, last_name;
    """,
        (last_name, first_name, first_name, department, department),
    )

    columns = (
        "id",
        "first_name",
        "last_name",
        "academic_title",
        "position",
        "user_type",
        "department",
        "email",
        "phone",
        "source_url",
    )
    return [row_to_dict(columns, row) for row in cursor.fetchall()]


def count_staff_not_seen_in_run(cursor: DBCursor) -> tuple[int, int]:
    """Count the staff members the current ingestion run has not written.

    A row written in this run has ``last_updated`` equal to ``CURRENT_TIMESTAMP``, which Postgres fixes at the start of
    the
    transaction. So this only works while the whole run is one transaction.

    Args:
        cursor: Database cursor of the ingestion run.

    Returns:
        tuple[int, int]: Staff members not written in this run, and all
        staff members stored.
    """
    cursor.execute("""
        SELECT COUNT(*) FILTER (WHERE last_updated IS DISTINCT FROM CURRENT_TIMESTAMP),
               COUNT(*)
        FROM staff;
    """)
    not_seen, total = cursor.fetchone()
    return not_seen, total


def delete_staff_not_seen_in_run(cursor: DBCursor) -> int:
    """Delete the staff members the current ingestion run has not written.

    Their office hours and functions go with them (``ON DELETE CASCADE``). Call this only after the whole staff list was
    fetched: a run cut short would delete everyone it did not reach. Like ``count_staff_not_seen_in_run``, it relies on
    the whole run being one transaction.

    Args:
        cursor: Database cursor of the ingestion run.

    Returns:
        int: Number of staff members deleted.
    """
    cursor.execute("""
        DELETE FROM staff
        WHERE last_updated IS DISTINCT FROM CURRENT_TIMESTAMP;
    """)
    return cursor.rowcount


def delete_staff(cursor: DBCursor) -> None:
    """Delete every staff member (used by ``clean_db.py``).

    Args:
        cursor: Database cursor to run the SQL on.
    """
    cursor.execute("""
        DELETE FROM staff;
    """)
