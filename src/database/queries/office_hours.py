"""SQL for the ``office_hours`` table: weekly office hours (konsultacje) of staff members."""

from __future__ import annotations

from psycopg import Cursor as DBCursor

from database.serialization import row_to_dict


def delete_office_hours_of_staff(cursor: DBCursor, staff_id: int) -> None:
    """Delete all of one person's office hours.

    Ingestion calls this before inserting the person's current hours, so the database always holds exactly what
    Informator shows now.

    Args:
        cursor: Database cursor to run the SQL on.
        staff_id: Id of the person.
    """
    cursor.execute(
        """
        DELETE FROM office_hours
        WHERE staff_id = %s;
    """,
        (staff_id,),
    )


def insert_office_hour(
    cursor: DBCursor,
    staff_id: int,
    day_of_week: str,
    start_time: str,
    end_time: str,
    location: str | None,
    comment: str | None,
    source_url: str,
) -> None:
    """Store one weekly office-hours slot of one person.

    Args:
        cursor: Database cursor to run the SQL on.
        staff_id: Id of the person.
        day_of_week: Polish weekday name in lower case, e.g. "poniedziałek".
        start_time: Start time as ``HH:MM``.
        end_time: End time as ``HH:MM``.
        location: Building and room, "online", or None if not given.
        comment: Note attached to this slot (e.g. "only by appointment"), or None.
        source_url: Link to the person's Informator card.
    """
    # Two slots on the same day starting at the same time: keep the first.
    cursor.execute(
        """
        INSERT INTO office_hours
        (staff_id, day_of_week, start_time, end_time, location, comment ,source_url)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (staff_id, day_of_week, start_time) DO NOTHING;
    """,
        (staff_id, day_of_week, start_time, end_time, location, comment, source_url),
    )


def get_office_hour(
    cursor: DBCursor,
    staff_id: int,
    day_of_week: str | None = None,
) -> list[dict]:
    """Find one person's weekly office hours, Monday first.

    Args:
        cursor: Database cursor to run the SQL on.
        staff_id: The person's id, from ``get_staff``.
        day_of_week: Only this Polish weekday (case and diacritics ignored); None for all days.

    Returns:
        list[dict]: The person's slots ordered by day and start time. Empty means the person has published no office
            hours — not that the person
        does not exist. Each dict:
            ``day_of_week`` (str) — Polish weekday.
        ``start_time`` / ``end_time`` (str) — e.g. "09: 45:00". ``location`` (str | None) — room or "online".
            ``comment`` (str | None) — note for this slot. ``source_url`` (str) — Informator card. ``last_updated``
            (str) — when the row was last written (ISO date-time).
    """
    # Weekday names would sort alphabetically, so the CASE orders them
    # Monday → Sunday instead.
    cursor.execute(
        """
        SELECT day_of_week, start_time, end_time, location, comment,
               source_url, last_updated
        FROM office_hours
        WHERE staff_id = %s
          AND (%s::text IS NULL OR LOWER(unaccent(day_of_week)) = LOWER(unaccent(%s::text)))
        ORDER BY
            CASE day_of_week
                WHEN 'poniedziałek' THEN 1
                WHEN 'wtorek' THEN 2
                WHEN 'środa' THEN 3
                WHEN 'czwartek' THEN 4
                WHEN 'piątek' THEN 5
                WHEN 'sobota' THEN 6
                WHEN 'niedziela' THEN 7
                ELSE 8
            END,
            start_time;
    """,
        (staff_id, day_of_week, day_of_week),
    )

    columns = ("day_of_week", "start_time", "end_time", "location", "comment", "source_url", "last_updated")
    return [row_to_dict(columns, row) for row in cursor.fetchall()]


def delete_office_hours(cursor: DBCursor) -> None:
    """Delete every office-hours slot (used by ``clean_db.py``).

    Args:
        cursor: Database cursor to run the SQL on.
    """
    cursor.execute("""
        DELETE FROM office_hours;
    """)
