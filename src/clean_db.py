"""Delete all loaded data so ingestion can start from an empty database.

Run with ``uv run src/clean_db.py``. Tables are kept, only their rows are
deleted — faster than wiping the Docker volume with ``down -v``.
"""

from database.connection import close_pool, get_db_connection
from database.queries.admission_thresholds import delete_admission_thresholds
from database.queries.courses import delete_courses
from database.queries.departments import delete_departments
from database.queries.majors import delete_majors
from database.queries.office_hours import delete_office_hours
from database.queries.staff import delete_staff
from database.queries.staff_functions import delete_staff_functions


def clean_db() -> None:
    """Delete every row from every table in one transaction.

    If anything fails, the error is printed and nothing is deleted.
    """
    print("Cleaning database...")
    try:
        with get_db_connection() as connection:
            print("Connected to database.")
            cursor = connection.cursor()
            # Children before parents, so no row is deleted while a
            # foreign key still points at it.
            delete_office_hours(cursor)
            delete_staff_functions(cursor)
            delete_staff(cursor)
            delete_departments(cursor)
            delete_courses(cursor)
            delete_admission_thresholds(cursor)
            delete_majors(cursor)
            print("Done.")
    except Exception as e:
        print(e)
        print("Error while cleaning database.")
    finally:
        close_pool()


if __name__ == "__main__":
    clean_db()
