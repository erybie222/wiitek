"""Load all WIiT data into the database: ``uv run src/run_ingestion.py``.

Steps, in order (later steps depend on earlier ones):

1. the WIiT faculty row;
2. every major, and the courses from its ECTS catalog page;
3. admission thresholds from the PDF — matched to the majors from step 2 by name;
4. faculty units, staff, their functions and office hours from Informator;
5. the sections of every course's card PDF — read from the links stored in
   step 2, downloaded once into ``data/course_cards/`` and parsed from there;
6. a vector for every section from step 5, for the course card search —
   computed only for new or changed text.

Everything runs in one transaction that is committed at the end, so a crash
leaves the database as it was. Two steps are exceptions, reported and skipped
instead of undoing the rest: step 3 when the PDF is missing or not laid out as
expected, and step 4, which needs a browser cookie that often expires (staff
saved before the failure are kept). Step 5 skips, one by one, the course cards it
cannot download or read, keeping their stored sections. Step 6 is skipped
without an OpenAI key and stops at an API error, keeping the vectors computed
so far; the next run computes the missing ones.

Do not commit partway through: step 2 deletes courses and step 4 staff and
units that this run did not write, and both recognise them by a
``last_updated`` older than the transaction's start. A commit in between would
make everything written before it look stale.
"""

from datetime import date

from database.connection import close_pool, get_db_connection
from database.queries.faculties import insert_wiit_faculty
from database.queries.majors import upsert_major
from embeddings import embed_course_card_sections
from scrapers.html_ects import ingest_ects_catalog
from scrapers.informator_staff_office_hours import ingest_informator
from scrapers.pdf_course_cards import ingest_course_cards
from scrapers.pdf_thresholds import (
    ThresholdsPdfError,
    find_thresholds_pdf,
    ingest_thresholds_pdf,
)

# Exactly one file must match. Its years are read from the table header, so
# replacing it with a newer PDF (keeping this name prefix) needs no code change.
THRESHOLDS_PDF_PATTERN = "data/progi punktowe *.pdf"
# The Polish academic year starts on 1 October.
ACADEMIC_YEAR_START_MONTH = 10


def academic_year_start(today: date) -> int:
    """Return the calendar year in which the academic year in progress began.

    Args:
        today: The date to check.

    Returns:
        int: E.g. 2026 for both 2026-10-24 and 2027-03-01 (academic year 2026/2027).
    """
    return today.year if today.month >= ACADEMIC_YEAR_START_MONTH else today.year - 1


ACADEMIC_YEAR = academic_year_start(date.today())
# ECTS catalogs are published per academic year, e.g. .../2026_2027; the one of
# the year in progress always exists, the next one may not be published yet.
BASE_ECTS_URL = f"https://put.poznan.pl/karty-ects/{ACADEMIC_YEAR}_{ACADEMIC_YEAR + 1}"
# Last part of every catalog URL: full-time ("stacjonarne") first-cycle studies.
STUDY_PATH = "stacjonarne/studia%20pierwszego%20stopnia"

# Polish-taught first-cycle majors, spelled exactly as in the catalog URLs and
# the thresholds PDF (the PDF match ignores letter case).
WIIT_MAJORS_PL_FIRST_CYCLE = [
    "Informatyka",
    "Bioinformatyka",
    "Teleinformatyka",
    "Cyberbezpieczeństwo",
    "Mikroelektronika i komunikacja cyfrowa",
]

# English-taught first-cycle majors as (Polish name, English name). Both are
# stored, so a question may name the major either way, and both build the
# catalog URL.
WIIT_MAJORS_EN_FIRST_CYCLE = [
    ("Sztuczna inteligencja", "Artificial Intelligence"),
]

# TODO: add second-cycle (master's) majors and part-time studies. Both need
# their own catalog URL path and the matching degree / study_mode values.
# WIIT_MAJORS_PL_SECOND_CYCLE = []
#
# WIIT_MAJORS_EN_SECOND_CYCLE = []


def main() -> None:
    """Run every ingestion step on one database cursor and commit once at the end.

    The steps and why they run in this order are described at the top of this module.
    """
    print("Starting PUT_Bot ingestion pipeline...")

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            print("Connected to the database. Initializing basic structures...")

            insert_wiit_faculty(cursor)

            for major in WIIT_MAJORS_PL_FIRST_CYCLE:
                major_id = upsert_major(
                    cursor,
                    major_name=major,
                    degree="bachelor-of-engineering",
                    study_mode="full-time",
                    profile="general-academic",
                    faculty="WIiT",
                )
                # The major name is a URL path segment, so spaces become %20.
                normalized_major_name = major.replace(" ", "%20")
                major_catalog_url = f"{BASE_ECTS_URL}/{normalized_major_name}/{STUDY_PATH}"

                ingest_ects_catalog(cursor, major_catalog_url, major_id=major_id)

            for major in WIIT_MAJORS_EN_FIRST_CYCLE:
                major_id = upsert_major(
                    cursor,
                    major_name=major[0],
                    degree="bachelor-of-engineering",
                    study_mode="full-time",
                    profile="general-academic",
                    faculty="WIiT",
                    english_name=major[1],
                )
                # English-taught majors' catalog path is "<Polish name>_<English name>".
                major_name = f"{major[0]}_{major[1]}"
                normalized_major_name = major_name.replace(" ", "%20")
                major_catalog_url = f"{BASE_ECTS_URL}/{normalized_major_name}/{STUDY_PATH}"
                ingest_ects_catalog(cursor, major_catalog_url, major_id=major_id)

            # Needs the majors above: rows are matched to them by name. A missing
            # or unexpected PDF skips this step; stored thresholds stay, as
            # past thresholds remain true.
            try:
                ingest_thresholds_pdf(cursor, find_thresholds_pdf(THRESHOLDS_PDF_PATTERN))
            except ThresholdsPdfError as error:
                print(f"Thresholds PDF skipped, stored thresholds kept: {error}")

            # Informator needs a manual browser cookie that often expires; a
            # failure here skips staff data instead of rolling back the run.
            try:
                ingest_informator(cursor)
            except RuntimeError as error:
                print(f"Informator staff scraping aborted: {error}")

            # Last, because the first run downloads every course card (over
            # an hour) and the Informator cookie could expire meanwhile. Needs
            # the courses from step 2: it reads their course card links.
            ingest_course_cards(cursor)

            # Needs the sections just stored. Skips itself without an API key.
            embed_course_card_sections(cursor)
    finally:
        close_pool()


if __name__ == "__main__":
    main()
