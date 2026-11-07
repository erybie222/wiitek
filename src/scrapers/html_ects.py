"""Scraper for the ECTS course catalog on put.poznan.pl.

Each major has one catalog page per academic year listing every course by
semester. The page structure the scraper relies on:

- ``<h2>`` — a semester heading, e.g. "Semestr 1";
- ``<h3>`` — a course group under it, e.g. "Przedmioty obligatoryjne";
- the ``<table>`` right after each ``<h3>`` — one course per row, with the
  columns name+code, ECTS, hours (W/C/L/P) and links to the course card.

If the university changes this layout, the scraper silently finds fewer
courses, so check the printed course list after a run.
"""

from __future__ import annotations

import re

import requests
from bs4 import BeautifulSoup
from psycopg import Cursor as DBCursor

from database.queries.courses import delete_courses_of_major, upsert_course


def parse_hours(hours_str: str) -> tuple[int, bool]:
    """Read the number of hours, and the exam marker, from one hours cell.

    Args:
        hours_str: Cell text, e.g. ``"30"`` or ``"30 (E)"``. "(E)" in the lecture column means the course ends with an
            exam.

    Returns:
        tuple[int, bool]: The hours (0 for an empty cell) and whether the
        cell contains "(E)".
    """
    if not hours_str or hours_str.strip() == "":
        return 0, False

    # Check for the Exam flag (case-insensitive)
    has_exam = "(e)" in hours_str.lower()

    # Extract only the digits using regex
    digits = re.findall(r"\d+", hours_str)
    hours_int = int(digits[0]) if digits else 0

    return hours_int, has_exam


def ingest_ects_catalog(cursor: DBCursor, url: str, major_id: int) -> None:
    """Make the database hold exactly the courses on one major's catalog page.

    The major's stored courses are deleted and the ones on the page stored, so courses dropped from the catalog
    disappear here too. If the page cannot be downloaded, or yields no courses (more likely a layout change than an
    empty catalog), the stored courses are kept, a warning is printed and the rest of the ingestion continues.

    Args:
        cursor: Database cursor to run the SQL on.
        url: Address of the major's catalog page.
        major_id: Id of the major the courses belong to.
    """
    print(f"Fetching full catalog from: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the catalog, keeping stored courses: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")

    # Parsed first and stored only at the end, so a page that yields nothing
    # never wipes the major's courses.
    courses: list[dict] = []

    # Find all h3 headers (Obligatory / Elective subjects)
    sections = soup.find_all("h3")

    for section in sections:
        section_title = section.get_text(strip=True).lower()

        # Ignore h3 tags that do not relate to subjects
        if "przedmioty" not in section_title:
            continue

        # Check if course is obligatory or elective based on the section title
        is_obligatory = "obligatoryjne" in section_title

        # The semester is the nearest <h2> above this section, e.g. "Semestr 1"
        parent_h2 = section.find_previous("h2")
        if not parent_h2:
            continue

        semester_text = parent_h2.get_text(strip=True)  # e.g., "Semestr 1"
        semester_match = re.search(r"\d+", semester_text)

        # Skip the course insertion if semester was not found

        if semester_match:
            semester = int(semester_match.group())
        else:
            continue

        # The table containing courses immediately follows the H3 tag
        table = section.find_next_sibling("table")
        if not table:
            continue

        print(f"\n--- SEMESTER {semester} | {'Obligatory' if is_obligatory else 'Elective'} ---")

        # Find the tbody element and all rows within it (courses are set in table)
        tbody = table.find("tbody")
        rows = tbody.find_all("tr") if tbody else table.find_all("tr")

        for row in rows:
            # Check if the row is an elective group row and adjust the obligatory flag accordingly.
            # The flag is not reset: every later row in this section is treated as elective too.
            if "elective-group-row" in row.get("class", []):
                is_obligatory = False

            # Skip rows that don't have enough cells to extract data
            cells = row.find_all("td")
            if len(cells) < 8:
                continue

            # 1. Name and code, found by their CSS classes rather than position
            name_span = cells[0].find("span", class_="course-name")
            code_span = cells[0].find("span", class_="course-code")

            if not name_span or not code_span:
                continue

            subject_name = name_span.get_text(strip=True)
            subject_code = code_span.get_text(strip=True)

            # If course code endswith ang then change boolean flag to true
            is_english = subject_code.lower().endswith("ang")

            # 2. Extract ECTS points
            ects_text = cells[1].get_text(strip=True)
            ects = int(ects_text) if ects_text.isdigit() else 0

            # 3. Extract Hours and Exam info
            # Columns W/C/L/P; only the lecture (W) cell carries the (E) exam marker
            lecture_hours, has_exam = parse_hours(cells[2].get_text(strip=True))
            classes_hours, _ = parse_hours(cells[3].get_text(strip=True))
            lab_hours, _ = parse_hours(cells[4].get_text(strip=True))
            project_hours, _ = parse_hours(cells[5].get_text(strip=True))

            assessment_type = "Egzamin" if has_exam else "Zaliczenie z ocena"

            # 4. Extract the PDF Link in Polish language (from the 'PL' column - index 7)
            pdf_link = None
            pl_cell = cells[7]
            anchor = pl_cell.find("a", class_="course-link")
            if anchor and "href" in anchor.attrs:
                pdf_link = anchor["href"]
                # Prefix with the main domain if the link is relative
                if pdf_link.startswith("/"):
                    pdf_link = f"https://put.poznan.pl{pdf_link}"

            print(f"Parsed -> [{subject_code}] {subject_name} (ECTS: {ects})")

            courses.append(
                {
                    "code": subject_code,
                    "semester": semester,
                    "name": subject_name,
                    "is_obligatory": is_obligatory,
                    "is_english": is_english,
                    "ects": ects,
                    "hours_lecture": lecture_hours,
                    "hours_classes": classes_hours,
                    "hours_lab": lab_hours,
                    "hours_project": project_hours,
                    "assessment_type": assessment_type,
                    "source_url": pdf_link,
                }
            )

    if not courses:
        print("\nWARNING: no courses found on the page (layout change?), keeping stored courses.")
        return

    delete_courses_of_major(cursor, major_id)
    for course in courses:
        upsert_course(cursor=cursor, major_id=major_id, **course)

    print(f"\nCatalog processing complete: {len(courses)} course rows parsed.")
