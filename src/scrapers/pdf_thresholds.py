"""Scraper for the admission thresholds PDF published by the university.

The PDF has one table on its first page: one row per major, and for each of
the last five recruitment years two columns — number of places and minimum
points. Only the first recruitment round is listed.

Which column holds which year is read from the table's header (e.g.
"2025/2026 limit przyjęć", "2025/2026 min liczba pkt."), so a new PDF with a
shifted year range needs no code change. A header that does not look like
that stops the step instead of storing numbers under a guessed year.

Majors are matched to the database by name, so they must be ingested first.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import pdfplumber
from psycopg import Cursor as DBCursor

from database.queries.admission_thresholds import upsert_threshold
from database.queries.majors import get_major_id

# "2025/2026" in a header cell -> recruitment year 2025.
HEADER_YEAR = re.compile(r"(\d{4})/\d{4}")


class ThresholdsPdfError(Exception):
    """The thresholds PDF is missing, ambiguous or not laid out as expected."""


def find_thresholds_pdf(pattern: str) -> str:
    """Find the one thresholds PDF matching a glob pattern.

    Args:
        pattern: Glob pattern relative to the working directory, e.g. ``"data/progi punktowe *.pdf"``.

    Returns:
        str: Path of the file, with forward slashes (it is cited as the source).

    Raises:
        ThresholdsPdfError: No file or more than one file matches.
    """
    matches = sorted(Path(".").glob(pattern))
    if len(matches) != 1:
        found = [path.as_posix() for path in matches]
        raise ThresholdsPdfError(f"Expected exactly one file matching {pattern!r}, found {len(found)}: {found}")
    return matches[0].as_posix()


def read_year_columns(header: list[str | None]) -> dict[int, tuple[int, int]]:
    """Work out which columns hold each year's capacity and minimum points.

    Args:
        header: The table's first row; column 0 is the major name, every other cell names a year and either "limit
            przyjęć" (capacity) or "min liczba pkt." (minimum points).

    Returns:
        dict[int, tuple[int, int]]: Recruitment year -> (capacity column,
        min-points column), newest year first, e.g. ``{2025: (1, 2), ...}``.

    Raises:
        ThresholdsPdfError: A cell names no year or no column kind, or a year lacks one of its two columns.
    """
    capacity_columns: dict[int, int] = {}
    points_columns: dict[int, int] = {}
    for index, cell in enumerate(header[1:], start=1):
        text = " ".join((cell or "").split()).lower()
        year_match = HEADER_YEAR.search(text)
        if "limit" in text:
            columns = capacity_columns
        elif "pkt" in text:
            columns = points_columns
        else:
            columns = None
        if year_match is None or columns is None:
            raise ThresholdsPdfError(f"Unexpected header cell in column {index}: {cell!r}")
        year = int(year_match.group(1))
        if year in columns:
            raise ThresholdsPdfError(f"Year {year} appears twice in the header: {header}")
        columns[year] = index

    if not capacity_columns or capacity_columns.keys() != points_columns.keys():
        raise ThresholdsPdfError(f"Header does not pair capacity and points per year: {header}")
    return {year: (capacity_columns[year], points_columns[year]) for year in sorted(capacity_columns, reverse=True)}


def parse_polish_number(value: str | None, output_type: Callable[[str], int | float]) -> int | float | None:
    """Convert a number written the Polish way (comma as decimal point) to a Python number.

    Args:
        value: Cell text, e.g. ``"612,50"``.
        output_type: ``int`` or ``float`` — what to convert to.

    Returns:
        int | float | None: The number, or None if the cell is empty or "-".
    """
    if not value or value.strip() in ("", "-"):
        return None

    clean_value = value.replace(",", ".").strip()
    return output_type(clean_value)


def ingest_thresholds_pdf(cursor: DBCursor, file_path: str) -> None:
    """Read the thresholds table from the PDF and store every major's thresholds.

    A major in the PDF that is not in the database is reported and skipped.

    Args:
        cursor: Database cursor to run the SQL on.
        file_path: Path to the thresholds PDF.

    Raises:
        ThresholdsPdfError: The PDF has no table, its header is not laid out as expected, or a number cell cannot be
            read.
    """
    print("Starting PDF extraction process...")

    print(f"Opening file: {file_path}")
    with pdfplumber.open(file_path) as pdf:
        table = pdf.pages[0].extract_table()
    if not table:
        raise ThresholdsPdfError(f"No table found on the first page of {file_path}")

    year_columns = read_year_columns(table[0])
    print(f"Years in the PDF: {', '.join(str(year) for year in year_columns)}")

    for row in table[1:]:
        # An English-taught major is listed as "polska nazwa/English Name",
        # often next to a Polish-taught row with the same Polish name; the two
        # are different majors, told apart by the English name.
        polish_part, _, english_part = row[0].partition("/")
        clean_name = polish_part.strip()
        english_name = english_part.strip() or None

        label = f"{clean_name.capitalize()}{f' / {english_name}' if english_name else ''}"
        print(f"\nProcessing: {label}")
        major_id = get_major_id(cursor, clean_name, "bachelor-of-engineering", "full-time", english_name)

        if not major_id:
            print(f"  -> Error: Could not resolve ID for {label}")
            continue

        for year, (idx_capacity, idx_points) in year_columns.items():
            try:
                capacity = parse_polish_number(row[idx_capacity], int)
                points = parse_polish_number(row[idx_points], float)
            except ValueError as error:
                raise ThresholdsPdfError(f"Unreadable number for {clean_name!r}, year {year}: {error}") from error

            # Empty or "-" cell: no threshold published for that year.
            if capacity is None or points is None:
                continue

            upsert_threshold(
                cursor=cursor,
                major_id=major_id,
                year=year,
                points=points,
                capacity=capacity,
                source_url=f"PDF: {file_path}",
            )

            print(f"  -> Year {year}: Min Points = {points}, Capacity = {capacity}")

    print("\nThresholds PDF processed.")
