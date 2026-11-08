"""Scraper for course cards: the PDF each course row links to (``courses.source_url``).

A course card ("karta opisu przedmiotu") is generated from one template, so the parser relies on its layout:

- a section heading is set in the ArialNarrow font, body text in ArialMT. Headings are recognised by that font, never
  by their words: one syllabus pastes the words "Cel przedmiotu" into its body text;
- grey body text is template wording, the same in every file (e.g. "Efekty uczenia się przedstawione wyżej
  weryfikowane są w następujący sposób:"), and is left out;
- the page number sits alone in the bottom margin, which is cut off by position. A rule like "drop a line holding only
  a number" would also drop a book's year, which the literature list sometimes puts on its own line;
- the first page prints "Koordynatorzy" and "Wykładowcy" as two columns side by side. Plain text extraction merges
  them into one line ("dr X dr Y"), so they are split by the x position of the "Wykładowcy" heading. A list too long
  for the page continues on the next one under a repeated lone "Wykładowcy" heading, which belongs to the same section;
- "Bilans nakładu pracy przeciętnego studenta" is the last section: a table of numbers that text extraction scrambles,
  so reading stops there. ECTS and hours come from the catalog (``get_course``) instead.

PDFs are cached under ``data/course_cards/``, mirroring the link's path after ``/cards/`` (which starts with the academic
year). A cached file is used without asking the server: each academic year has new links, so a new year downloads
itself. The first run downloads every syllabus, with pauses (the server answers 503 every few files), which takes over
an hour; later runs only parse.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
import requests
from psycopg import Cursor as DBCursor

from database.queries.course_card_sections import (
    delete_sections_not_seen_in_run,
    upsert_course_card_section,
)
from database.queries.courses import get_courses_with_card_links

COURSE_CARD_CACHE_DIR = Path("data/course_cards")
# Every syllabus link starts with this; the rest of the path becomes the cache path.
COURSE_CARD_URL_PREFIX = "https://put.poznan.pl/cards/"
# Characters Windows does not allow in a file name; replaced by "_" in cache paths.
WINDOWS_UNSAFE_CHARACTERS = re.compile(r'[<>:"\\|?*]')

DOWNLOAD_PAUSE_SECONDS = 1.0
DOWNLOAD_ATTEMPTS = 5
# Doubled after every failed attempt: 5, 10, 20, 40 s.
FIRST_RETRY_WAIT_SECONDS = 5.0
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
REQUEST_TIMEOUT_SECONDS = 30

HEADING_FONT = "ArialNarrow"
# Height of the bottom band cut off every page: it holds only the page number (top ≈ 804 pt of 842,
# while body text ends by ≈ 790 pt).
FOOTER_HEIGHT = 45
# Words whose tops differ by at most this many points are on one line (11.2 and 11.3 pt text mix in a line).
LINE_TOLERANCE = 3
# A word that starts this close to the left of the "Wykładowcy" heading still belongs to its column.
COLUMN_TOLERANCE = 5

# Sections stored under their heading, as printed in the PDF.
SECTIONS_TO_STORE = (
    "Wymagania wstępne",
    "Cel przedmiotu",
    "Przedmiotowe efekty uczenia się",
    "Metody weryfikacji efektów uczenia się i kryteria oceny",
    "Treści programowe",
    "Tematyka zajęć",
    "Metody dydaktyczne",
    "Literatura",
)
# The first page's two-column heading, and the one section both columns are stored as.
STAFF_HEADING = "Koordynatorzy Wykładowcy"
LECTURERS_HEADING_WORD = "Wykładowcy"
STAFF_SECTION = "Koordynatorzy i wykładowcy"
# First-page headings of what the catalog already holds (name, major, hours, ECTS).
SECTIONS_TO_SKIP = ("KARTA OPISU PRZEDMIOTU - SYLABUS", "Przedmiot", "Liczba godzin", "Liczba punktów ECTS")
LAST_SECTION_HEADING = "Bilans nakładu pracy przeciętnego studenta"
# A section holding only one of these (case ignored) says nothing and is not stored.
EMPTY_SECTION_TEXTS = {"", "brak", "brak.", "-", "–", "—"}


class CourseCardUnavailableError(Exception):
    """A course's syllabus cannot be stored in this run; the message says why. Its stored sections are kept."""


@dataclass
class TextLine:
    """One line of a PDF page: its words, left to right, as ``pdfplumber.extract_words`` returns them."""

    words: list[dict]

    @property
    def text(self) -> str:
        """Return the line's words joined by single spaces."""
        return " ".join(word["text"] for word in self.words)

    @property
    def is_heading(self) -> bool:
        """Return True when every word is in the heading font."""
        return all(HEADING_FONT in word["fontname"] for word in self.words)

    @property
    def is_template_text(self) -> bool:
        """Return True when every word is grey, the colour of the template's own wording."""
        return all(is_grey(word["non_stroking_color"]) for word in self.words)


@dataclass
class ParsedCourseCard:
    """What ``parse_course_card`` read from one PDF.

    Attributes:
        sections: Section name → text, in the order of the PDF; only sections worth storing.
        empty_sections: Sections left out because they were empty or said only "brak".
        repeated_sections: Sections left out because an earlier section had the same text.
        unknown_headings: Headings the parser does not know; their text was not stored.
        reached_last_section: False when the "Bilans" heading never came: the text of the last stored section may then
            include the workload table.
    """

    sections: dict[str, str] = field(default_factory=dict)
    empty_sections: list[str] = field(default_factory=list)
    repeated_sections: list[str] = field(default_factory=list)
    unknown_headings: list[str] = field(default_factory=list)
    reached_last_section: bool = False


def is_grey(color: tuple | list | None) -> bool:
    """Say whether a fill colour is a mid grey.

    Args:
        color: ``non_stroking_color`` of a word: one value (grey scale) or three (RGB), each 0–1; e.g. ``(0,)`` is
            black, ``(0.525, 0.529, 0.529)`` the template's grey, ``(0.0, 0.384, 0.533)`` the heading blue.

    Returns:
        bool: True when all components are nearly equal and between 0.2 and 0.8.
    """
    if not isinstance(color, (tuple, list)) or len(color) not in (1, 3):
        return False
    return max(color) - min(color) < 0.05 and 0.2 < color[0] < 0.8


def cached_pdf_path(url: str, cache_dir: Path = COURSE_CARD_CACHE_DIR) -> Path | None:
    """Return where the cache keeps a syllabus PDF.

    Args:
        url: Syllabus link, e.g. "https://put.poznan.pl/cards/2026_2027/Informatyka/.../Algebra liniowa (ALIN).pdf".
        cache_dir: Root folder of the cache.

    Returns:
        Path | None: The link's path after ``/cards/`` under ``cache_dir``, e.g.
        ``data/course_cards/2026_2027/Informatyka/.../Algebra liniowa (ALIN).pdf``, with characters Windows forbids replaced
        by "_". None for a link outside ``/cards/`` or with an empty, "." or ".." path part (it could point outside
        the cache).
    """
    if not url.startswith(COURSE_CARD_URL_PREFIX):
        return None
    parts = url[len(COURSE_CARD_URL_PREFIX) :].split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None
    return cache_dir.joinpath(*(WINDOWS_UNSAFE_CHARACTERS.sub("_", part) for part in parts))


def download_course_card(session: requests.Session, url: str, path: Path) -> None:
    """Download one syllabus PDF into the cache.

    Retries a timeout, a dropped connection or a status in ``RETRY_STATUS_CODES`` (the server answers 503 every few
    files), waiting 5, 10, 20, 40 s. The file is written under a temporary name and renamed at the end, so an
    interrupted download never leaves a truncated PDF in the cache.

    Args:
        session: HTTP session shared by all downloads of the run.
        url: Syllabus link.
        path: Where to save it (from ``cached_pdf_path``).

    Raises:
        CourseCardUnavailableError: Still failing after ``DOWNLOAD_ATTEMPTS`` attempts, another error status (e.g. 404),
            or a response that is not a PDF.
    """
    wait = FIRST_RETRY_WAIT_SECONDS
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        except (requests.ConnectionError, requests.Timeout) as error:
            problem = type(error).__name__
        else:
            if response.status_code not in RETRY_STATUS_CODES:
                break
            problem = f"HTTP {response.status_code}"
        if attempt == DOWNLOAD_ATTEMPTS:
            raise CourseCardUnavailableError(f"download failed {DOWNLOAD_ATTEMPTS} times, last: {problem}")
        time.sleep(wait)
        wait *= 2

    if response.status_code != 200:
        raise CourseCardUnavailableError(f"download failed: HTTP {response.status_code}")
    if not response.content.startswith(b"%PDF"):
        raise CourseCardUnavailableError("the server sent something other than a PDF")

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(response.content)
    partial.replace(path)


def read_lines(pdf_path: Path) -> list[TextLine]:
    """Read every line of a PDF, page after page, without the page-number band at the bottom.

    Args:
        pdf_path: The PDF file.

    Returns:
        list[TextLine]: Lines from top to bottom of each page, pages in order.
    """
    lines: list[TextLine] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            body = page.crop((0, 0, page.width, page.height - FOOTER_HEIGHT))
            words = body.extract_words(extra_attrs=["fontname", "non_stroking_color"])
            # With extra_attrs, pdfplumber returns the words of one font and colour together, not in reading order
            # (the "Wykładowcy" column came after the whole "Koordynatorzy" column), so lines are rebuilt here:
            # top to bottom, then left to right within a line.
            words.sort(key=lambda word: (word["top"], word["x0"]))
            page_lines: list[TextLine] = []
            for word in words:
                if page_lines and word["top"] - page_lines[-1].words[0]["top"] <= LINE_TOLERANCE:
                    page_lines[-1].words.append(word)
                else:
                    page_lines.append(TextLine([word]))
            for line in page_lines:
                line.words.sort(key=lambda word: word["x0"])
            lines.extend(page_lines)
    return lines


def holds_only_labels(text: str) -> bool:
    """Say whether a section's text is only sub-headings with nothing under them.

    Args:
        text: Section text, one line per PDF line.

    Returns:
        bool: True when every non-blank line ends with ":" — e.g. the three lines "Wiedza:", "Umiejętności:",
            "Kompetencje społeczne:" of a template the author did not fill in. False for an empty text (handled as
            empty separately) and as soon as one line holds content.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines) and all(line.endswith(":") for line in lines)


def staff_columns_text(heading: TextLine, body: list[TextLine]) -> str:
    r"""Turn the first page's two columns, coordinators and lecturers, into one text.

    Args:
        heading: The "Koordynatorzy Wykładowcy" heading line; its "Wykładowcy" word marks where the second column
            starts.
        body: The lines under it, each possibly holding words of both columns.

    Returns:
        str: One line "Koordynatorzy:" followed by that column's lines (e.g. "dr Jan Kowalski",
            "jan.kowalski@put.poznan.pl"), a blank line, then the same for the lecturers' column. A column with no lines
            keeps its label with nothing under it, as the PDF does. Empty when both columns are.
    """
    second_column_x = next(word["x0"] for word in heading.words if word["text"] == LECTURERS_HEADING_WORD)
    coordinators: list[str] = []
    lecturers: list[str] = []
    for line in body:
        left = [word["text"] for word in line.words if word["x0"] < second_column_x - COLUMN_TOLERANCE]
        right = [word["text"] for word in line.words if word["x0"] >= second_column_x - COLUMN_TOLERANCE]
        if left:
            coordinators.append(" ".join(left))
        if right:
            lecturers.append(" ".join(right))
    if not coordinators and not lecturers:
        return ""
    return "\n".join(["Koordynatorzy:", *coordinators, "", "Wykładowcy:", *lecturers])


def parse_course_card(pdf_path: Path) -> ParsedCourseCard:
    """Cut a syllabus PDF into the sections worth storing.

    Args:
        pdf_path: The PDF file.

    Returns:
        ParsedCourseCard: The sections and what was left out; see its attributes. ``sections`` is empty when no known
            heading was found (not a syllabus, or a changed template).
    """
    parsed = ParsedCourseCard()

    # Group lines under the heading above them; text before the first heading is the page header.
    blocks: list[tuple[TextLine, list[TextLine]]] = []
    for line in read_lines(pdf_path):
        if line.is_heading:
            if line.text == LAST_SECTION_HEADING:
                parsed.reached_last_section = True
                break
            # A staff list too long for one page continues on the next under a repeated column heading (e.g. a lone
            # "Wykładowcy" on page 2 of the football card): it stays in the staff block, whose split by x still puts
            # each name in its column.
            continues_staff = blocks and blocks[-1][0].text == STAFF_HEADING
            if continues_staff and {word["text"] for word in line.words} <= set(STAFF_HEADING.split()):
                continue
            blocks.append((line, []))
        elif blocks and not line.is_template_text:
            blocks[-1][1].append(line)

    seen_texts: set[str] = set()
    for heading, body in blocks:
        if heading.text in SECTIONS_TO_SKIP:
            continue
        if heading.text == STAFF_HEADING:
            name, text = STAFF_SECTION, staff_columns_text(heading, body)
        elif heading.text in SECTIONS_TO_STORE:
            name, text = heading.text, "\n".join(line.text for line in body)
        else:
            parsed.unknown_headings.append(heading.text)
            continue
        if name in parsed.sections:
            parsed.unknown_headings.append(f"{heading.text} (second time)")
            continue

        # Compared without letter case and line breaks; stored as printed.
        comparable = " ".join(text.split()).lower()
        if comparable in EMPTY_SECTION_TEXTS or holds_only_labels(text):
            parsed.empty_sections.append(name)
        elif comparable in seen_texts:
            parsed.repeated_sections.append(name)
        else:
            seen_texts.add(comparable)
            parsed.sections[name] = text.strip()
    return parsed


def load_course_card(session: requests.Session, url: str, cache_dir: Path) -> tuple[ParsedCourseCard, bool]:
    """Get one syllabus from the cache, or download it into the cache, and parse it.

    Args:
        session: HTTP session shared by all downloads of the run.
        url: Syllabus link.
        cache_dir: Root folder of the cache.

    Returns:
        tuple[ParsedCourseCard, bool]: The parsed syllabus (with at least one section) and whether it was downloaded
        now (False: read from the cache).

    Raises:
        CourseCardUnavailableError: The link is not a syllabus link, the download failed, the PDF cannot be read or has
            no known section.
    """
    path = cached_pdf_path(url, cache_dir)
    if path is None:
        raise CourseCardUnavailableError(f"not a {COURSE_CARD_URL_PREFIX} link")
    downloaded = not path.exists()
    if downloaded:
        time.sleep(DOWNLOAD_PAUSE_SECONDS)
        download_course_card(session, url, path)
    try:
        parsed = parse_course_card(path)
    except Exception as error:  # pdfminer raises many different exceptions on a damaged file
        raise CourseCardUnavailableError(f"unreadable PDF ({type(error).__name__}: {error})") from error
    if not parsed.sections:
        raise CourseCardUnavailableError("no known section heading in the PDF")
    return parsed, downloaded


def ingest_course_cards(cursor: DBCursor, cache_dir: Path = COURSE_CARD_CACHE_DIR) -> None:
    """Store the sections of every stored course's syllabus, and print what was stored and what was not.

    Run after the courses: it reads their links from the database. For each course the syllabus sections are upserted
    and the course's sections its PDF no longer has are deleted. A syllabus that cannot be downloaded or read keeps its
    stored sections. Sections of courses that left the catalog are already gone with the course (``ON DELETE CASCADE``).

    Args:
        cursor: Database cursor of the ingestion run.
        cache_dir: Root folder of the PDF cache.
    """
    courses = get_courses_with_card_links(cursor)
    print(f"\nCourse cards: {len(courses)} courses link to one; downloading what is not cached yet...")

    session = requests.Session()
    # One course row per link today, but parse a link once even if several courses share it.
    parsed_by_url: dict[str, ParsedCourseCard | None] = {}
    files = Counter()
    sections_per_name = Counter()
    unknown_headings = Counter()
    skipped: list[str] = []
    without_last_section: list[str] = []
    courses_stored = empty_sections = repeated_sections = sections_removed = 0

    for number, course in enumerate(courses, start=1):
        url = course["source_url"]
        if url not in parsed_by_url:
            try:
                parsed, downloaded = load_course_card(session, url, cache_dir)
            except CourseCardUnavailableError as error:
                parsed_by_url[url] = None
                skipped.append(f"{error}: {url}")
                print(f"  skipped, stored sections kept — {error}: {url}")
                continue
            parsed_by_url[url] = parsed
            files["downloaded" if downloaded else "from cache"] += 1
            empty_sections += len(parsed.empty_sections)
            repeated_sections += len(parsed.repeated_sections)
            unknown_headings.update(parsed.unknown_headings)
            if not parsed.reached_last_section:
                without_last_section.append(url)

        parsed = parsed_by_url[url]
        if parsed is None:
            continue
        key = (course["course_code"], course["major_id"], course["semester"])
        for section, text in parsed.sections.items():
            upsert_course_card_section(cursor, *key, source_url=url, section=section, chunk_index=0, content=text)
            sections_per_name[section] += 1
        sections_removed += delete_sections_not_seen_in_run(cursor, *key)
        courses_stored += 1

        if number % 50 == 0:
            print(f"  {number}/{len(courses)} courses done")

    print(
        f"\nCourse cards done: {courses_stored} of {len(courses)} courses stored, "
        f"{sum(sections_per_name.values())} sections."
    )
    print(f"  PDFs: {files['downloaded']} downloaded, {files['from cache']} from the cache, {len(skipped)} skipped.")
    print("  Sections stored per name: " + ", ".join(f"{name} {count}" for name, count in sections_per_name.items()))
    print(
        f"  Left out: {empty_sections} empty sections, {repeated_sections} repeating an earlier section of their "
        f"PDF; {sections_removed} stored sections no longer in their PDF removed."
    )
    if unknown_headings:
        print(
            "  Unknown headings (their text not stored): "
            + ", ".join(f"{h!r} ×{n}" for h, n in unknown_headings.items())
        )
    if without_last_section:
        print(f"  WARNING: {len(without_last_section)} PDFs never reached '{LAST_SECTION_HEADING}' — check that no")
        print("  workload-table numbers ended up in their last section:")
        for url in without_last_section:
            print(f"    {url}")
    if skipped:
        print("  Skipped (their stored sections kept):")
        for line in skipped:
            print(f"    {line}")
