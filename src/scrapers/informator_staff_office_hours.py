"""Scraper for Informator PP, the university's staff directory.

Collects the faculty's units (institutes, zakłady, offices), its staff, their
administrative functions and their weekly office hours.

Informator is used through the same JSON API its website calls
(``https://informator.put.poznan.pl/?do=<Controller.Action>``). Most of it is
behind a captcha that has to be solved once per session in a browser, so the
scraper borrows that browser session: INFORMATOR_SESSION in ``.env`` holds
the session cookie. When the cookie is missing or has expired,
``InformatorSessionError`` is raised.

The API often sends ``null`` where an object or text is expected, which is
why the code below writes ``(x.get(key) or {})`` instead of
``x.get(key, {})`` — the latter only helps when the key is missing.
"""

from __future__ import annotations

import os
import time
from typing import Iterator

from psycopg import Cursor as DBCursor
from requests import Session
from requests.exceptions import RequestException

from database.queries.departments import (
    delete_departments_not_seen_in_run,
    upsert_department,
)
from database.queries.office_hours import (
    delete_office_hours_of_staff,
    insert_office_hour,
)
from database.queries.staff import (
    count_staff_not_seen_in_run,
    delete_staff_not_seen_in_run,
    upsert_staff,
)
from database.queries.staff_functions import (
    delete_staff_functions_of_staff,
    insert_staff_function,
)

BASE_URL = "https://informator.put.poznan.pl"
FACULTY_SYMBOL = "WIiT"
PAGE_SIZE = 100  # people per page when listing staff; the API's maximum
REQUEST_DELAY_S = 0.2  # pause between per-person requests, to go easy on the server
MAX_RETRIES = 3  # attempts per API call before giving up
RETRY_BACKOFF_S = 2.0  # wait before the 2nd attempt; doubles before the 3rd
# Above this share of stored staff missing from a run, nobody is deleted: more
# likely an incomplete staff list from the API than that many people leaving.
MAX_STAFF_REMOVAL_SHARE = 0.2

# Informator gives weekdays in English; the database stores them in Polish.
WEEKDAYS_PL = {
    "monday": "poniedziałek",
    "tuesday": "wtorek",
    "wednesday": "środa",
    "thursday": "czwartek",
    "friday": "piątek",
    "saturday": "sobota",
    "sunday": "niedziela",
}


class InformatorSessionError(RuntimeError):
    """The Informator session cookie is missing, has expired, or its captcha was not solved.

    Copy a fresh cookie into INFORMATOR_SESSION (see ``.env.example``).
    """


def create_session() -> Session:
    """Create an HTTP session that sends the browser's Informator cookie with every request.

    Returns:
        Session: A session ready for ``api_call``.

    Raises:
        InformatorSessionError: INFORMATOR_SESSION is not set.
    """
    cookie = os.getenv("INFORMATOR_SESSION", "").strip()
    if not cookie:
        raise InformatorSessionError("INFORMATOR_SESSION is not set.\n")

    session = Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (WIiTek ingestion; student project)"
    session.cookies.set("informator", cookie, domain="informator.put.poznan.pl")
    return session


def api_call(session: Session, action: str, post: bool = True) -> dict:
    """Call one Informator API action and return the data it sends back.

    Network errors and responses that are not JSON are retried, with a growing pause between attempts.

    Args:
        session: Session from ``create_session``.
        action: Everything after ``?do=`` in the URL, e.g. ``"Card.CardShow&uuid=..."``.
        post: Send a POST request if True, a GET request if False (the organization tree is the only GET endpoint used).

    Returns:
        dict: The ``result`` part of the response.

    Raises:
        InformatorSessionError: The session is not valid — the API redirected to the login page or reported an unsolved
            captcha.
        RuntimeError: The API reported a failure, or every attempt failed.
    """
    url = f"{BASE_URL}/?do={action}"
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            time.sleep(RETRY_BACKOFF_S * (attempt - 1))
        try:
            if post:
                response = session.post(url, json={"captcha": None}, allow_redirects=False, timeout=30)
            else:
                response = session.get(url, allow_redirects=False, timeout=30)
        except RequestException as error:
            last_error = f"network error: {error}"
            continue

        if response.status_code in (301, 302, 303):
            raise InformatorSessionError("Session cookie expired (redirected to login).\n")

        try:
            payload = response.json()
        except ValueError:
            last_error = f"HTTP {response.status_code}, non-JSON response: {response.text[:120]!r}"
            continue

        status = payload.get("status", {})
        if status.get("request") != "SUCCESS":
            if status.get("captcha") == "NOT_VERIFIED":
                raise InformatorSessionError("Captcha not verified.\n")
            raise RuntimeError(f"Informator API call failed: {action} -> {status}")
        return payload["result"]

    raise RuntimeError(f"Informator API call failed after {MAX_RETRIES} attempts: {action} -> {last_error}")


def find_unit_by_symbol(node: dict, symbol: str) -> dict | None:
    """Search the university's organization tree for the unit with a given symbol.

    Args:
        node: The unit to start from (with all units below it).
        symbol: Unit symbol, e.g. "WIiT".

    Returns:
        dict | None: The matching unit, including all units below it, or
        None if no unit has this symbol.
    """
    if node.get("symbol") == symbol:
        return node
    for child in node.get("children") or []:
        found = find_unit_by_symbol(child, symbol)
        if found:
            return found
    return None


def iter_units(node: dict, depth: int = 0) -> Iterator[tuple[dict, int]]:
    """Go through a unit and every unit below it, each unit before its sub-units.

    Args:
        node: The unit to start from.
        depth: How deep ``node`` is; the starting unit is 0, its direct sub-units 1, and so on.

    Returns:
        Iterator[tuple[dict, int]]: Pairs of (unit, its depth).
    """
    yield node, depth
    for child in node.get("children") or []:
        yield from iter_units(child, depth + 1)


def fetch_staff(session: Session, unit_id: int) -> Iterator[dict]:
    """List every employee of a unit, including the units below it.

    The API returns at most ``PAGE_SIZE`` people per request, so this keeps requesting the next page until all are
    fetched.

    Args:
        session: Session from ``create_session``.
        unit_id: Informator's id of the unit.

    Returns:
        Iterator[dict]: One employee entry per person, as the API returns it.
    """
    page = 1
    while True:
        result = api_call(
            session,
            f"Organization.OrganizationUnitDetails&id={unit_id}"
            f"&page={page}&max_results={PAGE_SIZE}&user_type=&query_name=",
            True,
        )
        employees = result["employees"]
        yield from employees["items"]
        if page * PAGE_SIZE >= employees["total"]:
            return
        page += 1


def pick_primary_position(
    positions: dict, unit_names_by_id: dict[int, tuple[str, int]]
) -> tuple[str | None, str | None]:
    """Choose which of a person's jobs describes them, and return its unit and title.

    A person can hold several positions. The one in the most specific WIiT unit wins (a zakład beats the institute it
    belongs to). The unit and the job title are taken from the same position so they always match.

    Args:
        positions: The person's positions, from ``employment.data.positions``.
        unit_names_by_id: WIiT unit id → (unit name, depth in the tree).

    Returns:
        tuple[str | None, str | None]: (unit name, job title); both None if
        none of the positions is in a WIiT unit.
    """
    best_name = None
    best_position = None
    best_depth = -1
    for position in positions.values():
        department = position.get("department") or {}
        unit = unit_names_by_id.get(department.get("id"))
        # unit is (name, depth); deeper = more specific (zakład beats instytut).
        if unit and unit[1] > best_depth:
            best_name, best_depth = unit[0], unit[1]
            best_position = ((position.get("name") or {}).get("pl") or "").strip() or None
    return best_name, best_position


def build_location(meeting: dict) -> str | None:
    """Describe where one office-hours slot takes place.

    Args:
        meeting: One slot from the person's card.

    Returns:
        str | None: "<building> <room>" (e.g. "A2 101"), "online" if there is
        only a meeting link, or None if neither is given.
    """
    parts = [meeting.get("building_symbol"), meeting.get("room_number")]
    location = " ".join(part for part in parts if part)
    if not location and meeting.get("online_link"):
        location = "online"
    return location[:100] or None


def sync_office_hours(cursor: DBCursor, session: Session, staff_id: int, uuid: str, source_url: str) -> int:
    """Make the database hold exactly the office hours currently on one person's card.

    The person's stored slots are deleted and the current ones inserted, so slots removed from Informator also disappear
    here. Only weekly slots are stored; one-off meetings on a specific date are skipped.

    Args:
        cursor: Database cursor to run the SQL on.
        session: Session from ``create_session``.
        staff_id: The person's id in the database.
        uuid: The person's card id in Informator.
        source_url: Link to the person's card, stored as the source.

    Returns:
        int: Number of slots stored.
    """
    result = api_call(session, f"Card.CardShow&uuid={uuid}", True)
    modules = result["card"]["modules"]
    meetings_data = (modules.get("meetings", {}) or {}).get("data", {}) or {}
    meetings = meetings_data.get("meetings", {}) or {}

    delete_office_hours_of_staff(cursor, staff_id)
    stored = 0
    for meeting in meetings.values():
        weekday = WEEKDAYS_PL.get(meeting.get("weekday"))
        if meeting.get("frequency") != "weekly" or not weekday:
            continue  # one-off meetings carry a date, not a weekday; out of scope
        insert_office_hour(
            cursor=cursor,
            staff_id=staff_id,
            day_of_week=weekday,
            start_time=meeting["start_hour"],
            end_time=meeting["end_hour"],
            location=build_location(meeting),
            comment=meeting.get("comment") or None,
            source_url=source_url,
        )
        stored += 1
    return stored


def sync_functions(cursor: DBCursor, staff_id: int, functions: dict, source_url: str) -> int:
    """Make the database hold exactly the administrative functions on one person's card.

    The person's stored functions (e.g. "Dyrektor instytutu") are deleted and the current ones inserted, so a function
    the person no longer holds disappears here too.

    Args:
        cursor: Database cursor to run the SQL on.
        staff_id: The person's id in the database.
        functions: The functions from the person's card (``employment.data.functions``); may be None.
        source_url: Link to the person's card, stored as the source.

    Returns:
        int: Number of functions stored.
    """
    delete_staff_functions_of_staff(cursor, staff_id)
    stored = 0
    for function in (functions or {}).values():
        name = ((function.get("name") or {}).get("pl") or "").strip()
        if not name:
            continue
        department = ((function.get("department") or {}).get("name") or {}).get("pl")
        department = (department or "").strip()[:100] or None
        insert_staff_function(
            cursor=cursor,
            staff_id=staff_id,
            function_name=name[:150],
            department_name=department,
            source_url=source_url,
        )
        stored += 1
    return stored


def remove_staff_and_units_not_seen(cursor: DBCursor) -> None:
    """Delete staff and units that Informator no longer lists, e.g. people who left WIiT.

    Runs only after the whole staff list was stored. If more than ``MAX_STAFF_REMOVAL_SHARE`` of the stored staff would
    go, nothing is deleted and a warning is printed; ``clean_db.py`` plus a full run is then the way to accept the
    change.

    Args:
        cursor: Database cursor of the ingestion run (one transaction).
    """
    not_seen, total = count_staff_not_seen_in_run(cursor)
    if total and not_seen / total > MAX_STAFF_REMOVAL_SHARE:
        # Units are skipped too: staff kept here may still point at them.
        print(
            f"WARNING: {not_seen} of {total} stored staff members are missing from this run "
            f"(over {MAX_STAFF_REMOVAL_SHARE:.0%}), so no staff or units were deleted."
        )
        return
    staff_deleted = delete_staff_not_seen_in_run(cursor)
    units_deleted = delete_departments_not_seen_in_run(cursor)
    print(f"Removed {staff_deleted} staff members and {units_deleted} units no longer in Informator.")


def ingest_informator(cursor: DBCursor) -> None:
    """Store WIiT's units, then every staff member with their functions and office hours.

    People without an e-mail or without a full name are skipped, because the e-mail is what identifies a person in the
    database. If one person's office hours fail to load, that is printed and the next person follows; an expired session
    stops the whole step. Only when every person was processed are staff and units that Informator no longer lists
    deleted (see ``remove_staff_and_units_not_seen``).

    Args:
        cursor: Database cursor to run the SQL on (the caller commits).

    Raises:
        InformatorSessionError: The session cookie is missing or no longer valid.
        RuntimeError: WIiT was not found in Informator's organization tree, or an API call failed.
    """
    print("Starting Informator PP staff scraper...")
    session = create_session()

    print("Fetching organization tree...")
    tree = api_call(session, "Organization.OrganizationUnitsTree", False)
    faculty = find_unit_by_symbol(tree, FACULTY_SYMBOL)
    if faculty is None:
        raise RuntimeError(f"Faculty with symbol {FACULTY_SYMBOL!r} not found in organization tree")

    # Parents come before children, so a unit's parent row already exists
    # when the unit is upserted (departments.parent is a foreign key).
    # Name slices here and below match the VARCHAR lengths in docker/init.sql.
    units = list(iter_units(faculty))
    unit_names_by_id: dict[int, tuple[str, int]] = {}
    for node, depth in units:
        if depth == 0:
            continue  # the faculty itself lives in `faculties`, not in `departments`
        name = node["name"]["pl"][:100]
        unit_names_by_id[node["id"]] = (name, depth)
        parent = unit_names_by_id.get(node["parent_id"])
        phones = ", ".join(p["number"] for p in node.get("phones", []))
        upsert_department(
            cursor=cursor,
            name=name,
            symbol=(node.get("symbol") or "")[:10] or None,
            parent=parent[0] if parent else None,
            unit_type=node.get("type"),
            faculty=FACULTY_SYMBOL,
            address=node.get("address"),
            phone=phones[:100] or None,
            email=node.get("email"),
            source_url=f"{BASE_URL}/app/unit/{node['id']}",
        )
    print(f"Stored {len(units) - 1} units of {FACULTY_SYMBOL}.")

    processed = skipped = hours_total = functions_total = 0
    # The faculty node includes everyone in its institutes and zakłady.
    for item in fetch_staff(session, faculty["id"]):
        basic = (item["modules"].get("basic", {}) or {}).get("data", {}) or {}
        contact = (item["modules"].get("contact", {}) or {}).get("data", {}) or {}
        employment = (item["modules"].get("employment", {}) or {}).get("data", {}) or {}

        email = (contact.get("email", "") or "").strip().lower()
        first_name = (basic.get("name", "") or "").strip()
        last_name = (basic.get("surname", "") or "").strip()
        if not (email and first_name and last_name):
            print(f"  -> Skipping (no email or name): {first_name} {last_name}")
            skipped += 1
            continue

        source_url = f"{BASE_URL}/app/card/view/{item['uuid']}"
        department, position = pick_primary_position(employment.get("positions", {}) or {}, unit_names_by_id)
        staff_id = upsert_staff(
            cursor=cursor,
            first_name=first_name[:50],
            last_name=last_name[:50],
            academic_title=(basic.get("title", "") or "").strip()[:100] or None,
            position=position,
            user_type=(item.get("user_type") or "").strip()[:50] or None,
            department=department,
            email=email[:100],
            phone=", ".join(contact.get("phones", []))[:100] or None,
            source_url=source_url,
        )

        functions_total += sync_functions(cursor, staff_id, employment.get("functions", {}) or {}, source_url)

        try:
            hours = sync_office_hours(cursor, session, staff_id, item["uuid"], source_url)
        except InformatorSessionError:
            raise  # dead session — no point continuing with the remaining people
        except (RuntimeError, RequestException) as error:
            hours = 0
            print(f"  -> Office hours failed for {last_name} {first_name}: {error}")
        hours_total += hours
        processed += 1
        print(f"  [{processed}] {last_name} {first_name} ({email}), office hours: {hours}")
        time.sleep(REQUEST_DELAY_S)

    print(
        f"\nInformator done: {processed} staff members stored "
        f"({skipped} skipped), {hours_total} office-hours slots, {functions_total} functions."
    )

    # Reached only if the loop above finished: an exception (e.g. an expired
    # session) leaves before this, so a partial run never deletes anyone.
    remove_staff_and_units_not_seen(cursor)
