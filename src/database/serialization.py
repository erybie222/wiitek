"""Turning database rows into dictionaries that can be converted to JSON.

Tool results are sent to the model as JSON, but some values the database
returns — ``Decimal`` (e.g. threshold points), ``datetime`` and ``time``
(e.g. office hours) — cannot be written as JSON directly. Every query
builds its result rows with ``row_to_dict`` so this never fails.
"""

from __future__ import annotations

from datetime import datetime, time
from decimal import Decimal
from typing import Any


def _to_json_safe(value: Any) -> Any:
    """Convert one value from the database into one that JSON can store.

    Args:
        value: A column value as the database returned it.

    Returns:
        Any: A ``float`` for a ``Decimal``, an ISO-format string such as
        ``"09: 45:00"`` for a ``datetime`` or ``time``, and any other value unchanged.
    """
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, time)):
        return value.isoformat()
    return value


def row_to_dict(columns: tuple[str, ...], row: tuple) -> dict:
    """Turn one database row into a dictionary keyed by column name.

    Args:
        columns: Column names, in the same order as in the SELECT.
        row: One row as returned by the database.

    Returns:
        dict: Column name → value, with every value safe to convert to JSON.
    """
    return {column: _to_json_safe(value) for column, value in zip(columns, row)}
