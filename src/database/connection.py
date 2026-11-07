"""One shared pool of database connections for the whole program.

Opening a PostgreSQL connection is slow, so connections are opened once and
reused. Everything that talks to the database borrows a connection with
``get_db_connection()``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from psycopg import Connection
from psycopg_pool import ConnectionPool

from config import DatabaseSettings

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    """Return the shared pool, creating it the first time it is needed.

    Returns:
        ConnectionPool: An open pool using the DB_* settings from ``.env``.
    """
    global _pool
    pool = _pool
    if pool is None:
        db_settings = DatabaseSettings()
        pool = ConnectionPool(
            kwargs=db_settings.to_dict(),
            min_size=1,
            max_size=10,
            open=True,
        )
        _pool = pool
    return pool


@contextmanager
def get_db_connection() -> Iterator[Connection]:
    """Borrow a connection from the pool for the duration of a ``with`` block.

    When the block ends normally its changes are committed; if it raises, they are rolled back. Either way the
    connection goes back to the pool.

    Returns:
        Iterator[Connection]: The borrowed connection (use it as
        ``with get_db_connection() as conn: ``).
    """
    with get_pool().connection() as conn:
        yield conn


def close_pool() -> None:
    """Close all pooled connections; call once when the program ends.

    If the database is used again afterwards, a new pool is opened.
    """
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
