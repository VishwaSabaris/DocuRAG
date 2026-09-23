from collections.abc import Iterator

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.core.config import get_settings


settings = get_settings()


DATABASE_URL = (
    f"postgresql://{settings.postgres_user}:"
    f"{settings.postgres_password}@"
    f"{settings.postgres_host}:"
    f"{settings.postgres_port}/"
    f"{settings.postgres_db}"
)


pool = ConnectionPool(
    conninfo=DATABASE_URL,
    min_size=1,
    max_size=10,
    open=True,
    kwargs={
        "row_factory": dict_row,
    },
)


def get_connection() -> Iterator[Connection]:
    with pool.connection() as connection:
        yield connection


def close_pool() -> None:
    pool.close()
