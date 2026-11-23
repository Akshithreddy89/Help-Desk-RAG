
# app/database/connection.py

from collections.abc import Generator

from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.config import settings

if not settings.DATABASE_URL:
    raise ValueError("DATABASE_URL is missing from .env")


def _configure_connection(conn) -> None:
    """Configure pgvector types and search_path for each pooled connection."""
    conn.execute("SET search_path TO public, extensions")
    register_vector(conn)
    conn.commit()


# Create a reusable PostgreSQL connection pool
db_pool = ConnectionPool(
    conninfo=settings.DATABASE_URL,

    # Minimum connections maintained in the pool
    min_size=1,

    # Maximum simultaneous connections
    max_size=10,

    # Maximum time to wait for an available connection
    timeout=10,

    # Close connections that have been idle for too long
    max_idle=300,

    # Register pgvector adapter on every new connection
    configure=_configure_connection,

    # Do not open connections automatically during import
    open=False
)


def open_pool():
    """Initialize the connection pool."""
    db_pool.open()
    db_pool.wait(timeout=10)

    print("Supabase connection pool initialized!")


def get_db_connection():
    """Get a reusable connection context manager from the pool."""
    return db_pool.connection()


def get_db() -> Generator:
    """FastAPI dependency that yields a pooled connection."""
    with db_pool.connection() as conn:
        yield conn


def close_pool():
    """Close all connections when the application shuts down."""
    db_pool.close()
    print("Supabase connection pool closed!")
