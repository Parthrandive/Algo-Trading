import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Docker TimescaleDB (docker-compose.db.yml)
DOCKER_DB_URL = "postgresql://sentinel:sentinel@localhost:5432/sentinel_db"

# Local PostgreSQL installation
LOCAL_DB_URL = "postgresql://postgres:optimus@localhost:5433/sentinel_db"

# Pick default based on DB_MODE env var: "local" or "docker" (default)
DEFAULT_DB_URL = LOCAL_DB_URL if os.getenv("DB_MODE", "docker").lower() == "local" else DOCKER_DB_URL

def get_engine(database_url: str | None = None):
    """
    Creates and returns a SQLAlchemy engine instance.

    Resolution order:
      1. Explicit *database_url* argument
      2. DATABASE_URL environment variable
      3. DEFAULT_DB_URL (chosen by DB_MODE: 'local' | 'docker')
    """
    url = database_url or os.getenv("DATABASE_URL", DEFAULT_DB_URL)
    return create_engine(
        url,
        pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
        pool_pre_ping=True, # Verify connections before using them
    )

def get_session(engine=None) -> sessionmaker:
    """
    Returns a configured sessionmaker bound to the given engine.
    """
    if engine is None:
        engine = get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def transaction(session_factory: sessionmaker) -> Generator[Session, None, None]:
    """
    Context manager for safe database transactions.
    Rolls back on exception, commits on success.
    """
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
