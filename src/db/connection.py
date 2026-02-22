import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Default connection string matching docker-compose.db.yml
DEFAULT_DB_URL = "postgresql://sentinel:sentinel@localhost:5432/sentinel_db"

def get_engine(database_url: str | None = None):
    """
    Creates and returns a SQLAlchemy engine instance.
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
