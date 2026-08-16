"""Database session management, context managers, and FastAPI dependencies."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from .connection import get_engine

_session_factory: Optional[sessionmaker[Session]] = None


def get_session_factory(engine: Optional[Engine] = None) -> sessionmaker[Session]:
    """Get or create SQLAlchemy sessionmaker factory."""
    global _session_factory
    target_engine = engine or get_engine()
    if _session_factory is None or engine is not None:
        factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
            bind=target_engine,
        )
        if engine is None:
            _session_factory = factory
        return factory
    return _session_factory


def SessionLocal(engine: Optional[Engine] = None) -> Session:
    """Create a new Session instance."""
    factory = get_session_factory(engine)
    return factory()


def get_db(engine: Optional[Engine] = None) -> Generator[Session, None, None]:
    """FastAPI dependency yielding a database session per request."""
    db = SessionLocal(engine)
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session(engine: Optional[Engine] = None) -> Generator[Session, None, None]:
    """Context manager for standalone scripts, background tasks, and testing."""
    db = SessionLocal(engine)
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
