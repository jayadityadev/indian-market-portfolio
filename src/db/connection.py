"""Database engine management, connection pooling, and resilient SQLite fallback."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import threading
from typing import Any, Optional

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool, StaticPool

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Constants
DEFAULT_SQLITE_RELATIVE_PATH = "data/portfolio_intel.db"
ENV_DATABASE_URL_KEY = "DATABASE_URL"

def get_project_root() -> Path:
    """Resolve project root directory (indian-market-portfolio/)."""
    return Path(__file__).resolve().parent.parent.parent

# Always load .env from project root if available
load_dotenv(get_project_root() / ".env")

_engine: Optional[Engine] = None
_engine_lock = threading.Lock()


def get_default_sqlite_url() -> str:
    """Generate default SQLite database URL with absolute path."""
    db_path = get_project_root() / DEFAULT_SQLITE_RELATIVE_PATH
    return f"sqlite:///{db_path.as_posix()}"


def normalize_database_url(url: Optional[str]) -> str:
    """Normalize database connection string for SQLAlchemy 2.0.

    - Replaces 'postgres://' with 'postgresql://'
    - Falls back to default SQLite path if url is empty or whitespace.
    """
    if not url or not url.strip():
        return get_default_sqlite_url()

    url = url.strip()
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]

    return url


def ensure_sqlite_directory_exists(sqlite_url: str) -> None:
    """Create parent directories if SQLite database is file-based."""
    if not sqlite_url.startswith("sqlite:///"):
        return

    path_part = sqlite_url[len("sqlite:///"):]
    if path_part in (":memory:", "", "?check_same_thread=False"):
        return

    # Strip any query parameters
    file_path_str = path_part.split("?")[0]
    file_path = Path(file_path_str)
    if not file_path.is_absolute():
        file_path = get_project_root() / file_path

    parent_dir = file_path.parent
    if parent_dir and not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created missing directory for SQLite database: %s", parent_dir)


def mask_database_url(url: str) -> str:
    """Mask credentials in database URL for safe logging and health check output."""
    if not url:
        return "None"
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)


def create_db_engine(
    db_url: Optional[str] = None,
    echo: bool = False,
    force_sqlite_fallback: bool | None = None,
) -> Engine:
    """Create a new SQLAlchemy Engine configured for Neon PostgreSQL or SQLite."""
    if force_sqlite_fallback is None:
        force_sqlite_fallback = os.getenv("APP_ENV", "development").lower() != "production"
    raw_url = db_url if db_url is not None else os.getenv(ENV_DATABASE_URL_KEY)
    normalized_url = normalize_database_url(raw_url)

    # 1. SQLite Engine Configuration
    if normalized_url.startswith("sqlite"):
        ensure_sqlite_directory_exists(normalized_url)
        is_memory = ":memory:" in normalized_url

        connect_args = {"check_same_thread": False}
        poolclass = StaticPool if is_memory else QueuePool

        logger.info("Initializing SQLite database engine: %s", mask_database_url(normalized_url))
        return create_engine(
            normalized_url,
            echo=echo,
            connect_args=connect_args,
            poolclass=poolclass,
        )

    # 2. PostgreSQL Engine Configuration (Neon with Pooling Resilience)
    try:
        logger.info("Initializing PostgreSQL engine: %s", mask_database_url(normalized_url))
        engine = create_engine(
            normalized_url,
            echo=echo,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            connect_args={"connect_timeout": 5},
        )

        # Test connection immediately
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        logger.info("Successfully connected to PostgreSQL database.")
        return engine

    except Exception as exc:
        logger.warning(
            "Failed to connect to PostgreSQL database (%s): %s",
            mask_database_url(normalized_url),
            exc,
        )
        if force_sqlite_fallback:
            fallback_url = get_default_sqlite_url()
            logger.warning("Falling back to local SQLite database: %s", fallback_url)
            ensure_sqlite_directory_exists(fallback_url)
            return create_engine(
                fallback_url,
                echo=echo,
                connect_args={"check_same_thread": False},
                poolclass=QueuePool,
            )
        raise


def get_engine(
    db_url: Optional[str] = None,
    echo: bool = False,
    force_sqlite_fallback: bool | None = None,
) -> Engine:
    """Thread-safe singleton getter for database engine."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = create_db_engine(
                    db_url=db_url,
                    echo=echo,
                    force_sqlite_fallback=force_sqlite_fallback,
                )
    return _engine


def reset_engine() -> None:
    """Dispose and reset the cached global engine (useful for testing)."""
    global _engine
    with _engine_lock:
        if _engine is not None:
            try:
                _engine.dispose()
            except Exception:
                pass
            _engine = None


def check_connection_health(engine: Optional[Engine] = None) -> dict[str, Any]:
    """Execute a lightweight health check against the database."""
    target_engine = engine or get_engine()
    url_str = str(target_engine.url)
    dialect_name = target_engine.dialect.name

    try:
        with target_engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        pool_status: dict[str, Any] = {}
        if hasattr(target_engine.pool, "size"):
            try:
                pool_status = {
                    "pool_size": target_engine.pool.size(),
                    "checked_in": target_engine.pool.checkedin(),
                    "checked_out": target_engine.pool.checkedout(),
                }
            except Exception:
                pass

        return {
            "status": "healthy",
            "dialect": dialect_name,
            "url_masked": mask_database_url(url_str),
            "pool": pool_status,
            "error": None,
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "dialect": dialect_name,
            "url_masked": mask_database_url(url_str),
            "pool": {},
            "error": str(exc),
        }
