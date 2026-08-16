"""Database persistence package for Indian Market Portfolio Intelligence platform."""
from __future__ import annotations

from .connection import (
    check_connection_health,
    create_db_engine,
    ensure_sqlite_directory_exists,
    get_default_sqlite_url,
    get_engine,
    get_project_root,
    mask_database_url,
    normalize_database_url,
    reset_engine,
)
from .crud import (
    count_backtests,
    count_benchmarks,
    count_regime_snapshots,
    create_backtest_log,
    create_benchmark_batch,
    create_benchmark_run,
    create_regime_snapshot,
    delete_backtest_by_id,
    get_backtest_by_id,
    get_benchmark_history,
    get_latest_benchmark_runs,
    get_latest_regime_snapshot,
    get_recent_backtests,
    get_recent_regime_snapshots,
)
from .models import (
    BacktestLog,
    Base,
    ModelBenchmarkRun,
    RegimeSnapshot,
    utc_now,
)
from .service import (
    init_db,
    save_analysis_record,
    save_backtest_record,
    save_benchmark_record,
)
from .session import (
    SessionLocal,
    get_db,
    get_db_session,
    get_session_factory,
)

__all__ = [
    # Models & Base
    "Base",
    "BacktestLog",
    "RegimeSnapshot",
    "ModelBenchmarkRun",
    "utc_now",
    # Connection & Engine
    "get_engine",
    "create_db_engine",
    "reset_engine",
    "check_connection_health",
    "normalize_database_url",
    "mask_database_url",
    "get_default_sqlite_url",
    "get_project_root",
    "ensure_sqlite_directory_exists",
    # Session Management
    "SessionLocal",
    "get_db",
    "get_db_session",
    "get_session_factory",
    # Services & Initialization
    "init_db",
    "save_analysis_record",
    "save_backtest_record",
    "save_benchmark_record",
    # CRUD Operations
    "create_backtest_log",
    "get_backtest_by_id",
    "get_recent_backtests",
    "count_backtests",
    "delete_backtest_by_id",
    "create_regime_snapshot",
    "get_latest_regime_snapshot",
    "get_recent_regime_snapshots",
    "count_regime_snapshots",
    "create_benchmark_run",
    "create_benchmark_batch",
    "get_latest_benchmark_runs",
    "get_benchmark_history",
    "count_benchmarks",
]
