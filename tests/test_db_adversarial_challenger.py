"""Adversarial stress test and empirical verification suite for Milestone M2 Database Persistence Layer.

Targets:
1. Malformed / unreachable PostgreSQL URLs and fallback to SQLite.
2. Concurrent session checkouts, pool exhaustion, and starvation handling.
3. Path creation for deeply nested and special SQLite database URLs.
4. Database URL credential masking robustness across varied character sets.
5. Database connection healthcheck behavior across healthy, degraded, broken, and closed engine states.
6. Edge case stress testing for background recording services, SQL injection resistance, and extreme payloads.
"""
from __future__ import annotations

import concurrent.futures
import math
import os
from pathlib import Path
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError, TimeoutError
from sqlalchemy.pool import NullPool, QueuePool, StaticPool

from src.db import (
    BacktestLog,
    Base,
    ModelBenchmarkRun,
    RegimeSnapshot,
    SessionLocal,
    check_connection_health,
    count_backtests,
    count_benchmarks,
    count_regime_snapshots,
    create_backtest_log,
    create_benchmark_batch,
    create_benchmark_run,
    create_db_engine,
    create_regime_snapshot,
    delete_backtest_by_id,
    ensure_sqlite_directory_exists,
    get_backtest_by_id,
    get_benchmark_history,
    get_db,
    get_db_session,
    get_default_sqlite_url,
    get_engine,
    get_latest_benchmark_runs,
    get_latest_regime_snapshot,
    get_project_root,
    get_recent_backtests,
    get_recent_regime_snapshots,
    get_session_factory,
    init_db,
    mask_database_url,
    normalize_database_url,
    reset_engine,
    save_analysis_record,
    save_backtest_record,
    save_benchmark_record,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_engine_state():
    """Ensure global engine singleton is reset before and after each test."""
    reset_engine()
    yield
    reset_engine()


# ---------------------------------------------------------------------------
# 1. Adversarial Tests: PostgreSQL URLs, Fallback & Failure Modes
# ---------------------------------------------------------------------------

class TestPostgresFallbackAdversarial:
    """Stress test PostgreSQL connection failure, fallback behavior, and driver resolution."""

    @pytest.mark.parametrize(
        "unreachable_url",
        [
            "postgresql://user:pass@127.0.0.1:59999/nonexistent_db",
            "postgresql://user:pass@invalid.domain.that.does.not.exist.internal:5432/neondb",
            "postgres://baduser:badpass@127.0.0.1:59999/neondb",
            "postgresql+psycopg2://user:pass@127.0.0.1:59999/neondb",
        ],
    )
    def test_unreachable_postgres_falls_back_to_functional_sqlite(self, unreachable_url: str):
        """Unreachable PostgreSQL hosts must automatically fallback to a fully working SQLite engine."""
        engine = create_db_engine(db_url=unreachable_url, force_sqlite_fallback=True)
        assert engine is not None
        assert engine.dialect.name == "sqlite"

        # Verify fallback engine can initialize tables and execute queries
        init_db(engine)
        with get_db_session(engine) as session:
            log = create_backtest_log(
                session=session,
                ticker="^NSEI",
                start_date="2023-01-01",
                end_date="2023-12-31",
                strategy="FallbackTestStrategy",
                initial_investment=50000.0,
            )
            created_id = log.id

        with get_db_session(engine) as session:
            retrieved = get_backtest_by_id(session, created_id)
            assert retrieved is not None
            assert retrieved.strategy == "FallbackTestStrategy"
        engine.dispose()

    def test_force_sqlite_fallback_disabled_raises_exception(self):
        """When force_sqlite_fallback=False, connection failure MUST raise exception rather than falling back."""
        bad_url = "postgresql://user:pass@127.0.0.1:59999/bad_db"
        with pytest.raises((SQLAlchemyError, OperationalError, Exception)):
            create_db_engine(db_url=bad_url, force_sqlite_fallback=False)

    def test_database_url_from_env_fallback(self, monkeypatch):
        """When DATABASE_URL is set in environment to unreachable host, fallback works transparently."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://envuser:envpass@127.0.0.1:59999/envdb")
        engine = get_engine()
        assert engine is not None
        assert engine.dialect.name == "sqlite"

        health = check_connection_health(engine)
        assert health["status"] == "healthy"
        assert health["dialect"] == "sqlite"

    def test_empty_and_whitespace_database_url_normalization(self):
        """Whitespace and None URLs normalize to default SQLite URL."""
        default_url = get_default_sqlite_url()
        assert normalize_database_url(None) == default_url
        assert normalize_database_url("") == default_url
        assert normalize_database_url("   \t\n  ") == default_url
        assert normalize_database_url("  postgres://user:pass@host/db  ").startswith("postgresql://")

    def test_non_postgres_non_sqlite_url_fallback(self):
        """Exotic/unsupported DB scheme falls back to SQLite when fallback is enabled."""
        exotic_url = "mysql+pymysql://user:pass@127.0.0.1:59999/testdb"
        engine = create_db_engine(db_url=exotic_url, force_sqlite_fallback=True)
        assert engine is not None
        assert engine.dialect.name == "sqlite"
        engine.dispose()


# ---------------------------------------------------------------------------
# 2. Concurrency & Pool Exhaustion Stress Tests
# ---------------------------------------------------------------------------

class TestConcurrencyAndPoolExhaustionStress:
    """Stress test concurrent session checkouts, pool exhaustion, and thread safety."""

    def test_high_concurrency_session_lifecycle(self, tmp_path: Path):
        """50 concurrent threads performing CRUD operations simultaneously must not deadlock or leak."""
        db_file = tmp_path / "concurrent_stress.db"
        url = f"sqlite:///{db_file.as_posix()}"
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False, "timeout": 30},
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
        )
        init_db(engine)

        num_threads = 50
        records_per_thread = 5
        errors: list[Exception] = []

        def worker_task(thread_id: int):
            try:
                for j in range(records_per_thread):
                    with get_db_session(engine) as session:
                        create_backtest_log(
                            session=session,
                            ticker=f"^NSE_{thread_id}",
                            start_date="2023-01-01",
                            end_date="2023-12-31",
                            strategy=f"Strategy_{thread_id}_{j}",
                            initial_investment=100000.0,
                            cagr=0.10 + (j * 0.01),
                            sharpe=1.0 + (j * 0.1),
                        )
                    time.sleep(0.005)
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Encountered concurrency errors: {errors}"

        # Verify all records persisted accurately
        with get_db_session(engine) as session:
            total_logs = len(get_recent_backtests(session, limit=1000))
            assert total_logs == num_threads * records_per_thread

        engine.dispose()

    def test_pool_exhaustion_timeout_and_recovery(self, tmp_path: Path):
        """Verify QueuePool raises TimeoutError when capacity is exceeded, then recovers after release."""
        db_file = tmp_path / "pool_exhaust.db"
        url = f"sqlite:///{db_file.as_posix()}"

        # Engine with capacity: pool_size=2 + max_overflow=1 = 3 connections total
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=QueuePool,
            pool_size=2,
            max_overflow=1,
            pool_timeout=0.5,  # 500ms timeout
        )
        init_db(engine)

        conns = []
        try:
            # Check out max capacity (3 connections)
            for _ in range(3):
                conns.append(engine.connect())

            # 4th connection attempt should timeout
            with pytest.raises(TimeoutError):
                engine.connect()

        finally:
            # Release all held connections
            for c in conns:
                c.close()

        # Verify pool recovery: new checkout must succeed immediately
        with engine.connect() as conn:
            val = conn.execute(text("SELECT 99")).scalar()
            assert val == 99

        engine.dispose()

    def test_get_db_generator_concurrent_cleanup(self, tmp_path: Path):
        """Ensure get_db generator properly closes sessions and returns connections under load."""
        db_file = tmp_path / "generator_stress.db"
        url = f"sqlite:///{db_file.as_posix()}"
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=5,
            pool_timeout=5,
        )
        init_db(engine)

        num_requests = 40
        errors: list[Exception] = []

        def mock_fastapi_request(req_id: int):
            try:
                gen = get_db(engine)
                session = next(gen)
                try:
                    stmt = select(BacktestLog)
                    _ = session.scalars(stmt).all()
                finally:
                    try:
                        next(gen)
                    except StopIteration:
                        pass
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(mock_fastapi_request, i) for i in range(num_requests)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Generator checkout failed: {errors}"
        # All connections returned
        assert engine.pool.checkedout() == 0
        engine.dispose()

    def test_concurrent_singleton_initialization(self):
        """Concurrent calls to get_engine and get_session_factory should not race or corrupt state."""
        reset_engine()
        results = []
        barrier = threading.Barrier(20)

        def init_task():
            barrier.wait()
            eng = get_engine("sqlite:///:memory:")
            fac = get_session_factory(eng)
            results.append((eng, fac))

        threads = [threading.Thread(target=init_task) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        first_engine, first_factory = results[0]
        for eng, fac in results:
            assert eng is first_engine
        reset_engine()

    def test_concurrent_rollback_and_commit_isolation(self, tmp_path: Path):
        """Verify concurrent failed transactions do not pollute or rollback successful concurrent transactions."""
        db_file = tmp_path / "rollback_isolation.db"
        engine = create_engine(
            f"sqlite:///{db_file.as_posix()}",
            connect_args={"check_same_thread": False, "timeout": 30},
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
        )
        init_db(engine)

        successful_commits = 0
        failed_attempts = 0
        lock = threading.Lock()

        def committer(idx: int):
            nonlocal successful_commits
            with get_db_session(engine) as session:
                create_backtest_log(
                    session=session,
                    ticker="^NSEI",
                    start_date="2023-01-01",
                    end_date="2023-12-31",
                    strategy=f"Success_{idx}",
                    initial_investment=100000.0,
                )
            with lock:
                successful_commits += 1

        def failer(idx: int):
            nonlocal failed_attempts
            try:
                with get_db_session(engine) as session:
                    create_backtest_log(
                        session=session,
                        ticker="^NSEI",
                        start_date="2023-01-01",
                        end_date="2023-12-31",
                        strategy=f"Fail_{idx}",
                        initial_investment=100000.0,
                    )
                    raise RuntimeError(f"Forced failure in worker {idx}")
            except RuntimeError:
                with lock:
                    failed_attempts += 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futs = []
            for i in range(20):
                if i % 2 == 0:
                    futs.append(executor.submit(committer, i))
                else:
                    futs.append(executor.submit(failer, i))
            concurrent.futures.wait(futs)

        assert successful_commits == 10
        assert failed_attempts == 10

        with get_db_session(engine) as session:
            logs = get_recent_backtests(session, limit=100)
            assert len(logs) == 10
            for l in logs:
                assert l.strategy.startswith("Success_")
                assert not l.strategy.startswith("Fail_")

        engine.dispose()


# ---------------------------------------------------------------------------
# 3. Path Creation & File System Probes
# ---------------------------------------------------------------------------

class TestSQLitePathCreationProbes:
    """Stress test path creation for deeply nested, query-parameterized, and edge SQLite paths."""

    def test_deeply_nested_sqlite_path_creation(self, tmp_path: Path):
        """Ensure directories 5 levels deep are auto-created."""
        nested_db = tmp_path / "a" / "b" / "c" / "d" / "e" / "deep.db"
        sqlite_url = f"sqlite:///{nested_db.as_posix()}"
        assert not nested_db.parent.exists()

        ensure_sqlite_directory_exists(sqlite_url)
        assert nested_db.parent.exists()
        assert nested_db.parent.is_dir()

        # Engine creation should succeed without manual mkdir
        engine = create_db_engine(sqlite_url)
        init_db(engine)
        with engine.connect() as conn:
            res = conn.execute(text("SELECT 101")).scalar()
            assert res == 101
        engine.dispose()
        assert nested_db.exists()

    def test_sqlite_url_with_query_parameters(self, tmp_path: Path):
        """Query parameters like ?check_same_thread=False must not be appended to directory paths."""
        target_dir = tmp_path / "query_test"
        db_file = target_dir / "test.db"
        sqlite_url = f"sqlite:///{db_file.as_posix()}?check_same_thread=False&timeout=10"

        ensure_sqlite_directory_exists(sqlite_url)
        assert target_dir.exists()
        assert not (tmp_path / "query_test?check_same_thread=False").exists()

    def test_special_sqlite_urls_no_directory_creation(self):
        """Special SQLite URLs (:memory:, empty) should not attempt file directory creation."""
        ensure_sqlite_directory_exists("sqlite:///:memory:")
        ensure_sqlite_directory_exists("sqlite:///")
        ensure_sqlite_directory_exists("sqlite://")
        ensure_sqlite_directory_exists("postgresql://localhost:5432/db")

    def test_relative_sqlite_path_resolution(self, tmp_path: Path, monkeypatch):
        """Relative SQLite URL creates directory under project root."""
        rel_url = "sqlite:///data/sub_test_dir/rel_portfolio.db"
        ensure_sqlite_directory_exists(rel_url)
        root = get_project_root()
        created_path = root / "data" / "sub_test_dir"
        assert created_path.exists()
        try:
            if created_path.exists():
                created_path.rmdir()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 4. URL Masking Probes
# ---------------------------------------------------------------------------

class TestDatabaseURLMaskingProbes:
    """Stress test URL masking across varied credential formats and character sets."""

    @pytest.mark.parametrize(
        ("raw_url", "expected_contains", "forbidden_contains"),
        [
            (
                "postgresql://admin:SecretPass123!@ep-cool-123.neon.tech/neondb",
                "admin:***@",
                "SecretPass123!",
            ),
            (
                "postgresql://readonly_user:SimplePass@localhost:5432/portfolio",
                "readonly_user:***@",
                "SimplePass",
            ),
            (
                "postgresql://app_user:p%40ssword@neon.tech:5432/db",
                "app_user:***@",
                "p%40ssword",
            ),
            (
                "postgresql://user:pass:with:colons@db.host.com:5432/main",
                "user:***@",
                "pass:with:colons",
            ),
            (
                "postgresql://user@db.host.com:5432/main",
                "postgresql://user@db.host.com:5432/main",
                "***",
            ),
            (
                "sqlite:///./data/portfolio_intel.db",
                "sqlite:///./data/portfolio_intel.db",
                "***",
            ),
            (
                "sqlite:///:memory:",
                "sqlite:///:memory:",
                "***",
            ),
            (
                "",
                "None",
                "***",
            ),
        ],
    )
    def test_mask_database_url_scenarios(self, raw_url: str, expected_contains: str, forbidden_contains: str):
        """mask_database_url must obscure passwords without crashing."""
        masked = mask_database_url(raw_url)
        assert expected_contains in masked
        if forbidden_contains != "***":
            assert forbidden_contains not in masked


# ---------------------------------------------------------------------------
# 5. Connection Healthcheck Probes
# ---------------------------------------------------------------------------

class TestConnectionHealthcheckProbes:
    """Stress test healthcheck behavior under normal, degraded, broken, and closed conditions."""

    def test_healthcheck_healthy_engine(self, tmp_path: Path):
        """Healthy engine returns valid health dictionary with pool metrics."""
        db_file = tmp_path / "health_ok.db"
        engine = create_db_engine(f"sqlite:///{db_file.as_posix()}")
        init_db(engine)

        health = check_connection_health(engine)
        assert health["status"] == "healthy"
        assert health["dialect"] == "sqlite"
        assert health["error"] is None
        assert isinstance(health["pool"], dict)
        assert "pool_size" in health["pool"]
        assert "checked_in" in health["pool"]
        assert "checked_out" in health["pool"]
        engine.dispose()

    def test_healthcheck_broken_engine_query_failure(self):
        """Engine that fails on SELECT 1 returns status unhealthy with error details."""
        mock_engine = MagicMock()
        mock_engine.dialect.name = "postgresql"
        mock_engine.url = "postgresql://user:pass@broken-host:5432/db"
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.execute.side_effect = OperationalError("connection lost", None, Exception("socket closed"))
        mock_engine.connect.return_value = mock_conn

        health = check_connection_health(mock_engine)
        assert health["status"] == "unhealthy"
        assert health["dialect"] == "postgresql"
        assert "connection lost" in health["error"]
        assert "pass" not in health["url_masked"]

    def test_healthcheck_pool_without_size_attribute(self):
        """Engine using a pool without .size() (e.g. NullPool) does not crash healthcheck."""
        engine = create_engine("sqlite:///:memory:", poolclass=NullPool)
        health = check_connection_health(engine)
        assert health["status"] == "healthy"
        assert health["dialect"] == "sqlite"
        assert health["pool"] == {}
        engine.dispose()

    def test_healthcheck_pool_attribute_exception_tolerance(self):
        """If pool.size() raises an exception, healthcheck degrades gracefully to empty pool dict."""
        mock_engine = MagicMock()
        mock_engine.dialect.name = "postgresql"
        mock_engine.url = "postgresql://user:secret@localhost:5432/test"
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_engine.connect.return_value = mock_conn

        # Pool whose methods throw
        mock_pool = MagicMock()
        mock_pool.size.side_effect = RuntimeError("Pool statistics unavailable")
        mock_engine.pool = mock_pool

        health = check_connection_health(mock_engine)
        assert health["status"] == "healthy"
        assert health["pool"] == {}

    def test_healthcheck_disposed_engine_reconnects_or_reports(self, tmp_path: Path):
        """Disposing an engine does not prevent subsequent healthcheck if DB file still exists."""
        db_file = tmp_path / "health_dispose.db"
        engine = create_db_engine(f"sqlite:///{db_file.as_posix()}")
        init_db(engine)
        engine.dispose()

        # In SQLAlchemy, connect() after dispose re-initializes connection pool
        health = check_connection_health(engine)
        assert health["status"] == "healthy"
        engine.dispose()

    def test_healthcheck_global_singleton_default(self):
        """Calling check_connection_health() without args uses global singleton engine."""
        health = check_connection_health()
        assert health["status"] == "healthy"
        assert health["dialect"] == "sqlite"
        assert health["error"] is None


# ---------------------------------------------------------------------------
# 6. Service Layer & Extreme Payload Stress Tests
# ---------------------------------------------------------------------------

class TestServiceExtremePayloadsAndSQLInjection:
    """Stress test background services with exotic payloads, extreme numbers, and SQL injection probes."""

    def test_save_analysis_record_massive_equity_curve(self, tmp_path: Path):
        """save_analysis_record handles 10,000 equity curve points without performance degradation."""
        db_file = tmp_path / "payload_stress.db"
        engine = create_db_engine(f"sqlite:///{db_file.as_posix()}")
        init_db(engine)

        large_equity_points = [100000.0 + (i * 1.5) for i in range(10000)]
        payload = {
            "ticker": "^NSEI",
            "start_date": "1990-01-01",
            "end_date": "2024-01-01",
            "strategy": "Dual Momentum",
            "initial_investment": 100000.0,
        }
        result = {
            "current_regime": "Bull",
            "recommended_strategy": "Dual Momentum",
            "recommendation_source": "XGBoostClassifier",
            "n_trading_days": 10000,
            "overall_metrics": {
                "Dual Momentum": {"CAGR": 0.21, "Sharpe": 1.75, "MaxDrawdown": -0.12},
            },
            "equity_curves": {
                "Dual Momentum": large_equity_points,
            },
            "regime_timeline": [
                {"regime": "Bull", "days": 6000},
                {"regime": "Bear", "days": 2000},
                {"regime": "Sideways", "days": 2000},
            ],
        }

        res = save_analysis_record(payload, result, engine=engine)
        assert res is not None
        assert "backtest_log_id" in res
        assert "regime_snapshot_id" in res

        with get_db_session(engine) as session:
            bt = get_backtest_by_id(session, res["backtest_log_id"])
            assert bt is not None
            # Equity curve summary extracts only summary start/end/points
            assert bt.equity_curve_summary["Dual Momentum"]["total_points"] == 10000
            assert bt.equity_curve_summary["Dual Momentum"]["start"] == 100000.0
            assert bt.equity_curve_summary["Dual Momentum"]["end"] == large_equity_points[-1]

            snap = get_latest_regime_snapshot(session, "^NSEI")
            assert snap is not None
            assert snap.total_trading_days == 10000
            assert snap.regime_distribution == {"Bull": 6000, "Bear": 2000, "Sideways": 2000}

        engine.dispose()

    def test_sql_injection_probe_in_ticker_and_strategy(self, tmp_path: Path):
        """Malicious SQL syntax in ticker and strategy fields must be safely parameterized."""
        db_file = tmp_path / "sqli_test.db"
        engine = create_db_engine(f"sqlite:///{db_file.as_posix()}")
        init_db(engine)

        malicious_ticker = "^NSEI'; DROP TABLE backtest_logs; --"
        malicious_strat = "1 OR 1=1; DELETE FROM regime_history;"

        with get_db_session(engine) as session:
            log = create_backtest_log(
                session=session,
                ticker=malicious_ticker,
                start_date="2020-01-01",
                end_date="2020-12-31",
                strategy=malicious_strat,
                initial_investment=100000.0,
            )
            created_id = log.id

        # Query using the malicious string as exact parameter
        with get_db_session(engine) as session:
            retrieved = get_recent_backtests(session, ticker=malicious_ticker)
            assert len(retrieved) == 1
            assert retrieved[0].id == created_id
            assert retrieved[0].ticker == malicious_ticker

            # Verify tables still exist and were not dropped
            count = count_backtests(session)
            assert count == 1
            snap_count = count_regime_snapshots(session)
            assert snap_count == 0

        engine.dispose()

    def test_save_backtest_record_with_corrupt_items_in_list(self, tmp_path: Path):
        """save_backtest_record skips non-dict or malformed elements without failing the entire batch."""
        db_file = tmp_path / "corrupt_batch.db"
        engine = create_db_engine(f"sqlite:///{db_file.as_posix()}")
        init_db(engine)

        payload = {"ticker": "^CNXIT", "start_date": "2021", "end_date": "2022"}
        result = {
            "results": [
                None,  # corrupt
                "invalid_string_item",  # corrupt
                {"strategy": "ValidStrategy1", "metrics": {"CAGR": 0.15}},
                12345,  # corrupt
                {"strategy": "ValidStrategy2", "metrics": {"CAGR": 0.18}},
            ]
        }

        saved_ids = save_backtest_record(payload, result, engine=engine)
        assert len(saved_ids) == 2

        with get_db_session(engine) as session:
            logs = get_recent_backtests(session, ticker="^CNXIT")
            assert len(logs) == 2
            strats = {l.strategy for l in logs}
            assert strats == {"ValidStrategy1", "ValidStrategy2"}

        engine.dispose()

    def test_save_benchmark_record_with_corrupt_models_list(self, tmp_path: Path):
        """save_benchmark_record filters out corrupt model items cleanly."""
        db_file = tmp_path / "corrupt_bench.db"
        engine = create_db_engine(f"sqlite:///{db_file.as_posix()}")
        init_db(engine)

        benchmark_data = {
            "ticker": "^NSEI",
            "evaluation_date": "2024-01-01",
            "models": [
                None,
                {"model_name": "XGBoost", "accuracy": 0.82, "f1_score": 0.81},
                {"corrupt_key_no_name": 123},
                {"model_name": "LSTM-DNN", "accuracy": 0.79, "f1_score": 0.78},
            ],
        }

        saved_ids = save_benchmark_record(benchmark_data, engine=engine)
        assert len(saved_ids) == 3  # XGBoost, corrupt_key (defaults to 'Unknown'), LSTM-DNN

        with get_db_session(engine) as session:
            history = get_benchmark_history(session, ticker="^NSEI")
            assert len(history) == 3

        engine.dispose()

    def test_orm_models_to_dict_complete_roundtrip(self, tmp_path: Path):
        """Verify all ORM models convert to dictionary with correct JSON serializable types."""
        db_file = tmp_path / "dict_test.db"
        engine = create_db_engine(f"sqlite:///{db_file.as_posix()}")
        init_db(engine)

        with get_db_session(engine) as session:
            bt = create_backtest_log(
                session=session,
                ticker="^NSEI",
                start_date="2020-01-01",
                end_date="2020-12-31",
                strategy="SMA",
                initial_investment=100000.0,
                metrics_json={"metric1": 123},
            )
            reg = create_regime_snapshot(
                session=session,
                ticker="^NSEI",
                as_of_date="2020-12-31",
                current_regime="Bull",
                regime_distribution={"Bull": 200},
                total_trading_days=200,
            )
            bm = create_benchmark_run(
                session=session,
                ticker="^NSEI",
                model_name="XGBoost",
                accuracy=0.85,
                f1_score=0.84,
            )

        with get_db_session(engine) as session:
            bt_dict = get_backtest_by_id(session, bt.id).to_dict()
            assert isinstance(bt_dict["created_at"], str)
            assert bt_dict["ticker"] == "^NSEI"

            reg_dict = get_latest_regime_snapshot(session, "^NSEI").to_dict()
            assert isinstance(reg_dict["created_at"], str)
            assert reg_dict["regime_distribution"] == {"Bull": 200}

            bm_dict = get_latest_benchmark_runs(session, "^NSEI")[0].to_dict()
            assert isinstance(bm_dict["created_at"], str)
            assert bm_dict["accuracy"] == 0.85

        engine.dispose()
