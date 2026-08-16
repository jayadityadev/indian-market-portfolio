"""Comprehensive unit and integration test suite for the database persistence layer (src/db/).

Covers:
- Connection pooling, URL normalization, masked logging, and resilient SQLite fallback.
- Session lifecycle, FastAPI get_db dependency, and get_db_session context manager.
- SQLAlchemy 2.0 ORM models (BacktestLog, RegimeSnapshot, ModelBenchmarkRun).
- CRUD operations (create, read, update, delete, count, pagination, filtering).
- Background execution recording services (save_analysis_record, save_backtest_record, save_benchmark_record).
- Mocked Neon PostgreSQL pool configuration and connection failover.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import StaticPool

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
    utc_now,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_engine(tmp_path: Path) -> Engine:
    """Provide a fresh file-based SQLite database engine for test isolation."""
    db_file = tmp_path / "test_portfolio.db"
    url = f"sqlite:///{db_file.as_posix()}"
    engine = create_db_engine(db_url=url, echo=False)
    init_db(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine: Engine):
    """Provide an isolated database session with auto-rollback."""
    with get_db_session(db_engine) as session:
        yield session


# ---------------------------------------------------------------------------
# 1. TestDatabaseConnection
# ---------------------------------------------------------------------------

class TestDatabaseConnection:
    """Test suite for database connection, pooling, URL normalization, and fallback."""

    def test_normalize_database_url_postgres_prefix(self):
        """postgres:// prefix must be converted to postgresql:// for SQLAlchemy 2.0."""
        raw_url = "postgres://user:secret@ep-cool-serverless.us-east-2.aws.neon.tech/neondb?sslmode=require"
        normalized = normalize_database_url(raw_url)
        assert normalized.startswith("postgresql://")
        assert "user:secret@ep-cool-serverless" in normalized

    def test_normalize_database_url_empty_and_whitespace(self):
        """Empty or whitespace URL must fallback to default SQLite path."""
        assert normalize_database_url(None) == get_default_sqlite_url()
        assert normalize_database_url("") == get_default_sqlite_url()
        assert normalize_database_url("   ") == get_default_sqlite_url()

    def test_normalize_database_url_valid_sqlite(self):
        """Valid SQLite URLs must remain unchanged."""
        sqlite_url = "sqlite:///:memory:"
        assert normalize_database_url(sqlite_url) == sqlite_url

    def test_mask_database_url(self):
        """Sensitive credentials in connection string must be masked."""
        url = "postgresql://myuser:supersecretpassword@neon.tech:5432/mydb"
        masked = mask_database_url(url)
        assert "supersecretpassword" not in masked
        assert "myuser:***@" in masked
        assert mask_database_url("") == "None"

    def test_ensure_sqlite_directory_exists(self, tmp_path: Path):
        """Missing parent directory for file SQLite URL must be auto-created."""
        nested_dir = tmp_path / "nested" / "subfolder"
        db_path = nested_dir / "test.db"
        assert not nested_dir.exists()
        sqlite_url = f"sqlite:///{db_path.as_posix()}"
        ensure_sqlite_directory_exists(sqlite_url)
        assert nested_dir.exists()

    def test_sqlite_in_memory_engine_creation(self):
        """In-memory SQLite engine must use StaticPool and enable multithreading."""
        engine = create_db_engine(db_url="sqlite:///:memory:", echo=False)
        assert isinstance(engine.pool, StaticPool)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 42")).scalar()
            assert result == 42
        engine.dispose()

    def test_engine_singleton_and_reset(self):
        """get_engine returns the same singleton until reset_engine is called."""
        reset_engine()
        eng1 = get_engine("sqlite:///:memory:")
        eng2 = get_engine()
        assert eng1 is eng2
        reset_engine()
        eng3 = get_engine("sqlite:///:memory:")
        assert eng3 is not eng1
        reset_engine()

    def test_check_connection_health_healthy_sqlite(self, db_engine: Engine):
        """Healthy database returns status 'healthy' with dialect and masked URL."""
        health = check_connection_health(db_engine)
        assert health["status"] == "healthy"
        assert health["dialect"] == "sqlite"
        assert health["error"] is None
        assert "url_masked" in health

    def test_check_connection_health_unhealthy(self):
        """Broken or disposed engine returns status 'unhealthy' with error details."""
        mock_engine = MagicMock()
        mock_engine.dialect.name = "postgresql"
        mock_engine.url = "postgresql://user:pass@localhost:5432/bad_db"
        mock_engine.connect.side_effect = Exception("Connection refused by peer")
        health = check_connection_health(mock_engine)
        assert health["status"] == "unhealthy"
        assert health["dialect"] == "postgresql"
        assert "Connection refused" in health["error"]

    def test_neon_postgres_success_and_pool_config(self):
        """PostgreSQL engine creation configures pool_pre_ping, pool_recycle, and pool sizing."""
        with patch("src.db.connection.create_engine") as mock_create_engine:
            mock_pg_engine = MagicMock(spec=Engine)
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_pg_engine.connect.return_value = mock_conn
            mock_create_engine.return_value = mock_pg_engine

            pg_url = "postgresql://user:pass@ep-shiny-123.neon.tech/neondb?sslmode=require"
            engine = create_db_engine(db_url=pg_url, echo=False)

            assert engine is mock_pg_engine
            mock_create_engine.assert_called_once_with(
                pg_url,
                echo=False,
                pool_pre_ping=True,
                pool_recycle=300,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                connect_args={"connect_timeout": 5},
            )

    def test_neon_postgres_failover_to_sqlite(self, tmp_path: Path):
        """When PostgreSQL connection ping fails, engine automatically falls back to SQLite."""
        with patch("src.db.connection.create_engine") as mock_create_engine:
            # First call for postgres fails
            mock_pg_engine = MagicMock()
            mock_pg_engine.connect.side_effect = SQLAlchemyError("Network unreachable")

            # Second call for fallback succeeds with real engine
            real_sqlite = create_db_engine("sqlite:///:memory:")

            def side_effect(url, **kwargs):
                if url.startswith("postgresql"):
                    return mock_pg_engine
                return real_sqlite

            mock_create_engine.side_effect = side_effect

            pg_url = "postgresql://user:pass@bad-host.neon.tech/neondb"
            engine = create_db_engine(db_url=pg_url, force_sqlite_fallback=True)
            assert engine is real_sqlite
            real_sqlite.dispose()


# ---------------------------------------------------------------------------
# 2. TestDatabaseSession
# ---------------------------------------------------------------------------

class TestDatabaseSession:
    """Test suite for sessionmaker, SessionLocal, get_db, and get_db_session."""

    def test_get_session_factory_and_session_local(self, db_engine: Engine):
        """SessionLocal produces valid active sessions bound to the target engine."""
        factory = get_session_factory(db_engine)
        assert factory is not None
        session = SessionLocal(db_engine)
        assert session.bind is db_engine
        session.close()

    def test_get_db_fastapi_dependency_lifecycle(self, db_engine: Engine):
        """get_db generator yields active session and closes it on generator exit."""
        gen = get_db(db_engine)
        session = next(gen)
        assert session.is_active
        with pytest.raises(StopIteration):
            next(gen)
        # Session should now be closed / inactive
        assert not session.is_active or session.get_transaction() is None

    def test_get_db_session_context_manager_commits(self, db_engine: Engine):
        """get_db_session commits database transaction on clean context exit."""
        with get_db_session(db_engine) as session:
            log = create_backtest_log(
                session=session,
                ticker="^NSEI",
                start_date="2022-01-01",
                end_date="2022-12-31",
                strategy="Momentum",
                initial_investment=100000.0,
            )
            created_id = log.id

        # Verify persisted in a new separate session
        with get_db_session(db_engine) as session:
            retrieved = get_backtest_by_id(session, created_id)
            assert retrieved is not None
            assert retrieved.strategy == "Momentum"

    def test_get_db_session_context_manager_rolls_back_on_error(self, db_engine: Engine):
        """get_db_session rolls back database transaction when an exception is raised."""
        with pytest.raises(ValueError, match="Intentional failure"):
            with get_db_session(db_engine) as session:
                create_backtest_log(
                    session=session,
                    ticker="^NSEI",
                    start_date="2022-01-01",
                    end_date="2022-12-31",
                    strategy="ShouldRollBack",
                    initial_investment=100000.0,
                )
                raise ValueError("Intentional failure")

        # Verify record was NOT saved
        with get_db_session(db_engine) as session:
            logs = get_recent_backtests(session, strategy="ShouldRollBack")
            assert len(logs) == 0


# ---------------------------------------------------------------------------
# 3. TestORMModels
# ---------------------------------------------------------------------------

class TestORMModels:
    """Test suite for DeclarativeBase models, fields, and JSON serialization."""

    def test_init_db_creates_all_tables(self, db_engine: Engine):
        """init_db creates all required tables and indexes."""
        inspector = inspect(db_engine)
        table_names = set(inspector.get_table_names())
        assert "backtest_logs" in table_names
        assert "regime_history" in table_names
        assert "model_benchmarks" in table_names

    def test_backtest_log_model_fields_and_to_dict(self, db_session):
        """BacktestLog model supports all financial metrics, JSON columns, and to_dict."""
        log = create_backtest_log(
            session=db_session,
            ticker="^NSEI",
            start_date="2021-01-01",
            end_date="2021-12-31",
            strategy="Dual Momentum",
            initial_investment=200000.0,
            commission_pct=0.001,
            slippage_pct=0.0005,
            cagr=0.185,
            sharpe=1.45,
            sortino=2.10,
            max_drawdown=-0.085,
            calmar=2.17,
            volatility=0.128,
            recommended_strategy="Dual Momentum",
            recommendation_source="XGBoostClassifier",
            current_regime="Bull",
            metrics_json={"Dual Momentum": {"Sharpe": 1.45, "CAGR": 0.185}},
            equity_curve_summary={"start": 200000.0, "end": 237000.0, "points": 250},
            notes="Annual backtest run",
        )
        assert log.id is not None
        assert "BacktestLog" in repr(log)
        d = log.to_dict()
        assert d["id"] == log.id
        assert d["ticker"] == "^NSEI"
        assert d["cagr"] == 0.185
        assert d["metrics_json"]["Dual Momentum"]["Sharpe"] == 1.45
        assert d["equity_curve_summary"]["end"] == 237000.0

    def test_regime_snapshot_model_fields_and_to_dict(self, db_session):
        """RegimeSnapshot stores HMM distribution, transition matrix, and to_dict."""
        snapshot = create_regime_snapshot(
            session=db_session,
            ticker="^NSEI",
            as_of_date="2024-06-30",
            current_regime="Bull",
            regime_distribution={"Bull": 120, "Sideways": 80, "Bear": 40},
            total_trading_days=240,
            transition_matrix=[[0.85, 0.10, 0.05], [0.12, 0.80, 0.08], [0.10, 0.15, 0.75]],
            recent_segments=[{"regime": "Bull", "days": 30}],
            stationary_distribution=[0.50, 0.30, 0.20],
            state_posteriors=[0.88, 0.09, 0.03],
        )
        assert snapshot.id is not None
        assert "RegimeSnapshot" in repr(snapshot)
        d = snapshot.to_dict()
        assert d["current_regime"] == "Bull"
        assert d["total_trading_days"] == 240
        assert d["transition_matrix"][0][0] == 0.85
        assert d["stationary_distribution"] == [0.50, 0.30, 0.20]

    def test_model_benchmark_run_fields_and_to_dict(self, db_session):
        """ModelBenchmarkRun stores precision, recall, F1, latency, and to_dict."""
        run = create_benchmark_run(
            session=db_session,
            ticker="^NSEI",
            model_name="LSTM-DNN",
            train_window_start="2018-01-01",
            train_window_end="2022-12-31",
            test_window_start="2023-01-01",
            test_window_end="2023-12-31",
            accuracy=0.745,
            precision=0.730,
            recall=0.760,
            f1_score=0.744,
            roc_auc=0.812,
            train_accuracy=0.780,
            overfitting_gap=0.035,
            training_time_sec=14.2,
            inference_latency_ms=1.8,
            architecture_summary="2x LSTM (64, 32) + 4x Dense",
            details={"confusion_matrix": [[45, 10], [8, 52]]},
        )
        assert run.id is not None
        assert "ModelBenchmarkRun" in repr(run)
        d = run.to_dict()
        assert d["model_name"] == "LSTM-DNN"
        assert d["f1_score"] == 0.744
        assert d["overfitting_gap"] == 0.035
        assert d["details"]["confusion_matrix"][0][0] == 45


# ---------------------------------------------------------------------------
# 4. TestCRUDOperations
# ---------------------------------------------------------------------------

class TestCRUDOperations:
    """Test suite for CRUD querying, filtering, pagination, and deletion."""

    def test_create_and_get_backtest_log(self, db_session):
        """create_backtest_log and get_backtest_by_id retrieve correct record."""
        bt = create_backtest_log(
            session=db_session,
            ticker="^NSEBANK",
            start_date="2020-01-01",
            end_date="2020-12-31",
            strategy="RSI",
            initial_investment=100000.0,
        )
        retrieved = get_backtest_by_id(db_session, bt.id)
        assert retrieved is not None
        assert retrieved.id == bt.id
        assert retrieved.ticker == "^NSEBANK"
        assert retrieved.strategy == "RSI"

        assert get_backtest_by_id(db_session, 999999) is None

    def test_get_recent_backtests_with_filtering_and_pagination(self, db_session):
        """get_recent_backtests correctly applies ticker, strategy filters, and limit/offset."""
        for i in range(10):
            create_backtest_log(
                session=db_session,
                ticker="^NSEI" if i % 2 == 0 else "^CNXIT",
                start_date="2020-01-01",
                end_date="2020-12-31",
                strategy="Momentum" if i < 5 else "Bollinger",
                initial_investment=100000.0,
            )

        all_logs = get_recent_backtests(db_session, limit=100)
        assert len(all_logs) == 10

        # Filter by ticker
        nsei_logs = get_recent_backtests(db_session, ticker="^NSEI")
        assert len(nsei_logs) == 5
        assert all(l.ticker == "^NSEI" for l in nsei_logs)

        # Filter by strategy
        mom_logs = get_recent_backtests(db_session, strategy="Momentum")
        assert len(mom_logs) == 5
        assert all(l.strategy == "Momentum" for l in mom_logs)

        # Pagination
        page1 = get_recent_backtests(db_session, limit=3, offset=0)
        page2 = get_recent_backtests(db_session, limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].id != page2[0].id

    def test_count_and_delete_backtest_log(self, db_session):
        """count_backtests and delete_backtest_by_id work as expected."""
        bt1 = create_backtest_log(session=db_session, ticker="^NSEI", start_date="2020", end_date="2021", strategy="SMA")
        bt2 = create_backtest_log(session=db_session, ticker="^NSEI", start_date="2021", end_date="2022", strategy="EMA")

        assert count_backtests(db_session, ticker="^NSEI") == 2
        assert count_backtests(db_session, strategy="SMA") == 1

        deleted = delete_backtest_by_id(db_session, bt1.id)
        assert deleted is True
        assert count_backtests(db_session, ticker="^NSEI") == 1
        assert get_backtest_by_id(db_session, bt1.id) is None

        # Deleting non-existent returns False
        assert delete_backtest_by_id(db_session, 99999) is False

    def test_create_and_get_latest_regime_snapshot(self, db_session):
        """get_latest_regime_snapshot returns the most recent record for the ticker."""
        create_regime_snapshot(
            session=db_session,
            ticker="^NSEI",
            as_of_date="2023-01-01",
            current_regime="Bear",
            regime_distribution={"Bear": 100},
            total_trading_days=100,
        )
        create_regime_snapshot(
            session=db_session,
            ticker="^NSEI",
            as_of_date="2023-06-01",
            current_regime="Bull",
            regime_distribution={"Bull": 100},
            total_trading_days=200,
        )

        latest = get_latest_regime_snapshot(db_session, ticker="^NSEI")
        assert latest is not None
        assert latest.current_regime == "Bull"
        assert latest.as_of_date == "2023-06-01"

        assert get_latest_regime_snapshot(db_session, ticker="^NONEXISTENT") is None

    def test_get_recent_regime_snapshots_and_count(self, db_session):
        """get_recent_regime_snapshots and count_regime_snapshots support pagination and filtering."""
        for i in range(5):
            create_regime_snapshot(
                session=db_session,
                ticker="^NSEI",
                as_of_date=f"2023-0{i+1}-01",
                current_regime="Bull" if i % 2 == 0 else "Sideways",
                regime_distribution={"Bull": 50},
                total_trading_days=50 * (i + 1),
            )

        assert count_regime_snapshots(db_session, ticker="^NSEI") == 5
        snapshots = get_recent_regime_snapshots(db_session, ticker="^NSEI", limit=2, offset=0)
        assert len(snapshots) == 2

    def test_create_benchmark_run_and_batch(self, db_session):
        """create_benchmark_batch inserts multiple model benchmark runs atomically."""
        runs_data = [
            {
                "ticker": "^NSEI",
                "model_name": "XGBoost",
                "accuracy": 0.78,
                "f1_score": 0.77,
            },
            {
                "ticker": "^NSEI",
                "model_name": "LSTM-DNN",
                "accuracy": 0.75,
                "f1_score": 0.74,
            },
            {
                "ticker": "^NSEI",
                "model_name": "RandomForest",
                "accuracy": 0.71,
                "f1_score": 0.70,
            },
        ]
        created = create_benchmark_batch(db_session, runs_data)
        assert len(created) == 3
        assert created[0].model_name == "XGBoost"
        assert created[1].model_name == "LSTM-DNN"
        assert created[2].model_name == "RandomForest"

    def test_get_latest_benchmark_runs_distinct_models(self, db_session):
        """get_latest_benchmark_runs returns one latest record per unique model."""
        # Insert older run
        create_benchmark_run(
            session=db_session,
            ticker="^NSEI",
            model_name="XGBoost",
            accuracy=0.70,
            f1_score=0.69,
        )
        # Insert newer runs
        create_benchmark_run(
            session=db_session,
            ticker="^NSEI",
            model_name="XGBoost",
            accuracy=0.82,
            f1_score=0.81,
        )
        create_benchmark_run(
            session=db_session,
            ticker="^NSEI",
            model_name="LSTM-DNN",
            accuracy=0.79,
            f1_score=0.78,
        )

        latest = get_latest_benchmark_runs(db_session, ticker="^NSEI")
        assert len(latest) == 2
        model_names = {r.model_name for r in latest}
        assert model_names == {"XGBoost", "LSTM-DNN"}
        xgb_run = next(r for r in latest if r.model_name == "XGBoost")
        assert xgb_run.accuracy == 0.82

    def test_get_benchmark_history_filtering_and_count(self, db_session):
        """get_benchmark_history filters by model name and ticker with count."""
        create_benchmark_run(session=db_session, ticker="^NSEI", model_name="XGBoost", accuracy=0.8, f1_score=0.8)
        create_benchmark_run(session=db_session, ticker="^NSEI", model_name="LSTM-DNN", accuracy=0.7, f1_score=0.7)
        create_benchmark_run(session=db_session, ticker="^NSEBANK", model_name="XGBoost", accuracy=0.85, f1_score=0.84)

        assert count_benchmarks(db_session, model_name="XGBoost") == 2
        assert count_benchmarks(db_session, ticker="^NSEI") == 2
        assert count_benchmarks(db_session, model_name="XGBoost", ticker="^NSEI") == 1

        history = get_benchmark_history(db_session, model_name="XGBoost", ticker="^NSEI")
        assert len(history) == 1
        assert history[0].accuracy == 0.8


# ---------------------------------------------------------------------------
# 5. TestDatabaseServiceAndBackgroundTasks
# ---------------------------------------------------------------------------

class TestDatabaseServiceAndBackgroundTasks:
    """Test suite for background recording services and exception safety."""

    def test_save_analysis_record_full_payload(self, db_engine: Engine):
        """save_analysis_record extracts and persists BacktestLog and RegimeSnapshot."""
        payload = {
            "ticker": "^NSEI",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "strategy": "all",
            "initial_investment": 150000.0,
            "commission_pct": 0.001,
            "slippage_pct": 0.0005,
        }
        result = {
            "ticker": "^NSEI",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "current_regime": "Bull",
            "recommended_strategy": "Momentum",
            "recommendation_source": "XGBoost",
            "n_trading_days": 250,
            "overall_metrics": {
                "Momentum": {
                    "CAGR": 0.22,
                    "Sharpe": 1.65,
                    "Sortino": 2.30,
                    "MaxDrawdown": -0.09,
                    "Calmar": 2.44,
                    "Volatility": 0.13,
                },
                "Buy & Hold": {
                    "CAGR": 0.15,
                    "Sharpe": 1.10,
                },
            },
            "equity_curves": {
                "Momentum": [150000.0, 160000.0, 183000.0],
            },
            "regime_timeline": [
                {"regime": "Sideways", "days": 50},
                {"regime": "Bull", "days": 200},
            ],
        }

        res = save_analysis_record(payload, result, notes="E2E Analysis Run", engine=db_engine)
        assert res is not None
        assert "backtest_log_id" in res
        assert "regime_snapshot_id" in res

        with get_db_session(db_engine) as session:
            bt = get_backtest_by_id(session, res["backtest_log_id"])
            assert bt is not None
            assert bt.ticker == "^NSEI"
            assert bt.strategy == "all"
            assert bt.recommended_strategy == "Momentum"
            assert bt.sharpe == 1.65
            assert bt.notes == "E2E Analysis Run"

            snap = get_latest_regime_snapshot(session, "^NSEI")
            assert snap is not None
            assert snap.id == res["regime_snapshot_id"]
            assert snap.current_regime == "Bull"
            assert snap.regime_distribution == {"Sideways": 50, "Bull": 200}
            assert snap.total_trading_days == 250

    def test_save_analysis_record_pydantic_object(self, db_engine: Engine):
        """save_analysis_record supports Pydantic models with model_dump."""
        class MockPydanticModel:
            def __init__(self, data: dict):
                self._data = data
            def model_dump(self):
                return self._data

        req = MockPydanticModel({"ticker": "^NSEBANK", "start_date": "2023-01-01", "end_date": "2023-12-31"})
        res = MockPydanticModel({"current_regime": "Sideways", "n_trading_days": 100, "overall_metrics": {}})

        result = save_analysis_record(req, res, engine=db_engine)
        assert result is not None
        assert "backtest_log_id" in result

    def test_save_analysis_record_resilience_on_invalid_data(self):
        """save_analysis_record handles malformed data or broken engine gracefully without throwing."""
        mock_broken_engine = MagicMock(spec=Engine)
        mock_broken_engine.connect.side_effect = Exception("DB down")

        res = save_analysis_record({"ticker": "^NSEI"}, {"invalid": "data"}, engine=mock_broken_engine)
        assert res is None  # Does not crash background worker

    def test_save_backtest_record_multiple_strategies(self, db_engine: Engine):
        """save_backtest_record persists logs for all strategy results."""
        payload = {
            "ticker": "^NSEI",
            "start_date": "2022-01-01",
            "end_date": "2022-12-31",
            "initial_investment": 100000.0,
            "commission_pct": 0.0,
            "slippage_pct": 0.0,
        }
        result = {
            "results": [
                {
                    "strategy": "Buy & Hold",
                    "metrics": {"CAGR": 0.12, "Sharpe": 1.05, "MaxDrawdown": -0.15},
                    "equity_curve_start": 100000.0,
                    "equity_curve_end": 112000.0,
                    "n_days": 252,
                },
                {
                    "strategy": "Momentum",
                    "metrics": {"CAGR": 0.20, "Sharpe": 1.55, "MaxDrawdown": -0.08},
                    "equity_curve_start": 100000.0,
                    "equity_curve_end": 120000.0,
                    "n_days": 252,
                },
            ]
        }

        saved_ids = save_backtest_record(payload, result, engine=db_engine)
        assert len(saved_ids) == 2

        with get_db_session(db_engine) as session:
            logs = get_recent_backtests(session, ticker="^NSEI")
            assert len(logs) == 2
            strats = {l.strategy for l in logs}
            assert strats == {"Buy & Hold", "Momentum"}

    def test_save_backtest_record_resilience_on_invalid_data(self):
        """save_backtest_record catches exceptions and returns empty list."""
        mock_broken_engine = MagicMock(spec=Engine)
        mock_broken_engine.connect.side_effect = Exception("DB connection timeout")
        saved_ids = save_backtest_record({"ticker": "^NSEI"}, {"results": "not-a-list"}, engine=mock_broken_engine)
        assert saved_ids == []

    def test_save_benchmark_record_batch(self, db_engine: Engine):
        """save_benchmark_record extracts model cards and saves batch records."""
        benchmark_response = {
            "ticker": "^NSEI",
            "evaluation_date": "2024-06-30",
            "models": [
                {
                    "model_name": "XGBoostClassifier",
                    "accuracy": 0.81,
                    "precision": 0.80,
                    "recall": 0.82,
                    "f1_score": 0.81,
                    "roc_auc": 0.88,
                    "train_accuracy": 0.84,
                    "overfitting_gap": 0.03,
                    "training_time_sec": 0.45,
                    "inference_latency_ms": 0.12,
                    "architecture_summary": "Gradient Boosted Trees (n_estimators=100, max_depth=4)",
                    "details": {"hyperparams": {"learning_rate": 0.05}},
                },
                {
                    "model_name": "PyTorch LSTM-DNN",
                    "accuracy": 0.78,
                    "precision": 0.77,
                    "recall": 0.79,
                    "f1_score": 0.78,
                    "roc_auc": 0.85,
                    "train_accuracy": 0.83,
                    "overfitting_gap": 0.05,
                    "training_time_sec": 12.8,
                    "inference_latency_ms": 1.45,
                    "architecture_summary": "2x LSTM + 4x Dense (Alam et al. 2024)",
                    "details": {"epochs": 50, "batch_size": 32},
                },
            ],
        }

        saved_ids = save_benchmark_record(benchmark_response, engine=db_engine)
        assert len(saved_ids) == 2

        with get_db_session(db_engine) as session:
            history = get_benchmark_history(session, ticker="^NSEI")
            assert len(history) == 2
            lstm_run = next(r for r in history if "LSTM" in r.model_name)
            assert lstm_run.f1_score == 0.78
            assert lstm_run.training_time_sec == 12.8

    def test_save_benchmark_record_resilience_on_invalid_data(self):
        """save_benchmark_record catches exceptions and returns empty list."""
        mock_broken_engine = MagicMock(spec=Engine)
        mock_broken_engine.connect.side_effect = Exception("DB error")
        saved_ids = save_benchmark_record({"models": "invalid"}, engine=mock_broken_engine)
        assert saved_ids == []
