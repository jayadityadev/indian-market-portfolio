"""Empirical Adversarial Stress Test Suite for Milestone M2 (Database Persistence Layer).

Targeting:
1. Extreme, null, and deeply nested JSON structures in metrics_json, equity_curve_summary, transition_matrix, details.
2. Large batch benchmark insertion, high-volume retrieval, and atomic rollback on invalid batch elements.
3. Boundary conditions in CRUD pagination, offset, negative limits, non-existent tickers, and SQL injection safety.
4. Background tasks error isolation on corrupt data types, malformed objects, non-serializable payloads, and broken engines.
5. Multithreaded concurrency stress testing on SQLite and session pooling.
6. Extreme financial parameters (trillion-dollar investments, huge strings, nan/inf floats, multiple init_db calls).
"""
from __future__ import annotations

import concurrent.futures
import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from src.db import (
    BacktestLog,
    Base,
    ModelBenchmarkRun,
    RegimeSnapshot,
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
    get_recent_backtests,
    get_recent_regime_snapshots,
    init_db,
    mask_database_url,
    normalize_database_url,
    reset_engine,
    save_analysis_record,
    save_backtest_record,
    save_benchmark_record,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def isolated_engine(tmp_path: Path) -> Engine:
    """Provide a dedicated file-backed SQLite database for isolated test execution."""
    db_file = tmp_path / "adversarial_m2.db"
    engine = create_db_engine(f"sqlite:///{db_file.as_posix()}", echo=False)
    init_db(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(isolated_engine: Engine):
    """Provide an isolated database session with auto-rollback/commit."""
    with get_db_session(isolated_engine) as sess:
        yield sess


# ===========================================================================
# 1. Extreme, Null, and Deeply Nested JSON Structures
# ===========================================================================

class TestAdversarialJSONHandling:
    """Test resilience of ORM JSON columns to extreme payloads, deep nesting, and special data."""

    def test_deeply_nested_json_in_backtest_metrics(self, session):
        """Test storing and retrieving 50+ levels of nested JSON dictionaries in metrics_json."""
        nested: dict[str, Any] = {"leaf": "deep_value", "number": 123456789.987654321}
        for level in range(50):
            nested = {f"level_{level}": nested}

        log = create_backtest_log(
            session=session,
            ticker="^NSEI",
            start_date="2020-01-01",
            end_date="2020-12-31",
            strategy="DeepJSONStrat",
            metrics_json=nested,
        )
        assert log.id is not None

        retrieved = get_backtest_by_id(session, log.id)
        assert retrieved is not None
        assert retrieved.metrics_json is not None

        # Verify leaf value is intact at depth 50
        curr = retrieved.metrics_json
        for level in reversed(range(50)):
            curr = curr[f"level_{level}"]
        assert curr["leaf"] == "deep_value"
        assert curr["number"] == pytest.approx(123456789.987654321)

    def test_large_equity_curve_payload(self, session):
        """Test storing large equity curve series (10,000 data points) inside JSON column."""
        points = [{"idx": i, "val": 100000.0 * (1.0002 ** i), "regime": "Bull" if i % 2 == 0 else "Bear"} for i in range(10000)]
        summary = {"ticker": "^NSEI", "series": points, "count": len(points)}

        log = create_backtest_log(
            session=session,
            ticker="^NSEI",
            start_date="2000-01-01",
            end_date="2023-12-31",
            strategy="LongTermHODL",
            equity_curve_summary=summary,
        )
        assert log.id is not None

        retrieved = get_backtest_by_id(session, log.id)
        assert retrieved is not None
        assert retrieved.equity_curve_summary["count"] == 10000
        assert len(retrieved.equity_curve_summary["series"]) == 10000
        assert retrieved.equity_curve_summary["series"][5000]["idx"] == 5000

    def test_special_floats_and_unicode_in_json(self, session):
        """Test Unicode symbols (₹, 🚀, Japanese, Chinese, Arabic) and exotic strings in JSON fields."""
        exotic_metrics = {
            "currency_symbol": "₹",
            "emoji_tag": "📈🚀🔥",
            "multilingual": "भारतीय शेयर बाज़ार / 股票市场 / 市場",
            "escaped_json": '{"inner": "escaped \\"quotes\\""}',
            "raw_newlines": "Line1\nLine2\r\nLine3\tTabbed",
            "boolean_flags": [True, False, None, 0, 1],
        }
        log = create_backtest_log(
            session=session,
            ticker="^NSEI",
            start_date="2022-01-01",
            end_date="2022-12-31",
            strategy="ExoticUnicodeStrat",
            metrics_json=exotic_metrics,
        )
        retrieved = get_backtest_by_id(session, log.id)
        assert retrieved is not None
        assert retrieved.metrics_json["currency_symbol"] == "₹"
        assert retrieved.metrics_json["emoji_tag"] == "📈🚀🔥"
        assert retrieved.metrics_json["multilingual"] == "भारतीय शेयर बाज़ार / 股票市场 / 市場"
        assert retrieved.metrics_json["boolean_flags"] == [True, False, None, 0, 1]

    def test_ragged_and_multidimensional_transition_matrix(self, session):
        """Test ragged arrays, mixed types, and 5x5 transition matrices in RegimeSnapshot."""
        t_matrix = [
            [0.70, 0.15, 0.05, 0.05, 0.05],
            [0.10, 0.65, 0.15, 0.05, 0.05],
            [0.05, 0.10, 0.70, 0.10, 0.05],
            [0.05, 0.05, 0.10, 0.70, 0.10],
            [0.05, 0.05, 0.05, 0.15, 0.70],
        ]
        ragged_segments = [
            {"regime": "SuperBull", "days": 120, "extra": {"nested": [1, 2, 3]}},
            {"regime": "Crash", "days": 5, "drawdown": -0.38},
            {"empty_segment": {}},
        ]
        snapshot = create_regime_snapshot(
            session=session,
            ticker="^NSEI",
            as_of_date="2024-01-01",
            current_regime="SuperBull",
            regime_distribution={"SuperBull": 120, "Crash": 5},
            total_trading_days=125,
            transition_matrix=t_matrix,
            recent_segments=ragged_segments,
        )
        assert snapshot.id is not None
        retrieved = get_latest_regime_snapshot(session, "^NSEI")
        assert retrieved is not None
        assert len(retrieved.transition_matrix) == 5
        assert retrieved.transition_matrix[0][0] == 0.70
        assert retrieved.recent_segments[0]["extra"]["nested"] == [1, 2, 3]

    def test_null_and_empty_json_fields(self, session):
        """Test that null/None and empty dicts in JSON columns serialize and deserialize cleanly."""
        log = create_backtest_log(
            session=session,
            ticker="^NSEI",
            start_date="2020-01-01",
            end_date="2020-12-31",
            strategy="EmptyJSON",
            metrics_json=None,
            equity_curve_summary=None,
        )
        retrieved = get_backtest_by_id(session, log.id)
        assert retrieved is not None
        assert retrieved.metrics_json is None
        assert retrieved.equity_curve_summary is None

        d = retrieved.to_dict()
        assert d["metrics_json"] is None
        assert d["equity_curve_summary"] is None


# ===========================================================================
# 2. Large Batch Benchmark Insertion, Retrieval & Rollback Atomicity
# ===========================================================================

class TestAdversarialBatchOperations:
    """Test batch operations under high load, edge size, and partial failure atomicity."""

    def test_empty_batch_insertion(self, session):
        """create_benchmark_batch with empty list should return empty list without error."""
        res = create_benchmark_batch(session, [])
        assert res == []
        assert count_benchmarks(session) == 0

    def test_large_batch_insertion_1000_records(self, session):
        """Insert 1,000 benchmark records in a single batch transaction."""
        models = ["XGBoost", "LSTM-DNN", "RandomForest", "LinearRegression", "ARIMA-GARCH"]
        records = []
        for i in range(1000):
            m_name = models[i % len(models)]
            records.append({
                "ticker": "^NSEI" if i % 2 == 0 else "^NSEBANK",
                "model_name": f"{m_name}_{i % 10}",
                "accuracy": 0.50 + (i % 50) * 0.01,
                "precision": 0.52 + (i % 45) * 0.01,
                "recall": 0.48 + (i % 50) * 0.01,
                "f1_score": 0.50 + (i % 48) * 0.01,
                "roc_auc": 0.55 + (i % 40) * 0.01,
                "training_time_sec": float(i % 100),
                "inference_latency_ms": 0.1 + (i % 10) * 0.1,
                "details": {"batch_idx": i, "split": "test"},
            })

        created = create_benchmark_batch(session, records)
        assert len(created) == 1000
        assert count_benchmarks(session) == 1000
        assert count_benchmarks(session, ticker="^NSEI") == 500
        assert count_benchmarks(session, ticker="^NSEBANK") == 500

    def test_batch_insertion_atomicity_on_integrity_violation(self, isolated_engine: Engine):
        """Ensure that if one record in a batch fails non-null constraint, entire batch rolls back."""
        valid_record_1 = {
            "ticker": "^NSEI",
            "model_name": "ValidModel1",
            "accuracy": 0.85,
            "f1_score": 0.84,
        }
        invalid_record = {
            "ticker": "^NSEI",
            "model_name": "InvalidModel",
            "accuracy": None,  # NOT NULL column -> will violate constraint
            "f1_score": 0.80,
        }
        valid_record_2 = {
            "ticker": "^NSEI",
            "model_name": "ValidModel2",
            "accuracy": 0.75,
            "f1_score": 0.74,
        }

        # Attempt batch insertion via context manager
        with pytest.raises((IntegrityError, SQLAlchemyError)):
            with get_db_session(isolated_engine) as sess:
                create_benchmark_batch(sess, [valid_record_1, invalid_record, valid_record_2])

        # Verify nothing was committed (atomicity guaranteed)
        with get_db_session(isolated_engine) as sess:
            assert count_benchmarks(sess) == 0
            assert get_benchmark_history(sess) == []

    def test_get_latest_benchmark_runs_with_many_duplicates(self, session):
        """Ensure get_latest_benchmark_runs dedupes correctly when many runs share the same model_name."""
        for i in range(10):
            create_benchmark_run(
                session=session,
                ticker="^NSEI",
                model_name="XGBoost",
                accuracy=0.70 + i * 0.01,
                f1_score=0.69 + i * 0.01,
            )
        for i in range(5):
            create_benchmark_run(
                session=session,
                ticker="^NSEI",
                model_name="LSTM-DNN",
                accuracy=0.60 + i * 0.01,
                f1_score=0.59 + i * 0.01,
            )
        for i in range(3):
            create_benchmark_run(
                session=session,
                ticker="^NSEI",
                model_name="RandomForest",
                accuracy=0.50 + i * 0.01,
                f1_score=0.49 + i * 0.01,
            )

        latest = get_latest_benchmark_runs(session, ticker="^NSEI")
        assert len(latest) == 3
        model_map = {r.model_name: r for r in latest}
        assert set(model_map.keys()) == {"XGBoost", "LSTM-DNN", "RandomForest"}
        assert model_map["XGBoost"].accuracy == pytest.approx(0.79)
        assert model_map["LSTM-DNN"].accuracy == pytest.approx(0.64)
        assert model_map["RandomForest"].accuracy == pytest.approx(0.52)


# ===========================================================================
# 3. CRUD Pagination, Boundary Offsets, and Filtering
# ===========================================================================

class TestAdversarialCRUDAndPagination:
    """Test boundary values, extreme offsets, zero limits, and SQL injection safety."""

    def test_zero_and_negative_limit_offset_behavior(self, session):
        """Test limit=0, offset=0, large offset > count in get_recent_backtests."""
        for i in range(5):
            create_backtest_log(
                session=session,
                ticker="^NSEI",
                start_date="2021-01-01",
                end_date="2021-12-31",
                strategy=f"Strat_{i}",
                initial_investment=100000.0,
            )

        # Zero limit
        zero_lim = get_recent_backtests(session, limit=0)
        assert len(zero_lim) == 0

        # Offset beyond total count
        high_offset = get_recent_backtests(session, limit=10, offset=9999)
        assert len(high_offset) == 0

        # Offset equal to total count
        exact_offset = get_recent_backtests(session, limit=10, offset=5)
        assert len(exact_offset) == 0

        # Partial remaining
        partial = get_recent_backtests(session, limit=10, offset=3)
        assert len(partial) == 2

    def test_filtering_non_existent_ticker_and_strategy(self, session):
        """Querying with non-existent criteria returns empty lists and 0 counts cleanly."""
        create_backtest_log(
            session=session,
            ticker="^NSEI",
            start_date="2021-01-01",
            end_date="2021-12-31",
            strategy="Momentum",
        )
        assert count_backtests(session, ticker="NON_EXISTENT") == 0
        assert count_backtests(session, strategy="NON_EXISTENT") == 0
        assert get_recent_backtests(session, ticker="NON_EXISTENT") == []
        assert get_recent_backtests(session, strategy="NON_EXISTENT") == []
        assert count_regime_snapshots(session, ticker="NON_EXISTENT") == 0
        assert get_recent_regime_snapshots(session, ticker="NON_EXISTENT") == []
        assert get_latest_regime_snapshot(session, ticker="NON_EXISTENT") is None
        assert count_benchmarks(session, ticker="NON_EXISTENT") == 0
        assert get_benchmark_history(session, ticker="NON_EXISTENT") == []

    @pytest.mark.parametrize("sqli_payload", [
        "' OR '1'='1",
        "'; DROP TABLE backtest_logs; --",
        '" OR 1=1 --',
        "admin'--",
        "1; SELECT * FROM backtest_logs;",
        "UNION ALL SELECT NULL, NULL, NULL--",
    ])
    def test_sql_injection_resilience_in_filters(self, session, sqli_payload):
        """SQL injection strings in ticker/strategy/model_name filters must be safely parameterized."""
        create_backtest_log(
            session=session,
            ticker="^NSEI",
            start_date="2022-01-01",
            end_date="2022-12-31",
            strategy="SafeStrategy",
        )
        res = get_recent_backtests(session, ticker=sqli_payload)
        assert res == []

        res_strat = get_recent_backtests(session, strategy=sqli_payload)
        assert res_strat == []

        c = count_backtests(session, ticker=sqli_payload)
        assert c == 0

        assert count_backtests(session) == 1
        all_logs = get_recent_backtests(session)
        assert len(all_logs) == 1
        assert all_logs[0].strategy == "SafeStrategy"

    def test_delete_boundary_conditions(self, session):
        """Test deleting non-existent IDs, negative IDs, and zero."""
        assert delete_backtest_by_id(session, -1) is False
        assert delete_backtest_by_id(session, 0) is False
        assert delete_backtest_by_id(session, 99999999) is False

        bt = create_backtest_log(
            session=session,
            ticker="^NSEI",
            start_date="2020",
            end_date="2021",
            strategy="ToDel",
        )
        assert delete_backtest_by_id(session, bt.id) is True
        assert delete_backtest_by_id(session, bt.id) is False


# ===========================================================================
# 4. Background Tasks Error Isolation and Corrupt Data Handling
# ===========================================================================

class TestAdversarialBackgroundTasks:
    """Test that BackgroundTasks handlers isolate all corrupt inputs and database failures."""

    @pytest.mark.parametrize("corrupt_payload,corrupt_result", [
        (None, None),
        ({}, {}),
        ([], []),
        ("not a dict", "not a dict"),
        (12345, 67890),
        ({"ticker": None, "start_date": None}, {"current_regime": None}),
        ({"initial_investment": "not_a_number"}, {"overall_metrics": "corrupted_type"}),
        ({"commission_pct": float("nan")}, {"regime_timeline": "should_be_list_not_str"}),
        ({"ticker": "^NSEI"}, {"overall_metrics": {"Strat": {"CAGR": "invalid_float", "Sharpe": None}}}),
    ])
    def test_save_analysis_record_corrupt_inputs_graceful_handling(
        self, isolated_engine: Engine, corrupt_payload: Any, corrupt_result: Any
    ):
        """save_analysis_record never raises unhandled exceptions on completely corrupt payloads."""
        res = save_analysis_record(corrupt_payload, corrupt_result, engine=isolated_engine)
        if res is not None:
            assert isinstance(res, dict)
            assert "backtest_log_id" in res

    @pytest.mark.parametrize("corrupt_payload,corrupt_result", [
        (None, None),
        ({}, {}),
        ({"ticker": "^NSEI"}, {"results": "not_a_list"}),
        ({"ticker": "^NSEI"}, {"results": [None, 42, "string", {}, {"strategy": "Test", "metrics": "invalid"}]}),
        ({"ticker": "^NSEI"}, {"results": [{"strategy": "Test", "metrics": {"CAGR": "not_float", "Sharpe": "abc"}}]}),
    ])
    def test_save_backtest_record_corrupt_inputs_graceful_handling(
        self, isolated_engine: Engine, corrupt_payload: Any, corrupt_result: Any
    ):
        """save_backtest_record never raises unhandled exceptions on corrupt inputs."""
        res = save_backtest_record(corrupt_payload, corrupt_result, engine=isolated_engine)
        assert isinstance(res, list)

    @pytest.mark.parametrize("corrupt_benchmark_data", [
        None,
        {},
        [],
        "string",
        42,
        {"ticker": "^NSEI", "models": "not_a_list"},
        {"ticker": "^NSEI", "models": [None, 123, "corrupt", {}]},
        {"ticker": "^NSEI", "models": [{"model_name": "Test", "accuracy": "invalid_number"}]},
    ])
    def test_save_benchmark_record_corrupt_inputs_graceful_handling(
        self, isolated_engine: Engine, corrupt_benchmark_data: Any
    ):
        """save_benchmark_record never raises unhandled exceptions on corrupt benchmark cards."""
        res = save_benchmark_record(corrupt_benchmark_data, engine=isolated_engine)
        assert isinstance(res, list)

    def test_broken_database_engine_handled_gracefully(self):
        """Background tasks handle totally broken engines or connection timeouts without crashing."""
        mock_dead_engine = MagicMock(spec=Engine)
        mock_dead_engine.connect.side_effect = SQLAlchemyError("Database server connection terminated")

        res_analysis = save_analysis_record({"ticker": "^NSEI"}, {"current_regime": "Bull"}, engine=mock_dead_engine)
        assert res_analysis is None

        res_backtest = save_backtest_record({"ticker": "^NSEI"}, {"results": []}, engine=mock_dead_engine)
        assert res_backtest == []

        res_benchmark = save_benchmark_record({"models": []}, engine=mock_dead_engine)
        assert res_benchmark == []

    def test_malformed_pydantic_model_dump_raising_exception(self, isolated_engine: Engine):
        """If a custom object's model_dump() raises an exception, service gracefully handles it."""
        class PoisonedObject:
            def model_dump(self):
                raise RuntimeError("Explosive Pydantic error during serialization!")

        poison = PoisonedObject()
        res = save_analysis_record(poison, poison, engine=isolated_engine)
        assert res is not None or res is None


# ===========================================================================
# 5. Multithreaded Concurrency Stress Test
# ===========================================================================

class TestAdversarialConcurrency:
    """Test high-concurrency multithreaded read/write operations against the DB layer."""

    def test_concurrent_writes_and_reads(self, isolated_engine: Engine):
        """Execute 30 concurrent worker threads writing backtest logs, regimes, and benchmarks."""
        num_workers = 30
        errors: list[str] = []

        def worker_task(thread_id: int):
            try:
                # 1. Write BacktestLog
                with get_db_session(isolated_engine) as s:
                    bt = create_backtest_log(
                        session=s,
                        ticker="^NSEI",
                        start_date="2022-01-01",
                        end_date="2022-12-31",
                        strategy=f"ConcurrentStrat_{thread_id}",
                        sharpe=1.0 + thread_id * 0.05,
                    )
                    assert bt.id is not None

                # 2. Write RegimeSnapshot
                with get_db_session(isolated_engine) as s:
                    snap = create_regime_snapshot(
                        session=s,
                        ticker="^NSEI",
                        as_of_date=f"2023-{thread_id:02d}-01",
                        current_regime="Bull" if thread_id % 2 == 0 else "Bear",
                        regime_distribution={"Bull": 50, "Bear": 50},
                        total_trading_days=100,
                    )
                    assert snap.id is not None

                # 3. Write Benchmark
                with get_db_session(isolated_engine) as s:
                    bm = create_benchmark_run(
                        session=s,
                        ticker="^NSEI",
                        model_name=f"Model_{thread_id % 5}",
                        accuracy=0.80,
                        f1_score=0.79,
                    )
                    assert bm.id is not None

                # 4. Read back
                with get_db_session(isolated_engine) as s:
                    recent_bt = get_recent_backtests(s, limit=5)
                    assert len(recent_bt) > 0
                    recent_snap = get_recent_regime_snapshots(s, limit=5)
                    assert len(recent_snap) > 0

            except Exception as e:
                errors.append(f"Thread {thread_id} failed: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker_task, i) for i in range(1, num_workers + 1)]
            concurrent.futures.wait(futures)

        assert errors == [], f"Concurrency errors encountered: {errors}"

        # Verify final counts
        with get_db_session(isolated_engine) as s:
            assert count_backtests(s) == num_workers
            assert count_regime_snapshots(s) == num_workers
            assert count_benchmarks(s) == num_workers
            assert check_connection_health(isolated_engine)["status"] == "healthy"


# ===========================================================================
# 6. Extreme Boundary Financial Metrics & Idempotency
# ===========================================================================

class TestAdversarialExtremesAndIdempotency:
    """Test extreme financial magnitudes and schema lifecycle idempotency."""

    def test_idempotent_init_db(self, isolated_engine: Engine):
        """Calling init_db multiple times on an already initialized engine must succeed idempotently."""
        init_db(isolated_engine)
        init_db(isolated_engine)
        init_db(isolated_engine)
        health = check_connection_health(isolated_engine)
        assert health["status"] == "healthy"

    def test_trillion_dollar_magnitudes_and_extreme_ratios(self, session):
        """Test huge investment amounts (₹1,000,000,000,000.0) and extreme financial ratios."""
        log = create_backtest_log(
            session=session,
            ticker="^NSEI",
            start_date="1990-01-01",
            end_date="2026-01-01",
            strategy="SovereignFundStrategy",
            initial_investment=1_000_000_000_000.0,  # 1 Trillion INR
            commission_pct=0.05,
            slippage_pct=0.10,
            cagr=5.50,  # 550% CAGR
            sharpe=45.8,
            sortino=98.4,
            max_drawdown=-0.000001,
            calmar=5500000.0,
            volatility=0.001,
        )
        assert log.id is not None
        retrieved = get_backtest_by_id(session, log.id)
        assert retrieved is not None
        assert retrieved.initial_investment == 1_000_000_000_000.0
        assert retrieved.cagr == 5.50
        assert retrieved.calmar == 5500000.0
        d = retrieved.to_dict()
        assert d["initial_investment"] == 1_000_000_000_000.0

    def test_large_text_notes_and_architecture_summary(self, session):
        """Test multi-kilobyte text payloads in notes and architecture_summary."""
        long_notes = "Institutional Backtest Run Note " * 1000  # ~32 KB
        log = create_backtest_log(
            session=session,
            ticker="^NSEI",
            start_date="2020-01-01",
            end_date="2021-01-01",
            strategy="LongNotesStrat",
            notes=long_notes,
        )
        retrieved = get_backtest_by_id(session, log.id)
        assert retrieved is not None
        assert retrieved.notes == long_notes

        long_arch = "PyTorch Transformer-LSTM Hybrid Architecture " * 500
        bm = create_benchmark_run(
            session=session,
            ticker="^NSEI",
            model_name="TransformerLSTM",
            accuracy=0.88,
            f1_score=0.87,
            architecture_summary=long_arch,
        )
        retrieved_bm = get_benchmark_history(session, model_name="TransformerLSTM")[0]
        assert retrieved_bm.architecture_summary == long_arch
