# E2E Test Infra: Indian Market Portfolio Intelligence Platform

## Test Philosophy
- Requirement-driven, opaque-box testing based on `ORIGINAL_REQUEST.md`.
- No hardcoded mocks bypassing logic — offline resilient fixtures and parameter variations.
- Methodology: Category-Partition, Boundary Value Analysis, Pairwise Combinatorial Testing, Real-World Workload Testing.

## Feature Inventory & Test Matrix
| # | Feature | Requirement Source | Tier 1 (Coverage) | Tier 2 (Boundary) | Tier 3 (Pairwise) | Tier 4 (Workloads) |
|---|---------|-------------------|:-----------------:|:-----------------:|:-----------------:|:------------------:|
| 1 | Gaussian HMM Regime Detection | R1 | 5 cases | 5 cases | ✓ | ✓ |
| 2 | XGBoost Strategy Recommendation | R1 | 5 cases | 5 cases | ✓ | ✓ |
| 3 | LSTM-DNN Academic Benchmark | R1 | 5 cases | 5 cases | ✓ | ✓ |
| 4 | SQLAlchemy Database Persistence | R2 | 5 cases | 5 cases | ✓ | ✓ |
| 5 | AI Market Analyst Report | R3 | 5 cases | 5 cases | ✓ | ✓ |
| 6 | REST API `/api/v1/*` Endpoints | R4 | 5 cases | 5 cases | ✓ | ✓ |
| 7 | Next.js Frontend Build & Types | R4 | 5 cases | 5 cases | ✓ | ✓ |

## Test Architecture
- Framework: `pytest`, `pytest-asyncio`, `httpx` for API testing, `npm run build` / `npm run lint` for Next.js.
- Database: Test DB isolation (in-memory or file SQLite `sqlite:///./test_portfolio.db`).
- LLM Provider Testing: Mock provider fallbacks and structured output schema verification.
- Pass/Fail Semantics:
  - Exit code 0 for `pytest tests/`
  - Exit code 0 for `npm run build` in `frontend/`

## Coverage Thresholds
- Tier 1: >= 35 test cases (>= 5 per feature)
- Tier 2: >= 35 boundary test cases
- Tier 3: >= 10 cross-feature integration test cases
- Tier 4: >= 5 realistic multi-asset real-world workload pipelines
- Total Target: >= 85 test assertions
