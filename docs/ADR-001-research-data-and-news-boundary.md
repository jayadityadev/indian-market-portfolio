# ADR-001: Research Data and News Boundary

## Status

Accepted and implemented for demo platform operation.

## Decision

Use one canonical, versioned NIFTY 50 daily dataset as source for features, regimes, labels, backtests, and recommendations. Rebuild derived artifacts from that source. Extend historical coverage toward 2005-present and reserve latest observations as untouched out-of-sample data.

Add live news as a separate context subsystem. Prefer free provider adapters, starting with GDELT and reputable RSS feeds. Cache successful responses, show fetched-at timestamps, and never fabricate fallback content. AI summaries must retain source links; raw article mode remains available.

News must not modify quantitative strategy probabilities until a separately validated news model exists.

## Rationale

The platform now rebuilds one canonical NIFTY 50 dataset and derived artifacts. Historical data is frozen through `2026-08-14`; strict contracts and hashes prevent accidental mixing of incompatible artifacts.

Separating news from recommendation avoids unvalidated sentiment becoming hidden model logic. It also makes outages, stale cache, licensing, and provenance visible to users.

## Consequences

- Dataset rebuild precedes news dashboard work.
- Strict validation can block research-grade recommendations.
- Demo mode must use real cached data or explicitly marked fixtures.
- Portfolio optimization remains future work.
- News ingestion needs deduplication, relevance filtering, rate-limit handling, and source attribution.

## Rejected Alternatives

- Keep independently generated parquet artifacts: rejected because date coverage and model lineage are unclear.
- Hardcoded news or heuristic market narratives: rejected because user requires actual data.
- Feed news sentiment directly into recommendations now: rejected because no labelled validation set exists.
