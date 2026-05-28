"""Streamlit dashboard for Indian Market Portfolio Intelligence platform."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backtester import run_backtest
from data_pipeline import engineer_features, fetch_data
from regime_detector import fit_regimes, get_current_regime, get_regime_performance
from strategies import (
    buy_and_hold, ma_crossover, momentum_strategy, rsi_strategy,
    bollinger_bands, dual_momentum,
)
from classifier_inference import get_strategy_probabilities
from risk_forecaster import simulate_drawdowns
from market_outlook import get_market_outlook

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Indian Market Portfolio Intelligence",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Indian Market Portfolio Intelligence")
st.caption(
    "Strategy evaluation system for Indian equities — powered by ML regime detection"
)

# ============================================================================
# CONSTANTS & CONFIG
# ============================================================================

DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "nifty50.parquet"
STRATEGY_FUNCTIONS = {
    "Buy & Hold": buy_and_hold,
    "MA Crossover": ma_crossover,
    "RSI": rsi_strategy,
    "Momentum": momentum_strategy,
    "Bollinger Bands": bollinger_bands,
    "Dual Momentum": dual_momentum,
}
STRATEGY_NAMES = list(STRATEGY_FUNCTIONS.keys())
MIN_RELIABLE_DAYS = 252
MIN_REGIME_DAYS = 60

# ============================================================================
# CACHING LAYER
# ============================================================================


@st.cache_data
def load_or_fetch_data(
    ticker: str, start_date: datetime, end_date: datetime
) -> pd.DataFrame | None:
    """Load cached parquet or fetch fresh data from yfinance.
    
    **Doing:** Using @st.cache_data to avoid redundant yfinance calls on each interaction.
    **Why:** Streamlit reruns entire script on sidebar changes. Without caching, every
    date range adjustment triggers expensive network fetch + feature engineering.
    """
    # Try loading from parquet if it exists
    if DEFAULT_DATA_PATH.exists():
        try:
            cached_df = pd.read_parquet(DEFAULT_DATA_PATH)
            cached_df.index = pd.to_datetime(cached_df.index)
            
            # Check if cached data covers requested range
            if len(cached_df) > 0 and cached_df.index[0] <= start_date and cached_df.index[-1] >= end_date:
                return cached_df.loc[start_date:end_date].copy()
        except Exception:
            pass  # Fall through to fresh fetch
    
    # Fetch fresh data
    try:
        raw_df = fetch_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        featured_df = engineer_features(raw_df)
        return featured_df
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return None


@st.cache_data
def load_regime_data() -> pd.DataFrame | None:
    """Load regime labels from Day 4 output."""
    try:
        regime_path = Path(__file__).parent.parent / "data" / "nifty50_regimes.parquet"
        if regime_path.exists():
            regime_df = pd.read_parquet(regime_path)
            regime_df.index = pd.to_datetime(regime_df.index)
            return regime_df
    except Exception as e:
        st.error(f"Regime data load failed: {e}")
    return None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def run_all_strategies(
    df: pd.DataFrame, commission_pct: float = 0.0, slippage_pct: float = 0.0
) -> dict[str, dict]:
    """Run all 6 strategies through backtester.
    
    **Doing:** Running all strategies even if user selects single strategy.
    **Why:** Need all 6 for regime comparison table and Auto recommendation.
    Also provides benchmark context (Buy & Hold must always be visible).
    Strategies requiring more data than available are skipped with warnings.
    """
    # Minimum data requirements per strategy
    min_days_required = {
        "MA Crossover": 200,
        "Momentum": 252,
        "Dual Momentum": 252,
    }
    n_days = len(df)

    results = {}
    for strategy_name, strategy_func in STRATEGY_FUNCTIONS.items():
        req = min_days_required.get(strategy_name, 0)
        if req > 0 and n_days < req:
            st.warning(
                f"⏭️ Skipping **{strategy_name}** — needs {req}+ trading days, "
                f"only {n_days} available."
            )
            continue
        try:
            signals = strategy_func(df)
            backtest_result = run_backtest(
                df["Close"],
                signals,
                commission_pct=commission_pct,
                slippage_pct=slippage_pct,
            )
            results[strategy_name] = backtest_result
        except Exception as e:
            st.warning(f"Strategy '{strategy_name}' failed: {e}")
    return results


def get_filtered_results(
    all_results: dict[str, dict], selected_strategy: str
) -> dict[str, dict]:
    """Filter results based on user selection."""
    if selected_strategy == "Auto (Best for Current Regime)":
        return all_results
    else:
        # Show selected strategy + Buy & Hold for comparison
        filtered = {"Buy & Hold": all_results.get("Buy & Hold", {})}
        if selected_strategy in all_results:
            filtered[selected_strategy] = all_results[selected_strategy]
        return filtered if len(filtered) > 1 else all_results


def interpret_current_regime(df: pd.DataFrame) -> str:
    """Detect current market regime from most recent observations.
    
    **Doing:** Using Day 4's regime detection model to classify current market state.
    **Why:** Feeds into Auto recommendation logic and regime-wise performance display.
    """
    try:
        current_regime = get_current_regime(df)
        return current_regime if current_regime else "Unknown"
    except Exception as e:
        st.warning(f"Regime detection failed: {e}")
        return "Unknown"


def compute_regime_performance(
    df: pd.DataFrame, results: dict[str, dict]
) -> dict[str, dict]:
    """Compute per-strategy, per-regime Sharpe ratios.
    
    **Doing:** Using portfolio_analytics skill pattern for regime breakdown.
    **Why:** Heatmap shows strategy × regime matrix. This reveals which strategies
    excel in which market conditions — core insight for regime-aware trading.
    """
    try:
        perf_df = get_regime_performance(df, results)
        
        # Transform dataframe into nested dict: strategy -> regime -> Sharpe
        perf_dict = {}
        for _, row in perf_df.iterrows():
            strategy = str(row["strategy"])
            regime = str(row["regime"])
            sharpe = float(row["Sharpe"])
            
            if strategy not in perf_dict:
                perf_dict[strategy] = {}
            
            perf_dict[strategy][regime] = sharpe
        
        return perf_dict if perf_dict else {}
    except Exception as e:
        st.warning(f"Regime performance computation failed: {e}")
        return {}


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Stock / Index selector
    ticker = st.selectbox(
        "Stock / Index",
        ["^NSEI"],
        help="MVP scope: NIFTY 50 only. Multi-asset portfolio support in future."
    )
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.today() - timedelta(days=730),
            help="At least 2 years of data is recommended for better regime inference."
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.today(),
        )
    
    # Strategy selector
    strategy_options = ["Auto (Best for Current Regime)"] + STRATEGY_NAMES
    selected_strategy = st.selectbox(
        "Strategy",
        strategy_options,
    )
    
    # Transaction Costs
    st.markdown("---")
    st.subheader("💸 Transaction Costs")
    commission_pct = st.slider(
        "Commission (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="Brokerage and other transaction taxes (STT, etc.) per trade."
    ) / 100.0
    
    slippage_pct = st.slider(
        "Slippage (%)",
        min_value=0.0,
        max_value=2.0,
        value=0.05,
        step=0.01,
        help="Price impact due to order execution latency or liquidity."
    ) / 100.0
    
    # Run button
    run_btn = st.sidebar.button("🚀 Run Analysis", width="stretch")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

if run_btn:
    # Validate date range
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()
    
    # Step 1: Fetch & engineer data
    with st.spinner("📊 Fetching data and engineering features..."):
        df = load_or_fetch_data(ticker, start_date, end_date)
    
    if df is None or len(df) == 0:
        st.error("No data available for selected date range.")
        st.stop()
    
    if len(df) < MIN_REGIME_DAYS:
        st.error(
            f"Too little data for regime detection (need {MIN_REGIME_DAYS}+ trading days). "
            "Expand date range."
        )
        st.stop()

    if len(df) < MIN_RELIABLE_DAYS:
        st.warning(
            f"Only {len(df)} trading days in range. Metrics are less reliable below "
            f"{MIN_RELIABLE_DAYS} days. Proceeding anyway."
        )
    
    # Step 2: Run all strategies
    with st.spinner("🎯 Running backtests..."):
        all_results = run_all_strategies(
            df, commission_pct=commission_pct, slippage_pct=slippage_pct
        )
    
    if not all_results:
        st.error("Backtesting failed for all strategies.")
        st.stop()
    
    # Step 3: Load regime data
    with st.spinner("📈 Loading regime labels..."):
        regime_df = load_regime_data()
    
    if regime_df is None:
        st.error("Regime data not available. Please run Day 4 regime detection first.")
        st.stop()
    
    # Align regime_df to df's date range
    regime_df = regime_df.loc[regime_df.index.isin(df.index)]
    
    # Step 4: Detect current regime
    current_regime = interpret_current_regime(df)
    
    # Step 4b: Warn if single-regime period
    unique_regimes = regime_df["regime"].dropna().astype(str).nunique()
    if unique_regimes <= 1:
        st.warning(
            "⚠️ Only one market regime detected in this date range. "
            "Regime-wise comparison is unreliable — consider expanding the date range."
        )
    
    # Step 5: Compute regime performance
    with st.spinner("🔍 Analyzing regime performance..."):
        regime_perf = compute_regime_performance(df, all_results)
    
    # Step 6: Filter results for display
    display_results = get_filtered_results(all_results, selected_strategy)
    
    # ========================================================================
    # SECTION 1: EQUITY CURVE CHART
    # ========================================================================
    
    st.markdown("---")
    st.subheader("📊 Equity Curve — Strategy Performance")
    
    """
    **Doing:** Plotting normalized equity curves for all strategies side-by-side.
    **Why:** Users need visual comparison of how each strategy grew capital over time.
    Normalized to 100 makes them directly comparable. Hover shows exact values.
    **Technical:** Using Plotly's hovermode='x unified' to sync values at same date.
    """
    
    fig_equity = go.Figure()
    
    for strategy_name, result in display_results.items():
        if "equity_curve" not in result:
            continue
        
        equity_curve = result["equity_curve"]
        # Normalize to 100
        normalized_equity = 100 * equity_curve / equity_curve.iloc[0]
        
        fig_equity.add_trace(
            go.Scatter(
                x=normalized_equity.index,
                y=normalized_equity.values,
                name=strategy_name,
                mode="lines",
                hovertemplate=f"<b>{strategy_name}</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: ₹%{{y:.2f}}<extra></extra>"
            )
        )
    
    fig_equity.update_layout(
        title="Normalized Equity Curve (Base = ₹100)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (₹)",
        hovermode="x unified",
        template="plotly_dark",
        height=400,
    )
    
    st.plotly_chart(fig_equity, width='stretch')
    
    # ========================================================================
    # SECTION 2: METRICS TABLE
    # ========================================================================
    
    st.markdown("---")
    st.subheader("📋 Performance Metrics")
    
    """
    **Doing:** Displaying CAGR, Sharpe, Max Drawdown, and Volatility for each strategy.
    **Why:** These 4 metrics tell the full story:
    - CAGR = total return
    - Sharpe = risk-adjusted return (best for comparing strategies)
    - MaxDD = worst loss (psychological impact)
    - Volatility = risk level
    **Financial gotcha avoided:** Any one metric alone can mislead. Sharpe might be
    high but MaxDD extreme, or CAGR high but Volatility dangerous.
    """
    
    metrics_rows = []
    for strategy_name, result in display_results.items():
        if "metrics" not in result:
            continue
        
        metrics = result["metrics"]
        metrics_rows.append(
            {
                "Strategy": strategy_name,
                "CAGR": f"{metrics.get('CAGR', 0):.1%}",
                "Sharpe Ratio": f"{metrics.get('Sharpe', 0):.2f}",
                "Max Drawdown": f"{metrics.get('MaxDrawdown', 0):.1%}",
                "Volatility": f"{metrics.get('Volatility', 0):.1%}",
            }
        )
    
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        st.dataframe(metrics_df, width='stretch', hide_index=True)
    else:
        st.warning("No metrics available.")
    
    # ========================================================================
    # SECTION 3: REGIME TIMELINE
    # ========================================================================
    
    st.markdown("---")
    st.subheader("🎨 Market Regime Timeline")
    
    """
    **Doing:** Rendering color-coded regime labels over historical dates.
    **Why:** Core differentiator of this platform. Visualizes WHEN regimes occurred.
    User can correlate equity curve peaks/valleys with market conditions. E.g.,
    "Momentum crashed 40% because Bull → Bear transition happened unexpectedly."
    **Alternative considered:** Gantt-style px.timeline. Rejected because it doesn't
    handle daily granularity well with 2,000+ rows of regime data.
    """
    
    if regime_df is not None and len(regime_df) > 0:
        # Create regime timeline
        regime_colors = {"Bull": "green", "Bear": "red", "Sideways": "orange"}
        
        fig_regime = go.Figure()
        
        for regime_name, color in regime_colors.items():
            mask = regime_df["regime"].astype(str) == regime_name
            regime_dates = regime_df.index[mask]
            
            if len(regime_dates) > 0:
                fig_regime.add_trace(
                    go.Scatter(
                        x=regime_dates,
                        y=[1] * len(regime_dates),
                        mode="markers",
                        marker=dict(color=color, size=5, symbol="square"),
                        name=regime_name,
                        hovertemplate=f"<b>{regime_name}</b><br>Date: %{{x|%Y-%m-%d}}<extra></extra>"
                    )
                )
        
        fig_regime.update_layout(
            title="Market Regime Timeline",
            xaxis_title="Date",
            yaxis=dict(visible=False),
            height=120,
            showlegend=True,
            template="plotly_dark",
            margin=dict(t=40, b=20, l=20, r=20),
        )
        
        st.plotly_chart(fig_regime, width='stretch')
    else:
        st.warning("Regime timeline data not available.")
    
    # ========================================================================
    # SECTION 4: REGIME-WISE PERFORMANCE HEATMAP
    # ========================================================================
    
    st.markdown("---")
    st.subheader("🔥 Strategy Performance by Regime")
    
    """
    **Doing:** Creating a heatmap showing strategy × regime Sharpe ratios.
    **Why:** Reveals which strategies dominate which regimes. Patterns visible instantly:
    - Green = good Sharpe (strategy profitable in that regime)
    - Red = bad Sharpe (strategy lost money or high volatility)
    **Financial insight:** Momentum often crushes in Bull, fails in Sideways.
    MA Crossover smooth in Sideways, whipsawed in trending regimes.
    **Why Sharpe not CAGR:** Sharpe is risk-adjusted. High CAGR but extreme drawdown
    in one regime is flagged by low Sharpe. Catches dangerous strategies.
    """
    
    if regime_perf:
        # Build heatmap data: rows = strategies, columns = regimes, values = Sharpe
        heatmap_data = []
        for strategy_name, regimes in regime_perf.items():
            row = {"Strategy": strategy_name}
            for regime_name in ["Bull", "Bear", "Sideways"]:
                sharpe = regimes.get(regime_name)
                row[regime_name] = np.nan if sharpe is None else sharpe
            heatmap_data.append(row)
        
        perf_df = pd.DataFrame(heatmap_data).set_index("Strategy")
        text_df = perf_df.astype(object).copy()
        for column in text_df.columns:
            text_df[column] = text_df[column].map(
                lambda value: "N/A" if pd.isna(value) else f"{value:.2f}"
            )
        
        fig_heatmap = px.imshow(
            perf_df,
            color_continuous_scale="RdYlGn",
            title="Sharpe Ratio by Strategy × Regime",
            labels=dict(color="Sharpe Ratio"),
        )
        fig_heatmap.update_traces(text=text_df.values, texttemplate="%{text}")
        
        fig_heatmap.update_layout(
            template="plotly_dark",
            height=300,
            coloraxis_colorbar=dict(title="Sharpe"),
        )
        
        st.plotly_chart(fig_heatmap, width='stretch')
    else:
        st.warning("Regime performance data not available.")
    
    # ========================================================================
    # SECTION 5: STRATEGY RECOMMENDATION BOX
    # ========================================================================
    
    st.markdown("---")
    st.subheader("📊 Strategy Recommendation")
    
    """
    **Doing:** Auto-selecting best strategy using ML probabilities with fallback to historical Sharpe.
    **Why:** Core feature differentiating this platform. Instead of static "buy & hold" or
    simple lookup, system uses a Random Forest classifier to predict probability of outperforming.
    **Financial gotcha avoided:** Showing disclaimer and applying fallback if confidence < 0.55.
    Honest system > overconfident system.
    """
    
    # Calculate probabilities
    probs = {}
    if len(df) >= 252:
        # Use the last 252 days for feature extraction
        df_recent = df.iloc[-252:]
        probs = get_strategy_probabilities(df_recent, current_regime, models_dir=Path(__file__).parent.parent / "models")
        
    # Find best strategy historically for current regime (fallback)
    best_historical_strategy = "No strategy available"
    best_sharpe = np.nan
    
    if regime_perf and current_regime != "Unknown":
        candidates: dict[str, float] = {}
        for strategy_name, regimes in regime_perf.items():
            if strategy_name == "Buy & Hold":
                continue
            sharpe = regimes.get(current_regime)
            if sharpe is None or pd.isna(sharpe):
                sharpe = -999.0
            candidates[strategy_name] = float(sharpe)

        if candidates:
            best_historical_strategy = max(candidates, key=candidates.get)
            best_sharpe = candidates[best_historical_strategy]

    sharpe_label = "N/A" if pd.isna(best_sharpe) or best_sharpe <= -999 else f"{best_sharpe:.2f}"
    
    max_prob = max(probs.values()) if probs else 0.0
    best_ml_strategy = max(probs, key=probs.get) if probs else "Unknown"
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Market Regime", current_regime)
    with col2:
        if max_prob >= 0.55:
            st.metric("Recommended Strategy", best_ml_strategy)
        else:
            st.metric("Recommended Strategy", best_historical_strategy)
            
    # Display probabilities bar chart if available
    if probs:
        prob_df = pd.DataFrame([{"Strategy": k, "Probability": v} for k, v in probs.items()])
        prob_df = prob_df.sort_values(by="Probability", ascending=True)
        
        fig_prob = px.bar(
            prob_df, 
            x="Probability", 
            y="Strategy", 
            orientation="h",
            title="Model Confidence (Probability to Beat Buy & Hold in Next ~63 Days)",
            range_x=[0, 1]
        )
        # Add a vertical line for the 0.55 threshold
        fig_prob.add_vline(x=0.55, line_dash="dash", line_color="red", annotation_text="Confidence Threshold")
        fig_prob.update_layout(template="plotly_dark", height=250)
        st.plotly_chart(fig_prob, width='stretch')
        
    if max_prob >= 0.55:
        st.success(
            f"Based on current market features, the ML model predicts **{best_ml_strategy}** "
            f"has a **{max_prob:.0%} probability** of beating Buy & Hold over the next ~63 days."
        )
    else:
        st.warning(
            "Low confidence from ML model (all probabilities < 55%). "
            "Falling back to historical regime performance."
        )
        st.info(
            f"In **{current_regime}** markets, **{best_historical_strategy}** historically achieved "
            f"the best risk-adjusted return (Sharpe: {sharpe_label}). "
        )
        
    st.caption(
        "**⚠️ Disclaimer:** This recommendation is based on historical probabilities and regime performance, "
        "not a guarantee of future returns. Past performance does not imply future results. "
        "Always conduct your own research and consult a financial advisor."
    )
    
    # ========================================================================
    # SECTION 6: RISK FORECAST
    # ========================================================================
    
    if current_regime != "Unknown":
        forecast_strategy = best_ml_strategy if max_prob >= 0.55 else best_historical_strategy
        
        if forecast_strategy in all_results and "equity_curve" in all_results[forecast_strategy]:
            strat_equity = all_results[forecast_strategy]["equity_curve"]
            strat_returns = strat_equity.pct_change().dropna()
            
            regime_dates = regime_df.index[regime_df["regime"].astype(str) == current_regime]
            valid_dates = strat_returns.index.intersection(regime_dates)
            regime_returns = strat_returns.loc[valid_dates]
            
            if len(regime_returns) >= 30:
                forecast = simulate_drawdowns(regime_returns, n_simulations=1000, horizon=63)
                
                st.markdown("---")
                st.subheader("🛡️ Downside Risk Forecast (Next 63 Days)")
                st.info(
                    f"If you deploy **{forecast_strategy}** in the current **{current_regime}** regime, "
                    f"historical bootstrapping (1,000 simulations) estimates the following maximum drawdowns:"
                )
                
                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    st.metric("Best Case (10% chance)", f"{forecast['best_case_90']:.1%}")
                with rc2:
                    st.metric("Median (50% chance)", f"{forecast['median_50']:.1%}")
                with rc3:
                    st.metric("Worst Case (90% chance)", f"{forecast['worst_case_10']:.1%}")
                    
                st.caption(
                    "**Note:** 'Worst Case (90%)' means there is a 10% chance the drawdown could be worse than this number. "
                    "Bootstrapping assumes future risk distributions will match past distributions within the same regime."
                )

    # ========================================================================
    # SECTION 7: MARKET OUTLOOK (NEW)
    # ========================================================================
    
    st.markdown("---")
    st.subheader("🔮 Market Outlook & Forecast")
    
    with st.spinner("Generating market outlook..."):
        try:
            outlook = get_market_outlook(df, current_regime, probs)
            
            # Using st.info/warning/success based on outlook label
            color_map = {
                "Strongly Bullish": "🟢",
                "Bullish": "📈",
                "Neutral": "⚖️",
                "Bearish": "📉",
                "Strongly Bearish": "🔴"
            }
            label = outlook.get("outlook", "Neutral")
            emoji = color_map.get(label, "⚖️")
            
            # Layout for outlook
            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                st.metric("Outlook Label", f"{emoji} {label}")
                st.metric("Confidence Level", outlook.get("confidence", "Medium"))
            with oc2:
                st.metric("Outperformance Prob", f"{outlook.get('probability', 0):.1%}")
                st.metric("Forecast Horizon", outlook.get("horizon", "N/A"))
            with oc3:
                st.metric("Expected Volatility", outlook.get("expected_volatility", "N/A"))

            # Disclaimer
            st.markdown(f"**Analysis Note:** {outlook.get('disclaimer', '')}")
            
        except Exception as e:
            st.warning(f"Could not generate market outlook: {e}")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.caption(
        "🔧 MVP Platform — Indian Market Portfolio Intelligence | "
        "Data: yfinance | Backtester: 252 trading days | Risk-free rate: 6% | "
        "Regime model: KMeans k=3"
    )


else:
    st.info(
        "👈 Configure your analysis in the sidebar and click **Run Analysis** to get started.\n\n"
        "**Quick Start:**\n"
        "- Select date range (e.g., 2015–2024 for full historical context)\n"
        "- Choose strategy or 'Auto' for regime-aware recommendation\n"
        "- Click Run to see equity curves, metrics, and regime breakdown"
    )