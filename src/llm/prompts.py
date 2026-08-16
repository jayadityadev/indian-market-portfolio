"""Prompt engineering templates, context builders, and deterministic mock generator

Specialized for Indian equities (NIFTY 50, BSE SENSEX), RBI monetary policy, FII/DII liquidity,
Gaussian HMM market regimes, and quantitative risk management.
"""
from __future__ import annotations

import math
import re
from typing import Any


def _safe_float(val: Any, default: float | None = 0.0) -> float | None:
    """Safely convert value to float, handling None, string percentages, and invalid types."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return default
        try:
            if s.endswith("%"):
                return float(s[:-1].strip()) / 100.0
            return float(s)
        except (ValueError, TypeError):
            return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _format_pct(val_float: float | None, default: str = "0.0%") -> str:
    """Safely format a float value as percentage string."""
    if val_float is None:
        return default
    if math.isnan(val_float):
        return "nan"
    if math.isinf(val_float):
        return "inf" if val_float > 0 else "-inf"
    return f"{val_float * 100:.1f}%" if abs(val_float) < 10 else f"{val_float:.1f}%"


def _format_num(val_float: float | None, default: str = "0.00") -> str:
    """Safely format a float value with 2 decimal places."""
    if val_float is None:
        return default
    if math.isnan(val_float):
        return "nan"
    if math.isinf(val_float):
        return "inf" if val_float > 0 else "-inf"
    return f"{val_float:.2f}"


# ---------------------------------------------------------------------------
# Institutional Analyst System Prompt
# ---------------------------------------------------------------------------

def get_analyst_system_prompt() -> str:
    """Return the authoritative institutional quantitative analyst persona for Indian Equities."""
    return """You are the Senior Quantitative Investment Strategist and Lead Portfolio Manager for Indian Equities at an institutional asset management firm.

Your role is to produce rigorous, institutional-grade quantitative market commentary, regime diagnostics, and strategy recommendations for Indian benchmark indices (e.g., NIFTY 50, BSE SENSEX) and sectoral constituents.

### Analytical & Domain Framework:
1. **Indian Macroeconomic & Microstructure Dynamics**:
   - **NIFTY 50 Sectoral Weightings**: Heavy exposure to BFSI (~33-35%), Information Technology (~12-14%), Oil & Gas/Energy (~10-12%), FMCG (~8%), Automobile (~6-7%), and Metals/Pharma.
   - **Monetary Policy**: Reserve Bank of India (RBI) Monetary Policy Committee (MPC) repo rate stance, liquidity management, and 10-Year Indian Government Securities (G-Sec) risk-free benchmark rate (assumed ~6.50% - 7.00%).
   - **Institutional Flow Dynamics**: Interaction between Foreign Institutional Investors (FII/FPI) global risk-on/off capital flows and Domestic Institutional Investors (DII) structural liquidity buffer powered by systematic investment plans (SIPs).
   - **Volatility Surface & Sentiment**: India VIX regime benchmarks:
     * Sub-13.0: Low Volatility / Complacency / Momentum expansion.
     * 13.0 - 18.0: Normal / Equilibrium trading range.
     * 18.0 - 24.0: Elevated Volatility / Hedging demand / Sector rotation.
     * 24.0+: Tail Risk / High Uncertainty / Liquidity contraction.

2. **Quantitative Machine Learning & Statistical Models**:
   - **3-State Gaussian Hidden Markov Model (HMM)**: Classifying market conditions into `Bull` (trend persistence, positive mean returns, low-to-moderate variance), `Bear` (high downside variance, negative drift, capital preservation priority), and `Sideways` (mean-reverting oscillations, band contraction).
   - **Strategy Evaluation Suite (6 Quantitative Strategies)**: Buy & Hold, MA Crossover (Trend Following), RSI Mean Reversion (Oscillator), Momentum (Relative Strength), Bollinger Bands (Volatility Breakout/Reversal), and Dual Momentum (Absolute + Relative Trend).
   - **Risk & Attribution Metrics**: Compound Annual Growth Rate (CAGR), Annualized Volatility, Sharpe Ratio (rf=6.5%), Sortino Ratio (Downside Deviation penalty), Maximum Drawdown (peak-to-trough decline), and Calmar Ratio (CAGR / |Max Drawdown|).
   - **Monte Carlo Forward Stress Testing**: 63-day forward simulated path distribution capturing 10th percentile (worst-case tail risk), 50th percentile (median expectation), and 90th percentile (favorable scenario).

### Output Formatting Constraints:
You MUST structure your entire response using the following EXACT 5 Markdown Section Headers with their specific emoji prefixes:
## 🏛️ 1. Executive Market & Regime Diagnosis
## 🔄 2. Regime Shift Dynamics & Transition Probability
## 📊 3. Quantitative Strategy Evaluation & Justification
## 🛡️ 4. Risk Budgeting & Drawdown Guardrails
## 🚀 5. Tactical Capital Allocation & Action Plan

Be precise, analytical, quantitative, and actionable. Avoid generic filler. Incorporate exact statistics from the provided data into the narrative.
"""


# ---------------------------------------------------------------------------
# Context Formatters
# ---------------------------------------------------------------------------

def format_backtest_metrics_table(metrics: dict[str, Any] | None) -> str:
    """Format backtest metrics dictionary into a clean Markdown table."""
    if not metrics:
        return "*No backtest metrics provided.*"

    headers = ["Strategy", "CAGR", "Sharpe", "Sortino", "Max Drawdown", "Calmar", "Volatility"]
    rows = []

    for strat_name, data in metrics.items():
        if isinstance(data, dict):
            # Extract fields with safe defaults
            cagr = _safe_float(data.get("CAGR"), 0.0)
            sharpe = _safe_float(data.get("Sharpe"), 0.0)
            sortino = _safe_float(data.get("Sortino"), 0.0)
            max_dd = _safe_float(data.get("MaxDrawdown"), 0.0)
            calmar = _safe_float(data.get("Calmar"), 0.0)
            vol = _safe_float(data.get("Volatility"), 0.0)

            cagr_str = _format_pct(cagr)
            sharpe_str = _format_num(sharpe)
            sortino_str = _format_num(sortino)
            max_dd_str = _format_pct(max_dd)
            calmar_str = _format_num(calmar)
            vol_str = _format_pct(vol)

            rows.append(f"| {strat_name} | {cagr_str} | {sharpe_str} | {sortino_str} | {max_dd_str} | {calmar_str} | {vol_str} |")

    if not rows:
        return "*No valid strategy metrics available.*"

    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---" for _ in headers]) + " |",
    ] + rows
    return "\n".join(table)


def format_regime_context(regime_data: dict[str, Any] | None) -> str:
    """Format Gaussian HMM regime and ML classifier probability context."""
    if not regime_data:
        return "- Current Market Regime: **Bull** (Default Baseline)\n- Strategy Classification Confidence: High"

    lines = []
    current_regime = regime_data.get("current_regime", "Bull") or "Bull"
    lines.append(f"- **Current Detected Regime (HMM)**: `{current_regime}`")

    if "regime_distribution" in regime_data and isinstance(regime_data["regime_distribution"], dict):
        dist_str = ", ".join([f"{k}: {v} days" for k, v in regime_data["regime_distribution"].items()])
        lines.append(f"- **Historical Regime Distribution**: {dist_str}")

    if "transition_matrix" in regime_data:
        matrix = regime_data["transition_matrix"]
        lines.append(f"- **HMM Transition Probability Matrix**: `{matrix}`")

    probs = regime_data.get("probabilities")
    if probs:
        if isinstance(probs, dict):
            prob_items = []
            for k, v in probs.items():
                if v is None:
                    continue
                v_num = _safe_float(v, default=None)
                if v_num is not None:
                    prob_str = f"{v_num * 100:.1f}%" if v_num <= 1.0 else f"{v_num:.1f}%"
                    prob_items.append(f"{k}: {prob_str}")
            if prob_items:
                lines.append(f"- **ML Strategy Suitability Probabilities**: {', '.join(prob_items)}")
        elif isinstance(probs, list):
            prob_items = []
            for p in probs:
                if isinstance(p, dict):
                    s = p.get("strategy", "")
                    prob_raw = p.get("probability")
                    if prob_raw is not None:
                        prob_val = _safe_float(prob_raw, default=None)
                        if prob_val is not None:
                            prob_str = f"{prob_val * 100:.1f}%" if prob_val <= 1.0 else f"{prob_val:.1f}%"
                            prob_items.append(f"{s}: {prob_str}")
            if prob_items:
                lines.append(f"- **ML Strategy Suitability Probabilities**: {', '.join(prob_items)}")

    if "recommended_strategy" in regime_data and regime_data["recommended_strategy"]:
        lines.append(f"- **Recommended Strategy**: `{regime_data['recommended_strategy']}`")

    return "\n".join(lines)


def format_risk_forecast_context(risk_metrics: dict[str, Any] | None) -> str:
    """Format Monte Carlo risk forecast and drawdown percentiles."""
    if not risk_metrics:
        return "- **Monte Carlo 63-Day Forward Drawdown Simulation**: Baseline risk profile."

    lines = ["- **Monte Carlo 63-Day Forward Drawdown Projections**:"]
    worst_10 = risk_metrics.get("worst_case_10")
    med_50 = risk_metrics.get("median_50")
    best_90 = risk_metrics.get("best_case_90")
    exposure = risk_metrics.get("recommended_exposure")

    def _format_risk_val(val: Any) -> str | None:
        if val is None:
            return None
        v_float = _safe_float(val, default=None)
        if v_float is not None:
            return _format_pct(v_float)
        if isinstance(val, str) and val.strip():
            return val.strip()
        return None

    w_str = _format_risk_val(worst_10)
    if w_str is not None:
        lines.append(f"  * **10th Percentile (Tail Risk / Adverse Scenario)**: {w_str}")

    m_str = _format_risk_val(med_50)
    if m_str is not None:
        lines.append(f"  * **50th Percentile (Median Expected Drawdown)**: {m_str}")

    b_str = _format_risk_val(best_90)
    if b_str is not None:
        lines.append(f"  * **90th Percentile (Favorable Path Drawdown)**: {b_str}")

    if exposure:
        lines.append(f"- **Suggested Exposure Limit**: `{exposure}`")

    return "\n".join(lines)


def build_analyst_user_prompt(
    backtest_metrics: dict[str, Any] | None = None,
    regime_data: dict[str, Any] | None = None,
    risk_metrics: dict[str, Any] | None = None,
    ticker: str = "^NSEI",
    custom_prompt: str | None = None,
) -> str:
    """Construct the comprehensive structured prompt for the AI Market Analyst."""
    metrics_table = format_backtest_metrics_table(backtest_metrics)
    regime_block = format_regime_context(regime_data)
    risk_block = format_risk_forecast_context(risk_metrics)

    user_prompt_str = ""
    if isinstance(custom_prompt, str) and custom_prompt.strip():
        user_prompt_str = custom_prompt.strip()
    elif isinstance(custom_prompt, dict):
        user_prompt_str = custom_prompt.get("outlook") or custom_prompt.get("disclaimer") or str(custom_prompt)

    user_instructions = (
        f"\n### User Custom Request / Specific Focus:\n{user_prompt_str}\n"
        if user_prompt_str
        else ""
    )

    prompt = f"""### Market Analysis Request for Asset: {ticker}

Please generate an institutional AI Market Analyst report for **{ticker}** incorporating the quantitative data below:

### 1. Gaussian HMM Regime Detection & ML Strategy Suitability:
{regime_block}

### 2. Backtest Performance Metrics Across 6 Quantitative Strategies:
{metrics_table}

### 3. Quantitative Risk Forecast & Monte Carlo Simulation:
{risk_block}
{user_instructions}
### Required Deliverables:
Provide a rigorous, institutional analysis following the exact 5-section structure:
1. `## 🏛️ 1. Executive Market & Regime Diagnosis` (Macro context, NIFTY 50 sectoral posture, HMM state diagnosis, FII/DII balance)
2. `## 🔄 2. Regime Shift Dynamics & Transition Probability` (Transition matrix analysis, stability, potential catalysts for regime shifts)
3. `## 📊 3. Quantitative Strategy Evaluation & Justification` (Detailed performance attribution of top strategy vs benchmarks using CAGR, Sharpe, Sortino, and Calmar)
4. `## 🛡️ 4. Risk Budgeting & Drawdown Guardrails` (Monte Carlo tail risk, India VIX implications, stop-loss & max drawdown thresholds)
5. `## 🚀 5. Tactical Capital Allocation & Action Plan` (Concrete capital deployment percentages, hedge ratios, and execution roadmap)
"""
    return prompt


# ---------------------------------------------------------------------------
# Deterministic High-Fidelity Mock Report Generator
# ---------------------------------------------------------------------------

def generate_mock_report(
    system_prompt: str = "",
    user_prompt: str = "",
    model: str = "mock-deterministic-v1",
    backtest_metrics: dict[str, Any] | None = None,
    regime_data: dict[str, Any] | None = None,
    risk_metrics: dict[str, Any] | None = None,
    ticker: str = "^NSEI",
) -> str:
    """Generate a high-fidelity, deterministic 5-section Markdown analyst report.

    Guaranteed 100% offline uptime with realistic domain metrics and dynamic context extraction.
    """
    # 1. Resolve ticker
    resolved_ticker = ticker.strip() if ticker and isinstance(ticker, str) and ticker.strip() else "^NSEI"
    if resolved_ticker == "^NSEI" and user_prompt:
        m = re.search(r"### Market Analysis Request for Asset:\s*`?([A-Za-z0-9_\^\.\-]+)`?", user_prompt)
        if not m:
            m = re.search(r"- \*\*Ticker / Asset\*\*:\s*`?([A-Za-z0-9_\^\.\-]+)`?", user_prompt)
        if not m:
            m = re.search(r"- \*\*Primary Ticker / Asset\*\*:\s*`?([A-Za-z0-9_\^\.\-]+)`?", user_prompt)
        if not m:
            m = re.search(r"Asset:\s*`?([A-Za-z0-9_\^\.\-]+)`?", user_prompt)
        if m:
            resolved_ticker = m.group(1).strip()

    # 2. Resolve regime
    current_regime = "Bull"
    if regime_data and "current_regime" in regime_data and regime_data["current_regime"]:
        current_regime = str(regime_data["current_regime"]).strip()
    elif user_prompt:
        m = re.search(r"- \*\*Current Detected Regime \(HMM\)\*\*:\s*`?([^`\n\r]+)`?", user_prompt)
        if not m:
            m = re.search(r"- \*\*Detected Market Regime\*\*:\s*`?([^`\n\r]+)`?", user_prompt)
        if not m:
            m = re.search(r"- \*\*Active Quantitative State\*\*:\s*\*\*([^*]+)\*\*", user_prompt)
        if m:
            extracted = m.group(1).replace("Market Regime", "").replace("Regime", "").strip()
            if "bear" in extracted.lower():
                current_regime = "Bear"
            elif "sideway" in extracted.lower():
                current_regime = "Sideways"
            else:
                current_regime = "Bull"
        else:
            if re.search(r"\bBear\b", user_prompt, re.IGNORECASE):
                current_regime = "Bear"
            elif re.search(r"\bSideways\b", user_prompt, re.IGNORECASE):
                current_regime = "Sideways"
            else:
                current_regime = "Bull"

    # Normalize regime string
    if "bear" in current_regime.lower():
        current_regime = "Bear"
    elif "sideway" in current_regime.lower():
        current_regime = "Sideways"
    else:
        current_regime = "Bull"

    # 3. Resolve recommended strategy
    recommended_strategy = "Momentum"
    if current_regime == "Bear":
        recommended_strategy = "RSI Mean Reversion"
    elif current_regime == "Sideways":
        recommended_strategy = "Bollinger Bands"

    if regime_data and "recommended_strategy" in regime_data and regime_data["recommended_strategy"]:
        recommended_strategy = str(regime_data["recommended_strategy"]).strip()
    elif user_prompt:
        m = re.search(r"- \*\*Recommended Strategy\*\*:\s*`?([^`\n\r]+)`?", user_prompt)
        if not m:
            m = re.search(r"- \*\*Recommended Quantitative Framework\*\*:\s*`?([^`\n\r]+)`?", user_prompt)
        if m:
            rec_extracted = m.group(1).strip().strip("*")
            if rec_extracted:
                recommended_strategy = rec_extracted

    # 4. Resolve recommended exposure
    recommended_exposure = "80% - 100% Equity Exposure"
    if risk_metrics and "recommended_exposure" in risk_metrics and risk_metrics["recommended_exposure"]:
        recommended_exposure = str(risk_metrics["recommended_exposure"]).strip()
    elif user_prompt:
        m = re.search(r"- \*\*Suggested Exposure Limit\*\*:\s*`?([^`\n\r]+)`?", user_prompt)
        if m:
            recommended_exposure = m.group(1).strip()

    # 5. Resolve risk percentiles
    worst_case_10 = "-12.4%"
    median_50 = "-5.8%"
    best_case_90 = "-1.8%"

    if risk_metrics:
        if "worst_case_10" in risk_metrics and risk_metrics["worst_case_10"] is not None:
            w = risk_metrics["worst_case_10"]
            w_f = _safe_float(w, default=None)
            worst_case_10 = _format_pct(w_f) if w_f is not None else str(w)
        if "median_50" in risk_metrics and risk_metrics["median_50"] is not None:
            m_val = risk_metrics["median_50"]
            m_f = _safe_float(m_val, default=None)
            median_50 = _format_pct(m_f) if m_f is not None else str(m_val)
        if "best_case_90" in risk_metrics and risk_metrics["best_case_90"] is not None:
            b = risk_metrics["best_case_90"]
            b_f = _safe_float(b, default=None)
            best_case_90 = _format_pct(b_f) if b_f is not None else str(b)
    elif user_prompt:
        m10 = re.search(r"10th Percentile[^\*:]*(?:\*\*|\:)?:\s*(?:\*\*)?([^\n\r\*]+)(?:\*\*)?", user_prompt)
        if m10:
            worst_case_10 = m10.group(1).strip()
        m50 = re.search(r"50th Percentile[^\*:]*(?:\*\*|\:)?:\s*(?:\*\*)?([^\n\r\*]+)(?:\*\*)?", user_prompt)
        if m50:
            median_50 = m50.group(1).strip()
        m90 = re.search(r"90th Percentile[^\*:]*(?:\*\*|\:)?:\s*(?:\*\*)?([^\n\r\*]+)(?:\*\*)?", user_prompt)
        if m90:
            best_case_90 = m90.group(1).strip()

    # 6. Parse backtest metrics table
    top_cagr = "18.6%"
    top_sharpe = "1.38"
    top_sortino = "1.95"
    top_maxdd = "-13.8%"
    top_calmar = "1.35"
    bh_cagr = "12.4%"
    bh_sharpe = "0.82"
    bh_maxdd = "-28.2%"

    parsed_table: dict[str, dict[str, str]] = {}
    if user_prompt:
        for line in user_prompt.splitlines():
            line_str = line.strip()
            if line_str.startswith("|") and line_str.endswith("|"):
                parts = [p.strip() for p in line_str.split("|")[1:-1]]
                if (
                    len(parts) >= 6
                    and parts[0] != "Strategy"
                    and not parts[0].startswith(":-")
                    and not parts[0].startswith("Performance Dimension")
                ):
                    s_name = parts[0]
                    parsed_table[s_name] = {
                        "CAGR": parts[1] if len(parts) > 1 else "0.0%",
                        "Sharpe": parts[2] if len(parts) > 2 else "0.00",
                        "Sortino": parts[3] if len(parts) > 3 else "0.00",
                        "MaxDrawdown": parts[4] if len(parts) > 4 else "0.0%",
                        "Calmar": parts[5] if len(parts) > 5 else "0.00",
                        "Volatility": parts[6] if len(parts) > 6 else "0.0%",
                    }

    if backtest_metrics and recommended_strategy in backtest_metrics:
        strat_m = backtest_metrics[recommended_strategy]
        if isinstance(strat_m, dict):
            cagr_val = _safe_float(strat_m.get("CAGR"), 0.186)
            top_cagr = _format_pct(cagr_val)
            top_sharpe = _format_num(_safe_float(strat_m.get("Sharpe"), 1.38))
            top_sortino = _format_num(_safe_float(strat_m.get("Sortino"), 1.95))
            dd_val = _safe_float(strat_m.get("MaxDrawdown"), -0.138)
            top_maxdd = _format_pct(dd_val)
            top_calmar = _format_num(_safe_float(strat_m.get("Calmar"), 1.35))
    elif parsed_table:
        if recommended_strategy in parsed_table:
            top_cagr = parsed_table[recommended_strategy]["CAGR"]
            top_sharpe = parsed_table[recommended_strategy]["Sharpe"]
            top_sortino = parsed_table[recommended_strategy]["Sortino"]
            top_maxdd = parsed_table[recommended_strategy]["MaxDrawdown"]
            top_calmar = parsed_table[recommended_strategy]["Calmar"]
        else:
            non_bh = [k for k in parsed_table if k not in ("Buy & Hold", "Buy and Hold", "Benchmark")]
            chosen = non_bh[0] if non_bh else list(parsed_table.keys())[0]
            top_cagr = parsed_table[chosen]["CAGR"]
            top_sharpe = parsed_table[chosen]["Sharpe"]
            top_sortino = parsed_table[chosen]["Sortino"]
            top_maxdd = parsed_table[chosen]["MaxDrawdown"]
            top_calmar = parsed_table[chosen]["Calmar"]

    if backtest_metrics and "Buy & Hold" in backtest_metrics:
        bh_m = backtest_metrics["Buy & Hold"]
        if isinstance(bh_m, dict):
            c_val = _safe_float(bh_m.get("CAGR"), 0.124)
            bh_cagr = _format_pct(c_val)
            bh_sharpe = _format_num(_safe_float(bh_m.get("Sharpe"), 0.82))
            d_val = _safe_float(bh_m.get("MaxDrawdown"), -0.282)
            bh_maxdd = _format_pct(d_val)
    elif parsed_table:
        for bh_name in ("Buy & Hold", "Buy and Hold", "Benchmark"):
            if bh_name in parsed_table:
                bh_cagr = parsed_table[bh_name]["CAGR"]
                bh_sharpe = parsed_table[bh_name]["Sharpe"]
                bh_maxdd = parsed_table[bh_name]["MaxDrawdown"]
                break

    # Domain specific narrative per regime
    if current_regime == "Bull":
        regime_diag = f"The 3-state Gaussian Hidden Markov Model (HMM) classifies **{resolved_ticker}** into a robust **Bull Regime**. Market microstructure exhibits persistent upward drift with muted downside dispersion. Heavyweight constituents across BFSI (HDFC Bank, ICICI Bank) and IT services are commanding positive institutional breadth, underpinned by resilient domestic mutual fund SIP inflows (~₹21,000+ Cr monthly) absorbing periodic foreign portfolio investor (FPI) volatility."
        transition_commentary = f"Transition probability estimates indicate high regime persistence ($P(S_{{t+1}}=\\text{{Bull}} | S_t=\\text{{Bull}}) \\approx 0.88$). Key macro watchpoints for potential degradation include RBI MPC rate policy commentary, crude oil price spikes above $85/bbl affecting current account deficits, and India VIX breaches above 17.5."
        strat_justification = f"In this persistent upward drift regime, **{recommended_strategy}** significantly outperforms benchmark Buy & Hold. By compounding winners and reducing drag from lagging sectors, the strategy achieves an annualized **CAGR of {top_cagr}** and **Sharpe Ratio of {top_sharpe}** (vs Buy & Hold CAGR of {bh_cagr} and Sharpe of {bh_sharpe}). Downside risk is tightly controlled with a Sortino Ratio of **{top_sortino}** and Calmar Ratio of **{top_calmar}**."
        alloc_action = f"- **Equity Capital Allocation**: 85% - 95% actively deployed in high-momentum constituents.\n- **Cash / Overnight Collateral**: 5% - 15% dry powder.\n- **Hedging**: OTM NIFTY Put options (~200 pts below spot) to cap tail risks."
    elif current_regime == "Bear":
        regime_diag = f"The 3-state Gaussian Hidden Markov Model (HMM) classifies **{resolved_ticker}** into a high-volatility **Bear Regime**. Price dynamics reflect elevated downside variance, negative momentum, and persistent distribution from Foreign Institutional Investors (FII). Breadth deterioration across high-beta sectors (Metals, Realty, Auto) warrants strict defensive capital preservation."
        transition_commentary = f"HMM persistence indicates ongoing risk clustering ($P(S_{{t+1}}=\\text{{Bear}} | S_t=\\text{{Bear}}) \\approx 0.79$). Transition toward a stabilization or sideways state requires sustained India VIX contraction below 15.5 and positive net institutional liquidity confluence."
        strat_justification = f"During bear regimes, unhedged Buy & Hold suffers severe capital degradation (Max Drawdown {bh_maxdd}). The recommended **{recommended_strategy}** dynamic risk posture limits exposure, capturing oversold bounces while capping structural drawdown to **{top_maxdd}**, producing a resilient Calmar Ratio of **{top_calmar}**."
        alloc_action = f"- **Equity Capital Allocation**: 30% - 45% (Defensive FMCG/Pharma and Low-Beta dividend yielders).\n- **Cash / Liquid TREPS / Arbitrage**: 45% - 55% capital preservation.\n- **Hedging**: Systematic Bear Put Spreads or trailing index futures short overlays."
    else:  # Sideways
        regime_diag = f"The 3-state Gaussian Hidden Markov Model (HMM) identifies **{resolved_ticker}** in a range-bound **Sideways / Mean-Reverting Regime**. India VIX is oscillating between 12.5 and 15.0 with low directional conviction. Sector rotation is rapid between domestic cyclicals and defensive export plays without decisive index breakout."
        transition_commentary = f"HMM state persistence reflects equilibrium consolidation ($P(S_{{t+1}}=\\text{{Sideways}} | S_t=\\text{{Sideways}}) \\approx 0.82$). A structural breakout requires high-volume expansion and institutional sector leadership."
        strat_justification = f"In range-bound markets, trend-following models suffer whipsaws. **{recommended_strategy}** excels by exploiting mean-reverting boundary oscillations, yielding a **CAGR of {top_cagr}** with an exceptional Sortino Ratio of **{top_sortino}** and bounded drawdown of **{top_maxdd}**."
        alloc_action = f"- **Equity Capital Allocation**: 60% - 70% deployed in mean-reversion swing baskets.\n- **Cash / Overnight Arbitrage**: 25% - 30% opportunistic liquidity.\n- **Options Overlay**: Iron Condors / Short Strangle strategies around 1-standard-deviation support and resistance."

    report = f"""## 🏛️ 1. Executive Market & Regime Diagnosis

{regime_diag}

- **Primary Ticker / Asset**: `{resolved_ticker}`
- **Active Quantitative State**: **{current_regime} Market Regime**
- **Recommended Quantitative Framework**: **{recommended_strategy}**
- **Effective Exposure Posture**: `{recommended_exposure}`

---

## 🔄 2. Regime Shift Dynamics & Transition Probability

{transition_commentary}

- **State Stability Assessment**: High internal coherence within the `{current_regime}` state.
- **Liquidity Buffer**: DII institutional liquidity remains structurally supportive through monthly SIP inflows, mitigating steep flash-crash risks.
- **Key Invalidation Signals**: Rapid India VIX expansion (>18.0) or unexpected RBI MPC liquidity tightening.

---

## 📊 3. Quantitative Strategy Evaluation & Justification

{strat_justification}

### Quantitative Metric Comparison:
| Performance Dimension | Benchmark (Buy & Hold) | Recommended ({recommended_strategy}) | Alpha / Spread |
| :--- | :--- | :--- | :--- |
| **Annualized Return (CAGR)** | {bh_cagr} | **{top_cagr}** | *Outperformance* |
| **Sharpe Ratio (rf=6.5%)** | {bh_sharpe} | **{top_sharpe}** | *Superior Risk-Adjusted Return* |
| **Sortino Ratio (Downside)** | 1.10 | **{top_sortino}** | *Strong Downside Protection* |
| **Maximum Drawdown** | {bh_maxdd} | **{top_maxdd}** | *Controlled Capital Risk* |
| **Calmar Ratio** | 0.44 | **{top_calmar}** | *Enhanced Recovery Speed* |

---

## 🛡️ 4. Risk Budgeting & Drawdown Guardrails

Risk management guardrails are parameterized using 63-trading-day forward Monte Carlo simulations (1,000 iterations):

- **10th Percentile Tail Risk (Stress Scenario)**: **{worst_case_10}**
- **50th Percentile Median Drawdown**: **{median_50}**
- **90th Percentile Favorable Scenario**: **{best_case_90}**
- **Hard Stop-Loss Floor**: Systematic exit triggered if portfolio equity breaches a trailing 8.5% drawdown.
- **Volatility Sizing**: Scale position sizes inversely to rolling 20-day realized volatility.

---

## 🚀 5. Tactical Capital Allocation & Action Plan

Based on the quantitative synthesis of HMM regimes, strategy attribution, and Monte Carlo risk boundaries, the institutional action plan is:

{alloc_action}

### Execution Checklist:
1. **Rebalance Cadence**: Bi-weekly regime validation check or immediate rebalance upon HMM state transition trigger.
2. **Execution Slippage & Impact Cost**: Limit orders routed via VWAP algorithms across high-liquidity NSE trading windows (10:00 - 14:30 IST).
3. **Institutional Governance**: Maintain strict adherence to risk budgets and stop-loss limits without emotional discretion.
"""
    return report.strip()
