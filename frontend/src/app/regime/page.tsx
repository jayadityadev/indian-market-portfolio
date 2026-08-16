"use client";

import React, { useEffect, useState } from "react";
import { fetchRegime, RegimeResponse } from "@/lib/regime";
import { errorMessage } from "@/lib/errors";
import { Activity, TrendingUp, TrendingDown, RefreshCw, Shield, Layers, BarChart2, HelpCircle, CheckCircle2, AlertTriangle, XCircle, Info } from "lucide-react";

const REGIME_CONFIG: Record<string, { label: string; desc: string; icon: typeof TrendingUp; color: string; bg: string; stance: string; favored: string }> = {
  Bull: {
    label: "Bull Regime (Expansion)",
    desc: "Sustained upward drift, moderate volatility, and positive momentum across large caps.",
    icon: TrendingUp,
    color: "var(--bull)",
    bg: "var(--bull-bg)",
    stance: "Full Exposure (100% Equity)",
    favored: "Momentum, Buy & Hold, MA Crossover",
  },
  Bear: {
    label: "Bear Regime (Contraction)",
    desc: "High annualized volatility, sharp drawdowns, and negative return distribution.",
    icon: TrendingDown,
    color: "var(--bear)",
    bg: "var(--bear-bg)",
    stance: "Defensive (0–50% Equity / Cash Hedge)",
    favored: "Dual Momentum (Cash Exit), Risk Off",
  },
  Sideways: {
    label: "Sideways Regime (Consolidation)",
    desc: "Range-bound price action with low directional trend and frequent mean-reverting swings.",
    icon: Activity,
    color: "var(--sideways)",
    bg: "var(--sideways-bg)",
    stance: "Selective Exposure (75% Position Sizing)",
    favored: "RSI Mean Reversion, Bollinger Bands",
  },
};

const STRATEGY_EXPLAINERS = [
  {
    name: "Momentum",
    rule: "Buys top-performing stocks over a 12-month lookback window.",
    bull: { rating: "★★★★★", tag: "Highest Alpha", desc: "Sustained upward trends amplify winners; generates maximum compounding.", status: "win" },
    sideways: { rating: "★★☆☆☆", tag: "False Breakouts", desc: "Choppy range whipsaws momentum triggers; buys at tops right before pullbacks.", status: "neutral" },
    bear: { rating: "★☆☆☆☆", tag: "Severe Drawdown", desc: "Continues holding falling assets until lookback updates; suffers massive drops.", status: "lose" },
  },
  {
    name: "MA Crossover",
    rule: "Buys when Fast MA (50-day) crosses above Slow MA (200-day); sells when it crosses below.",
    bull: { rating: "★★★★☆", tag: "Trend Capture", desc: "Reliably stays in long positions throughout multi-month bull runs.", status: "win" },
    sideways: { rating: "★☆☆☆☆", tag: "Whipsaw Losses", desc: "Repeated false crossovers generate multiple small transaction and friction losses.", status: "lose" },
    bear: { rating: "★★★☆☆", tag: "Timely Cash Exit", desc: "Death Cross triggers early exit, preserving cash during the deepest crashes.", status: "win" },
  },
  {
    name: "RSI Mean Reversion",
    rule: "Buys when RSI < 30 (oversold) and sells when RSI > 70 (overbought).",
    bull: { rating: "★★☆☆☆", tag: "Premature Exit", desc: "Sells early as soon as RSI hits 70, missing the explosive middle of bull runs.", status: "neutral" },
    sideways: { rating: "★★★★★", tag: "Optimal Alpha", desc: "Thrives in range-bound markets by buying support and selling resistance.", status: "win" },
    bear: { rating: "★☆☆☆☆", tag: "Catching Knives", desc: "RSI stays oversold for months during crashes; repeatedly buys into steep drops.", status: "lose" },
  },
  {
    name: "Bollinger Bands",
    rule: "Buys at Lower Band (2 standard deviations below 20 MA), sells at Upper Band.",
    bull: { rating: "★★★☆☆", tag: "Walking the Band", desc: "Price 'walks' upper band during strong rallies, causing early exits or short stalls.", status: "neutral" },
    sideways: { rating: "★★★★★", tag: "High Win Rate", desc: "Standard deviation envelopes act as solid boundaries, capturing mean-reversion swings.", status: "win" },
    bear: { rating: "★★☆☆☆", tag: "Band Expansion", desc: "Lower band rapidly widens downward; stop-outs occur frequently.", status: "neutral" },
  },
  {
    name: "Dual Momentum",
    rule: "Combines Relative Momentum (vs peers) + Absolute Momentum (must beat cash/T-Bills).",
    bull: { rating: "★★★★☆", tag: "Strong Alpha", desc: "Invests in high-relative-strength equities when broad market trend is positive.", status: "win" },
    sideways: { rating: "★★★☆☆", tag: "Controlled Drag", desc: "Moderate turnover with modest returns while filtering out weak sectors.", status: "neutral" },
    bear: { rating: "★★★★★", tag: "100% Cash Defense", desc: "Absolute filter fails when index drops below cash rate $\\rightarrow$ moves 100% to risk-free cash.", status: "win" },
  },
  {
    name: "Buy & Hold (Baseline)",
    rule: "100% invested in index at all times with zero trading turnover.",
    bull: { rating: "★★★★★", tag: "Max Compound", desc: "Zero slippage, zero tax friction, and full capture of long-term economic expansion.", status: "win" },
    sideways: { rating: "★★★☆☆", tag: "Stagnant", desc: "Capital is locked up with zero alpha generation during multi-year consolidations.", status: "neutral" },
    bear: { rating: "★☆☆☆☆", tag: "100% Crash Hit", desc: "Takes 100% of historical crashes (e.g. -50% in 2008, -38% in 2020) with no defense.", status: "lose" },
  },
];

export default function RegimePage() {
  const [data, setData] = useState<RegimeResponse | null>(null);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [expandedStrategy, setExpandedStrategy] = useState<string | null>(null);

  const loadData = async () => {
    setLoading(true);
    setError("");
    try {
      const result = await fetchRegime();
      setData(result);
    } catch (err: unknown) {
      setError(errorMessage(err, "Failed to load causal regime data"));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadData();
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  const total = data ? (data.total_days || Object.values(data.regime_distribution).reduce((a, b) => a + b, 0)) : 0;
  const currentConfig = data ? (REGIME_CONFIG[data.current_regime] || REGIME_CONFIG.Sideways) : null;
  const CurrentIcon = currentConfig?.icon || Activity;

  return (
    <div className="subpage-container">
      {/* Header */}
      <div className="subpage-header">
        <div className="subpage-badge">
          <Layers size={13} />
          <span>Gaussian Hidden Markov Model • 3-State Classifier</span>
        </div>
        <h1 className="subpage-title">Market Regime Intelligence</h1>
        <p className="subpage-desc">
          Statistically unsupervised Gaussian HMM state discovery on NIFTY 50. Identifies market expansion,
          contraction, and consolidation phases to condition trading strategies and dynamic exposure sizing.
        </p>
      </div>

      {loading && (
        <div className="card" style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: 280, gap: 12 }}>
          <RefreshCw size={20} className="spin" style={{ color: "var(--accent-100)" }} />
          <span style={{ color: "var(--text-200)", fontSize: 14 }}>Computing causal regime distributions...</span>
        </div>
      )}

      {error && (
        <div className="error-banner">
          <div>{error}</div>
          <button onClick={loadData} className="export-btn" style={{ marginLeft: "auto" }}>
            <RefreshCw size={12} /> Retry
          </button>
        </div>
      )}

      {data && currentConfig && (
        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          {/* Current Regime Hero Card */}
          <div
            className="hero-card"
            style={{
              background: `linear-gradient(135deg, ${currentConfig.bg}, var(--bg-200))`,
              borderColor: currentConfig.color,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 16 }}>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                  <span
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 6,
                      padding: "4px 12px",
                      borderRadius: 999,
                      fontSize: 11,
                      fontWeight: 700,
                      background: currentConfig.bg,
                      color: currentConfig.color,
                      border: `1px solid ${currentConfig.color}`,
                    }}
                  >
                    <CurrentIcon size={14} /> Current Market Phase
                  </span>
                  <span style={{ fontSize: 11, color: "var(--text-200)" }}>NIFTY 50 (Latest Session)</span>
                </div>
                <div style={{ fontSize: 32, fontWeight: 800, color: "var(--text-100)", marginBottom: 6 }}>
                  {currentConfig.label}
                </div>
                <p style={{ fontSize: 13, color: "var(--text-200)", maxWidth: 640, lineHeight: 1.6, margin: 0 }}>
                  {currentConfig.desc}
                </p>
              </div>

              <div style={{ background: "var(--bg-200)", padding: "16px 20px", borderRadius: 16, border: "1px solid var(--card-border)", minWidth: 220 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text-200)", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: 4 }}>
                  Recommended Stance
                </div>
                <div style={{ fontSize: 15, fontWeight: 700, color: currentConfig.color, marginBottom: 8 }}>
                  {currentConfig.stance}
                </div>
                <div style={{ fontSize: 11, color: "var(--text-200)" }}>
                  Optimal strategies: <strong style={{ color: "var(--text-100)" }}>{currentConfig.favored}</strong>
                </div>
              </div>
            </div>
          </div>

          {/* Plain English Guide for Beginners */}
          <div className="card" style={{ background: "linear-gradient(145deg, var(--bg-100), var(--bg-200))", border: "1px solid var(--card-border)" }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text-100)", marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
              <HelpCircle size={16} style={{ color: "var(--accent-100)" }} />
              <span>What does this page tell you? (Beginner Guide)</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 14, marginTop: 12 }}>
              <div style={{ padding: 14, borderRadius: 14, background: "var(--bg-200)", border: "1px solid var(--bg-300)" }}>
                <strong style={{ fontSize: 12, color: "var(--text-100)", display: "block", marginBottom: 4 }}>1. Markets have &quot;Weather&quot; (Regimes)</strong>
                <span style={{ fontSize: 11, color: "var(--text-200)", lineHeight: 1.5 }}>
                  Just like weather changes between sunny, stormy, or cloudy, the stock market cycles between Bull (growth), Bear (decline), and Sideways (flat).
                </span>
              </div>
              <div style={{ padding: 14, borderRadius: 14, background: "var(--bg-200)", border: "1px solid var(--bg-300)" }}>
                <strong style={{ fontSize: 12, color: "var(--text-100)", display: "block", marginBottom: 4 }}>2. No Strategy Wins in All Weather</strong>
                <span style={{ fontSize: 11, color: "var(--text-200)", lineHeight: 1.5 }}>
                  Buying & holding works in Bull markets but loses in Bear markets. Knowing the active regime tells you which rule to follow today.
                </span>
              </div>
              <div style={{ padding: 14, borderRadius: 14, background: "var(--bg-200)", border: "1px solid var(--bg-300)" }}>
                <strong style={{ fontSize: 12, color: "var(--text-100)", display: "block", marginBottom: 4 }}>3. Mathematical Detection (HMM)</strong>
                <span style={{ fontSize: 11, color: "var(--text-200)", lineHeight: 1.5 }}>
                  Our Gaussian Hidden Markov Model automatically detects shifts from daily price drift and volatility without human guessing or emotion.
                </span>
              </div>
            </div>
          </div>

          {/* Regime Distribution Breakdown */}
          <div className="card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <div className="card-title" style={{ margin: 0 }}>
                <BarChart2 size={16} /> Historical Regime Distribution ({total} Trading Days)
              </div>
              <span style={{ fontSize: 11, color: "var(--text-200)" }}>
                Source: {data.regime_source || "Walk-Forward Dataset"}
              </span>
            </div>

            {/* Visual Distribution Progress Bar */}
            <div style={{ height: 28, borderRadius: 10, overflow: "hidden", display: "flex", marginBottom: 18, border: "1px solid var(--card-border)" }}>
              {Object.entries(data.regime_distribution).map(([regime, count]) => {
                const cfg = REGIME_CONFIG[regime] || REGIME_CONFIG.Sideways;
                const pct = total > 0 ? (count / total) * 100 : 0;
                return (
                  <div
                    key={regime}
                    title={`${regime}: ${count} days (${pct.toFixed(1)}%)`}
                    style={{
                      flex: `${pct} 0 0`,
                      background: cfg.color,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      color: "#fff",
                      fontSize: 10,
                      fontWeight: 700,
                      transition: "all 0.3s ease",
                    }}
                  >
                    {pct > 12 ? `${regime} ${pct.toFixed(0)}%` : ""}
                  </div>
                );
              })}
            </div>

            {/* Metric Cards for Each Regime */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 14 }}>
              {["Bull", "Sideways", "Bear"].map((regime) => {
                const count = data.regime_distribution[regime] || 0;
                const pct = total > 0 ? ((count / total) * 100).toFixed(1) : "0.0";
                const cfg = REGIME_CONFIG[regime] || REGIME_CONFIG.Sideways;
                const isCurrent = data.current_regime === regime;

                return (
                  <div
                    key={regime}
                    className="metric-card"
                    style={{
                      borderColor: isCurrent ? cfg.color : "var(--card-border)",
                      position: "relative",
                      background: isCurrent ? `linear-gradient(180deg, ${cfg.bg}, var(--bg-200))` : "var(--bg-200)",
                    }}
                  >
                    {isCurrent && (
                      <span
                        style={{
                          position: "absolute",
                          top: 12,
                          right: 12,
                          fontSize: 9,
                          fontWeight: 800,
                          padding: "2px 8px",
                          borderRadius: 999,
                          background: cfg.color,
                          color: "#fff",
                          textTransform: "uppercase",
                        }}
                      >
                        Active Now
                      </span>
                    )}
                    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
                      <div style={{ width: 8, height: 8, borderRadius: "50%", background: cfg.color }} />
                      <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text-100)" }}>{regime} Market</span>
                    </div>
                    <div style={{ fontSize: 24, fontWeight: 800, color: cfg.color, marginBottom: 4 }}>
                      {pct}%
                    </div>
                    <div style={{ fontSize: 11, color: "var(--text-200)", marginBottom: 10 }}>
                      {count.toLocaleString()} historical trading sessions
                    </div>
                    <div style={{ fontSize: 11, color: "var(--text-100)", borderTop: "1px solid var(--bg-300)", paddingTop: 8, lineHeight: 1.4 }}>
                      <span style={{ color: "var(--text-200)" }}>Stance:</span> {cfg.stance}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Deep Explainability: Strategy Suitability Matrix & Why It Works */}
          <div className="card">
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8, flexWrap: "wrap", gap: 10 }}>
              <div className="card-title" style={{ margin: 0 }}>
                <Shield size={16} /> Strategy Suitability by Market Regime (Explainability Matrix)
              </div>
              <span style={{ fontSize: 11, color: "var(--accent-100)", fontWeight: 600 }}>
                Click any row for strategy mechanics & why it wins/fails
              </span>
            </div>

            <p style={{ fontSize: 13, color: "var(--text-200)", lineHeight: 1.6, marginBottom: 16 }}>
              <strong>How this is determined:</strong> Empirical quantitative backtests across 10+ years of NIFTY 50 price history
              sliced by Gaussian HMM regime periods. Star ratings reflect real risk-adjusted returns (Sharpe Ratio), annualized CAGR,
              and maximum drawdown avoidance during each regime phase.
            </p>

            {/* Color Legend & Semantics */}
            <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap", padding: "8px 14px", background: "var(--bg-100)", borderRadius: 10, marginBottom: 14, fontSize: 11 }}>
              <span style={{ color: "var(--text-200)", fontWeight: 700 }}>Color Legend:</span>
              <span style={{ display: "inline-flex", alignItems: "center", gap: 5, color: "var(--bull)", fontWeight: 600 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--bull)" }} />
                Green: High Sharpe & Alpha (Recommended)
              </span>
              <span style={{ display: "inline-flex", alignItems: "center", gap: 5, color: "var(--sideways)", fontWeight: 600 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--sideways)" }} />
                Yellow: Moderate Alpha / Elevated Friction
              </span>
              <span style={{ display: "inline-flex", alignItems: "center", gap: 5, color: "var(--bear)", fontWeight: 600 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--bear)" }} />
                Red: High Drawdown / False Signal Risk
              </span>
            </div>

            <div style={{ overflowX: "auto" }}>
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Strategy</th>
                    <th style={{ background: data.current_regime === "Bull" ? "var(--bull-bg)" : undefined, color: data.current_regime === "Bull" ? "var(--bull)" : undefined }}>
                      Bull (Expansion) {data.current_regime === "Bull" && "• ACTIVE TODAY"}
                    </th>
                    <th style={{ background: data.current_regime === "Sideways" ? "var(--sideways-bg)" : undefined, color: data.current_regime === "Sideways" ? "var(--sideways)" : undefined }}>
                      Sideways (Consolidation) {data.current_regime === "Sideways" && "• ACTIVE TODAY"}
                    </th>
                    <th style={{ background: data.current_regime === "Bear" ? "var(--bear-bg)" : undefined, color: data.current_regime === "Bear" ? "var(--bear)" : undefined }}>
                      Bear (Contraction) {data.current_regime === "Bear" && "• ACTIVE TODAY"}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {STRATEGY_EXPLAINERS.map((strat) => {
                    const isExpanded = expandedStrategy === strat.name;
                    return (
                      <React.Fragment key={strat.name}>
                        <tr
                          onClick={() => setExpandedStrategy(isExpanded ? null : strat.name)}
                          style={{ cursor: "pointer", background: isExpanded ? "rgba(var(--glow-rgb), 0.1)" : undefined }}
                          title="Click to view algorithmic mechanism"
                        >
                          <td>
                            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                              <span style={{ fontWeight: 700, color: "var(--text-100)" }}>{strat.name}</span>
                              <Info size={12} style={{ color: "var(--text-200)" }} />
                            </div>
                            <div style={{ fontSize: 11, color: "var(--text-200)" }}>{strat.rule}</div>
                          </td>
                          <td>
                            <div style={{ color: "var(--bull)", fontWeight: 700, fontSize: 12 }}>{strat.bull.rating}</div>
                            <div style={{ fontSize: 11, fontWeight: 600, color: strat.bull.status === "win" ? "var(--bull)" : "var(--text-200)" }}>
                              {strat.bull.tag}
                            </div>
                          </td>
                          <td>
                            <div style={{ color: strat.sideways.status === "win" ? "var(--bull)" : strat.sideways.status === "lose" ? "var(--bear)" : "var(--sideways)", fontWeight: 700, fontSize: 12 }}>
                              {strat.sideways.rating}
                            </div>
                            <div style={{ fontSize: 11, fontWeight: 600, color: strat.sideways.status === "win" ? "var(--bull)" : strat.sideways.status === "lose" ? "var(--bear)" : "var(--sideways)" }}>
                              {strat.sideways.tag}
                            </div>
                          </td>
                          <td>
                            <div style={{ color: strat.bear.status === "win" ? "var(--bull)" : "var(--bear)", fontWeight: 700, fontSize: 12 }}>
                              {strat.bear.rating}
                            </div>
                            <div style={{ fontSize: 11, fontWeight: 600, color: strat.bear.status === "win" ? "var(--bull)" : "var(--bear)" }}>
                              {strat.bear.tag}
                            </div>
                          </td>
                        </tr>

                        {isExpanded && (
                          <tr>
                            <td colSpan={4} style={{ padding: 16, background: "var(--bg-100)", borderRadius: 12 }}>
                              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 12 }}>
                                <div style={{ background: "var(--bg-200)", padding: 12, borderRadius: 10, border: "1px solid rgba(16, 185, 129, 0.2)" }}>
                                  <div style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--bull)", fontWeight: 700, fontSize: 12, marginBottom: 4 }}>
                                    <CheckCircle2 size={13} /> Bull Behavior
                                  </div>
                                  <div style={{ fontSize: 11, color: "var(--text-200)", lineHeight: 1.5 }}>
                                    {strat.bull.desc}
                                  </div>
                                </div>

                                <div style={{ background: "var(--bg-200)", padding: 12, borderRadius: 10, border: "1px solid rgba(245, 158, 11, 0.2)" }}>
                                  <div style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--sideways)", fontWeight: 700, fontSize: 12, marginBottom: 4 }}>
                                    <AlertTriangle size={13} /> Sideways Behavior
                                  </div>
                                  <div style={{ fontSize: 11, color: "var(--text-200)", lineHeight: 1.5 }}>
                                    {strat.sideways.desc}
                                  </div>
                                </div>

                                <div style={{ background: "var(--bg-200)", padding: 12, borderRadius: 10, border: "1px solid rgba(239, 68, 68, 0.2)" }}>
                                  <div style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--bear)", fontWeight: 700, fontSize: 12, marginBottom: 4 }}>
                                    <XCircle size={13} /> Bear Behavior
                                  </div>
                                  <div style={{ fontSize: 11, color: "var(--text-200)", lineHeight: 1.5 }}>
                                    {strat.bear.desc}
                                  </div>
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Actionable Inference Takeaway */}
            <div style={{ marginTop: 18, padding: "14px 18px", borderRadius: 14, background: "linear-gradient(135deg, var(--bg-100), var(--bg-200))", border: "1px solid var(--card-border)" }}>
              <strong style={{ fontSize: 13, color: "var(--text-100)", display: "block", marginBottom: 4 }}>
                🎯 What should you infer from this matrix?
              </strong>
              <p style={{ fontSize: 12, color: "var(--text-200)", lineHeight: 1.6, margin: 0 }}>
                1. <strong>Do not use a single rigid strategy across all market cycles.</strong> A strategy optimized for Bull runs (like Momentum) suffers massive drawdowns during Bear crashes.
                <br />
                2. <strong>Regime-Switching provides alpha & downside protection:</strong> When Gaussian HMM flags a Bull regime, allocate to Momentum/Buy & Hold; when it detects Sideways, switch to RSI/Bollinger Bands; when Bear is detected, switch to Dual Momentum (100% Cash Defense) to preserve capital.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
