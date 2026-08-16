"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { 
  BookOpen, 
  CheckCircle2, 
  XCircle, 
  Sliders
} from "lucide-react";

interface StrategyItem {
  id: string;
  name: string;
  category: "Trend Following" | "Mean Reversion" | "Multi-Factor" | "Passive Baseline";
  formula: string;
  description: string;
  bestRegime: "Bull" | "Sideways" | "Bear";
  worstRegime: "Bull" | "Sideways" | "Bear";
  annualizedAlpha: string;
  holdingPeriod: string;
  coreRules: string[];
  failureModes: string[];
  visualType: "momentum" | "ma" | "rsi" | "bb" | "dual" | "bh";
}

const STRATEGIES: StrategyItem[] = [
  {
    id: "momentum",
    name: "Momentum Alpha",
    category: "Trend Following",
    formula: "R_i(t) = \\frac{P_i(t) - P_i(t - 252)}{P_i(t - 252)}",
    description: "Capitalizes on continuation of existing market trends by buying top-quartile performers over a 12-month lookback period.",
    bestRegime: "Bull",
    worstRegime: "Bear",
    annualizedAlpha: "+4.2% vs Baseline",
    holdingPeriod: "1 to 3 Months",
    coreRules: [
      "Rank universe by 12-month cumulative return minus most recent month (12-1 momentum).",
      "Allocate to top decile with monthly rebalancing.",
      "Cut exposure when benchmark drops below 200-day moving average.",
    ],
    failureModes: [
      "Sharp V-shaped market reversals cause momentum crashes.",
      "High turnover in choppy consolidation regimes.",
    ],
    visualType: "momentum",
  },
  {
    id: "ma-crossover",
    name: "MA Crossover (50/200)",
    category: "Trend Following",
    formula: "\\text{Signal}(t) = \\text{sign}(\\text{SMA}_{50}(t) - \\text{SMA}_{200}(t))",
    description: "Classic systematic trend filter using Golden Cross (bullish) and Death Cross (bearish) moving average crossovers.",
    bestRegime: "Bull",
    worstRegime: "Sideways",
    annualizedAlpha: "+2.8% vs Baseline",
    holdingPeriod: "3 to 9 Months",
    coreRules: [
      "Go 100% Long when 50-day SMA crosses above 200-day SMA.",
      "Exit 100% to Cash when 50-day SMA crosses below 200-day SMA.",
      "Lagged execution (1-day delay) prevents lookahead bias.",
    ],
    failureModes: [
      "Whipsaw losses during range-bound sideways markets.",
      "Lagging signal enters late after rallies begin.",
    ],
    visualType: "ma",
  },
  {
    id: "rsi",
    name: "RSI Mean Reversion",
    category: "Mean Reversion",
    formula: "\\text{RSI}_{14} = 100 - \\frac{100}{1 + \\frac{\\text{EMA}(\\text{Gains}, 14)}{\\text{EMA}(\\text{Losses}, 14)}}",
    description: "Exploits short-term overextensions by buying deeply oversold conditions and taking profit at overbought extremes.",
    bestRegime: "Sideways",
    worstRegime: "Bear",
    annualizedAlpha: "+5.1% in Range Markets",
    holdingPeriod: "3 to 14 Days",
    coreRules: [
      "BUY signal triggered when 14-period RSI drops below 30 (Oversold).",
      "SELL signal triggered when RSI crosses above 70 (Overbought).",
      "Hard stop-loss at -4% from entry price.",
    ],
    failureModes: [
      "Catching falling knives during sustained market crashes where RSI stays < 30.",
      "Exits strong bull rallies prematurely at RSI 70.",
    ],
    visualType: "rsi",
  },
  {
    id: "bollinger",
    name: "Bollinger Bands",
    category: "Mean Reversion",
    formula: "\\text{Bands} = \\text{SMA}_{20}(P) \\pm 2 \\cdot \\sigma_{20}(P)",
    description: "Volatility envelope model that measures standard deviation swings away from a 20-period moving average.",
    bestRegime: "Sideways",
    worstRegime: "Bear",
    annualizedAlpha: "+3.9% in Range Markets",
    holdingPeriod: "5 to 20 Days",
    coreRules: [
      "Enter Long when price touches or pierces Lower 2-sigma Band.",
      "Take profit when price reaches Upper 2-sigma Band or 20-day Mean.",
      "Dynamic band width adapts automatically to market volatility.",
    ],
    failureModes: [
      "Strong trends 'walk the bands' continuously, generating premature exits or stop-outs.",
      "Volatility squeeze breakouts can cause sudden losses.",
    ],
    visualType: "bb",
  },
  {
    id: "dual-momentum",
    name: "Dual Momentum",
    category: "Multi-Factor",
    formula: "\\text{Invested} = (R_{\\text{Equity}} > R_{\\text{Peers}}) \\land (R_{\\text{Equity}} > R_{\\text{Cash}})",
    description: "Gary Antonacci's framework combining Relative Momentum (asset vs peers) and Absolute Momentum (asset vs cash floor).",
    bestRegime: "Bear",
    worstRegime: "Sideways",
    annualizedAlpha: "+6.4% in Crash Regimes",
    holdingPeriod: "1 to 6 Months",
    coreRules: [
      "Relative Check: Identify sector/asset with strongest 12-month return.",
      "Absolute Check: Ensure asset return is strictly higher than 91-day T-Bill rate.",
      "If absolute check fails, move 100% of portfolio to risk-free cash/treasuries.",
    ],
    failureModes: [
      "Lagged shift into cash during sudden, sharp market drops.",
      "Underperforms in rapid whipsaw recoveries.",
    ],
    visualType: "dual",
  },
  {
    id: "buy-and-hold",
    name: "Buy & Hold (Baseline)",
    category: "Passive Baseline",
    formula: "V(t) = V_0 \\prod_{s=1}^t (1 + r_s)",
    description: "Passive, 100% invested index allocation capturing full long-term Indian macroeconomic growth without market timing.",
    bestRegime: "Bull",
    worstRegime: "Bear",
    annualizedAlpha: "0.0% (Benchmark Baseline)",
    holdingPeriod: "Multi-Year / Infinite",
    coreRules: [
      "Maintain 100% static equity allocation in NIFTY 50.",
      "Zero turnover, zero transaction costs, zero timing friction.",
      "Reinvest dividends periodically.",
    ],
    failureModes: [
      "Absorbs 100% of major macro market crashes (-50% in 2008, -38% in 2020).",
      "Prolonged multi-year drawdown recovery periods during sideways regimes.",
    ],
    visualType: "bh",
  },
];

function StrategySimulation({ type }: { type: StrategyItem["visualType"] }) {
  if (type === "momentum") {
    return (
      <div style={{ position: "relative", height: 160, background: "var(--bg-100)", borderRadius: 14, overflow: "hidden", padding: 14, display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text-200)" }}>
          <span style={{ color: "var(--bull)", fontWeight: 700 }}>● Active Long (12M Momentum)</span>
          <span>Target: Top Decile Alpha</span>
        </div>
        <svg viewBox="0 0 300 80" style={{ width: "100%", height: 70 }}>
          <motion.path
            d="M 10 70 Q 70 65, 120 40 T 200 25 T 290 10"
            fill="none"
            stroke="var(--bull)"
            strokeWidth="2.5"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 2, repeat: Infinity, repeatDelay: 1 }}
          />
          <line x1="10" y1="55" x2="290" y2="35" stroke="var(--text-200)" strokeWidth="1" strokeDasharray="4 4" opacity="0.4" />
          <motion.circle cx="200" cy="25" r="4" fill="var(--bull)" animate={{ scale: [1, 1.4, 1] }} transition={{ duration: 1.5, repeat: Infinity }} />
        </svg>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--text-200)" }}>
          <span>Entry: Acceleration Point</span>
          <span style={{ color: "var(--bull)" }}>Trailing Alpha: +18.4%</span>
        </div>
      </div>
    );
  }

  if (type === "ma") {
    return (
      <div style={{ position: "relative", height: 160, background: "var(--bg-100)", borderRadius: 14, overflow: "hidden", padding: 14, display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
          <span style={{ color: "var(--accent-100)", fontWeight: 700 }}>Fast MA (50)</span>
          <span style={{ color: "var(--text-200)" }}>Slow MA (200)</span>
        </div>
        <svg viewBox="0 0 300 80" style={{ width: "100%", height: 70 }}>
          {/* Slow MA */}
          <path d="M 10 50 Q 150 45, 290 30" fill="none" stroke="var(--text-200)" strokeWidth="2" strokeDasharray="6 4" />
          {/* Fast MA */}
          <motion.path
            d="M 10 65 Q 100 60, 150 42 T 290 15"
            fill="none"
            stroke="var(--accent-100)"
            strokeWidth="2.5"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 2.2, repeat: Infinity, repeatDelay: 1 }}
          />
          {/* Golden Cross Marker */}
          <circle cx="150" cy="42" r="5" fill="var(--bull)" />
        </svg>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--text-200)" }}>
          <span style={{ color: "var(--bull)", fontWeight: 700 }}>⚡ Golden Cross (BUY)</span>
          <span>Death Cross Protection</span>
        </div>
      </div>
    );
  }

  if (type === "rsi") {
    return (
      <div style={{ position: "relative", height: 160, background: "var(--bg-100)", borderRadius: 14, overflow: "hidden", padding: 14, display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
          <span style={{ color: "var(--bear)" }}>Overbought (70)</span>
          <span style={{ color: "var(--bull)" }}>Oversold (30)</span>
        </div>
        <svg viewBox="0 0 300 80" style={{ width: "100%", height: 70 }}>
          <line x1="0" y1="20" x2="300" y2="20" stroke="var(--bear)" strokeWidth="1" strokeDasharray="3 3" opacity="0.4" />
          <line x1="0" y1="60" x2="300" y2="60" stroke="var(--bull)" strokeWidth="1" strokeDasharray="3 3" opacity="0.4" />
          <motion.path
            d="M 0 50 Q 50 15, 100 50 T 200 68 T 300 30"
            fill="none"
            stroke="var(--sideways)"
            strokeWidth="2.5"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 2.5, repeat: Infinity, repeatDelay: 0.8 }}
          />
          <circle cx="200" cy="68" r="4" fill="var(--bull)" />
          <circle cx="50" cy="15" r="4" fill="var(--bear)" />
        </svg>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--text-200)" }}>
          <span style={{ color: "var(--bull)", fontWeight: 700 }}>BUY at RSI &lt; 30</span>
          <span style={{ color: "var(--bear)", fontWeight: 700 }}>SELL at RSI &gt; 70</span>
        </div>
      </div>
    );
  }

  if (type === "bb") {
    return (
      <div style={{ position: "relative", height: 160, background: "var(--bg-100)", borderRadius: 14, overflow: "hidden", padding: 14, display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text-200)" }}>
          <span>Upper +2σ</span>
          <span>Lower -2σ</span>
        </div>
        <svg viewBox="0 0 300 80" style={{ width: "100%", height: 70 }}>
          {/* Upper band */}
          <path d="M 0 15 Q 150 25, 300 15" fill="none" stroke="var(--accent-100)" strokeWidth="1" strokeDasharray="2 2" opacity="0.6" />
          {/* Lower band */}
          <path d="M 0 65 Q 150 55, 300 65" fill="none" stroke="var(--accent-100)" strokeWidth="1" strokeDasharray="2 2" opacity="0.6" />
          {/* Price oscillation */}
          <motion.path
            d="M 0 40 Q 60 18, 120 62 T 220 20 T 300 45"
            fill="none"
            stroke="var(--bull)"
            strokeWidth="2"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 3, repeat: Infinity }}
          />
        </svg>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--text-200)" }}>
          <span style={{ color: "var(--bull)" }}>Mean Reversion Entry</span>
          <span>Adaptive Volatility</span>
        </div>
      </div>
    );
  }

  if (type === "dual") {
    return (
      <div style={{ position: "relative", height: 160, background: "var(--bg-100)", borderRadius: 14, overflow: "hidden", padding: 14, display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
          <span style={{ color: "var(--bull)", fontWeight: 700 }}>Relative Check: PASS</span>
          <span style={{ color: "var(--sideways)", fontWeight: 700 }}>Absolute Check: ACTIVE</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 16, height: 60 }}>
          <div style={{ padding: "8px 12px", borderRadius: 10, background: "var(--bull-bg)", color: "var(--bull)", fontSize: 12, fontWeight: 700 }}>
            Top Equity Decile
          </div>
          <span style={{ fontSize: 18, color: "var(--text-200)" }}>→</span>
          <div style={{ padding: "8px 12px", borderRadius: 10, background: "var(--bear-bg)", color: "var(--bear)", fontSize: 12, fontWeight: 700 }}>
            100% Cash Defense
          </div>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--text-200)" }}>
          <span>Crash Filter: Enabled</span>
          <span style={{ color: "var(--bull)" }}>Max Drawdown: -8.2%</span>
        </div>
      </div>
    );
  }

  // Buy & Hold
  return (
    <div style={{ position: "relative", height: 160, background: "var(--bg-100)", borderRadius: 14, overflow: "hidden", padding: 14, display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text-200)" }}>
        <span>Index Compounding</span>
        <span>0% Trading Friction</span>
      </div>
      <svg viewBox="0 0 300 80" style={{ width: "100%", height: 70 }}>
        <motion.path
          d="M 10 70 Q 100 65, 160 45 T 290 10"
          fill="none"
          stroke="var(--accent-100)"
          strokeWidth="2.5"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 2.5, repeat: Infinity, repeatDelay: 1 }}
        />
      </svg>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--text-200)" }}>
        <span>Long-Term Capital Gain</span>
        <span style={{ color: "var(--bear)" }}>Drawdown Exposure: 100%</span>
      </div>
    </div>
  );
}

export default function StrategiesPage() {
  const [selectedCategory, setSelectedCategory] = useState<string>("All");

  const categories = ["All", "Trend Following", "Mean Reversion", "Multi-Factor", "Passive Baseline"];
  const filtered = selectedCategory === "All" ? STRATEGIES : STRATEGIES.filter((s) => s.category === selectedCategory);

  return (
    <div className="subpage-container">
      {/* Header */}
      <div className="subpage-header">
        <div className="subpage-badge">
          <BookOpen size={13} />
          <span>Quantitative Algorithm Library • 6 Systemic Strategies</span>
        </div>
        <h1 className="subpage-title">Strategy Library & Quantitative Mechanics</h1>
        <p className="subpage-desc">
          Mathematical formulations, execution triggers, and regime-conditional behavior for all six trading
          strategies implemented in the Indian Market Portfolio backtesting engine.
        </p>
      </div>

      {/* Category Filter */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 20 }}>
        <span style={{ fontSize: 12, color: "var(--text-200)", display: "inline-flex", alignItems: "center", gap: 4, marginRight: 6 }}>
          <Sliders size={13} /> Filter Strategy Type:
        </span>
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className="navbar-link"
            style={{
              background: selectedCategory === cat ? "var(--accent-100)" : "var(--bg-200)",
              color: selectedCategory === cat ? "#ffffff" : "var(--text-200)",
              border: "1px solid var(--card-border)",
              cursor: "pointer",
              padding: "6px 14px",
              fontSize: 12,
              fontWeight: 600,
            }}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Strategy Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: 20 }}>
        {filtered.map((strat) => (
          <div key={strat.id} className="card" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
            <div>
              {/* Header */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10, gap: 8 }}>
                <div>
                  <span
                    style={{
                      fontSize: 10,
                      fontWeight: 700,
                      color: "var(--accent-100)",
                      textTransform: "uppercase",
                      letterSpacing: "0.04em",
                      display: "block",
                      marginBottom: 2,
                    }}
                  >
                    {strat.category}
                  </span>
                  <h3 style={{ fontSize: 18, fontWeight: 700, color: "var(--text-100)", margin: 0 }}>
                    {strat.name}
                  </h3>
                </div>

                <div style={{ display: "flex", gap: 6 }}>
                  <span
                    style={{
                      fontSize: 10,
                      fontWeight: 700,
                      padding: "3px 8px",
                      borderRadius: 999,
                      background: strat.bestRegime === "Bull" ? "var(--bull-bg)" : strat.bestRegime === "Bear" ? "var(--bear-bg)" : "var(--sideways-bg)",
                      color: strat.bestRegime === "Bull" ? "var(--bull)" : strat.bestRegime === "Bear" ? "var(--bear)" : "var(--sideways)",
                      border: "1px solid var(--card-border)",
                    }}
                  >
                    Best: {strat.bestRegime}
                  </span>
                </div>
              </div>

              <p style={{ fontSize: 12, color: "var(--text-200)", lineHeight: 1.5, marginBottom: 14 }}>
                {strat.description}
              </p>

              {/* Live Interactive Visualization */}
              <div style={{ marginBottom: 16 }}>
                <StrategySimulation type={strat.visualType} />
              </div>

              {/* Mathematical Formula */}
              <div
                style={{
                  background: "var(--bg-100)",
                  padding: "10px 14px",
                  borderRadius: 12,
                  border: "1px solid var(--card-border)",
                  marginBottom: 16,
                  fontFamily: "monospace",
                  fontSize: 11,
                  color: "var(--text-100)",
                  overflowX: "auto",
                }}
              >
                <div style={{ fontSize: 9, textTransform: "uppercase", color: "var(--text-200)", marginBottom: 2 }}>Mathematical Formulation</div>
                <code>{strat.formula}</code>
              </div>

              {/* Execution Rules */}
              <div style={{ marginBottom: 14 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text-100)", marginBottom: 6, display: "flex", alignItems: "center", gap: 4 }}>
                  <CheckCircle2 size={13} style={{ color: "var(--bull)" }} /> Execution Triggers
                </div>
                <ul style={{ margin: 0, paddingLeft: 18, fontSize: 11, color: "var(--text-200)", lineHeight: 1.5 }}>
                  {strat.coreRules.map((rule, i) => (
                    <li key={i}>{rule}</li>
                  ))}
                </ul>
              </div>

              {/* Known Failure Modes */}
              <div>
                <div style={{ fontSize: 11, fontWeight: 700, color: "var(--bear)", marginBottom: 6, display: "flex", alignItems: "center", gap: 4 }}>
                  <XCircle size={13} /> Failure Vulnerabilities
                </div>
                <ul style={{ margin: 0, paddingLeft: 18, fontSize: 11, color: "var(--text-200)", lineHeight: 1.5 }}>
                  {strat.failureModes.map((fm, i) => (
                    <li key={i}>{fm}</li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Bottom Metrics Pill */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderTop: "1px solid var(--bg-300)", paddingTop: 12, marginTop: 16, fontSize: 11 }}>
              <span style={{ color: "var(--text-200)" }}>Holding Horizon: <strong>{strat.holdingPeriod}</strong></span>
              <span style={{ color: "var(--bull)", fontWeight: 700 }}>{strat.annualizedAlpha}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
