"use client";
import { Book, CheckCircle2, Info } from "lucide-react";

const STRATEGIES = [
  {
    name: "Buy & Hold",
    description: "The classic 'coffee-can' approach. Stay invested regardless of market conditions.",
    bestFor: "Long-term Bull markets",
    pros: "Zero transaction costs, tax efficient, simple.",
    cons: "Maximum exposure to crashes (like 2008 or 2020).",
  },
  {
    name: "MA Crossover",
    description: "Uses a fast (50-day) and slow (200-day) moving average. Buy when fast crosses above slow.",
    bestFor: "Trending markets",
    pros: "Avoids major bear markets by exiting early.",
    cons: "Gets 'whipsawed' (many small losses) in sideways markets.",
  },
  {
    name: "RSI Mean Reversion",
    description: "Relative Strength Index. Buy when 'oversold' (<30) and sell when 'overbought' (>70).",
    bestFor: "Sideways / Mean-reverting markets",
    pros: "Capitalizes on short-term price extremes.",
    cons: "Extremely dangerous in strong trending markets.",
  },
  {
    name: "Momentum",
    description: "Riding the wave. Buys stocks that have performed best over the last 12 months.",
    bestFor: "Strong Bull markets",
    pros: "Can capture massive returns during sustained rallies.",
    cons: "High volatility and sharp reversals at trend changes.",
  },
  {
    name: "Bollinger Bands",
    description: "Uses standard deviation bands around a mean. Sells at upper band, buys at lower band.",
    bestFor: "Volatile / Sideways markets",
    pros: "Provides clear entry/exit points based on volatility.",
    cons: "Bands can 'walk' during strong trends, causing premature exits.",
  },
  {
    name: "Dual Momentum",
    description: "Combines Absolute (vs cash) and Relative (vs others) momentum. Only buys if trend is positive.",
    bestFor: "Regime changes",
    pros: "Strongest risk-adjusted performance historically.",
    cons: "Complex calculation, higher turnover.",
  }
];

export function StrategyLibrary() {
  return (
    <div className="card">
      <div className="card-title"><Book size={16} /> Strategy Library</div>
      <div style={{ fontSize: 12, color: "var(--text-200)", marginBottom: 16 }}>Learn how our algorithm thinks about these strategies</div>
      
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 12 }}>
        {STRATEGIES.map(s => (
          <div key={s.name} className="strategy-card" style={{ padding: 14, borderRadius: 10, background: "var(--bg-200)", border: "1px solid var(--bg-300)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: 8 }}>
              <div style={{ fontWeight: 700, color: "var(--text-100)" }}>{s.name}</div>
              <div style={{ fontSize: 10, background: "var(--accent-200)", color: "#fff", padding: "2px 8px", borderRadius: 12 }}>{s.bestFor}</div>
            </div>
            <div style={{ fontSize: 12, color: "var(--text-200)", marginBottom: 12, lineHeight: 1.5 }}>{s.description}</div>
            
            <div style={{ display: "flex", gap: 10 }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: "var(--bull)", display: "flex", alignItems: "center", gap: 4, marginBottom: 4 }}><CheckCircle2 size={10} /> Pros</div>
                <div style={{ fontSize: 11, color: "var(--text-200)" }}>{s.pros}</div>
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: "var(--bear)", display: "flex", alignItems: "center", gap: 4, marginBottom: 4 }}><Info size={10} /> Cons</div>
                <div style={{ fontSize: 11, color: "var(--text-200)" }}>{s.cons}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
