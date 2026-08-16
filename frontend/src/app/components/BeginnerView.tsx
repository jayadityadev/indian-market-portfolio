"use client";

import { useMemo, useState } from "react";
import { AnalyzeResponse } from "@/lib/api";
import { CandlestickChart } from "./CandlestickChart";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from "recharts";
import { TrendingUp, Shield, Lightbulb, Activity, ChevronDown, ChevronUp, Sparkles, CheckCircle2 } from "lucide-react";

const RC: Record<string, string> = { Bull: "var(--bull)", Bear: "var(--bear)", Sideways: "var(--sideways)" };
const RB: Record<string, string> = { Bull: "var(--bull-bg)", Bear: "var(--bear-bg)", Sideways: "var(--sideways-bg)" };
const rLabel = (r: string) => r === "Bull" ? "Market was growing 📈" : r === "Bear" ? "Market was falling 📉" : "Market was moving sideways ↔️";
const rLabelShort = (r: string) => r === "Bull" ? "Bullish" : r === "Bear" ? "Bearish" : "Sideways";
const COMPARISON_STRATEGIES = ["MA Crossover", "RSI", "Momentum", "Bollinger Bands"];

export function BeginnerView({ data }: { data: AnalyzeResponse }) {
  const [showTechnical, setShowTechnical] = useState(false);
  const m = data.overall_metrics[data.recommended_strategy] || Object.values(data.overall_metrics)[0];
  const curve = data.equity_curves[data.recommended_strategy] || [];
  const bh = data.equity_curves["Buy & Hold"] || [];
  
  const baseVal = curve[0]?.value || 100000;
  const bhBase = bh[0]?.value || 100000;
  
  // Calculate % return from actual initial capital base
  const stratReturn = curve.length > 0 ? ((curve[curve.length - 1].value / baseVal) - 1) * 100 : 0;
  const bhReturn = bh.length > 0 ? ((bh[bh.length - 1].value / bhBase) - 1) * 100 : 0;
  const outperformance = stratReturn - bhReturn;

  // Merge all equity series by date into a single unified time-series
  const chartData = useMemo(() => {
    if (!data.equity_curves) return [];
    const dateMap = new Map<string, Record<string, string | number>>();
    
    Object.entries(data.equity_curves).forEach(([strat, points]) => {
      points.forEach((pt) => {
        if (!dateMap.has(pt.date)) {
          dateMap.set(pt.date, { date: pt.date });
        }
        const entry = dateMap.get(pt.date)!;
        entry[strat] = pt.value;
      });
    });

    return Array.from(dateMap.values()).sort((a, b) => (a.date as string).localeCompare(b.date as string));
  }, [data.equity_curves]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      {data.recommendation_status !== "validated_ml" && (
        <div className="card" style={{ border: "1px solid var(--sideways)", background: "var(--sideways-bg)" }}>
          <div className="card-title" style={{ fontSize: 13 }}>💡 Prudent Fallback Active</div>
          <div style={{ fontSize: 13, color: "var(--text-200)", lineHeight: 1.6 }}>
            Our algorithm recommends <strong>{data.recommended_strategy}</strong> based on historical regime backtests to ensure capital safety.
          </div>
        </div>
      )}

      {/* Level 1: Action Hero Card */}
      <div className="hero-card" style={{ padding: 26 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, fontWeight: 700, color: "var(--accent-100)", marginBottom: 8 }}>
          <Sparkles size={14} /> Clear Action Recommendation
        </div>
        <div style={{ fontSize: 13, color: "var(--text-200)", marginBottom: 4 }}>Best strategy for this market condition:</div>
        <div style={{ fontSize: 32, fontWeight: 800, color: "var(--text-100)", marginBottom: 8 }}>{data.recommended_strategy}</div>
        
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", fontSize: 13, color: "var(--text-200)", marginBottom: 12 }}>
          <span>Gained <strong style={{ color: stratReturn >= 0 ? "var(--bull)" : "var(--bear)" }}>{stratReturn >= 0 ? "+" : ""}{stratReturn.toFixed(1)}%</strong></span>
          <span>•</span>
          <span>Regular Buy &amp; Hold: <strong>{bhReturn >= 0 ? "+" : ""}{bhReturn.toFixed(1)}%</strong></span>
          <span>•</span>
          <span style={{ color: outperformance >= 0 ? "var(--bull)" : "var(--bear)", fontWeight: 700 }}>
            ({outperformance >= 0 ? "+" : ""}{outperformance.toFixed(1)}% {outperformance >= 0 ? "extra profit" : "difference"})
          </span>
        </div>

        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", borderTop: "1px solid var(--card-border)", paddingTop: 12, fontSize: 12 }}>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 5, color: "var(--text-100)" }}>
            <CheckCircle2 size={13} style={{ color: "var(--bull)" }} /> Position Sizing: <strong>{data.recommended_exposure}</strong>
          </span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 5, color: "var(--text-100)" }}>
            <CheckCircle2 size={13} style={{ color: "var(--bull)" }} /> Market Phase: <strong>{rLabel(data.current_regime)}</strong>
          </span>
        </div>
      </div>

      {/* Level 2: 3 Simplified Metric Cards */}
      <div className="metric-grid">
        <div className="metric-card">
          <div className="metric-card__label">Annual Return Rate</div>
          <div className="metric-card__value" style={{ color: m.CAGR >= 0 ? "var(--bull)" : "var(--bear)" }}>
            {(m.CAGR * 100).toFixed(1)}%
          </div>
          <div className="metric-card__desc">{m.CAGR >= 0.1 ? "Strong" : m.CAGR >= 0.05 ? "Moderate" : "Low"} yearly growth</div>
        </div>

        <div className="metric-card">
          <div className="metric-card__label">Risk-Adjusted Rating</div>
          <div className="metric-card__value" style={{ color: m.Sharpe >= 1 ? "var(--bull)" : m.Sharpe >= 0.5 ? "var(--sideways)" : "var(--bear)" }}>
            {m.Sharpe >= 1.5 ? "Excellent" : m.Sharpe >= 1 ? "Good" : m.Sharpe >= 0.5 ? "Fair" : "Cautious"}
          </div>
          <div className="metric-card__desc">Score: {m.Sharpe.toFixed(2)} vs volatility</div>
        </div>

        <div className="metric-card">
          <div className="metric-card__label">Worst Market Drop</div>
          <div className="metric-card__value" style={{ color: "var(--bear)" }}>
            {(m.MaxDrawdown * 100).toFixed(0)}%
          </div>
          <div className="metric-card__desc">Max peak-to-trough pullback</div>
        </div>
      </div>

      {/* Level 2: Simplified Performance Comparison (Recommended vs Baseline) */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
          <div className="card-title" style={{ margin: 0 }}>
            <TrendingUp size={16} /> Portfolio Growth (Starting with ₹1,00,000)
          </div>
          <span style={{ fontSize: 11, color: "var(--bull)", fontWeight: 700 }}>
            ₹{Number(curve[curve.length - 1]?.value || 100000).toLocaleString("en-IN", { maximumFractionDigits: 0 })} Final Value
          </span>
        </div>
        <div style={{ fontSize: 12, color: "var(--text-200)", marginBottom: 12 }}>
          Clean comparison: Your recommended strategy (Green) vs standard Buy &amp; Hold (Dashed)
        </div>
        <div style={{ height: 260 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid stroke="var(--bg-300)" strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: "var(--text-200)" }} tickFormatter={(d: string) => d.slice(0, 7)} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 10, fill: "var(--text-200)" }} tickFormatter={(v: number) => `${(((v - baseVal) / baseVal) * 100).toFixed(0)}%`} />
              <Tooltip contentStyle={{ background: "var(--bg-200)", border: "1px solid var(--bg-300)", borderRadius: 8, fontSize: 12, color: "var(--text-100)" }} formatter={(v: unknown) => { const pct = (((Number(v) - baseVal) / baseVal) * 100).toFixed(1); return [`₹${Number(v).toLocaleString('en-IN', { maximumFractionDigits: 0 })} (${pct.startsWith('-') ? '' : '+'}${pct}%)`, ""]; }} />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
              <Line dataKey={data.recommended_strategy} name={`${data.recommended_strategy} (Recommended)`} stroke="var(--bull)" strokeWidth={2.5} dot={false} connectNulls />
              {data.recommended_strategy !== "Buy & Hold" && (
                <Line dataKey="Buy & Hold" name="Buy & Hold (Baseline)" stroke="var(--text-200)" strokeWidth={1.5} strokeDasharray="8 4" dot={false} connectNulls />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Why this Strategy & Safe Exposure */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 14 }}>
        <div className="card">
          <div className="card-title"><Lightbulb size={16} /> Why this strategy?</div>
          <div style={{ display: "inline-block", padding: "3px 12px", borderRadius: 20, fontSize: 11, fontWeight: 600, background: RB[data.current_regime], color: RC[data.current_regime], marginBottom: 12 }}>
            {rLabel(data.current_regime)}
          </div>
          <div style={{ fontSize: 13, color: "var(--text-200)", lineHeight: 1.7 }}>
            {data.recommended_strategy} performs best in {data.current_regime.toLowerCase()} markets. Based on historical backtests during similar conditions, it delivered superior risk-adjusted stability.
          </div>
        </div>

        <div className="card">
          <div className="card-title"><Shield size={16} /> Risk &amp; Exposure Guidance</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: data.recommended_exposure.includes("100%") ? "var(--bull)" : data.recommended_exposure.includes("75%") ? "var(--sideways)" : "var(--bear)", marginBottom: 6 }}>
            {data.recommended_exposure.includes("100%") ? "Low Risk Stance ✅" : data.recommended_exposure.includes("75%") ? "Moderate Stance ⚠️" : "High Defensive Stance 🔴"}
          </div>
          <div style={{ fontSize: 12, color: "var(--text-200)", lineHeight: 1.6 }}>
            {data.recommended_exposure.includes("100%") ? "Normal market conditions. Full 100% equity exposure is safe." : data.recommended_exposure.includes("75%") ? "Elevated volatility. Keep 25% cash buffer and 75% equity." : "High drawdown risk. Move 50–75% into cash reserves."}
          </div>
        </div>
      </div>

      {/* Level 3: Progressive Disclosure Toggle for Advanced Technical Breakdown */}
      <div style={{ marginTop: 8 }}>
        <button
          onClick={() => setShowTechnical(!showTechnical)}
          className="export-btn"
          style={{
            width: "100%",
            justifyContent: "center",
            padding: "12px 16px",
            fontSize: 13,
            fontWeight: 600,
            background: showTechnical ? "var(--bg-100)" : "var(--bg-200)",
          }}
        >
          <span>{showTechnical ? "Hide Detailed Technical Breakdown" : "🔬 Show Detailed Technical Breakdown (Candlesticks, Timeline & Multi-Strategy)"}</span>
          {showTechnical ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
        </button>

        {showTechnical && (
          <div style={{ display: "flex", flexDirection: "column", gap: 18, marginTop: 16 }}>
            {/* Candlestick Chart */}
            {data.ohlc_data.length > 0 && (
              <div className="card">
                <div className="card-title"><Activity size={16} /> NIFTY 50 Daily Price Action (Candlesticks)</div>
                <CandlestickChart data={data.ohlc_data} regimeTimeline={data.regime_timeline} height={340} />
              </div>
            )}

            {/* All 6 Strategies Performance Comparison */}
            <div className="card">
              <div className="card-title"><TrendingUp size={16} /> Multi-Strategy Overlay (All 6 Algorithms)</div>
              <div style={{ height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid stroke="var(--bg-300)" strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fontSize: 10, fill: "var(--text-200)" }} tickFormatter={(d: string) => d.slice(0, 7)} interval="preserveStartEnd" />
                    <YAxis tick={{ fontSize: 10, fill: "var(--text-200)" }} tickFormatter={(v: number) => `${(((v - baseVal) / baseVal) * 100).toFixed(0)}%`} />
                    <Tooltip contentStyle={{ background: "var(--bg-200)", border: "1px solid var(--bg-300)", borderRadius: 8, fontSize: 12, color: "var(--text-100)" }} formatter={(v: unknown) => { const pct = (((Number(v) - baseVal) / baseVal) * 100).toFixed(1); return [`₹${Number(v).toLocaleString('en-IN', { maximumFractionDigits: 0 })} (${pct.startsWith('-') ? '' : '+'}${pct}%)`, ""]; }} />
                    <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                    <Line dataKey={data.recommended_strategy} name={`${data.recommended_strategy} (Recommended)`} stroke="var(--bull)" strokeWidth={2.5} dot={false} connectNulls />
                    <Line dataKey="Buy & Hold" name="Buy & Hold (Baseline)" stroke="var(--text-200)" strokeWidth={1.5} strokeDasharray="8 4" dot={false} connectNulls />
                    {COMPARISON_STRATEGIES.map((s, idx) => {
                      if (s === data.recommended_strategy) return null;
                      return <Line key={s} dataKey={s} name={s} stroke={`var(--accent-${(idx % 3) + 1}00)`} strokeWidth={1} dot={false} connectNulls opacity={0.3} />;
                    })}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Regime Timeline */}
            <div className="card">
              <div className="card-title"><Activity size={16} /> Market Regime Timeline (HMM History)</div>
              <div style={{ fontSize: 12, color: "var(--text-200)", marginBottom: 12 }}>Detailed sequence of Bull, Bear, and Sideways periods.</div>
              <div className="regime-bar" style={{ height: 48, display: "flex", borderRadius: 8, overflow: "hidden", border: "1px solid var(--card-border)" }}>
                {data.regime_timeline.map((seg, i) => {
                  const total = data.regime_timeline.reduce((a, x) => a + Math.max(x.days, 1), 0);
                  const pct = (Math.max(seg.days, 1) / total) * 100;
                  return <div key={i} title={`${seg.regime}: ${seg.start} → ${seg.end} (${seg.days} days)`} style={{ flex: `${pct} 0 0`, background: RB[seg.regime], borderRight: "1px solid var(--card-border)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 700, color: RC[seg.regime], textAlign: "center", padding: "0 4px" }}>{pct > 8 ? rLabelShort(seg.regime) : ""}</div>;
                })}
              </div>
              <div className="regime-legend">{["Bull","Bear","Sideways"].map(r => <div key={r} className="regime-legend-item"><div className="regime-dot" style={{ background: RC[r] }} />{rLabel(r)}</div>)}</div>
            </div>

            {/* Monte Carlo Risk Forecast */}
            {data.risk_forecast && (
              <div className="card">
                <div className="card-title"><Shield size={16} /> Monte Carlo 63-Day Forward Risk Forecast</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginTop: 10 }}>
                  <div style={{ textAlign: "center", padding: 12, background: "var(--bg-100)", borderRadius: 10 }}>
                    <div style={{ fontSize: 11, color: "var(--text-200)" }}>Best Case (90th)</div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: "var(--bull)" }}>-{(Math.abs(data.risk_forecast.best_case_90)*100).toFixed(0)}%</div>
                  </div>
                  <div style={{ textAlign: "center", padding: 12, background: "var(--bg-100)", borderRadius: 10 }}>
                    <div style={{ fontSize: 11, color: "var(--text-200)" }}>Median (50th)</div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: "var(--text-100)" }}>-{(Math.abs(data.risk_forecast.median_50)*100).toFixed(0)}%</div>
                  </div>
                  <div style={{ textAlign: "center", padding: 12, background: "var(--bg-100)", borderRadius: 10 }}>
                    <div style={{ fontSize: 11, color: "var(--text-200)" }}>Worst Case (10th)</div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: "var(--bear)" }}>-{(Math.abs(data.risk_forecast.worst_case_10)*100).toFixed(0)}%</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
