"use client";
import { AnalyzeResponse } from "@/lib/api";
import { CandlestickChart } from "./CandlestickChart";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from "recharts";
import { TrendingUp, Shield, Lightbulb, Activity } from "lucide-react";

const RC: Record<string, string> = { Bull: "var(--bull)", Bear: "var(--bear)", Sideways: "var(--sideways)" };
const RB: Record<string, string> = { Bull: "var(--bull-bg)", Bear: "var(--bear-bg)", Sideways: "var(--sideways-bg)" };
const fmt = (n: number) => n >= 10000000 ? `₹${(n/10000000).toFixed(2)} Cr` : n >= 100000 ? `₹${(n/100000).toFixed(2)} L` : `₹${n.toLocaleString("en-IN")}`;
const rLabel = (r: string) => r === "Bull" ? "Market was growing 📈" : r === "Bear" ? "Market was falling 📉" : "Market was moving sideways ↔️";
const rLabelShort = (r: string) => r === "Bull" ? "Bullish" : r === "Bear" ? "Bearish" : "Sideways";
const COMPARISON_STRATEGIES = ["MA Crossover", "RSI", "Momentum", "Bollinger Bands"];

export function BeginnerView({ data, strategy }: { data: AnalyzeResponse; strategy: string }) {
  const m = data.overall_metrics[data.recommended_strategy] || Object.values(data.overall_metrics)[0];
  const curve = data.equity_curves[data.recommended_strategy] || [];
  const bh = data.equity_curves["Buy & Hold"] || [];
  
  // Calculate % return (assuming 100 = 1x)
  const stratReturn = curve.length > 0 ? ((curve[curve.length - 1].value / 100) - 1) * 100 : 0;
  const bhReturn = bh.length > 0 ? ((bh[bh.length - 1].value / 100) - 1) * 100 : 0;
  const outperformance = stratReturn - bhReturn;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div className="hero-card">
        <div style={{ fontSize: 13, color: "var(--text-200)", marginBottom: 6 }}>Best strategy for this market:</div>
        <div style={{ fontSize: 34, fontWeight: 800, color: "var(--text-100)", marginBottom: 4 }}>{data.recommended_strategy}</div>
        <div style={{ fontSize: 14, color: "var(--text-200)" }}>Returned <strong style={{ color: stratReturn >= 0 ? "var(--bull)" : "var(--bear)" }}>{stratReturn.toFixed(1)}%</strong> vs Buy&Hold <strong>{bhReturn.toFixed(1)}%</strong> ({outperformance >= 0 ? "+" : ""}{outperformance.toFixed(1)}% outperformance)</div>
      </div>

      <div className="metric-grid">
        <div className="metric-card"><div className="metric-card__label">Annual Return</div><div className="metric-card__value" style={{ color: m.CAGR >= 0 ? "var(--bull)" : "var(--bear)" }}>{(m.CAGR*100).toFixed(1)}%</div><div className="metric-card__desc">{m.CAGR >= 0.1 ? "Strong" : m.CAGR >= 0.05 ? "Moderate" : "Low"} growth</div></div>
        <div className="metric-card"><div className="metric-card__label">Risk-Adjusted Score</div><div className="metric-card__value" style={{ color: m.Sharpe >= 1 ? "var(--bull)" : m.Sharpe >= 0.5 ? "var(--sideways)" : "var(--bear)" }}>{m.Sharpe >= 1.5 ? "Excellent" : m.Sharpe >= 1 ? "Good" : m.Sharpe >= 0.5 ? "Fair" : "Poor"}</div><div className="metric-card__desc">Score: {m.Sharpe.toFixed(2)}</div></div>
        <div className="metric-card"><div className="metric-card__label">Max Drawdown</div><div className="metric-card__value" style={{ color: "var(--bear)" }}>{(m.MaxDrawdown*100).toFixed(0)}%</div><div className="metric-card__desc">Worst peak-to-trough loss</div></div>
      </div>

      {data.ohlc_data.length > 0 && <div className="card"><div className="card-title"><Activity size={16} /> Price Action</div><CandlestickChart data={data.ohlc_data} regimeTimeline={data.regime_timeline} height={340} /></div>}

      <div className="card">
        <div className="card-title"><TrendingUp size={16} /> Strategy Performance Comparison</div>
        <div style={{ fontSize: 12, color: "var(--text-200)", marginBottom: 12 }}>How each strategy would have performed over this period</div>
        <div style={{ height: 280 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart>
              <CartesianGrid stroke="var(--bg-300)" strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: "var(--text-200)" }} tickFormatter={(d: string) => d.slice(0,7)} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 10, fill: "var(--text-200)" }} tickFormatter={(v: number) => `${((v/100 - 1)*100).toFixed(0)}%`} />
              <Tooltip contentStyle={{ background: "var(--bg-200)", border: "1px solid var(--bg-300)", borderRadius: 8, fontSize: 12, color: "var(--text-100)" }} formatter={(v: any) => {const pct = ((Number(v)/100 - 1)*100).toFixed(1); return [`${pct}%`, ""]}} />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
              <Line data={curve} dataKey="value" name={`${data.recommended_strategy} (Recommended)`} stroke="var(--bull)" strokeWidth={2.5} dot={false} connectNulls />
              {bh.length > 0 && data.recommended_strategy !== "Buy & Hold" && (
                <Line data={bh} dataKey="value" name="Buy & Hold (Baseline)" stroke="var(--text-200)" strokeWidth={1.5} strokeDasharray="8 4" dot={false} connectNulls />
              )}
              {COMPARISON_STRATEGIES.map((s, idx) => {
                if (s === data.recommended_strategy) return null;
                const c = data.equity_curves[s];
                if (!c) return null;
                return <Line key={s} data={c} dataKey="value" name={s} stroke={`var(--accent-${(idx % 3) + 1}00)`} strokeWidth={1} dot={false} connectNulls opacity={0.3} />;
              })}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <div className="card-title"><Activity size={16} /> What the market was doing (Regime Timeline)</div>
        <div style={{ fontSize: 12, color: "var(--text-200)", marginBottom: 12 }}>Market conditions over your investment period. Different regimes favor different strategies.</div>
        <div className="regime-bar" style={{ height: 48, display: "flex", borderRadius: 8, overflow: "hidden", border: "1px solid var(--card-border)" }}>
          {data.regime_timeline.map((seg, i) => {
            const total = data.regime_timeline.reduce((a, x) => a + Math.max(x.days, 1), 0);
            const pct = (Math.max(seg.days, 1) / total) * 100;
            return <div key={i} title={`${seg.regime}: ${seg.start} → ${seg.end} (${seg.days} days)`} style={{ flex: `${pct} 0 0`, background: RB[seg.regime], borderRight: "1px solid var(--card-border)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 700, color: RC[seg.regime], textAlign: "center", padding: "0 4px" }}>{pct > 8 ? rLabelShort(seg.regime) : ""}</div>;
          })}
        </div>
        <div className="regime-legend">{["Bull","Bear","Sideways"].map(r => <div key={r} className="regime-legend-item"><div className="regime-dot" style={{ background: RC[r] }} />{rLabel(r)}</div>)}</div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <div className="card">
          <div className="card-title"><Lightbulb size={16} /> Why this strategy?</div>
          <div style={{ display: "inline-block", padding: "3px 12px", borderRadius: 20, fontSize: 11, fontWeight: 600, background: RB[data.current_regime], color: RC[data.current_regime], marginBottom: 12 }}>{rLabel(data.current_regime)}</div>
          <div style={{ fontSize: 13, color: "var(--text-200)", lineHeight: 1.7 }}>{data.recommended_strategy} performs best in {data.current_regime.toLowerCase()} markets. Based on historical backtests during similar conditions, it had the highest risk-adjusted returns.</div>
        </div>
        <div className="card">
          <div className="card-title"><Shield size={16} /> Risk Level</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: data.recommended_exposure.includes("100%") ? "var(--bull)" : data.recommended_exposure.includes("75%") ? "var(--sideways)" : "var(--bear)", marginBottom: 10 }}>
            {data.recommended_exposure.includes("100%") ? "Low Risk ✅" : data.recommended_exposure.includes("75%") ? "Moderate ⚠️" : "High Risk 🔴"}
          </div>
          <div style={{ fontSize: 13, color: "var(--text-200)", lineHeight: 1.7 }}>{data.recommended_exposure.includes("100%") ? "Market conditions look normal. Full exposure is reasonable." : data.recommended_exposure.includes("75%") ? "Elevated volatility. Use smaller position sizes." : "High drawdown risk. Trade cautiously."}</div>
          {data.risk_forecast && <div style={{ marginTop: 16, padding: "16px 12px", background: "linear-gradient(135deg, var(--bg-300), var(--bg-200))", borderRadius: 12, border: "1px solid var(--sideways)", boxShadow: "0 4px 12px rgba(0,0,0,0.1)" }}>
            <div style={{ fontSize: 12, color: "var(--sideways)", fontWeight: 700, marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
              <Shield size={14} /> 3-Month Probability Forecast
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
              <div style={{ textAlign: "center" }}><div style={{ fontSize: 10, color: "var(--text-200)" }}>Best Case</div><div style={{ fontSize: 14, fontWeight: 700, color: "var(--bull)" }}>-{(Math.abs(data.risk_forecast.best_case_90)*100).toFixed(0)}%</div></div>
              <div style={{ textAlign: "center" }}><div style={{ fontSize: 10, color: "var(--text-200)" }}>Median</div><div style={{ fontSize: 14, fontWeight: 700, color: "var(--text-100)" }}>-{(Math.abs(data.risk_forecast.median_50)*100).toFixed(0)}%</div></div>
              <div style={{ textAlign: "center" }}><div style={{ fontSize: 10, color: "var(--text-200)" }}>Worst Case</div><div style={{ fontSize: 14, fontWeight: 700, color: "var(--bear)" }}>-{(Math.abs(data.risk_forecast.worst_case_10)*100).toFixed(0)}%</div></div>
            </div>
            <div style={{ fontSize: 11, color: "var(--text-200)", marginTop: 10, fontStyle: "italic" }}>*90% confidence interval based on {data.current_regime} historical returns.</div>
          </div>}
        </div>
      </div>
    </div>
  );
}
