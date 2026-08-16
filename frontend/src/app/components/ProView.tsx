"use client";

import { useMemo } from "react";
import { AnalyzeResponse } from "@/lib/api";
import { CandlestickChart } from "./CandlestickChart";
import { ComposedChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, BarChart, Bar, Cell } from "recharts";
import { Download, TrendingUp, Shield, Brain, Activity, Flame } from "lucide-react";

const RC: Record<string, string> = { Bull: "var(--bull)", Bear: "var(--bear)", Sideways: "var(--sideways)" };
const RB: Record<string, string> = { Bull: "var(--bull-bg)", Bear: "var(--bear-bg)", Sideways: "var(--sideways-bg)" };
const SC: Record<string, string> = { "Buy & Hold": "#64748b", "MA Crossover": "#2563eb", RSI: "#0f766e", Momentum: "#16a34a", "Bollinger Bands": "#d97706", "Dual Momentum": "#be123c" };
const fmt = (n: number) => n >= 10000000 ? `₹${(n/10000000).toFixed(2)}Cr` : n >= 100000 ? `₹${(n/100000).toFixed(1)}L` : `₹${n.toLocaleString("en-IN")}`;
function hc(v: number) { return v >= 0.15 ? "rgba(16,185,129,0.35)" : v >= 0.05 ? "rgba(16,185,129,0.18)" : v >= 0 ? "rgba(16,185,129,0.08)" : v >= -0.10 ? "rgba(239,68,68,0.18)" : "rgba(239,68,68,0.35)"; }
function ht(v: number) { return v >= 0 ? "var(--bull)" : "var(--bear)"; }

export function ProView({ data, strategy }: { data: AnalyzeResponse; strategy: string }) {
  const exportCSV = () => {
    const rows = [["Strategy","CAGR","Sharpe","Sortino","MaxDrawdown","Calmar","Volatility"]];
    for (const [n, m] of Object.entries(data.overall_metrics)) rows.push([n, (m.CAGR*100).toFixed(2), m.Sharpe.toFixed(2), m.Sortino.toFixed(2), (m.MaxDrawdown*100).toFixed(2), m.Calmar.toFixed(2), (m.Volatility*100).toFixed(2)]);
    const blob = new Blob([rows.map(r => r.join(",")).join("\n")], { type: "text/csv" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `${data.ticker}_metrics.csv`; a.click();
  };

  const showComparison = strategy === "Recommend Strategy";

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
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {data.recommendation_status !== "validated_ml" && (
        <div className="card" style={{ border: "1px solid var(--sideways)", background: "var(--sideways-bg)" }}>
          <div className="card-title">ML needs more work</div>
          <div style={{ fontSize: 13, color: "var(--text-200)", lineHeight: 1.6 }}>
            ML is not driving this recommendation. Historical regime performance is active until validation clears the promotion gate.
          </div>
          <div style={{ fontSize: 11, color: "var(--text-200)", marginTop: 8 }}>{data.validation_reason}</div>
        </div>
      )}
      {/* Strategy Comparison Table — only when "Recommend Strategy" selected */}
      {showComparison && <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
          <div className="card-title"><TrendingUp size={16} /> Strategy Comparison</div>
          <button onClick={exportCSV} className="export-btn"><Download size={12} /> Export CSV</button>
        </div>
        <div style={{ overflowX: "auto" }}>
          <table className="metrics-table">
            <thead><tr>{["Strategy","CAGR","Sharpe","Sortino","Max DD","Calmar","Volatility"].map(h => <th key={h}>{h}</th>)}</tr></thead>
            <tbody>
              {Object.entries(data.overall_metrics).map(([n, m]) => (
                <tr key={n} style={{ background: "transparent" }}>
                  <td style={{ fontWeight: 700, color: "var(--text-100)" }}>{n}</td>
                  <td style={{ color: m.CAGR >= 0 ? "var(--bull)" : "var(--bear)" }}>{(m.CAGR*100).toFixed(1)}%</td>
                  <td style={{ color: m.Sharpe >= 1 ? "var(--bull)" : "var(--text-100)" }}>{m.Sharpe.toFixed(2)}</td>
                  <td style={{ color: m.Sortino >= 1 ? "var(--bull)" : "var(--text-100)" }}>{m.Sortino.toFixed(2)}</td>
                  <td style={{ color: "var(--bear)" }}>{(m.MaxDrawdown*100).toFixed(1)}%</td>
                  <td>{m.Calmar.toFixed(2)}</td>
                  <td>{(m.Volatility*100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>}

      {/* Candlestick */}
      {data.ohlc_data.length > 0 && <div className="card"><div className="card-title"><Activity size={16} /> Price Action (OHLC)</div><CandlestickChart data={data.ohlc_data} regimeTimeline={data.regime_timeline} height={380} /></div>}

      {/* Equity Curves */}
      <div className="card">
        <div className="card-title"><TrendingUp size={16} /> Equity Curves (₹{(data.initial_investment/100000).toFixed(0)}L initial)</div>
        <div style={{ height: 320 }}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData}>
              <CartesianGrid stroke="var(--bg-300)" strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: "var(--text-200)" }} tickFormatter={(d: string) => d.slice(0,7)} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 10, fill: "var(--text-200)" }} tickFormatter={(v: number) => v >= 100000 ? `${(v/100000).toFixed(1)}L` : `${(v/1000).toFixed(0)}k`} />
              <Tooltip contentStyle={{ background: "var(--bg-200)", border: "1px solid var(--bg-300)", borderRadius: 8, fontSize: 12, color: "var(--text-100)" }} formatter={(v: unknown) => [fmt(Math.round(Number(v))), ""]} />
              <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
              {Object.keys(data.equity_curves).map((n) => (
                <Line key={n} dataKey={n} name={n} stroke={SC[n] || "#3b82f6"} strokeWidth={n === strategy ? 3 : 1.5} strokeDasharray={n === "Buy & Hold" ? "8 4" : undefined} dot={false} connectNulls strokeOpacity={n === strategy || n === "Buy & Hold" ? 1 : 0.3} />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Regime Timeline */}
      <div className="card">
        <div className="card-title"><Activity size={16} /> Regime Timeline</div>
        <div style={{ fontSize: 12, color: "var(--text-200)", marginBottom: 12 }}>Market regimes detected over analysis period. Each regime may require different strategy selection.</div>
        <div style={{ height: 48, display: "flex", borderRadius: 8, overflow: "hidden", border: "1px solid var(--card-border)" }}>
          {data.regime_timeline.map((seg, i) => {
            const total = data.regime_timeline.reduce((a, x) => a + Math.max(x.days, 1), 0);
            const pct = (Math.max(seg.days, 1) / total) * 100;
            return <div key={i} title={`${seg.regime}: ${seg.start} → ${seg.end} (${seg.days} days)`} style={{ flex: `${pct} 0 0`, background: RB[seg.regime], borderRight: "1px solid var(--card-border)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 600, color: RC[seg.regime], textAlign: "center", padding: "0 8px" }}>{pct > 10 ? seg.regime : ""}</div>;
          })}
        </div>
        <div className="regime-legend">{["Bull","Bear","Sideways"].map(r => <div key={r} className="regime-legend-item"><div className="regime-dot" style={{ background: RC[r] }} />{r}</div>)}</div>
      </div>

      {/* Bottom Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 14 }}>
        {/* Heatmap */}
        <div className="card">
          <div className="card-title"><Flame size={16} /> Regime-Conditional CAGR</div>
          {(() => {
            const regimes = ["Bull","Bear","Sideways"];
            const strats = [...new Set(data.regime_heatmap.map(h => h.strategy))];
            return <div style={{ display: "grid", gridTemplateColumns: `110px repeat(${regimes.length}, 1fr)`, gap: 4 }}>
              <div />
              {regimes.map(r => <div key={r} style={{ textAlign: "center", fontSize: 12, color: RC[r], fontWeight: 700, padding: "6px 0" }}>{r}</div>)}
              {strats.map(st => <div key={st} style={{ display: "contents" }}>
                <div style={{ fontSize: 11, color: "var(--text-200)", display: "flex", alignItems: "center", fontWeight: st === strategy ? 700 : 500 }}>{st}{st === strategy ? " ⭐" : ""}</div>
                {regimes.map(r => { const v = data.regime_heatmap.find(h => h.strategy === st && h.regime === r)?.CAGR ?? 0; return <div key={`${st}-${r}`} className="heatmap-cell" style={{ background: hc(v), color: ht(v) }}>{(v*100).toFixed(1)}%</div>; })}
              </div>)}
            </div>;
          })()}
        </div>

        {/* Right Column */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div className="card">
            <div style={{ fontSize: 11, color: "var(--accent-200)", fontWeight: 600, marginBottom: 8 }}>💡 Current Recommendation</div>
            <div style={{ display: "inline-block", padding: "3px 10px", borderRadius: 20, fontSize: 11, fontWeight: 600, background: RB[data.current_regime], color: RC[data.current_regime], marginBottom: 8 }}>{data.current_regime} regime</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: "var(--text-100)", marginBottom: 4 }}>{data.recommended_strategy}</div>
            <div style={{ fontSize: 12, color: "var(--text-200)", lineHeight: 1.6 }}>{data.recommendation_reason}</div>
          </div>
          <div className="card">
            <div className="card-title"><Shield size={16} /> Risk Forecast (63-day)</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <div className="risk-row"><span style={{ color: "var(--text-200)" }}>Exposure Limit</span><span style={{ fontWeight: 700, color: data.recommended_exposure.includes("25%") || data.recommended_exposure.includes("50%") ? "var(--bear)" : "var(--text-100)" }}>{data.recommended_exposure}</span></div>
              <div className="risk-row"><span style={{ color: "var(--text-200)" }}>Expected DD</span><span style={{ fontWeight: 700, color: "var(--sideways)" }}>{data.risk_forecast ? `${(data.risk_forecast.best_case_90*100).toFixed(0)}% to ${(data.risk_forecast.median_50*100).toFixed(0)}%` : "N/A"}</span></div>
              <div className="risk-row"><span style={{ color: "var(--text-200)" }}>Worst Case (10th)</span><span style={{ fontWeight: 700, color: "var(--bear)" }}>{data.risk_forecast ? `${(data.risk_forecast.worst_case_10*100).toFixed(1)}%` : "N/A"}</span></div>
            </div>
          </div>
        </div>
      </div>

      {/* ML Strategy Selection Confidence */}
      {Object.keys(data.probabilities).length > 0 && <div className="card">
        <div className="card-title"><Brain size={16} /> Strategy Selection Confidence (ML Classifier)</div>
        <div style={{ fontSize: 12, color: "var(--text-200)", marginBottom: 12 }}>Probability that each strategy will outperform in current market regime</div>
        <div style={{ height: 180 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={Object.entries(data.probabilities).map(([k, v]) => ({ name: k, probability: v * 100 }))}>
              <CartesianGrid stroke="var(--bg-300)" strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: "var(--text-200)" }} />
              <YAxis tick={{ fontSize: 10, fill: "var(--text-200)" }} domain={[0, 100]} />
              <Tooltip contentStyle={{ background: "var(--bg-200)", border: "1px solid var(--bg-300)", borderRadius: 8, fontSize: 12 }} />
              <Bar dataKey="probability" radius={[4,4,0,0]}>
                {Object.entries(data.probabilities).map(([k]) => <Cell key={k} fill={SC[k] || "#3b82f6"} fillOpacity={k === strategy ? 1 : 0.4} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>}
    </div>
  );
}
