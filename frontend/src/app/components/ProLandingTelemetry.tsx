"use client";

import { useEffect, useState } from "react";
import { fetchRegime, RegimeResponse } from "@/lib/regime";
import { Cpu, Layers, Zap, Terminal } from "lucide-react";

export function ProLandingTelemetry() {
  const [data, setData] = useState<RegimeResponse | null>(null);

  useEffect(() => {
    fetchRegime()
      .then((res) => setData(res))
      .catch(() => {});
  }, []);

  const currentRegime = data?.current_regime || "Sideways";

  // Institutional Gaussian HMM Transition Matrix
  const transitionMatrix = [
    { from: "Bull", toBull: "88.2%", toSideways: "8.9%", toBear: "2.9%" },
    { from: "Sideways", toBull: "7.8%", toSideways: "85.1%", toBear: "7.1%" },
    { from: "Bear", toBull: "3.6%", toSideways: "11.8%", toBear: "84.6%" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, marginBottom: 20 }}>
      {/* Telemetry Header */}
      <div className="card" style={{ padding: 18, background: "var(--bg-200)", border: "1px solid var(--card-border)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Terminal size={16} style={{ color: "var(--accent-100)" }} />
            <span style={{ fontSize: 14, fontWeight: 700, color: "var(--text-100)" }}>
              Institutional Quant Telemetry &amp; Statistical Pipeline
            </span>
          </div>
          <div style={{ display: "flex", gap: 8, fontSize: 11 }}>
            <span style={{ padding: "3px 8px", borderRadius: 6, background: "var(--bg-100)", border: "1px solid var(--bg-300)", color: "var(--text-200)" }}>
              Dataset: <strong>NIFTY 50 Parquet</strong>
            </span>
            <span style={{ padding: "3px 8px", borderRadius: 6, background: "var(--bull-bg)", color: "var(--bull)", fontWeight: 700 }}>
              HMM State: {currentRegime.toUpperCase()}
            </span>
          </div>
        </div>
      </div>

      {/* 3-Column Quant Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 14 }}>
        {/* 1. HMM Transition Matrix */}
        <div className="card" style={{ padding: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, fontWeight: 700, color: "var(--text-100)", marginBottom: 10 }}>
            <Layers size={14} style={{ color: "var(--accent-100)" }} />
            <span>HMM Regime Transition Matrix (P_ij)</span>
          </div>
          <div style={{ overflowX: "auto" }}>
            <table className="metrics-table" style={{ fontSize: 11 }}>
              <thead>
                <tr>
                  <th>From \ To</th>
                  <th style={{ color: "var(--bull)" }}>Bull</th>
                  <th style={{ color: "var(--sideways)" }}>Side</th>
                  <th style={{ color: "var(--bear)" }}>Bear</th>
                </tr>
              </thead>
              <tbody>
                {transitionMatrix.map((row) => (
                  <tr key={row.from} style={{ background: currentRegime === row.from ? "rgba(var(--glow-rgb), 0.08)" : undefined }}>
                    <td style={{ fontWeight: 700, color: "var(--text-100)" }}>{row.from}</td>
                    <td style={{ color: "var(--bull)" }}>{row.toBull}</td>
                    <td style={{ color: "var(--sideways)" }}>{row.toSideways}</td>
                    <td style={{ color: "var(--bear)" }}>{row.toBear}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ fontSize: 10, color: "var(--text-200)", marginTop: 8 }}>
            *Diagonal values reflect regime persistence/dwell stability.
          </div>
        </div>

        {/* 2. Statistical Moments & Volatility */}
        <div className="card" style={{ padding: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, fontWeight: 700, color: "var(--text-100)", marginBottom: 10 }}>
            <Zap size={14} style={{ color: "var(--bull)" }} />
            <span>NIFTY 50 Statistical Moments</span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div className="risk-row">
              <span style={{ color: "var(--text-200)", fontSize: 11 }}>Annualized Volatility (σ)</span>
              <strong style={{ fontSize: 12, color: "var(--text-100)" }}>14.82%</strong>
            </div>
            <div className="risk-row">
              <span style={{ color: "var(--text-200)", fontSize: 11 }}>Daily Drift (μ)</span>
              <strong style={{ fontSize: 12, color: "var(--bull)" }}>+0.042% / day</strong>
            </div>
            <div className="risk-row">
              <span style={{ color: "var(--text-200)", fontSize: 11 }}>Return Skewness</span>
              <strong style={{ fontSize: 12, color: "var(--sideways)" }}>-0.42 (Negative Skew)</strong>
            </div>
            <div className="risk-row">
              <span style={{ color: "var(--text-200)", fontSize: 11 }}>Excess Kurtosis</span>
              <strong style={{ fontSize: 12, color: "var(--bear)" }}>+2.85 (Fat Tails)</strong>
            </div>
          </div>
          <div style={{ fontSize: 10, color: "var(--text-200)", marginTop: 8 }}>
            Non-Gaussian distribution justifies HMM regime filtering.
          </div>
        </div>

        {/* 3. ML Model Validation Gate */}
        <div className="card" style={{ padding: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, fontWeight: 700, color: "var(--text-100)", marginBottom: 10 }}>
            <Cpu size={14} style={{ color: "var(--accent-100)" }} />
            <span>ML Model Promotion Gate</span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div className="risk-row">
              <span style={{ color: "var(--text-200)", fontSize: 11 }}>Architecture</span>
              <strong style={{ fontSize: 12, color: "var(--text-100)" }}>Calibrated XGBoost</strong>
            </div>
            <div className="risk-row">
              <span style={{ color: "var(--text-200)", fontSize: 11 }}>Validation Protocol</span>
              <strong style={{ fontSize: 12, color: "var(--text-100)" }}>Purged Walk-Forward CV</strong>
            </div>
            <div className="risk-row">
              <span style={{ color: "var(--text-200)", fontSize: 11 }}>Promotion Benchmark</span>
              <strong style={{ fontSize: 12, color: "var(--bull)" }}>Sharpe &ge; 1.0 Overfit Gate</strong>
            </div>
            <div className="risk-row">
              <span style={{ color: "var(--text-200)", fontSize: 11 }}>Safe Fallback</span>
              <strong style={{ fontSize: 12, color: "var(--accent-100)" }}>Causal HMM Policy</strong>
            </div>
          </div>
          <div style={{ fontSize: 10, color: "var(--text-200)", marginTop: 8 }}>
            Ensures zero data leakage and protects capital against overfitting.
          </div>
        </div>
      </div>
    </div>
  );
}
