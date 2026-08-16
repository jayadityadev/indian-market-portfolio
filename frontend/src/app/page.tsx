"use client";

import { useState } from "react";
import { fetchAnalysis, AnalyzeResponse } from "@/lib/api";
import { BeginnerView } from "./components/BeginnerView";
import { ProView } from "./components/ProView";
import { MagicBento } from "./components/MagicBento";
import { NewsPanel } from "./components/NewsPanel";
import { BeginnerLandingBrief } from "./components/BeginnerLandingBrief";
import { ProLandingTelemetry } from "./components/ProLandingTelemetry";
import { errorMessage } from "@/lib/errors";
import {
  Play,
  Loader2,
  AlertTriangle,
  Target,
  Settings,
} from "lucide-react";

const TICKER_OPTIONS = [
  { label: "NIFTY 50", value: "^NSEI" },
];

const STRATEGY_OPTIONS = [
  "Recommend Strategy",
  "Buy & Hold",
  "MA Crossover",
  "RSI",
  "Momentum",
  "Bollinger Bands",
  "Dual Momentum",
];

export default function Dashboard() {
  const [mode, setMode] = useState<"beginner" | "pro">("beginner");
  const today = new Date().toISOString().split('T')[0];
  const threeYearsAgo = new Date(new Date().setFullYear(new Date().getFullYear() - 3)).toISOString().split('T')[0];

  const [ticker, setTicker] = useState("^NSEI");
  const [startDate, setStartDate] = useState(threeYearsAgo);
  const [endDate, setEndDate] = useState(today);
  const [strategy, setStrategy] = useState("Recommend Strategy");
  const [investment, setInvestment] = useState(100000);
  const [data, setData] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showWarning, setShowWarning] = useState(false);

  const runAnalysis = async () => {
    setLoading(true);
    setError("");
    try {
      const analysisStrategy = mode === "beginner" ? "Recommend Strategy" : strategy;
      const result = await fetchAnalysis(
        ticker,
        startDate,
        endDate,
        analysisStrategy,
        investment
      );
      setData(result);
    } catch (error: unknown) {
      setError(errorMessage(error, "Analysis failed"));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-grid">
      {/* Professional Mode Warning Modal */}
      {showWarning && (
        <div className="modal-overlay">
          <div className="modal-box">
            <div
              style={{
                fontSize: 18,
                fontWeight: 700,
                color: "var(--sideways)",
                marginBottom: 12,
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              <AlertTriangle size={20} /> Professional Mode
            </div>
            <p
              style={{
                fontSize: 13,
                color: "var(--text-200)",
                lineHeight: 1.8,
                marginBottom: 20,
              }}
            >
              Professional mode shows raw quantitative metrics: Sharpe, Sortino,
              Calmar ratios, regime-conditional CAGR breakdowns, ML
              probabilities, and statistical risk distributions.
              <br />
              <br />
              <strong style={{ color: "var(--text-100)" }}>
                This requires familiarity with quantitative finance.
              </strong>{" "}
              If you{"'"}re new to investing, Beginner mode gives the same
              insights in plain language.
            </p>
            <div style={{ display: "flex", gap: 10 }}>
              <button
                onClick={() => {
                  setMode("pro");
                  setShowWarning(false);
                }}
                className="run-btn"
                style={{ flex: 1, marginTop: 0 }}
              >
                Switch to Professional
              </button>
              <button
                onClick={() => setShowWarning(false)}
                className="export-btn"
                style={{ flex: 1, padding: 12, fontSize: 13, justifyContent: "center" }}
              >
                Stay in Beginner
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ===== SIDEBAR ===== */}
      <aside className="sidebar">
        <div className="sidebar-header" style={{ borderBottom: "none", paddingBottom: 0 }}>
          <div className="sidebar-section-label" style={{ margin: 0, fontSize: 11, color: "var(--accent-100)" }}>
            Strategy Engine &amp; Controls
          </div>
        </div>

        {/* Mode Toggle */}
        <div className="mode-toggle">
          <button
            onClick={() => setMode("beginner")}
            className={`mode-btn ${mode === "beginner" ? "mode-btn--active" : "mode-btn--inactive"}`}
          >
            <Target size={13} /> Beginner
          </button>
          <button
            onClick={() =>
              mode === "beginner" ? setShowWarning(true) : setMode("beginner")
            }
            className={`mode-btn ${mode === "pro" ? "mode-btn--active" : "mode-btn--inactive"}`}
          >
            <Settings size={13} /> Professional
          </button>
        </div>

        {/* Inputs */}
        <div>
          <div className="sidebar-section-label">
            {mode === "beginner"
              ? "What do you want to analyse?"
              : "Analysis Parameters"}
          </div>

          <label className="sidebar-label">
            {mode === "beginner" ? "Which stock or index?" : "Ticker"}
          </label>
          <select
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            className="sidebar-select"
          >
            {TICKER_OPTIONS.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
        </div>

        {/* Strategy Selection — Pro mode only */}
        {mode === "pro" && (
          <div>
            <label className="sidebar-label">Strategy</label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className="sidebar-select"
            >
              {STRATEGY_OPTIONS.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Date Range */}
        <div>
          <label className="sidebar-label">
            {mode === "beginner" ? "Time period" : "Date Range"}
          </label>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="sidebar-input"
            />
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="sidebar-input"
            />
          </div>
        </div>

        {/* Initial Capital — Pro mode only */}
        {mode === "pro" && (
          <div>
            <label className="sidebar-label">Initial Capital (₹)</label>
            <input
              type="number"
              value={investment}
              onChange={(e) => setInvestment(Number(e.target.value))}
              className="sidebar-input"
            />
          </div>
        )}

        {/* Run Button */}
        <button onClick={runAnalysis} disabled={loading} className="run-btn">
          {loading ? (
            <>
              <Loader2 size={16} className="spin" /> Analysing...
            </>
          ) : (
            <>
              <Play size={14} />{" "}
              {mode === "beginner" ? "Analyse for me" : "Run Analysis"}
            </>
          )}
        </button>

        <div style={{ flexGrow: 1 }} />

        {/* User Profile */}
        <div className="sidebar-profile" style={{ display: "flex", alignItems: "center", gap: 12, marginTop: "auto", paddingTop: 20, borderTop: "1px solid var(--bg-300)" }}>
          <div style={{ width: 36, height: 36, borderRadius: "50%", background: "var(--accent-100)", display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontWeight: "bold", fontSize: 14 }}>
            HB
          </div>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <span style={{ fontSize: 14, fontWeight: 600, color: "var(--text-100)" }}>Hue Brew</span>
            <span style={{ fontSize: 12, color: "var(--text-200)" }}>Designer</span>
          </div>
          <Settings size={16} style={{ marginLeft: "auto", color: "var(--text-200)", cursor: "pointer" }} />
        </div>
      </aside>

      {/* ===== MAIN CONTENT ===== */}
      <main className="main-content">
        <NewsPanel />

        {/* Pre-Analysis Adaptive Landing */}
        {!data && !error && !loading && (
          <>
            {mode === "beginner" ? <BeginnerLandingBrief /> : <ProLandingTelemetry />}
            <MagicBento />
          </>
        )}

        {error && (
          <div className="error-banner">
            <AlertTriangle size={16} /> {error}
          </div>
        )}

        {/* Post-Analysis Views with Progressive Disclosure */}
        {data && mode === "beginner" && (
          <BeginnerView data={data} />
        )}
        {data && mode === "pro" && (
          <ProView data={data} strategy={strategy} />
        )}
      </main>
    </div>
  );
}
