"use client";

import { useEffect, useState } from "react";
import { fetchBenchmark, BenchmarkResponse, BenchmarkMetrics } from "@/lib/benchmark";
import { errorMessage } from "@/lib/errors";
import { Cpu, Zap, RefreshCw, BarChart3, Database, ShieldAlert } from "lucide-react";

export default function BenchmarkPage() {
  const [data, setData] = useState<BenchmarkResponse | null>(null);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);

  const loadBenchmark = async () => {
    setLoading(true);
    setError("");
    try {
      const result = await fetchBenchmark();
      setData(result);
    } catch (err: unknown) {
      setError(errorMessage(err, "Failed to load model benchmark data"));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadBenchmark();
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  const metricsList: { key: keyof BenchmarkMetrics; label: string; desc: string }[] = [
    { key: "accuracy", label: "Accuracy", desc: "Overall classification accuracy across 6 strategy classes" },
    { key: "f1_score", label: "F1 Score (Weighted)", desc: "Harmonic mean of precision and recall accounting for class balance" },
    { key: "precision", label: "Precision", desc: "True positive rate over all predicted strategy recommendations" },
    { key: "recall", label: "Recall", desc: "True positive rate over actual optimal historical regimes" },
  ];

  return (
    <div className="subpage-container">
      {/* Header */}
      <div className="subpage-header">
        <div className="subpage-badge">
          <Cpu size={13} />
          <span>Academic Benchmark Replication • IEEE Access 2024</span>
        </div>
        <h1 className="subpage-title">Model Benchmark Comparison</h1>
        <p className="subpage-desc">
          Head-to-head empirical evaluation between our production XGBoost Strategy Classifier and the
          PyTorch LSTM-DNN architecture proposed by Alam et al. (IEEE Access 2024).
        </p>
      </div>

      {/* Validation Gate Alert */}
      <div
        className="card"
        style={{
          border: "1px solid var(--primary-300)",
          background: "linear-gradient(135deg, var(--bg-200), var(--primary-100))",
          marginBottom: 24,
        }}
      >
        <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
          <ShieldAlert size={20} style={{ color: "var(--accent-100)", flexShrink: 0, marginTop: 2 }} />
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text-100)", marginBottom: 4 }}>
              Academic Benchmark Notice & Promotion Gating
            </div>
            <p style={{ fontSize: 13, color: "var(--text-200)", lineHeight: 1.6, margin: 0 }}>
              This benchmark tracks academic model architectures on NIFTY 50 price sequences. Production strategy recommendations
              are subject to walk-forward validation and causal regime gating to prevent out-of-sample overfit.
            </p>
          </div>
        </div>
      </div>

      {loading && (
        <div className="card" style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: 280, gap: 12 }}>
          <RefreshCw size={20} className="spin" style={{ color: "var(--accent-100)" }} />
          <span style={{ color: "var(--text-200)", fontSize: 14 }}>Evaluating models against test dataset...</span>
        </div>
      )}

      {error && (
        <div className="error-banner">
          <div>{error}</div>
          <button onClick={loadBenchmark} className="export-btn" style={{ marginLeft: "auto" }}>
            <RefreshCw size={12} /> Retry
          </button>
        </div>
      )}

      {data && (
        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          {/* Side-by-Side Model Architecture Cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 20 }}>
            {/* XGBoost Card */}
            <div className="card" style={{ borderTop: "3px solid var(--accent-100)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{ padding: 6, borderRadius: 8, background: "var(--primary-100)", color: "var(--accent-100)" }}>
                    <Zap size={16} />
                  </div>
                  <div>
                    <h2 style={{ fontSize: 16, fontWeight: 700, color: "var(--text-100)", margin: 0 }}>XGBoost Classifier</h2>
                    <span style={{ fontSize: 11, color: "var(--text-200)" }}>Gradient Boosted Decision Trees</span>
                  </div>
                </div>
                <span style={{ fontSize: 10, fontWeight: 800, padding: "2px 8px", borderRadius: 999, background: "var(--bull-bg)", color: "var(--bull)" }}>
                  PRODUCTION READY
                </span>
              </div>

              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 16 }}>
                <span className="subpage-badge" style={{ fontSize: 10, padding: "2px 8px", margin: 0 }}>6 Classes</span>
                <span className="subpage-badge" style={{ fontSize: 10, padding: "2px 8px", margin: 0 }}>Purged CV</span>
                <span className="subpage-badge" style={{ fontSize: 10, padding: "2px 8px", margin: 0 }}>Calibrated Probabilities</span>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {metricsList.map((m) => {
                  const val = (data.xgboost_metrics[m.key] as number) ?? 0;
                  return (
                    <div key={m.key}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                        <span style={{ color: "var(--text-200)", fontWeight: 600 }}>{m.label}</span>
                        <strong style={{ color: "var(--text-100)" }}>{(val * 100).toFixed(1)}%</strong>
                      </div>
                      <div style={{ height: 6, borderRadius: 999, background: "var(--bg-300)", overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${Math.min(val * 100, 100)}%`, background: "var(--accent-100)", borderRadius: 999 }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* LSTM-DNN Card */}
            <div className="card" style={{ borderTop: "3px solid #3b82f6" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{ padding: 6, borderRadius: 8, background: "rgba(59, 130, 246, 0.12)", color: "#3b82f6" }}>
                    <Cpu size={16} />
                  </div>
                  <div>
                    <h2 style={{ fontSize: 16, fontWeight: 700, color: "var(--text-100)", margin: 0 }}>PyTorch LSTM-DNN</h2>
                    <span style={{ fontSize: 11, color: "var(--text-200)" }}>IEEE Access 2024 Topology</span>
                  </div>
                </div>
                <span style={{ fontSize: 10, fontWeight: 800, padding: "2px 8px", borderRadius: 999, background: "var(--primary-100)", color: "var(--text-200)" }}>
                  ACADEMIC BENCHMARK
                </span>
              </div>

              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 16 }}>
                <span className="subpage-badge" style={{ fontSize: 10, padding: "2px 8px", margin: 0 }}>2-Layer LSTM</span>
                <span className="subpage-badge" style={{ fontSize: 10, padding: "2px 8px", margin: 0 }}>4 Dense Layers</span>
                <span className="subpage-badge" style={{ fontSize: 10, padding: "2px 8px", margin: 0 }}>Dropout + LayerNorm</span>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {metricsList.map((m) => {
                  const val = (data.lstm_metrics[m.key] as number) ?? 0;
                  return (
                    <div key={m.key}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                        <span style={{ color: "var(--text-200)", fontWeight: 600 }}>{m.label}</span>
                        <strong style={{ color: "var(--text-100)" }}>{(val * 100).toFixed(1)}%</strong>
                      </div>
                      <div style={{ height: 6, borderRadius: 999, background: "var(--bg-300)", overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${Math.min(val * 100, 100)}%`, background: "#3b82f6", borderRadius: 999 }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Comparative Metrics Table */}
          <div className="card">
            <div className="card-title">
              <BarChart3 size={16} /> Quantitative Comparative Breakdown
            </div>
            <div style={{ overflowX: "auto" }}>
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Evaluation Metric</th>
                    <th>XGBoost Recommender</th>
                    <th>PyTorch LSTM-DNN</th>
                    <th>Delta / Advantage</th>
                  </tr>
                </thead>
                <tbody>
                  {metricsList.map((m) => {
                    const xgb = (data.xgboost_metrics[m.key] as number) ?? 0;
                    const lstm = (data.lstm_metrics[m.key] as number) ?? 0;
                    const delta = (xgb - lstm) * 100;
                    return (
                      <tr key={m.key}>
                        <td>
                          <div style={{ fontWeight: 700, color: "var(--text-100)" }}>{m.label}</div>
                          <div style={{ fontSize: 11, color: "var(--text-200)" }}>{m.desc}</div>
                        </td>
                        <td style={{ fontSize: 14, fontWeight: 700, color: "var(--accent-100)" }}>
                          {(xgb * 100).toFixed(2)}%
                        </td>
                        <td style={{ fontSize: 14, fontWeight: 700, color: "#3b82f6" }}>
                          {(lstm * 100).toFixed(2)}%
                        </td>
                        <td style={{ fontWeight: 700, color: delta >= 0 ? "var(--bull)" : "var(--bear)" }}>
                          {delta >= 0 ? `+${delta.toFixed(2)}% (XGBoost)` : `${delta.toFixed(2)}% (LSTM)`}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Provenance Card */}
          {data.data_provenance && (
            <div className="card" style={{ background: "var(--bg-100)", border: "1px solid var(--card-border)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, fontSize: 13, fontWeight: 700, color: "var(--text-100)" }}>
                <Database size={15} /> Dataset Provenance & Reproducibility
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12, fontSize: 12 }}>
                <div><span style={{ color: "var(--text-200)" }}>Ticker:</span> <strong style={{ color: "var(--text-100)" }}>{data.data_provenance.ticker}</strong></div>
                <div><span style={{ color: "var(--text-200)" }}>Start Date:</span> <strong style={{ color: "var(--text-100)" }}>{data.data_provenance.start}</strong></div>
                <div><span style={{ color: "var(--text-200)" }}>End Date:</span> <strong style={{ color: "var(--text-100)" }}>{data.data_provenance.end}</strong></div>
                <div><span style={{ color: "var(--text-200)" }}>Evaluation Purpose:</span> <strong style={{ color: "var(--text-100)" }}>{data.data_provenance.purpose}</strong></div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
