"use client";

import { useCallback, useEffect, useState } from "react";
import { fetchAnalysis } from "@/lib/api";
import { fetchReport, ReportResponse } from "@/lib/report";
import { errorMessage } from "@/lib/errors";
import { Sparkles, Copy, Check, RefreshCw, Cpu, Clock, Layers } from "lucide-react";

const PROVIDERS = [
  { label: "⚡ Auto (Waterfall Cascade)", value: "" },
  { label: "Google Gemini", value: "gemini" },
  { label: "Groq (Llama 3.3 70B)", value: "groq" },
  { label: "NVIDIA NIM (Llama 3.3 70B)", value: "nvidia:meta/llama-3.3-70b-instruct" },
  { label: "OpenRouter", value: "openrouter" },
];

/** Clean custom markdown formatting for financial commentary without heavy external dependencies */
function renderMarkdown(raw: string) {
  if (!raw) return null;

  const lines = raw.split("\n");
  const elements: React.ReactNode[] = [];
  let currentList: string[] = [];

  const flushList = () => {
    if (currentList.length > 0) {
      elements.push(
        <ul key={`ul-${elements.length}`} style={{ paddingLeft: 20, marginBottom: 14 }}>
          {currentList.map((item, idx) => (
            <li key={idx} style={{ marginBottom: 4, color: "var(--text-100)", fontSize: 13, lineHeight: 1.6 }}>
              <span dangerouslySetInnerHTML={{ __html: formatInline(item) }} />
            </li>
          ))}
        </ul>
      );
      currentList = [];
    }
  };

  const formatInline = (text: string) => {
    return text
      .replace(/\*\*(.*?)\*\*/g, "<strong style='color: var(--text-100); font-weight: 700;'>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/`([^`]+)`/g, "<code style='background: var(--bg-300); padding: 2px 6px; border-radius: 6px; font-size: 12px;'>$1</code>");
  };

  lines.forEach((line, index) => {
    const trimmed = line.trim();

    if (trimmed.startsWith("# ")) {
      flushList();
      elements.push(
        <h1 key={index} style={{ fontFamily: "var(--font-display)", fontSize: 22, fontWeight: 700, marginTop: 24, marginBottom: 12, color: "var(--text-100)" }}>
          {trimmed.replace("# ", "")}
        </h1>
      );
    } else if (trimmed.startsWith("## ")) {
      flushList();
      elements.push(
        <h2 key={index} style={{ fontFamily: "var(--font-display)", fontSize: 18, fontWeight: 700, marginTop: 20, marginBottom: 10, color: "var(--accent-100)", borderBottom: "1px solid var(--bg-300)", paddingBottom: 4 }}>
          {trimmed.replace("## ", "")}
        </h2>
      );
    } else if (trimmed.startsWith("### ")) {
      flushList();
      elements.push(
        <h3 key={index} style={{ fontSize: 15, fontWeight: 700, marginTop: 16, marginBottom: 8, color: "var(--text-100)" }}>
          {trimmed.replace("### ", "")}
        </h3>
      );
    } else if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      currentList.push(trimmed.substring(2));
    } else if (trimmed.startsWith("> ")) {
      flushList();
      elements.push(
        <blockquote key={index} style={{ borderLeft: "3px solid var(--accent-100)", background: "var(--bg-100)", padding: "10px 16px", borderRadius: "0 10px 10px 0", margin: "12px 0", color: "var(--text-200)", fontSize: 13, fontStyle: "italic" }}>
          <span dangerouslySetInnerHTML={{ __html: formatInline(trimmed.replace("> ", "")) }} />
        </blockquote>
      );
    } else if (trimmed.length > 0) {
      flushList();
      elements.push(
        <p key={index} style={{ marginBottom: 12, fontSize: 13, lineHeight: 1.7, color: "var(--text-100)" }}>
          <span dangerouslySetInnerHTML={{ __html: formatInline(trimmed) }} />
        </p>
      );
    }
  });

  flushList();
  return elements;
}

export default function ReportPage() {
  const [provider, setProvider] = useState<string>("");
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [copied, setCopied] = useState<boolean>(false);

  const generateReport = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const analysis = await fetchAnalysis();
      const result = await fetchReport(analysis, provider || undefined);
      setReport(result);
    } catch (err: unknown) {
      setError(errorMessage(err, "Failed to generate analyst report"));
    } finally {
      setLoading(false);
    }
  }, [provider]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void generateReport();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [generateReport]);

  const copyToClipboard = () => {
    if (!report?.report) return;
    navigator.clipboard.writeText(report.report);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="subpage-container">
      {/* Header */}
      <div className="subpage-header">
        <div className="subpage-badge">
          <Sparkles size={13} />
          <span>Multi-Provider Waterfall Cascade • LLM Analyst Subsystem</span>
        </div>
        <h1 className="subpage-title">AI Market Analyst Report</h1>
        <p className="subpage-desc">
          Institutional macroeconomic synthesis and quantitative strategy assessment. Dynamically orchestrated
          across leading inference engines with automatic fallback and zero-hallucination guardrails.
        </p>
      </div>

      {/* Controls Bar */}
      <div
        className="card"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: 14,
          padding: "16px 20px",
          marginBottom: 24,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", flex: 1 }}>
          <span style={{ fontSize: 12, fontWeight: 700, color: "var(--text-200)" }}>Inference Provider:</span>
          <select
            value={provider}
            onChange={(e) => setProvider(e.target.value)}
            className="sidebar-select"
            style={{ maxWidth: 300, padding: "8px 12px", fontSize: 13 }}
            disabled={loading}
          >
            {PROVIDERS.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))}
          </select>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {report && (
            <button
              onClick={copyToClipboard}
              className="export-btn"
              title="Copy Markdown Report"
              style={{ padding: "9px 14px", fontSize: 12 }}
            >
              {copied ? <Check size={14} style={{ color: "var(--bull)" }} /> : <Copy size={14} />}
              <span>{copied ? "Copied!" : "Copy Report"}</span>
            </button>
          )}

          <button
            onClick={generateReport}
            disabled={loading}
            className="run-btn"
            style={{ margin: 0, padding: "9px 18px", fontSize: 13, width: "auto" }}
          >
            {loading ? <RefreshCw size={14} className="spin" /> : <Sparkles size={14} />}
            <span>{loading ? "Synthesizing..." : "Regenerate Report"}</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="error-banner">
          <div>{error}</div>
          <button onClick={generateReport} className="export-btn" style={{ marginLeft: "auto" }}>
            <RefreshCw size={12} /> Retry
          </button>
        </div>
      )}

      {loading && !report && (
        <div className="card" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: 360, gap: 14 }}>
          <RefreshCw size={24} className="spin" style={{ color: "var(--accent-100)" }} />
          <div style={{ fontSize: 15, fontWeight: 700, color: "var(--text-100)" }}>
            Generating Institutional Analyst Report
          </div>
          <p style={{ fontSize: 13, color: "var(--text-200)", maxWidth: 440, textAlign: "center", margin: 0 }}>
            Analyzing Gaussian HMM regime shifts, backtest metrics, risk distributions, and latest market news context...
          </p>
        </div>
      )}

      {report && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Metadata Chips Bar */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              flexWrap: "wrap",
              gap: 12,
              padding: "10px 16px",
              background: "var(--bg-200)",
              borderRadius: 14,
              border: "1px solid var(--card-border)",
              fontSize: 12,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <Cpu size={14} style={{ color: "var(--accent-100)" }} />
              <span style={{ color: "var(--text-200)" }}>Provider:</span>
              <strong style={{ color: "var(--text-100)", textTransform: "capitalize" }}>{report.provider_used}</strong>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <Layers size={14} style={{ color: "#3b82f6" }} />
              <span style={{ color: "var(--text-200)" }}>Model:</span>
              <code style={{ fontSize: 11, background: "var(--bg-100)", padding: "2px 6px", borderRadius: 4, color: "var(--text-100)" }}>
                {report.model_used}
              </code>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: "auto" }}>
              <Clock size={14} style={{ color: "var(--text-200)" }} />
              <span style={{ color: "var(--text-200)" }}>Generated:</span>
              <strong style={{ color: "var(--text-100)" }}>
                {new Date(report.generated_at).toLocaleString()}
              </strong>
            </div>
          </div>

          {/* Report Container */}
          <div className="card" style={{ padding: "32px 36px", position: "relative" }}>
            <div className="report-prose">
              {renderMarkdown(report.report)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
