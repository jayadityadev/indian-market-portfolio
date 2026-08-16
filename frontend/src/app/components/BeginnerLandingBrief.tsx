"use client";

import { useEffect, useState } from "react";
import { fetchRegime, RegimeResponse } from "@/lib/regime";
import { Sparkles, Shield, Compass, ArrowRight } from "lucide-react";

export function BeginnerLandingBrief() {
  const [regimeData, setRegimeData] = useState<RegimeResponse | null>(null);

  useEffect(() => {
    fetchRegime()
      .then((data) => setRegimeData(data))
      .catch(() => {
        // Fallback default
      });
  }, []);

  const regime = regimeData?.current_regime || "Sideways";

  const config = {
    Bull: {
      weather: "Bull Market (Rising Trend 📈)",
      color: "var(--bull)",
      bg: "var(--bull-bg)",
      action: "Stay invested in high-momentum stocks or Buy & Hold index funds.",
      riskStance: "Normal (Full Exposure 100%)",
      topTip: "Ride the upward trend, but set trailing stop losses.",
    },
    Bear: {
      weather: "Bear Market (Declining / Stormy 📉)",
      color: "var(--bear)",
      bg: "var(--bear-bg)",
      action: "Protect capital. Switch to cash hedges or absolute momentum.",
      riskStance: "Defensive (Hold 50–100% in Cash)",
      topTip: "Do not try to catch falling stocks without systematic stop-losses.",
    },
    Sideways: {
      weather: "Sideways Market (Range-Bound / Flat ↔️)",
      color: "var(--sideways)",
      bg: "var(--sideways-bg)",
      action: "Use range-trading strategies (Bollinger Bands, RSI) that buy low and sell high.",
      riskStance: "Moderate (Keep 25% Cash Buffer)",
      topTip: "Avoid buying breakout rallies—they often reverse back into the range.",
    },
  }[regime as "Bull" | "Bear" | "Sideways"] || {
    weather: "Sideways Market (Range-Bound ↔️)",
    color: "var(--sideways)",
    bg: "var(--sideways-bg)",
    action: "Use range-trading strategies that buy low and sell high.",
    riskStance: "Moderate (Keep 25% Cash Buffer)",
    topTip: "Avoid buying breakout rallies.",
  };

  return (
    <div
      className="card"
      style={{
        background: `linear-gradient(135deg, ${config.bg}, var(--bg-200))`,
        border: `1px solid ${config.color}`,
        padding: 24,
        marginBottom: 20,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 10, marginBottom: 12 }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, fontWeight: 700, color: config.color }}>
          <Sparkles size={14} /> Today&apos;s 60-Second Market Briefing for Beginners
        </div>
        <span style={{ fontSize: 11, color: "var(--text-200)" }}>NIFTY 50 Live Context</span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 16 }}>
        <div>
          <div style={{ fontSize: 11, color: "var(--text-200)", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: 2 }}>
            Current Market Condition
          </div>
          <div style={{ fontSize: 20, fontWeight: 800, color: "var(--text-100)", marginBottom: 8 }}>
            {config.weather}
          </div>
          <p style={{ fontSize: 13, color: "var(--text-200)", lineHeight: 1.5, margin: 0 }}>
            {config.action}
          </p>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 8, background: "var(--bg-200)", padding: 14, borderRadius: 14, border: "1px solid var(--card-border)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, fontWeight: 700, color: "var(--text-100)" }}>
            <Shield size={14} style={{ color: config.color }} /> Safe Position Sizing: <span style={{ color: config.color }}>{config.riskStance}</span>
          </div>
          <div style={{ display: "flex", alignItems: "flex-start", gap: 6, fontSize: 11, color: "var(--text-200)", lineHeight: 1.4 }}>
            <Compass size={13} style={{ flexShrink: 0, marginTop: 2, color: "var(--accent-100)" }} />
            <span><strong>Rule of Thumb:</strong> {config.topTip}</span>
          </div>
        </div>
      </div>

      <div style={{ borderTop: "1px solid var(--card-border)", marginTop: 16, paddingTop: 12, display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8, fontSize: 12, color: "var(--text-200)" }}>
        <span>Ready to test your strategy? Select your date range on the left sidebar.</span>
        <span style={{ color: "var(--accent-100)", fontWeight: 700, display: "inline-flex", alignItems: "center", gap: 4 }}>
          Click &ldquo;Analyse for me&rdquo; <ArrowRight size={13} />
        </span>
      </div>
    </div>
  );
}
