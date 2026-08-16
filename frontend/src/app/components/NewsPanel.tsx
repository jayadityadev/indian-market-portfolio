"use client";

import { useEffect, useState } from "react";
import { ExternalLink, Loader2, Newspaper, Sparkles, RefreshCw, Radio, Globe } from "lucide-react";
import { fetchNews, NewsResponse } from "@/lib/news";

function formatPublisher(publisher: string): string {
  if (!publisher) return "Financial News";
  const p = publisher.toLowerCase();
  if (p.includes("economictimes") || p.includes("indiatimes")) return "The Economic Times";
  if (p.includes("bbc")) return "BBC Business";
  if (p.includes("reuters")) return "Reuters";
  if (p.includes("bloomberg")) return "Bloomberg";
  if (p.includes("moneycontrol")) return "Moneycontrol";
  if (p.includes("livemint") || p.includes("mint")) return "Mint";
  if (p.includes("business-standard")) return "Business Standard";
  if (p.includes("financialexpress")) return "Financial Express";
  if (p.includes("cnbc")) return "CNBC TV18";
  if (p.startsWith("http://") || p.startsWith("https://")) {
    try {
      return new URL(publisher).hostname.replace(/^www\./, "");
    } catch {
      return "Market Wire";
    }
  }
  return publisher;
}

export function NewsPanel() {
  const [data, setData] = useState<NewsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadNews = (forceRefresh = false) => {
    setLoading(true);
    setError("");
    fetchNews(12, forceRefresh)
      .then((result) => {
        setData(result);
        setLoading(false);
      })
      .catch((reason: Error) => {
        setError(reason.message || "Failed to load market news");
        setLoading(false);
      });
  };

  useEffect(() => {
    const timer = window.setTimeout(() => {
      loadNews();
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  return (
    <div className="card" style={{ marginBottom: 20, padding: 22 }}>
      {/* Card Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14, flexWrap: "wrap", gap: 10 }}>
        <div className="card-title" style={{ margin: 0, fontSize: 15 }}>
          <Newspaper size={17} /> Live Market News & Macro Intelligence
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {data && (
            <span
              style={{
                fontSize: 10,
                fontWeight: 700,
                padding: "3px 10px",
                borderRadius: 999,
                background: data.status === "live" ? "var(--bull-bg)" : "var(--sideways-bg)",
                color: data.status === "live" ? "var(--bull)" : "var(--sideways)",
                border: `1px solid ${data.status === "live" ? "rgba(16,185,129,0.25)" : "rgba(245,158,11,0.25)"}`,
                display: "inline-flex",
                alignItems: "center",
                gap: 5,
              }}
            >
              <Radio size={10} />
              {data.status === "live" ? "Live Feed (GDELT & ET/BBC)" : "Cached Feed"}
            </span>
          )}

          <button
            onClick={() => loadNews(true)}
            disabled={loading}
            className="export-btn"
            style={{ padding: "6px 12px", fontSize: 11, margin: 0 }}
            title="Force refresh live news feed"
          >
            <RefreshCw size={11} className={loading ? "spin" : ""} />
            <span>{loading ? "Fetching..." : "Refresh"}</span>
          </button>
        </div>
      </div>

      {loading && !data && (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", padding: "32px 0", color: "var(--text-200)", fontSize: 13, gap: 8 }}>
          <Loader2 size={16} className="spin" style={{ color: "var(--accent-100)" }} />
          <span>Ingesting live market articles...</span>
        </div>
      )}

      {error && (
        <div style={{ padding: "12px 16px", background: "var(--bear-bg)", color: "var(--bear)", borderRadius: 12, fontSize: 13 }}>
          {error}
        </div>
      )}

      {data && (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {/* AI Macro Synthesis Banner */}
          {data.summary && (
            <div
              style={{
                background: "linear-gradient(135deg, rgba(var(--glow-rgb), 0.1), var(--bg-100))",
                border: "1px solid var(--card-border)",
                borderRadius: 16,
                padding: "16px 18px",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, fontWeight: 700, color: "var(--accent-100)", marginBottom: 6 }}>
                <Sparkles size={14} /> AI Macro Synthesis ({data.summary.provider_used.toUpperCase()})
              </div>
              <p style={{ color: "var(--text-100)", fontSize: 13, lineHeight: 1.6, margin: 0 }}>
                {data.summary.text}
              </p>
            </div>
          )}

          {/* Article Cards Grid - Balanced Responsive Grid */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 12 }}>
            {data.articles.slice(0, 6).map((article) => {
              const pubName = formatPublisher(article.publisher);
              return (
                <a
                  key={article.url}
                  href={article.url}
                  target="_blank"
                  rel="noreferrer"
                  className="metric-card"
                  style={{
                    textDecoration: "none",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "space-between",
                    padding: "14px 16px",
                    borderRadius: 16,
                    cursor: "pointer",
                    overflow: "hidden",
                    minHeight: 110,
                  }}
                  title={`Open article from ${pubName} in a new tab`}
                >
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6, gap: 8 }}>
                      <span
                        style={{
                          fontSize: 10,
                          fontWeight: 700,
                          color: "var(--accent-100)",
                          textTransform: "uppercase",
                          letterSpacing: "0.04em",
                          display: "inline-flex",
                          alignItems: "center",
                          gap: 4,
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                      >
                        <Globe size={11} style={{ flexShrink: 0 }} />
                        {pubName}
                      </span>
                      <div
                        style={{
                          padding: 4,
                          borderRadius: 6,
                          background: "var(--bg-100)",
                          color: "var(--text-200)",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          flexShrink: 0,
                        }}
                      >
                        <ExternalLink size={12} />
                      </div>
                    </div>
                    <h4
                      style={{
                        fontSize: 13,
                        fontWeight: 600,
                        color: "var(--text-100)",
                        lineHeight: 1.45,
                        margin: 0,
                        display: "-webkit-box",
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: "vertical",
                        overflow: "hidden",
                        wordBreak: "break-word",
                      }}
                    >
                      {article.title}
                    </h4>
                  </div>

                  {article.published_at && (
                    <div style={{ fontSize: 10, color: "var(--text-200)", marginTop: 8 }}>
                      {article.published_at.length > 10 ? article.published_at.slice(0, 16) : article.published_at}
                    </div>
                  )}
                </a>
              );
            })}
          </div>

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, color: "var(--text-200)", borderTop: "1px solid var(--bg-300)", paddingTop: 10 }}>
            <span>
              {data.fetched_at ? `Feed updated: ${new Date(data.fetched_at).toLocaleTimeString()}` : ""}
            </span>
            <span>{data.articles.length} verified news sources</span>
          </div>
        </div>
      )}
    </div>
  );
}
