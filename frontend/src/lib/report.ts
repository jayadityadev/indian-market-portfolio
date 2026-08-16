import type { AnalyzeResponse } from "./api";

/** Richer response from POST /api/v1/llm-report */
export interface ReportResponse {
  /** Markdown-formatted analyst commentary */
  report: string;
  /** Which provider actually generated the report (gemini | groq | nvidia | openrouter | mock) */
  provider_used: string;
  /** Exact model identifier used */
  model_used: string;
  /** ISO-8601 UTC timestamp */
  generated_at: string;
  /** Cascade log — which providers were tried / skipped */
  fallback_history: string[];
}

const API_V1 = `${process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"}/api/v1`;

/**
 * Generate an AI Market Analyst report from an AnalyzeResponse payload.
 *
 * @param payload  - Full analysis result from /api/v1/analyze
 * @param provider - Optional provider pin: "gemini" | "groq" | "nvidia" | "openrouter" | "mock"
 *                   Supports model suffix, e.g. "nvidia:meta/llama-3.3-70b-instruct"
 */
export async function fetchReport(
  payload: AnalyzeResponse,
  provider?: string
): Promise<ReportResponse> {
  const url = provider
    ? `${API_V1}/llm-report?provider=${encodeURIComponent(provider)}`
    : `${API_V1}/llm-report`;

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "LLM report fetch failed");
  }
  return res.json();
}
