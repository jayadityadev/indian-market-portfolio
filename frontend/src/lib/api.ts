export interface MetricsResponse {
  CAGR: number;
  Sharpe: number;
  Sortino: number;
  MaxDrawdown: number;
  Calmar: number;
  Volatility: number;
}

export interface EquityPoint {
  date: string;
  value: number;
}

export interface OHLCPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface RegimeHeatmapEntry {
  strategy: string;
  regime: string;
  CAGR: number;
  Sharpe: number;
  Sortino: number;
  MaxDrawdown: number;
  Calmar: number;
  Volatility: number;
}

export interface RegimeTimelineSegment {
  regime: string;
  start: string;
  end: string;
  days: number;
}

export interface AnalyzeResponse {
  ticker: string;
  start_date: string;
  end_date: string;
  n_trading_days: number;
  initial_investment: number;
  current_regime: string;
  recommended_strategy: string;
  recommendation_source: string;
  recommendation_reason: string;
  recommended_exposure: string;
  probabilities: Record<string, number>;
  overall_metrics: Record<string, MetricsResponse>;
  equity_curves: Record<string, EquityPoint[]>;
  ohlc_data: OHLCPoint[];
  regime_heatmap: RegimeHeatmapEntry[];
  regime_timeline: RegimeTimelineSegment[];
  risk_forecast: { worst_case_10: number; median_50: number; best_case_90: number } | null;
}

const API_BASE = "http://localhost:8000";

export async function fetchAnalysis(
  ticker: string = "^NSEI",
  startDate: string = "2015-01-01",
  endDate: string = "2024-12-31",
  strategy: string = "all",
  initialInvestment: number = 100000
): Promise<AnalyzeResponse> {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ticker,
      start_date: startDate,
      end_date: endDate,
      strategy,
      initial_investment: initialInvestment,
    }),
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "Analysis failed");
  }
  return res.json();
}
