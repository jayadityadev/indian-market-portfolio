export interface BenchmarkMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  training_time_sec?: number;
}

export interface BenchmarkResponse {
  xgboost_metrics: BenchmarkMetrics;
  lstm_metrics: BenchmarkMetrics;
  data_provenance?: {
    ticker: string;
    start: string;
    end: string;
    purpose: string;
  };
}

export async function fetchBenchmark(): Promise<BenchmarkResponse> {
  const res = await fetch(`${process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"}/api/v1/benchmark`, {
    method: "GET",
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "Benchmark fetch failed");
  }
  return res.json();
}
