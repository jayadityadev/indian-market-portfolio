export interface RegimeResponse {
  current_regime: string;
  regime_distribution: Record<string, number>;
  total_days: number;
  regime_source?: string;
}

export async function fetchRegime(): Promise<RegimeResponse> {
  const res = await fetch(`${process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"}/api/v1/regime`, {
    method: "GET",
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "Regime fetch failed");
  }
  return res.json();
}
