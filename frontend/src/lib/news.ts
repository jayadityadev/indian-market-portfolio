export interface NewsArticle {
  title: string;
  url: string;
  publisher: string;
  published_at: string | null;
  snippet: string | null;
  source: string;
}

export interface NewsSummary {
  text: string;
  provider_used: string;
  generated_at: string;
  source_urls: string[];
}

export interface NewsResponse {
  query: string;
  fetched_at: string | null;
  articles: NewsArticle[];
  status: "live" | "cached" | "unavailable";
  stale: boolean;
  errors: string[];
  summary: NewsSummary | null;
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

let newsClientCache: { data: NewsResponse; timestamp: number } | null = null;
const CACHE_TTL_MS = 5 * 60 * 1000; // 5-minute cache TTL

export async function fetchNews(limit = 12, forceRefresh = false): Promise<NewsResponse> {
  const now = Date.now();
  if (!forceRefresh && newsClientCache && now - newsClientCache.timestamp < CACHE_TTL_MS) {
    return newsClientCache.data;
  }

  const response = await fetch(`${API_BASE}/api/v1/news?limit=${limit}&summarize=true`, {
    cache: "no-store",
  });

  if (!response.ok) {
    if (newsClientCache) {
      return newsClientCache.data;
    }
    throw new Error("News unavailable");
  }

  const data: NewsResponse = await response.json();
  newsClientCache = { data, timestamp: now };
  return data;
}
