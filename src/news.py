"""Free live-news retrieval with provenance-preserving cache fallback."""
from __future__ import annotations

import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import httpx

from llm.client import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path(__file__).parent.parent / "data" / "news_cache.json"
DEFAULT_RSS_URLS = (
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
)
DEFAULT_GDELT_QUERY = "NIFTY OR RBI OR rupee OR India stock market OR crude oil"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _parse_rss(content: bytes, source_url: str) -> list[dict[str, Any]]:
    root = ET.fromstring(content)
    rows: list[dict[str, Any]] = []
    for item in root.findall(".//item"):
        title = _clean_text(item.findtext("title"))
        url = _clean_text(item.findtext("link"))
        if not title or not url:
            continue
        rows.append(
            {
                "title": title,
                "url": url,
                "publisher": _clean_text(item.findtext("source")) or source_url,
                "published_at": _clean_text(item.findtext("pubDate")) or None,
                "snippet": _clean_text(item.findtext("description")) or None,
                "source": "rss",
            }
        )
    return rows


def _fetch_rss(url: str, limit: int, client: httpx.Client) -> list[dict[str, Any]]:
    response = client.get(url)
    response.raise_for_status()
    return _parse_rss(response.content, url)[:limit]


def _fetch_gdelt(query: str, limit: int, client: httpx.Client) -> list[dict[str, Any]]:
    endpoint = (
        "https://api.gdeltproject.org/api/v2/doc/doc?query="
        f"{quote_plus(query)}&mode=artlist&maxrecords={limit}&format=json&sort=HybridRel"
    )
    response = client.get(endpoint)
    response.raise_for_status()
    payload = response.json()
    rows: list[dict[str, Any]] = []
    for item in payload.get("articles", []):
        title = _clean_text(item.get("title"))
        url = _clean_text(item.get("url"))
        if not title or not url:
            continue
        rows.append(
            {
                "title": title,
                "url": url,
                "publisher": _clean_text(item.get("domain")) or "GDELT source",
                "published_at": _clean_text(item.get("seendate")) or None,
                "snippet": None,
                "source": "gdelt",
            }
        )
    return rows


def _deduplicate(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = row["url"].lower().rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
        if len(result) >= limit:
            break
    return result


def _load_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Unable to read news cache: %s", exc)
        return None


def fetch_news(
    query: str = DEFAULT_GDELT_QUERY,
    limit: int = 20,
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> dict[str, Any]:
    """Fetch live India/global market news; return timestamped cache on failure."""
    rss_urls = tuple(
        value.strip()
        for value in os.getenv("NEWS_RSS_URLS", ",".join(DEFAULT_RSS_URLS)).split(",")
        if value.strip()
    )
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        with httpx.Client(timeout=12.0, follow_redirects=True) as client:
            try:
                rows.extend(_fetch_gdelt(query, limit, client))
            except Exception as exc:
                errors.append(f"gdelt: {exc}")
            for url in rss_urls:
                try:
                    rows.extend(_fetch_rss(url, limit, client))
                except Exception as exc:
                    errors.append(f"rss:{url}: {exc}")
    except Exception as exc:
        errors.append(f"http client: {exc}")

    articles = _deduplicate(rows, limit)
    if articles:
        payload = {
            "query": query,
            "fetched_at": _now(),
            "articles": articles,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return {**payload, "status": "live", "errors": errors, "stale": False}

    cached = _load_cache(cache_path)
    if cached and cached.get("articles"):
        return {
            **cached,
            "status": "cached",
            "errors": errors,
            "stale": True,
        }
    return {
        "query": query,
        "fetched_at": None,
        "articles": [],
        "status": "unavailable",
        "errors": errors,
        "stale": True,
    }


def summarize_news(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Summarize fetched articles only through a real configured LLM provider."""
    articles = payload.get("articles", [])
    if not articles:
        return None
    article_text = "\n".join(
        f"- {article.get('title')} | {article.get('publisher')} | {article.get('url')}"
        for article in articles[:20]
    )
    client = LLMClient()
    result = client.generate(
        system_prompt=(
            "You summarize retrieved financial news. Use only supplied articles. "
            "Do not invent facts. State uncertainty. Return: What happened, likely market relevance, "
            "and what remains unknown. Keep under 180 words."
        ),
        user_prompt=f"Retrieved articles:\n{article_text}",
        max_tokens=400,
        temperature=0.1,
    )
    if result["provider_used"] == "mock":
        return None
    return {
        "text": result["content"],
        "provider_used": result["provider_used"],
        "generated_at": _now(),
        "source_urls": [article.get("url") for article in articles if article.get("url")],
    }
