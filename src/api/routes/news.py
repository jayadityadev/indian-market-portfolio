"""Live market news route with cache fallback."""
from __future__ import annotations

from fastapi import APIRouter, Query

from news import fetch_news, summarize_news

router = APIRouter()


@router.get("/news", summary="Fetch cited India and global market news", tags=["News"])
def get_news(
    query: str = Query(default="NIFTY OR RBI OR rupee OR India stock market OR crude oil"),
    limit: int = Query(default=12, ge=1, le=50),
    summarize: bool = Query(default=True),
):
    payload = fetch_news(query=query, limit=limit)
    payload["summary"] = summarize_news(payload) if summarize and payload["articles"] else None
    return payload
