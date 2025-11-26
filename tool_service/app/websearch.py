import asyncio
import logging
import os
from typing import Any

import httpx
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("LLM_GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("LLM_GOOGLE_CSE_ID") or os.getenv("GOOGLE_CSE_ID")
DEFAULT_MAX_RESULTS = 5
GOOGLE_SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"


class WebSearchError(RuntimeError):
    """Raised when the web search providers fail."""


async def perform_web_search(
    query: str,
    *,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> dict[str, Any]:
    """Run a web search using Google Custom Search or DuckDuckGo fallback."""

    if not isinstance(query, str) or not query.strip():
        raise WebSearchError("Search query cannot be empty")

    limit = _normalize_limit(max_results)
    normalized_query = query.strip()

    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        try:
            results = await _google_custom_search(normalized_query, limit)
            if results:
                return {
                    "provider": "google-custom-search",
                    "query": normalized_query,
                    "results": results,
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Google Custom Search failed, falling back to DuckDuckGo: %s",
                exc,
            )

    results = await _duckduckgo_search(normalized_query, limit)
    if results:
        return {
            "provider": "duckduckgo-search",
            "query": normalized_query,
            "results": results,
        }

    raise WebSearchError("No search provider returned results")


def _normalize_limit(raw_limit: Any) -> int:
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return DEFAULT_MAX_RESULTS
    return max(1, min(limit, 10))


async def _google_custom_search(query: str, limit: int) -> list[dict[str, Any]]:
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": limit,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(GOOGLE_SEARCH_ENDPOINT, params=params)
        response.raise_for_status()
        data = response.json()

    items = data.get("items") or []
    return [_normalize_google_item(item) for item in items][:limit]


def _normalize_google_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": item.get("title"),
        "url": item.get("link"),
        "snippet": item.get("snippet") or item.get("htmlSnippet"),
        "source": item.get("displayLink"),
    }


async def _duckduckgo_search(query: str, limit: int) -> list[dict[str, Any]]:
    loop = asyncio.get_running_loop()

    def _run_search() -> list[dict[str, Any]]:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=limit))
        normalized = []
        for result in results:
            normalized.append(
                {
                    "title": result.get("title"),
                    "url": result.get("href"),
                    "snippet": result.get("body"),
                    "source": result.get("source"),
                }
            )
        return normalized

    return await loop.run_in_executor(None, _run_search)
