from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import quote

import feedparser
import httpx
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (compatible; LearningAgentFrameworks/1.0)"


@dataclass
class SourceResult:
    source: str
    items: list[str]
    notes: str | None = None


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned


def scrape_wikipedia(topic: str) -> SourceResult:
    url = f"https://en.wikipedia.org/wiki/{quote(topic.replace(' ', '_'))}"
    response = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    headings = [
        _normalize_text(tag.get_text())
        for tag in soup.select("h2 span.mw-headline")
        if tag.get_text(strip=True)
    ]
    paragraphs = [
        _normalize_text(p.get_text())
        for p in soup.select("p")
        if p.get_text(strip=True)
    ]
    snippets = paragraphs[:3]
    items = headings[:10] + snippets
    return SourceResult(source="wikipedia", items=[item for item in items if item])


def scrape_google_news(topic: str) -> SourceResult:
    feed_url = f"https://news.google.com/rss/search?q={quote(topic)}"
    feed = feedparser.parse(feed_url)
    titles = [_normalize_text(entry.title) for entry in feed.entries[:15]]
    return SourceResult(source="google_news", items=[title for title in titles if title])


def scrape_social_media(topic: str) -> SourceResult:
    """Best-effort scrape for TikTok/Instagram public tags.

    Both providers rate-limit and often require authentication, so this returns
    a note with any parsed hashtag tokens when available.
    """
    tag = re.sub(r"\s+", "", topic)
    results: list[str] = []
    notes: list[str] = []
    for platform, url in {
        "tiktok": f"https://www.tiktok.com/tag/{quote(tag)}",
        "instagram": f"https://www.instagram.com/explore/tags/{quote(tag)}/",
    }.items():
        try:
            response = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
            for script in scripts:
                try:
                    payload = json.loads(script.string or "")
                except json.JSONDecodeError:
                    continue
                keywords = payload.get("keywords")
                if keywords:
                    results.extend([_normalize_text(word) for word in keywords.split(",")])
            if not scripts:
                notes.append(f"{platform} returned limited metadata without login.")
        except httpx.HTTPError as exc:
            notes.append(f"{platform} request failed: {exc}")
    return SourceResult(
        source="tiktok_instagram",
        items=[item for item in results if item],
        notes=" ".join(notes) if notes else None,
    )


def clean_topics(*results: SourceResult) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for result in results:
        for item in result.items:
            normalized = _normalize_text(item)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized)
    return cleaned


def truncate_topics(topics: Iterable[str], limit: int = 12) -> list[str]:
    return list(topics)[:limit]
