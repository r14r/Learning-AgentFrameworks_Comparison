from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from urllib.parse import quote

import httpx


@dataclass
class TopicImages:
    topic: str
    images: list[str]


def search_images(topic: str, limit: int = 2) -> TopicImages:
    search_url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&format=json&prop=pageimages&generator=search"
        f"&gsrsearch={quote(topic)}&gsrlimit={limit}&piprop=thumbnail&pithumbsize=640"
    )
    response = httpx.get(search_url, timeout=30)
    response.raise_for_status()
    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    images: list[str] = []
    for page in pages.values():
        thumbnail = page.get("thumbnail", {})
        if "source" in thumbnail:
            images.append(thumbnail["source"])
        if len(images) >= limit:
            break
    while len(images) < limit:
        images.append(f"https://picsum.photos/seed/{quote(topic)}-{len(images)}/640/360")
    return TopicImages(topic=topic, images=images)


def batch_search_images(topics: Iterable[str], limit: int = 2) -> list[TopicImages]:
    return [search_images(topic, limit=limit) for topic in topics]
