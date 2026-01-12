from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

from .blog import BlogEntry, build_context, format_blog_entry
from .images import TopicImages, batch_search_images
from .scraper import (
    SourceResult,
    clean_topics,
    scrape_google_news,
    scrape_social_media,
    scrape_wikipedia,
    truncate_topics,
)


@dataclass
class PipelineOutput:
    topic: str
    sources: dict[str, list[str]]
    topics: list[str]
    images: list[TopicImages]
    context: str


def _run_sources(topic: str) -> list[SourceResult]:
    return [
        scrape_wikipedia(topic),
        scrape_google_news(topic),
        scrape_social_media(topic),
    ]


def run_pipeline(topic: str, limit_topics: int = 12) -> PipelineOutput:
    results = _run_sources(topic)
    sources = {result.source: result.items for result in results}
    topics = truncate_topics(clean_topics(*results), limit=limit_topics)
    images = batch_search_images(topics, limit=2)
    context = build_context(topic, sources, images)
    return PipelineOutput(
        topic=topic,
        sources=sources,
        topics=topics,
        images=images,
        context=context,
    )


async def run_pipeline_async(topic: str, limit_topics: int = 12) -> PipelineOutput:
    loop = asyncio.get_running_loop()

    results = await loop.run_in_executor(None, _run_sources, topic)
    sources = {result.source: result.items for result in results}
    topics = truncate_topics(clean_topics(*results), limit=limit_topics)
    images = await loop.run_in_executor(None, batch_search_images, topics, 2)
    context = build_context(topic, sources, images)
    return PipelineOutput(
        topic=topic,
        sources=sources,
        topics=topics,
        images=images,
        context=context,
    )


def build_blog_entry(
    topic: str,
    context: str,
    images: list[TopicImages],
    summarizer: Callable[[str], str],
) -> BlogEntry:
    prompt = (
        "You are a research blog assistant. Write a concise summary (150-200 words) "
        "based on the provided context. Include relevant keywords and keep the tone "
        "professional.\n\nContext:\n"
        f"{context}\n\nSummary:"
    )
    summary = summarizer(prompt)
    return format_blog_entry(topic, summary, images)
