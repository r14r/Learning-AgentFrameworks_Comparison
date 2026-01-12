from __future__ import annotations

import argparse
import asyncio
from typing import Iterable

from swarms import Agent

from common.blog import format_blog_entry
from common.images import batch_search_images
from common.scraper import scrape_google_news, scrape_social_media, scrape_wikipedia


def _parse_topics(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*")
        if cleaned:
            lines.append(cleaned)
    return lines


def _build_agent(name: str) -> Agent:
    return Agent(
        agent_name=name,
        model_name="llama3",
        llm="ollama",
        system_prompt=(
            "Extract 5-8 distinct subtopics from the provided source content. "
            "Return each subtopic on its own line."
        ),
    )


async def _run_source_agent(agent: Agent, topic: str, source_items: Iterable[str]) -> list[str]:
    content = "\n".join(source_items)
    prompt = f"Topic: {topic}\nSource items:\n{content}\n\nSubtopics:"
    response = await asyncio.to_thread(agent.run, prompt)
    return _parse_topics(response)


async def run(topic: str) -> str:
    wiki_result = await asyncio.to_thread(scrape_wikipedia, topic)
    news_result = await asyncio.to_thread(scrape_google_news, topic)
    social_result = await asyncio.to_thread(scrape_social_media, topic)

    wiki_agent = _build_agent("WikipediaAgent")
    news_agent = _build_agent("NewsAgent")
    social_agent = _build_agent("SocialAgent")

    wiki_topics, news_topics, social_topics = await asyncio.gather(
        _run_source_agent(wiki_agent, topic, wiki_result.items),
        _run_source_agent(news_agent, topic, news_result.items),
        _run_source_agent(social_agent, topic, social_result.items),
    )

    merged_topics = list(dict.fromkeys([*wiki_topics, *news_topics, *social_topics]))
    images = batch_search_images(merged_topics[:8], limit=2)

    summary_agent = Agent(
        agent_name="SummaryAgent",
        model_name="llama3",
        llm="ollama",
        system_prompt=(
            "You are a research blog assistant. Write a concise summary (150-200 words) "
            "about the topic and the subtopics."),
    )
    prompt = (
        f"Topic: {topic}\nSubtopics: {', '.join(merged_topics)}\n"
        f"Images: {', '.join(', '.join(entry.images) for entry in images)}\n\nSummary:"
    )
    summary = await asyncio.to_thread(summary_agent.run, prompt)
    blog_entry = format_blog_entry(topic, summary, images)
    return blog_entry.markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Swarms multi-agent demo")
    parser.add_argument("topic", help="Topic to research")
    parser.add_argument("--output", default="swarms_blog.md")
    args = parser.parse_args()

    markdown = asyncio.run(run(args.topic))
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(f"Wrote blog entry to {args.output}")


if __name__ == "__main__":
    main()
