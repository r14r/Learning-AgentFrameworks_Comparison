from __future__ import annotations

import argparse
import asyncio
from typing import Iterable

from autogen import AssistantAgent

from common.blog import format_blog_entry
from common.images import batch_search_images
from common.scraper import scrape_google_news, scrape_social_media, scrape_wikipedia


def _build_llm_config():
    return {
        "config_list": [
            {
                "model": "llama3",
                "api_type": "ollama",
                "base_url": "http://localhost:11434",
            }
        ],
        "temperature": 0.2,
    }


def _parse_topics(text: str) -> list[str]:
    topics = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*")
        if cleaned:
            topics.append(cleaned)
    return topics


async def _run_agent(agent: AssistantAgent, topic: str, source_items: Iterable[str]) -> list[str]:
    content = "\n".join(source_items)
    message = (
        f"Extract 5-8 distinct subtopics from the following items about '{topic}'. "
        "Return one per line.\n\n"
        f"Items:\n{content}\n\nSubtopics:"
    )
    response = await asyncio.to_thread(agent.generate_reply, messages=[{"content": message, "role": "user"}])
    return _parse_topics(response)


async def run(topic: str) -> str:
    wiki_result = await asyncio.to_thread(scrape_wikipedia, topic)
    news_result = await asyncio.to_thread(scrape_google_news, topic)
    social_result = await asyncio.to_thread(scrape_social_media, topic)

    llm_config = _build_llm_config()
    wiki_agent = AssistantAgent("WikipediaAgent", llm_config=llm_config)
    news_agent = AssistantAgent("NewsAgent", llm_config=llm_config)
    social_agent = AssistantAgent("SocialAgent", llm_config=llm_config)

    wiki_topics, news_topics, social_topics = await asyncio.gather(
        _run_agent(wiki_agent, topic, wiki_result.items),
        _run_agent(news_agent, topic, news_result.items),
        _run_agent(social_agent, topic, social_result.items),
    )

    merged_topics = list(dict.fromkeys([*wiki_topics, *news_topics, *social_topics]))
    images = batch_search_images(merged_topics[:8], limit=2)

    summary_agent = AssistantAgent("SummaryAgent", llm_config=llm_config)
    summary_prompt = (
        f"Write a concise blog summary (150-200 words) about '{topic}' using these "
        f"subtopics: {', '.join(merged_topics)}. Provide a professional tone."
    )
    summary = await asyncio.to_thread(
        summary_agent.generate_reply, messages=[{"content": summary_prompt, "role": "user"}]
    )
    blog_entry = format_blog_entry(topic, summary, images)
    return blog_entry.markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoGen multi-agent demo")
    parser.add_argument("topic", help="Topic to research")
    parser.add_argument("--output", default="autogen_blog.md")
    args = parser.parse_args()

    markdown = asyncio.run(run(args.topic))
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(f"Wrote blog entry to {args.output}")


if __name__ == "__main__":
    main()
