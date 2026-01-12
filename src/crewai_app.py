from __future__ import annotations

import argparse
import asyncio
from typing import Iterable

from crewai import Agent, Crew, Process, Task
from crewai_tools import tool

from common.blog import format_blog_entry
from common.images import batch_search_images
from common.scraper import scrape_google_news, scrape_social_media, scrape_wikipedia


@tool("wikipedia_scraper")
def wikipedia_scraper(topic: str) -> list[str]:
    """Scrape Wikipedia headings and snippets for a topic."""
    return scrape_wikipedia(topic).items


@tool("google_news_scraper")
def google_news_scraper(topic: str) -> list[str]:
    """Scrape Google News RSS titles for a topic."""
    return scrape_google_news(topic).items


@tool("social_scraper")
def social_scraper(topic: str) -> list[str]:
    """Scrape TikTok/Instagram public tag data for a topic."""
    return scrape_social_media(topic).items


def _parse_topics(text: str) -> list[str]:
    topics = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*")
        if cleaned:
            topics.append(cleaned)
    return topics


async def run(topic: str) -> str:
    wiki_agent = Agent(
        role="Wikipedia analyst",
        goal="Extract subtopics from Wikipedia",
        backstory="A researcher focusing on encyclopedic sources.",
        tools=[wikipedia_scraper],
        llm="ollama/llama3",
    )
    news_agent = Agent(
        role="News analyst",
        goal="Extract subtopics from Google News",
        backstory="A researcher focusing on news sources.",
        tools=[google_news_scraper],
        llm="ollama/llama3",
    )
    social_agent = Agent(
        role="Social analyst",
        goal="Extract subtopics from TikTok and Instagram",
        backstory="A researcher focusing on social signals.",
        tools=[social_scraper],
        llm="ollama/llama3",
    )

    wiki_task = Task(
        description=(
            f"Use the wikipedia_scraper tool on '{topic}' and list 5-8 subtopics, one per line."
        ),
        agent=wiki_agent,
    )
    news_task = Task(
        description=(
            f"Use the google_news_scraper tool on '{topic}' and list 5-8 subtopics, one per line."
        ),
        agent=news_agent,
    )
    social_task = Task(
        description=(
            f"Use the social_scraper tool on '{topic}' and list 5-8 subtopics, one per line."
        ),
        agent=social_agent,
    )

    crew = Crew(
        agents=[wiki_agent, news_agent, social_agent],
        tasks=[wiki_task, news_task, social_task],
        process=Process.parallel,
    )
    results = await asyncio.to_thread(crew.kickoff)

    topic_lists = []
    if isinstance(results, dict):
        topic_lists.extend(_parse_topics(result) for result in results.values())
    elif isinstance(results, list):
        topic_lists.extend(_parse_topics(result) for result in results)
    else:
        topic_lists.append(_parse_topics(str(results)))

    merged_topics = list(dict.fromkeys([topic for topics in topic_lists for topic in topics]))
    images = batch_search_images(merged_topics[:8], limit=2)

    summary_agent = Agent(
        role="Blog writer",
        goal="Summarize the research into a blog entry",
        backstory="A concise professional writer.",
        llm="ollama/llama3",
    )
    summary_task = Task(
        description=(
            f"Write a 150-200 word summary about '{topic}' using these subtopics: "
            f"{', '.join(merged_topics)}."
        ),
        agent=summary_agent,
    )
    summary = await asyncio.to_thread(Crew(agents=[summary_agent], tasks=[summary_task]).kickoff)
    blog_entry = format_blog_entry(topic, str(summary), images)
    return blog_entry.markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="CrewAI multi-agent demo")
    parser.add_argument("topic", help="Topic to research")
    parser.add_argument("--output", default="crewai_blog.md")
    args = parser.parse_args()

    markdown = asyncio.run(run(args.topic))
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(f"Wrote blog entry to {args.output}")


if __name__ == "__main__":
    main()
