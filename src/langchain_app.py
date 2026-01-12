from __future__ import annotations

import argparse
import asyncio

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from common.blog import format_blog_entry
from common.images import batch_search_images
from common.scraper import clean_topics, scrape_google_news, scrape_social_media, scrape_wikipedia


async def _run_source_agents(topic: str):
    wiki_agent = RunnableLambda(lambda input_topic: scrape_wikipedia(input_topic))
    news_agent = RunnableLambda(lambda input_topic: scrape_google_news(input_topic))
    social_agent = RunnableLambda(lambda input_topic: scrape_social_media(input_topic))

    return await asyncio.gather(
        wiki_agent.ainvoke(topic),
        news_agent.ainvoke(topic),
        social_agent.ainvoke(topic),
    )


def _build_summarizer():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a research blog assistant. Write a concise summary (150-200 words) "
                "about the topic using the provided signals and image references.",
            ),
            ("human", "{context}"),
        ]
    )
    llm = ChatOllama(model="llama3", base_url="http://localhost:11434")
    return prompt | llm | StrOutputParser()


def build_context(topic: str, sources: dict[str, list[str]], images: dict[str, list[str]]) -> str:
    lines = [f"Primary topic: {topic}", "", "Signals:"]
    for source, items in sources.items():
        lines.append(f"- {source}: {', '.join(items[:10])}")
    lines.append("")
    lines.append("Images:")
    for image_topic, urls in images.items():
        lines.append(f"- {image_topic}: {', '.join(urls)}")
    return "\n".join(lines)


async def run(topic: str) -> str:
    wiki_result, news_result, social_result = await _run_source_agents(topic)

    topics = clean_topics(wiki_result, news_result, social_result)
    images = batch_search_images(topics[:8], limit=2)
    image_map = {entry.topic: entry.images for entry in images}

    sources = {
        "wikipedia": wiki_result.items,
        "google_news": news_result.items,
        "tiktok_instagram": social_result.items,
    }

    context = build_context(topic, sources, image_map)
    summarizer = _build_summarizer()
    summary = summarizer.invoke({"context": context})
    blog_entry = format_blog_entry(topic, summary, images)
    return blog_entry.markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain multi-agent demo")
    parser.add_argument("topic", help="Topic to research")
    parser.add_argument("--output", default="langchain_blog.md")
    args = parser.parse_args()

    markdown = asyncio.run(run(args.topic))
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(f"Wrote blog entry to {args.output}")


if __name__ == "__main__":
    main()
