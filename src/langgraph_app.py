from __future__ import annotations

import argparse
import asyncio
from typing import TypedDict

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph

from common.blog import format_blog_entry
from common.images import batch_search_images
from common.scraper import clean_topics, scrape_google_news, scrape_social_media, scrape_wikipedia


class BlogState(TypedDict):
    topic: str
    sources: dict[str, list[str]]
    images: dict[str, list[str]]
    image_list: list
    summary: str


def _build_context(topic: str, sources: dict[str, list[str]], images: dict[str, list[str]]) -> str:
    lines = [f"Primary topic: {topic}", "", "Signals:"]
    for source, items in sources.items():
        lines.append(f"- {source}: {', '.join(items[:10])}")
    lines.append("")
    lines.append("Images:")
    for image_topic, urls in images.items():
        lines.append(f"- {image_topic}: {', '.join(urls)}")
    return "\n".join(lines)


async def scrape_sources(state: BlogState) -> BlogState:
    topic = state["topic"]
    loop = asyncio.get_running_loop()
    wiki_task = loop.run_in_executor(None, scrape_wikipedia, topic)
    news_task = loop.run_in_executor(None, scrape_google_news, topic)
    social_task = loop.run_in_executor(None, scrape_social_media, topic)
    wiki_result, news_result, social_result = await asyncio.gather(
        wiki_task, news_task, social_task
    )

    topics = clean_topics(wiki_result, news_result, social_result)
    images = await loop.run_in_executor(None, batch_search_images, topics[:8], 2)
    image_map = {entry.topic: entry.images for entry in images}
    sources = {
        "wikipedia": wiki_result.items,
        "google_news": news_result.items,
        "tiktok_instagram": social_result.items,
    }

    state["sources"] = sources
    state["images"] = image_map
    state["image_list"] = images
    return state


def summarize(state: BlogState) -> BlogState:
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
    chain = prompt | llm | StrOutputParser()

    context = _build_context(state["topic"], state["sources"], state["images"])
    state["summary"] = chain.invoke({"context": context})
    return state


def build_graph():
    graph = StateGraph(BlogState)
    graph.add_node("scrape", scrape_sources)
    graph.add_node("summarize", summarize)
    graph.set_entry_point("scrape")
    graph.add_edge("scrape", "summarize")
    graph.set_finish_point("summarize")
    return graph.compile()


async def run(topic: str) -> str:
    graph = build_graph()
    state: BlogState = {
        "topic": topic,
        "sources": {},
        "images": {},
        "image_list": [],
        "summary": "",
    }
    result = await graph.ainvoke(state)

    blog_entry = format_blog_entry(topic, result["summary"], result["image_list"])
    return blog_entry.markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph multi-agent demo")
    parser.add_argument("topic", help="Topic to research")
    parser.add_argument("--output", default="langgraph_blog.md")
    args = parser.parse_args()

    markdown = asyncio.run(run(args.topic))
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(f"Wrote blog entry to {args.output}")


if __name__ == "__main__":
    main()
