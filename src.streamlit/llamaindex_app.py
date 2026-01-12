from __future__ import annotations

import asyncio

import streamlit as st

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

from common.blog import format_blog_entry
from common.images import batch_search_images
from common.scraper import scrape_google_news, scrape_social_media, scrape_wikipedia


def _parse_topics(text: str) -> list[str]:
    topics = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*")
        if cleaned:
            topics.append(cleaned)
    return topics


def _build_agent(tool_fn, name: str, llm: Ollama) -> ReActAgent:
    tool = FunctionTool.from_defaults(fn=tool_fn, name=name)
    return ReActAgent.from_tools([tool], llm=llm, verbose=False)


async def _run_agent(agent: ReActAgent, topic: str, prompt_prefix: str) -> list[str]:
    prompt = (
        f"{prompt_prefix} {topic}. Use the tool and return 5-8 subtopics, one per line."
    )
    response = await asyncio.to_thread(agent.chat, prompt)
    return _parse_topics(str(response))


async def run(topic: str) -> str:
    llm = Ollama(model="llama3", base_url="http://localhost:11434")

    wiki_agent = _build_agent(lambda t: scrape_wikipedia(t).items, "wikipedia", llm)
    news_agent = _build_agent(lambda t: scrape_google_news(t).items, "google_news", llm)
    social_agent = _build_agent(lambda t: scrape_social_media(t).items, "social", llm)

    wiki_topics, news_topics, social_topics = await asyncio.gather(
        _run_agent(wiki_agent, topic, "Scrape Wikipedia for"),
        _run_agent(news_agent, topic, "Scrape Google News for"),
        _run_agent(social_agent, topic, "Scrape TikTok and Instagram for"),
    )

    merged_topics = list(dict.fromkeys([*wiki_topics, *news_topics, *social_topics]))
    images = batch_search_images(merged_topics[:8], limit=2)

    summary_prompt = (
        f"Write a concise blog summary (150-200 words) about '{topic}' using these "
        f"subtopics: {', '.join(merged_topics)}."
    )
    summary = await asyncio.to_thread(llm.complete, summary_prompt)
    blog_entry = format_blog_entry(topic, str(summary), images)
    return blog_entry.markdown


def main():
    st.title("LlamaIndex Multi-Agent Blog Generator")
    topic = st.text_input("Topic to research", "Streamlit")
    run_button = st.button("Generate Blog Entry")
    if run_button and topic:
        with st.spinner("Generating blog entry..."):
            markdown = asyncio.run(run(topic))
            st.markdown(markdown)


if __name__ == "__main__":
    main()
