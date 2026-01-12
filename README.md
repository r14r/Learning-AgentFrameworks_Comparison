# Learning Agent Frameworks

This repository is a learning playground that compares multi-agent workflows across popular
agent frameworks. Each framework script runs three agents in parallel to scrape and parse
signals from Wikipedia, Google News, and TikTok/Instagram, then merges the topics, fetches
images, and generates a short blog entry using a local Ollama LLM.

## Features

- Parallel source agents (Wikipedia, Google News, TikTok/Instagram)
- Topic de-duplication and enrichment
- Two images per topic via the Wikipedia API with a fallback placeholder
- Blog-style summary generated with Ollama

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with the `llama3` model pulled
  (`ollama pull llama3`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the demos

Each script accepts a topic and writes a markdown blog entry to disk.

```bash
python src/langchain_app.py "Edge AI" --output langchain_blog.md
python src/langgraph_app.py "Edge AI" --output langgraph_blog.md
python src/swarms_app.py "Edge AI" --output swarms_blog.md
python src/autogen_app.py "Edge AI" --output autogen_blog.md
python src/llamaindex_app.py "Edge AI" --output llamaindex_blog.md
python src/crewai_app.py "Edge AI" --output crewai_blog.md
```

## Project layout

- `src/common/` – shared scraping, image lookup, and blog formatting utilities
- `src/langchain_app.py` – LangChain multi-agent workflow
- `src/langgraph_app.py` – LangGraph multi-agent workflow
- `src/swarms_app.py` – Swarms multi-agent workflow
- `src/autogen_app.py` – Microsoft AutoGen multi-agent workflow
- `src/llamaindex_app.py` – LlamaIndex multi-agent workflow
- `src/crewai_app.py` – CrewAI multi-agent workflow

## Notes

- TikTok/Instagram scraping can be limited without authentication. The scripts do a
  best-effort public scrape and fall back gracefully when content is unavailable.
- Google News uses the public RSS feed, which is more reliable than scraping the HTML UI.
