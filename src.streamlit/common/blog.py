from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .images import TopicImages


@dataclass
class BlogEntry:
    topic: str
    markdown: str


def build_context(topic: str, sources: dict[str, list[str]], images: Iterable[TopicImages]) -> str:
    lines = [f"Primary topic: {topic}", "", "Signals:"]
    for source, items in sources.items():
        lines.append(f"- {source}: {', '.join(items[:10])}")
    lines.append("")
    lines.append("Images:")
    for entry in images:
        lines.append(f"- {entry.topic}: {', '.join(entry.images)}")
    return "\n".join(lines)


def format_blog_entry(topic: str, summary: str, images: Iterable[TopicImages]) -> BlogEntry:
    image_lines = []
    for entry in images:
        for image_url in entry.images:
            image_lines.append(f"![{entry.topic}]({image_url})")
    markdown = "\n\n".join(
        [
            f"# {topic}",
            summary.strip(),
            "## Images",
            "\n".join(image_lines),
        ]
    )
    return BlogEntry(topic=topic, markdown=markdown)
