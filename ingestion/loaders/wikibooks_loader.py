from dataclasses import dataclass, field
from typing import List, Optional

from bs4 import BeautifulSoup, Tag  # type: ignore


@dataclass
class _Section:
    title: str
    level: int
    parent: Optional[str]
    content: List[str] = field(default_factory=list)


class WikibooksLoader:
    """Parse a Wikibooks HTML page into per-section document dicts.

    Accepts raw HTML text (already read from S3) so no network I/O is done
    inside the loader.  Mirrors the interface of the other ingestion loaders:
    ``__init__(text, source)`` → ``load() -> list[dict]``.
    """

    def __init__(self, text: str, source: str = ""):
        self.text = text
        self.source = source

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> List[dict]:
        soup = BeautifulSoup(self.text, "html.parser")
        title = self._get_title(soup)
        sections = self._parse_sections(soup)
        return self._sections_to_dicts(sections, title)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_title(self, soup: BeautifulSoup) -> str:
        tag = soup.find("h1", {"id": "firstHeading"})
        return tag.get_text(strip=True) if tag else "Unknown"

    def _parse_sections(self, soup: BeautifulSoup) -> List[_Section]:
        content = soup.find("div", {"id": "mw-content-text"})
        if not content:
            return []

        sections: List[_Section] = []
        stack: List[_Section] = []

        for el in content.descendants:
            if not isinstance(el, Tag):
                continue

            if el.name in ("h2", "h3", "h4"):
                heading_text = el.get_text(strip=True)
                level = int(el.name[1])

                while stack and stack[-1].level >= level:
                    stack.pop()

                parent = stack[-1].title if stack else None
                section = _Section(title=heading_text, level=level, parent=parent)
                stack.append(section)
                sections.append(section)

            elif el.name == "p" and stack:
                text = el.get_text(strip=True)
                if text:
                    stack[-1].content.append(text)

        return sections

    def _sections_to_dicts(self, sections: List[_Section], page_title: str) -> List[dict]:
        docs = []
        for sec in sections:
            if not sec.content:
                continue

            section_path = f"{sec.parent} > {sec.title}" if sec.parent else sec.title

            docs.append({
                "page_content": "\n\n".join(sec.content),
                "metadata": {
                    "source": self.source,
                    "book_title": page_title,
                    "section": sec.title,
                    "parent_section": sec.parent,
                    "section_path": section_path,
                    "level": sec.level,
                },
            })
        return docs
