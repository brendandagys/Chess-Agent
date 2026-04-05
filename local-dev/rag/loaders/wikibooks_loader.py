import requests # type: ignore
from typing import List, Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup, Tag # type: ignore

from langchain_core.documents import Document # type: ignore


@dataclass
class Section:
    title: str
    level: int
    parent: Optional[str]
    content: List[str]


class WikibooksLoader:
    def __init__(self, url: str):
        self.url = url

    def load(self) -> List[Document]:
        html = self._fetch_html()
        soup = BeautifulSoup(html, "html.parser")

        title = self._get_title(soup)
        sections = self._parse_sections(soup)

        docs = self._sections_to_documents(sections, title)
        return docs

    def _fetch_html(self) -> str:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; chess-agent/1.0; +https://github.com/local)"
        }
        resp = requests.get(self.url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.text

    def _get_title(self, soup: BeautifulSoup) -> str:
        title_tag = soup.find("h1", {"id": "firstHeading"})
        return title_tag.get_text(strip=True) if title_tag else "Unknown"

    def _parse_sections(self, soup: BeautifulSoup) -> List[Section]:
        content = soup.find("div", {"id": "mw-content-text"})
        if not content:
            return []

        sections: List[Section] = []
        stack: List[Section] = []

        for el in content.descendants:
            if not isinstance(el, Tag):
                continue

            # Headings
            if el.name in ["h2", "h3", "h4"]:
                title = el.get_text(strip=True)
                level = int(el.name[1])

                # Maintain hierarchy stack
                while stack and stack[-1].level >= level:
                    stack.pop()

                parent = stack[-1].title if stack else None

                section = Section(
                    title=title,
                    level=level,
                    parent=parent,
                    content=[]
                )

                stack.append(section)
                sections.append(section)

            # Paragraphs
            elif el.name == "p" and stack:
                text = el.get_text(strip=True)
                if text:
                    stack[-1].content.append(text)

        return sections

    def _sections_to_documents(
        self,
        sections: List[Section],
        page_title: str
    ) -> List[Document]:

        docs = []

        for sec in sections:
            if not sec.content:
                continue

            text = "\n\n".join(sec.content)

            # Build hierarchical path
            path = sec.title
            if sec.parent:
                path = f"{sec.parent} > {sec.title}"

            doc = Document(
                page_content=text,
                metadata={
                    "source": self.url,
                    "book_title": page_title,
                    "section": sec.title,
                    "parent_section": sec.parent,
                    "section_path": path,
                    "level": sec.level,
                },
            )
            docs.append(doc)

        return docs
