from typing import List

from bs4 import BeautifulSoup, Tag  # type: ignore

# Tags whose entire subtree should be removed before extracting text
_NOISE_TAGS = {
    "script", "style", "noscript", "nav", "header", "footer",
    "aside", "form", "button", "iframe", "svg", "figure",
}

# Candidate selectors for the main content area, tried in order
_CONTENT_SELECTORS = [
    {"name": "main"},
    {"name": "article"},
    {"name": "div", "attrs": {"id": "main-content"}},
    {"name": "div", "attrs": {"id": "content"}},
    {"name": "div", "attrs": {"id": "post"}},
    {"name": "div", "attrs": {"class": "entry-content"}},
    {"name": "div", "attrs": {"class": "post-content"}},
    {"name": "div", "attrs": {"class": "article-body"}},
    {"name": "div", "attrs": {"class": "content"}},
    {"name": "body"},
]

_HEADING_TAGS = {"h1", "h2", "h3", "h4"}


class HTMLLoader:
    """Extract readable text from a generic chess-related HTML page.

    Tries to isolate the main content area, splits it into sections at
    heading boundaries, and returns one document dict per section (or a
    single doc if no headings are present).

    Accepts raw HTML text already read from S3 — no network I/O.
    Interface: ``__init__(text, source)`` → ``load() -> list[dict]``.
    """

    def __init__(self, text: str, source: str = ""):
        self.text = text
        self.source = source

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> List[dict]:
        soup = BeautifulSoup(self.text, "html.parser")
        self._strip_noise(soup)

        page_title = self._get_title(soup)
        content = self._find_content_area(soup)
        return self._extract_sections(content, page_title)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _strip_noise(self, soup: BeautifulSoup) -> None:
        for tag in soup.find_all(_NOISE_TAGS):
            tag.decompose()

    def _get_title(self, soup: BeautifulSoup) -> str:
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text(strip=True)
        h1 = soup.find("h1")
        return h1.get_text(strip=True) if h1 else self.source

    def _find_content_area(self, soup: BeautifulSoup) -> Tag:
        for sel in _CONTENT_SELECTORS:
            tag_name = sel["name"]
            attrs = sel.get("attrs")
            el = soup.find(tag_name, attrs) if attrs else soup.find(tag_name)
            if el:
                return el
        return soup  # fallback: whole document

    def _extract_sections(self, content: Tag, page_title: str) -> List[dict]:
        sections: List[dict] = []
        current_heading = page_title
        current_paragraphs: List[str] = []

        for el in content.descendants:
            if not isinstance(el, Tag):
                continue

            if el.name in _HEADING_TAGS:
                # Flush accumulated paragraphs under the previous heading
                if current_paragraphs:
                    sections.append(self._make_doc(current_heading, current_paragraphs))
                    current_paragraphs = []

                current_heading = el.get_text(strip=True) or current_heading

            elif el.name == "p":
                text = el.get_text(strip=True)
                if text:
                    current_paragraphs.append(text)

        # Flush final section
        if current_paragraphs:
            sections.append(self._make_doc(current_heading, current_paragraphs))

        if not sections:
            # Last-resort: grab all text from the content area
            fallback = content.get_text(separator="\n", strip=True)
            if fallback:
                sections.append(self._make_doc(page_title, [fallback]))

        return sections

    def _make_doc(self, section_heading: str, paragraphs: List[str]) -> dict:
        return {
            "page_content": "\n\n".join(paragraphs),
            "metadata": {
                "source": self.source,
                "section": section_heading,
            },
        }
