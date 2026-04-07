class TXTLoader:
    """Load plain text content into a single document dict."""

    def __init__(self, text: str, source: str = ""):
        self.text = text
        self.source = source

    def load(self) -> list[dict]:
        return [{"page_content": self.text, "metadata": {"source": self.source}}]
