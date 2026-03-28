import os
import re
from typing import List

from langchain_core.documents import Document


class PGNLoader:
    """Load annotated PGN files, yielding one Document per game.

    Each Document's ``page_content`` contains a human-readable representation
    of the game: key header info (players, event, date, ECO, opening) followed
    by the move text with annotations preserved.  Raw tag pairs that don't add
    analytic value (e.g. Site, Round) are dropped to keep chunks focused.

    Annotations inside ``{ }`` are the highest-value content for RAG, so they
    are always retained.
    """

    # Headers worth surfacing in the text representation
    _KEEP_HEADERS = {
        "Event", "White", "Black", "Date", "Result",
        "ECO", "Opening", "Variation", "WhiteElo", "BlackElo",
        "Annotator",
    }

    _TAG_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        with open(self.file_path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()

        games = self._split_games(raw)
        docs: List[Document] = []
        filename = os.path.basename(self.file_path)

        for idx, game_text in enumerate(games):
            headers, moves = self._parse_game(game_text)
            if not moves.strip():
                continue

            readable = self._format_game(headers, moves)
            metadata = {
                "source": filename,
                "game_index": idx,
                "white": headers.get("White", ""),
                "black": headers.get("Black", ""),
                "result": headers.get("Result", ""),
                "eco": headers.get("ECO", ""),
                "opening": headers.get("Opening", ""),
                "event": headers.get("Event", ""),
            }
            docs.append(Document(page_content=readable, metadata=metadata))

        return docs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_games(raw: str) -> List[str]:
        """Split a multi-game PGN string into individual game strings."""
        games: List[str] = []
        current: List[str] = []

        for line in raw.splitlines(keepends=True):
            # A new game starts when we see a tag pair after move text.
            if line.startswith("[") and current and not all(
                l.startswith("[") or l.strip() == "" for l in current
            ):
                games.append("".join(current))
                current = []
            current.append(line)

        if current:
            games.append("".join(current))
        return games

    def _parse_game(self, game_text: str) -> tuple[dict, str]:
        """Return (headers_dict, move_text) for a single PGN game."""
        headers: dict = {}
        move_lines: List[str] = []
        in_moves = False

        for line in game_text.splitlines():
            tag_match = self._TAG_RE.match(line)
            if tag_match and not in_moves:
                headers[tag_match.group(1)] = tag_match.group(2)
            else:
                in_moves = True
                move_lines.append(line)

        moves = " ".join(move_lines).strip()
        # Collapse multiple spaces
        moves = re.sub(r"\s{2,}", " ", moves)
        return headers, moves

    def _format_game(self, headers: dict, moves: str) -> str:
        """Build a human-readable string for a single game."""
        parts: List[str] = []

        white = headers.get("White", "?")
        black = headers.get("Black", "?")
        result = headers.get("Result", "?")
        parts.append(f"{white} vs {black} ({result})")

        event = headers.get("Event")
        date = headers.get("Date")
        if event:
            label = event
            if date:
                label += f", {date}"
            parts.append(label)

        opening = headers.get("Opening", "")
        variation = headers.get("Variation", "")
        eco = headers.get("ECO", "")
        if opening:
            opening_str = opening
            if variation:
                opening_str += f": {variation}"
            if eco:
                opening_str += f" [{eco}]"
            parts.append(opening_str)

        elos: List[str] = []
        for side in ("White", "Black"):
            elo = headers.get(f"{side}Elo")
            if elo:
                elos.append(f"{side}: {elo}")
        if elos:
            parts.append("Elo " + ", ".join(elos))

        annotator = headers.get("Annotator")
        if annotator:
            parts.append(f"Annotated by {annotator}")

        parts.append("")  # blank line before moves
        parts.append(moves)

        return "\n".join(parts)
