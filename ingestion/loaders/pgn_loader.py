import re
from typing import List


class PGNLoader:
    """Load annotated PGN data, yielding one Document per game.

    Accepts raw PGN text directly (no file I/O) so it works in Lambda where
    the content has already been read from S3.
    """

    _KEEP_HEADERS = {
        "Event",
        "White",
        "Black",
        "Date",
        "Result",
        "ECO",
        "Opening",
        "Variation",
        "WhiteElo",
        "BlackElo",
        "Annotator",
    }

    _TAG_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')

    def __init__(self, text: str, source: str = ""):
        self.raw = text
        self.source = source

    def load(self) -> List[dict]:
        games = self._split_games(self.raw)
        docs: List[dict] = []

        for idx, game_text in enumerate(games):
            headers, moves = self._parse_game(game_text)
            if not moves.strip():
                continue

            readable = self._format_game(headers, moves)
            metadata = {
                "source": self.source,
                "game_index": idx,
                "white": headers.get("White", ""),
                "black": headers.get("Black", ""),
                "result": headers.get("Result", ""),
                "eco": headers.get("ECO", ""),
                "opening": headers.get("Opening", ""),
                "event": headers.get("Event", ""),
            }
            docs.append({"page_content": readable, "metadata": metadata})

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
            if (
                line.startswith("[")
                and current
                and not all(l.startswith("[") or l.strip() == "" for l in current)
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
