from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.data.text import normalize_transcript


class CharVocabulary:
    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.blank_id = self.token_to_id["<blank>"]
        self.unk_id = self.token_to_id["<unk>"]

    @classmethod
    def default(cls) -> "CharVocabulary":
        tokens = ["<blank>", "<unk>", "'", " "] + list("abcdefghijklmnopqrstuvwxyz")
        return cls({token: idx for idx, token in enumerate(tokens)})

    @classmethod
    def load(cls, path: str | Path) -> "CharVocabulary":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "token_to_id" in data:
            data = data["token_to_id"]
        return cls({str(token): int(idx) for token, idx in data.items()})

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    def encode(self, text: str) -> list[int]:
        text = normalize_transcript(text)
        return [self.token_to_id.get(char, self.unk_id) for char in text]

    def decode(self, ids: Iterable[int]) -> str:
        chars = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), "")
            if token in {"<blank>", "<unk>"}:
                continue
            chars.append(token)
        return "".join(chars).strip()
