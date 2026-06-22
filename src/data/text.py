from __future__ import annotations

import re

_ALLOWED = set("abcdefghijklmnopqrstuvwxyz' ")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_transcript(text: str) -> str:
    text = text.lower()
    chars = []
    for char in text:
        if char in _ALLOWED:
            chars.append(char)
        elif char.isspace():
            chars.append(" ")
        else:
            chars.append(" ")
    return _WHITESPACE_RE.sub(" ", "".join(chars)).strip()
