import re
from collections.abc import Iterable


_NON_TEXT = re.compile(r"[^a-z' ]")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = _NON_TEXT.sub("", text.lower())
    return _WHITESPACE.sub(" ", text).strip()


class Vocabulary:
    def __init__(self) -> None:
        self.characters = " abcdefghijklmnopqrstuvwxyz'"
        self.blank_id = 0
        self.char_to_id = {
            character: index + 1
            for index, character in enumerate(self.characters)
        }

    def encode(self, text: str) -> list[int]:
        normalized = normalize_text(text)
        unknown = set(normalized) - self.char_to_id.keys()
        if unknown:
            raise ValueError(f"Text contains unsupported characters: {sorted(unknown)}")
        return [self.char_to_id[character] for character in normalized]

    def decode(self, tokens: Iterable[int]) -> str:
        characters: list[str] = []
        previous = None
        for token in tokens:
            token = int(token)
            if token != self.blank_id and token != previous:
                if token < 0 or token > len(self.characters):
                    raise ValueError(f"Token ID is outside the vocabulary: {token}")
                characters.append(self.characters[token - 1])
            previous = token
        return "".join(characters).strip()

    @property
    def size(self) -> int:
        return len(self.characters) + 1
