from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


_END = "__end__"


@dataclass
class SemanticPrefixTrie:
    """Prefix trie over semantic-ID code sequences."""

    root: dict = field(default_factory=dict)

    def insert(self, code_sequence: Iterable[int]) -> None:
        node = self.root
        for value in code_sequence:
            value = int(value)
            node = node.setdefault(value, {})
        node[_END] = True

    def allowed_next(self, prefix: Iterable[int]) -> list[int]:
        node = self.root
        for value in prefix:
            value = int(value)
            if value not in node:
                return []
            node = node[value]
        return sorted(int(key) for key in node.keys() if key != _END)

    def has_prefix(self, prefix: Iterable[int]) -> bool:
        node = self.root
        for value in prefix:
            value = int(value)
            if value not in node:
                return False
            node = node[value]
        return True

    def has_code(self, code_sequence: Iterable[int]) -> bool:
        node = self.root
        for value in code_sequence:
            value = int(value)
            if value not in node:
                return False
            node = node[value]
        return bool(node.get(_END, False))

    @classmethod
    def from_item_to_code(cls, item_to_code: dict[str, list[int]]) -> "SemanticPrefixTrie":
        trie = cls()
        for code_sequence in item_to_code.values():
            trie.insert(code_sequence)
        return trie
