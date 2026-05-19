import json
from pathlib import Path
from typing import List

import bm25s

from .models import MinimalSource

CHUNKS_META = Path("data/processed/chunks/chunks_meta.json")
INDEX_PATH = Path("data/processed/bm25_index")


class Retriever:

    def __init__(self) -> None:
        self._retriever = bm25s.BM25.load(str(INDEX_PATH), load_corpus=False)
        with open(CHUNKS_META) as fp:
            self._meta = json.load(fp)

    def search(self, query: str, k: int = 10) -> List[MinimalSource]:
        tokens = bm25s.tokenize([query])
        results, _ = self._retriever.retrieve(tokens, k=min(k, len(self._meta)))
        sources: List[MinimalSource] = []
        for idx in results[0]:
            meta = self._meta[idx]
            sources.append(
                MinimalSource(
                    file_path=meta["file_path"],
                    first_character_index=meta["first_character_index"],
                    last_character_index=meta["last_character_index"],
                )
            )
        return sources