import json
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import bm25s

from .models import MinimalSource

CHUNKS_META = Path("data/processed/chunks/chunks_meta.json")
INDEX_PATH = Path("data/processed/bm25_index")
QUERY_CACHE_SIZE = 256


class Retriever:

    def __init__(self) -> None:
        self._retriever = bm25s.BM25.load(str(INDEX_PATH), load_corpus=False)
        with open(CHUNKS_META) as fp:
            self._meta = json.load(fp)
        self._cached_search = lru_cache(maxsize=QUERY_CACHE_SIZE)(self._search_raw)

    def _search_raw(self, query: str, k: int) -> Tuple[int, ...]:
        tokens = bm25s.tokenize([query], show_progress=False)
        results, _ = self._retriever.retrieve(
            tokens, k=min(k, len(self._meta)), show_progress=False
        )
        return tuple(int(idx) for idx in results[0])

    def search(self, query: str, k: int = 10) -> List[MinimalSource]:
        if k <= 0 or not query.strip():
            return []
        indices = self._cached_search(query, k)
        sources: List[MinimalSource] = []
        for idx in indices:
            meta = self._meta[idx]
            sources.append(
                MinimalSource(
                    file_path=meta["file_path"],
                    first_character_index=meta["first_character_index"],
                    last_character_index=meta["last_character_index"],
                )
            )
        return sources
