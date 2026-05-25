import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bm25s
import numpy as np

from .embedder import Embedder
from .models import MinimalSource
from .query_expander import expand_query

CHUNKS_META = Path("data/processed/chunks/chunks_meta.json")
CORPUS_PATH = Path("data/processed/chunks/corpus.json")
INDEX_PATH = Path("data/processed/bm25_index")
EMBED_PATH = Path("data/processed/embeddings.npy")

QUERY_CACHE_SIZE = 256
RRF_K = 60
CANDIDATE_POOL = 60

W_ORIGINAL = 5.0
W_EXPANDED = 1.0
W_SEMANTIC = 2.0


class Retriever:

    def __init__(self) -> None:
        self._bm25 = bm25s.BM25.load(str(INDEX_PATH), load_corpus=False)
        with open(CHUNKS_META) as fp:
            self._meta = json.load(fp)

        try:
            self._embeddings: Optional[np.ndarray] = np.load(EMBED_PATH)
            self._embedder: Optional[Embedder] = Embedder()
        except (FileNotFoundError, OSError):
            self._embeddings = None
            self._embedder = None

        self._cached_search = lru_cache(maxsize=QUERY_CACHE_SIZE)(self._search_raw)

    def _bm25_search(self, query: str, n: int) -> List[int]:
        tokens = bm25s.tokenize([query], show_progress=False)
        results, _ = self._bm25.retrieve(
            tokens, k=min(n, len(self._meta)), show_progress=False
        )
        return [int(idx) for idx in results[0]]

    def _semantic_search(self, query: str, n: int) -> List[int]:
        if self._embeddings is None or self._embedder is None:
            return []
        q_vec = self._embedder.encode_query(query)
        scores = self._embeddings @ q_vec
        top_n = min(n, len(scores))
        idx = np.argpartition(-scores, top_n - 1)[:top_n]
        idx = idx[np.argsort(-scores[idx])]
        return idx.tolist()

    def _search_raw(self, query: str, k: int) -> Tuple[int, ...]:
        pool = max(CANDIDATE_POOL, k)

        original_ranking = self._bm25_search(query, pool)
        fused: Dict[int, float] = {
            idx: W_ORIGINAL / (RRF_K + rank + 1)
            for rank, idx in enumerate(original_ranking)
        }

        expanded = expand_query(query)
        if expanded != query:
            candidates = set(original_ranking)
            for rank, idx in enumerate(self._bm25_search(expanded, pool)):
                if idx in candidates:
                    fused[idx] += W_EXPANDED / (RRF_K + rank + 1)

        for rank, idx in enumerate(self._semantic_search(query, pool)):
            fused[idx] = fused.get(idx, 0.0) + W_SEMANTIC / (RRF_K + rank + 1)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return tuple(idx for idx, _ in ordered[:k])

    def search_batch(self, queries: List[str], k: int = 10) -> List[List[MinimalSource]]:
        pool = max(CANDIDATE_POOL, k)
        n = len(queries)

        tokens = bm25s.tokenize(queries, show_progress=False)
        bm25_results, _ = self._bm25.retrieve(
            tokens, k=min(pool, len(self._meta)), show_progress=False
        )

        expanded_queries = [expand_query(q) for q in queries]
        expand_mask = [eq != q for q, eq in zip(queries, expanded_queries)]
        exp_results = None
        if any(expand_mask):
            exp_tokens = bm25s.tokenize(expanded_queries, show_progress=False)
            exp_results, _ = self._bm25.retrieve(
                exp_tokens, k=min(pool, len(self._meta)), show_progress=False
            )

        semantic_rankings: List[List[int]] = [[] for _ in range(n)]
        if self._embeddings is not None and self._embedder is not None:
            q_vecs = self._embedder.encode_corpus(queries)
            all_scores = self._embeddings @ q_vecs.T
            top_n = min(pool, len(self._embeddings))
            for i in range(n):
                scores_i = all_scores[:, i]
                idx = np.argpartition(-scores_i, top_n - 1)[:top_n]
                idx = idx[np.argsort(-scores_i[idx])]
                semantic_rankings[i] = idx.tolist()

        all_sources: List[List[MinimalSource]] = []
        for i in range(n):
            original_ranking = [int(x) for x in bm25_results[i]]
            fused: Dict[int, float] = {
                idx: W_ORIGINAL / (RRF_K + rank + 1)
                for rank, idx in enumerate(original_ranking)
            }
            if expand_mask[i] and exp_results is not None:
                candidates = set(original_ranking)
                for rank, idx in enumerate(int(x) for x in exp_results[i]):
                    if idx in candidates:
                        fused[idx] += W_EXPANDED / (RRF_K + rank + 1)
            for rank, idx in enumerate(semantic_rankings[i]):
                fused[idx] = fused.get(idx, 0.0) + W_SEMANTIC / (RRF_K + rank + 1)

            ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
            all_sources.append([
                MinimalSource(
                    file_path=self._meta[idx]["file_path"],
                    first_character_index=self._meta[idx]["first_character_index"],
                    last_character_index=self._meta[idx]["last_character_index"],
                )
                for idx, _ in ordered[:k]
            ])
        return all_sources

    def search(self, query: str, k: int = 10) -> List[MinimalSource]:
        if k <= 0 or not query.strip():
            return []
        indices = self._cached_search(query, k)
        return [
            MinimalSource(
                file_path=self._meta[idx]["file_path"],
                first_character_index=self._meta[idx]["first_character_index"],
                last_character_index=self._meta[idx]["last_character_index"],
            )
            for idx in indices
        ]
