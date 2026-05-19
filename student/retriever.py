import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
CANDIDATE_POOL = 100

W_ORIGINAL = 5.0
W_EXPANDED = 1.0
W_PRF = 0.5
W_SEMANTIC = 0.8

PRF_DOCS = 2
PRF_TERMS = 3
PRF_MIN_TOKEN_LEN = 5

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{3,}")

PRF_STOPWORDS = {
    "this", "that", "these", "those", "with", "from", "into", "have", "been",
    "will", "would", "could", "should", "their", "there", "where", "which",
    "what", "when", "while", "more", "than", "some", "such", "only", "also",
    "each", "other", "they", "them", "your", "many", "most", "very",
    "used", "using", "uses", "user", "make", "made", "must", "need",
    "needs", "needed", "shown", "below", "above", "after", "before", "between",
    "about", "however", "though", "still", "type", "types", "first", "second",
    "default", "different", "support", "supports", "supported", "include",
    "includes", "included", "true", "false", "none", "name", "names",
    "param", "params", "parameter", "parameters", "function", "functions",
    "class", "classes", "method", "methods", "value", "values", "return",
    "returns", "args", "kwargs", "self", "cls", "test", "tests", "example",
    "examples", "code", "see", "note", "notes", "doc", "docs", "documentation",
    "import", "imports", "module", "modules", "file", "files",
    "vllm", "the", "and", "for", "are", "you", "can", "any",
    "all", "not", "but",
}


class Retriever:

    def __init__(self) -> None:
        self._bm25 = bm25s.BM25.load(str(INDEX_PATH), load_corpus=False)
        with open(CHUNKS_META) as fp:
            self._meta = json.load(fp)
        try:
            with open(CORPUS_PATH) as fp:
                self._corpus: List[str] = json.load(fp)
        except FileNotFoundError:
            self._corpus = []

        try:
            self._embeddings: Optional[np.ndarray] = np.load(EMBED_PATH)
            self._embedder: Optional[Embedder] = Embedder()
        except (FileNotFoundError, OSError):
            self._embeddings = None
            self._embedder = None

        self._cached_search = lru_cache(
            maxsize=QUERY_CACHE_SIZE)(
            self._search_raw)

    def _bm25_search(self, query: str, n: int) -> List[int]:
        tokens = bm25s.tokenize([query], show_progress=False)
        results, _ = self._bm25.retrieve(
            tokens, k=min(n, len(self._meta)), show_progress=False
        )
        return [int(idx) for idx in results[0]]

    def _prf_terms(
            self,
            original_query: str,
            top_indices: List[int]) -> List[str]:
        if not self._corpus or not top_indices:
            return []

        query_tokens = {tok.lower()
                        for tok in _TOKEN_RE.findall(original_query)}
        per_doc_sets: List[Set[str]] = []
        total_counts: Counter = Counter()

        for idx in top_indices[:PRF_DOCS]:
            if idx >= len(self._corpus):
                continue
            text = self._corpus[idx]
            doc_tokens: Set[str] = set()
            for tok in _TOKEN_RE.findall(text):
                lower = tok.lower()
                if len(lower) < PRF_MIN_TOKEN_LEN:
                    continue
                if lower in query_tokens or lower in PRF_STOPWORDS:
                    continue
                doc_tokens.add(lower)
                total_counts[lower] += 1
            per_doc_sets.append(doc_tokens)

        if len(per_doc_sets) < 2:
            return []

        shared = per_doc_sets[0]
        for s in per_doc_sets[1:]:
            shared = shared & s
        if not shared:
            return []

        ranked = sorted(
            shared,
            key=lambda t: math.log1p(total_counts[t]),
            reverse=True,
        )
        return ranked[:PRF_TERMS]

    def _semantic_ranking(
            self,
            query: str,
            candidates: List[int]) -> List[int]:
        if (
            self._embeddings is None
            or self._embedder is None
            or not candidates
        ):
            return []
        cand_array = np.asarray(candidates, dtype=np.int64)
        cand_vectors = self._embeddings[cand_array]
        q_vec = self._embedder.encode_query(query)
        scores = cand_vectors @ q_vec
        order = np.argsort(-scores)
        return [int(candidates[i]) for i in order]

    def _search_raw(self, query: str, k: int) -> Tuple[int, ...]:
        pool = max(CANDIDATE_POOL, k)
        original_ranking = self._bm25_search(query, pool)
        candidates = set(original_ranking)

        fused: Dict[int, float] = {
            idx: W_ORIGINAL / (RRF_K + rank + 1)
            for rank, idx in enumerate(original_ranking)
        }

        expanded = expand_query(query)
        if expanded != query:
            for rank, idx in enumerate(self._bm25_search(expanded, pool)):
                if idx in candidates:
                    fused[idx] += W_EXPANDED / (RRF_K + rank + 1)

        prf_terms = self._prf_terms(query, original_ranking)
        if prf_terms:
            prf_query = f"{query} {' '.join(prf_terms)}"
            for rank, idx in enumerate(self._bm25_search(prf_query, pool)):
                if idx in candidates:
                    fused[idx] += W_PRF / (RRF_K + rank + 1)

        semantic_ranking = self._semantic_ranking(query, original_ranking)
        for rank, idx in enumerate(semantic_ranking):
            fused[idx] += W_SEMANTIC / (RRF_K + rank + 1)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return tuple(idx for idx, _ in ordered[:k])

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
