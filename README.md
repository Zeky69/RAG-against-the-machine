*This project has been created as part of the 42 curriculum by Zeky69.*

# RAG Against the Machine

A Retrieval-Augmented Generation (RAG) system built on the vLLM codebase.
Given a natural language question, the system retrieves the most relevant
source chunks from the repository and generates a grounded answer using a
local language model (Qwen/Qwen3-0.6B by default).

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Chunking Strategy](#chunking-strategy)
3. [Retrieval Method](#retrieval-method)
4. [Answer Generation](#answer-generation)
5. [Performance Analysis](#performance-analysis)
6. [Design Decisions & Trade-offs](#design-decisions--trade-offs)
7. [Bonus Features](#bonus-features)
8. [Example Usage](#example-usage)
9. [Project Layout](#project-layout)

---

## System Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │              INDEXING (offline)              │
                 │                                              │
   repo files ──▶│  Chunker  ──▶  BM25 index    (lexical)       │
                 │           ──▶  Embeddings    (semantic)      │
                 │           ──▶  chunks_meta.json              │
                 └──────────────────────────────────────────────┘
                                       │
                                       ▼
                 ┌──────────────────────────────────────────────┐
   Query  ──────▶│              RETRIEVAL (online)              │
                 │                                              │
                 │   ┌──────────────┐    ┌──────────────────┐   │
                 │   │ Query        │    │ BM25 (original)  │   │
                 │   │ expansion    │───▶│ BM25 (expanded)  │   │
                 │   │ (synonyms)   │    │ BM25 (PRF)       │   │
                 │   └──────────────┘    │ Semantic re-rank │   │
                 │                       └─────────┬────────┘   │
                 │                                 │            │
                 │                       Weighted RRF fusion    │
                 │                                 │            │
                 │                                 ▼            │
                 │                      Top-k MinimalSource     │
                 └──────────────────────────────────┬───────────┘
                                                    │
                                                    ▼
                 ┌──────────────────────────────────────────────┐
                 │           GENERATION (Qwen3-0.6B)            │
                 │                                              │
                 │   Top-k chunks ──▶ Context window ──▶ LLM    │
                 │                                       │      │
                 │                                       ▼      │
                 │                              Grounded answer │
                 └──────────────────────────────────────────────┘
```

The pipeline has three stages:

1. **Indexing** (`indexer.py`) — repo files are chunked, then two parallel
   indices are built:
   - A **BM25** sparse index (`bm25s`) for lexical retrieval.
   - A **semantic** index of normalised sentence-transformer embeddings
     (`sentence-transformers/all-MiniLM-L6-v2`) for dense retrieval.
   Per-chunk metadata (`file_path`, `first_character_index`,
   `last_character_index`) is persisted to disk so re-indexing is not
   needed across runs.
2. **Retrieval** (`retriever.py`) — the query is expanded with a synonym
   table, then BM25 is queried on the original query, on the expanded
   query, and on a PRF-augmented query. A semantic re-ranking of the
   BM25 candidate pool is computed using the dense embeddings. The four
   rankings are merged with weighted **Reciprocal Rank Fusion (RRF)**.
   Per-query results are memoised via `functools.lru_cache`.
3. **Generation** (`generator.py`) — the top-k chunks are read from disk
   and concatenated as context, then passed to `Qwen/Qwen3-0.6B` (GGUF,
   served via `llama-cpp-python`) with a system prompt that forces a
   self-contained, source-cited answer.

---

## Chunking Strategy

Two strategies are implemented depending on file type (`chunker.py`).
**Maximum chunk size is 2000 characters**, configurable via the
`--max_chunk_size` flag on `index`.

### Python files — AST-aware chunker (`chunk_python`)

The file is parsed with Python's `ast` module. The top-level body is walked
(non-recursively) to preserve natural code boundaries:

- **Functions and classes** that fit within `max_chunk_size` are kept as
  single chunks, including their decorators.
- **Large classes** are split into a header chunk (class declaration +
  class-level attributes) plus one chunk per method.
- **Large standalone functions** are split by character.
- **Module-level code** (imports, constants, type aliases) between
  definitions is accumulated into prose blocks and chunked separately.

This avoids both **redundancy** (no recursive double-indexing of nested
defs) and **data loss** (module-level constants are fully indexed).

### All other files — paragraph chunker (`chunk_text`)

Content is split on double newlines. Paragraphs are accumulated until
`max_chunk_size` is reached, then flushed. Single paragraphs larger than
`max_chunk_size` are split by character.

### Impact of chunk size

- **Too small** (<500 chars) → context is fragmented, the LLM cannot
  reconstruct meaning, and Recall@k drops because relevant content is
  spread across multiple chunks.
- **Too large** (>3000 chars) → BM25 length normalisation penalises
  long chunks, the LLM context window is wasted on irrelevant code, and
  the moulinette `max_context_length` constraint is violated.
- **2000 chars** was chosen as the sweet spot: aligned with the
  moulinette validation cap, large enough to keep most functions
  intact, small enough to keep BM25 discriminative.

---

## Retrieval Method

The retriever is **hybrid**: it combines lexical BM25, semantic
embeddings, query expansion, and pseudo-relevance feedback through a
weighted RRF fusion.

### BM25 (primary lexical ranker)

Implementation: `bm25s` library with default parameters (`k1 = 1.5`,
`b = 0.75`).

BM25 ranks documents by term-frequency / inverse-document-frequency with
length normalisation:

```
score(d, q) = Σ_t  IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))
```

BM25 is the natural fit for code retrieval because identifiers
(function names, class names, command flags) are best matched by exact
keywords.

### Semantic embeddings (dense ranker)

A **MiniLM** sentence-transformer (`all-MiniLM-L6-v2`, 384-dim,
normalised) embeds every chunk at index time and the query at search
time. Cosine similarity (dot product on normalised vectors) re-ranks the
BM25 candidate pool to recover paraphrases that lexical search misses.

### Query expansion (`query_expander.py`)

A curated synonym table maps domain terms onto common alternatives:
`k8s↔kubernetes`, `tp↔tensor parallel`, `kv-cache↔kvcache`,
`pagedattention↔paged attention`, etc. The expanded query is run as a
**separate BM25 query** and merged into the fusion.

### Pseudo-Relevance Feedback (PRF)

The top-2 documents from the initial BM25 ranking are mined for terms
that:
- appear in **both** documents (intersection),
- are not in the original query,
- pass a stopword filter, and
- have a minimum length of 5 characters.

The 3 highest-frequency terms are appended to the query and BM25 is
re-run. This recovers domain-specific vocabulary the user could not have
known.

### Reciprocal Rank Fusion

The four rankings (BM25 original, BM25 expanded, BM25 PRF, semantic) are
merged with weighted RRF:

```
fused(d) = Σ_i  w_i / (RRF_K + rank_i(d) + 1)
```

with `RRF_K = 60`, weights `W_ORIGINAL = 5.0`, `W_EXPANDED = 1.0`,
`W_PRF = 0.5`, `W_SEMANTIC = 0.8`. The candidate pool is fixed at 100
documents; fusion always returns the top-k.

### Caching

- **Persisted index** — BM25 model, embeddings (`.npy`) and chunk
  metadata are written to `data/processed/` and loaded once per process.
- **Query cache** — `functools.lru_cache(maxsize=256)` memoises
  `(query, k)` so repeated questions in a dataset are free.

---

## Answer Generation

**Default model**: `Qwen/Qwen3-0.6B-GGUF`, served locally via
`llama-cpp-python` on CPU. Greedy decoding (`temperature=0.0`) is used
for deterministic, hallucination-resistant answers.

Context construction (`generator.py`):

1. For each retrieved source, the original file is re-read and sliced by
   character offsets.
2. Per-source content is truncated to **900 characters**.
3. Total context is capped at **3000 characters** to fit comfortably in
   the 4096-token context window.
4. Sources are stitched with `--- ` separators and labelled with file
   path and character range.

The system prompt forces:
- a direct answer first (with the exact identifier from the context),
- one sentence of supporting detail,
- a final `Source: <file_path>` line,
- no hallucinated content beyond the context.

---

## Performance Analysis

Results on the **public** datasets (100 questions each):

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---------|----------|----------|----------|-----------|
| Docs    | 53%      | 78%      | **82%**  | 88%       |
| Code    | 36%      | 48%      | **53%**  | 60%       |

Pass thresholds:
- Docs Recall@5 ≥ 80% — **PASS** (82%)
- Code Recall@5 ≥ 50% — **PASS** (53%)

Key factors driving performance:
- **`Path.as_posix()`** for file paths ensures cross-platform
  compatibility with the moulinette path comparison (otherwise Windows
  backslash paths break ground-truth matching).
- **Chunk size = 2000 chars** keeps every chunk within the moulinette
  `max_context_length` validation limit.
- The **AST-aware Python chunker** indexes module-level constants and
  type aliases that purely structural chunkers miss.
- **Hybrid fusion** lifts recall on paraphrased queries that pure BM25
  misses; **PRF** lifts recall on queries that use vocabulary slightly
  different from the docs.

### Ablation (qualitative)

| Configuration                                | Docs R@5 | Code R@5 |
|----------------------------------------------|----------|----------|
| BM25 only                                    | ~76%     | ~48%     |
| BM25 + query expansion                       | ~78%     | ~50%     |
| BM25 + query expansion + semantic re-ranking | ~81%     | ~52%     |
| Full hybrid (+ PRF, full RRF)                | **82%**  | **53%**  |

---

## Design Decisions & Trade-offs

| Decision                                  | Rationale                                                         |
|-------------------------------------------|-------------------------------------------------------------------|
| BM25 over pure TF-IDF                     | Better length normalisation; higher recall on code identifiers    |
| Hybrid retrieval over single ranker       | Recovers paraphrases (semantic) without losing exact matches (BM25) |
| Weighted RRF (not score-sum)              | Rank-based fusion is robust to score scale mismatches between rankers |
| AST-aware Python chunking                 | Preserves function/class boundaries; keeps decorators with bodies |
| Module-level code as prose blocks         | Avoids losing constants and type aliases between definitions      |
| `Path.as_posix()` in index                | Ground-truth datasets use `/`; Windows `\` would break matching   |
| Chunk size = 2000 chars                   | Matches moulinette max_context_length validation                  |
| Qwen3-0.6B (GGUF) via llama-cpp           | CPU-only, no GPU required; small enough to run on the corrector's laptop |
| Greedy decoding (`temperature=0.0`)       | Deterministic, reproducible, hallucination-resistant              |
| MiniLM-L6-v2 for embeddings               | 384-dim, very fast on CPU, strong retrieval quality per parameter |
| `lru_cache` on search                     | Free re-queries during dataset evaluation                         |

**Trade-offs accepted**:
- Embedding the whole corpus at index time costs ~1–2 min on CPU but
  removes all latency at query time.
- BM25 + semantic fusion is two passes; this is fine because the
  candidate pool is fixed at 100 and the semantic re-rank is a single
  matrix multiplication.
- Qwen3-0.6B is small and occasionally produces shallow answers; this
  was preferred to GPU dependency, which the moulinette environment
  cannot assume.

---

## Bonus Features

The following bonus features (from the subject's bonus list) are
implemented and exercised by the default retrieval pipeline:

| Feature                                       | Worth | Where                                               |
|-----------------------------------------------|-------|-----------------------------------------------------|
| **Query expansion** (synonym table)           | 1 pt  | [`query_expander.py`](student/query_expander.py)    |
| **Semantic embeddings** (MiniLM, normalised)  | 1 pt  | [`embedder.py`](student/embedder.py), [`indexer.py`](student/indexer.py) |
| **Caching** (persisted index + `lru_cache`)   | 1 pt  | [`indexer.py`](student/indexer.py), [`retriever.py`](student/retriever.py) |
| **Hybrid retrieval** (BM25 + semantic, RRF)   | 2 pt  | [`retriever.py`](student/retriever.py)              |

Additional retrieval enhancement (counted under hybrid fusion):

- **Pseudo-Relevance Feedback (PRF)** — top-document term mining with
  stopword filtering and intersection across top docs, fused as a
  third BM25 pass.

Total: **5 bonus points** (cap reached).

> Note: vLLM-based LLM serving is **not** used. Generation is done via
> `llama-cpp-python` with the Qwen3-0.6B GGUF model so the system runs
> on a pure CPU machine without requiring GPU drivers or vLLM
> dependencies on the corrector's environment.

---

## Example Usage

All commands are run from the repository root.

### Install dependencies

```bash
uv sync
```

### Index the repository

```bash
uv run python -m student index data/raw/vllm-0.10.1
```

Optional flag: `--max_chunk_size 2000`.

This produces:

```
data/processed/
├── bm25_index/        # bm25s index
├── chunks/
│   ├── chunks_meta.json
│   └── corpus.json
└── embeddings.npy
```

### Search for a single query

```bash
uv run python -m student search "How to configure the OpenAI-compatible server?" --k 10
```

### Batch-search a dataset

```bash
uv run python -m student search_dataset \
    data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10
```

Output is written to `data/output/search_results/<dataset_name>.json` as
a valid `StudentSearchResults` JSON.

### Generate an answer

```bash
uv run python -m student answer "What is PagedAttention?" --k 10
```

### Generate answers for a whole dataset

```bash
uv run python -m student answer_dataset \
    --student_search_results_path data/output/search_results/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer
```

Output is written as a valid `StudentSearchResultsAndAnswer` JSON.

### Evaluate retrieval quality (Recall@k)

```bash
uv run python -m student evaluate \
    data/output/search_results/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10
```

### Run the full evaluation pipeline

```bash
make evaluate
```

This runs `search_dataset` on both public datasets and invokes the
moulinette evaluator against the Docs (80% threshold) and Code (50%
threshold) targets.

### Lint and type-check

```bash
make lint          # flake8 + mypy
make lint-strict   # flake8 + mypy --strict
```

---

## Project Layout

```
.
├── student/                  # main Python module (python -m student ...)
│   ├── __main__.py           # Fire CLI entrypoint (RAGSystem)
│   ├── models.py             # Pydantic models (MinimalSource, RagDataset, ...)
│   ├── chunker.py            # AST + text chunking strategies
│   ├── indexer.py            # builds BM25 + embeddings + meta on disk
│   ├── retriever.py          # hybrid BM25 / semantic / RRF / PRF / cache
│   ├── query_expander.py     # synonym-table query expansion
│   ├── embedder.py           # sentence-transformers wrapper
│   ├── generator.py          # Qwen3-0.6B via llama-cpp-python
│   └── evaluator.py          # Recall@k
├── data/                     # datasets and persisted indices
├── exams/                    # corrector scripts (exam_retrieval.sh, ...)
├── evaluations/              # past evaluation outputs
├── stubs/                    # type stubs for mypy
├── pyproject.toml
├── uv.lock
└── Makefile
```
