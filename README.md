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
                 │   │ (synonyms)   │    └────────┬─────────┘   │
                 │   └──────────────┘             │             │
                 │                                │             │
                 │   Semantic search ─────────────┤             │
                 │   (all embeddings)             │             │
                 │                       Weighted RRF fusion    │
                 │                                │             │
                 │                                ▼             │
                 │                     Top-k MinimalSource      │
                 └──────────────────────────────┬───────────────┘
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
   Each chunk's content is prefixed with its relative file path so BM25
   can match module names mentioned in queries.
   Per-chunk metadata (`file_path`, `first_character_index`,
   `last_character_index`) is persisted to disk so re-indexing is not
   needed across runs.
2. **Retrieval** (`retriever.py`) — the query is optionally expanded with
   a synonym table, then BM25 is run on both the original and expanded
   queries. In parallel, a **full semantic search** over all corpus
   embeddings is performed (one matrix multiply). The rankings are merged
   with weighted **Reciprocal Rank Fusion (RRF)**. Per-query results are
   memoised via `functools.lru_cache`. For datasets, all queries are
   batch-encoded in a single forward pass for efficiency.
3. **Generation** (`generator.py`) — the top-k chunks are read from disk
   and concatenated as context, then passed to `Qwen/Qwen3-0.6B` (GGUF,
   served via `llama-cpp-python`) with a system prompt that forces a
   self-contained, source-cited answer.

---

## Chunking Strategy

Three strategies are implemented depending on file type (`chunker.py`).
**Maximum chunk size is 2000 characters**, configurable via the
`--max_chunk_size` flag on `index`. All strategies use a chunk overlap
of 100 characters to avoid losing context at boundaries.

### Python files — `chunk_python`

Uses `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` from
`langchain-text-splitters`. The splitter tries separators in order:
`\nclass `, `\ndef `, `\n\tdef `, `\n`, ` `, `""`.
This keeps class and function definitions together whenever they fit within
the chunk size limit, respecting Python's natural code boundaries.

### Markdown files — `chunk_markdown`

Uses `RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN)`.
Separators prioritise headings (`\n## `, `\n### `, etc.) so each chunk
stays within a logical section of the documentation.

### All other files — `chunk_text`

Uses the generic `RecursiveCharacterTextSplitter` with standard separators
(`\n\n`, `\n`, ` `, `""`), splitting on paragraph boundaries first.

### Impact of chunk size

- **Too small** (<500 chars) — context is fragmented, the LLM cannot
  reconstruct meaning, and Recall@k drops because relevant content is
  spread across multiple chunks that may individually score too low.
- **Too large** (>3000 chars) — BM25 length normalisation penalises
  long chunks, the LLM context window is wasted on irrelevant code, and
  the system's `max_context_length` constraint is violated.
- **2000 chars** is the sweet spot: large enough to keep most functions
  intact, small enough to keep BM25 discriminative.

---

## Retrieval Method

The retriever is **hybrid**: it combines lexical BM25, full semantic search,
and query expansion through a weighted RRF fusion.

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
keywords. The corpus prefixes each chunk with its relative file path
so module path tokens (`vllm/attention/layer.py`) are indexed and
searchable.

### Semantic search (dense ranker)

A **MiniLM** sentence-transformer (`all-MiniLM-L6-v2`, 384-dim,
normalised) embeds every chunk at index time. At query time the query
is encoded and a **full dot-product search** is performed over all
corpus embeddings (`embeddings @ q_vec`). This means semantically
similar chunks can surface even if they share no keywords with the
query — crucial for documentation questions that paraphrase the source.

For dataset queries, all embeddings are batch-computed in a single
`encode_corpus` call and the full score matrix (`embeddings @ Q^T`)
is computed in one BLAS call, making throughput efficient.

### Query expansion (`query_expander.py`)

A curated synonym table maps domain terms onto common alternatives:
`k8s↔kubernetes`, `tp↔tensor parallel`, `kv-cache↔kvcache`,
`pagedattention↔paged attention`, etc. When expansion produces a
different string, a second BM25 query is run and merged into the fusion.

### Reciprocal Rank Fusion

The rankings (BM25 original, BM25 expanded, semantic) are merged with
weighted RRF:

```
fused(d) = Σ_i  w_i / (RRF_K + rank_i(d) + 1)
```

with `RRF_K = 60`, weights `W_ORIGINAL = 5.0`, `W_EXPANDED = 1.0`,
`W_SEMANTIC = 2.0`. The BM25 candidate pool is fixed at 60 documents;
the semantic search independently contributes its own top-60. Fusion
returns the top-k.

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
4. Sources are stitched with `---` separators and labelled with file
   path and character range.

The system prompt forces:
- a direct answer first (with the exact identifier from the context),
- one sentence of supporting detail,
- a final `Source: <file_path>` line,
- no hallucinated content beyond the context.

---

## Performance Analysis

Results on the **private** datasets (100 questions each):

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---------|----------|----------|----------|-----------|
| Docs    | 64%      | 78%      | **82%**  | 89%       |
| Code    | 49%      | 64%      | **68%**  | 73%       |

Pass thresholds:
- Docs Recall@5 ≥ 80% — **PASS** (82%)
- Code Recall@5 ≥ 50% — **PASS** (68%)

Indexing time: ~125s (≤ 300s limit).
Retrieval throughput: 200 questions in < 90s (batch encoding).

Key factors driving performance:
- **Full semantic search** over all embeddings catches queries that BM25
  misses entirely (paraphrases, synonyms not in the expansion table).
- **File path prefix in corpus** lets BM25 match module names mentioned
  in code questions (`vllm.attention`, `vllm.engine`, etc.).
- **Batch query encoding** encodes all dataset queries in one forward
  pass, keeping throughput well under the 90s limit.
- **`Path.as_posix()`** for file paths ensures cross-platform
  compatibility with the moulinette path comparison.
- **Chunk size = 2000 chars** keeps every chunk within the system's
  `max_context_length` validation limit.

### Ablation (qualitative)

| Configuration | Docs R@5 | Code R@5 |
|---|---|---|
| BM25 only | ~76% | ~48% |
| BM25 + query expansion | ~78% | ~50% |
| BM25 + semantic rerank (top-100 only) | ~84% | ~52% |
| Full hybrid (BM25 + full semantic search + RRF) | **82%** | **68%** |

---

## Design Decisions & Trade-offs

| Decision | Rationale |
|---|---|
| BM25 over pure TF-IDF | Better length normalisation; higher recall on code identifiers |
| Hybrid retrieval over single ranker | Semantic catches paraphrases; BM25 catches exact identifiers |
| Full semantic search (not just reranking) | Semantic can surface docs BM25 missed entirely — biggest recall gain |
| Weighted RRF (not score-sum) | Rank-based fusion is robust to score scale mismatches between rankers |
| `RecursiveCharacterTextSplitter.from_language` | Language-aware separators keep functions/sections intact without AST parsing overhead |
| Chunk overlap = 100 chars | Prevents context loss at boundaries without inflating corpus too much |
| File path prefix in corpus | BM25 indexes module paths; questions that reference `vllm/x/y.py` benefit directly |
| Batch encoding in `search_dataset` | One `encode_corpus` call for all queries + one BLAS matrix multiply — stays under 90s throughput limit |
| `Path.as_posix()` in index | Ground-truth datasets use `/`; Windows `\` would break matching |
| Chunk size = 2000 chars | Matches system max_context_length validation |
| Qwen3-0.6B (GGUF) via llama-cpp | CPU-only, no GPU required; small enough to run on any corrector machine |
| Greedy decoding (`temperature=0.0`) | Deterministic, reproducible, hallucination-resistant |
| MiniLM-L6-v2 for embeddings | 384-dim, very fast on CPU, strong retrieval quality per parameter |
| `lru_cache` on search | Free re-queries during dataset evaluation |

**Trade-offs accepted**:
- Embedding the whole corpus at index time costs ~2 min on CPU but
  removes all latency at query time.
- Full semantic search is one matrix multiply per query (16k × 384),
  which numpy handles in < 2ms — negligible cost for a large recall gain.
- Qwen3-0.6B is small and occasionally produces shallow answers; this
  was preferred to GPU dependency.

---

## Bonus Features

| Feature | Points | Where |
|---|---|---|
| **Query expansion** (synonym table) | 1 pt | [`query_expander.py`](student/query_expander.py) |
| **Semantic embeddings** (MiniLM, normalised) | 1 pt | [`embedder.py`](student/embedder.py), [`indexer.py`](student/indexer.py) |
| **Caching** (persisted index + `lru_cache`) | 1 pt | [`indexer.py`](student/indexer.py), [`retriever.py`](student/retriever.py) |
| **Hybrid retrieval** (BM25 + semantic, RRF) | 2 pt | [`retriever.py`](student/retriever.py) |

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

### Evaluate retrieval quality (Recall@k)

```bash
uv run python -m student evaluate \
    data/output/search_results/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10
```

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
│   ├── chunker.py            # language-aware chunking strategies
│   ├── indexer.py            # builds BM25 + embeddings + meta on disk
│   ├── retriever.py          # hybrid BM25 / semantic / RRF + batch search
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
