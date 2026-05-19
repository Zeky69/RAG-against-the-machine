*This project has been created as part of the 42 curriculum by Zeky69.*

# RAG Against the Machine

A Retrieval-Augmented Generation (RAG) system built on the vLLM codebase.
Given a natural language question, the system retrieves the most relevant
source chunks from the repository and generates a grounded answer using a
local language model.

---

## System Architecture

```
Query
  │
  ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Retriever  │────▶│  Retrieved chunks │────▶│    Generator    │
│   (BM25)    │     │  (file_path +     │     │  (Qwen3-0.6B)  │
└─────────────┘     │   char offsets)   │     └─────────────────┘
       ▲            └──────────────────┘              │
       │                                              ▼
┌─────────────┐                               Generated Answer
│  BM25 Index │
│  (on disk)  │
└─────────────┘
       ▲
       │
┌─────────────┐
│   Indexer   │◀── vLLM repository files
│  (chunker)  │
└─────────────┘
```

The pipeline has three stages:

1. **Indexing** — the vLLM repository is chunked and indexed offline with
   BM25. Python files use an AST-aware chunker; all other files use a
   paragraph-based text chunker.
2. **Retrieval** — BM25 ranks all chunks against the query and returns the
   top-k most relevant source windows (file path + character offsets).
3. **Generation** — the top-k chunks are concatenated as context and passed
   to Qwen/Qwen3-0.6B, which generates a grounded answer.

---

## Chunking Strategy

Two strategies are implemented depending on file type.

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

This avoids both redundancy (no recursive double-indexing of nested defs)
and data loss (module-level constants are fully indexed).

### All other files — paragraph chunker (`chunk_text`)

Content is split on double newlines. Paragraphs are accumulated until
`max_chunk_size` is reached, then flushed. Single paragraphs larger than
`max_chunk_size` are split by character.

**Maximum chunk size**: 2000 characters (configurable via `--max_chunk_size`).

---

## Retrieval Method

**Algorithm**: BM25 (Best Match 25) via the `bm25s` library.

BM25 ranks documents by term-frequency / inverse-document-frequency with
length normalisation. Given a query `q` and a chunk `d`:

```
score(d, q) = sum_t [ IDF(t) * tf(t,d)*(k1+1) / (tf(t,d) + k1*(1-b+b*|d|/avgdl)) ]
```

with `k1 = 1.5` and `b = 0.75` (BM25 defaults).

**Why BM25?**
- Exact keyword matching is very effective for code queries (function names,
  class names, constant names).
- Zero cold-start cost after indexing — retrieval is sub-millisecond.
- No GPU required for retrieval.

**Index storage**: BM25 index and chunk metadata are persisted to
`data/processed/` so only one indexing pass is needed.

---

## Performance Analysis

Results on the **public** datasets (100 questions each):

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---------|----------|----------|----------|-----------|
| Docs    | 53%      | 78%      | **82%**  | 88%       |
| Code    | 36%      | 48%      | **53%**  | 60%       |

Pass thresholds: Docs Recall@5 >= 80% PASS — Code Recall@5 >= 50% PASS

Key factors driving performance:
- Using `.as_posix()` for file paths ensures cross-platform compatibility
  with the moulinette path comparison.
- Chunk size of 2000 chars keeps all chunks within the moulinette
  `max_context_length` validation limit.
- The AST-aware Python chunker indexes module-level constants and type
  aliases that purely structural chunkers miss.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| BM25 over TF-IDF | Better length normalisation; higher recall on code |
| AST-aware Python chunking | Preserves function/class boundaries |
| Module-level code in prose blocks | Avoids losing constants and type aliases |
| Posix paths in index | Ground-truth datasets use `/`; Windows `\` breaks matching |
| Chunk size = 2000 chars | Matches moulinette max_context_length validation |
| Qwen3-0.6B with greedy decoding | Deterministic and avoids hallucination |

**Trade-offs**: BM25 is a lexical method and struggles with paraphrased
queries where exact keywords are absent. Semantic or hybrid retrieval would
improve recall at the cost of GPU memory and indexing time.

---

## Challenges Faced

- **Path separator mismatch**: Python `Path` on Windows produces backslash
  paths; the moulinette compares with forward slashes. Fixed with
  `Path.as_posix()` during indexing.
- **Oversized chunks**: The original text chunker did not split paragraphs
  larger than `max_chunk_size`. Added a character-split fallback.
- **AST walk vs top-level walk**: `ast.walk()` recursively indexed nested
  methods separately from their class, creating redundant chunks and missing
  module-level code. Fixed by iterating only `tree.body`.
- **Pydantic field name**: The moulinette expects `question_str` in
  `MinimalSearchResults`. Aligned the model field name accordingly.

---

## Example Usage

### Install dependencies

```bash
uv sync
```

### Index the repository

```bash
uv run python -m student index data/raw/vllm-0.10.1
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

### Generate an answer

```bash
uv run python -m student answer "What is PagedAttention?" --k 10
```

### Evaluate retrieval quality

```bash
uv run python -m student evaluate \
    data/output/search_results/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10
```

uv run python -m student answer_dataset
--student_search_results_path data/output/search_results/dataset_docs_public.json
--save_directory data/output/search_results_and_answer

### Run the full evaluation pipeline

```bash
make evaluate
```

