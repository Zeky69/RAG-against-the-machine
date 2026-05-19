import json
from pathlib import Path
from typing import List

import bm25s
from tqdm import tqdm

from .chunker import Chunk, chunk_file

CHUNKS_PATH = Path("data/processed/chunks")
INDEX_PATH = Path("data/processed/bm25_index")


def build_index(repo_path: str, max_chunk_size: int = 2000) -> None:
    CHUNKS_PATH.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Chunk] = []
    files = [
        f for f in Path(repo_path).rglob("*")
        if f.is_file()
    ]

    for f in tqdm(files, desc="Chunking files"):
        all_chunks.extend(chunk_file(f.as_posix(), max_size=max_chunk_size))

    meta = [
        {
            "file_path": c.file_path,
            "first_character_index": c.first_character_index,
            "last_character_index": c.last_character_index,
        }
        for c in all_chunks
    ]
    with open(CHUNKS_PATH / "chunks_meta.json", "w") as fp:
        json.dump(meta, fp)

    corpus = [c.content for c in all_chunks]
    tokenized = bm25s.tokenize(corpus)
    retriever = bm25s.BM25()
    retriever.index(tokenized)
    retriever.save(str(INDEX_PATH))

    print(f"Ingestion complete! Indices saved under data/processed/")
    print(f"Total chunks: {len(all_chunks)}")