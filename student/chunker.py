from dataclasses import dataclass
from pathlib import Path
from typing import List

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    file_path: str
    first_character_index: int
    last_character_index: int
    content: str


def _docs_to_chunks(file_path: str, docs: list) -> List[Chunk]:
    return [
        Chunk(
            file_path=file_path,
            first_character_index=doc.metadata["start_index"],
            last_character_index=doc.metadata["start_index"] + len(doc.page_content),
            content=doc.page_content,
        )
        for doc in docs
    ]


def chunk_python(file_path: str, content: str, max_size: int) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=max_size,
        chunk_overlap=0,
        add_start_index=True,
    )
    return _docs_to_chunks(file_path, splitter.create_documents([content]))


def chunk_text(file_path: str, content: str, max_size: int) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=0,
        add_start_index=True,
    )
    return _docs_to_chunks(file_path, splitter.create_documents([content]))


def chunk_markdown(file_path: str, content: str, max_size: int) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=max_size,
        chunk_overlap=0,
        add_start_index=True,
    )
    return _docs_to_chunks(file_path, splitter.create_documents([content]))


def chunk_file(file_path: str, max_size: int = 2000) -> List[Chunk]:
    path = Path(file_path)
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    if path.suffix == ".py":
        return chunk_python(file_path, content, max_size)
    if path.suffix == ".md":
        return chunk_markdown(file_path, content, max_size)
    return chunk_text(file_path, content, max_size)
