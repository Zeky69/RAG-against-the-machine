import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class Chunk:
    file_path: str
    first_character_index: int
    last_character_index: int
    content: str


def _build_offsets(content: str) -> List[int]:
    offsets = [0]
    for line in content.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _node_span(
    node: ast.stmt,
    offsets: List[int],
    content_len: int,
) -> Tuple[int, int]:
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        first_line = decorators[0].lineno
    else:
        first_line = node.lineno
    start = offsets[first_line - 1]
    end_line = node.end_lineno or node.lineno
    end = offsets[min(end_line, len(offsets) - 1)]
    return start, min(end, content_len)


def _raw_chunks(
        file_path: str,
        text: str,
        offset: int,
        max_size: int) -> List[Chunk]:
    result = []
    for i in range(0, len(text), max_size):
        sub = text[i:i + max_size]
        result.append(Chunk(file_path, offset + i, offset + i + len(sub), sub))
    return result


def chunk_python(file_path: str, content: str, max_size: int) -> List[Chunk]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return chunk_text(file_path, content, max_size)

    offsets = _build_offsets(content)
    content_len = len(content)

    def span(node: ast.stmt) -> Tuple[int, int]:
        return _node_span(node, offsets, content_len)

    chunks: List[Chunk] = []
    # Accumulate consecutive non-def module-level nodes (imports, constants,
    # etc.)
    prose_start: Optional[int] = None
    prose_end: int = 0

    def flush_prose() -> None:
        nonlocal prose_start, prose_end
        if prose_start is not None and prose_end > prose_start:
            text = content[prose_start:prose_end]
            if text.strip():
                chunks.extend(
                    _raw_chunks(
                        file_path,
                        text,
                        prose_start,
                        max_size))
        prose_start = None
        prose_end = 0

    def add_def(start: int, end: int) -> None:
        text = content[start:end]
        if not text.strip():
            return
        if len(text) <= max_size:
            chunks.append(Chunk(file_path, start, end, text))
        else:
            chunks.extend(_raw_chunks(file_path, text, start, max_size))

    for node in tree.body:
        start, end = span(node)

        if not isinstance(
            node,
            (ast.FunctionDef,
             ast.AsyncFunctionDef,
             ast.ClassDef)):
            # Module-level code: accumulate into a prose block
            if prose_start is None:
                prose_start = start
            prose_end = end
            continue

        flush_prose()

        node_text = content[start:end]
        if len(node_text) <= max_size:
            if node_text.strip():
                chunks.append(Chunk(file_path, start, end, node_text))
        elif isinstance(node, ast.ClassDef):
            # Class too large: keep class header + each method as separate
            # chunks
            methods = [
                span(child)
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if methods:
                # Class header = from class start to first method
                add_def(start, methods[0][0])
                for ms, me in methods:
                    add_def(ms, me)
            else:
                # No methods (e.g. dataclass with only fields): split as raw
                # text
                chunks.extend(
                    _raw_chunks(
                        file_path,
                        node_text,
                        start,
                        max_size))
        else:
            # Large standalone function: split as raw text
            chunks.extend(_raw_chunks(file_path, node_text, start, max_size))

    flush_prose()

    return chunks if chunks else chunk_text(file_path, content, max_size)


def chunk_text(file_path: str, content: str, max_size: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    paragraphs = content.split("\n\n")
    current = ""
    current_start = 0
    offset = 0

    for para in paragraphs:
        if current and len(current) + 2 + len(para) > max_size:
            chunks.append(
                Chunk(
                    file_path,
                    current_start,
                    current_start +
                    len(current),
                    current))
            current_start = offset
            current = ""

        if not current and len(para) > max_size:
            for i in range(0, len(para), max_size):
                sub = para[i:i + max_size]
                chunks.append(
                    Chunk(
                        file_path,
                        offset +
                        i,
                        offset +
                        i +
                        len(sub),
                        sub))
            current_start = offset + len(para) + 2
        else:
            current += ("\n\n" if current else "") + para

        offset += len(para) + 2

    if current:
        chunks.append(
            Chunk(
                file_path,
                current_start,
                current_start +
                len(current),
                current))
    return chunks


def chunk_file(file_path: str, max_size: int = 2000) -> List[Chunk]:
    path = Path(file_path)
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    if path.suffix == ".py":
        return chunk_python(file_path, content, max_size)
    return chunk_text(file_path, content, max_size)
