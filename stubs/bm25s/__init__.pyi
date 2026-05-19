from typing import Any, Sequence, Tuple

import numpy as np


class Tokenized:
    ...


def tokenize(
    corpus: Sequence[str],
    show_progress: bool = ...,
    **kwargs: Any,
) -> Tokenized: ...


class BM25:
    def __init__(self, **kwargs: Any) -> None: ...

    def index(self, tokens: Tokenized) -> None: ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str, load_corpus: bool = ...) -> "BM25": ...

    def retrieve(
        self,
        tokens: Tokenized,
        k: int = ...,
        show_progress: bool = ...,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]: ...
