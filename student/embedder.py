import os
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH = 256
MAX_SEQ_LENGTH = 128


class Embedder:
    def __init__(self) -> None:
        torch.set_num_threads(os.cpu_count() or 4)
        self._model = SentenceTransformer(EMBED_MODEL_ID, device="cpu")
        self._model.max_seq_length = MAX_SEQ_LENGTH

    def encode_corpus(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=EMBED_BATCH,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)[0]
