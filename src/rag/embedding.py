"""Thin wrapper around sentence-transformers/all-MiniLM-L6-v2.

The model produces 384-dim embeddings and silently truncates inputs longer than
256 word pieces, so long paragraphs should be chunked upstream before embedding.
Embeddings are L2-normalized by default so dot product == cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_SEQ_LENGTH = 256


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class EmbedderConfig:
    model_name: str = DEFAULT_MODEL_NAME
    device: str | None = None  # None lets sentence-transformers pick cuda/mps/cpu
    batch_size: int = 64
    normalize: bool = True


class SentenceTransformerEmbedder:
    """Loads the model lazily on first use so imports stay cheap."""

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        self.config = config or EmbedderConfig()
        self.model_name = self.config.model_name
        self.dim = EMBEDDING_DIM
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.config.model_name, device=self.config.device)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        self.load()
        return self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
