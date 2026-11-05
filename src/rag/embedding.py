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
    revision: str | None = None


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

        self._model = SentenceTransformer(
            self.config.model_name, device=self.config.device, revision=self.config.revision
        )
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        self.load()
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def cached_embeddings(embedder: Embedder, texts: list[str], directory, *, cache_tag=""):
    """Content-addressed cache; order, model config, and text changes invalidate it."""
    import hashlib
    import json
    from dataclasses import asdict, is_dataclass
    from pathlib import Path
    import uuid

    config = getattr(embedder, "config", None)
    identity = {"model": embedder.model_name, "tag": cache_tag,
                "config": asdict(config) if is_dataclass(config) else None, "texts": texts}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(directory) / f"{key}.npy"
    if path.exists():
        vectors = np.load(path, allow_pickle=False)
    else:
        vectors = np.asarray(embedder.embed(texts), dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
            raise ValueError("invalid embedding shape or values")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp.npy")
        np.save(temporary, vectors, allow_pickle=False)
        temporary.replace(path)
    if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
        raise ValueError("invalid cached embedding shape or values")
    return vectors
