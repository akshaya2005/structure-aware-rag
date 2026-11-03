"""FastAPI app that serves embeddings from an Embedder.

Endpoints:
  - GET  /health  liveness + which model is loaded.
  - POST /embed   {"texts": [...]} -> {"embeddings": [[...], ...], "dim": 384}
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.embedding import Embedder, SentenceTransformerEmbedder

MAX_TEXTS_PER_REQUEST = 256


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class EmbedResponse(BaseModel):
    model: str
    dim: int
    embeddings: list[list[float]]


def create_app(embedder: Embedder | None = None) -> FastAPI:
    """Build the app. Pass a custom embedder (e.g. a fake) for tests."""
    embedder = embedder or SentenceTransformerEmbedder()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Warm the model at startup so the first request doesn't pay the load cost.
        if hasattr(embedder, "load"):
            embedder.load()
        yield

    app = FastAPI(title="structure-aware-rag embeddings", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": embedder.model_name, "dim": embedder.dim}

    @app.post("/embed", response_model=EmbedResponse)
    def embed(req: EmbedRequest) -> EmbedResponse:
        if len(req.texts) > MAX_TEXTS_PER_REQUEST:
            raise HTTPException(
                status_code=413,
                detail=f"at most {MAX_TEXTS_PER_REQUEST} texts per request",
            )
        # Sync handler: FastAPI runs it in a threadpool, so encoding doesn't block the event loop.
        vectors = embedder.embed(req.texts)
        return EmbedResponse(model=embedder.model_name, dim=embedder.dim, embeddings=vectors.tolist())

    return app
