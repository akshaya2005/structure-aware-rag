import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag.serving import MAX_TEXTS_PER_REQUEST, create_app  # noqa: E402


class FakeEmbedder:
    """Deterministic stand-in so tests don't download the real model."""

    model_name = "fake-minilm"
    dim = 384

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([[float(len(t))] * self.dim for t in texts], dtype=np.float32)


def make_client() -> TestClient:
    return TestClient(create_app(FakeEmbedder()))


def test_health_reports_model_and_dim():
    resp = make_client().get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "fake-minilm", "dim": 384}


def test_embed_returns_one_vector_per_text_in_order():
    resp = make_client().post("/embed", json={"texts": ["a", "abc"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["dim"] == 384
    assert len(body["embeddings"]) == 2
    assert body["embeddings"][0][0] == 1.0
    assert body["embeddings"][1][0] == 3.0


def test_embed_rejects_empty_list():
    resp = make_client().post("/embed", json={"texts": []})
    assert resp.status_code == 422


def test_embed_rejects_oversized_batch():
    resp = make_client().post("/embed", json={"texts": ["x"] * (MAX_TEXTS_PER_REQUEST + 1)})
    assert resp.status_code == 413
