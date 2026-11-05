"""FAISS cosine indexes with stable chunk IDs and explicit retrieval scope."""
from collections import defaultdict
from pathlib import Path
import json

import numpy as np


def normalized(matrix):
    matrix = np.array(matrix, dtype=np.float32, order="C", copy=True)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("embeddings must be a finite two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero embeddings have no cosine similarity")
    return matrix / norms


def embedding_text(chunk, titles, tokenizer, max_tokens=254, *, contextual=True):
    """Budget title prefixes without truncating evidence text."""
    body_tokens = len(tokenizer.token_spans(chunk.text))
    if body_tokens > max_tokens:
        raise ValueError("chunk exceeds embedding token budget; reduce chunk size")
    if not contextual:
        return chunk.text
    prefix = f"{titles.get(chunk.paper_id, '')}\n{chunk.section_name or ''}".strip()
    spans = tokenizer.token_spans(prefix)
    for count in range(min(len(spans), max_tokens - body_tokens), -1, -1):
        text = (prefix[:spans[count - 1][1]] + "\n" if count else "") + chunk.text
        if len(tokenizer.token_spans(text)) <= max_tokens:
            return text
    return chunk.text


class FaissIndex:
    def __init__(self, doc_ids, embeddings, kind="flat", m=32, ef_construction=200, ef_search=64):
        import faiss
        if kind not in ("flat", "hnsw"):
            raise ValueError("kind must be flat or hnsw")
        vectors = normalized(embeddings)
        self.doc_ids = list(doc_ids)
        if len(self.doc_ids) != len(vectors) or len(set(self.doc_ids)) != len(self.doc_ids):
            raise ValueError("one unique document ID is required per vector")
        if vectors.shape[1] == 0 or min(m, ef_construction, ef_search) <= 0:
            raise ValueError("dimensions and HNSW parameters must be positive")
        self.index = (faiss.IndexFlatIP(vectors.shape[1]) if kind == "flat" else
                      faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT))
        if kind == "hnsw":
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
        self.index.add(vectors)

    def search(self, queries, k):
        if k < 0:
            raise ValueError("k must be nonnegative")
        queries = normalized(queries)
        if queries.shape[1] != self.index.d:
            raise ValueError("query dimension differs from index")
        if k == 0 or not self.doc_ids:
            return [[] for _ in queries]
        scores, indices = self.index.search(queries, min(k, len(self.doc_ids)))
        return [[(self.doc_ids[i], float(s)) for i, s in zip(row, values) if i >= 0]
                for row, values in zip(indices, scores)]

    def save(self, directory):
        import faiss
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        (path / "doc_ids.json").write_text(json.dumps(self.doc_ids), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        import faiss
        path = Path(directory)
        obj = cls.__new__(cls)
        obj.index = faiss.read_index(str(path / "index.faiss"))
        obj.doc_ids = json.loads((path / "doc_ids.json").read_text(encoding="utf-8"))
        if len(obj.doc_ids) != obj.index.ntotal or len(set(obj.doc_ids)) != len(obj.doc_ids):
            raise ValueError("index metadata does not match vectors")
        return obj


class FaissRetriever:
    """Build separate indexes per paper to avoid filtering away valid neighbors."""
    def __init__(self, embedder, chunks, embeddings, kind="flat", scope="paper", **kwargs):
        if scope not in ("paper", "corpus"):
            raise ValueError("scope must be paper or corpus")
        if len(chunks) != len(embeddings):
            raise ValueError("chunk and embedding counts differ")
        self.name = f"faiss_{kind}"
        self.scope, self.chunks, self.embedder = scope, chunks, embedder
        self.by_id = {c.chunk_id: c for c in chunks}
        if len(self.by_id) != len(chunks):
            raise ValueError("chunk IDs must be unique")
        groups = defaultdict(list)
        for i, chunk in enumerate(chunks):
            groups[chunk.paper_id if scope == "paper" else None].append(i)
        self.indices = {key: FaissIndex([chunks[i].chunk_id for i in ids],
                                       embeddings[ids], kind=kind, **kwargs)
                        for key, ids in groups.items()}

    def retrieve_batch(self, questions, k):
        if not questions:
            return []
        vectors = self.embedder.embed([q.question for q in questions])
        results = []
        for q, vector in zip(questions, vectors):
            index = self.indices.get(q.paper_id if self.scope == "paper" else None)
            hits = index.search(vector[None, :], k)[0] if index else []
            results.append([self.by_id[doc_id] for doc_id, _ in hits])
        return results

    def save(self, directory):
        """Store indexes, chunk provenance, and scope together for serving."""
        from dataclasses import asdict
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        groups = []
        for i, (key, index) in enumerate(self.indices.items()):
            index.save(path / str(i))
            groups.append({"key": key, "directory": str(i)})
        (path / "manifest.json").write_text(json.dumps({
            "scope": self.scope, "name": self.name,
            "chunks": [asdict(c) for c in self.chunks], "groups": groups}), encoding="utf-8")

    @classmethod
    def load(cls, directory, embedder):
        from rag.chunking import Chunk
        path = Path(directory)
        meta = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        obj = cls.__new__(cls)
        obj.scope, obj.name, obj.embedder = meta["scope"], meta["name"], embedder
        obj.chunks = [Chunk(**{**c, "unit_ids": tuple(c["unit_ids"])}) for c in meta["chunks"]]
        obj.by_id = {c.chunk_id: c for c in obj.chunks}
        obj.indices = {g["key"]: FaissIndex.load(path / g["directory"]) for g in meta["groups"]}
        indexed = [cid for index in obj.indices.values() for cid in index.doc_ids]
        if len(indexed) != len(obj.chunks) or set(indexed) != set(obj.by_id):
            raise ValueError("saved index chunk mapping is inconsistent")
        return obj
