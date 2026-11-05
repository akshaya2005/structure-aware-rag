"""Tests for embedding/indexing and its evaluation adapters."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from rag.indexing import FaissIndex, embedding_text
from rag.chunking import Chunk, RegexTokenizer


@pytest.mark.parametrize("kind", ["flat", "hnsw"])
def test_cosine_search_and_roundtrip(kind, tmp_path):
    index = FaissIndex(["a", "b", "c"], np.array([[10, 0], [1, 1], [0, 2]]), kind)
    hits = index.search(np.array([[5, 0]]), 20)[0]
    assert [doc for doc, _ in hits] == ["a", "b", "c"]
    assert [s for _, s in hits] == pytest.approx([1, 2**-0.5, 0])
    index.save(tmp_path)
    assert FaissIndex.load(tmp_path).search([[1, 0]], 3) == [hits]
    assert index.search([[1, 0]], 0) == [[]]


def test_invalid_vectors():
    with pytest.raises(ValueError):
        FaissIndex(["a"], [[0, 0]])
    with pytest.raises(ValueError):
        FaissIndex(["a", "a"], [[1, 0], [0, 1]])


def test_context_keeps_evidence():
    c = Chunk("p", "c", "paragraph", "Results", ("u",), "all evidence stays intact", 4)
    text = embedding_text(c, {"p": "A very long paper title"}, RegexTokenizer(), max_tokens=6)
    assert text.endswith(c.text)
    assert len(RegexTokenizer().token_spans(text)) <= 6


def test_flat_matches_numpy_and_hnsw_recovers_neighbors():
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(200, 16)).astype(np.float32)
    queries = rng.normal(size=(4, 16)).astype(np.float32)
    from rag.indexing import normalized
    expected = np.argsort(-(normalized(queries) @ normalized(vectors).T), axis=1)[:, :10]
    ids = [str(i) for i in range(len(vectors))]
    for kind in ("flat", "hnsw"):
        index = FaissIndex(ids, vectors, kind=kind, ef_search=200)
        actual = [[int(cid) for cid, _ in hits] for hits in index.search(queries, 10)]
        np.testing.assert_array_equal(actual, expected)


def test_scope_and_retriever_persistence(tmp_path):
    from rag.indexing import FaissRetriever
    from rag.benchmark import EvalQuestion
    chunks = [Chunk(p, p, "paragraph", "S", (p,), p, 1) for p in ["a", "b"]]
    class Embedder:
        def embed(self, texts):
            return np.array([[0, 1]] * len(texts), dtype=np.float32)
    question = EvalQuestion("a", "q", "test", frozenset({"a"}), "unknown", "text")
    vectors = np.eye(2, dtype=np.float32)
    for scope, expected in [("paper", "a"), ("corpus", "b")]:
        retriever = FaissRetriever(Embedder(), chunks, vectors, scope=scope)
        retriever.save(tmp_path / scope)
        loaded = FaissRetriever.load(tmp_path / scope, Embedder())
        assert loaded.retrieve_batch([question], 1)[0][0].chunk_id == expected
        assert loaded.retrieve_batch([], 3) == []


def test_embedding_cache_invalidation(tmp_path):
    from rag.embedding import cached_embeddings
    class Embedder:
        model_name = "test"
        calls = 0
        def embed(self, texts):
            self.calls += 1
            return np.array([[len(t), 1] for t in texts], dtype=np.float32)
    embedder = Embedder()
    first = cached_embeddings(embedder, ["one", "two"], tmp_path)
    np.testing.assert_array_equal(first, cached_embeddings(embedder, ["one", "two"], tmp_path))
    assert embedder.calls == 1
    cached_embeddings(embedder, ["two", "one"], tmp_path)
    assert embedder.calls == 2
    embedder.model_name = "another"
    cached_embeddings(embedder, ["two", "one"], tmp_path)
    assert embedder.calls == 3


from rag.answer_evaluation import answer_f1, evaluate_answers


def test_official_normalization_and_special_answers():
    assert answer_f1("The Red, fox", "red fox") == 1
    assert answer_f1("red", "red fox") == 2 / 3
    assert answer_f1("", "") == 0
    assert answer_f1("yes", "yes indeed", hotpot=True) == 0
    assert answer_f1("Unanswerable", "unanswerable") == 1


def test_qasper_multiple_references_and_missing_predictions():
    data = {"p": {"qas": [
        {"question_id": "q1", "answers": [{"answer": {"unanswerable": False,
            "extractive_spans": ["red", "fox"]}}, {"answer": {"unanswerable": False,
            "free_form_answer": "a fox"}}]},
        {"question_id": "q2", "answers": [{"answer": {"unanswerable": True}}]}
    ]}}
    report = evaluate_answers(data, {"q1": "fox"})
    assert report["answer_f1"] == 0.5
    assert report["missing_predictions"] == 1
    subset = evaluate_answers(data, {"q2": "unanswerable"}, question_ids=["q2"])
    assert subset["answer_f1"] == 1


from rag.hotpot import build_hotpot_corpus, unit_id
from rag.chunking import RegexTokenizer, ChunkerConfig
from rag.benchmark import score_question


def test_two_article_support_provenance():
    data = [{"_id": "q", "question": "red fox blue bird", "answer": "fox",
             "context": [["Fox", ["red fox", "in forest"]], ["Bird", ["blue bird"]]],
             "supporting_facts": [["Fox", 0], ["Bird", 0]], "type": "bridge"}]
    corpus = build_hotpot_corpus(data, RegexTokenizer(), ChunkerConfig(5, 1))
    assert corpus.questions[0].gold_unit_ids == {unit_id("q", "Fox", 0), unit_id("q", "Bird", 0)}
    summary = score_question(corpus.chunks, corpus.questions[0].gold_unit_ids, ks=(5,))
    assert summary["recall@5"] == 1
