import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag.bm25 import BM25Index, tokenize  # noqa: E402


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Zebra migration, patterns!") == ["zebra", "migration", "patterns"]


def test_postings_and_doc_frequency():
    index = BM25Index.build(
        doc_ids=["d0", "d1", "d2"],
        texts=["zebra migration patterns", "penguin videos", "zebra videos"],
    )

    assert sorted(index.postings["zebra"]) == [(0, 1), (2, 1)]
    assert index.doc_freq["zebra"] == 2
    assert index.doc_freq["videos"] == 2
    assert index.doc_freq["penguin"] == 1
    assert index.doc_lengths == [3, 2, 2]
    assert index.avg_doc_length == (3 + 2 + 2) / 3


def test_search_ranks_doc_with_more_query_term_matches_first():
    index = BM25Index.build(
        doc_ids=["d0", "d1", "d2", "d3", "d4"],
        texts=[
            "zebra migration patterns",
            "penguin swims in a pond",
            "zebra visits a new forest trail",
            "giraffes eat leaves happily",
            "lion roars at dawn",
        ],
    )

    # d0 matches both query terms, d2 only "zebra" -> d0 must outscore d2.
    hits = index.search("zebra migration", top_k=2)
    assert [doc_id for doc_id, _ in hits] == ["d0", "d2"]


def test_term_absent_from_corpus_does_not_crash_and_contributes_nothing():
    index = BM25Index.build(doc_ids=["d0"], texts=["zebra migration"])
    assert index.search("giraffe", top_k=5) == []


def test_empty_query_returns_no_hits():
    index = BM25Index.build(doc_ids=["d0", "d1"], texts=["zebra migration", "penguin videos"])
    assert index.search("", top_k=5) == []


def test_common_term_gets_epsilon_floored_nonnegative_idf():
    # "the" appears in every doc, so its raw idf (log((n-df+0.5)/(df+0.5))) is
    # negative; rank_bm25 floors it at epsilon * average_idf instead of letting it
    # subtract from the score, and we match that behavior.
    index = BM25Index.build(
        doc_ids=["d0", "d1", "d2"],
        texts=["the zebra runs", "the penguin swims", "the giraffe walks"],
    )
    assert index.idf["the"] >= 0
    assert index.idf["the"] < index.idf["zebra"]
