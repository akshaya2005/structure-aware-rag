import re
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag.benchmark import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    EvalQuestion,
    RandomRetriever,
    breakdown,
    build_corpus,
    classify_evidence_type,
    classify_question_type,
    evaluate,
    score_question,
)
from rag.chunking import Chunk, ChunkerConfig, RegexTokenizer  # noqa: E402

TOK = RegexTokenizer()
# Small chunks so each short paragraph below lands in its own chunk.
CONFIG = ChunkerConfig(chunk_size=6, overlap=1)


class KeywordEmbedder:
    """Hashed bag-of-words: texts sharing words get high cosine similarity."""

    model_name = "keyword"
    dim = 256

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                out[row, zlib.crc32(word.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def answer(evidence, unanswerable=False, yes_no=None, extractive_spans=None, free_form_answer=""):
    return {
        "answer": {
            "unanswerable": unanswerable,
            "evidence": evidence,
            "yes_no": yes_no,
            "extractive_spans": extractive_spans or [],
            "free_form_answer": free_form_answer,
        }
    }


def qa(qid, question, answers):
    return {"question_id": qid, "question": question, "answers": answers}


DATA = {
    "p1": {
        "full_text": [
            {"section_name": "Intro", "paragraphs": ["we study zebra migration patterns"]},
            {"section_name": "Data", "paragraphs": ["the corpus has penguin videos"]},
        ],
        "figures_and_tables": [{"caption": "Table 1: Data Statistics."}],
        "qas": [
            qa("q_zebra", "what do zebra migration patterns show",
               [answer(["we study zebra migration patterns"],
                       extractive_spans=["zebra migration patterns"])]),
            qa("q_union", "which data",
               [answer(["the corpus has penguin videos"], free_form_answer="penguin video data"),
                answer(["FLOAT SELECTED: Table 1: Data Statistics."], yes_no=True)]),
            qa("q_unans", "unanswerable?", [answer([], unanswerable=True)]),
            qa("q_noev", "no evidence?", [answer([])]),
            qa("q_miss", "unmatched?", [answer(["text that is not in the paper"])]),
        ],
    },
    "p2": {
        "full_text": [{"section_name": "Intro", "paragraphs": ["zebra migration patterns again"]}],
        "figures_and_tables": [],
        "qas": [],
    },
}


def test_build_corpus_labels_and_skips_questions():
    corpus = build_corpus(DATA, TOK, CONFIG)
    by_id = {q.question_id: q for q in corpus.questions}

    assert set(by_id) == {"q_zebra", "q_union"}
    assert by_id["q_zebra"].gold_unit_ids == {"p1::para::Intro::0"}
    # Evidence is unioned across annotators, including FLOAT SELECTED captions.
    assert by_id["q_union"].gold_unit_ids == {"p1::para::Data::0", "p1::caption::0"}

    assert by_id["q_zebra"].question_type == "extractive"
    assert by_id["q_zebra"].evidence_type == "text"
    # q_union's annotators disagree on type (abstractive vs yes_no) and evidence
    # type (text vs FLOAT SELECTED table), so both classify as "mixed".
    assert by_id["q_union"].question_type == "mixed"
    assert by_id["q_union"].evidence_type == "mixed"

    assert corpus.stats["questions_total"] == 5
    assert corpus.stats["skipped_unanswerable"] == 1
    assert corpus.stats["skipped_no_evidence"] == 1
    assert corpus.stats["skipped_unmatched_evidence"] == 1
    assert corpus.stats["evidence_strings_unmatched"] == 1


def make_chunk(cid, unit_ids):
    return Chunk("p", cid, "paragraph", "S", tuple(unit_ids), "", 0)


def test_score_question_metrics():
    ranked = [make_chunk("c0", ["u9"]), make_chunk("c1", ["u1", "u2"]), make_chunk("c2", ["u3"])]
    gold = frozenset({"u1", "u3", "u4"})
    scores = score_question(ranked, gold, ks=(1, 3))

    assert scores["mrr"] == 0.5
    assert scores["hit@1"] == 0.0
    assert scores["hit@3"] == 1.0
    assert scores["recall@3"] == 2 / 3
    assert scores["precision@3"] == 2 / 3

    # total_relevant defaults to the 2 relevant chunks already in `ranked` (c1, c2);
    # the ideal top-3 ranking puts both of them first, but they actually land at
    # ranks 2 and 3, so ndcg@3 is less than 1.
    assert scores["ndcg@1"] == 0.0
    assert round(scores["ndcg@3"], 6) == 0.693426


def test_dense_retriever_ranks_matching_chunk_first_and_stays_in_paper():
    corpus = build_corpus(DATA, TOK, CONFIG)
    retriever = DenseRetriever(KeywordEmbedder(), corpus.chunks, scope="paper")
    summary, rows = evaluate(retriever, corpus.questions, ks=(1, 3))

    zebra = next(r for r in rows if r["question_id"] == "q_zebra")
    assert zebra["scores"]["hit@1"] == 1.0
    assert all(cid.startswith("p1::") for r in rows for cid in r["retrieved_chunk_ids"])
    assert set(summary) == {
        "mrr",
        "hit@1", "recall@1", "precision@1", "ndcg@1",
        "hit@3", "recall@3", "precision@3", "ndcg@3",
    }


def test_corpus_scope_can_retrieve_from_other_papers():
    corpus = build_corpus(DATA, TOK, CONFIG)
    retriever = DenseRetriever(KeywordEmbedder(), corpus.chunks, scope="corpus")
    _, rows = evaluate(retriever, corpus.questions, ks=(5,))

    zebra = next(r for r in rows if r["question_id"] == "q_zebra")
    assert any(cid.startswith("p2::") for cid in zebra["retrieved_chunk_ids"])


def test_bm25_retriever_ranks_matching_chunk_first_and_stays_in_paper():
    corpus = build_corpus(DATA, TOK, CONFIG)
    retriever = BM25Retriever(corpus.chunks, scope="paper")
    summary, rows = evaluate(retriever, corpus.questions, ks=(1, 3))

    zebra = next(r for r in rows if r["question_id"] == "q_zebra")
    assert zebra["scores"]["hit@1"] == 1.0
    assert all(cid.startswith("p1::") for r in rows for cid in r["retrieved_chunk_ids"])
    assert set(summary) == {
        "mrr",
        "hit@1", "recall@1", "precision@1", "ndcg@1",
        "hit@3", "recall@3", "precision@3", "ndcg@3",
    }


def test_bm25_corpus_scope_can_retrieve_from_other_papers():
    corpus = build_corpus(DATA, TOK, CONFIG)
    retriever = BM25Retriever(corpus.chunks, scope="corpus")

    # "again" only occurs in p2's chunk; a corpus-scoped index must reach it even
    # though the question is (falsely, for this test) attributed to p1.
    query = EvalQuestion(
        paper_id="p1", question_id="q", question="again",
        gold_unit_ids=frozenset(), question_type="unknown", evidence_type="text",
    )
    [ranked] = retriever.retrieve_batch([query], k=1)

    assert ranked[0].paper_id == "p2"


def test_random_retriever_is_seeded_and_paper_scoped():
    corpus = build_corpus(DATA, TOK, CONFIG)
    _, rows_a = evaluate(RandomRetriever(corpus.chunks, seed=7), corpus.questions, ks=(3,))
    _, rows_b = evaluate(RandomRetriever(corpus.chunks, seed=7), corpus.questions, ks=(3,))

    assert [r["retrieved_chunk_ids"] for r in rows_a] == [r["retrieved_chunk_ids"] for r in rows_b]
    assert all(cid.startswith("p1::") for r in rows_a for cid in r["retrieved_chunk_ids"])


def test_classify_question_type_flags_disagreement_as_mixed():
    assert classify_question_type([{"extractive_spans": ["x"], "yes_no": None, "free_form_answer": ""}]) == "extractive"
    assert classify_question_type([{"yes_no": True, "extractive_spans": [], "free_form_answer": ""}]) == "yes_no"
    assert classify_question_type([{"free_form_answer": "a summary", "yes_no": None, "extractive_spans": []}]) == "abstractive"
    assert classify_question_type([
        {"extractive_spans": ["x"], "yes_no": None, "free_form_answer": ""},
        {"free_form_answer": "a summary", "yes_no": None, "extractive_spans": []},
    ]) == "mixed"
    assert classify_question_type([{"yes_no": None, "extractive_spans": [], "free_form_answer": ""}]) == "unknown"


def test_classify_evidence_type_flags_table_text_and_mixed():
    assert classify_evidence_type(["plain paragraph text"]) == "text"
    assert classify_evidence_type(["FLOAT SELECTED: Table 1: Stats."]) == "table"
    assert classify_evidence_type(["plain paragraph text", "FLOAT SELECTED: Table 1: Stats."]) == "mixed"


def test_breakdown_groups_rows_and_averages_scores():
    corpus = build_corpus(DATA, TOK, CONFIG)
    _, rows = evaluate(BM25Retriever(corpus.chunks, scope="paper"), corpus.questions, ks=(1,))

    by_type = breakdown(rows, "question_type")
    assert by_type["extractive"]["n"] == 1  # q_zebra
    assert by_type["mixed"]["n"] == 1  # q_union
    assert set(by_type["extractive"]) == {"n", "mrr", "hit@1", "recall@1", "precision@1", "ndcg@1"}
