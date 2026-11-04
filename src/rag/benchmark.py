"""Step 2: retrieval test bench on QASPER.

Pipeline: raw QASPER split -> parse_paper -> chunk_paper -> retrieve top-k chunks per
question -> score against gold evidence.

Gold labels: a question's gold units are the union of evidence over all of its
answerable annotations, mapped to unit_ids by exact text (FLOAT SELECTED evidence maps
to caption units). A question is skipped if every annotation is unanswerable, if no
annotation cites evidence, or if none of its evidence matches a parsed unit.

Retrieval is scoped to the question's own paper by default, matching QASPER's setup
where each question is asked about a known paper; scope="corpus" searches every chunk.

Metrics (a chunk is relevant iff it contains >= 1 gold unit):
  - hit@k:       1 if any of the top-k chunks is relevant.
  - recall@k:    fraction of gold units covered by the union of the top-k chunks.
  - precision@k: fraction of the top-k chunks that are relevant.
  - ndcg@k:      binary-relevance nDCG; the ideal ranking puts min(k, total relevant
                 chunks in scope) relevant chunks first, where "scope" is the
                 question's paper (or the whole corpus, for scope="corpus").
  - mrr:         reciprocal rank of the first relevant chunk within the top max(ks).
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rag.bm25 import DEFAULT_B, DEFAULT_K1, BM25Index
from rag.chunking import (
    FLOAT_SELECTED_PREFIX,
    Chunk,
    ChunkerConfig,
    Tokenizer,
    chunk_paper,
    evidence_to_unit_ids,
    is_relevant,
)
from rag.embedding import Embedder
from rag.parsing import parse_paper

DEFAULT_KS = (1, 3, 5, 10, 20)


def classify_question_type(answers: list[dict]) -> str:
    """Classify a question's answerable annotations as extractive/abstractive/yes_no.

    QASPER lets each annotator answer differently (e.g. one gives an extractive span,
    another a free-form summary); "mixed" surfaces that disagreement instead of
    silently picking one annotator's type.
    """
    types = set()
    for a in answers:
        if a.get("yes_no") is not None:
            types.add("yes_no")
        elif a.get("extractive_spans"):
            types.add("extractive")
        elif a.get("free_form_answer", "").strip():
            types.add("abstractive")
    if len(types) == 1:
        return next(iter(types))
    return "mixed" if types else "unknown"


def classify_evidence_type(evidence: list[str]) -> str:
    """Classify a question's gold evidence as text/table based on FLOAT SELECTED."""
    has_table = any(ev.startswith(FLOAT_SELECTED_PREFIX) for ev in evidence)
    has_text = any(not ev.startswith(FLOAT_SELECTED_PREFIX) for ev in evidence)
    if has_table and has_text:
        return "mixed"
    return "table" if has_table else "text"


@dataclass(frozen=True)
class EvalQuestion:
    paper_id: str
    question_id: str
    question: str
    gold_unit_ids: frozenset[str]
    question_type: str
    evidence_type: str


@dataclass
class Corpus:
    chunks: list[Chunk]
    questions: list[EvalQuestion]
    stats: Counter


def build_corpus(data: dict, tokenizer: Tokenizer, config: ChunkerConfig | None = None) -> Corpus:
    """Parse, chunk and label one raw QASPER split ({paper_id: paper_dict})."""
    chunks: list[Chunk] = []
    questions: list[EvalQuestion] = []
    stats: Counter = Counter()

    for paper_id, paper in data.items():
        units = parse_paper(paper_id, paper)
        unit_ids = [u.unit_id for u in units]
        if len(unit_ids) != len(set(unit_ids)):
            # Colliding unit_ids make relevance labels ambiguous for this paper.
            stats["papers_with_duplicate_unit_ids"] += 1
        chunks.extend(chunk_paper(units, tokenizer, config))
        stats["papers"] += 1

        for qa in paper["qas"]:
            stats["questions_total"] += 1
            answers = [a["answer"] for a in qa["answers"] if not a["answer"]["unanswerable"]]
            if not answers:
                stats["skipped_unanswerable"] += 1
                continue

            evidence = sorted({ev for a in answers for ev in a["evidence"]})
            if not evidence:
                stats["skipped_no_evidence"] += 1
                continue

            unmatched = [ev for ev in evidence if not evidence_to_unit_ids([ev], units)]
            stats["evidence_strings"] += len(evidence)
            stats["evidence_strings_unmatched"] += len(unmatched)

            gold = evidence_to_unit_ids(evidence, units)
            if not gold:
                stats["skipped_unmatched_evidence"] += 1
                continue

            questions.append(
                EvalQuestion(
                    paper_id=paper_id,
                    question_id=qa["question_id"],
                    question=qa["question"],
                    gold_unit_ids=frozenset(gold),
                    question_type=classify_question_type(answers),
                    evidence_type=classify_evidence_type(evidence),
                )
            )

    stats["questions_evaluated"] = len(questions)
    stats["chunks"] = len(chunks)
    return Corpus(chunks=chunks, questions=questions, stats=stats)


class Retriever(Protocol):
    name: str

    def retrieve_batch(self, questions: list[EvalQuestion], k: int) -> list[list[Chunk]]: ...


class DenseRetriever:
    """Embeds every chunk once, then ranks by dot product (cosine for normalized embedders)."""

    name = "dense"

    def __init__(self, embedder: Embedder, chunks: list[Chunk], scope: str = "paper") -> None:
        if scope not in ("paper", "corpus"):
            raise ValueError("scope must be 'paper' or 'corpus'")
        self.embedder = embedder
        self.chunks = chunks
        self.scope = scope
        self.matrix = embedder.embed([c.text for c in chunks])

        by_paper: dict[str, list[int]] = defaultdict(list)
        for i, chunk in enumerate(chunks):
            by_paper[chunk.paper_id].append(i)
        self._paper_idx = {pid: np.array(idx) for pid, idx in by_paper.items()}
        self._all_idx = np.arange(len(chunks))

    def retrieve_batch(self, questions: list[EvalQuestion], k: int) -> list[list[Chunk]]:
        query_vecs = self.embedder.embed([q.question for q in questions])
        results = []
        for q, qv in zip(questions, query_vecs):
            if self.scope == "paper":
                candidates = self._paper_idx.get(q.paper_id, np.array([], dtype=int))
            else:
                candidates = self._all_idx
            scores = self.matrix[candidates] @ qv
            top = candidates[np.argsort(-scores, kind="stable")[:k]]
            results.append([self.chunks[i] for i in top])
        return results


class BM25Retriever:
    """Ranks chunks by Okapi BM25 over an inverted index built from their text."""

    name = "bm25"

    def __init__(
        self, chunks: list[Chunk], scope: str = "paper", k1: float = DEFAULT_K1, b: float = DEFAULT_B
    ) -> None:
        if scope not in ("paper", "corpus"):
            raise ValueError("scope must be 'paper' or 'corpus'")
        self.chunks = chunks
        self.scope = scope
        self._by_chunk_id = {c.chunk_id: c for c in chunks}

        if scope == "corpus":
            by_scope: dict[str | None, list[Chunk]] = {None: chunks}
        else:
            by_scope = defaultdict(list)
            for c in chunks:
                by_scope[c.paper_id].append(c)

        self._indices = {
            key: BM25Index.build([c.chunk_id for c in group], [c.text for c in group], k1, b)
            for key, group in by_scope.items()
        }

    def retrieve_batch(self, questions: list[EvalQuestion], k: int) -> list[list[Chunk]]:
        results = []
        for q in questions:
            index = self._indices.get(None if self.scope == "corpus" else q.paper_id)
            hits = index.search(q.question, k) if index else []
            results.append([self._by_chunk_id[chunk_id] for chunk_id, _ in hits])
        return results


class RandomRetriever:
    """Sanity floor: k random chunks from the question's paper."""

    name = "random"
    scope = "paper"

    def __init__(self, chunks: list[Chunk], seed: int = 0) -> None:
        self.chunks = chunks
        self._by_paper: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            self._by_paper[chunk.paper_id].append(chunk)
        self._rng = random.Random(seed)

    def retrieve_batch(self, questions: list[EvalQuestion], k: int) -> list[list[Chunk]]:
        results = []
        for q in questions:
            pool = self._by_paper.get(q.paper_id, [])
            results.append(self._rng.sample(pool, min(k, len(pool))))
        return results


def _dcg(relevance_flags: list[bool]) -> float:
    return sum(1.0 / math.log2(i + 2) for i, rel in enumerate(relevance_flags) if rel)


def _ndcg_at_k(relevant: list[bool], total_relevant: int, k: int) -> float:
    ideal = _dcg([True] * min(k, total_relevant))
    return _dcg(relevant[:k]) / ideal if ideal > 0 else 0.0


def score_question(
    ranked: list[Chunk], gold: frozenset[str], ks=DEFAULT_KS, total_relevant: int | None = None
) -> dict[str, float]:
    """Score one question's ranked chunks.

    total_relevant is the number of relevant chunks available in the retriever's
    search scope (paper or corpus), used as the ideal ranking for ndcg@k. If not
    given, it falls back to the relevant chunks already present in `ranked` -- exact
    when the caller already searched the whole scope, an underestimate otherwise.
    """
    relevant = [is_relevant(c, gold) for c in ranked]
    if total_relevant is None:
        total_relevant = sum(relevant)
    first_hit = next((i for i, r in enumerate(relevant) if r), None)

    scores = {"mrr": 0.0 if first_hit is None else 1.0 / (first_hit + 1)}
    for k in ks:
        covered = gold & {uid for c in ranked[:k] for uid in c.unit_ids}
        scores[f"hit@{k}"] = float(any(relevant[:k]))
        scores[f"recall@{k}"] = len(covered) / len(gold)
        scores[f"precision@{k}"] = sum(relevant[:k]) / k
        scores[f"ndcg@{k}"] = _ndcg_at_k(relevant, total_relevant, k)
    return scores


def _count_relevant_in_scope(
    chunks: list[Chunk] | None, scope: str, questions: list[EvalQuestion]
) -> list[int | None]:
    """Per question, how many of the retriever's candidate chunks are relevant."""
    if chunks is None:
        return [None] * len(questions)

    by_paper: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_paper[c.paper_id].append(c)

    counts = []
    for q in questions:
        pool = chunks if scope == "corpus" else by_paper.get(q.paper_id, [])
        counts.append(sum(1 for c in pool if is_relevant(c, q.gold_unit_ids)))
    return counts


def evaluate(
    retriever: Retriever, questions: list[EvalQuestion], ks=DEFAULT_KS
) -> tuple[dict[str, float], list[dict]]:
    """Return (mean metrics over questions, per-question rows for error analysis)."""
    ranked_lists = retriever.retrieve_batch(questions, max(ks))
    scope = getattr(retriever, "scope", "paper")
    total_relevant = _count_relevant_in_scope(getattr(retriever, "chunks", None), scope, questions)

    rows = []
    for q, ranked, n_relevant in zip(questions, ranked_lists, total_relevant):
        rows.append(
            {
                "question_id": q.question_id,
                "paper_id": q.paper_id,
                "question": q.question,
                "question_type": q.question_type,
                "evidence_type": q.evidence_type,
                "gold_unit_ids": sorted(q.gold_unit_ids),
                "retrieved_chunk_ids": [c.chunk_id for c in ranked],
                "scores": score_question(ranked, q.gold_unit_ids, ks, n_relevant),
            }
        )

    metric_names = rows[0]["scores"].keys() if rows else []
    summary = {m: float(np.mean([r["scores"][m] for r in rows])) for m in metric_names}
    return summary, rows


def breakdown(rows: list[dict], key: str) -> dict[str, dict[str, float]]:
    """Group evaluate()'s per-question rows by row[key] and average their scores.

    Use key="question_type" or key="evidence_type" for the 1.7 error-analysis cuts.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)

    out = {}
    for group_key, group_rows in groups.items():
        metric_names = group_rows[0]["scores"].keys()
        out[group_key] = {
            "n": len(group_rows),
            **{m: float(np.mean([r["scores"][m] for r in group_rows])) for m in metric_names},
        }
    return out
