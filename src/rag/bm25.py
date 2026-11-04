"""Step 1.4: BM25 inverted index over chunk text.

An inverted index maps each term to its posting list: the chunks it appears in and
the term frequency there. Alongside it we keep per-term document frequency and
per-document length, which is all Okapi BM25 needs to score a query without ever
touching document text again.

Tokenization is lowercased \\w+ words, no stemming or stopword removal. This is kept
identical to what scripts/check_bm25_against_reference.py feeds rank_bm25.BM25Okapi,
and the idf formula and epsilon floor below match rank_bm25's defaults too, so the
two implementations can be checked against each other term-for-term.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

_TOKEN_RE = re.compile(r"\w+")

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
DEFAULT_EPSILON = 0.25


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    epsilon: float = DEFAULT_EPSILON
    doc_ids: list[str] = field(default_factory=list)
    doc_lengths: list[int] = field(default_factory=list)
    avg_doc_length: float = 0.0
    # term -> [(doc_idx, term_freq), ...], one entry per doc that contains the term.
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    doc_freq: dict[str, int] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        doc_ids: Iterable[str],
        texts: Iterable[str],
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        epsilon: float = DEFAULT_EPSILON,
    ) -> "BM25Index":
        index = cls(k1=k1, b=b, epsilon=epsilon)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for doc_idx, (doc_id, text) in enumerate(zip(doc_ids, texts)):
            tokens = tokenize(text)
            index.doc_ids.append(doc_id)
            index.doc_lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                postings[term].append((doc_idx, tf))

        index.postings = dict(postings)
        index.doc_freq = {term: len(plist) for term, plist in index.postings.items()}
        index.avg_doc_length = (
            sum(index.doc_lengths) / len(index.doc_lengths) if index.doc_lengths else 0.0
        )
        index.idf = index._compute_idf()
        return index

    def _compute_idf(self) -> dict[str, float]:
        # Robertson/Sparck-Jones idf: log((N - df + 0.5) / (df + 0.5)). This goes
        # negative for terms in more than half the corpus, so floor those at
        # epsilon * average_idf instead of letting them subtract score (rank_bm25's
        # BM25Okapi does the same, with the same default epsilon=0.25).
        n = len(self.doc_ids)
        idf: dict[str, float] = {}
        negative_terms: list[str] = []
        idf_sum = 0.0
        for term, df in self.doc_freq.items():
            score = math.log(n - df + 0.5) - math.log(df + 0.5)
            idf[term] = score
            idf_sum += score
            if score < 0:
                negative_terms.append(term)

        average_idf = idf_sum / len(idf) if idf else 0.0
        floor = self.epsilon * average_idf
        for term in negative_terms:
            idf[term] = floor
        return idf

    def score_all(self, query: str) -> dict[int, float]:
        """Return {doc_idx: bm25_score} for every doc sharing >=1 query term."""
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            idf = self.idf.get(term)
            if not postings or not idf:
                continue
            for doc_idx, tf in postings:
                dl = self.doc_lengths[doc_idx]
                norm = 1 - self.b + self.b * (dl / self.avg_doc_length if self.avg_doc_length else 0)
                scores[doc_idx] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return scores

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return the top_k (doc_id, score) pairs for query, highest score first."""
        scores = self.score_all(query)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self.doc_ids[doc_idx], score) for doc_idx, score in ranked]
