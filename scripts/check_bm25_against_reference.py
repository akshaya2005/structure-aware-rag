"""Milestone check for 1.4: does our BM25Index agree with rank_bm25?

Builds the dev-split chunk corpus, picks ~50 evaluable questions, and for each one
ranks its paper's chunks with both our BM25Index and rank_bm25.BM25Okapi fed the
exact same tokenization (rag.bm25.tokenize). Reports, over those queries, the mean
Jaccard overlap of the two top-10 chunk-id sets and the fraction of queries whose
top-1 chunk agrees.

Usage: python3 scripts/check_bm25_against_reference.py [--source hf|json] [--n 50]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rank_bm25 import BM25Okapi  # noqa: E402

from rag.benchmark import build_corpus  # noqa: E402
from rag.bm25 import BM25Index, tokenize  # noqa: E402
from rag.chunking import RegexTokenizer  # noqa: E402
from rag.parsing import load_qasper, load_qasper_hf  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"


def top_k_reference(chunks, query: str, k: int) -> list[str]:
    corpus_tokens = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i].chunk_id for i in ranked]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="hf", choices=("hf", "json"))
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    if args.source == "hf":
        data = load_qasper_hf("dev")
    else:
        data = load_qasper(RAW_DIR / "qasper-dev-v0.3.json")

    # RegexTokenizer is dependency-free and fast enough for this many papers; the
    # comparison only cares that chunk text matches, not which tokenizer chunked it.
    corpus = build_corpus(data, RegexTokenizer())
    chunks_by_paper: dict[str, list] = {}
    for c in corpus.chunks:
        chunks_by_paper.setdefault(c.paper_id, []).append(c)

    questions = corpus.questions[: args.n]
    if not questions:
        raise SystemExit("no evaluable questions found")

    top1_matches = 0
    jaccard_sum = 0.0
    for q in questions:
        paper_chunks = chunks_by_paper[q.paper_id]
        ours_index = BM25Index.build([c.chunk_id for c in paper_chunks], [c.text for c in paper_chunks])
        ours = [cid for cid, _ in ours_index.search(q.question, args.k)]
        ref = top_k_reference(paper_chunks, q.question, args.k)

        if ours[:1] == ref[:1]:
            top1_matches += 1
        a, b = set(ours), set(ref)
        jaccard_sum += len(a & b) / len(a | b) if (a or b) else 1.0

    n = len(questions)
    print(f"queries checked:       {n}")
    print(f"top-1 agreement:       {top1_matches}/{n} ({top1_matches / n:.1%})")
    print(f"mean top-{args.k} jaccard overlap: {jaccard_sum / n:.3f}")


if __name__ == "__main__":
    main()
