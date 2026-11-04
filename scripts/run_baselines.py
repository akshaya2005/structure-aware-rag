"""CLI: run baselines (1.7) -- BM25, dense, and a random floor on the same chunk
corpus and the same question subset, so their numbers are directly comparable.

Writes one results/{run}.json with, per retriever: overall metrics plus a
breakdown by question type (extractive/abstractive/yes_no/mixed) and by evidence
type (text/table/mixed).

Usage:
    python3 scripts/run_baselines.py                    # dev split, per-paper scope
    python3 scripts/run_baselines.py --limit 200         # faster iteration
    python3 scripts/run_baselines.py --scope corpus --device mps
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag.benchmark import (  # noqa: E402
    DEFAULT_KS,
    BM25Retriever,
    DenseRetriever,
    RandomRetriever,
    breakdown,
    build_corpus,
    evaluate,
)
from rag.chunking import ChunkerConfig, HFTokenizer  # noqa: E402
from rag.embedding import DEFAULT_MODEL_NAME, EmbedderConfig, SentenceTransformerEmbedder  # noqa: E402
from rag.parsing import load_qasper, load_qasper_hf  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"


def print_table(summaries: dict[str, dict[str, float]], ks: tuple[int, ...]) -> None:
    cols = ["mrr"] + [f"hit@{k}" for k in ks] + ["ndcg@10" if 10 in ks else f"ndcg@{ks[-1]}"]
    print(f"\n{'retriever':<10}" + "".join(f"{c:>10}" for c in cols))
    for name, summary in summaries.items():
        print(f"{name:<10}" + "".join(f"{summary.get(c, float('nan')):>10.4f}" for c in cols))


def print_breakdown(label: str, summaries: dict[str, dict[str, dict[str, float]]], metric: str) -> None:
    groups = sorted({g for s in summaries.values() for g in s})
    print(f"\nby {label} ({metric}, n):")
    print(f"{'retriever':<10}" + "".join(f"{g:>16}" for g in groups))
    for name, group_summary in summaries.items():
        cells = [
            f"{group_summary[g][metric]:.3f} (n={group_summary[g]['n']})" if g in group_summary else "-"
            for g in groups
        ]
        print(f"{name:<10}" + "".join(f"{c:>16}" for c in cells))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    parser.add_argument("--source", default="hf", choices=("hf", "json"))
    parser.add_argument("--scope", default="paper", choices=("paper", "corpus"))
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=None, help="cpu | cuda | mps (default: auto)")
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=25)
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    chunk_config = ChunkerConfig(chunk_size=args.chunk_size, overlap=args.overlap)
    data = load_qasper_hf(args.split) if args.source == "hf" else load_qasper(
        RAW_DIR / f"qasper-{args.split}-v0.3.json"
    )
    corpus = build_corpus(data, HFTokenizer(args.model), chunk_config)
    print(f"[{args.split}] " + " ".join(f"{k}={v}" for k, v in sorted(corpus.stats.items())))

    questions = corpus.questions[: args.limit] if args.limit else corpus.questions
    ks = tuple(sorted(args.ks))

    embedder = SentenceTransformerEmbedder(EmbedderConfig(model_name=args.model, device=args.device))
    retrievers = {
        "random": RandomRetriever(corpus.chunks, seed=args.seed),
        "bm25": BM25Retriever(corpus.chunks, scope=args.scope),
        "dense": DenseRetriever(embedder, corpus.chunks, scope=args.scope),
    }

    run_name = f"{args.split}_baselines_{args.scope}_c{args.chunk_size}_o{args.overlap}"
    report = {
        "run": run_name,
        "args": vars(args),
        "chunker": asdict(chunk_config),
        "stats": dict(corpus.stats),
        "n_questions": len(questions),
        "retrievers": {},
    }

    summaries, by_qtype, by_evtype = {}, {}, {}
    for name, retriever in retrievers.items():
        summary, rows = evaluate(retriever, questions, ks)
        summaries[name] = summary
        by_qtype[name] = breakdown(rows, "question_type")
        by_evtype[name] = breakdown(rows, "evidence_type")
        report["retrievers"][name] = {
            "summary": summary,
            "by_question_type": by_qtype[name],
            "by_evidence_type": by_evtype[name],
        }

    print_table(summaries, ks)
    print_breakdown("question_type", by_qtype, metric=f"hit@{ks[-1]}")
    print_breakdown("evidence_type", by_evtype, metric=f"hit@{ks[-1]}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{run_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
