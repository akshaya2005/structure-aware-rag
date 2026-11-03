"""CLI: run the QASPER retrieval test bench and write results/.

Loads the split from the HF hub (allenai/qasper) by default, or from
data/raw/qasper-{split}-v0.3.json with --source json. Parsing + chunking happen in
memory, so scripts/run_parsing.py and scripts/run_chunking.py are not prerequisites.

Usage:
    python3 scripts/run_benchmark.py                       # dense, dev split, per-paper
    python3 scripts/run_benchmark.py --retriever random    # sanity floor
    python3 scripts/run_benchmark.py --scope corpus --device mps
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
    DenseRetriever,
    RandomRetriever,
    build_corpus,
    evaluate,
)
from rag.chunking import ChunkerConfig, HFTokenizer  # noqa: E402
from rag.embedding import DEFAULT_MODEL_NAME, EmbedderConfig, SentenceTransformerEmbedder  # noqa: E402
from rag.parsing import load_qasper, load_qasper_hf  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    parser.add_argument("--source", default="hf", choices=("hf", "json"))
    parser.add_argument("--retriever", default="dense", choices=("dense", "random"))
    parser.add_argument("--scope", default="paper", choices=("paper", "corpus"))
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=None, help="cpu | cuda | mps (default: auto)")
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=25)
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    chunk_config = ChunkerConfig(chunk_size=args.chunk_size, overlap=args.overlap)
    if args.source == "hf":
        data = load_qasper_hf(args.split)
    else:
        data = load_qasper(RAW_DIR / f"qasper-{args.split}-v0.3.json")
    corpus = build_corpus(data, HFTokenizer(args.model), chunk_config)
    print(f"[{args.split}] " + " ".join(f"{k}={v}" for k, v in sorted(corpus.stats.items())))

    if args.retriever == "dense":
        embedder = SentenceTransformerEmbedder(EmbedderConfig(model_name=args.model, device=args.device))
        retriever = DenseRetriever(embedder, corpus.chunks, scope=args.scope)
    else:
        retriever = RandomRetriever(corpus.chunks, seed=args.seed)

    ks = tuple(sorted(args.ks))
    summary, rows = evaluate(retriever, corpus.questions, ks)

    print(f"\n{'metric':<14}{'value':>8}")
    for name, value in summary.items():
        print(f"{name:<14}{value:>8.4f}")

    scope = args.scope if args.retriever == "dense" else "paper"
    run_name = f"{args.split}_{args.retriever}_{scope}_c{args.chunk_size}_o{args.overlap}"
    RESULTS_DIR.mkdir(exist_ok=True)
    summary_path = RESULTS_DIR / f"{run_name}.json"
    rows_path = RESULTS_DIR / f"{run_name}_questions.jsonl"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run": run_name,
                "args": vars(args),
                "chunker": asdict(chunk_config),
                "stats": dict(corpus.stats),
                "metrics": summary,
            },
            f,
            indent=2,
        )
    with open(rows_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nwrote {summary_path}\nwrote {rows_path}")


if __name__ == "__main__":
    main()
