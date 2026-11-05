"""Reproducible BM25 / exact FAISS / HNSW comparison with plain/contextual ablations."""
import argparse
import csv
from dataclasses import asdict
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from rag.benchmark import BM25Retriever, breakdown, evaluate, DEFAULT_KS
from rag.indexing import FaissRetriever


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", type=Path, help="local QASPER or HotpotQA JSON")
    p.add_argument("--dataset", choices=["qasper", "hotpot"], default="qasper")
    p.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    p.add_argument("--scope", choices=["paper", "corpus"], default="paper")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variants", choices=["plain", "contextual"], nargs="+", default=["plain", "contextual"])
    p.add_argument("--contextual", action="store_true", help="shortcut for --variants contextual")
    p.add_argument("--device", default=None)
    p.add_argument("--revision", default=None, help="model commit SHA for pinned experiments")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument("--overlap", type=int, default=25)
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-search", type=int, default=64)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--cache", type=Path, default=ROOT / "artifacts" / "embeddings")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "indexing" / "qasper.json")
    args = p.parse_args()
    if min(args.limit, args.batch_size, args.threads, args.m, args.ef_search, args.ef_construction, *args.ks) <= 0:
        p.error("limits, cutoffs, and index parameters must be positive")
    if args.dataset == "hotpot" and (args.json is None or args.scope != "paper"):
        p.error("HotpotQA requires --json and --scope paper (each question's supplied context)")
    import subprocess
    import faiss
    import numpy as np
    from rag.benchmark import EvalQuestion
    from rag.chunking import Chunk
    faiss.omp_set_num_threads(args.threads)
    ks = tuple(sorted(set(args.ks)))
    args.variants = ["contextual"] if args.contextual else list(dict.fromkeys(args.variants))
    artifacts = args.output.parent / (args.output.stem + "_artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config_path = artifacts / "config.json"
    config_path.write_text(json.dumps(config))
    # Separate processes avoid duplicate OpenMP runtimes from torch and FAISS wheels.
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_index_experiment.py"),
                    str(config_path)], check=True)
    prepared = json.loads((artifacts / "prepared.json").read_text())
    chunks = [Chunk(**{**c, "unit_ids": tuple(c["unit_ids"])})
              for line in (artifacts / "chunks.jsonl").read_text().splitlines()
              for c in [json.loads(line)]]
    questions = [EvalQuestion(**{**q, "gold_unit_ids": frozenset(q["gold_unit_ids"])})
                 for q in prepared.pop("questions")]
    query_vectors = np.load(artifacts / "queries.npy", allow_pickle=False)
    class CachedQueries:
        def embed(self, texts):
            if texts != [q.question for q in questions]:
                raise ValueError("unexpected query batch")
            return query_vectors
    report = {**prepared, "config": config, "environment": {
        "python": platform.python_version(), "platform": platform.platform(),
        "faiss": faiss.__version__, "numpy": np.__version__, "threads": args.threads}, "indexes": {}}

    def measure(retriever, name, build_seconds, size):
        retriever.retrieve_batch(questions, max(ks))
        timings = []
        for _ in range(3):
            start = time.perf_counter()
            retriever.retrieve_batch(questions, max(ks))
            timings.append(time.perf_counter() - start)
        metrics, rows = evaluate(retriever, questions, ks)
        report["indexes"][name] = {"metrics": metrics, "build_seconds": build_seconds,
            "search_batch_seconds": timings,
            "search_ms_per_question": 1000 * float(np.median(timings)) / len(questions),
            "serialized_bytes": size, "by_question_type": breakdown(rows, "question_type"),
            "by_evidence_type": breakdown(rows, "evidence_type"), "questions": rows}
        print(name, metrics, flush=True)
        return [r["retrieved_chunk_ids"] for r in rows]

    for variant in args.variants:
        variant_dir = artifacts / variant
        embeddings = np.load(variant_dir / "embeddings.npy", allow_pickle=False)
        texts = json.loads((variant_dir / "embedding_metadata.json").read_text())["indexed_texts"]
        embedding_seconds = prepared["embedding_times"][variant]
        start = time.perf_counter()
        # Same IDs/provenance and chunk boundaries; only index input text differs.
        from dataclasses import replace
        sparse_chunks = [replace(c, text=text) for c, text in zip(chunks, texts)]
        sparse = BM25Retriever(sparse_chunks, scope=args.scope)
        build_seconds = time.perf_counter() - start
        sparse_size = 0
        sparse_manifest = []
        for i, (key, index) in enumerate(sparse._indices.items()):
            path = variant_dir / f"bm25_{i}.json"
            path.write_text(json.dumps(asdict(index)))
            sparse_size += path.stat().st_size
            sparse_manifest.append({"scope_key": key, "file": path.name})
        (variant_dir / "bm25_manifest.json").write_text(json.dumps(sparse_manifest))
        measure(sparse, f"{variant}_bm25", build_seconds, sparse_size)
        exact = None
        for kind in ("flat", "hnsw"):
            start = time.perf_counter()
            retriever = FaissRetriever(CachedQueries(), chunks, embeddings, kind, args.scope,
                                       m=args.m, ef_construction=args.ef_construction, ef_search=args.ef_search)
            build_seconds = time.perf_counter() - start
            name = f"{variant}_{kind}"
            rankings = measure(retriever, name, build_seconds,
                              sum(len(faiss.serialize_index(i.index)) for i in retriever.indices.values()))
            if kind == "flat":
                exact = rankings
            report["indexes"][name]["agreement"] = {f"exact_overlap@{k}": float(np.mean([
                len(set(a[:k]) & set(b[:k])) / len(a[:k]) if a[:k] else 1.0
                for a, b in zip(exact, rankings)])) for k in ks}
            report["indexes"][name]["embedding_or_cache_seconds"] = embedding_seconds
            retriever.save(variant_dir / kind)
        # Checkpoint after each ablation to retain completed expensive work.
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    table = [{"system": name, **r["metrics"], "search_ms_per_question": r["search_ms_per_question"],
              "build_seconds": r["build_seconds"], "serialized_bytes": r["serialized_bytes"]}
             for name, r in report["indexes"].items()]
    with args.output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"Saved {args.output} and {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
