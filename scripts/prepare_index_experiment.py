"""Internal worker: isolate PyTorch/tokenizers from FAISS's OpenMP runtime."""
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from rag.benchmark import build_corpus
from rag.chunking import HFTokenizer, ChunkerConfig
from rag.embedding import SentenceTransformerEmbedder, EmbedderConfig, cached_embeddings
from rag.hotpot import build_hotpot_corpus
from rag.indexing import embedding_text
from rag.parsing import load_qasper, load_qasper_hf


def main():
    config_path = Path(sys.argv[1])
    args = json.loads(config_path.read_text())
    import numpy as np
    import torch
    torch.set_num_threads(args["threads"])
    tokenizer = HFTokenizer()
    data = load_qasper(args["json"]) if args["json"] else load_qasper_hf(args["split"])
    config = ChunkerConfig(args["chunk_size"], args["overlap"])
    corpus = (build_corpus(data, tokenizer, config) if args["dataset"] == "qasper" else
              build_hotpot_corpus(data, tokenizer, config))
    questions = random.Random(args["seed"]).sample(sorted(corpus.questions, key=lambda q: q.question_id),
                                               min(args["limit"], len(corpus.questions)))
    if not questions or not corpus.chunks:
        raise ValueError("dataset must contain chunks and evaluable questions")
    embedder = SentenceTransformerEmbedder(EmbedderConfig(device=args["device"],
        batch_size=args["batch_size"], revision=args["revision"]))
    embedder.load()
    max_tokens = embedder._model.max_seq_length - embedder._model.tokenizer.num_special_tokens_to_add(False)
    titles = ({pid: paper.get("title", "") for pid, paper in data.items()} if args["dataset"] == "qasper" else {})
    start = time.perf_counter()
    query_vectors = cached_embeddings(embedder, [q.question for q in questions], args["cache"])
    query_seconds = time.perf_counter() - start
    artifacts = config_path.parent
    np.save(artifacts / "queries.npy", query_vectors, allow_pickle=False)
    (artifacts / "question_ids.json").write_text(json.dumps([q.question_id for q in questions]))
    with (artifacts / "chunks.jsonl").open("w") as f:
        for chunk in corpus.chunks:
            f.write(json.dumps(asdict(chunk)) + "\n")
    import importlib.metadata
    report = {"dataset_sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
              "model": embedder.model_name, "chunker": asdict(config), "stats": dict(corpus.stats),
              "n_questions": len(questions), "query_embedding_or_cache_seconds": query_seconds,
              "query_inputs_over_limit": sum(len(tokenizer.token_spans(q.question)) > max_tokens for q in questions),
              "encoder_environment": {"torch": torch.__version__, "device": str(embedder._model.device),
                  "sentence_transformers": importlib.metadata.version("sentence-transformers"),
                  "transformers": importlib.metadata.version("transformers"),
                  "model_commit": getattr(embedder._model[0].auto_model.config, "_commit_hash", None)},
              "questions": [{**asdict(q), "gold_unit_ids": sorted(q.gold_unit_ids)} for q in questions],
              "embedding_times": {}}
    for variant in args["variants"]:
        texts = [embedding_text(c, titles if variant == "contextual" else {}, tokenizer, max_tokens,
                                contextual=variant == "contextual") for c in corpus.chunks]
        start = time.perf_counter()
        embeddings = cached_embeddings(embedder, texts, args["cache"])
        report["embedding_times"][variant] = time.perf_counter() - start
        variant_dir = artifacts / variant
        variant_dir.mkdir(exist_ok=True)
        np.save(variant_dir / "embeddings.npy", embeddings, allow_pickle=False)
        (variant_dir / "embedding_metadata.json").write_text(json.dumps({
            "model": embedder.model_name, "revision": args["revision"], "normalized": True,
            "chunk_ids": [c.chunk_id for c in corpus.chunks], "indexed_texts": texts,
            "shape": list(embeddings.shape), "variant": variant}, indent=2))
        print(f"Prepared {variant}: {len(corpus.chunks)} chunks", flush=True)
    (artifacts / "prepared.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main()
