"""CLI: chunk data/processed/{split}_units.jsonl into data/processed/{split}_chunks.jsonl.

Run scripts/run_parsing.py first.

Usage: python3 scripts/run_chunking.py [--chunk-size 200] [--overlap 25]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag.chunking import ChunkerConfig, HFTokenizer, chunk_units, write_chunks_jsonl  # noqa: E402
from rag.parsing import read_jsonl  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=25)
    args = parser.parse_args()

    config = ChunkerConfig(chunk_size=args.chunk_size, overlap=args.overlap)
    tokenizer = HFTokenizer()

    for split in ("train", "dev"):
        in_path = PROCESSED_DIR / f"{split}_units.jsonl"
        out_path = PROCESSED_DIR / f"{split}_chunks.jsonl"

        chunks = list(chunk_units(read_jsonl(in_path), tokenizer, config))
        write_chunks_jsonl(chunks, out_path)

        sizes = [c.num_tokens for c in chunks]
        num_caption = sum(1 for c in chunks if c.chunk_type == "caption")
        num_multi_unit = sum(1 for c in chunks if len(c.unit_ids) > 1)

        print(f"[{split}] chunks={len(chunks)} captions={num_caption} "
              f"multi_unit={num_multi_unit} mean_tokens={sum(sizes) / len(sizes):.1f} "
              f"max_tokens={max(sizes)}")
        print(f"[{split}] wrote {out_path}")


if __name__ == "__main__":
    main()
