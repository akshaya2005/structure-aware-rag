"""CLI: parse QASPER train/dev into data/processed/{split}_units.jsonl.

Usage: python3 scripts/run_parsing.py [--source hf|json]
  hf   (default) load allenai/qasper from the HF hub
  json read data/raw/qasper-{split}-v0.3.json
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag.parsing import load_qasper, load_qasper_hf, parse_paper, write_jsonl  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="hf", choices=("hf", "json"))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev"):
        if args.source == "hf":
            data = load_qasper_hf(split)
        else:
            data = load_qasper(RAW_DIR / f"qasper-{split}-v0.3.json")
        out_path = OUT_DIR / f"{split}_units.jsonl"

        units = [unit for paper_id, paper in data.items() for unit in parse_paper(paper_id, paper)]
        write_jsonl(iter(units), out_path)

        num_papers = len({u.paper_id for u in units})
        num_para = sum(1 for u in units if u.unit_type == "paragraph")
        num_caption = sum(1 for u in units if u.unit_type == "caption")
        num_empty = sum(1 for u in units if not u.text.strip())

        print(f"[{split}] papers={num_papers} units={len(units)} "
              f"paragraphs={num_para} captions={num_caption} empty_text={num_empty}")
        print(f"[{split}] wrote {out_path}")


if __name__ == "__main__":
    main()
