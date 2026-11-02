"""CLI: parse QASPER train/dev JSON into data/processed/{split}_units.jsonl.

Usage: python3 scripts/run_parsing.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag.parsing import parse_qasper_file, write_jsonl  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev"):
        in_path = RAW_DIR / f"qasper-{split}-v0.3.json"
        out_path = OUT_DIR / f"{split}_units.jsonl"

        units = list(parse_qasper_file(in_path))
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
