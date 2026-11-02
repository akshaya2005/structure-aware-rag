"""Step 1.1: parse QASPER's raw JSON into flat, addressable units.

Each paper contributes two kinds of units:
  - "paragraph": one entry per paragraph in full_text, keyed by section_name + para_idx.
  - "caption":   one entry per figures_and_tables item (figure or table caption).

Captions must be parsed as their own units because gold evidence for table/figure
questions is stored as a string like "FLOAT SELECTED: Table 1: Data Statistics.",
which only matches a caption's text, never a paragraph's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class ParsedUnit:
    paper_id: str
    unit_id: str
    unit_type: str  # "paragraph" | "caption"
    section_name: str | None
    para_idx: int
    text: str


def load_qasper(path: str | Path) -> dict:
    """Load a raw qasper-{split}-v0.3.json file into {paper_id: paper_dict}."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_paper(paper_id: str, paper: dict) -> list[ParsedUnit]:
    """Flatten one paper's full_text paragraphs and figure/table captions into units."""
    units: list[ParsedUnit] = []

    for section in paper["full_text"]:
        section_name = section["section_name"]  # can be None in the raw data
        for para_idx, text in enumerate(section["paragraphs"]):
            units.append(
                ParsedUnit(
                    paper_id=paper_id,
                    unit_id=f"{paper_id}::para::{section_name}::{para_idx}",
                    unit_type="paragraph",
                    section_name=section_name,
                    para_idx=para_idx,
                    text=text,
                )
            )

    for fig_idx, fig in enumerate(paper["figures_and_tables"]):
        units.append(
            ParsedUnit(
                paper_id=paper_id,
                unit_id=f"{paper_id}::caption::{fig_idx}",
                unit_type="caption",
                section_name=None,
                para_idx=fig_idx,
                text=fig["caption"],
            )
        )

    return units


def parse_qasper_file(path: str | Path) -> Iterator[ParsedUnit]:
    """Parse every paper in a qasper-{split}-v0.3.json file into ParsedUnits."""
    data = load_qasper(path)
    for paper_id, paper in data.items():
        yield from parse_paper(paper_id, paper)


def write_jsonl(units: Iterator[ParsedUnit], out_path: str | Path) -> int:
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for unit in units:
            f.write(json.dumps(asdict(unit), ensure_ascii=False) + "\n")
            count += 1
    return count
