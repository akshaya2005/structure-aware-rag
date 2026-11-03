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


HF_DATASET = "allenai/qasper"
# The hub repo only ships a loading script, which datasets>=4 refuses to run;
# the hub's auto-converted parquet branch loads without it.
HF_REVISION = "refs/convert/parquet"
HF_SPLITS = {"train": "train", "dev": "validation", "test": "test"}


def hf_row_to_paper(row: dict) -> dict:
    """Convert one HF row (lists of dicts stored as dicts of lists) to the raw-JSON paper shape."""
    full_text = row["full_text"]
    figs = row["figures_and_tables"]
    qas = row["qas"]

    questions = []
    for i in range(len(qas["question_id"])):
        qa = {key: qas[key][i] for key in qas if key != "answers"}
        answers = qas["answers"][i]
        qa["answers"] = [
            {"answer": answer, "annotation_id": annotation_id, "worker_id": worker_id}
            for answer, annotation_id, worker_id in zip(
                answers["answer"], answers["annotation_id"], answers["worker_id"]
            )
        ]
        questions.append(qa)

    return {
        "title": row["title"],
        "abstract": row["abstract"],
        "full_text": [
            {"section_name": name, "paragraphs": paragraphs}
            for name, paragraphs in zip(full_text["section_name"], full_text["paragraphs"])
        ],
        "qas": questions,
        "figures_and_tables": [
            {"caption": caption, "file": file} for caption, file in zip(figs["caption"], figs["file"])
        ],
    }


def load_qasper_hf(split: str) -> dict:
    """Load a QASPER split ("train" | "dev" | "test") from the HF hub into {paper_id: paper_dict}."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, revision=HF_REVISION, split=HF_SPLITS[split])
    return {row["id"]: hf_row_to_paper(row) for row in ds}


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


def read_jsonl(path: str | Path) -> Iterator[ParsedUnit]:
    """Inverse of write_jsonl: stream ParsedUnits back from a {split}_units.jsonl file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield ParsedUnit(**json.loads(line))


def write_jsonl(units: Iterator[ParsedUnit], out_path: str | Path) -> int:
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for unit in units:
            f.write(json.dumps(asdict(unit), ensure_ascii=False) + "\n")
            count += 1
    return count
