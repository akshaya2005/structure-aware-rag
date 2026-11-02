import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag.parsing import parse_paper  # noqa: E402

SAMPLE_PAPER = {
    "full_text": [
        {"section_name": "Introduction", "paragraphs": ["Para zero.", "Para one."]},
        {"section_name": None, "paragraphs": [""]},
    ],
    "figures_and_tables": [
        {"file": "1-Table1-1.png", "caption": "Table 1: Data Statistics."},
    ],
}


def test_paragraph_units_carry_section_and_index():
    units = parse_paper("p1", SAMPLE_PAPER)
    paras = [u for u in units if u.unit_type == "paragraph"]
    assert len(paras) == 3
    assert paras[0].section_name == "Introduction"
    assert paras[0].para_idx == 0
    assert paras[0].text == "Para zero."
    assert paras[1].para_idx == 1


def test_null_section_name_is_preserved_not_crashed_on():
    units = parse_paper("p1", SAMPLE_PAPER)
    null_section_units = [u for u in units if u.section_name is None and u.unit_type == "paragraph"]
    assert len(null_section_units) == 1
    assert null_section_units[0].text == ""


def test_caption_is_its_own_unit_and_matches_gold_evidence_after_stripping_prefix():
    units = parse_paper("p1", SAMPLE_PAPER)
    captions = [u for u in units if u.unit_type == "caption"]
    assert len(captions) == 1

    gold_evidence = "FLOAT SELECTED: Table 1: Data Statistics."
    stripped = gold_evidence.removeprefix("FLOAT SELECTED: ")
    assert captions[0].text == stripped


def test_unit_ids_are_unique_within_a_paper():
    units = parse_paper("p1", SAMPLE_PAPER)
    ids = [u.unit_id for u in units]
    assert len(ids) == len(set(ids))


def test_real_dev_file_float_selected_evidence_matches_a_parsed_caption():
    """Regression test for the gotcha called out in the plan: without caption
    units, FLOAT SELECTED evidence for table/figure questions can't be retrieved."""
    raw_path = ROOT / "data" / "raw" / "qasper-dev-v0.3.json"
    if not raw_path.exists():
        return  # dataset not downloaded in this environment; skip

    with open(raw_path, encoding="utf-8") as f:
        data = json.load(f)

    paper_id = "1912.01214"
    units = parse_paper(paper_id, data[paper_id])
    caption_texts = {u.text for u in units if u.unit_type == "caption"}

    gold_evidence = "FLOAT SELECTED: Table 1: Data Statistics."
    stripped = gold_evidence.removeprefix("FLOAT SELECTED: ")
    assert stripped in caption_texts
