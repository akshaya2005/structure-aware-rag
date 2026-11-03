import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag.chunking import (  # noqa: E402
    ChunkerConfig,
    RegexTokenizer,
    chunk_paper,
    evidence_to_unit_ids,
    is_relevant,
)
from rag.parsing import parse_paper  # noqa: E402

TOK = RegexTokenizer()


def words(start: int, stop: int) -> str:
    return " ".join(f"w{i}" for i in range(start, stop))


def test_long_paragraph_uses_fixed_windows_with_overlap():
    paper = {"full_text": [{"section_name": "Intro", "paragraphs": [words(0, 500)]}],
             "figures_and_tables": []}
    chunks = chunk_paper(parse_paper("p1", paper), TOK, ChunkerConfig(chunk_size=200, overlap=25))

    assert [c.num_tokens for c in chunks] == [200, 200, 150]
    assert chunks[0].text.split()[-25:] == chunks[1].text.split()[:25]
    assert chunks[1].text.startswith("w175 ")
    assert chunks[2].text.endswith("w499")


def test_short_consecutive_paragraphs_are_grouped_into_one_chunk():
    paper = {"full_text": [{"section_name": "Intro", "paragraphs": ["a b", "c d", "e f"]}],
             "figures_and_tables": []}
    units = parse_paper("p1", paper)
    [chunk] = chunk_paper(units, TOK)

    assert chunk.unit_ids == tuple(u.unit_id for u in units)
    assert chunk.text == "a b\n\nc d\n\ne f"
    assert chunk.num_tokens == 6


def test_paragraph_split_by_window_edge_is_recorded_in_both_chunks():
    paper = {"full_text": [{"section_name": "Intro", "paragraphs": [words(0, 6), words(6, 12)]}],
             "figures_and_tables": []}
    units = parse_paper("p1", paper)
    chunks = chunk_paper(units, TOK, ChunkerConfig(chunk_size=8, overlap=2))

    assert chunks[0].unit_ids == (units[0].unit_id, units[1].unit_id)
    assert chunks[1].unit_ids == (units[1].unit_id,)


def test_chunks_never_cross_section_boundaries():
    paper = {"full_text": [{"section_name": "Intro", "paragraphs": ["a b c"]},
                           {"section_name": "Method", "paragraphs": ["d e f"]}],
             "figures_and_tables": []}
    chunks = chunk_paper(parse_paper("p1", paper), TOK, ChunkerConfig(chunk_size=10, overlap=2))

    assert [(c.section_name, c.text) for c in chunks] == [("Intro", "a b c"), ("Method", "d e f")]


def test_window_edges_snap_back_to_word_boundaries():
    # Tokens: a b c d , e  -> the 4-token window would end between "d" and its glued ",".
    paper = {"full_text": [{"section_name": "Intro", "paragraphs": ["a b c d, e"]}],
             "figures_and_tables": []}
    chunks = chunk_paper(parse_paper("p1", paper), TOK, ChunkerConfig(chunk_size=4, overlap=0))

    assert [c.text for c in chunks] == ["a b c", "d, e"]


def test_captions_are_separate_chunks_and_empty_paragraphs_are_dropped():
    paper = {"full_text": [{"section_name": None, "paragraphs": ["", "body text"]}],
             "figures_and_tables": [{"caption": "Table 1: Stats."}, {"caption": "Figure 1: Arch."}]}
    units = parse_paper("p1", paper)
    chunks = chunk_paper(units, TOK)

    assert [c.chunk_type for c in chunks] == ["paragraph", "caption", "caption"]
    assert chunks[0].unit_ids == (units[1].unit_id,)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_relevance_matches_gold_paragraph_and_float_selected_caption():
    paper = {"full_text": [{"section_name": "Intro", "paragraphs": ["first para", "second para"]},
                           {"section_name": "Data", "paragraphs": ["data para"]}],
             "figures_and_tables": [{"caption": "Table 1: Data Statistics."}]}
    units = parse_paper("p1", paper)
    chunks = chunk_paper(units, TOK)

    gold = evidence_to_unit_ids(["second para", "FLOAT SELECTED: Table 1: Data Statistics."], units)
    assert [is_relevant(c, gold) for c in chunks] == [True, False, True]


def test_config_rejects_overlap_not_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        ChunkerConfig(chunk_size=10, overlap=10)
