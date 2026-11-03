"""Step 1.2: section-aware sliding-window chunking of ParsedUnits.

Consecutive paragraphs of the same section are concatenated into one token stream,
and a fixed-size window (default 200 tokens, 25-token / 12.5% overlap) slides over it.
Windows never cross a section boundary, so a chunk always has a single section_name.
Figure/table captions are unrelated to each other, so each caption is its own "section".

Token counts use the embedding model's own WordPiece tokenizer, so a 200-token chunk
(+2 special tokens) always fits under all-MiniLM-L6-v2's 256-token input limit.
Window edges are snapped back to a word boundary so chunks never start or end
mid-word (e.g. on a "##ing" continuation piece).

Each chunk records the unit_ids it overlaps. Relevance is defined at that level:
a chunk is relevant to a question iff it contains at least one gold-evidence unit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from itertools import groupby
from pathlib import Path
from typing import Iterable, Iterator, Protocol

from rag.embedding import DEFAULT_MODEL_NAME
from rag.parsing import ParsedUnit

FLOAT_SELECTED_PREFIX = "FLOAT SELECTED: "

Span = tuple[int, int]  # (char_start, char_end) of one token within its paragraph


class Tokenizer(Protocol):
    def token_spans(self, text: str) -> list[Span]: ...


class HFTokenizer:
    """The embedding model's WordPiece tokenizer, exposing character offsets per token."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_name)
        # We tokenize whole paragraphs on purpose; silence the "longer than max length" warning.
        self._tok.model_max_length = 10**9

    def token_spans(self, text: str) -> list[Span]:
        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [tuple(span) for span in enc["offset_mapping"]]


class RegexTokenizer:
    """Dependency-free tokenizer (words and punctuation) for tests and quick experiments."""

    _PATTERN = re.compile(r"\w+|[^\w\s]")

    def token_spans(self, text: str) -> list[Span]:
        return [m.span() for m in self._PATTERN.finditer(text)]


@dataclass(frozen=True)
class ChunkerConfig:
    chunk_size: int = 200
    overlap: int = 25  # 12.5% of chunk_size, inside the 10-15% band

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.overlap < self.chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")


@dataclass(frozen=True)
class Chunk:
    paper_id: str
    chunk_id: str
    chunk_type: str  # "paragraph" | "caption"
    section_name: str | None
    unit_ids: tuple[str, ...]  # ParsedUnit.unit_ids this chunk overlaps, in document order
    text: str
    num_tokens: int


def _section_key(unit: ParsedUnit):
    # Paragraphs group by consecutive section_name; every caption stands alone.
    return unit.unit_id if unit.unit_type == "caption" else ("paragraph", unit.section_name)


def _snap_to_word_start(i: int, lo: int, word_starts: list[bool]) -> int:
    """Move i back to the nearest word start above lo; keep i if one word fills the whole window."""
    j = i
    while j > lo and not word_starts[j]:
        j -= 1
    return j if j > lo else i


def _windows(word_starts: list[bool], config: ChunkerConfig) -> Iterator[tuple[int, int]]:
    """Yield [start, end) token windows over a section's token stream."""
    n = len(word_starts)
    start = 0
    while start < n:
        end = min(start + config.chunk_size, n)
        if end < n:
            end = _snap_to_word_start(end, start, word_starts)
        yield start, end
        if end == n:
            return
        next_start = _snap_to_word_start(end - config.overlap, start, word_starts)
        start = max(next_start, start + 1)


def _chunk_section(
    units: list[ParsedUnit], tokenizer: Tokenizer, config: ChunkerConfig
) -> Iterator[tuple[tuple[str, ...], str, int]]:
    """Yield (unit_ids, text, num_tokens) for each window over one section."""
    tokens: list[tuple[int, int, int]] = []  # (index into units, char_start, char_end)
    for ui, unit in enumerate(units):
        tokens.extend((ui, s, e) for s, e in tokenizer.token_spans(unit.text))

    # A window may only begin at a token that isn't glued to the previous one.
    word_starts = [
        i == 0 or tok[0] != tokens[i - 1][0] or tok[1] > tokens[i - 1][2]
        for i, tok in enumerate(tokens)
    ]

    for start, end in _windows(word_starts, config):
        pieces, unit_ids = [], []
        for ui, group in groupby(tokens[start:end], key=lambda t: t[0]):
            group = list(group)
            pieces.append(units[ui].text[group[0][1]:group[-1][2]])
            unit_ids.append(units[ui].unit_id)
        yield tuple(unit_ids), "\n\n".join(pieces), end - start


def chunk_paper(
    paper_units: list[ParsedUnit], tokenizer: Tokenizer, config: ChunkerConfig | None = None
) -> list[Chunk]:
    """Chunk one paper's units (in parse order) without crossing section boundaries."""
    config = config or ChunkerConfig()
    chunks: list[Chunk] = []
    for _, section in groupby(paper_units, key=_section_key):
        section = list(section)
        head = section[0]
        for unit_ids, text, num_tokens in _chunk_section(section, tokenizer, config):
            chunks.append(
                Chunk(
                    paper_id=head.paper_id,
                    chunk_id=f"{head.paper_id}::chunk::{len(chunks)}",
                    chunk_type=head.unit_type,
                    section_name=head.section_name,
                    unit_ids=unit_ids,
                    text=text,
                    num_tokens=num_tokens,
                )
            )
    return chunks


def chunk_units(
    units: Iterable[ParsedUnit], tokenizer: Tokenizer, config: ChunkerConfig | None = None
) -> Iterator[Chunk]:
    """Chunk a stream of units from many papers (units of a paper must be contiguous)."""
    for _, paper_units in groupby(units, key=lambda u: u.paper_id):
        yield from chunk_paper(list(paper_units), tokenizer, config)


def evidence_to_unit_ids(evidence: Iterable[str], paper_units: list[ParsedUnit]) -> set[str]:
    """Map QASPER gold evidence strings to the unit_ids of one paper by exact text match."""
    by_text: dict[str, list[str]] = {}
    for unit in paper_units:
        if unit.text.strip():
            by_text.setdefault(unit.text.strip(), []).append(unit.unit_id)

    gold: set[str] = set()
    for ev in evidence:
        gold.update(by_text.get(ev.removeprefix(FLOAT_SELECTED_PREFIX).strip(), ()))
    return gold


def is_relevant(chunk: Chunk, gold_unit_ids: set[str]) -> bool:
    """A chunk is relevant iff it contains at least one gold-evidence unit."""
    return not gold_unit_ids.isdisjoint(chunk.unit_ids)


def write_chunks_jsonl(chunks: Iterable[Chunk], out_path: str | Path) -> int:
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            count += 1
    return count
