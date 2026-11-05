"""HotpotQA distractor-context adapter, with sentence-level evidence provenance.

Each question's supplied articles are its own retrieval scope. This is not a
full-Wikipedia experiment. Sentence IDs are namespaced by question and title.
"""
from collections import Counter
import json

from rag.benchmark import Corpus, EvalQuestion
from rag.chunking import chunk_paper
from rag.parsing import ParsedUnit


def unit_id(qid, title, sentence):
    return json.dumps([qid, title, sentence], ensure_ascii=False, separators=(",", ":"))


def build_hotpot_corpus(data, tokenizer, config=None):
    chunks, questions = [], []
    stats = Counter()
    seen = set()
    for record in data:
        qid = record["_id"]
        if qid in seen:
            raise ValueError(f"duplicate HotpotQA question ID: {qid}")
        seen.add(qid)
        units = []
        for title, sentences in record["context"]:
            for i, sentence in enumerate(sentences):
                units.append(ParsedUnit(qid, unit_id(qid, title, i), "paragraph", title, i, sentence))
        ids = {u.unit_id for u in units}
        gold = frozenset(unit_id(qid, title, i) for title, i in record["supporting_facts"])
        if not gold or not gold <= ids:
            raise ValueError(f"missing supporting facts in HotpotQA context: {qid}")
        chunks.extend(chunk_paper(units, tokenizer, config))
        questions.append(EvalQuestion(qid, qid, record["question"], gold,
                                      record.get("type", "unknown"), "text"))
        stats["questions_total"] += 1
    stats.update(chunks=len(chunks), questions_evaluated=len(questions))
    return Corpus(chunks, questions, stats)
