"""Read-only numerical audit of the team's BM25 against rank_bm25.

Does not modify the implementation. Saves score/rank differences for its owner.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag.bm25 import BM25Index, tokenize


def audit():
    texts = ["zebra zebra migration", "penguin videos", "zebra videos", "a new forest", "lion roars"]
    ids = [str(i) for i in range(len(texts))]
    ours = BM25Index.build(ids, texts)
    reference = BM25Okapi([tokenize(t) for t in texts])
    rows = []
    for query in ["zebra migration", "zebra zebra", "unknown", "videos"]:
        sparse = ours.score_all(query)
        actual = np.array([sparse.get(i, 0) for i in range(len(texts))])
        expected = reference.get_scores(tokenize(query))
        rows.append({"query": query, "max_score_error": float(np.max(np.abs(actual - expected))),
                     "scores_match": bool(np.allclose(actual, expected)),
                     "ours_top5": [cid for cid, _ in ours.search(query, 5)],
                     "reference_top5": [ids[i] for i in np.argsort(-expected, kind="stable")]})
    return {"all_scores_match": all(r["scores_match"] for r in rows), "queries": rows,
            "note": "Existing sparse search omits unmatched chunks; reference ranks every chunk. Repeated query terms may differ. Implementation left unchanged for its owner."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/indexing/sparse_audit.json"))
    a = p.parse_args()
    report = audit()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
