"""Create a Markdown comparison table and paired bootstrap intervals from a run."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--metric", default="recall@5")
    p.add_argument("--baseline", default="plain_flat")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.resamples <= 0:
        p.error("resamples must be positive")
    report = json.loads(args.report.read_text())
    systems = report["indexes"]
    if args.baseline not in systems:
        p.error("baseline not found in report")
    baseline = {r["question_id"]: r for r in systems[args.baseline]["questions"]}
    # Cluster by paper: questions from the same paper are not independent.
    clusters = {}
    for qid, row in baseline.items():
        clusters.setdefault(row["paper_id"], []).append(qid)
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(len(clusters), size=(args.resamples, len(clusters)))
    lines = [f"# Index comparison: {args.metric}", "",
        f"Questions: {len(baseline)}. Baseline: {args.baseline}. Seed: {args.seed}.", "",
        "| System | Score | Difference vs baseline (95% CI) | Search ms/query | Index bytes |",
        "|---|---:|---:|---:|---:|"]
    for name, result in systems.items():
        rows = {r["question_id"]: r for r in result["questions"]}
        if rows.keys() != baseline.keys():
            raise ValueError("paired comparison requires identical question IDs")
        sums, counts = [], []
        for qids in clusters.values():
            sums.append(sum(rows[q]["scores"][args.metric] - baseline[q]["scores"][args.metric] for q in qids))
            counts.append(len(qids))
        sampled = np.array(sums)[draws].sum(axis=1) / np.array(counts)[draws].sum(axis=1)
        lo, hi = np.quantile(sampled, [0.025, 0.975])
        delta = sum(sums) / sum(counts)
        lines.append(f"| {name} | {result['metrics'][args.metric]:.4f} | {delta:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                     f"{result['search_ms_per_question']:.3f} | {result['serialized_bytes']} |")
    lines += ["", "Intervals use paired bootstrap resampling by paper (question context for HotpotQA).",
              "Small samples give unstable intervals. These timings exclude query encoding and generation.",
              "BM25 uses the team's existing implementation; see the sparse audit for reference differences."]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
