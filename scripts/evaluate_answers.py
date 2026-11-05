"""Score saved answers; generation and LLM serving are separate team components."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag.answer_evaluation import evaluate_answers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--dataset", choices=["qasper", "hotpot"], default="qasper")
    p.add_argument("--question-ids", type=Path, help="JSON array defining an explicit evaluation subset")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    predictions = json.loads(a.predictions.read_text())
    if a.dataset == "hotpot" and "answer" in predictions:
        predictions = predictions["answer"]
    report = evaluate_answers(json.loads(a.gold.read_text()), predictions, a.dataset,
                              json.loads(a.question_ids.read_text()) if a.question_ids else None)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k != "questions"})


if __name__ == "__main__":
    main()
