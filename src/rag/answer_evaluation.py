"""Answer F1 following the official QASPER and HotpotQA evaluation conventions.

References:
https://github.com/allenai/qasper-led-baseline/blob/main/scripts/evaluator.py
https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
Scores are fractions, not percentages. Missing predictions count as zero.
"""
import re
import string
from collections import Counter


def normalize_answer(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_f1(prediction, reference, *, hotpot=False):
    prediction, reference = normalize_answer(prediction), normalize_answer(reference)
    if hotpot and prediction != reference and ({prediction, reference} & {"yes", "no", "noanswer"}):
        return 0.0
    p, r = prediction.split(), reference.split()
    overlap = sum((Counter(p) & Counter(r)).values())
    return 2 * overlap / (len(p) + len(r)) if overlap else 0.0


def qasper_references(data):
    gold = {}
    for paper in data.values():
        for qa in paper["qas"]:
            refs = []
            for annotation in qa["answers"]:
                a = annotation["answer"]
                if a["unanswerable"]:
                    answer, kind = "Unanswerable", "none"
                elif a.get("extractive_spans"):
                    answer, kind = ", ".join(a["extractive_spans"]), "extractive"
                elif a.get("free_form_answer"):
                    answer, kind = a["free_form_answer"], "abstractive"
                elif a.get("yes_no") is not None:
                    answer, kind = "Yes" if a["yes_no"] else "No", "boolean"
                else:
                    raise ValueError(f"missing gold answer: {qa['question_id']}")
                refs.append((answer, kind))
            if not refs or qa["question_id"] in gold:
                raise ValueError("missing references or duplicate question ID")
            gold[qa["question_id"]] = refs
    return gold


def evaluate_answers(data, predictions, dataset="qasper", question_ids=None):
    if dataset not in ("qasper", "hotpot"):
        raise ValueError("dataset must be qasper or hotpot")
    gold = (qasper_references(data) if dataset == "qasper" else
            {r["_id"]: [(r["answer"], r.get("type", "unknown"))] for r in data})
    if question_ids is not None:
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("duplicate evaluation question IDs")
        gold = {qid: gold[qid] for qid in question_ids}
    if not gold:
        raise ValueError("no questions to evaluate")
    rows, missing = [], 0
    for qid, refs in gold.items():
        pred = predictions.get(qid)
        if isinstance(pred, dict):
            pred = pred["answer"]
        if pred is None:
            missing += 1
            score, kind, em = 0.0, "missing", 0.0
        else:
            if not isinstance(pred, str):
                raise ValueError(f"prediction must be text: {qid}")
            score, kind = max(((answer_f1(pred, ref, hotpot=dataset == "hotpot"), kind)
                              for ref, kind in refs), key=lambda pair: pair[0])
            em = float(any(normalize_answer(pred) == normalize_answer(ref) for ref, _ in refs))
        rows.append({"question_id": qid, "answer_f1": score, "exact_match": em, "answer_type": kind})
    return {"n_questions": len(rows), "missing_predictions": missing,
            "answer_f1": sum(r["answer_f1"] for r in rows) / len(rows),
            "exact_match": sum(r["exact_match"] for r in rows) / len(rows), "questions": rows}
