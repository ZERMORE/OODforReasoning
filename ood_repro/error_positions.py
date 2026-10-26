"""E3: export a human-review template; analyze error buckets with correct controls."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .aggregation import aggregate
from .evaluation import read_jsonl
from .sequence_analysis import load_replay
from .study_io import independent_output, write_json


def validate_annotations(items, rows):
    lookup = {r["sample_id"]: r for r in rows}
    if len({a["sample_id"] for a in items}) != len(items):
        raise ValueError("Duplicate annotation IDs")
    for annotation in items:
        if annotation["sample_id"] not in lookup or annotation.get("reviewed") is not True:
            raise ValueError("Annotations must be reviewed and match source IDs")
        if type(annotation.get("process_valid")) is not bool:
            raise ValueError("process_valid must be a reviewed boolean")
        position = annotation.get("first_error_position")
        row = lookup[annotation["sample_id"]]
        if annotation.get("source_row_hash") != row["source_row_hash"]:
            raise ValueError("Annotation source hash mismatch")
        if position is not None and (type(position) is not int or position not in {t["position"] for t in row["sequence_tokens"]}):
            raise ValueError("first_error_position must be a scored generated-token index")
        if annotation["process_valid"] and position is not None:
            raise ValueError("A valid process cannot have an error position")
    return {a["sample_id"]: a for a in items}


def analyze_positions(rows, calibration, annotations, metric, aggregation, bootstrap=2000, seed=0):
    def confidence(row):
        values = [t["scores"][metric] for t in row["sequence_tokens"]]
        return aggregate(values)[aggregation]
    calibration_correct = [confidence(r) for r in calibration if r["is_correct"]]
    if not calibration_correct:
        raise ValueError("Calibration needs correct answers")
    threshold = float(np.quantile(calibration_correct, .05))
    annotated_errors = [r for r in rows if not r["is_correct"] and r["sample_id"] in annotations
                        and annotations[r["sample_id"]]["first_error_position"] is not None]
    controls = [r for r in rows if r["is_correct"]]
    if not controls:
        raise ValueError("Need a correct-answer control pool for AUROC")
    control_scores = np.array([confidence(r) for r in controls])
    errors = np.array([confidence(r) for r in annotated_errors])
    positions = np.array([min(3, int(4 * annotations[r["sample_id"]]["first_error_position"] / len(r["generated_token_ids"])))
                          for r in annotated_errors])
    samples = {i: {"tpr": [], "auroc": []} for i in range(4)}
    all_scores = np.r_[control_scores, errors]
    all_positions = np.r_[np.full(len(controls), -1), positions]
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap):
        # One draw of question IDs shared by all buckets, including reused controls.
        indices = rng.integers(0, len(all_scores), len(all_scores))
        draw_score, draw_position = all_scores[indices], all_positions[indices]
        correct = draw_position == -1
        for bucket in range(4):
            wrong = draw_position == bucket
            if wrong.any():
                samples[bucket]["tpr"].append(float((draw_score[wrong] < threshold).mean()))
                if correct.any():
                    selected = correct | wrong
                    samples[bucket]["auroc"].append(float(roc_auc_score(wrong[selected], -draw_score[selected])))
    output = []
    for bucket in range(4):
        values = errors[positions == bucket]
        combined = np.r_[control_scores, values]
        labels = np.r_[np.zeros(len(controls)), np.ones(len(values))]
        intervals = {key: np.quantile(value, [.025, .975]).tolist() if value else None
                     for key, value in samples[bucket].items()}
        output.append({"position_bucket": bucket, "n_errors": len(values), "n_controls": len(controls),
                       "tpr": float((values < threshold).mean()) if len(values) else None,
                       "auroc": float(roc_auc_score(labels, -combined)) if len(values) else None,
                       "ci95": intervals, "descriptive_only": len(values) < 20})
    return {"metric": metric, "aggregation": aggregation, "threshold": threshold,
            "calibration_correct_n": len(calibration_correct),
            "calibration_actual_fpr": float((np.array(calibration_correct) < threshold).mean()),
            "test_actual_fpr": float((control_scores < threshold).mean()), "buckets": output,
            "annotated_errors_with_position": len(annotated_errors),
            "unlocated_or_unreviewed_final_errors": sum(not r["is_correct"] for r in rows) - len(annotated_errors),
            "final_correct_with_process_error": sum(r["is_correct"] and r["sample_id"] in annotations
                                                     and not annotations[r["sample_id"]]["process_valid"] for r in rows),
            "warning": "Observational, shared correct controls, no length matching; small buckets descriptive"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "analyze"])
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--annotations")
    parser.add_argument("--calibration", help="Completed replay of an independent train run")
    parser.add_argument("--metrics", default="fdbd_all,nci,logit_margin,deepconf_top20")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    source, output = independent_output(args.replay, args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new empty directory")
    manifest, rows = load_replay(source)
    if args.command == "prepare":
        output.mkdir(parents=True, exist_ok=True)
        with (output / "annotations.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps({"sample_id": row["sample_id"], "question": row["question"],
                    "source_row_hash": row["source_row_hash"],
                    "response": row["response"], "gold_answer": row["gold_answer"], "final_correct": row["is_correct"],
                    "parse_failure": row["parsed_answer"] is None, "truncated": row["stop_reason"] == "max_new_tokens",
                    "generated_token_ids": row["generated_token_ids"], "token_pieces": row.get("token_pieces"),
                    "reviewed": False, "process_valid": None, "first_error_position": None,
                    "recovered": None, "steps": [], "reviewer": "", "notes": ""}, ensure_ascii=False) + "\n")
        print(f"Fill and independently review: {output / 'annotations.jsonl'}")
        return
    if not args.annotations or not args.calibration or args.bootstrap < 2:
        parser.error("analyze requires --annotations, --calibration and bootstrap >= 2")
    calibration_manifest, calibration = load_replay(args.calibration)
    if calibration_manifest["source_manifest"]["dataset"]["split"] != "train":
        raise ValueError("Calibration must use train, never a test split")
    from .data import question_key
    if {r["sample_id"] for r in rows} & {r["sample_id"] for r in calibration} or {
        question_key(r["question"]) for r in rows} & {question_key(r["question"]) for r in calibration}:
        raise ValueError("Calibration overlaps evaluated questions")
    left, right = manifest["source_manifest"], calibration_manifest["source_manifest"]
    for key in ("task", "model_revision", "tokenizer_revision", "dtype", "attention", "temperature", "max_new_tokens", "include_eos"):
        if left["config"].get(key) != right["config"].get(key):
            raise ValueError(f"Calibration protocol mismatch: {key}")
    if left.get("pack_hash") != right.get("pack_hash"):
        raise ValueError("Calibration prompt pack differs")
    for key in ("model_commit", "tokenizer_commit", "chat_template_hash"):
        if left["environment"].get(key) != right["environment"].get(key):
            raise ValueError(f"Calibration checkpoint mismatch: {key}")
    annotations = validate_annotations(read_jsonl(args.annotations), rows)
    result = [analyze_positions(rows, calibration, annotations, metric, aggregation, args.bootstrap, args.seed)
              for metric in args.metrics.split(",") for aggregation in ("mean", "lowest_w64")]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "positions.json", result)
    print(f"E3 observational position report: {output}")


if __name__ == "__main__":
    main()
