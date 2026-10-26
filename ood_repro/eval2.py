"""Eval section 2: freeze splits or prepare blinded first-error annotation files."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .eval2_data import build_split, compatible
from .sequence_analysis import load_replay
from .study_io import file_hash, independent_output, seal_record, source_run, start_output, write_json
from .data import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["split", "prepare-annotations", "import-online"])
    parser.add_argument("--test", required=True)
    parser.add_argument("--pool")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--correct-count", type=int, help="Default: review as many final-correct as final-error cases")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source, output = independent_output(args.test, args.output)
    if args.command == "import-online":
        manifest, rows = source_run(source)
        settings = {"kind": "eval2-online-import", "schema": 1, "count": len(rows), "source": str(source),
                    "source_predictions_sha256": file_hash(source / "predictions.jsonl"),
                    "source_manifest_sha256": file_hash(source / "manifest.json")}
        signature, previous, saved = start_output(output, settings, args.resume)
        if len(previous) > len(rows) or any(old["source_row_hash"] != digest(row) for old, row in zip(previous, rows)):
            raise ValueError("Imported prefix differs from source")
        saved = {"settings": settings, "signature": signature, "complete": False, "source_manifest": manifest,
                 "dataset": {"sample_ids": [r["sample_id"] for r in rows]}}
        write_json(output / "manifest.json", saved)
        with (output / "predictions.jsonl").open("a", encoding="utf-8") as stream:
            for row in rows[len(previous):]:
                tokens = row.get("token_scores", [])
                if not tokens or len(tokens) != row["scored_tokens"]:
                    raise ValueError("Online tokens missing: generate the pool using --save-token-scores, or use replay_scores")
                keys = set(tokens[0]["scores"])
                if any(set(t["scores"]) != keys for t in tokens):
                    raise ValueError("Inconsistent online score keys")
                scores = {key: float(np.mean([t["scores"][key] for t in tokens])) for key in keys}
                if "mean_logprob" in scores:
                    scores["log_predictive_probability"] = scores["mean_logprob"] * len(tokens)
                audit = {}
                for key, old in row["scores"].items():
                    if key not in scores or not np.isfinite(scores[key]) or not np.isclose(scores[key], old, rtol=1e-5, atol=1e-6):
                        raise ValueError(f"Online aggregate mismatch: {row['sample_id']}/{key}")
                    audit[key] = {"old": old, "replayed": scores[key], "abs_diff": abs(scores[key] - old)}
                record = seal_record({**row, "source_row_hash": digest(row), "sequence_tokens": tokens,
                                      "sequence_scores": scores, "score_audit": audit})
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
        saved["complete"] = True
        write_json(output / "manifest.json", saved)
        print(f"Imported online scores without another model pass: {output}")
        return
    if output.exists() and any(output.iterdir()):
        parser.error("New empty output required")
    manifest, rows = load_replay(source)
    if args.command == "split":
        if not args.pool:
            parser.error("split requires --pool")
        independent_output(args.pool, output)
        pool_manifest, pool = load_replay(args.pool)
        compatible(pool_manifest, manifest)
        if pool_manifest["source_manifest"]["dataset"]["split"] != "train" or manifest["source_manifest"]["dataset"]["split"] == "train":
            raise ValueError("Use a train pool and held-out test replay")
        ids = build_split(pool, rows, args.seed)
        lookup = {r["sample_id"]: r for r in pool}
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "splits.json", {"schema": 1, "seed": args.seed, "ids": ids,
            "counts": {name: {"n": len(items), "errors": sum(not lookup[i]["is_correct"] for i in items)} for name, items in ids.items()},
            "pool_sha256": file_hash(Path(args.pool) / "predictions.jsonl"), "test_sha256": file_hash(source / "predictions.jsonl"),
            "test_ids": [r["sample_id"] for r in rows], "note": "70/15/15 question-group split; group class stratified by any final error; freeze before fitting"})
        print(f"Frozen splits: {output / 'splits.json'}")
        return
    wrong = [r for r in rows if not r["is_correct"]]
    correct = [r for r in rows if r["is_correct"]]
    count = len(wrong) if args.correct_count is None else args.correct_count
    if count < 0:
        parser.error("correct-count must be nonnegative")
    rng = np.random.default_rng(args.seed)
    sampled = [correct[i] for i in rng.permutation(len(correct))[:count]]
    selected = wrong + sampled
    rng.shuffle(selected)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "annotations.jsonl").open("w", encoding="utf-8") as stream, (output / "tokens.tsv").open("w", encoding="utf-8-sig", newline="") as tokens:
        writer = csv.writer(tokens, delimiter="\t")
        writer.writerow(["sample_id", "generated_position_0_based", "token_id", "token_piece"])
        for row in selected:
            stream.write(json.dumps({"sample_id": row["sample_id"], "source_row_hash": row["source_row_hash"],
                "question": row["question"], "response": row["response"], "gold_answer": row["gold_answer"],
                "parse_failure": row["parsed_answer"] is None, "truncated": row["stop_reason"] == "max_new_tokens",
                "reviewed": False, "process_valid": None, "first_error_position": None, "first_error_span_end": None,
                "first_error_step": None, "error_type": None, "recovered": None, "steps": [], "reviewer": "", "notes": ""}, ensure_ascii=False) + "\n")
            for token in row["sequence_tokens"]:
                pieces = row.get("token_pieces") or []
                piece = pieces[token["position"]] if pieces else ""
                writer.writerow([row["sample_id"], token["position"], token["token_id"], piece])
    write_json(output / "selection.json", {"seed": args.seed, "final_error_n": len(wrong), "sampled_final_correct_n": len(sampled),
        "sample_ids": [r["sample_id"] for r in selected], "source_sha256": file_hash(source / "predictions.jsonl"),
        "note": "No token scores or final correctness flags shown in review files; gold/response still permit correctness inference. Not fully blinded."})
    print(f"Review annotations.jsonl using tokens.tsv, then set reviewed=true: {output}")


if __name__ == "__main__":
    main()
