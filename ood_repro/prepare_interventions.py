"""Prepare editable E4 events from independently reviewed correct trajectories."""
import argparse
import json

import numpy as np

from .aggregation import aggregate
from .error_positions import validate_annotations
from .evaluation import read_jsonl
from .sequence_analysis import load_replay
from .study_io import file_hash, independent_output, write_json


def select_steps(rows, annotations, count=100, seed=0):
    candidates = []
    for row in rows:
        annotation = annotations.get(row["sample_id"])
        if not annotation or not annotation["process_valid"] or not row["is_correct"]:
            continue
        steps = annotation.get("steps", [])
        if not steps:
            continue
        aggregate(np.zeros(row["scored_tokens"]), steps=steps)  # Validate exhaustive scored-position partition.
        eligible = [(start, end) for start, end in steps if end <= .9 * row["scored_tokens"]]
        if eligible:
            candidates.append((row, eligible))
    rng = np.random.default_rng(seed)
    selected = [candidates[i] for i in rng.permutation(len(candidates))[:count]]
    result = []
    for row, steps in selected:
        used = set()
        for target in (.25, .75):
            remaining = [step for step in steps if step not in used]
            if not remaining:
                break
            step = min(remaining, key=lambda pair: abs(sum(pair) / 2 / row["scored_tokens"] - target))
            used.add(step)
            start, end = step
            result.append({"event_id": f"{row['sample_id']}@{target}", "sample_id": row["sample_id"],
                           "source_row_hash": row["source_row_hash"], "original_process_valid": True,
                           "start": row["sequence_tokens"][start]["position"],
                           "end": row["sequence_tokens"][end-1]["position"] + 1,
                           "target_fraction": target, "reviewed": False,
                           "error_step_text": None, "sham_step_text": None, "reviewer": "", "notes": ""})
    return result, len(candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", help="Tokenizer at same checkpoint's new local path")
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    source, output = independent_output(args.replay, args.output)
    if args.max_questions < 1 or (output.exists() and any(output.iterdir())):
        parser.error("Positive question count and new output required")
    manifest, rows = load_replay(source)
    annotations = validate_annotations(read_jsonl(args.annotations), rows)
    events, candidate_count = select_steps(rows, annotations, args.max_questions, args.seed)
    if not events:
        raise ValueError("No reviewed correct trajectories with eligible annotated steps")
    from transformers import AutoTokenizer
    config = manifest["source_manifest"]["config"]
    tokenizer = AutoTokenizer.from_pretrained(args.model or config["model"], local_files_only=True, trust_remote_code=False,
                                             revision=config.get("tokenizer_revision") or config["model_revision"])
    lookup = {r["sample_id"]: r for r in rows}
    for event in events:
        row = lookup[event["sample_id"]]
        event["original_step_text"] = tokenizer.decode(row["generated_token_ids"][event["start"]:event["end"]], skip_special_tokens=False)
        event["question"] = row["question"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "events.jsonl").open("w", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(output / "selection.json", {"candidate_questions": candidate_count, "selected_questions": len({e["sample_id"] for e in events}),
               "events": len(events), "seed": args.seed, "annotations_hash": file_hash(args.annotations),
               "note": "Candidate steps require human verification of arithmetic/editability; review both edited branches before running."})
    print(f"Review and fill error/sham edits, then mark reviewed=true: {output / 'events.jsonl'}")


if __name__ == "__main__":
    main()
