"""Export all seven Hendrycks MATH subjects at a pinned Hub revision."""
import argparse
import json
from pathlib import Path
import re

from .data import SPECS, last_boxed, question_key
from .study_io import file_hash, write_json


def main():
    from huggingface_hub import snapshot_download
    from pyarrow.parquet import read_table
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", args.revision):
        parser.error("Supply the full Hub dataset commit SHA")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory")
    dataset = SPECS["math"][0]
    snapshot = Path(snapshot_download(dataset, repo_type="dataset", revision=args.revision,
                                      allow_patterns=["*/train-*.parquet", "*/test-*.parquet"], max_workers=2))
    subjects = sorted(path.name for path in snapshot.iterdir() if path.is_dir())
    if len(subjects) != 7:
        raise ValueError(f"Expected 7 MATH configurations, found {subjects}")
    splits = {}
    for split, expected in (("train", 7500), ("test", 5000)):
        rows = []
        for subject in subjects:
            shards = sorted((snapshot / subject).glob(f"{split}-*.parquet"))
            if not shards:
                raise ValueError(f"Missing {split} Parquet for {subject}")
            subject_rows = (row for shard in shards for row in read_table(shard).to_pylist())
            for i, row in enumerate(subject_rows):
                rows.append({**row, "id": f"math:{split}:{subject}:{i}"})
        if len(rows) != expected:
            raise ValueError(f"Unexpected {split} size: {len(rows)}")
        splits[split] = rows
    test_keys = {question_key(row["problem"]) for row in splits["test"]}
    if len(test_keys) != len(splits["test"]):
        raise ValueError("Duplicate test questions require a grouped evaluation protocol")
    for row in splits["test"]:
        if not last_boxed(row["solution"]):
            raise ValueError(f"Missing boxed MATH test reference: {row['id']}")
    splits["train"] = [row for row in splits["train"] if question_key(row["problem"]) not in test_keys]
    overlap = 7500 - len(splits["train"])
    ungraded = [row["id"] for row in splits["train"] if not last_boxed(row["solution"])]
    excluded = set(ungraded)
    splits["train"] = [row for row in splits["train"] if row["id"] not in excluded]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": dataset, "revision": args.revision, "subjects": subjects,
                "train_test_overlap_excluded": overlap, "train_missing_reference_ids": ungraded, "splits": {}}
    for split, rows in splits.items():
        path = args.output / f"{split}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {"n": len(rows), "sha256": file_hash(path)}
    write_json(args.output / "manifest.json", manifest)
    print(f"MATH export: {len(splits['test'])} test, {len(splits['train'])} eligible train -> {args.output}")


if __name__ == "__main__":
    main()
