"""Frozen question-level train/validation/calibration splits for eval section 2."""
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .data import question_key
from .sequence_analysis import load_replay
from .study_io import file_hash


def compatible(left, right):
    a, b = left["source_manifest"], right["source_manifest"]
    for key in ("task", "model_revision", "tokenizer_revision", "dtype", "attention", "temperature",
                "top_p", "max_new_tokens", "include_eos", "prompt_mode"):
        if a["config"].get(key) != b["config"].get(key):
            raise ValueError(f"Pool/test protocol mismatch: {key}")
    if a.get("pack_hash") != b.get("pack_hash"):
        raise ValueError("Prompt pack mismatch")
    for key in ("model_commit", "tokenizer_commit", "chat_template_hash", "packages"):
        if a["environment"].get(key) != b["environment"].get(key):
            raise ValueError(f"Checkpoint/tokenizer mismatch: {key}")


def build_split(pool, test, seed=1729):
    pool_groups, test_groups = {}, {question_key(r["question"]) for r in test}
    if {r["sample_id"] for r in pool} & {r["sample_id"] for r in test}:
        raise ValueError("Pool/test sample IDs overlap")
    for row in pool:
        key = question_key(row["question"])
        if key in test_groups:
            raise ValueError("Pool/test question content overlaps")
        pool_groups.setdefault(key, []).append(row)
    keys = sorted(pool_groups)
    labels = [int(any(not r["is_correct"] for r in pool_groups[k])) for k in keys]
    if len(set(labels)) != 2 or min(np.bincount(labels)) < 8:
        raise ValueError("Pool needs >=8 correct and >=8 error question groups; collect more train data")
    train, rest = train_test_split(keys, test_size=.3, random_state=seed, stratify=labels)
    rest_labels = [int(any(not r["is_correct"] for r in pool_groups[k])) for k in rest]
    validation, calibration = train_test_split(rest, test_size=.5, random_state=seed + 1, stratify=rest_labels)
    return {name: [r["sample_id"] for key in group for r in pool_groups[key]]
            for name, group in (("train", train), ("validation", validation), ("calibration", calibration))}


def load_study(pool_path, test_path, split_path):
    pool_manifest, pool = load_replay(pool_path)
    test_manifest, test = load_replay(test_path)
    compatible(pool_manifest, test_manifest)
    if pool_manifest["source_manifest"]["dataset"]["split"] != "train":
        raise ValueError("Supervised pool must come from the dataset train split")
    if test_manifest["source_manifest"]["dataset"]["split"] == "train":
        raise ValueError("Evaluation replay must be a held-out dataset split")
    if len({question_key(r["question"]) for r in test}) != len(test):
        raise ValueError("This evaluator requires one test trajectory per question for question bootstrap")
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    for field, path in (("pool_sha256", pool_path), ("test_sha256", test_path)):
        if split[field] != file_hash(Path(path) / "predictions.jsonl"):
            raise ValueError(f"Frozen split content mismatch: {field}")
    lookup = {r["sample_id"]: r for r in pool}
    ids = [i for key in ("train", "validation", "calibration") for i in split["ids"][key]]
    if len(ids) != len(set(ids)) or set(ids) != set(lookup):
        raise ValueError("Split IDs are missing, overlapping or unknown")
    partitions = {key: [lookup[i] for i in split["ids"][key]] for key in ("train", "validation", "calibration")}
    groups = [{question_key(r["question"]) for r in partitions[key]} for key in partitions]
    if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]):
        raise ValueError("Question groups cross training partitions")
    if set(ids) & {r["sample_id"] for r in test} or set.union(*groups) & {question_key(r["question"]) for r in test}:
        raise ValueError("Pool/test overlap")
    return partitions, test, split


def sequences(rows, metric):
    arrays = [np.array([t["scores"][metric] for t in row["sequence_tokens"]], dtype=np.float32)[:, None] for row in rows]
    if any(not len(x) or not np.isfinite(x).all() for x in arrays):
        raise ValueError("Invalid input sequences")
    return arrays


def pooling(values):
    """Keep q10 distinct from the MEAN of the lowest 10%; preserve legacy E1 API."""
    from .aggregation import aggregate
    result = aggregate(values)
    result["quantile10"] = float(np.quantile(values, .1))
    return result
