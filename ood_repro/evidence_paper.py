"""Shannon Mean/Drop paper-style selective evaluation on a frozen half split."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .eval2_data import sequences
from .evidence import detection_summary, fixed_scores, render_distributions
from .sequence_analysis import load_replay
from .study_io import file_hash, independent_output, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    source, output = independent_output(args.replay, args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new output directory")
    manifest, rows = load_replay(source)
    if manifest["source_manifest"]["dataset"]["split"] != "test":
        raise ValueError("Paper-style evaluation expects the dataset test split")
    errors = np.array([not row["is_correct"] for row in rows], dtype=int)
    cal, test = train_test_split(np.arange(len(rows)), test_size=.5, random_state=args.seed, stratify=errors)
    scores, _ = fixed_scores(sequences(rows, "shannon_top20"))
    results = {name: detection_summary(errors[test], scores[name][test], errors[cal], scores[name][cal])
               for name in ("mean", "evidence_drop")}
    import matplotlib
    matplotlib.use("Agg")
    output.mkdir(parents=True, exist_ok=True)
    render_distributions({f"shannon_top20/{name}": {"fixed": scores[name][test]} for name in results}, errors[test], output)
    write_json(output / "paper_style.json", {"source_sha256": file_hash(source / "predictions.jsonl"),
               "model": manifest["source_manifest"]["config"].get("model"), "seed": args.seed,
               "calibration_ids": [rows[i]["sample_id"] for i in cal], "evaluation_ids": [rows[i]["sample_id"] for i in test],
               "evaluation_accuracy": float(1-errors[test].mean()), "results": results,
               "threshold_rule": "Accept strictly below the lower-tail empirical wrong-answer risk cutoff; empirical protocol, no finite-sample guarantee claimed."})


if __name__ == "__main__":
    main()
