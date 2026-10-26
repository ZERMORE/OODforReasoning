"""Prepare a train-only calibration config inheriting the frozen original protocol."""
import argparse
import json
from pathlib import Path

from .study_io import independent_output, source_run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True, help="New directory containing config.json")
    parser.add_argument("--run-output", required=True, help="New raw calibration run directory")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--save-token-scores", action="store_true", help="Retain online scores for eval2, avoiding a second model pass")
    args = parser.parse_args()
    source, output = independent_output(args.source, args.output)
    _, run_output = independent_output(source, args.run_output)
    if args.limit < 1 or (output.exists() and any(output.iterdir())) or (run_output.exists() and any(run_output.iterdir())):
        parser.error("Positive limit and new output/config directories required")
    manifest, _ = source_run(source)
    if manifest["dataset"]["split"] == "train":
        parser.error("Source should be the existing evaluation run")
    config = dict(manifest["config"])
    config.update(split="train", limit=args.limit, seed=args.seed, data_jsonl=None, k_selection=None,
                  output=str(run_output), bootstrap=0, resume=True, save_token_scores=args.save_token_scores)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    print(f"python -m ood_repro run --config {output / 'config.json'}")
    print("Training questions exclude prompt demonstrations using the existing loader; verify calibration/test overlap before E3.")


if __name__ == "__main__":
    main()
