"""Prepare immutable Qwen assets, run three paper cells, and summarize results.

All large files live under --data-root. No Llama assets or gated access required.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .data import SPECS, load_examples

MODEL = "Qwen/Qwen2.5-7B-Instruct"
TASKS = ("csqa", "gsm8k", "aqua")
PAPER = {
    "csqa": {"mean_logprob": 61.94, "nci": 71.60, "fdbd_all": 71.50, "fdbd_k100": 72.47},
    "gsm8k": {"mean_logprob": 71.54, "nci": 75.83, "fdbd_all": 77.19},
    "aqua": {"mean_logprob": 71.66, "nci": 78.19, "fdbd_all": 77.08, "fdbd_k100": 78.22},
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def storage(data_root):
    root = Path(data_root).expanduser().resolve()
    locations = {"HF_HOME": root / "cache/huggingface", "HF_HUB_CACHE": root / "cache/huggingface/hub",
                 "HF_DATASETS_CACHE": root / "cache/huggingface/datasets", "TMPDIR": root / "tmp"}
    for key, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    return root


def experiment_dir(root, name):
    if not name or name in {".", ".."} or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Experiment name must contain only letters, digits, underscores and hyphens")
    return root / "results/OODmethod" / name


def read_recipe(path):
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    if recipe["model_id"] != MODEL or set(recipe["datasets"]) != set(TASKS):
        raise ValueError("Recipe is not a complete Qwen2.5-7B three-task recipe")
    model_path = Path(recipe["model_path"])
    if not (model_path / "config.json").is_file() or not list(model_path.glob("*.safetensors")):
        raise ValueError(f"Missing model weights/config at {model_path}; rerun prepare after checking storage")
    index = model_path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        missing = [shard for shard in set(weight_map.values()) if not (model_path / shard).is_file()]
        if missing:
            raise ValueError(f"Incomplete model download, missing shards: {missing}")
    return recipe


def prepare(root, name, revision):
    recipe_path = experiment_dir(root, name) / "recipe.json"
    if recipe_path.exists():
        recipe = read_recipe(recipe_path)
        if recipe["requested_model_revision"] != revision:
            raise ValueError("Existing recipe uses another revision; use a new --name")
        print(f"Reusing pinned recipe: {recipe_path}", flush=True)
        return recipe
    # Import only after storage() so Hub/Datasets see the requested cache paths.
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=False)
    sha = api.model_info(MODEL, revision=revision).sha
    model_path = root / "models/Qwen2.5-7B-Instruct" / sha
    print(f"Downloading {MODEL}@{sha} to {model_path}", flush=True)
    snapshot_download(MODEL, revision=sha, token=False, local_dir=str(model_path),
                      allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.jinja", "merges.txt", "vocab.*"])
    datasets = {}
    for task in TASKS:
        repo_id, config, split, expected = SPECS[task]
        data_sha = api.dataset_info(repo_id).sha
        examples, metadata = load_examples(task, revision=data_sha)
        if len(examples) != expected:
            raise ValueError(f"Unexpected {task} count: {len(examples)}")
        datasets[task] = {"repo_id": repo_id, "config": config, "split": split, "revision": data_sha,
                          "expected_count": expected, "sample_hash": metadata["sample_hash"]}
        print(f"Prepared {task}: {expected} questions, revision={data_sha}", flush=True)
    recipe = {"model_id": MODEL, "requested_model_revision": revision, "model_revision": sha,
              "model_path": str(model_path), "data_root": str(root), "datasets": datasets,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": "paper v4 fixed-k comparison; not independent k selection"}
    write_json(recipe_path, recipe)
    read_recipe(recipe_path)
    print(f"Prepared recipe: {recipe_path}", flush=True)
    return recipe


def build_command(recipe, result_root, task, phase, smoke_limit, dtype, bootstrap):
    repo_root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "ood_repro", "run", "--config", str(repo_root / "configs" / f"qwen_{task}.json"),
               "--model", recipe["model_path"], "--model-revision", recipe["model_revision"],
               "--tokenizer-revision", recipe["model_revision"],
               "--dataset-revision", recipe["datasets"][task]["revision"],
               "--local-files-only", "--dtype", dtype, "--resume", "--verify-replay", "3",
               "--output", str(result_root / phase / f"qwen_{task}")]
    if phase == "smoke":
        # A cheap alignment/protocol check, not the paper's selected-k experiment.
        command += ["--limit", str(smoke_limit), "--ks", "100", "--bootstrap", "0", "--save-token-scores"]
    else:
        command += ["--bootstrap", str(bootstrap)]
    return command


def summarize(result_root, recipe, phase):
    rows = []
    for task in TASKS:
        cell = result_root / phase / f"qwen_{task}"
        manifest = json.loads((cell / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((cell / "metrics.json").read_text(encoding="utf-8"))
        if manifest["config"]["model"] != recipe["model_path"] or manifest["config"]["task"] != task:
            raise ValueError(f"Model/task mismatch in {cell}")
        if manifest["config"]["dataset_revision"] != recipe["datasets"][task]["revision"]:
            raise ValueError(f"Dataset revision mismatch in {cell}")
        expected = len(manifest["dataset"]["sample_ids"])
        if metrics["n"] != expected:
            raise ValueError(f"Incomplete {cell}: {metrics['n']}/{expected}")
        if phase == "main" and expected != recipe["datasets"][task]["expected_count"]:
            raise ValueError(f"Main cell is not a full evaluation split: {cell}")
        methods = PAPER[task] if phase == "main" else {key: None for key in ("mean_logprob", "nci", "fdbd_k100")}
        for method, paper in methods.items():
            value = metrics["auroc"].get(method)
            ci = metrics.get("auroc_ci95", {}).get(method)
            rows.append({"task": task, "method": method, "auroc_x100": None if value is None else value * 100,
                         "paper_auroc_x100": paper if phase == "main" else None,
                         "delta_points": None if value is None or phase != "main" else value * 100 - paper,
                         "ci95_x100": None if ci is None else [v * 100 for v in ci],
                         "n": metrics["n"], "n_scored": metrics["n_scored"], "accuracy": metrics["accuracy"],
                         "parse_failure_rate": metrics["parse_failure_rate"], "truncation_rate": metrics["truncation_rate"]})
    output = result_root / phase
    write_json(output / "summary.json", {"phase": phase, "paper": "arXiv:2602.07253v4 Table 4", "recipe": recipe, "rows": rows})
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# Qwen2.5-7B-Instruct ({phase})", "", "Paper values are references, not measured results.", "",
             "| Task | Score | Measured AUROC x100 | Paper | Delta points | Scored / total | Parse failure | Truncation |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda value: "undefined" if value is None else f"{value:.2f}"
    for row in rows:
        lines.append(f"| {row['task']} | {row['method']} | {fmt(row['auroc_x100'])} | {fmt(row['paper_auroc_x100'])} | {fmt(row['delta_points'])} | {row['n_scored']}/{row['n']} | {row['parse_failure_rate']:.1%} | {row['truncation_rate']:.1%} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return rows


def run_cells(root, name, phase, smoke_limit, dtype, bootstrap):
    result_root = experiment_dir(root, name)
    recipe = read_recipe(result_root / "recipe.json")
    # Cache contents may have been removed: verify pinned data again before GPU load.
    for task in TASKS:
        _, metadata = load_examples(task, revision=recipe["datasets"][task]["revision"])
        if metadata["sample_hash"] != recipe["datasets"][task]["sample_hash"]:
            raise ValueError(f"Dataset content changed for {task}")
    import torch
    if not torch.cuda.is_available():
        raise ValueError("A CUDA GPU is required for the 7B suite; this is not a tiny-model test")
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 unsupported; select --dtype float16 explicitly and report the protocol difference")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)}, free={free / 2**30:.1f}/{total / 2**30:.1f} GiB", flush=True)
    if phase == "main":
        for task in TASKS:
            smoke_path = result_root / "smoke" / f"qwen_{task}" / "metrics.json"
            if not smoke_path.exists():
                raise ValueError("Run the three-task smoke phase before main")
            metrics = json.loads(smoke_path.read_text(encoding="utf-8"))
            manifest = json.loads((smoke_path.parent / "manifest.json").read_text(encoding="utf-8"))
            if metrics["n"] != len(manifest["dataset"]["sample_ids"]) or metrics["n_scored"] == 0:
                raise ValueError(f"Smoke is incomplete or has no scored responses for {task}")
            protocol = {"model": recipe["model_path"], "model_revision": recipe["model_revision"],
                        "dataset_revision": recipe["datasets"][task]["revision"], "dtype": dtype}
            if any(manifest["config"][key] != value for key, value in protocol.items()):
                raise ValueError(f"Smoke/main protocol mismatch for {task}; rerun with consistent settings")
    repo_root = Path(__file__).resolve().parents[1]
    for task in TASKS:
        command = build_command(recipe, result_root, task, phase, smoke_limit, dtype, bootstrap)
        print(f"Starting {phase}/{task}; output: {result_root / phase / ('qwen_' + task)}", flush=True)
        subprocess.run(command, cwd=repo_root, check=True)
    return summarize(result_root, recipe, phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "smoke", "main", "report"])
    parser.add_argument("--data-root", default="/mnt/data/wxh/wdl")
    parser.add_argument("--name", default="qwen25-paper-v4")
    parser.add_argument("--model-revision", default="main", help="Resolved to an immutable model commit during prepare")
    parser.add_argument("--smoke-limit", type=int, default=20)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.smoke_limit < 1 or args.bootstrap < 0:
        parser.error("smoke-limit must be positive; bootstrap must be nonnegative")
    root = storage(args.data_root)
    if args.phase == "prepare":
        prepare(root, args.name, args.model_revision)
    elif args.phase == "report":
        result_root = experiment_dir(root, args.name)
        summarize(result_root, read_recipe(result_root / "recipe.json"), "main")
    else:
        run_cells(root, args.name, args.phase, args.smoke_limit, args.dtype, args.bootstrap)


if __name__ == "__main__":
    main()
