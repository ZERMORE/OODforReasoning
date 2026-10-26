"""Score existing trajectories with DeepConf confidence; never regenerate answers."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .cli import environment, load_model, model_args, write_json
from .data import digest
from .engine import replay
from .evaluation import evaluate, read_jsonl


class TopKConfidence:
    def __init__(self, k=20):
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k

    def score(self, z, logits, actual_token):
        logits = logits.float()
        if self.k > logits.numel():
            raise ValueError("k exceeds vocabulary size")
        logp = logits.log_softmax(-1)
        # Normalize over the FULL vocabulary, not over the selected top-k.
        return {"deepconf_mean": -logp.topk(self.k).values.mean(),
                "mean_logprob": logp[actual_token]}


def window_scores(values, window=64):
    """Stride-one full windows; traces shorter than window use one full trace."""
    values = np.asarray(values, dtype=float)
    if window < 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid window or confidence values")
    width = min(window, len(values))
    sums = np.r_[0., values.cumsum()]
    windows = (sums[width:] - sums[:-width]) / width
    count = max(1, math.ceil(.1 * len(windows)))
    return {"deepconf_mean": float(values.mean()),
            "deepconf_lowest": float(windows.min()),
            "deepconf_bottom10": float(np.sort(windows)[:count].mean())}


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def augment_row(model, row, scorer, config, stop_ids, window):
    result = replay(model, row["prompt_token_ids"], row["generated_token_ids"], scorer,
                    mode="matched", stop_ids=stop_ids, include_eos=config["include_eos"], save_tokens=True)
    if result["scored_tokens"] != row["scored_tokens"]:
        raise ValueError(f"Scored token count mismatch: {row['sample_id']}")
    # Confirm the replay agrees with the existing run before mixing new scores in.
    if not np.isclose(result["scores"]["mean_logprob"], row["scores"]["mean_logprob"],
                      rtol=config.get("verify_rtol", .01), atol=config.get("verify_atol", .001)):
        raise ValueError(f"Replay mean_logprob mismatch: {row['sample_id']}; check model and environment")
    values = [item["scores"]["deepconf_mean"] for item in result["token_scores"]]
    return {**row, "scores": {**row["scores"], **window_scores(values, window)},
            "deepconf_token_confidence": values,
            "deepconf_replay_mean_logprob": result["scores"]["mean_logprob"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_args(parser)
    parser.set_defaults(model_revision=None, tokenizer_revision=None, dtype=None, attention=None)
    parser.add_argument("--source", type=Path, required=True, help="Existing single-task run directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--top-k", type=int, choices=[20], default=20)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if args.window < 1 or args.top_k < 1 or args.bootstrap < 0:
        parser.error("window/top-k must be positive; bootstrap must be nonnegative")
    if output == source or source in output.parents or output in source.parents:
        parser.error("Output must be separate from source")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    rows = read_jsonl(source / "predictions.jsonl")
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(manifest["dataset"]["sample_ids"]):
        raise ValueError("Source run must be complete and contain unique IDs")
    for key in ("model", "model_revision", "tokenizer_revision", "dtype", "attention"):
        original = config.get(key)
        if key != "model" and getattr(args, key) not in (original, None):
            raise ValueError(f"Match source --{key.replace('_', '-')}: {original}")
        setattr(args, key, getattr(args, key) if key == "model" and args.model else original)
    args.seed = config.get("seed", 0)
    settings = {"top_k": args.top_k, "window": args.window, "stride": 1,
                "short_trace": "one full-trace window", "bottom_count": "ceil(0.1 * window_count)",
                "source_predictions_sha256": file_hash(source / "predictions.jsonl"),
                "source_manifest_sha256": file_hash(source / "manifest.json"),
                "source": str(source), "model": args.model, "mode": "matched teacher-forced replay",
                "scope": "single-trace confidence only; no voting, filtering or early stopping",
                "reference": "https://arxiv.org/abs/2508.15260"}
    signature = digest(settings)
    previous = []
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            raise ValueError("Output nonempty; use a new directory or --resume")
        saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if saved.get("deepconf", {}).get("signature") != signature:
            raise ValueError("Resume source/settings mismatch")
        previous = read_jsonl(output / "predictions.jsonl") if (output / "predictions.jsonl").exists() else []
        if len(previous) > len(rows) or any(
            {k: v for k, v in old.items() if k not in {"scores", "deepconf_token_confidence", "deepconf_replay_mean_logprob"}}
            != {k: v for k, v in src.items() if k != "scores"}
            or any(old["scores"].get(k) != v for k, v in src["scores"].items())
            or not all(k in old["scores"] for k in ("deepconf_mean", "deepconf_lowest", "deepconf_bottom10"))
            for old, src in zip(previous, rows)
        ):
            raise ValueError("Resume rows differ from source or lack new scores")
    output.mkdir(parents=True, exist_ok=True)
    if len(previous) < len(rows):
        model, tokenizer = load_model(args)
        env = environment(model, tokenizer)
        source_env = manifest["environment"]
        for key in ("model_commit", "tokenizer_commit", "chat_template_hash", "packages"):
            if source_env.get(key) != env.get(key):
                raise ValueError(f"Source/replay environment mismatch: {key}")
        if previous and saved["deepconf"]["environment"] != env:
            raise ValueError("Resume replay environment mismatch")
        copied_manifest = {**manifest, "deepconf": {**settings, "signature": signature, "environment": env}}
        write_json(output / "manifest.json", copied_manifest)
        from tqdm import tqdm
        scorer = TopKConfidence(args.top_k)
        with (output / "predictions.jsonl").open("a", encoding="utf-8") as stream:
            for row in tqdm(rows[len(previous):], desc=f"DeepConf/{config['task']}", initial=len(previous), total=len(rows)):
                augmented = augment_row(model, row, scorer, config, set(manifest["eos_token_ids"]), args.window)
                stream.write(json.dumps(augmented, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
    report = evaluate(read_jsonl(output / "predictions.jsonl"), args.bootstrap, args.seed)
    write_json(output / "metrics.json", report)
    for reference in ("nci", "logit_margin", "fdbd_all", "fdbd_k100"):
        if reference in rows[0]["scores"]:
            paired_rows = [{**r, "scores": {k: v for k, v in r["scores"].items()
                           if k.startswith("deepconf_") or k == reference}}
                           for r in read_jsonl(output / "predictions.jsonl")]
            write_json(output / f"deepconf_vs_{reference}.json",
                       evaluate(paired_rows, args.bootstrap, args.seed, reference=reference))
    print(f"DeepConf scores, AUROC, CI and paired comparisons saved to {output}")


if __name__ == "__main__":
    main()
