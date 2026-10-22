"""Error-positive AUROC and paired, question-level bootstrap."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(rows, bootstrap=2000, seed=0, reference="mean_logprob"):
    if not rows:
        raise ValueError("Empty prediction file")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs; evaluate seeds separately")
    valid = [r for r in rows if r.get("scores")]
    report = {"n": len(rows), "n_scored": len(valid), "n_unscored": len(rows) - len(valid),
              "accuracy": float(np.mean([r["is_correct"] for r in rows])),
              "parse_failure_rate": float(np.mean([r["parsed_answer"] is None for r in rows])),
              "truncation_rate": float(np.mean([r["stop_reason"] == "max_new_tokens" for r in rows])),
              "score_orientation": "all stored scores: higher confidence; AUROC uses error=1 and -score",
              "auroc": {}, "paired_delta_vs_reference": {}, "reference": reference}
    if not valid:
        return report
    keys = sorted(set.intersection(*(set(row["scores"]) for row in valid)))
    y = np.array([not r["is_correct"] for r in valid], dtype=int)
    values = {k: -np.array([r["scores"][k] for r in valid], dtype=float) for k in keys}
    if any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError("Nonfinite scores in prediction file")
    if len(np.unique(y)) < 2:
        report["undefined_reason"] = "AUROC requires both correct and incorrect answers"
        return report
    report["auroc"] = {k: float(roc_auc_score(y, v)) for k, v in values.items()}
    if bootstrap <= 0:
        return report
    samples = {k: [] for k in keys}
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        for k in keys:
            samples[k].append(float(roc_auc_score(y[idx], values[k][idx])))
    accepted = len(next(iter(samples.values()), []))
    report["bootstrap"] = {"requested": bootstrap, "accepted": accepted, "seed": seed,
                           "unit": "question", "ci": "percentile 95%"}
    if accepted:
        report["auroc_ci95"] = {k: np.quantile(v, [.025, .975]).tolist() for k, v in samples.items()}
        if reference in keys:
            for k in keys:
                delta = np.array(samples[k]) - np.array(samples[reference])
                report["paired_delta_vs_reference"][k] = {
                    "delta": report["auroc"][k] - report["auroc"][reference],
                    "ci95": np.quantile(delta, [.025, .975]).tolist()}
    return report


def select_k(run_dir, output):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset"]["split"] != "train":
        raise ValueError("k selection is restricted to split=train; do not tune on evaluation splits")
    rows = read_jsonl(run_dir / "predictions.jsonl")
    if len(rows) != len(manifest["dataset"]["sample_ids"]) or {r["sample_id"] for r in rows} != set(manifest["dataset"]["sample_ids"]):
        raise ValueError("Calibration run is incomplete")
    report = evaluate(rows, bootstrap=0)
    candidates = {k: v for k, v in report["auroc"].items() if k.startswith("fdbd_")}
    if not candidates:
        raise ValueError("No defined fDBD AUROC for k selection")
    winner = max(candidates, key=candidates.get)
    result = {"selected_metric": winner, "selected_k": "all" if winner == "fdbd_all" else int(winner[6:]),
              "calibration_auroc": candidates, "calibration_manifest": manifest,
              "rule": "maximum train-subset AUROC; ties use deterministic metric-name order"}
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result
