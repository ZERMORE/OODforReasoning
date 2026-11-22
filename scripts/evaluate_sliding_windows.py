"""Evaluate minimum-confidence sliding windows on existing token trajectories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", default="4,8,16,32,64,128,256")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    widths = sorted(set(int(w) for w in args.windows.split(",")))
    if not widths or min(widths) < 1 or args.bootstrap < 2:
        parser.error("Positive windows and at least two bootstrap draws required")
    if args.output.exists():
        parser.error("Choose a new output directory")
    source = args.source / "predictions.jsonl" if args.source.is_dir() else args.source
    metrics = ("fdbd_all", "nci")
    values, labels, lengths, ids = [], [], [], []
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            tokens = row.get("sequence_tokens") or row.get("token_scores")
            if not tokens:
                raise ValueError(f"Token scores missing for {row['sample_id']}; replay required")
            n = row["scored_tokens"]
            if len(tokens) != n or not n:
                raise ValueError("Token count mismatch")
            if any(t["position"] != i or t["token_id"] != row["generated_token_ids"][i]
                   for i, t in enumerate(tokens)):
                raise ValueError("Token alignment mismatch")
            x = np.array([[t["scores"][metric] for metric in metrics] for t in tokens])
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite token scores")
            sums = np.vstack([np.zeros((1, len(metrics))), x.cumsum(0)])
            scores = [x.mean(0)]
            for width in widths:
                w = min(width, n)
                scores.append(((sums[w:] - sums[:-w]) / w).min(0))
            values.append(-np.array(scores).T.reshape(-1))
            labels.append(not row["is_correct"])
            lengths.append(n)
            ids.append(row["sample_id"])
    if len(set(ids)) != len(ids) or len(set(labels)) != 2:
        raise ValueError("Unique sample IDs and both label classes required")
    values, y, lengths = np.array(values), np.array(labels), np.array(lengths)
    names = [(metric, name) for metric in metrics for name in ["mean", *widths]]
    rng = np.random.default_rng(args.seed)
    draws = np.empty((args.bootstrap, len(names)))
    for b in range(args.bootstrap):
        index = rng.integers(len(y), size=len(y))
        positive = y[index].sum()
        if positive == 0 or positive == len(y):
            raise ValueError("Single-class bootstrap draw; use a larger sample")
        ranks = rankdata(values[index], axis=0)
        draws[b] = (ranks[y[index]].sum(0) - positive * (positive + 1) / 2) / (positive * (len(y) - positive))
    result = []
    for j, (metric, name) in enumerate(names):
        base = names.index((metric, "mean"))
        precision, recall, _ = precision_recall_curve(y, values[:, j])
        area = roc_auc_score(y, values[:, j])
        result.append(dict(metric=metric, window=name, auroc=area,
                           ci95=np.quantile(draws[:, j], [.025, .975]).tolist(),
                           delta_vs_mean=area-roc_auc_score(y, values[:, base]),
                           delta_ci95=np.quantile(draws[:, j]-draws[:, base], [.025, .975]).tolist(),
                           pr_auc=float(auc(recall, precision)),
                           ap=float(average_precision_score(y, values[:, j])),
                           whole_response_fraction=1. if name=="mean" else float((lengths<=name).mean())))
    args.output.mkdir(parents=True)
    report = dict(source=str(source.resolve()), source_sha256=digest.hexdigest(), n=len(y),
                  errors=int(y.sum()), windows=widths, bootstrap=args.bootstrap, seed=args.seed,
                  note="Exploratory fixed-window comparison, no held-out window selection. Error is positive; risk is negative confidence. Stride 1; shorter responses use their whole mean. CIs resample questions, unadjusted for multiple comparisons.",
                  records=result)
    (args.output/"results.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    lines = ["# Sliding-window comparison", "", f"n={len(y)}, errors={y.sum()}.",
             "Risk is the negative minimum window mean, stride 1. Windows longer than the response use its whole mean.",
             "Exploratory comparison. 95% paired bootstrap intervals, unadjusted for multiple comparisons.", "",
             "| Metric | Window | AUROC x100 [95% CI] | Delta vs mean [95% CI] | PR-AUC x100 | AP x100 | Whole-response fallback |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in result:
        lo, hi = np.array(r["ci95"])*100
        dl, dh = np.array(r["delta_ci95"])*100
        lines.append(f"| {r['metric']} | {r['window']} | {r['auroc']*100:.2f} [{lo:.2f}, {hi:.2f}] | "
                     f"{r['delta_vs_mean']*100:+.2f} [{dl:+.2f}, {dh:+.2f}] | {r['pr_auc']*100:.2f} | "
                     f"{r['ap']*100:.2f} | {r['whole_response_fraction']*100:.1f}% |")
    (args.output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines),flush=True)


if __name__ == "__main__":
    main()

