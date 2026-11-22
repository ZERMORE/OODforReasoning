"""Evaluate prompt-only first-token confidence against final-answer correctness."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.bootstrap < 2 or args.output.exists():
        parser.error("At least two bootstrap draws and a new output directory required")
    source = args.source / "predictions.jsonl" if args.source.is_dir() else args.source
    metrics = ("fdbd_all", "nci")
    values, errors, ids = [], [], []
    digest = hashlib.sha256()
    for line in source.open("rb"):
        digest.update(line)
        if not line.strip():
            continue
        row = json.loads(line)
        tokens = row.get("sequence_tokens") or row.get("token_scores")
        if not tokens:
            raise ValueError(f"Per-token scores missing for {row['sample_id']}; replay required")
        if len(tokens) != row["scored_tokens"] or tokens[0]["position"] != 0:
            raise ValueError("Unexpected scoring alignment")
        if tokens[0]["token_id"] != row["generated_token_ids"][0]:
            raise ValueError("First token mismatch")
        scores = np.array([[t["scores"][m] for m in metrics] for t in tokens])
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite scores")
        values.append(-np.array([scores[0], scores[:16].mean(0), scores.mean(0)]).T.reshape(-1))
        errors.append(not row["is_correct"])
        ids.append(row["sample_id"])
    if len(set(ids)) != len(ids) or len(set(errors)) != 2:
        raise ValueError("Unique IDs and both classes required")
    y, x = np.array(errors), np.array(values)
    rng = np.random.default_rng(args.seed)
    draws = []
    for _ in range(args.bootstrap):
        index = rng.integers(len(y), size=len(y))
        n = y[index].sum()
        if n == 0 or n == len(y):
            raise ValueError("Single-class bootstrap draw")
        draws.append((rankdata(x[index], axis=0)[y[index]].sum(0) - n*(n+1)/2) / (n*(len(y)-n)))
    draws = np.array(draws)
    results = []
    for j, (metric, stage) in enumerate((m,s) for m in metrics for s in ("first","first16","whole")):
        lo, hi = np.quantile(draws[:,j], [.025,.975])
        area = roc_auc_score(y,x[:,j])
        results.append(dict(metric=metric,stage=stage,auroc=area,ci95=[lo,hi]))
        print(metric,stage,round(area*100,2),np.round(100*np.array([lo,hi]),2),flush=True)
    args.output.mkdir(parents=True)
    report = dict(source=str(source),source_sha256=digest.hexdigest(),n=len(y),errors=int(y.sum()),
                  bootstrap=args.bootstrap,seed=args.seed,records=results,
                  note="First scored token position 0, before generated response tokens enter the context. Lower confidence indicates final error. Question bootstrap, exploratory unadjusted CIs.")
    (args.output/"results.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("n/errors",len(y),int(y.sum()),flush=True)


if __name__ == "__main__":
    main()

