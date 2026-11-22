"""Compare when confidence separates correct and wrong MATH responses."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


METRICS = ("fdbd_all", "nci", "shannon_top20")
STAGES = ("first", "first16", "first64", "Q1", "Q2", "Q3", "Q4", "whole")


def load(source):
    records, scores = [], []
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            tokens = row["sequence_tokens"]
            n = len(tokens)
            if n < 64 or n != row["scored_tokens"] or any(
                t["position"] != i or t["token_id"] != row["generated_token_ids"][i]
                for i, t in enumerate(tokens)
            ):
                raise ValueError(f"Unexpected token alignment: {row['sample_id']}")
            x = np.array([[t["scores"][key] for key in METRICS] for t in tokens])
            if not np.isfinite(x).all():
                raise ValueError(f"Nonfinite scores: {row['sample_id']}")
            quarters = np.array_split(x, 4)
            scores.append([x[0], x[:16].mean(0), x[:64].mean(0),
                           *[q.mean(0) for q in quarters], x.mean(0)])
            records.append(dict(
                id=row["sample_id"], correct=bool(row["is_correct"]),
                length=n, subject=row["subject"], level=row["level"],
                eligible=row["stop_reason"] == "eos" and row["parsed_answer"] is not None,
            ))
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate sample IDs")
    return records, np.array(scores)


def match(records, caliper):
    groups = defaultdict(lambda: ([], []))
    for i, row in enumerate(records):
        if row["eligible"]:
            groups[row["subject"], row["level"]][not row["correct"]].append(i)
    pairs = []
    for correct, wrong in groups.values():
        key = lambda i: (records[i]["length"], records[i]["id"])
        correct.sort(key=key)
        wrong.sort(key=key)
        i = j = 0
        while i < len(correct) and j < len(wrong):
            delta = records[correct[i]]["length"] - records[wrong[j]]["length"]
            if delta < -caliper:
                i += 1
            elif delta > caliper:
                j += 1
            else:
                pairs.append((correct[i], wrong[j]))
                i += 1
                j += 1
    return np.array(pairs, dtype=int)


def summary(values):
    rng = np.random.default_rng(1729)
    idx = rng.integers(len(values), size=(2000, len(values)))
    samples = values[idx].mean(axis=1)
    return {"estimate": float(values.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records, scores = load(args.source)
    errors = np.array([not r["correct"] for r in records])
    report = {"source": str(args.source), "n": len(records), "errors": int(errors.sum()),
              "eligible": sum(r["eligible"] for r in records), "draws": 2000,
              "match": "Within subject and level; sorted by length and ID, one-to-one, no score-based selection. EOS plus parseable answer.",
              "full_auroc": {m: {s: roc_auc_score(errors, -scores[:, j, k])
                                  for j, s in enumerate(STAGES)} for k, m in enumerate(METRICS)}}
    for caliper in (10, 20, 40):
        pairs = match(records, caliper)
        differences = scores[pairs[:, 0]] - scores[pairs[:, 1]]
        lengths = np.array([records[i]["length"] - records[j]["length"] for i, j in pairs])
        result = {"n_pairs": len(pairs), "median_abs_length_difference": float(np.median(abs(lengths))),
                  "mean_signed_length_difference": float(lengths.mean()), "scores": {}}
        print("caliper", caliper, "pairs", len(pairs), "median_abs_length", np.median(abs(lengths)))
        for k, metric in enumerate(METRICS):
            z = differences[:, :, k]
            item = {stage: {"win": summary((z[:, j] > 0) + .5 * (z[:, j] == 0)),
                            "gap": summary(z[:, j])}
                    for j, stage in enumerate(STAGES)}
            item["Q4_minus_Q1_gap"] = summary(z[:, 6] - z[:, 3])
            quarter_wins = (z[:, 3:7] > 0).sum(axis=1)
            item["quarters_positive_counts"] = np.bincount(quarter_wins, minlength=5).tolist()
            result["scores"][metric] = item
            if caliper == 20:
                for stage in STAGES:
                    d = item[stage]
                    print(metric,stage,"AUROC",round(100*report["full_auroc"][metric][stage],2),
                          "win",np.round(100*np.array([d["win"]["estimate"],*d["win"]["ci95"]]),2),
                          "gap x1000",np.round(1000*np.array([d["gap"]["estimate"],*d["gap"]["ci95"]]),3))
                print("quarters positive counts", item["quarters_positive_counts"],
                      "Q4-Q1",item["Q4_minus_Q1_gap"])
        if caliper == 20:
            result["pair_ids"] = [[records[i]["id"], records[j]["id"]] for i,j in pairs]
        report[str(caliper)] = result
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

