"""Experiment 1: frozen-threshold error-position TPR and length-matched AUROC."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .error_positions import validate_annotations
from .eval2_data import load_study, pooling
from .evaluation import read_jsonl
from .study_io import file_hash, independent_output, write_json
from .visualize import weighted_auc, save_figure


def validate_reviews(items, rows):
    """Explicitly allow reviewed-but-undecidable cases, never silently invent a label."""
    if len({a["sample_id"] for a in items}) != len(items):
        raise ValueError("Duplicate annotation IDs")
    known = [a for a in items if a.get("process_valid") is not None]
    result = validate_annotations(known, rows)
    lookup = {r["sample_id"]: r for r in rows}
    for annotation in items:
        if annotation.get("process_valid") is not None:
            continue
        row = lookup.get(annotation["sample_id"])
        if row is None or annotation.get("source_row_hash") != row["source_row_hash"] or annotation.get("reviewed") is not True:
            raise ValueError("Unknown-process annotation must be reviewed and match the source")
        if annotation.get("first_error_position") is not None or not annotation.get("notes", "").strip():
            raise ValueError("Undecidable cases need no invented location and an explicit notes explanation")
        result[annotation["sample_id"]] = annotation
    return result


def match_controls(rows, error_indices, correct_indices, seed=0):
    """One-to-one nearest log-length, without replacement WITHIN each bucket."""
    rng = np.random.default_rng(seed)
    available = list(rng.permutation(correct_indices))
    lengths = np.log1p([r["scored_tokens"] for r in rows])
    selected, distances = [], []
    for index in sorted(error_indices, key=lambda i: (lengths[i], i)):
        if not available:
            raise ValueError("Insufficient correct controls for 1:1 length matching")
        choice = min(available, key=lambda i: abs(lengths[i] - lengths[index]))
        selected.append(choice)
        distances.append(float(abs(lengths[choice] - lengths[index])))
        available.remove(choice)
    return selected, distances


def analyze(rows, calibration, annotations, metrics, aggregations, bootstrap=2000, seed=0):
    annotations = validate_reviews(annotations, rows)
    final_errors = [r for r in rows if not r["is_correct"]]
    missing = [r["sample_id"] for r in final_errors if r["sample_id"] not in annotations]
    if missing:
        raise ValueError(f"Review ALL final-error cases first; {len(missing)} missing, e.g. {missing[:3]}")
    n = len(rows)
    correct = np.array([r["is_correct"] for r in rows], dtype=bool)
    buckets = np.full(n, -2, dtype=int)
    for i, row in enumerate(rows):
        annotation = annotations.get(row["sample_id"])
        if not row["is_correct"] and annotation and annotation["first_error_position"] is not None:
            buckets[i] = min(3, int(4 * annotation["first_error_position"] / len(row["generated_token_ids"])))
    rng = np.random.default_rng(seed)
    counts = np.array([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(bootstrap)], dtype=np.int16)
    matches = {}
    for bucket in range(4):
        indices = np.flatnonzero(buckets == bucket).tolist()
        controls, distances = match_controls(rows, indices, np.flatnonzero(correct).tolist(), seed + bucket)
        matches[bucket] = {"controls": controls, "log_length_distances": distances}
    records = []
    for metric in metrics:
        print(f"Experiment1: {metric} position bootstrap", flush=True)
        test_values = [pooling([t["scores"][metric] for t in r["sequence_tokens"]]) for r in rows]
        calibration_values = [pooling([t["scores"][metric] for t in r["sequence_tokens"]]) for r in calibration]
        for aggregation in aggregations:
            score = np.array([v[aggregation] for v in test_values])
            cal_correct = np.array([v[aggregation] for r, v in zip(calibration, calibration_values) if r["is_correct"]])
            if not len(cal_correct):
                raise ValueError("Calibration has no correct answers")
            threshold = float(np.quantile(cal_correct, .05))
            item = {"metric": metric, "aggregation": aggregation, "threshold": threshold,
                    "calibration_correct_n": len(cal_correct), "calibration_fpr": float((cal_correct < threshold).mean()),
                    "test_fpr": float((score[correct] < threshold).mean()), "buckets": []}
            tpr_draws, auc_draws = {}, {}
            for bucket in range(4):
                wrong = buckets == bucket
                matching = np.zeros(n, dtype=bool)
                matching[matches[bucket]["controls"]] = True
                entry = {"bucket": bucket, "n_errors": int(wrong.sum()), "descriptive_only": int(wrong.sum()) < 20,
                         "n_shared_controls": int(correct.sum()), "n_matched_controls": int(matching.sum()),
                         "matched_correct_ids": [rows[i]["sample_id"] for i in matches[bucket]["controls"]],
                         "matching_log_length_distance_median": float(np.median(matches[bucket]["log_length_distances"])) if wrong.any() else None}
                if not wrong.any():
                    item["buckets"].append(entry)
                    continue
                denom = counts[:, wrong].sum(1)
                tprs = np.divide((counts[:, wrong] * (score[wrong] < threshold)).sum(1), denom,
                                 out=np.full(bootstrap, np.nan), where=denom > 0)
                tpr_draws[bucket] = tprs
                entry.update(tpr=float((score[wrong] < threshold).mean()),
                             tpr_ci95=np.nanquantile(tprs, [.025, .975]).tolist())
                for name, control in (("shared", correct), ("length_matched", matching)):
                    mask = wrong | control
                    labels, risk = wrong[mask].astype(int), -score[mask]
                    values = np.array([weighted_auc(labels, risk, draw[mask]) for draw in counts])
                    auc_draws[(bucket, name)] = values
                    entry[name + "_auroc"] = float(roc_auc_score(labels, risk))
                    entry[name + "_ci95"] = np.nanquantile(values, [.025, .975]).tolist()
                item["buckets"].append(entry)
            if 0 in tpr_draws and 3 in tpr_draws:
                early, late = item["buckets"][0], item["buckets"][3]
                item["early_minus_late_tpr"] = {"delta": early["tpr"] - late["tpr"],
                    "ci95": np.nanquantile(tpr_draws[0] - tpr_draws[3], [.025, .975]).tolist()}
                item["early_minus_late_matched_auroc"] = {"delta": early["length_matched_auroc"] - late["length_matched_auroc"],
                    "ci95": np.nanquantile(auc_draws[(0, "length_matched")] - auc_draws[(3, "length_matched")], [.025, .975]).tolist()}
            records.append(item)
    return {"n": n, "final_errors": len(final_errors), "located_final_errors": int((buckets >= 0).sum()),
            "unlocated_final_errors": int(((buckets == -2) & ~correct).sum()),
            "correct_final_with_process_error": sum(r["is_correct"] and r["sample_id"] in annotations
                                                     and annotations[r["sample_id"]]["process_valid"] is False for r in rows),
            "reviewed_undecidable_processes": sum(a.get("process_valid") is None for a in annotations.values()),
            "records": records, "notes": ["Position is observational, not a causal treatment.",
                "Shared and matched controls may be reused BETWEEN buckets; the same question-bootstrap counts are reused across buckets.",
                "Length matching is fixed before bootstrap; CIs conditional on this matching, not its selection uncertainty.",
                "Process-valid final errors (e.g. extraction failures) and unknown locations are not forced into a position bucket.",
                "Exploratory unadjusted 95% CIs; unknown locations may induce selection bias."]}


def render(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    primary = [r for r in result["records"] if r["aggregation"] == "mean"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for record in primary:
        for ax, key, ci_key in ((axes[0], "tpr", "tpr_ci95"), (axes[1], "length_matched_auroc", "length_matched_ci95")):
            valid = [b for b in record["buckets"] if key in b]
            if not valid:
                continue
            values = np.array([b[key] for b in valid]) * 100
            ci = np.array([b[ci_key] for b in valid]) * 100
            ax.errorbar([b["bucket"] for b in valid], values, yerr=np.array([values-ci[:, 0], ci[:, 1]-values]).clip(0), marker="o", capsize=2, label=record["metric"])
    for ax, title in zip(axes, ("Frozen calibration threshold: error TPR (%)", "Length-matched AUROC x 100")):
        ax.set_xticks(range(4), ["0-25%", "25-50%", "50-75%", "75-100%"])
        ax.set(title=title, xlabel="First-error relative position")
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.tight_layout()
    save_figure(fig, output, "experiment1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("pool", "test", "split", "annotations", "output"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--metrics", default="fdbd_all,nci")
    parser.add_argument("--aggregations", default="mean,min,quantile10,token_bottom10,head_weighted,tail_weighted,lowest_w64")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error("bootstrap >=2 required")
    _, output = independent_output(args.test, args.output)
    independent_output(args.pool, output)
    if output.exists() and any(output.iterdir()):
        parser.error("New empty output required")
    partitions, test, split = load_study(args.pool, args.test, args.split)
    annotations = read_jsonl(args.annotations)
    judge_path = Path(args.annotations).parent / "judge-manifest.json"
    judge = None
    if judge_path.exists():
        judge = json.loads(judge_path.read_text(encoding="utf-8"))
        if not judge.get("complete") or judge.get("annotations_sha256") != file_hash(args.annotations):
            raise ValueError("API annotations are incomplete or their hash differs from the judge manifest")
        if judge["settings"]["source_sha256"] != file_hash(Path(args.test) / "predictions.jsonl"):
            raise ValueError("API judge source differs from experiment1 test replay")
    auto_n = sum(a.get("annotation_kind") == "llm_auto" for a in annotations)
    if auto_n and judge is None:
        raise ValueError("Automatic annotations require their original sibling judge-manifest.json")
    result = analyze(test, partitions["calibration"], annotations, args.metrics.split(","), args.aggregations.split(","), args.bootstrap)
    result["annotation_provenance"] = {"automatic_n": auto_n, "other_n": len(annotations)-auto_n, "judge": judge}
    if auto_n:
        result["notes"].append("LLM automatic process judgments, not human ground truth. CIs exclude judge errors; quote start is an operational location, not a causal origin.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "experiment1.json", {**result, "config": vars(args), "split": split,
                                           "annotations_sha256": file_hash(args.annotations)})
    fields = ["metric", "aggregation", "bucket", "n_errors", "tpr", "tpr_ci95", "shared_auroc", "shared_ci95", "length_matched_auroc", "length_matched_ci95"]
    with (output / "positions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in result["records"]:
            for bucket in record["buckets"]:
                writer.writerow({key: {**record, **bucket}.get(key) for key in fields})
    render(result, output)
    lines = ["# Eval section 2 — experiment 1", "",
             f"Final errors: {result['final_errors']}; located: {result['located_final_errors']}; unlocated: {result['unlocated_final_errors']}.", "",
             "|Metric|Aggregation|First-error bucket|Errors|TPR % [95% CI]|Length-matched AUROC x100 [95% CI]|",
             "|---|---|---:|---:|---:|---:|"]
    if auto_n:
        lines[2:2] = [f"Automatic LLM annotations: {auto_n}; NOT human verified. Confidence intervals do not include judge uncertainty.", ""]
    for record in result["records"]:
        for b in record["buckets"]:
            tpr = "—" if "tpr" not in b else f"{100*b['tpr']:.2f} [{100*b['tpr_ci95'][0]:.2f}, {100*b['tpr_ci95'][1]:.2f}]"
            auc = "—" if "length_matched_auroc" not in b else f"{100*b['length_matched_auroc']:.2f} [{100*b['length_matched_ci95'][0]:.2f}, {100*b['length_matched_ci95'][1]:.2f}]"
            lines.append(f"|{record['metric']}|{record['aggregation']}|{b['bucket']}|{b['n_errors']}|{tpr}|{auc}|")
    lines += ["", "Buckets 0/1/2/3 correspond to 0-25/25-50/50-75/75-100%.",
              "Buckets with <20 errors are descriptive. Matching is fixed, CIs are unadjusted; location associations are not causal effects.",
              "Threshold, actual FPR, shared-control AUROC and paired early-minus-late intervals are in experiment1.json."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Experiment1 complete: {output}; location counts and unknown cases are in experiment1.json")


if __name__ == "__main__":
    main()
