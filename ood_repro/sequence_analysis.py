"""E1 aggregation matrix, paired bootstrap and E2 order-destruction controls."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .aggregation import aggregate, permutation_indices, window_means
from .evaluation import read_jsonl
from .study_io import check_record, file_hash, independent_output, write_json
from .visualize import bootstrap_aucs, save_figure


def load_replay(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    rows = read_jsonl(path / "predictions.jsonl")
    if not manifest.get("complete") or [r["sample_id"] for r in rows] != manifest["dataset"]["sample_ids"]:
        raise ValueError("Replay is incomplete or sample order differs")
    for row in rows:
        check_record(row)
        tokens = row["sequence_tokens"]
        if len(tokens) != row["scored_tokens"] or not tokens:
            raise ValueError("Invalid token count")
        expected = [i for i, token in enumerate(row["generated_token_ids"])
                    if manifest["source_manifest"]["config"]["include_eos"]
                    or token not in manifest["source_manifest"]["eos_token_ids"]]
        if [t["position"] for t in tokens] != expected or any(
            t["token_id"] != row["generated_token_ids"][t["position"]] for t in tokens
        ):
            raise ValueError("Token alignment mismatch")
        keys = set(tokens[0]["scores"])
        if any(set(t["scores"]) != keys for t in tokens):
            raise ValueError("Token metrics differ within a trajectory")
        for key in keys:
            values = np.array([t["scores"][key] for t in tokens])
            if not np.isfinite(values).all() or not np.isclose(values.mean(), row["sequence_scores"][key], rtol=1e-5, atol=1e-6):
                raise ValueError("Token/aggregate score mismatch")
    return manifest, rows


def metrics_matrix(rows, widths, annotations=None):
    keys = set(rows[0]["sequence_tokens"][0]["scores"])
    if any(set(r["sequence_tokens"][0]["scores"]) != keys for r in rows):
        raise ValueError("Different token metric sets across rows")
    use_steps = annotations is not None
    columns = {}
    for row in rows:
        steps = annotations[row["sample_id"]]["steps"] if use_steps else None
        for metric in sorted(keys):
            x = [t["scores"][metric] for t in row["sequence_tokens"]]
            for name, value in aggregate(x, widths, steps).items():
                columns.setdefault(f"{metric}/{name}", []).append(value)
    return {key: np.array(value) for key, value in columns.items()}


def summarize(rows, scores, bootstrap=2000, seed=0):
    errors = np.array([not r["is_correct"] for r in rows], dtype=int)
    if np.unique(errors).size != 2:
        raise ValueError("Need both correct and incorrect answers")
    samples = bootstrap_aucs(errors, scores, bootstrap, seed)
    records = []
    lengths = np.array([r["scored_tokens"] for r in rows])
    for key, values in scores.items():
        metric, aggregation = key.split("/", 1)
        reference = metric + "/mean"
        auroc = float(roc_auc_score(errors, -values))
        delta = samples[key] - samples[reference]
        strata = {}
        for label, mask in (("1-64", lengths <= 64), ("65-128", (lengths > 64) & (lengths <= 128)),
                            ("129-256", (lengths > 128) & (lengths <= 256)), ("257+", lengths > 256)):
            strata[label] = {"n": int(mask.sum()), "errors": int(errors[mask].sum()),
                            "auroc": float(roc_auc_score(errors[mask], -values[mask])) if np.unique(errors[mask]).size == 2 else None}
        records.append({"metric": metric, "aggregation": aggregation, "n": len(rows), "error_rate": float(errors.mean()),
                        "auroc": auroc, "ci95": np.quantile(samples[key], [.025, .975]).tolist(),
                        "auprc": float(average_precision_score(errors, -values)),
                        "delta_vs_mean": auroc - float(roc_auc_score(errors, -scores[reference])),
                        "delta_ci95": np.quantile(delta, [.025, .975]).tolist(), "length_strata": strata})
    return records


def order_controls(rows, metric, width=64, repeats=1000, seed=0, bootstrap=2000):
    errors = np.array([not r["is_correct"] for r in rows], dtype=int)
    arrays = [np.array([t["scores"][metric] for t in row["sequence_tokens"]]) for row in rows]
    original_values = np.array([window_means(x, width).min() for x in arrays])
    original = float(roc_auc_score(errors, -original_values))
    rng = np.random.default_rng(seed)
    result = {}
    from tqdm import tqdm
    for mode in ("shuffle", "phase", "block", "reverse"):
        values = []
        null_scores = {}
        for _ in tqdm(range(1 if mode == "reverse" else repeats), desc=f"E2/{metric}/{mode}"):
            reordered = [x[permutation_indices(len(x), rng, mode)] for x in arrays]
            score = np.array([window_means(x, width).min() for x in reordered])
            if mode == "reverse" and not np.allclose(score, original_values, atol=1e-9, rtol=1e-10):
                raise ValueError("Reverse-window invariance failed")
            # Enforce exact mathematical invariants, avoiding rank changes from FP noise.
            short = np.array([len(x) <= width for x in arrays])
            score[short] = original_values[short]
            if mode == "reverse":
                score = original_values.copy()
            values.append(float(roc_auc_score(errors, -score)))
            if len(null_scores) < 16:
                null_scores[f"null_{len(null_scores)}"] = score
        values = np.array(values)
        left = (1 + int((values <= original).sum())) / (1 + len(values))
        right = (1 + int((values >= original).sum())) / (1 + len(values))
        result[mode] = {"original_auroc": original, "permuted_mean": float(values.mean()),
                        "delta": original - float(values.mean()), "distribution95": np.quantile(values, [.025, .975]).tolist(),
                        "diagnostic_p_two_sided": None if mode == "reverse" else min(1., 2 * min(left, right)),
                        "auroc_draws": values.tolist(), "repeats": len(values),
                        "warning": "Order-destruction diagnostic; exchangeability is not guaranteed; not causal proof"}
        draws = bootstrap_aucs(errors, {"original": original_values, **null_scores}, bootstrap, seed + 17)
        differences = draws["original"] - np.mean([draws[key] for key in null_scores], axis=0)
        result[mode]["paired_question_ci95_fixed_null_subset"] = np.quantile(differences, [.025, .975]).tolist()
        result[mode]["ci_null_draws"] = len(null_scores)
        result[mode]["ci_estimand"] = "Original AUROC minus mean of first <=16 FIXED order controls; conditional diagnostic, distinct from all-draw mean"
    return result


def figures(records, orders, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = sorted({r["metric"] for r in records})
    names = [r["aggregation"] for r in records if r["metric"] == methods[0]]
    matrix = np.array([[next(r["auroc"] * 100 for r in records if r["metric"] == m and r["aggregation"] == a)
                        for a in names] for m in methods])
    fig, ax = plt.subplots(figsize=(max(12, len(names) * 1.1), 1.8 + .6 * len(methods)))
    artist = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(methods)), methods)
    for i in range(len(methods)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    color="black" if artist.norm(matrix[i,j]) > .55 else "white", fontsize=8)
    ax.set_title("E1: AUROC x 100; all predeclared aggregation variants")
    fig.colorbar(artist, ax=ax)
    fig.tight_layout()
    save_figure(fig, output, "aggregation_matrix")
    if orders:
        fig, axes = plt.subplots(len(orders), 1, figsize=(10, 3.5 * len(orders)), squeeze=False)
        for ax, (metric, modes) in zip(axes[:, 0], orders.items()):
            for mode in ("shuffle", "phase", "block"):
                ax.hist(np.array(modes[mode]["auroc_draws"]) * 100, bins=25, alpha=.4, label=mode)
            ax.axvline(modes["shuffle"]["original_auroc"] * 100, color="black", label="original order")
            ax.set(xlabel="AUROC x 100", ylabel="Permutation draws", title=metric)
            ax.legend()
        fig.tight_layout()
        save_figure(fig, output, "order_controls")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--windows", default="32,64,128")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1000, help="0 skips E2")
    parser.add_argument("--permutation-metrics", default="fdbd_all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", help="Optional JSONL: sample_id and exhaustive [start,end) scored-position steps")
    args = parser.parse_args()
    source, output = independent_output(args.replay, args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new empty analysis output directory")
    widths = tuple(dict.fromkeys(int(x) for x in args.windows.split(",")))
    if args.bootstrap < 2 or args.permutations < 0 or not widths or min(widths) < 1 or 64 not in widths:
        parser.error("bootstrap >= 2, permutations >= 0 and positive windows including primary 64 required")
    manifest, rows = load_replay(source)
    print(f"E1: loaded {len(rows)} audited trajectories; aggregating all declared variants", flush=True)
    annotations = None
    if args.steps:
        items = read_jsonl(args.steps)
        annotations = {r["sample_id"]: r for r in items}
        if len(annotations) != len(items) or set(annotations) != {r["sample_id"] for r in rows}:
            raise ValueError("Step annotations must cover each replay sample exactly once")
    scores = metrics_matrix(rows, widths, annotations)
    print(f"E1: paired question bootstrap, {args.bootstrap} draws for {len(scores)} score/aggregation combinations", flush=True)
    records = summarize(rows, scores, args.bootstrap, args.seed)
    print("E1 complete; starting order controls" if args.permutations else "E1 complete; E2 skipped", flush=True)
    orders = {}
    if args.permutations:
        for metric in args.permutation_metrics.split(","):
            if metric + "/mean" not in scores:
                raise ValueError(f"Permutation metric not present: {metric}")
            orders[metric] = order_controls(rows, metric, 64, args.permutations, args.seed, args.bootstrap)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "analysis.json", {"config": vars(args), "source_sha256": file_hash(source / "predictions.jsonl"),
               "replay_manifest": manifest, "primary": "GSM8K fDBD-All lowest_w64 vs mean",
               "short_trace_counts": {str(w): sum(r["scored_tokens"] < w for r in rows) for w in widths},
               "records": records, "order_controls": orders,
               "notes": ["Existing test-set follow-up; no new causal experiment", "Unadjusted exploratory CIs; no best-window selection",
                         "E2 always uses the predeclared window 64; reverse is an invariance control"]})
    with (output / "aggregation.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", "aggregation", "n", "error_rate", "auroc_x100", "ci_low", "ci_high",
                                                    "auprc", "delta_x100", "delta_ci_low", "delta_ci_high"])
        writer.writeheader()
        for r in records:
            writer.writerow({"metric": r["metric"], "aggregation": r["aggregation"], "n": r["n"], "error_rate": r["error_rate"],
                             "auroc_x100": r["auroc"] * 100, "ci_low": r["ci95"][0] * 100, "ci_high": r["ci95"][1] * 100,
                             "auprc": r["auprc"], "delta_x100": r["delta_vs_mean"] * 100,
                             "delta_ci_low": r["delta_ci95"][0] * 100, "delta_ci_high": r["delta_ci95"][1] * 100})
    figures(records, orders, output)
    report = ["# E1/E2 sequence study", "", f"Questions: {len(rows)}; all scores are higher-confidence-oriented.",
              "", "|Score|Aggregation|AUROC x100 [95% CI]|Delta vs mean [95% CI]|", "|---|---|---:|---:|"]
    for r in records:
        report.append(f"|{r['metric']}|{r['aggregation']}|{100*r['auroc']:.2f} [{100*r['ci95'][0]:.2f}, {100*r['ci95'][1]:.2f}]|"
                      f"{100*r['delta_vs_mean']:+.2f} [{100*r['delta_ci95'][0]:+.2f}, {100*r['delta_ci95'][1]:+.2f}]|")
    report += ["", "E2 results: see analysis.json/order_controls and order_controls.png.",
               "No window selection on test AUROC. Local aggregation gains do not alone establish order dependence or error propagation.",
               "If a trace is shorter than the window, its single window equals its mean. Reverse-window equality is expected."]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"E1/E2 report and figures: {output}")


if __name__ == "__main__":
    main()
