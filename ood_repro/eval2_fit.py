"""Experiment 2: supervised aggregation with frozen question splits and set controls."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .data import digest
from .eval2_data import load_study, pooling, sequences
from .study_io import file_hash, independent_output, write_json
from .visualize import bootstrap_aucs, save_figure


def statistics(arrays):
    return np.array([[float(x.mean()), float(x.std()), *np.quantile(x[:, 0], [0, .1, .25, .5, .75, .9, 1]).tolist(),
                      float(np.sort(x[:, 0])[:max(1, int(np.ceil(.1 * len(x))))].mean()), float(np.log1p(len(x)))] for x in arrays])


def normalization(arrays):
    tokens = np.concatenate(arrays)
    lengths = np.log1p([len(x) for x in arrays])
    return {"mean": tokens.mean(0).tolist(), "std": np.maximum(tokens.std(0), 1e-6).tolist(),
            "length_mean": float(lengths.mean()), "length_std": float(max(lengths.std(), 1e-6))}


def batch(arrays, indices, norm, device, rng=None, reverse=False):
    import torch
    items = []
    for i in indices:
        x = (arrays[i] - np.array(norm["mean"])) / np.array(norm["std"])
        if rng is not None:
            x = x[rng.permutation(len(x))]
        if reverse:
            x = x[::-1]
        items.append(torch.tensor(x.copy(), dtype=torch.float32))
    lengths = torch.tensor([len(x) for x in items], device=device)
    padded = torch.nn.utils.rnn.pad_sequence(items, batch_first=True).to(device)
    feature = (torch.log1p(lengths.float()) - norm["length_mean"]) / norm["length_std"]
    return padded, lengths, feature


def predict(model, arrays, norm, device, batch_size=32, mode="original", seed=0):
    import torch
    model.eval()
    results = []
    rng = np.random.default_rng(seed) if mode == "shuffle" else None
    with torch.inference_mode():
        for start in range(0, len(arrays), batch_size):
            inputs = batch(arrays, range(start, min(start + batch_size, len(arrays))), norm, device, rng, mode == "reverse")
            results.extend((-model(*inputs)).cpu().tolist())  # Higher is confidence, not error risk.
    return np.array(results)


def train_sequence(kind, train, validation, labels, val_labels, seed, args):
    import torch
    from .eval2_models import SequenceClassifier
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    norm = normalization(train)
    model = SequenceClassifier("gru" if kind == "gru_shuffled" else kind, hidden=args.hidden).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(labels) - labels.sum()) / labels.sum(), device=args.device))
    rng = np.random.default_rng(seed)
    order_rng = np.random.default_rng(seed + 10000)
    best, best_state, best_epoch = -1., None, None
    for epoch in range(args.epochs):
        model.train()
        indices = rng.permutation(len(train))
        for start in range(0, len(train), args.batch_size):
            selected = indices[start:start + args.batch_size]
            inputs = batch(train, selected, norm, args.device, order_rng if kind == "gru_shuffled" else None)
            targets = torch.tensor(labels[selected], dtype=torch.float32, device=args.device)
            optimizer.zero_grad()
            loss = loss_fn(model(*inputs), targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        mode = "shuffle" if kind == "gru_shuffled" else "original"
        confidence = predict(model, validation, norm, args.device, args.batch_size, mode, seed + 20000)
        value = float(roc_auc_score(val_labels, -confidence))
        print(f"  {kind} seed={seed} epoch={epoch+1}/{args.epochs} validation_AUROC={value:.4f}", flush=True)
        if value > best:
            best, best_epoch = value, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, norm, {"best_validation_auroc": best, "best_epoch": best_epoch,
                         "parameters": sum(p.numel() for p in model.parameters()), "state_dict": best_state}


def summarize(test, models, bootstrap=2000, seed=0):
    errors = np.array([not r["is_correct"] for r in test], dtype=int)
    flattened = {f"{name}::{rep}": values for name, replicas in models.items() for rep, values in replicas.items()}
    samples = bootstrap_aucs(errors, flattened, bootstrap, seed)
    records = []
    for name, replicas in models.items():
        metric, method = name.split("/", 1)
        draws = np.mean([samples[f"{name}::{rep}"] for rep in replicas], axis=0)
        points = [float(roc_auc_score(errors, -values)) for values in replicas.values()]
        record = {"metric": metric, "method": method, "auroc": float(np.mean(points)),
                  "ci95": np.quantile(draws, [.025, .975]).tolist(), "replica_aurocs": dict(zip(replicas, points)),
                  "seed_std": float(np.std(points)), "paired_deltas": {}}
        for reference in ("mean", "logreg", "deepsets", "attention_nopos", "gru"):
            key = metric + "/" + reference
            if key not in models:
                continue
            base = np.mean([samples[f"{key}::{rep}"] for rep in models[key]], axis=0)
            base_point = np.mean([roc_auc_score(errors, -v) for v in models[key].values()])
            record["paired_deltas"][reference] = {"delta": record["auroc"] - float(base_point),
                "ci95": np.quantile(draws - base, [.025, .975]).tolist()}
        records.append(record)
    return records


def render(records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = list(dict.fromkeys(r["metric"] for r in records))
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4.5 * len(metrics)), squeeze=False)
    for ax, metric in zip(axes[:, 0], metrics):
        selected = [r for r in records if r["metric"] == metric]
        positions = np.arange(len(selected))
        points = np.array([r["auroc"] for r in selected]) * 100
        ci = np.array([r["ci95"] for r in selected]) * 100
        ax.errorbar(positions, points, yerr=np.array([points-ci[:, 0], ci[:, 1]-points]).clip(0), fmt="o", capsize=3)
        ax.set_xticks(positions, [r["method"] for r in selected], rotation=40, ha="right")
        ax.set(title=metric, ylabel="AUROC x 100 [question-bootstrap 95% CI]")
        ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    save_figure(fig, output, "experiment2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", default="fdbd_all,nci")
    parser.add_argument("--models", default="logreg,deepsets,gru,cnn,attention,attention_nopos,gru_shuffled")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--min-train-errors", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import torch
    import matplotlib  # Fail before fitting models if plotting dependency is absent.
    import sklearn
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if min(args.epochs, args.batch_size, args.threads, args.min_train_errors) < 1 or args.bootstrap < 2 or args.lr <= 0:
        parser.error("Positive training parameters and bootstrap >=2 required")
    if args.hidden < 4 or args.hidden % 2:
        parser.error("hidden must be even and >=4")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu or check the original torch environment")
    seeds = [int(x) for x in args.seeds.split(",")]
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        parser.error("Distinct nonnegative seeds required")
    allowed = {"logreg", "deepsets", "gru", "cnn", "attention", "attention_nopos", "gru_shuffled"}
    kinds = args.models.split(",")
    if set(kinds) - allowed or len(kinds) != len(set(kinds)):
        parser.error("Unknown or duplicated models")
    _, output = independent_output(args.test, args.output)
    independent_output(args.pool, output)
    partitions, test, split = load_study(args.pool, args.test, args.split)
    counts = {key: {"n": len(rows), "errors": sum(not r["is_correct"] for r in rows)} for key, rows in partitions.items()}
    if counts["train"]["errors"] < args.min_train_errors:
        raise ValueError(f"Too few train errors: {counts}; enlarge train pool, do not search seeds for a favorable split")
    for key, count in counts.items():
        if min(count["errors"], count["n"] - count["errors"]) < 2:
            raise ValueError(f"Need >=2 of each class per partition: {counts}")
    config = {k: v for k, v in vars(args).items() if k not in {"resume", "output"}}
    code = {name: file_hash(Path(__file__).parent / name) for name in (
        "eval2_fit.py", "eval2_models.py", "eval2_data.py", "aggregation.py", "visualize.py")}
    environment = {"torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
                   "cuda": torch.version.cuda, "device": args.device,
                   "gpu": torch.cuda.get_device_name(args.device) if str(args.device).startswith("cuda") else None}
    signature = digest({"config": config, "split": split, "code": code, "environment": environment})
    if output.exists() and any(output.iterdir()):
        if not args.resume or json.loads((output / "manifest.json").read_text())["signature"] != signature:
            raise ValueError("Use new output or matching --resume")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", {"signature": signature, "config": config, "split": split, "counts": counts,
        "code": code, "environment": environment, "complete": False})
    models, jobs, thresholds = {}, {}, {}
    errors = {name: np.array([not r["is_correct"] for r in rows], dtype=int) for name, rows in partitions.items()}
    for metric in args.metrics.split(","):
        arrays = {name: sequences(rows, metric) for name, rows in {**partitions, "test": test}.items()}
        for aggregation in ("mean", "min", "quantile10", "token_bottom10", "head_weighted", "tail_weighted", "lowest_w64"):
            name = metric + "/" + aggregation
            models[name] = {"fixed": np.array([pooling(x[:, 0])[aggregation] for x in arrays["test"]])}
        models[metric + "/length_only"] = {"fixed": -np.array([len(x) for x in arrays["test"]], dtype=float)}
        for kind in kinds:
            replicas = seeds[:1] if kind == "logreg" else seeds
            for seed in replicas:
                name, rep = metric + "/" + kind, str(seed)
                job = output / f"{metric}-{kind}-seed{seed}.json"
                print(f"Experiment2: {name}, seed={seed}", flush=True)
                if job.exists():
                    saved = json.loads(job.read_text())
                    if saved.get("job_hash") != digest({k: v for k, v in saved.items() if k != "job_hash"}):
                        raise ValueError("Cached job integrity mismatch")
                    confidence = np.array(saved["test_confidence"])
                    if len(confidence) != len(test):
                        raise ValueError("Cached predictions count mismatch")
                    if kind != "logreg":
                        checkpoint = output / f"{metric}-{kind}-seed{seed}.pt"
                        if not checkpoint.is_file() or file_hash(checkpoint) != saved["checkpoint_sha256"]:
                            raise ValueError("Cached neural checkpoint missing or changed")
                else:
                    if kind == "logreg":
                        scaler = StandardScaler().fit(statistics(arrays["train"]))
                        classifier = LogisticRegression(C=1., class_weight="balanced", max_iter=1000, random_state=seed)
                        classifier.fit(scaler.transform(statistics(arrays["train"])), errors["train"])
                        confidence = -classifier.decision_function(scaler.transform(statistics(arrays["test"])))
                        calibration = -classifier.decision_function(scaler.transform(statistics(arrays["calibration"])))
                        saved = {"parameters": int(classifier.coef_.size + 1), "coefficients": classifier.coef_.tolist(),
                                 "intercept": classifier.intercept_.tolist(), "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()}
                    else:
                        model, norm, saved = train_sequence(kind, arrays["train"], arrays["validation"], errors["train"], errors["validation"], seed, args)
                        torch.save({"state_dict": saved.pop("state_dict"), "normalization": norm, "kind": kind,
                                    "hidden": args.hidden}, output / f"{metric}-{kind}-seed{seed}.pt")
                        saved["checkpoint_sha256"] = file_hash(output / f"{metric}-{kind}-seed{seed}.pt")
                        mode = "shuffle" if kind == "gru_shuffled" else "original"
                        confidence = predict(model, arrays["test"], norm, args.device, args.batch_size, mode, seed + 20000)
                        calibration = predict(model, arrays["calibration"], norm, args.device, args.batch_size, mode, seed + 20000)
                        if kind in {"gru", "cnn", "attention"}:
                            saved["test_controls"] = {control: predict(model, arrays["test"], norm, args.device, args.batch_size,
                                                                      control, seed + 20000).tolist() for control in ("shuffle", "reverse")}
                    threshold = float(np.quantile(calibration[errors["calibration"] == 0], .05))
                    saved.update(test_confidence=confidence.tolist(), threshold=threshold,
                                 calibration_actual_fpr=float((calibration[errors["calibration"] == 0] < threshold).mean()))
                    saved["job_hash"] = digest(saved)
                    write_json(job, saved)
                models.setdefault(name, {})[rep] = confidence
                thresholds[f"{name}::{rep}"] = {"threshold": saved["threshold"], "calibration_actual_fpr": saved["calibration_actual_fpr"],
                    "test_actual_fpr": float((confidence[np.array([r["is_correct"] for r in test])] < saved["threshold"]).mean()),
                    "test_tpr": float((confidence[np.array([not r["is_correct"] for r in test])] < saved["threshold"]).mean())}
                jobs[f"{name}::{rep}"] = {k: v for k, v in saved.items() if k not in {"test_confidence", "test_controls"}}
                for control, values in saved.get("test_controls", {}).items():
                    models.setdefault(name + "_test_" + control, {})[rep] = np.array(values)
    print("Experiment2: paired question bootstrap and report", flush=True)
    records = summarize(test, models, args.bootstrap)
    write_json(output / "experiment2.json", {"records": records, "jobs": jobs, "thresholds": thresholds,
        "note": "Neural checkpoints chosen by validation only. CIs resample test questions; training-seed std reported separately. Exploratory, unadjusted CIs."})
    with (output / "test_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for i, row in enumerate(test):
            stream.write(json.dumps({"sample_id": row["sample_id"], "is_correct": row["is_correct"],
                "confidence": {name: {rep: float(values[i]) for rep, values in replicas.items()} for name, replicas in models.items()}}, allow_nan=False) + "\n")
    lines = ["# Eval section 2 — experiment 2", "", "|Metric|Method|AUROC x100 [95% CI]|Delta vs mean [95% CI]|Seed std (points)|",
             "|---|---|---:|---:|---:|"]
    for r in records:
        d = r["paired_deltas"]["mean"]
        lines.append(f"|{r['metric']}|{r['method']}|{100*r['auroc']:.2f} [{100*r['ci95'][0]:.2f}, {100*r['ci95'][1]:.2f}]|"
                     f"{100*d['delta']:+.2f} [{100*d['ci95'][0]:+.2f}, {100*d['ci95'][1]:+.2f}]|{100*r['seed_std']:.2f}|")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    render(records, output)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["complete"] = True
    write_json(output / "manifest.json", manifest)
    print(f"Experiment2 complete: {output}")


if __name__ == "__main__":
    main()
