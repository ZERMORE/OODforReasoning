"""Frozen transfer, window comparisons and ProcessBench evaluation."""
import csv
import re
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ood_repro.evidence import evidence_drop
from ood_repro.study_io import file_hash, independent_output
from .extension import GEOMETRY, PROBABILITY, SIGNALS, WIDTHS, WINDOW_SIGNALS
from .run import read, verified_rows, write


RADIAL = "fisher_radial"
FUSED = "probability_plus_angle_fisher"
PROB = "probability_model"


def parse_option(text, n_options):
    matches = re.findall(r"(?im)^[ \t]*\**[ \t]*Final answer[ \t]*:[ \t]*\**[ \t]*\(?([A-J])\)?[.* \t]*$", text)
    return matches[-1].upper() if matches and matches[-1].upper() in "ABCDEFGHIJ"[:n_options] else None


def completed(root):
    manifest = read(root/"manifest.json")
    if not manifest["complete"]:
        raise ValueError(f"Incomplete scoring run: {root}")
    return manifest


def width_for(width, length):
    return min(length, max(1, round(.1*length)) if width == "10pct" else int(width))


def window(values, width, budget=None):
    width = width_for(width, len(values))
    totals = np.r_[0., np.cumsum(values, dtype=float)]
    means = (totals[width:] - totals[:-width])/width
    if budget is None:
        start = int(np.argmax(means))
    else:
        indices = np.linspace(0, len(means)-1, min(len(means), budget), dtype=int)
        start = int(indices[np.argmax(means[indices])])
    return float(means[start]), start, width


def arrays(trace):
    return {k: np.asarray([t["scores"][k] for t in trace], dtype=float) for k in SIGNALS}


def summarize(values, signs, budget=None):
    features = {}
    for key, raw in values.items():
        confidence = signs[key]*raw
        features[f"{key}/mean"] = -float(confidence.mean())
        if key in WINDOW_SIGNALS:
            for width in WIDTHS:
                features[f"{key}/w{width}"] = window(-confidence, width, budget)[0]
        if key == "shannon_top20" and budget is None:
            features[f"{key}/evidence_drop"] = evidence_drop(confidence)[0]
    features["length_only"] = float(len(values[RADIAL]))
    return features


def fit_pair(rows, labels):
    if len(set(labels)) != 2:
        raise ValueError("Fusion fitting requires both classes")
    keys = [f"{k}/mean" for k in PROBABILITY]
    result = {}
    for name, features in ((PROB, keys), (FUSED, keys+[f"{k}/mean" for k in GEOMETRY])):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=500))
        model.fit([[row[k] for k in features] for row in rows], labels)
        scaler, classifier = [step[1] for step in model.steps]
        result[name] = dict(keys=features, center=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                            weights=classifier.coef_[0].tolist(), intercept=float(classifier.intercept_[0]))
    return result


def fuse(features, models):
    for name, model in models.items():
        x = np.asarray([features[k] for k in model["keys"]])
        features[name] = float(((x-model["center"])/model["scale"]) @ model["weights"] + model["intercept"])


def freeze(args):
    settings, fits, targets, calibration = {}, [], [], defaultdict(list)
    for task in ("gsm8k", "math"):
        source = args.source_g1/task
        manifest = completed(source)
        settings[task] = manifest["signature"]
        for row in verified_rows(source, manifest):
            if row["split"] not in ("fit", "calibration") or "excluded" in row:
                continue
            label = int(row["error"])
            values = arrays(row["token_scores"])
            if row["split"] == "fit":
                fits.append({k: float(v.mean()) for k, v in values.items()})
                targets.append(label)
            else:
                calibration[task].append((label, values[RADIAL]))
    signs = {k: 1 for k in SIGNALS}
    for key in (*GEOMETRY, RADIAL):
        signs[key] = 1 if roc_auc_score(targets, [-r[key] for r in fits]) >= .5 else -1
    fusion = fit_pair([{f"{k}/mean": -signs[k]*v for k, v in row.items()} for row in fits], targets)
    by_width = {}
    for size in (*WIDTHS, "mean"):
        ap = []
        for task in settings:
            y, risks = [], []
            for label, raw in calibration[task]:
                risk = -signs[RADIAL]*raw
                y.append(label)
                risks.append(float(risk.mean()) if size == "mean" else window(risk, size)[0])
            if len(set(y)) != 2:
                raise ValueError(f"Calibration requires both classes: {task}")
            ap.append(average_precision_score(y, risks))
        by_width[str(size)] = float(np.mean(ap))
    policy = dict(signs=signs, fusion=fusion, radial_width=max(by_width, key=by_width.get),
                  comparison="Macro calibration AP, GSM8K and MATH", source_signatures=settings,
                  fit_count=len(fits), calibration_counts={k: len(v) for k,v in calibration.items()},
                  window_selection_ap=by_width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and read(args.output) != policy:
        raise ValueError("Frozen policy differs. Use a new result root.")
    write(args.output, policy)
    print(f"Frozen width: {policy['radial_width']} -> {args.output}", flush=True)


def metrics(y, scores):
    if len(set(y)) != 2:
        return dict(auroc=None, ap=None, pr_auc=None)
    precision, recall, _ = precision_recall_curve(y, scores)
    return dict(auroc=float(roc_auc_score(y, scores)), ap=float(average_precision_score(y, scores)),
                pr_auc=float(auc(recall, precision)))


def table(path, rows):
    if not rows:
        return
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def paired(rows, labels, features, draws):
    if len(set(labels)) != 2:
        return []
    comparisons = {
        "radial_window64_vs_mean": [(1, "fisher_radial/w64"), (-1, "fisher_radial/mean")],
        "radial_frozen_vs_mean": [(1, "fisher_radial/frozen_window"), (-1, "fisher_radial/mean")],
        "geometry_increment": [(1, FUSED), (-1, PROB)],
        "normal_window64_vs_mean": [(1, "fisher_normal/w64"), (-1, "fisher_normal/mean")],
    }
    for baseline in ("nci", "fdbd_all", "shannon_top20"):
        comparisons[f"radial_window_gain_vs_{baseline}"] = comparisons["radial_window64_vs_mean"] + [
            (-1, f"{baseline}/w64"), (1, f"{baseline}/mean")]
    keys = sorted({key for spec in comparisons.values() for _, key in spec})
    blocks = defaultdict(list)
    for i, row in enumerate(rows):
        blocks[row["question_hash"]].append(i)
    blocks = list(blocks.values())
    y = np.asarray(labels)
    vals = {key: np.asarray([r[key] for r in features]) for key in keys}
    rng = np.random.default_rng(42)

    def effects(ix):
        scores = {key: np.array(list(metrics(y[ix], vals[key][ix]).values())) for key in keys}
        return np.array([sum(weight*scores[key] for weight,key in spec) for spec in comparisons.values()])

    estimates = effects(np.arange(len(y)))
    samples = []
    for _ in range(draws):
        ix = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        if len(set(y[ix])) == 2:
            samples.append(effects(ix))
    lows, highs = np.quantile(samples, [.025, .975], axis=0) if samples else (
        np.full(estimates.shape, np.nan), np.full(estimates.shape, np.nan))
    return [dict(comparison=key, metric=metric, delta=estimates[i,j], ci_low=lows[i,j] if samples else None,
                 ci_high=highs[i,j] if samples else None, draws=len(samples))
            for i,key in enumerate(comparisons) for j,metric in enumerate(("auroc", "ap", "pr_auc"))]


def make_groups(rows, task):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups["all"].append(i)
        if task == "supergpqa":
            groups["calculation" if row["is_calculation"] else "non-calculation"].append(i)
            groups["discipline/"+row["discipline"]].append(i)
        if task == "processbench":
            groups[row["subset"]].append(i)
        length = row["scored_tokens"]
        cutoff = next((c for c in (32, 64, 128, 256) if length <= c), None)
        groups[f"length/{cutoff or '257+'}"].append(i)
    return groups


def metric_table(path, scores, labels, groups):
    output = []
    for group, indices in groups.items():
        if not indices:
            continue
        y = [labels[i] for i in indices]
        for key in scores[0]:
            output.append(dict(group=group, method=key, n=len(y), errors=sum(y),
                               **metrics(y, [scores[i][key] for i in indices])))
    table(path, output)


def localize(row, values, step_ids, signs):
    gold = np.flatnonzero(step_ids == row["first_error"])
    a, b = int(gold[0]), int(gold[-1])
    for signal in (*WINDOW_SIGNALS, "mean_logprob"):
        risk = -signs[signal]*values[signal]
        windows = [(size, *window(risk, size)[1:]) for size in WIDTHS]
        _, drops = evidence_drop(-risk)
        windows.append(("evidence_drop", drops[0] if drops else None, 1))
        for size, start, width in windows:
            options = len(risk)-width+1
            random_hits = max(0, min(b, options-1)-max(0, a-width+1)+1)
            if start is None:
                relation, nearest = "no drop", None
            else:
                end = start+width
                relation = "before" if end <= a else "after" if start > b else "overlap"
                covered = [int(t) for t in step_ids[start:end] if t >= 0]
                nearest = min(covered, key=lambda s: abs(s-row["first_error"])) if covered else None
            yield dict(id=row["id"], subset=row["subset"], signal=signal, width=size, tokens=width,
                       first_error=row["first_error"], start_token=start, relation=relation, hit=relation=="overlap",
                       step_distance=None if nearest is None else nearest-row["first_error"],
                       random_hit=random_hits/options)


def step_features(row, values, step_ids, signs):
    results = []
    for idx in range(row["n_steps"]):
        positions = np.flatnonzero(step_ids == idx)
        end = int(positions[-1])+1
        features = {f"{key}/mean": -float(signs[key]*raw[positions].mean()) for key,raw in values.items()}
        for key in (*WINDOW_SIGNALS, "mean_logprob"):
            features[f"{key}/step"] = features[f"{key}/mean"]
            features[f"{key}/trailing64"] = -float(signs[key]*values[key][max(0,end-64):end].mean())
        features["shannon_top20/step_evidence_drop"] = evidence_drop(values["shannon_top20"][positions])[0]
        results.append(features)
    return results


def alarm_stats(gold, pred):
    bad = [(g,p) for g,p in zip(gold,pred) if g >= 0]
    clean = [p for g,p in zip(gold,pred) if g < 0]
    exact = sum(g == p for g,p in bad)/len(bad) if bad else None
    clean_ok = clean.count(-1)/len(clean) if clean else None
    f1 = None
    if exact is not None and clean_ok is not None:
        f1 = 2*exact*clean_ok/(exact+clean_ok) if exact+clean_ok else 0.
    return dict(exact_first_error=exact, official_f1=f1, n_solutions=len(gold),
                clean_false_alarm=None if clean_ok is None else 1-clean_ok,
                hit_within_one=sum(p >= 0 and abs(g-p) <= 1 for g,p in bad)/len(bad) if bad else None)


def eligible(gold, count):
    return range(count if gold < 0 else gold+1)


def alarm_eval(out, dev, test, groups):
    x, y = [], []
    for gold, features in dev:
        for step in eligible(gold, len(features)):
            x.append(features[step])
            y.append(int(step == gold))
    if len(set(y)) != 2 or not any(gold < 0 for gold,_ in dev):
        raise ValueError("ProcessBench development sample needs both erroneous and clean trajectories")
    models = fit_pair(x, y)
    for _, features in (*dev, *test):
        for row in features:
            fuse(row, models)
    methods = [key for key in test[0][1][0] if not key.endswith("/mean")]
    metrics_out, alarms_out, step_out, thresholds = [], [], [], {}
    for method in methods:
        dev_gold = [g for g,_ in dev]
        test_gold = [g for g,_ in test]
        dev_scores = [[r[method] for r in feat] for _,feat in dev]
        test_scores = [[r[method] for r in feat] for _,feat in test]

        def predict(values,t):
            return [next((i for i,v in enumerate(seq) if v > t), -1) for seq in values]

        flat = [v for seq in dev_scores for v in seq]
        candidates = np.unique(np.r_[np.nextafter(min(flat), -np.inf), np.quantile(flat, np.linspace(0,1,201))])
        threshold = max(candidates, key=lambda t: (alarm_stats(dev_gold,predict(dev_scores,t))["official_f1"],t))
        thresholds[method] = float(threshold)
        predicted = predict(test_scores, threshold)
        for i,(gold,seq) in enumerate(zip(test_gold,test_scores)):
            alarms_out.append(dict(row_index=i, method=method, gold_step=gold, alarm_step=predicted[i]))
            step_out.extend(dict(row_index=i, method=method, step=j, risk=seq[j], positive=int(j==gold))
                            for j in eligible(gold,len(seq)))
        for group,indices in groups.items():
            if group.startswith("length/"):
                continue
            pairs = [(int(j==test_gold[i]),test_scores[i][j]) for i in indices
                     for j in eligible(test_gold[i],len(test_scores[i]))]
            labels, scores = zip(*pairs)
            metrics_out.append(dict(method=method, group=group, n_steps=len(labels), positive_steps=sum(labels),
                                    **metrics(labels, scores), **alarm_stats([test_gold[i] for i in indices],
                                                                           [predicted[i] for i in indices])))
    table(out/"step-metrics.csv", metrics_out)
    table(out/"step-alarms.csv", alarms_out)
    table(out/"step-scores.csv", step_out)
    write(out/"step-policy.json", dict(fusion=models, thresholds=thresholds, n_dev=len(dev)))


def plot_windows(out, scores, labels):
    if len(set(labels)) != 2:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7,4))
    widths = list(WIDTHS[:-1])
    for signal in WINDOW_SIGNALS:
        ap = [100*average_precision_score(labels,[r[f"{signal}/w{w}"] for r in scores]) for w in widths]
        ax.plot(widths,ap,marker="o",label=signal)
    ax.set(xlabel="Window width (tokens)",ylabel="Average precision (%)",xscale="log",xticks=widths)
    ax.set_xticklabels(widths)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"window-ap.png",dpi=160)
    plt.close(fig)


def analyze(args):
    source, out = independent_output(args.input,args.output)
    manifest = completed(source)
    policy = read(args.policy)
    process = args.task == "processbench"
    if manifest["settings"].get("task", args.task) != args.task:
        raise ValueError("Dataset does not match analysis task")
    provenance = dict(task=args.task, input_signature=manifest["signature"],
                      policy_sha256=file_hash(args.policy), bootstrap=args.bootstrap)
    out.mkdir(parents=True,exist_ok=True)
    stamp = out/"analysis-protocol.json"
    if stamp.exists() and read(stamp) != provenance:
        raise ValueError("Analysis inputs changed. Use a new output directory.")
    write(stamp,provenance)
    signs = policy["signs"]
    rows, scores, capped, loc, dev_steps, test_steps, excluded = [], [], [], [], [], [], []
    elapsed, tokens, parse_failures, truncations = 0., 0, 0, 0
    for row in verified_rows(source,manifest):
        elapsed += row.get("elapsed_seconds",0.)
        tokens += row.get("generated_tokens",0)
        if args.task == "supergpqa":
            parse_failures += row["answer"] is None
            truncations += row["stop_reason"] == "max_new_tokens"
        if "excluded" in row:
            excluded.append(dict(id=row["id"],reason=row["excluded"]))
            continue
        trace = row.pop("token_scores")
        values = arrays(trace)
        if process:
            ids = np.array([-1 if t["step"] is None else t["step"] for t in trace])
            steps = (row["first_error"], step_features(row,values,ids,signs))
            if row["split"] == "dev":
                dev_steps.append(steps)
                continue
            test_steps.append(steps)
            if row["first_error"] >= 0:
                loc.extend(localize(row,values,ids,signs))
        row["scored_tokens"] = len(trace)
        features = summarize(values,signs)
        fuse(features,policy["fusion"])
        width = policy["radial_width"]
        features[f"{RADIAL}/frozen_window"] = features[f"{RADIAL}/"+("mean" if width=="mean" else f"w{width}")]
        scores.append(features)
        rows.append(row)
        if not process:
            capped.append({k:v for k,v in summarize(values,signs,budget=8).items() if "/w" in k})
    if not rows:
        raise ValueError("No scored test trajectories")
    groups = make_groups(rows,args.task)
    final = [int(r["error"]) for r in rows]
    metric_table(out/"final-answer-metrics.csv",scores,final,groups)
    labels = [int(r["first_error"] >= 0) for r in rows] if process else final
    prefix = "process-error" if process else "final-answer"
    table(out/f"{prefix}-paired.csv",paired(rows,labels,scores,args.bootstrap))
    table(out/"trajectory-scores.csv",[dict(row_index=i,id=r["id"],question_hash=r["question_hash"],
          scored_tokens=r["scored_tokens"],final_error=int(r["error"]),process_error=labels[i] if process else None,
          **scores[i]) for i,r in enumerate(rows)])
    if process:
        metric_table(out/"process-error-metrics.csv",scores,labels,
                     {**groups,"final-answer-correct":[i for i,r in enumerate(rows) if not r["error"]]})
        table(out/"localization.csv",loc)
        grouped = defaultdict(list)
        for item in loc:
            for group in ("all",item["subset"]):
                grouped[group,item["signal"],item["width"]].append(item)
        table(out/"localization-summary.csv",[dict(group=g,signal=s,width=w,n=len(items),
              hit_rate=float(np.mean([i["hit"] for i in items])),
              random_hit=float(np.mean([i["random_hit"] for i in items])),
              hit_gain=float(np.mean([i["hit"]-i["random_hit"] for i in items])),
              before=sum(i["relation"]=="before" for i in items),after=sum(i["relation"]=="after" for i in items))
              for (g,s,w),items in grouped.items()])
        alarm_eval(out,dev_steps,test_steps,groups)
    else:
        metric_table(out/"window-budget-8.csv",capped,labels,groups)
    table(out/"window-fallback.csv",[dict(group=g,width=w,n=len(ix),
          fallback=float(np.mean([width_for(w,rows[i]["scored_tokens"])==rows[i]["scored_tokens"] for i in ix])))
          for g,ix in groups.items() for w in WIDTHS])
    plot_windows(out,scores,labels)
    write(out/"summary.json",dict(**provenance,n=len(rows),input_count=manifest["count"],excluded=excluded,
          final_errors=sum(final),process_errors=sum(labels) if process else None,process_dev=len(dev_steps),
          parse_failures=parse_failures if args.task=="supergpqa" else None,
          truncations=truncations if args.task=="supergpqa" else None,
          scored_or_generated_tokens=tokens,seconds=elapsed,
          seconds_per_1000_tokens=1000*elapsed/tokens if tokens else None,radial_width=policy["radial_width"]))
    print(f"Analysis complete: {out}",flush=True)
