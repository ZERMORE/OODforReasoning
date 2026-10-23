"""Audit saved Qwen experiments and export reproducible PNG/SVG figures.

Reads inputs only. No torch, model downloads, GPU or network access required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .data import SPECS, digest, extract_answer
from .qwen_suite import PAPER, TASKS

LABELS = {"mean_logprob": "PPL (ranking equivalent)", "nci": "NCI", "fdbd_all": "fDBD (All)",
          "fdbd_k100": "fDBD (k=100)", "logit_margin": "Logit margin"}
COLORS = {"mean_logprob": "#64748b", "nci": "#2563eb", "fdbd_all": "#0d9488",
          "fdbd_k100": "#ea580c", "logit_margin": "#8b5cf6"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_main(path):
    path = Path(path).expanduser().resolve()
    candidates = [path, path / "main"] + list(path.glob("*/main"))
    matches = list(dict.fromkeys(p for p in candidates if (p / "qwen_csqa/metrics.json").is_file()))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one main experiment under {path}; found {len(matches)}. Specify the experiment/main directory explicitly.")
    return matches[0]


def selected(task):
    return "fdbd_all" if task == "gsm8k" else "fdbd_k100"


def load_cell(path, task, expected_count=None):
    path = Path(path)
    manifest, metrics = read_json(path / "manifest.json"), read_json(path / "metrics.json")
    rows = [json.loads(line) for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = SPECS[task][3] if expected_count is None else expected_count
    ids = [r["sample_id"] for r in rows]
    if len(rows) != expected or len(set(ids)) != expected or set(ids) != set(manifest["dataset"]["sample_ids"]):
        raise ValueError(f"{task}: incomplete/duplicate/mismatched sample IDs")
    if manifest["config"]["task"] != task or manifest["dataset"]["split"] != SPECS[task][2]:
        raise ValueError(f"{task}: task/split mismatch")
    for row in rows:
        parsed = extract_answer(row["response"], task)
        if parsed != row["parsed_answer"] or row["is_correct"] != (parsed == row["gold_answer"]):
            raise ValueError(f"{task}: stored answer/label mismatch at {row['sample_id']}")
    canonical = {r["sample_id"]: {k: r[k] for k in ("sample_id", "question", "options", "gold_answer")} for r in rows}
    sample_hash = digest([canonical[key] for key in manifest["dataset"]["sample_ids"]])
    if sample_hash != manifest["dataset"]["sample_hash"]:
        raise ValueError(f"{task}: normalized input content hash mismatch")
    keys = [key for key in LABELS if key != "fdbd_k100" or task != "gsm8k"]
    if any(not row.get("scores") for row in rows):
        raise ValueError(f"{task}: unscored rows require an explicit missing-score analysis")
    scores = {key: np.array([row["scores"][key] for row in rows], dtype=float) for key in keys}
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise ValueError(f"{task}: nonfinite scores")
    errors = np.array([not row["is_correct"] for row in rows], dtype=int)
    if np.unique(errors).size != 2:
        raise ValueError(f"{task}: AUROC requires both classes")
    recomputed = {key: float(roc_auc_score(errors, -value)) for key, value in scores.items()}
    if any(not np.isclose(value, metrics["auroc"][key], atol=1e-12, rtol=0) for key, value in recomputed.items()):
        raise ValueError(f"{task}: prediction AUROC differs from metrics.json")
    parse_fail = np.array([row["parsed_answer"] is None for row in rows])
    truncated = np.array([row["stop_reason"] == "max_new_tokens" for row in rows])
    for key, value in {"n": len(rows), "n_scored": len(rows), "n_unscored": 0,
                       "accuracy": float(1 - errors.mean()), "parse_failure_rate": float(parse_fail.mean()),
                       "truncation_rate": float(truncated.mean())}.items():
        if not np.isclose(value, metrics[key], atol=1e-12, rtol=0):
            raise ValueError(f"{task}: stored {key} mismatch")
    checks = [row["online_replay_verification"] for row in rows if row.get("online_replay_verification") is not None]
    if len(checks) < manifest["config"]["verify_replay"] or any(not check["passed"] for check in checks):
        raise ValueError(f"{task}: replay verification incomplete/failed")
    return {"task": task, "path": path, "rows": rows, "manifest": manifest, "metrics": metrics,
            "scores": scores, "errors": errors, "auroc": recomputed, "parse_fail": parse_fail,
            "truncated": truncated, "lengths": np.array([len(row["generated_token_ids"]) for row in rows]),
            "sample_hash": sample_hash, "verification_count": len(checks)}


def weighted_auc(errors, risk, counts):
    """Independent rank formula, with 0.5 credit for ties and bootstrap multiplicity."""
    _, groups = np.unique(risk, return_inverse=True)
    return grouped_auc(errors, groups, counts)


def grouped_auc(errors, groups, counts):
    positive = np.bincount(groups, weights=counts * errors)
    negative = np.bincount(groups, weights=counts * (1 - errors), minlength=len(positive))
    denominator = positive.sum() * negative.sum()
    if not denominator:
        return float("nan")
    wins = (positive * (negative.cumsum() - negative + .5 * negative)).sum()
    return float(wins / denominator)


def bootstrap_aucs(errors, scores, repeats=2000, seed=0):
    """Question resampling shared by all methods; matches evaluation.py RNG scheme.

    Replicating a question changes its multiplicity, not the score order. Grouped
    ranks therefore avoid sorting every method again for every bootstrap draw.
    """
    if repeats < 2:
        raise ValueError("At least two bootstrap draws are required")
    errors = np.asarray(errors, dtype=int)
    groups = {key: np.unique(-value, return_inverse=True)[1] for key, value in scores.items()}
    samples = {key: [] for key in scores}
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        counts = np.bincount(rng.integers(0, len(errors), len(errors)), minlength=len(errors))
        if not 0 < counts @ errors < len(errors):
            continue
        for key in scores:
            samples[key].append(grouped_auc(errors, groups[key], counts))
    if not len(next(iter(samples.values()))):
        raise ValueError("No valid bootstrap samples")
    return {key: np.array(value) for key, value in samples.items()}


def paired_result(cell, first, second):
    differences = cell["bootstrap"][first] - cell["bootstrap"][second]
    return {"first": first, "second": second,
            "delta_points": 100 * (cell["auroc"][first] - cell["auroc"][second]),
            "ci95_points": (100 * np.quantile(differences, [.025, .975])).tolist()}


def risk_coverage(errors, confidence):
    """Retain scores >= each distinct threshold, never break ties by row order."""
    order = np.argsort(-confidence, kind="stable")
    values = confidence[order]
    ends = np.r_[np.flatnonzero(values[:-1] != values[1:]) + 1, len(values)]
    return ends / len(values), np.cumsum(errors[order])[ends - 1] / ends


def save_figure(fig, output, stem):
    for extension in ("png", "svg"):
        fig.savefig(output / f"{stem}.{extension}", dpi=220, facecolor="white", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def plots(cells, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "svg.fonttype": "none"})
    ordered = list(LABELS)
    all_bounds = [bound * 100 for cell in cells for values in cell["bootstrap"].values()
                  for bound in np.quantile(values, [.025, .975])]
    all_bounds += [value for cell in cells for value in PAPER[cell["task"]].values()]
    auc_limits = (max(0, min(all_bounds) - 3), min(100, max(all_bounds) + 3))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, cell in zip(axes, cells):
        task = cell["task"]
        for i, key in enumerate(ordered):
            if key not in cell["scores"]:
                ax.text(np.mean(auc_limits), i, "not run (selected = All)", ha="center", va="center", color="#94a3b8", fontsize=8)
                continue
            value = 100 * cell["auroc"][key]
            low, high = 100 * np.quantile(cell["bootstrap"][key], [.025, .975])
            ax.plot([low, high], [i - .09] * 2, color=COLORS[key], lw=2)
            ax.plot(value, i - .09, "o", color=COLORS[key], ms=6)
            ax.annotate(f"{value:.2f}", (value, i - .09), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
            if key in PAPER[task]:
                ax.plot(PAPER[task][key], i + .13, marker="D", mfc="white", mec="#334155", ms=5)
        ax.set(yticks=range(len(ordered)), yticklabels=[LABELS[key] for key in ordered], xlabel="AUROC x 100",
               title=f"{task.upper()}  |  n={len(cell['rows']):,}", xlim=auc_limits, ylim=(4.65, -.65))
        ax.grid(axis="x", alpha=.15)
    fig.legend(handles=[Line2D([], [], marker="o", color="#2563eb", label="Reproduction + question-bootstrap 95% CI"),
                        Line2D([], [], marker="D", mfc="white", mec="#334155", ls="", label="Paper Table 4 (CI unavailable)")],
               loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.04), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save_figure(fig, output, "01_auroc_paper_comparison")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    gains = [bound for cell in cells for key in cell["scores"] if key != "mean_logprob"
             for bound in paired_result(cell, key, "mean_logprob")["ci95_points"]]
    gain_limits = (min(-1, min(gains)) - 1, max(1, max(gains)) + 1)
    for ax, cell in zip(axes, cells):
        keys = [key for key in ordered if key in cell["scores"] and key != "mean_logprob"]
        for i, key in enumerate(keys):
            result = paired_result(cell, key, "mean_logprob")
            lo, hi = result["ci95_points"]
            ax.plot([lo, hi], [i, i], color=COLORS[key], lw=2)
            ax.plot(result["delta_points"], i, "o", color=COLORS[key])
        ax.axvline(0, color="#475569", ls="--", lw=1)
        ax.set(yticks=range(len(keys)), yticklabels=[LABELS[key] for key in keys], title=cell["task"].upper(),
               xlabel="AUROC gain over PPL (percentage points)", xlim=gain_limits, ylim=(len(keys) - .5, -.5))
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("Paired question bootstrap: 95% intervals", y=1.02, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output, "02_paired_gain_over_ppl")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, cell in zip(axes, cells):
        for key, values in cell["scores"].items():
            fpr, tpr, _ = roc_curve(cell["errors"], -values)
            ax.plot(fpr, tpr, color=COLORS[key], lw=1.6, label=f"{LABELS[key]} ({cell['auroc'][key]:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="#cbd5e1", lw=1)
        ax.set(title=cell["task"].upper(), xlabel="False positive rate", ylabel="True positive rate (error detection)",
               xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    save_figure(fig, output, "03_roc_curves")

    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for row, cell in enumerate(cells):
        for col, key in enumerate(("mean_logprob", "nci", selected(cell["task"]))):
            ax = axes[row, col]
            values = cell["scores"][key]
            bins = np.histogram_bin_edges(values, bins=30)
            for is_error, color, name in [(False, "#2563eb", "Correct"), (True, "#dc2626", "Incorrect")]:
                ax.hist(values[cell["errors"] == is_error], bins=bins, density=True, histtype="stepfilled", alpha=.23,
                        color=color, edgecolor=color, label=name)
            ax.set(title=f"{cell['task'].upper()} / {LABELS[key]}", xlabel="Confidence score (higher = more confident)", ylabel="Class-normalized density")
            if row == 0 and col == 0:
                ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output, "04_confidence_distributions")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
    zoom_max = 0.
    for col, cell in enumerate(cells):
        for key, values in cell["scores"].items():
            coverage, risk = risk_coverage(cell["errors"], values)
            zoom_max = max(zoom_max, float(risk[coverage >= .2].max() * 100))
            for row in range(2):
                axes[row, col].plot(coverage * 100, risk * 100, color=COLORS[key], lw=1.5, label=LABELS[key])
        for row in range(2):
            axes[row, col].axhline(cell["errors"].mean() * 100, color="#94a3b8", ls="--", lw=1, label="Random retention (expected)")
            axes[row, col].set(title=cell["task"].upper() + (" / full range" if row == 0 else " / coverage >= 20%"),
                              xlabel="Retained answers (%)", ylabel="Retained-answer error rate (%)",
                              xlim=(0, 100) if row == 0 else (20, 100))
        axes[0, col].set_ylim(0, 100)
    for ax in axes[1]:
        ax.set_ylim(0, min(100, max(10, zoom_max * 1.15)))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(.5, .98), frameon=False, fontsize=9)
    fig.suptitle("Descriptive risk-coverage; no deployment guarantee / very low coverage is noisy", y=1.02, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .9))
    save_figure(fig, output, "05_risk_coverage")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, cell in zip(axes, cells):
        ax.boxplot([cell["lengths"][cell["errors"] == i] for i in (0, 1)], positions=[1, 2],
                   widths=.5, medianprops={"color": "#ea580c"}, flierprops={"markersize": 3, "alpha": .3})
        ax.set_xticks([1, 2], ["Correct", "Incorrect"])
        ax.set(title=cell["task"].upper(), ylabel="Generated tokens (log scale)", yscale="log")
        ax.grid(axis="y", alpha=.15)
    fig.tight_layout()
    save_figure(fig, output, "06_answer_lengths")


def analyze(main, output, repeats=2000, seed=0):
    from scipy.stats import spearmanr
    main, output = Path(main), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cells = [load_cell(main / f"qwen_{task}", task) for task in TASKS]
    recipe_path = main.parent / "recipe.json"
    recipe = read_json(recipe_path) if recipe_path.exists() else None
    shared = [cell["manifest"]["environment"]["implementation_hash"] for cell in cells]
    if len(set(shared)) != 1:
        raise ValueError("The three cells used different scoring implementations")
    audit, comparisons, failures = {}, [], []
    for cell in cells:
        task = cell["task"]
        if recipe and cell["sample_hash"] != recipe["datasets"][task]["sample_hash"]:
            raise ValueError(f"{task}: recipe/prediction input hash mismatch")
        if recipe and (cell["manifest"]["config"]["model_revision"] != recipe["model_revision"] or
                       cell["manifest"]["config"]["dataset_revision"] != recipe["datasets"][task]["revision"]):
            raise ValueError(f"{task}: recipe/model/dataset revision mismatch")
        cell["bootstrap"] = bootstrap_aucs(cell["errors"], cell["scores"], repeats, seed)
        winner = selected(task)
        clean = ~cell["parse_fail"] & ~cell["truncated"]
        clean_auroc = {key: float(roc_auc_score(cell["errors"][clean], -values[clean])) for key, values in cell["scores"].items()} if len(np.unique(cell["errors"][clean])) == 2 else {}
        pairs = [paired_result(cell, first, second) for first, second in [("nci", "mean_logprob"),
                 (winner, "mean_logprob"), (winner, "nci"), (winner, "logit_margin")]]
        if winner != "fdbd_all":
            pairs.append(paired_result(cell, winner, "fdbd_all"))
        for pair in pairs:
            pair["task"] = task
        comparisons.extend(pairs)
        stored_ci_difference = {key: float(np.max(np.abs(np.quantile(value, [.025, .975]) - cell["metrics"]["auroc_ci95"][key])))
                                for key, value in cell["bootstrap"].items()} if repeats == cell["metrics"].get("bootstrap", {}).get("requested") and seed == cell["metrics"].get("bootstrap", {}).get("seed") else {}
        audit[task] = {"n": len(cell["rows"]), "correct": int((1 - cell["errors"]).sum()), "incorrect": int(cell["errors"].sum()),
                       "parse_failures": int(cell["parse_fail"].sum()), "truncated": int(cell["truncated"].sum()),
                       "parse_and_truncated": int((cell["parse_fail"] & cell["truncated"]).sum()),
                       "replay_verifications": cell["verification_count"], "point_metrics_verified": True,
                       "input_hash_verified": True, "bootstrap_ci_max_abs_difference_from_saved": stored_ci_difference,
                       "clean_subset_n": int(clean.sum()), "clean_subset_auroc": clean_auroc,
                       "median_tokens_correct": float(np.median(cell["lengths"][cell["errors"] == 0])),
                       "median_tokens_incorrect": float(np.median(cell["lengths"][cell["errors"] == 1])),
                       "length_score_spearman": {key: float(spearmanr(cell["lengths"], values).statistic) for key, values in cell["scores"].items()},
                       "configuration": cell["manifest"]["config"], "environment": cell["manifest"]["environment"]}
        for key, paper in PAPER[task].items():
            comparisons.append({"task": task, "kind": "paper_reference", "method": key,
                                "measured_x100": 100 * cell["auroc"][key], "paper_x100": paper,
                                "difference_points": 100 * cell["auroc"][key] - paper,
                                "ci95_x100": (100 * np.quantile(cell["bootstrap"][key], [.025, .975])).tolist()})
        for row in cell["rows"]:
            if row["parsed_answer"] is None or row["stop_reason"] == "max_new_tokens":
                failures.append({"task": task, "sample_id": row["sample_id"], "gold_answer": row["gold_answer"],
                                 "parsed_answer": row["parsed_answer"], "stop_reason": row["stop_reason"],
                                 "generated_tokens": len(row["generated_token_ids"]), "response": row["response"]})
    plots(cells, output)
    sources = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
               for cell in cells for path in [cell["path"] / f for f in ("manifest.json", "metrics.json", "predictions.jsonl")]}
    results = {"created": datetime.now(timezone(timedelta(hours=8))).isoformat(), "source_main": str(main.resolve()),
               "analysis_environment": {"python": platform.python_version(),
                                        "packages": {key: importlib.metadata.version(key) for key in ("numpy", "scikit-learn", "scipy", "matplotlib")},
                                        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
               "bootstrap": {"requested": repeats, "seed": seed, "unit": "question", "interval": "percentile 95%; same question resample for all methods"},
               "audit": audit, "comparisons": comparisons, "source_sha256": sources,
               "limitations": ["One model checkpoint and greedy run; bootstrap reflects question sampling, not model/prompt/seed uncertainty.",
                               "Paper references have no supplied raw predictions or CIs; cross-study point differences are not paired significance tests.",
                               "Clean subset is a sensitivity analysis with changed sample composition; it does not repair or replace full-set results.",
                               "Risk-coverage is descriptive on the evaluation set, not calibrated deployment risk."]}
    (output / "analysis.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "paper_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        items = [row for row in comparisons if row.get("kind") == "paper_reference"]
        writer = csv.DictWriter(stream, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)
    with (output / "failure_cases.jsonl").open("w", encoding="utf-8") as stream:
        for row in failures:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report(cells, results, output)
    print(f"Audited all three full splits. Analysis and PNG/SVG figures: {output.resolve()}")
    return results


def report(cells, results, output):
    lines = ["# Qwen2.5-7B-Instruct 复现结果分析", "", "## 结论与完整性", "",
             "三组为完整评测 split；已独立核对问题 ID、输入内容 hash、严格答案抽取、正确性标签、分数有限性及 AUROC 与保存指标的一致性。",
             "单次 greedy 实验只能评估本配置的复现，不等于跨模型、prompt 或随机种子的稳定结论。", "",
             "|任务|正确 / 总数|错误|抽取失败|截断|在线重放验证数|", "|---|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        a = results["audit"][task]
        lines.append(f"|{task}|{a['correct']}/{a['n']}|{a['incorrect']}|{a['parse_failures']}|{a['truncated']}|{a['replay_verifications']}|")
    lines += ["", "## 与论文 Table 4 对照", "", "以下为 AUROC×100；差值单位为百分点。CI 来自本次数据的问题级 bootstrap，不包含论文实验的不确定性。",
              "", "|任务|方法|本次 [95% CI]|论文|本次−论文|", "|---|---|---:|---:|---:|"]
    for row in results["comparisons"]:
        if row.get("kind") == "paper_reference":
            lo, hi = row["ci95_x100"]
            lines.append(f"|{row['task']}|{LABELS[row['method']]}|{row['measured_x100']:.2f} [{lo:.2f}, {hi:.2f}]|{row['paper_x100']:.2f}|{row['difference_points']:+.2f}|")
    lines += ["", "## 同一批问题的配对比较", "", "CI 不跨 0 才在本次 percentile bootstrap 标准下支持方向性差异；这里未做多重比较校正，不把探索性比较解释为确证检验。",
              "", "|任务|比较|AUROC差值 [95% CI]，百分点|判断|", "|---|---|---:|---|"]
    for row in results["comparisons"]:
        if row.get("kind") == "paper_reference":
            continue
        lo, hi = row["ci95_points"]
        verdict = "CI跨0，无法区分" if lo <= 0 <= hi else ("正向差异" if lo > 0 else "负向差异")
        lines.append(f"|{row['task']}|{LABELS[row['first']]} − {LABELS[row['second']]}|{row['delta_points']:+.2f} [{lo:+.2f}, {hi:+.2f}]|{verdict}|")
    lines += ["", "## 格式与截断敏感性", "", "主结果保留所有题并将抽取失败计错；以下排除抽取失败和截断，改变了样本组成，只用于排查，不能作为新的主成绩。", "",
              "|任务|保留题数|selected fDBD 原始→子集 AUROC×100|NCI 原始→子集|", "|---|---:|---:|---:|"]
    for cell in cells:
        task = cell["task"]
        a = results["audit"][task]
        key = selected(task)
        def clean_fmt(method):
            value = a["clean_subset_auroc"].get(method)
            return "undefined (single class)" if value is None else f"{100 * value:.2f}"
        lines.append(f"|{task}|{a['clean_subset_n']}|{100 * cell['auroc'][key]:.2f} → {clean_fmt(key)}|{100 * cell['auroc']['nci']:.2f} → {clean_fmt('nci')}|")
    lines += ["", "AQuA 应优先检查 `failure_cases.jsonl` 中的长推理、重复生成和未遵循答案模板的例子。抽取失败不等于被抽取器漏掉的正确答案，需逐例审阅；不能仅放宽正则来提高成绩。",
              "", "## 解释边界与下一步", "",
              "- 结合配对 CI 判断几何分数是否优于 PPL；不要仅看单个 AUROC 点值。",
              "- 直接比较 selected fDBD 与 NCI、logit margin，检查几何计算是否提供额外信息；不要从本次评测集重新选择 k。",
              "- AQuA 与论文差距需先审查 prompt 来源、序列化、停止规则及生成上限，已有 v4 复现协议差异仍有效。",
              "- 512-token 上限敏感性应使用独立新实验目录并对完整 split 重跑；不能只重跑错误题或挑最好配置替换主结果。",
              "- `length_score_spearman` 和长度图仅提示混杂关联，不提供因果解释。风险覆盖曲线也不提供部署保证。",
              "- 原论文没有提供本次可用的逐题预测或 CI，不能对本次−论文差值做配对显著性检验。",
              "- 本次同时计算多个评分，日志中的整体耗时不是单个检测器的增量延迟；当前 RTX 6000 Ada 运行也不能直接验证论文 A100 的低开销比例。",
              "", "## 图表", ""]
    for stem, title in [("01_auroc_paper_comparison", "主表对照与本次95% CI"), ("02_paired_gain_over_ppl", "相对PPL的配对增益"),
                        ("03_roc_curves", "错误检测ROC"), ("04_confidence_distributions", "正确/错误回答的置信分布"),
                        ("05_risk_coverage", "同一评测集上的描述性风险覆盖曲线"), ("06_answer_lengths", "生成长度分布")]:
        lines += [f"### {title}", "", f"![{title}]({stem}.png)", ""]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="OODResults, experiment root, or main directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory for report and PNG/SVG artifacts")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    source = find_main(args.results)
    output = args.output.expanduser().resolve()
    if output == source or source in output.parents or output in source.parents:
        parser.error("Output must be separate from input main and its ancestors")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is nonempty; use a new directory to preserve previous analyses")
    analyze(source, output, args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
