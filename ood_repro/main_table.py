"""Export a paper-style table of ALL scores actually evaluated in the Qwen run."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .qwen_suite import TASKS
from .visualize import find_main, load_cell

METHODS = [
    ("mean_logprob", "Perplexity (ranking equivalent)"),
    ("log_predictive_probability", "Predictive probability (log domain)"),
    ("max_probability", "Mean maximum token probability"),
    ("negative_entropy", "Negative token entropy"),
    ("logit_margin", "Logit margin"),
    ("nci", "NCI"),
    ("nci_decoded", "NCI (decoded token)"),
    ("fdbd_all", "fDBD (All)"),
    ("fdbd_k100", "fDBD (k = 100)"),
    ("selected", "fDBD (paper-selected k)"),
]


def verify_all_scores(cells):
    """The complete table includes diagnostic scores beyond the five plot methods."""
    for task, cell in cells.items():
        for key, saved in cell["metrics"]["auroc"].items():
            scores = np.array([row["scores"][key] for row in cell["rows"]])
            if not np.isfinite(scores).all() or not np.isclose(roc_auc_score(cell["errors"], -scores), saved, atol=1e-12, rtol=0):
                raise ValueError(f"{task}: invalid or stale metric {key}")


def build_rows(cells):
    rows = []
    extra = [("deepconf_mean", "DeepConf top-20 (mean)"),
             ("deepconf_lowest", "DeepConf top-20 (lowest window)"),
             ("deepconf_bottom10", "DeepConf top-20 (bottom 10% windows)")]
    methods = METHODS[:5] + [(k, label) for k, label in extra
                            if any(k in cell["metrics"]["auroc"] for cell in cells.values())] + METHODS[5:]
    for key, label in methods:
        row = {"method": key, "label": label, "single_sample": True, "tasks": {}}
        for task in TASKS:
            resolved = ("fdbd_all" if task == "gsm8k" else "fdbd_k100") if key == "selected" else key
            metrics = cells[task]["metrics"]
            value = metrics["auroc"].get(resolved)
            ci = metrics.get("auroc_ci95", {}).get(resolved)
            row["tasks"][task] = {"metric_key": resolved, "auroc": None if value is None else 100 * value,
                                   "ci95": None if ci is None else [100 * v for v in ci]}
        rows.append(row)
    return rows


def rank(value, values):
    if value is None:
        return None
    distinct = []
    for other in sorted((v for v in values if v is not None), reverse=True):
        if not any(np.isclose(other, x, atol=1e-10, rtol=0) for x in distinct):
            distinct.append(other)
    return next(i + 1 for i, x in enumerate(distinct) if np.isclose(value, x, atol=1e-10, rtol=0))


def cell_text(value, with_ci=False):
    if value["auroc"] is None:
        return "—"
    text = f"{value['auroc']:.2f}"
    if with_ci and value["ci95"] is not None:
        text += f" [{value['ci95'][0]:.2f}, {value['ci95'][1]:.2f}]"
    return text


def render(rows, output, with_ci=False, deepconf_note=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(17 if with_ci else 13.5, 7.3))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    fig.subplots_adjust(left=.025, right=.985, top=.96, bottom=.03)
    ax.text(0, .98, "Qwen2.5-7B-Instruct: main experimental results", fontsize=17, weight="bold", va="top")
    subtitle = "AUROC x 100 (higher is better)" + ("  |  question-bootstrap 95% CI" if with_ci else "  |  greedy decoding, single generated response")
    ax.text(0, .928, subtitle, fontsize=11, color="#475569", va="top")
    xs = [.067, .161, .572, .686, .819, .947]
    ax.plot([0, 1], [.872, .872], color="black", lw=1.5)
    for x, label in zip(xs, ["Model", "Method", "Single\nsample", "CSQA\n(n = 1,221)", "GSM8K\n(n = 1,319)", "AQuA\n(n = 254)"]):
        ax.text(x, .827, label, ha="left" if label == "Method" else "center", va="center", fontsize=11, weight="bold")
    ax.plot([0, 1], [.784, .784], color="black", lw=.9)
    row_height = .49 / max(10, len(rows))
    ax.text(xs[0], .784 - row_height * len(rows) / 2, "Qwen2.5-7B\n-Instruct", ha="center", va="center", fontsize=11, linespacing=1.5)
    underline = []
    for i, row in enumerate(rows):
        y = .784 - row_height * (i + .5)
        if row["method"] in {"nci", "nci_decoded", "fdbd_all", "fdbd_k100", "selected"}:
            ax.add_patch(Rectangle((.148, y - row_height / 2), .852, row_height, fc="#edf2f7", ec="none", zorder=-1))
        if row["method"] == "nci":
            ax.plot([.148, 1], [y + row_height / 2] * 2, color="#94a3b8", lw=.6)
        ax.text(xs[1], y, row["label"], ha="left", va="center", fontsize=10.5, weight="bold" if row["method"] == "selected" else "normal")
        ax.text(xs[2], y, "Yes", ha="center", va="center", fontsize=10.5)
        for task, x in zip(TASKS, xs[3:]):
            value = row["tasks"][task]
            position = rank(value["auroc"], [r["tasks"][task]["auroc"] for r in rows])
            text = ax.text(x, y, cell_text(value, with_ci), ha="center", va="center", fontsize=10.2 if with_ci else 11,
                           weight="bold" if position == 1 else "normal", color="#94a3b8" if position is None else "black")
            if position == 2:
                underline.append(text)
    bottom = .784 - row_height * len(rows)
    ax.plot([0, 1], [bottom, bottom], color="black", lw=1.5)
    notes = ["Bold: highest point estimate. Underline: second distinct estimate; ties share rank. Ranking does not imply statistical significance.",
             "Paper-selected k: CSQA = 100, GSM8K = All, AQuA = 100. This row reuses the corresponding measurements.",
             "NCI (decoded token) equals NCI under greedy decoding. Dash: not run; no values are imputed.",
             "All values are this reproduction's measurements, not paper reference values. PPL is represented by mean log probability.",
             "Mean maximum token probability is a diagnostic baseline; it is not asserted to reproduce the paper's Max P implementation."]
    if deepconf_note:
        notes.append(deepconf_note)
    for i, note in enumerate(notes):
        ax.text(0, bottom - .035 - i * .031, note, fontsize=9.2, color="#334155", va="top")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in underline:
        box = text.get_window_extent(renderer=renderer)
        (x0, y0), (x1, _) = ax.transData.inverted().transform([[box.x0, box.y0 - 1], [box.x1, box.y0 - 1]])
        ax.plot([x0, x1], [y0, y0], color="black", lw=.75)
    stem = "main_results_with_ci" if with_ci else "main_results"
    for extension in ("png", "svg"):
        fig.savefig(output / f"{stem}.{extension}", dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def export(rows, output):
    lines = ["# Qwen2.5-7B-Instruct 完整主实验表", "", "AUROC×100，越高越好；粗体为最高点值，下划线为第二个不同点值，等值并列。", "",
             "|模型|方法|单次生成|CSQA|GSM8K|AQuA|", "|---|---|:---:|---:|---:|---:|"]
    ci_lines = ["# 完整主实验表（含问题级 bootstrap 95% CI）", "", "|方法|CSQA|GSM8K|AQuA|", "|---|---:|---:|---:|"]
    csv_rows, tex_rows = [], []
    for i, row in enumerate(rows):
        values, tex_values = [], []
        item = {"model": "Qwen2.5-7B-Instruct", "method": row["method"], "method_label": row["label"], "single_sample": True}
        for task in TASKS:
            value = row["tasks"][task]
            position = rank(value["auroc"], [r["tasks"][task]["auroc"] for r in rows])
            raw = cell_text(value)
            values.append(f"**{raw}**" if position == 1 else f"<u>{raw}</u>" if position == 2 else raw)
            latex = "---" if value["auroc"] is None else raw
            if position == 1:
                latex = "\\textbf{" + latex + "}"
            elif position == 2:
                latex = "\\underline{" + latex + "}"
            tex_values.append(latex)
            item[f"{task}_auroc_x100"] = value["auroc"]
            item[f"{task}_ci95_low"] = value["ci95"][0] if value["ci95"] else None
            item[f"{task}_ci95_high"] = value["ci95"][1] if value["ci95"] else None
        lines.append("|" + ("Qwen2.5-7B-Instruct" if i == 0 else "") + f"|{row['label']}|Yes|" + "|".join(values) + "|")
        ci_lines.append(f"|{row['label']}|" + "|".join(cell_text(row["tasks"][task], True) for task in TASKS) + "|")
        prefix = "\\multirow{" + str(len(rows)) + "}{*}{Qwen2.5-7B-Instruct}" if i == 0 else ""
        if i == 5:
            tex_rows.append("\\cmidrule(l){2-6}")
        tex_rows.append(prefix + " & " + row["label"] + " & Yes & " + " & ".join(tex_values) + " \\\\")
        csv_rows.append(item)
    notes = ["", "- selected k 按论文预先固定为 100 / All / 100；该行复用已测分数，不是新增实验。",
             "- NCI decoded 与 NCI 在 greedy 下相同；同分并列，不当作独立证据。",
             "- GSM8K 未运行 k=100，保留缺失；未运行的论文基线（CoE、P(True)、多样本方法等）不填造数字。",
             "- 排名依据未舍入的点值，不表示显著优于其他方法；统计判断见配对分析报告。",
             "- mean maximum probability 为本实现诊断基线，不声称与论文 Max P 完全一致。"]
    (output / "main_results.md").write_text("\n".join(lines + notes) + "\n", encoding="utf-8")
    (output / "main_results_with_ci.md").write_text("\n".join(ci_lines + notes) + "\n", encoding="utf-8")
    with (output / "main_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    latex = ["% Requires booktabs and multirow.", "\\begin{table*}[t]", "\\centering", "\\small",
             "\\caption{Reproduced Qwen2.5-7B-Instruct results. AUROC (percent), higher is better. Bold: best; underline: second distinct score.}",
             "\\begin{tabular}{llcrrr}", "\\toprule", "Model & Method & Single sample & CSQA & GSM8K & AQuA \\\\", "\\midrule",
             *tex_rows, "\\bottomrule", "\\end{tabular}",
             "\\par\\smallskip\\footnotesize Selected $k$: 100 / All / 100. Selected and decoded-token rows reuse measurements; --- means not run.", "\\end{table*}"]
    (output / "main_results.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = find_main(args.results)
    output = args.output.expanduser().resolve()
    if output == source or source in output.parents or output in source.parents or (output.exists() and any(output.iterdir())):
        parser.error("Use a new, empty output directory separate from input main and its ancestors")
    cells = {task: load_cell(source / f"qwen_{task}", task) for task in TASKS}
    verify_all_scores(cells)
    settings = {tuple(cell["manifest"].get("deepconf", {}).get(k) for k in ("top_k", "window", "stride"))
                for cell in cells.values() if "deepconf_mean" in cell["metrics"]["auroc"]}
    if len(settings) > 1 or (settings and (None in next(iter(settings)) or next(iter(settings))[0] != 20)):
        parser.error("DeepConf table requires consistent k=20, window and stride metadata")
    note = None
    if settings:
        k, window, stride = next(iter(settings))
        note = f"DeepConf: top-k = {k}, window = {window}, stride = {stride}; shorter traces use one full-trace window. Single-trace scoring only."
    output.mkdir(parents=True, exist_ok=True)
    rows = build_rows(cells)
    export(rows, output)
    render(rows, output, deepconf_note=note)
    render(rows, output, with_ci=True, deepconf_note=note)
    if note:
        for name in ("main_results.md", "main_results_with_ci.md"):
            with (output / name).open("a", encoding="utf-8") as stream:
                stream.write("\n" + note + "\n")
    (output / "main_results.json").write_text(json.dumps({"source_main": str(source), "units": "AUROC x 100", "deepconf_note": note, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Full main table exported to {output}")


if __name__ == "__main__":
    main()
