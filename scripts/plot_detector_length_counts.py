"""Plot detector decisions by response length at a fixed screening budget."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image

METHODS = (
    ("nci/mean", "NCI + mean", "#0072B2", "o", "-"),
    ("fdbd_all/mean", "fDBD + mean", "#A34000", "s", "--"),
    ("shannon_top20/evidence_drop", "Shannon Drop", "#007C62", "^", ":"),
)
SOURCES = (
    ("MATH", "evidence-math-half-v1/math/fit-no-shuffle", 128),
    ("GSM8K", "evidence-crossover-v1/gsm8k/fit", 32),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=.25)
    args = parser.parse_args()
    if not 0 < args.fraction < 1 or args.output.exists():
        parser.error("Need a new output directory and fraction strictly between zero and one")
    data = []
    records = []
    for task, relative, width in SOURCES:
        source = args.root / relative / "test_predictions.jsonl"
        with source.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        ids = np.array([row["sample_id"] for row in rows])
        assert len(set(ids)) == len(rows)
        length = np.array([row["scored_tokens"] for row in rows])
        wrong = np.array([not row["is_correct"] for row in rows])
        edges = np.arange(0, length.max() + width + 1, width)
        center = (edges[:-1] + edges[1:]) / 2
        k = int(np.ceil(len(rows) * args.fraction))
        decisions = {}
        table = {"task": task, "source": str(source), "n": len(rows), "screened": k,
                 "bin_width": width, "bin_edges": edges.tolist(), "methods": {}}
        for key, name, color, marker, style in METHODS:
            risk = -np.array([row["confidence"][key]["fixed"] for row in rows])
            order = np.lexsort((ids, -risk))
            predicted_wrong = np.zeros(len(rows), dtype=bool)
            predicted_wrong[order[:k]] = True
            counts = {}
            for label, mask in (
                ("predicted_wrong", predicted_wrong),
                ("false_error", predicted_wrong & ~wrong),
                ("false_correct", ~predicted_wrong & wrong),
            ):
                hist, _ = np.histogram(length[mask], bins=edges)
                assert hist.sum() == mask.sum()
                counts[label] = hist
            decisions[key] = counts
            table["methods"][key] = {label: count.tolist() for label, count in counts.items()}
            table["methods"][key]["predicted_wrong_ids"] = ids[predicted_wrong].tolist()
        table["all_trajectories"] = np.histogram(length, bins=edges)[0].tolist()
        table["median_tokens"] = float(np.median(length))
        assert sum(table["all_trajectories"]) == len(rows)
        data.append((task, center, edges, k, decisions, width))
        records.append(table)

    def decorate(ax, center, edges, task, title, decisions, kind, width):
        for (key, name, color, marker, style), hatch in zip(METHODS, ("///", "xxx", "...")):
            ax.stairs(decisions[key][kind], edges, baseline=0, fill=True,
                      label=name, facecolor=color, edgecolor=color, alpha=.28,
                      hatch=hatch, linewidth=1.)
        ax.set(title=f"{task} | {title}", xlabel="Response length (scored tokens)",
               ylabel="Number of trajectories", xlim=(0, edges[-1]), ylim=(0, None))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(.98, .96, f"Bin width = {width} tokens", transform=ax.transAxes,
                ha="right", va="top", fontsize=9)

    args.output.mkdir(parents=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    fig.subplots_adjust(left=.07, right=.98, bottom=.19, top=.76, wspace=.2)
    fig.suptitle("Trajectories flagged as incorrect, by response length", fontsize=16, y=.97)
    for ax, (task, center, edges, k, decisions, width) in zip(axes, data):
        decorate(ax, center, edges, task, f"{k} flagged by each detector", decisions, "predicted_wrong", width)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .9), ncol=3, frameon=False)
    fig.text(.5, .035, f"Each detector flags its highest-risk {args.fraction:.0%}. "
             "Counts include both actual-correct and actual-incorrect responses. Filled histograms, independently counted for each detector.",
             ha="center", fontsize=9, color="#465562")
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"flagged-by-length.{extension}", dpi=200, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.subplots_adjust(left=.07, right=.98, bottom=.12, top=.84, hspace=.4, wspace=.2)
    fig.suptitle("Detector mistakes, by response length", fontsize=16, y=.98)
    for rowaxes, (task, center, edges, k, decisions, width) in zip(axes, data):
        for ax, kind, title in zip(rowaxes, ("false_error", "false_correct"),
                                   ("Correct answer flagged incorrect", "Incorrect answer accepted")):
            decorate(ax, center, edges, task, title, decisions, kind, width)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
    fig.text(.5, .035, f"Same highest-risk {args.fraction:.0%} screening budget. "
             "Independent axis ranges per panel; filled areas show counts in length bins.",
             ha="center", fontsize=9, color="#465562")
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"mistakes-by-length.{extension}", dpi=200, facecolor="white")
    plt.close(fig)
    for row in records:
        task = row["task"]
        edges = np.array(row["bin_edges"])
        counts = np.array(row["all_trajectories"])
        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        ax.stairs(counts, edges, baseline=0, fill=True,
                  facecolor="#0072B2", edgecolor="#005987", alpha=.55, linewidth=1.)
        ax.set(title=f"{task} | All {row['n']:,} test trajectories",
               xlabel="Response length (scored tokens)", ylabel="Number of trajectories",
               xlim=(0, edges[-1]), ylim=(0, None))
        ax.text(.97, .95, f"Bin width = {row['bin_width']} tokens\nMedian = {row['median_tokens']:g} tokens",
                transform=ax.transAxes, ha="right", va="top")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines[["top", "right"]].set_visible(False)
        for extension in ("png", "pdf"):
            fig.savefig(args.output / f"all-trajectories-{task.lower()}.{extension}",
                        dpi=200, facecolor="white")
        plt.close(fig)
    for name in ("flagged-by-length.png", "mistakes-by-length.png",
                 "all-trajectories-math.png", "all-trajectories-gsm8k.png"):
        with Image.open(args.output / name) as image:
            image.verify()
    (args.output / "plot-data.json").write_text(
        json.dumps(dict(fraction=args.fraction,
                        tie_break="sample_id lexicographic, same as overlap analysis",
                        bin_semantics="[left,right), final bin right-inclusive",
                        datasets=records), indent=2) + "\n", encoding="utf-8")
    print(args.output)
    for r in records:
        print(r["task"], "flagged", r["screened"],
              [(key, sum(v["false_error"]), sum(v["false_correct"])) for key, v in r["methods"].items()])


if __name__ == "__main__":
    main()

