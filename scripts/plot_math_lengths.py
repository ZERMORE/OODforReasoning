"""Show MATH scored-token lengths and the relative size of a 256-token window."""
import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.source.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    length = np.array([r["scored_tokens"] for r in rows])
    correct = np.array([r["is_correct"] for r in rows], dtype=bool)
    assert len(set(r["sample_id"] for r in rows)) == len(rows)
    assert (length > 0).all()
    groups = [("All", np.ones(len(rows), dtype=bool)),
              ("Correct", correct), ("Incorrect", ~correct)]
    stats = {}
    for name, mask in groups:
        v = length[mask]
        stats[name] = dict(n=len(v), min=int(v.min()), max=int(v.max()),
                          mean=float(v.mean()), q25=float(np.quantile(v, .25)),
                          median=float(np.median(v)), q75=float(np.quantile(v, .75)),
                          p90=float(np.quantile(v, .9)),
                          at_most_256=int((v <= 256).sum()),
                          at_most_512=int((v <= 512).sum()),
                          at_most_1024=int((v <= 1024).sum()),
                          median_window_fraction=float(np.median(np.minimum(256, v) / v)))
    colors = ("#0072B2", "#A34000")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    fig.suptitle("MATH response lengths and a 256-token window | 2,500 trajectories", fontsize=15)
    edges = np.arange(0, max(2048, length.max()) + 128, 128)
    ax = axes[0]
    for (name, mask), color, style in zip(groups[1:], colors, ("-", "--")):
        counts, _ = np.histogram(length[mask], bins=edges)
        ax.stairs(counts / mask.sum() * 100, edges, label=f"{name} (n={mask.sum():,})",
                  color=color, linestyle=style, linewidth=1.6)
    ax.axvline(256, color="#333333", linestyle=":", label="256 tokens")
    ax.set(xlabel="Scored response tokens", ylabel="Within-group percentage",
           title="Length distribution | 128-token bins", xlim=(0, 2048), ylim=(0, None))
    ax.legend(fontsize=8)

    ax = axes[1]
    for (name, mask), color, style in zip(groups[1:], colors, ("-", "--")):
        values = np.sort(length[mask])
        ax.step(values, np.arange(1, len(values) + 1) / len(values) * 100,
                color=color, linestyle=style, label=name)
    for w in (256, 512, 1024):
        ax.axvline(w, color="#AAAAAA", linestyle=":", linewidth=.8)
    ax.set(xlabel="Scored response tokens", ylabel="Cumulative percentage",
           title="Fraction of responses below each length", xlim=(0, 2048), ylim=(0, 100))
    ax.legend(fontsize=8)

    ax = axes[2]
    # Effective window falls back to whole response when T < 256.
    fraction = np.minimum(256, length) / length
    bins = [0, .25, .5, .75, 1.]
    categories = ("0–25%", ">25–50%", ">50–75%", ">75–100%")
    positions = np.arange(4)
    for offset, (name, mask), color, hatch in zip((-.19, .19), groups[1:], colors, ("", "//")):
        values = fraction[mask]
        counts = [((values >= 0) & (values <= .25)).sum(),
                  ((values > .25) & (values <= .5)).sum(),
                  ((values > .5) & (values <= .75)).sum(),
                  ((values > .75) & (values <= 1)).sum()]
        pct = np.array(counts) / len(values) * 100
        bars = ax.bar(positions + offset, pct, width=.36, color=color,
                      hatch=hatch, label=name)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in pct], fontsize=8, padding=3)
    ax.set(xticks=positions, xticklabels=categories, xlabel="Effective window / response length",
           ylabel="Within-group percentage", title="How much of a response does 256 cover?",
           ylim=(0, 65))
    ax.legend(fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    args.output.mkdir(exist_ok=False, parents=True)
    fig.savefig(args.output / "math-token-lengths.png", dpi=180, facecolor="white")
    fig.savefig(args.output / "math-token-lengths.pdf", facecolor="white")
    plt.close(fig)
    with Image.open(args.output / "math-token-lengths.png") as im:
        im.verify()
    report = dict(source=str(args.source), definition="scored response tokens, excludes prompt and unscored EOS",
                  statistics=stats,
                  truncated=sum(r["stop_reason"] == "max_new_tokens" for r in rows),
                  histogram_bin_edges=edges.tolist(),
                  fraction_definition="min(256,T)/T; intervals (a,b] except first [0,.25]",
                  per_trajectory=[dict(sample_id=r["sample_id"], tokens=r["scored_tokens"],
                                       correct=r["is_correct"], stop_reason=r["stop_reason"]) for r in rows])
    (args.output / "length-statistics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(statistics=stats, truncated=report["truncated"]), indent=2))


if __name__ == "__main__":
    main()

