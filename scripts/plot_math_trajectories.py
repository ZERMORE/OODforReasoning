from argparse import ArgumentParser
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from ood_repro.evidence import evidence_drop

METRICS = (
    ("shannon_top20", "Top-20 negative Shannon entropy", "#0072B2"),
    ("fdbd_all", "fDBD All", "#A34000"),
    ("nci", "NCI", "#007C62"),
)
SPAN, WORST = 5, 5


def read_rows(source):
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def choose_pairs(source, seed, pairs_per_band=1):
    bands = [(350, 500), (650, 850)]
    eligible = [[], []]
    count = 0
    for row in read_rows(source):
        tokens = row["sequence_tokens"]
        n = len(tokens)
        if n != row["scored_tokens"] or any(
            token["position"] != i or any(
                not np.isfinite(token["scores"][key]) for key, _, _ in METRICS
            ) for i, token in enumerate(tokens)
        ):
            raise ValueError(f"Invalid token scores for {row['sample_id']}")
        count += 1
        if row["stop_reason"] != "eos" or row["parsed_answer"] is None:
            continue
        for bucket, (lo, hi) in zip(eligible, bands):
            if lo <= n <= hi:
                bucket.append({key: row[key] for key in
                               ("sample_id", "scored_tokens", "level", "subject", "is_correct")})
    rng = random.Random(seed)
    pairs = []
    for bucket, band in zip(eligible, bands):
        candidates = [
            (a["sample_id"], b["sample_id"])
            for a in bucket if a["is_correct"]
            for b in bucket if not b["is_correct"]
            and (a["subject"], a["level"]) == (b["subject"], b["level"])
            and abs(a["scored_tokens"] - b["scored_tokens"]) <= 20
        ]
        candidates.sort()
        for _ in range(pairs_per_band):
            if not candidates:
                raise ValueError(f"Not enough disjoint matched pairs for token band {band}")
            pair = rng.choice(candidates)
            pairs.append(pair)
            candidates = [candidate for candidate in candidates
                          if candidate[0] != pair[0] and candidate[1] != pair[1]]
    selected = set().union(*map(set, pairs))
    rows = {row["sample_id"]: row for row in read_rows(source) if row["sample_id"] in selected}
    return pairs, rows, count, bands


def token_series(row):
    tokens = row["sequence_tokens"]
    x = [token["position"] + 1 for token in tokens]
    return x, {
        key: np.array([token["scores"][key] for token in tokens])
        for key, _, _ in METRICS
    }


def draw_pair(pair, destination, number):
    fig, axes = plt.subplots(3, 2, figsize=(12, 7.6), sharex=True, sharey="row")
    fig.subplots_adjust(top=.82, bottom=.12, left=.09, right=.98, hspace=.34, wspace=.09)
    fig.suptitle(f"MATH token confidence | matched pair {number}", fontsize=18, y=.975)
    legend_handles = [
        Line2D([], [], color="#737C84", lw=.65, label="Raw score"),
        Line2D([], [], color="#253544", lw=1.4, label="EMA (span 5)"),
        Line2D([], [], color="#253544", lw=1, ls="--", label="Sequence mean"),
        Line2D([], [], color="#253544", ls="", marker="v", markersize=5,
               label="Endpoint of a selected drop"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(.5, .925),
               ncol=4, frameon=False)
    for col, row in enumerate(pair):
        x, series = token_series(row)
        label = "Correct" if row["is_correct"] else "Incorrect"
        axes[0, col].set_title(f"{label} | {row['scored_tokens']} tokens\n{row['sample_id']}",
                               fontsize=10, loc="left")
        for i, (key, name, color) in enumerate(METRICS):
            y = series[key]
            smoothed = y.copy()
            alpha = 2 / (SPAN + 1)
            for t in range(1, len(y)):
                smoothed[t] = alpha * y[t] + (1 - alpha) * smoothed[t - 1]
            risk, drops = evidence_drop(y, span=SPAN, worst=WORST)
            ax = axes[i, col]
            ax.plot(x, y, lw=.65, color="#737C84", alpha=.65)
            ax.plot(x, smoothed, lw=1.25, color=color)
            ax.axhline(y.mean(), lw=.9, ls="--", color="#253544")
            ax.scatter(np.array(x)[drops], smoothed[drops], marker="v", s=25,
                       facecolor="#253544", edgecolor="white", linewidth=.45, zorder=5)
            ax.text(.99, .96, f"Mean {y.mean():.4f} | Drop risk {risk:.4f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88})
            ax.grid(axis="y", alpha=.23)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(name.replace(" Shannon", "\nShannon"), fontsize=9)
    axes[0, 0].set_xlim(1, max(row["scored_tokens"] for row in pair))
    fig.supxlabel("Generated token position (1-based)", y=.062, fontsize=10)
    fig.text(.5, .027, "Raw units. The two panels in each row share the same axes. "
             "Markers show score drops, without step-level error labels.",
             ha="center", fontsize=8, color="#465562")
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)
    with Image.open(destination) as im:
        im.verify()
    print(destination)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--pairs-per-band", type=int, default=1)
    args = parser.parse_args()
    if args.pairs_per_band < 1:
        parser.error("--pairs-per-band must be positive")
    if args.output.exists():
        raise SystemExit(f"Choose a new output directory: {args.output}")
    pairs, rows, count, bands = choose_pairs(args.source, args.seed, args.pairs_per_band)
    print(f"Verified all three scores on {count} trajectories")
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    index = [
        "# MATH 逐 token 分数配对图",
        "",
        f"共 {len(pairs)} 对。抽样种子 {args.seed}，每个长度区间 {args.pairs_per_band} 对。",
        "每对科目与难度相同，两条轨迹长度相差不超过 20 token。每条轨迹至多使用一次。",
        "左列为答对，右列为答错。每行两侧共用坐标轴，各图纵轴范围随该组数据而定。",
        "灰线为原始分数，彩线为 EMA (span=5)，虚线为序列均值，倒三角标记最大五次下降的终点。",
        "",
        "[样例逐 token 分数和回答](sample-trajectories.json)",
        "",
        "| 图 | 长度区间 | 科目 | 难度 | 正确轨迹 | 错误轨迹 |",
        "|---|---:|---|---|---|---|",
    ]
    for i, ids in enumerate(pairs, start=1):
        pair = [rows[sample_id] for sample_id in ids]
        filename = f"pair-{i:02d}.png" if len(pairs) >= 10 else f"pair-{i}.png"
        draw_pair(pair, args.output / filename, i)
        lo, hi = bands[(i - 1) // args.pairs_per_band]
        correct, incorrect = pair
        index.append(
            f"| [{i:02d}]({filename}) | {lo}–{hi} | {correct['subject']} | {correct['level']} | "
            f"`{correct['sample_id']}` ({correct['scored_tokens']}) | "
            f"`{incorrect['sample_id']}` ({incorrect['scored_tokens']}) |"
        )
        for row in pair:
            x, series = token_series(row)
            record = {key: row[key] for key in ("sample_id", "subject", "level", "scored_tokens",
                       "question", "response", "parsed_answer", "gold_answer", "is_correct")}
            record.update(pair=i, position=x,
                          token_ids=[token["token_id"] for token in row["sequence_tokens"]],
                          scores={k: y.tolist() for k, y in series.items()},
                          means={k: float(y.mean()) for k, y in series.items()},
                          drops={k: {"risk": r, "positions": [t + 1 for t in drops]}
                                 for k, y in series.items()
                                 for r, drops in [evidence_drop(y, SPAN, WORST)]})
            records.append(record)
            print(record["sample_id"], record["scored_tokens"], record["is_correct"],
                  record["level"], record["means"],
                  {k: v["risk"] for k, v in record["drops"].items()})
    manifest = {"source": str(args.source), "validated_trajectories": count, "seed": args.seed,
                "length_bands": bands, "pairs_per_band": args.pairs_per_band,
                "selection": "Random disjoint same-subject and same-level pairs within 20 scored tokens. "
                "EOS and parsed answer required; scores unused for selection.",
                "span": SPAN, "worst": WORST, "trajectories": records}
    (args.output / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    (args.output / "sample-trajectories.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

