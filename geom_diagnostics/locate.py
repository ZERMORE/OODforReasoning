"""Optional G2 offline audit on existing first-error token annotations."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ood_repro.evidence import evidence_drop
from ood_repro.study_io import check_record, file_hash
from .run import read, verified_rows, write


def locate(root, annotations, analysis, dest, width):
    manifest = read(root / "manifest.json")
    protocol = read(root / analysis / "protocol.json")
    if not manifest["complete"] or manifest["signature"] != protocol["signature"]:
        raise ValueError("Use a completed G1 replay and its analysis")
    reviews = {}
    with annotations.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                review = json.loads(line)
                check_record(review)
                key = review["sample_id"]
                if key in reviews:
                    raise ValueError(f"Duplicate annotation: {key}")
                reviews[key] = review
    rows = []
    matched = 0
    for row in verified_rows(root, manifest):
        review = reviews.get(row["id"])
        if row["split"] != "test" or "excluded" in row or not review:
            continue
        if review["source_row_hash"] != row["source_hash"]:
            raise ValueError(f"Annotation source mismatch: {row['id']}")
        onset = review.get("first_error_position")
        if not row["error"] or onset is None or review.get("reviewed") is not True:
            continue
        trace = row["token_scores"]
        positions = np.array([t["position"] for t in trace])
        before = (positions >= onset - width) & (positions < onset)
        after = (positions > onset) & (positions <= onset + width)
        if not before.any() or not after.any():
            continue
        matched += 1
        top2 = [(t["winner_id"], t["competitor_id"]) for t in trace]
        switch = np.array([False, *[a != b for a,b in zip(top2, top2[1:])]])
        for signal in trace[0]["scores"]:
            direction = protocol["policies"][signal]["confidence_multiplier"]
            confidence = direction * np.array([t["scores"][signal] for t in trace])
            _, endpoints = evidence_drop(confidence)
            nearest = min((int(abs(positions[i]-onset)) for i in endpoints), default=None)
            rows.append(dict(id=row["id"], signal=signal, onset=onset,
                             human_reviewed=review.get("human_reviewed", False),
                             before_n=int(before.sum()), after_n=int(after.sum()),
                             before_switch_rate=float(switch[before].mean()), after_switch_rate=float(switch[after].mean()),
                             risk_increase=float(confidence[before].mean()-confidence[after].mean()),
                             nearest_drop_distance=nearest, hit_within_width=nearest is not None and nearest <= width))
    if not rows:
        raise ValueError("No matching located test errors with both pre/post windows")
    dest.mkdir(parents=True, exist_ok=False)
    with (dest/"per-question.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for signal in sorted({r["signal"] for r in rows}):
        group = [r for r in rows if r["signal"]==signal]
        summary[signal] = dict(n=len(group), hit_rate=float(np.mean([r["hit_within_width"] for r in group])),
                               mean_risk_increase=float(np.mean([r["risk_increase"] for r in group])))
    write(dest/"summary.json", dict(width=width, matched_errors=matched, summary=summary,
          replay_signature=manifest["signature"], annotations_sha256=file_hash(annotations),
          note="Offline pre/post diagnostic on located errors. No causal interpretation or correct-trajectory false-positive estimate."))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--analysis", default="analysis-v1")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=16)
    args = p.parse_args()
    if args.width < 1:
        p.error("--width must be positive")
    locate(args.replay,args.annotations,args.analysis,args.output,args.width)
