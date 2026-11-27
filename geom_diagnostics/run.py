"""Reproducible G1 replay and CPU-only aggregation."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ood_repro.data import digest
from ood_repro.study_io import check_record, file_hash, independent_output, seal_record


BASELINES = ("nci", "fdbd_all", "mean_logprob", "logit_margin", "negative_entropy", "shannon_top20")
FIXED_CONFIDENCE = set(BASELINES) | {"distance_top2"}
VIEWS = ("mean", "lowest_w64", "evidence_drop")
SCRIPT_FILES = ("engine.py", "scoring.py", "evidence.py", "cli.py")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def question_hash(question):
    return digest(" ".join(question.split()))


def split_of(qhash):
    index = int(hashlib.sha256(f"G1-v1:{qhash}".encode()).hexdigest()[:8], 16) % 10
    return "fit" if index < 2 else "calibration" if index < 4 else "test"


def load_source(path, kind):
    rows = []
    if kind == "ood":
        manifest = read(path / "manifest.json")
        config = manifest["config"]
        with (path / "predictions.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                entry = dict(id=row["sample_id"], question=row["question"], correct=row["is_correct"],
                             prompt=row["prompt_token_ids"], tokens=row["generated_token_ids"],
                             old=row["scores"], trace=row.get("token_scores"), source_hash=digest(row))
                expected = positions_for(entry, set(manifest["eos_token_ids"]), config["include_eos"])
                trace = cached_trace(entry, expected)
                entry["cached_fdbd"] = [t["scores"]["fdbd_all"] for t in trace] if trace else None
                entry.pop("trace")
                rows.append(entry)
        if {r["id"] for r in rows} != set(manifest["dataset"]["sample_ids"]):
            raise ValueError("Legacy OOD generation is incomplete")
        policy = dict(model=config["model"], dtype=config["dtype"], attention=config["attention"],
                      revision=config.get("model_revision", "main"), tokenizer_revision=config.get("tokenizer_revision"),
                      include_eos=config["include_eos"], stop_ids=manifest["eos_token_ids"],
                      model_hash=None, source_environment=manifest["environment"],
                      atol=config.get("verify_atol", .001), rtol=config.get("verify_rtol", .01))
    else:
        from step_ood.study import source_records
        policy = None
        for source_split in ("train", "validation", "test"):
            _, manifest, source = source_records(path, "gsm8k", source_split)
            config = manifest["config"]
            audit = read(path / "gsm8k" / source_split / "extraction.json")
            if audit["source_generation"] != manifest:
                raise ValueError("Step extraction source mismatch")
            kept = set(audit["included"])
            current = dict(model=config["model"], dtype="bfloat16", attention="sdpa",
                           revision="main", tokenizer_revision=None, include_eos=False,
                           stop_ids=None, model_hash=config["model_config"],
                           source_environment=None, atol=.001, rtol=.01)
            if policy is not None and policy != current:
                raise ValueError("Step checkpoint or precision differs across splits")
            policy = current
            for _, row in source:
                if row["id"] not in kept:
                    continue
                qhash = question_hash(row["question"])
                split = ("reference" if source_split == "train" else "test" if source_split == "test"
                         else "fit" if int(qhash[:8], 16) % 2 == 0 else "calibration")
                rows.append(dict(id=f"{source_split}/{row['id']}", question=row["question"],
                                 correct=row["is_correct"], prompt=row["input_ids"], split=split,
                                 tokens=row["full_seq_ids"][len(row["input_ids"]):],
                                 old={}, cached_fdbd=None, origin=row, source_hash=digest(row)))
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Empty source or duplicate IDs")
    partitions = {}
    for row in rows:
        qhash = question_hash(row["question"])
        row["split"] = row.get("split", split_of(qhash))
        if partitions.setdefault(qhash, row["split"]) != row["split"]:
            raise ValueError("Same question occurs in different partitions")
        if not row["prompt"] or not row["tokens"] or not isinstance(row["correct"], bool):
            raise ValueError(f"Invalid source record {row['id']}")
    return policy, rows


def positions_for(row, stops, include_eos):
    return [i for i, token in enumerate(row["tokens"]) if include_eos or token not in stops]


def cached_trace(row, expected):
    trace = row["trace"] or []
    if (len(trace) != len(expected) or any(
        t["position"] != i or t["token_id"] != row["tokens"][i] or
        not all(k in t["scores"] and np.isfinite(t["scores"][k]) for k in BASELINES if k != "shannon_top20")
        for t, i in zip(trace, expected)
    )):
        return None
    return trace


def step_map(tokenizer, row):
    from step_ood.study import content_spans
    from step_ood.token_alignment import original_token_offsets

    text, offsets, _ = original_token_offsets(tokenizer, row["tokens"])
    if text != row["origin"]["decoded_with_specials"]:
        raise ValueError("Step text does not match saved token IDs")
    spans = content_spans(text)
    return {i: j for j, (start, end) in enumerate(spans, 1)
            for i, (a, b) in enumerate(offsets) if start <= a < b <= end}


def fit_reference(model, tokenizer, rows, stops, scorer, path):
    import torch
    from ood_repro.engine import replay

    if path.is_file():
        return {int(k): torch.tensor(v, device=scorer.weight.device) for k,v in read(path).items()}
    sums, counts = {}, Counter()

    class Reference:
        position = 0

        def score(self, state, logits, token):
            index = current_mapping.get(current_positions[self.position], 0)
            self.position += 1
            vector = torch.nn.functional.normalize(state.float() - scorer.baselines.center, dim=0)
            for key in {0, index}:
                sums[key] = sums[key] + vector if key in sums else vector.clone()
                counts[key] += 1
            return {"placeholder": vector[0]}

    from tqdm import tqdm
    ref = Reference()
    for row in tqdm(rows, desc="unlabeled step reference"):
        current_mapping = step_map(tokenizer, row)
        current_positions = positions_for(row, stops, include_eos=False)
        ref.position = 0
        replay(model, row["prompt"], row["tokens"], ref, stop_ids=stops, mode="matched")
    directions = {k: torch.nn.functional.normalize(v, dim=0) for k,v in sums.items()
                  if counts[k] >= 10 or k == 0}
    if 0 not in directions:
        raise ValueError("Empty reference")
    write(path, {k: v.tolist() for k,v in directions.items()})
    return directions


def replay_job(args):
    from ood_repro.cli import environment, load_model
    from ood_repro.engine import eos_ids, replay
    from tqdm import tqdm
    from .scorer import DiagnosticScorer

    source, out = independent_output(args.source, args.output)
    policy, rows = load_source(source, args.kind)
    if args.limit:
        rows = [r for split in ("reference", "fit", "calibration", "test")
                for r in [r for r in rows if r["split"] == split][:args.limit]]
    model_path = Path(args.model or policy["model"]).resolve()
    if model_path.name != Path(policy["model"]).name:
        raise ValueError("Use the original checkpoint snapshot")
    checkpoint_hash = file_hash(model_path / "config.json")
    if policy["model_hash"] and checkpoint_hash != policy["model_hash"]:
        raise ValueError("Model config differs from Step source")
    root = Path(__file__).resolve().parents[1]
    code_files = [Path(__file__), Path(__file__).with_name("scorer.py"),
                  root / "step_ood/study.py", root / "step_ood/token_alignment.py",
                  *[root / "ood_repro" / k for k in SCRIPT_FILES]]
    settings = dict(source=str(source), kind=args.kind, model=str(model_path), checkpoint=checkpoint_hash,
                    policy=policy, limit=args.limit, seed=args.seed, device=args.device,
                    code={str(p.relative_to(root)):file_hash(p) for p in code_files},
                    rows=digest([r["source_hash"] for r in rows]))
    manifest_path = out / "manifest.json"
    manifest = {"signature": digest(settings), "settings": settings, "complete": False,
                "count": sum(r["split"] != "reference" for r in rows), "split_counts":dict(Counter(r["split"] for r in rows))}
    if manifest_path.exists():
        saved = read(manifest_path)
        if saved["signature"] != manifest["signature"]:
            raise ValueError("Resume settings changed. Use a new output directory.")
        if saved["complete"]:
            for _ in verified_rows(out, saved):
                pass
            print(f"Already complete: {out}", flush=True)
            return
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Nonempty output has no G1 manifest")
    (out / "rows").mkdir(parents=True, exist_ok=True)
    args.model = str(model_path)
    args.dtype, args.attention = policy["dtype"], policy["attention"]
    args.model_revision, args.tokenizer_revision = policy["revision"], policy["tokenizer_revision"]
    args.local_files_only = True
    model, tokenizer = load_model(args)
    env = environment(model, tokenizer)
    for key in ("model_commit", "tokenizer_commit", "chat_template_hash"):
        if policy["source_environment"] and env.get(key) != policy["source_environment"].get(key):
            raise ValueError(f"Source checkpoint/tokenizer mismatch: {key}")
    manifest["environment"] = env
    if manifest_path.exists():
        previous_env = read(manifest_path)["environment"]
        for key in ("packages", "gpu", "cuda", "model_commit", "tokenizer_commit", "chat_template_hash"):
            if env.get(key) != previous_env.get(key):
                raise ValueError(f"Resume environment mismatch: {key}")
    write(manifest_path, manifest)
    stops = eos_ids(model, tokenizer) if policy["stop_ids"] is None else set(policy["stop_ids"])
    scorer = DiagnosticScorer(model, cached_baselines=True, seed=args.seed)
    fit = [r for r in rows if r["split"] == "reference"]
    if fit:
        scorer.step_directions = fit_reference(model, tokenizer, fit, stops, scorer, out/"step-reference.json")
    for row in tqdm([r for r in rows if r["split"] != "reference"], desc="G1 replay"):
        dest = out / "rows" / (digest(row["id"]) + ".json")
        if dest.exists():
            cached = read(dest)
            check_record(cached)
            if cached["source_hash"] != row["source_hash"]:
                raise ValueError("Cached row source changed")
            continue
        expected = positions_for(row, stops, policy["include_eos"])
        historical = row["cached_fdbd"]
        scorer.baselines.ks = () if historical else ("all",)
        record = dict(id=row["id"], split=row["split"], question_hash=question_hash(row["question"]),
                      error=not row["correct"], source_hash=row["source_hash"])
        if not expected:
            write(dest, seal_record(dict(record, excluded="no scored tokens")))
            continue
        token_steps = step_map(tokenizer, row) if args.kind == "step" else {}
        scorer.begin_trajectory(expected, token_steps)
        result = replay(model, row["prompt"], row["tokens"], scorer, stop_ids=stops,
                        include_eos=policy["include_eos"], save_tokens=True, mode="matched")
        trace = result["token_scores"]
        if [t["position"] for t in trace] != expected or any(row["tokens"][t["position"]] != t["token_id"] for t in trace):
            raise ValueError("Replay token alignment changed")
        if args.kind == "step" and not any(i in scorer.token_steps for i in expected):
            raise ValueError("Step body contains no scored tokens")
        audit = {}
        if historical:
            # The token scores came from this exact saved prompt and continuation.
            for key in ("nci", "mean_logprob", "logit_margin", "negative_entropy", "shannon_top20"):
                if key in row["old"]:
                    new, old = result["scores"][key], row["old"][key]
                    audit[key] = abs(new-old)
                    if not np.isclose(new, old, atol=policy["atol"], rtol=policy["rtol"]):
                        raise ValueError(f"{row['id']}: replay audit failed for {key}: {new} vs {old}")
            old_mean = row["old"].get("fdbd_all")
            if old_mean is not None and not np.isclose(np.mean(historical), old_mean, atol=policy["atol"], rtol=policy["rtol"]):
                raise ValueError(f"{row['id']}: cached fDBD mean audit failed")
            for current, old in zip(trace, historical):
                current["scores"]["fdbd_all"] = old
            audit["fdbd_all"] = "reused exact source token scores"
        else:
            for key in BASELINES:
                if key in row["old"]:
                    new, old = result["scores"][key], row["old"][key]
                    audit[key] = abs(new-old)
                    if not np.isclose(new, old, atol=policy["atol"], rtol=policy["rtol"]):
                        raise ValueError(f"{row['id']}: replay audit failed for {key}: {new} vs {old}")
        scorer.attach_metadata(trace)
        write(dest, seal_record(dict(record, token_scores=trace, score_audit=audit)))
    for _ in verified_rows(out, manifest):
        pass
    write(manifest_path, dict(manifest, complete=True))
    print(f"G1 REPLAY COMPLETE: {out}", flush=True)


def verified_rows(root, manifest):
    paths = sorted((root/"rows").glob("*.json"))
    if len(paths) != manifest["count"]:
        raise ValueError("Incomplete G1 replay")
    seen = set()
    for p in paths:
        row = read(p)
        check_record(row)
        if row["id"] in seen or p.stem != digest(row["id"]):
            raise ValueError("Duplicate or inconsistent row")
        seen.add(row["id"])
        yield row


def pool(trace, key, direction):
    from ood_repro.evidence import evidence_drop

    confidence = direction * np.array([t["scores"][key] for t in trace], dtype=float)
    width = min(64, len(confidence))
    cumulative = np.r_[0., confidence.cumsum()]
    drop, _ = evidence_drop(confidence)
    output = dict(mean=float(confidence.mean()),
                  lowest_w64=float(((cumulative[width:] - cumulative[:-width])/width).min()),
                  evidence_drop=-drop)
    steps = defaultdict(list)
    for t, value in zip(trace, confidence):
        if t["step"] is not None:
            steps[t["step"]].append(value)
    if steps:
        means = [np.mean(x) for x in steps.values()]
        output.update(reasoning_token_mean=float(np.mean([v for x in steps.values() for v in x])),
                      step_mean=float(np.mean(means)), worst_step=float(min(means)))
        if key == "angle_step_ref":
            output = {k: v for k, v in output.items() if k in ("reasoning_token_mean", "step_mean", "worst_step")}
    return output


def analyze(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from ood_repro.evidence import risk_curve
    from step_ood.study import lower_threshold

    manifest = read(args.output/"manifest.json")
    if not manifest["complete"]:
        raise ValueError("Replay is not complete")
    audit, rows = Counter(), []
    for row in verified_rows(args.output, manifest):
        if "excluded" in row:
            audit[row["split"]] += 1
            continue
        trace = row.pop("token_scores")
        signals = sorted(trace[0]["scores"])
        row["pooled"] = {key:{s:pool(trace,key,s) for s in ([1] if key in FIXED_CONFIDENCE else [1,-1])}
                         for key in signals}
        rows.append(row)
    if not rows:
        raise ValueError("No scored trajectories")
    y = np.array([r["error"] for r in rows], dtype=int)
    group = np.array([r["split"] for r in rows])
    fit, cal, test = [group == s for s in ("fit", "calibration", "test")]
    if any(len(np.unique(y[m])) != 2 for m in (fit, cal, test)):
        raise ValueError("Each G1 partition needs both classes. Increase the subset for analysis.")
    q_test = np.array([r["question_hash"] for r in rows])[test]
    blocks = [np.flatnonzero(q_test == q) for q in sorted(set(q_test))]
    rng = np.random.default_rng(args.seed)
    draws = [np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))])
             for _ in range(args.bootstrap)]
    draws = [ix for ix in draws if len(np.unique(y[test][ix])) == 2]
    signals = sorted(rows[0]["pooled"])
    scores, policies = {}, {}
    for key in signals:
        directions = [1] if key in FIXED_CONFIDENCE else [1,-1]
        alternatives = {s:[r["pooled"][key][s] for r in rows] for s in directions}
        primary = "reasoning_token_mean" if key == "angle_step_ref" else "mean"
        direction = max(directions, key=lambda s: roc_auc_score(
            y[fit], -np.asarray([x[primary] for x in alternatives[s]])[fit]))
        policies[key] = {"confidence_multiplier":direction,"selected_on":"fixed" if len(directions)==1 else "fit"}
        for view in alternatives[direction][0]:
            scores[f"{key}/{view}"] = np.array([x[view] for x in alternatives[direction]])
    def combine(name, keys):
        x = np.stack([scores[k] for k in keys], axis=1)
        model = make_pipeline(StandardScaler(),LogisticRegression(C=1., max_iter=500))
        model.fit(x[fit], y[fit])
        scores[name] = -model.decision_function(x)
        policies[name] = dict(features=keys, mean=model[0].mean_.tolist(), scale=model[0].scale_.tolist(),
                              coefficients=model[1].coef_[0].tolist(), intercept=float(model[1].intercept_[0]))
    best = max((f"{k}/mean" for k in ("angle_boundary","fisher_normal","fisher_radial")),
               key=lambda key: roc_auc_score(y[fit], -scores[key][fit]))
    combine("angle_plus_fisher/mean", ["angle_boundary/mean","fisher_normal/mean"])
    probability = ["mean_logprob/mean","logit_margin/mean","shannon_top20/mean"]
    combine("probability_model/mean",probability)
    combine("probability_plus_angle_fisher/mean",probability+["angle_boundary/mean","fisher_normal/mean"])

    def metrics(labels, confidence):
        coverage, risk = risk_curve(labels, confidence)
        return np.array([roc_auc_score(labels, -confidence), average_precision_score(labels, -confidence),
                         np.diff(np.r_[0.,coverage]) @ risk])

    labels = y[test]
    estimates = {key:metrics(labels, v[test]) for key,v in scores.items()}
    samples = {key:np.asarray([metrics(labels[ix], v[test][ix]) for ix in draws]) for key,v in scores.items()}
    report = []
    for key,v in scores.items():
        correct_cal = v[cal & (y == 0)]
        threshold = lower_threshold(correct_cal, args.fpr)
        alarm = v[test] < threshold if threshold is not None else np.zeros(len(labels), dtype=bool)
        ref = ("probability_model/mean" if key == "probability_plus_angle_fisher/mean" else best
               if key == "angle_plus_fisher/mean" else key.split("/")[0]+"/mean")
        if ref not in scores:
            ref = key.split("/")[0]+"/reasoning_token_mean"
        item = dict(method=key, reference=ref, threshold=threshold,
                    calibration_fpr=float((correct_cal < threshold).mean()) if threshold is not None else 0.,
                    test_fpr=float(alarm[labels==0].mean()), test_tpr=float(alarm[labels==1].mean()))
        for j,m in enumerate(("auroc","ap","aurc")):
            item[m] = float(estimates[key][j])
            item[m+"_delta"] = float(estimates[key][j]-estimates[ref][j])
            if draws:
                for suffix, vals in [("_ci95", samples[key][:,j]),
                                     ("_delta_ci95", (samples[key]-samples[ref])[:,j])]:
                    item[m+suffix] = np.quantile(vals,[.025,.975]).tolist()
        report.append(item)
    dest = args.output / args.tag
    if dest.exists():
        raise ValueError("Analysis tag exists. Choose a new --tag.")
    temporary = dest.with_name(dest.name + ".tmp")
    temporary.mkdir(exist_ok=True)
    final = dest
    dest = temporary
    write(dest/"metrics.json", report)
    write(dest/"protocol.json", dict(signature=manifest["signature"], counts=dict(Counter(group.tolist())),
          excluded=dict(audit), test_errors=int(labels.sum()), test_n=len(labels),
          bootstrap_requested=args.bootstrap, bootstrap_accepted=len(draws), seed=args.seed,
          bootstrap_unit="paired test question clusters; direction, fusion and threshold fixed",
          fpr_target=args.fpr, policies=policies, best_single_on_fit=best,
          note="Exploratory reuse of previously studied trajectories; AP uses final-answer error. Lower AURC is better."))
    with (dest/"metrics.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    with (dest/"scores.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id","question_hash","split","error",*scores])
        for i,r in enumerate(rows):
            writer.writerow([r["id"],r["question_hash"],r["split"],int(y[i]),*[v[i] for v in scores.values()]])
    dest.replace(final)
    print(f"G1 ANALYSIS COMPLETE: {final}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    gpu = sub.add_parser("replay")
    gpu.add_argument("--source", type=Path, required=True)
    gpu.add_argument("--kind", choices=["ood","step"], required=True)
    gpu.add_argument("--model")
    gpu.add_argument("--device", default="cuda")
    gpu.add_argument("--limit", type=int, default=0, help="Limit per partition; 0 means full source")
    cpu = sub.add_parser("analyze")
    cpu.add_argument("--tag", default="analysis-v1")
    cpu.add_argument("--bootstrap", type=int, default=1000)
    cpu.add_argument("--fpr", type=float, default=.1)
    for p in (gpu,cpu):
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.phase == "replay":
        if args.limit < 0:
            parser.error("--limit must be nonnegative")
        replay_job(args)
    else:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag) or not 0<args.fpr<1 or args.bootstrap<0:
            parser.error("Invalid analysis tag, FPR or bootstrap count")
        analyze(args)


if __name__ == "__main__":
    main()
