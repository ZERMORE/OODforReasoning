"""E4: reviewed semantic-step edits, fresh-cache continuation, paired behavior report."""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import digest, extract_answer
from .evaluation import read_jsonl
from .study_io import (file_hash, implementation_hash, independent_output, seal_record,
                       source_model, source_run, start_output, write_json)


def validate_events(events, rows):
    lookup = {r["sample_id"]: r for r in rows}
    if not events or len({e["event_id"] for e in events}) != len(events):
        raise ValueError("Need nonempty, unique intervention event IDs")
    for event in events:
        row = lookup.get(event["sample_id"])
        if row is None or event.get("reviewed") is not True or event.get("original_process_valid") is not True:
            raise ValueError("Every event must have reviewed, verified correct original reasoning")
        start, end = event["start"], event["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end < len(row["generated_token_ids"]):
            raise ValueError("Step span must be inside the generated response, before its end")
        if not row["is_correct"] or event.get("source_row_hash") != digest(row):
            raise ValueError("Event source changed or original final answer was incorrect")
        for key in ("error_step_text", "sham_step_text"):
            if not isinstance(event.get(key), str) or not event[key].strip():
                raise ValueError(f"Fill reviewed {key}; no automatic error invention")
        if event["error_step_text"] == event["sham_step_text"]:
            raise ValueError("Error and sham edits must differ")
    return lookup


def paired_behavior(rows, bootstrap=2000, seed=0):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["sample_id"], {}).setdefault(row["branch"], []).append(not row["final_correct"])
    if not grouped or any(set(value) != {"original", "error", "sham"} for value in grouped.values()):
        raise ValueError("Each question must contain all three branches")
    ids = sorted(grouped)
    values = np.array([[np.mean(grouped[key][branch]) for branch in ("original", "error", "sham")] for key in ids])
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), (bootstrap, len(values)))].mean(axis=1)
    result = {"n_questions": len(ids), "n_continuations": len(rows),
              "mean_question_error_rate": dict(zip(("original", "error", "sham"), values.mean(axis=0).tolist())),
              "unit": "question; all positions, branches and seeds kept together", "paired_deltas": {},
              "truncation_rate": {b: float(np.mean([r["stop_reason"] == "max_new_tokens" for r in rows if r["branch"] == b]))
                                  for b in ("original", "error", "sham")},
              "note": "Behavioral effect only. Propagation/recovery require reviewed mechanism labels; nonsignificance is not equivalence."}
    for name, a, b in (("error-original", 1, 0), ("error-sham", 1, 2), ("sham-original", 2, 0)):
        result["paired_deltas"][name] = {"delta": float((values[:, a] - values[:, b]).mean()),
                                        "ci95": np.quantile(samples[:, a] - samples[:, b], [.025, .975]).tolist()}
    return result


def continuation_curves(rows, output, bootstrap=2000):
    """Question-paired score differences; variable termination is explicitly counted."""
    from .visualize import save_figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curves = {}
    ranges = [(0, 16), (16, 32), (32, 64), (64, 128)]
    metrics = sorted(set.intersection(*(set(r["scores"]) for r in rows)))
    metrics = [key for key in metrics if key in {"nci", "fdbd_all", "logit_margin", "deepconf_top20"}]
    if not metrics:
        write_json(output / "confidence_curves.json", {"warning": "No common nonempty confidence metrics"})
        return
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3.4 * len(metrics)), squeeze=False)
    for ax, metric in zip(axes[:, 0], metrics):
        curves[metric] = []
        for start, end in ranges:
            grouped = {}
            for row in rows:
                trace = row["token_scores"]
                if len(trace) >= end:
                    value = np.mean([t["scores"][metric] for t in trace[start:end]])
                    key = (row["sample_id"], row["event_id"], row["seed"])
                    grouped.setdefault(key, {})[row["branch"]] = value
            record = {"start": start, "end": end, "branch_question_counts": {
                b: len({key[0] for key, value in grouped.items() if b in value}) for b in ("original", "error", "sham")}}
            for reference in ("original", "sham"):
                question_differences = {}
                for key, value in grouped.items():
                    if "error" in value and reference in value:
                        question_differences.setdefault(key[0], []).append(value["error"] - value[reference])
                differences = np.array([np.mean(value) for value in question_differences.values()])
                if len(differences):
                    rng = np.random.default_rng(0)
                    samples = differences[rng.integers(0, len(differences), (bootstrap, len(differences)))].mean(axis=1)
                    record[f"error-{reference}"] = {"n_questions": len(differences), "delta": float(differences.mean()),
                                                     "ci95": np.quantile(samples, [.025, .975]).tolist()}
                else:
                    record[f"error-{reference}"] = None
            curves[metric].append(record)
        for reference in ("original", "sham"):
            valid = [(i, r[f"error-{reference}"]) for i, r in enumerate(curves[metric]) if r[f"error-{reference}"]]
            if valid:
                ax.errorbar([i for i, r in valid], [r["delta"] for i, r in valid],
                            yerr=np.array([[r["delta"]-r["ci95"][0], r["ci95"][1]-r["delta"]] for i, r in valid]).T.clip(0),
                            marker="o", label=f"error - {reference}")
        ax.axhline(0, color="gray", lw=1)
        ax.set_xticks(range(4), ["1-16", "17-32", "33-64", "65-128"])
        ax.set(title=metric, xlabel="Generated continuation token positions", ylabel="Confidence difference")
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.tight_layout()
    save_figure(fig, output, "continuation_confidence")
    write_json(output / "confidence_curves.json", {"curves": curves,
        "warning": "Matched event/seed pairs with complete intervals only; EOS survival selection remains. Not a recovery or equivalence test."})


def run_branch(model, tokenizer, row, event, branch, seed, scorer, manifest, args):
    from .engine import generate
    start, end = event["start"], event["end"]
    edited = row["generated_token_ids"][:end] if branch == "original" else (
        row["generated_token_ids"][:start] + tokenizer.encode(event[f"{branch}_step_text"], add_special_tokens=False))
    stop = set(manifest["eos_token_ids"])
    if not edited or any(token in stop for token in edited):
        raise ValueError("Edited prefix is empty or contains EOS")
    if branch != "original" and edited == row["generated_token_ids"][:end]:
        raise ValueError("Edited branch is token-identical to original; review the edit")
    prompt = row["prompt_token_ids"] + edited
    context = getattr(model.config, "max_position_embeddings", None)
    if context and len(prompt) + args.max_new_tokens > context:
        raise ValueError("Intervention prefix plus continuation exceeds model context")
    # generate() always creates a NEW DynamicCache, including across branches.
    generated = generate(model, prompt, scorer, seed=seed, temperature=args.temperature, top_p=1.,
                         max_new_tokens=args.max_new_tokens, stop_ids=stop,
                         include_eos=manifest["config"]["include_eos"], save_tokens=True)
    full_ids = edited + generated["generated_token_ids"]
    full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
    parsed = extract_answer(full_text, manifest["config"]["task"])
    return seal_record({**generated, "sample_id": row["sample_id"], "event_id": event["event_id"],
                        "branch": branch, "seed": seed, "edited_prefix_ids": edited, "event": event,
                        "full_response": full_text, "gold_answer": row["gold_answer"], "parsed_answer": parsed,
                        "final_correct": parsed == row["gold_answer"], "prefix_length_delta": len(edited) - end,
                        "score_scope": "new continuation only; injected and common prefix tokens excluded",
                        "used_injected_error": None, "self_corrected": None, "mechanism_reviewed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="ORIGINAL completed single-task run")
    parser.add_argument("--events", required=True, help="Reviewed JSONL semantic-step edits")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.temperature <= 0 or args.max_new_tokens < 1 or args.bootstrap < 2:
        parser.error("Positive temperature, token budget and bootstrap >= 2 required")
    seeds = [int(s) for s in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds) or not seeds:
        parser.error("Seeds must be distinct and nonempty")
    source, output = independent_output(args.source, args.output)
    if Path(args.events).resolve() == output or output in Path(args.events).resolve().parents:
        parser.error("Keep reviewed events outside output")
    manifest, rows = source_run(source)
    events = read_jsonl(args.events)
    lookup = validate_events(events, rows)
    jobs = [(event, branch, seed) for event in events for seed in seeds for branch in ("original", "error", "sham")]
    settings = {"schema": 1, "kind": "interventions", "source": str(source), "model_override": args.model,
                "source_hash": file_hash(source / "predictions.jsonl"), "manifest_hash": file_hash(source / "manifest.json"),
                "events_hash": file_hash(args.events), "seeds": seeds, "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens, "implementation_hash": implementation_hash()}
    signature, previous, saved = start_output(output, settings, args.resume)
    if len(previous) > len(jobs) or any((r["event_id"], r["branch"], r["seed"]) != (e["event_id"], b, s)
                                      for r, (e, b, s) in zip(previous, jobs)):
        raise ValueError("Resume job order mismatch")
    if len(previous) < len(jobs):
        from .replay_scores import StudyScorer
        from .cli import parse_ks
        from tqdm import tqdm
        model, tokenizer, env = source_model(args, manifest, saved.get("environment") if saved else None)
        scorer = StudyScorer(model, tuple(manifest.get("resolved_ks", parse_ks(manifest["config"]["ks"]))))
        saved = {"signature": signature, "settings": settings, "source_manifest": manifest, "environment": env, "complete": False}
        write_json(output / "manifest.json", saved)
        with (output / "predictions.jsonl").open("a", encoding="utf-8") as stream:
            for event, branch, seed in tqdm(jobs[len(previous):], initial=len(previous), total=len(jobs), desc="E4 continuation"):
                record = run_branch(model, tokenizer, lookup[event["sample_id"]], event, branch, seed, scorer, manifest, args)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
    saved["complete"] = True
    write_json(output / "manifest.json", saved)
    results = read_jsonl(output / "predictions.jsonl")
    write_json(output / "behavior.json", paired_behavior(results, args.bootstrap))
    continuation_curves(results, output, args.bootstrap)
    print(f"E4 continuations and paired behavior report: {output}; mechanism labels still require human review")


if __name__ == "__main__":
    main()
