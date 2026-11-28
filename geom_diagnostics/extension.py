"""Benchmark preparation and shared-pass G1 scoring."""
import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from ood_repro.data import digest, question_key
from ood_repro.study_io import check_record, file_hash, independent_output, seal_record
from .run import read, verified_rows, write


FILES = {
    "supergpqa": ("m-a-p/SuperGPQA", {"train": "SuperGPQA-all.jsonl"}, 26529),
    "processbench": ("Qwen/ProcessBench", {k: k+".json" for k in
                     ("gsm8k", "math", "olympiadbench", "omnimath")}, 3400),
}
PROBABILITY = ("mean_logprob", "logit_margin", "shannon_top20")
GEOMETRY = ("angle_boundary", "fisher_normal")
WINDOW_SIGNALS = ("fisher_radial", "fisher_normal", "nci", "fdbd_all", "shannon_top20")
SIGNALS = tuple(dict.fromkeys(WINDOW_SIGNALS + PROBABILITY + GEOMETRY))
WIDTHS = (8, 16, 32, 64, 128, 256, "10pct")


def normalize(row, subset, task):
    if task == "supergpqa":
        options = row["options"]
        gold = row["answer_letter"].strip().upper()
        if len(gold) != 1 or gold not in "ABCDEFGHIJ"[:len(options)]:
            raise ValueError(f"Invalid answer letter: {row['uuid']}")
        return dict(id=row["uuid"], question=row["question"], options=options, gold=gold,
                    discipline=row["discipline"], difficulty=row["difficulty"],
                    is_calculation=row["is_calculation"])
    steps, label = row["steps"], row["label"]
    if not steps or any(not s.strip() for s in steps) or not -1 <= label < len(steps):
        raise ValueError(f"Invalid ProcessBench annotation: {row['id']}")
    return dict(id=f"{subset}/{row['id']}", question=row["problem"], steps=steps,
                first_error=label, final_correct=row["final_answer_correct"],
                subset=subset, generator=row["generator"])


def prepare(args):
    import json
    from huggingface_hub import HfApi, hf_hub_download

    repo, files, expected = FILES[args.task]
    dest = args.data / args.task
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dest / "manifest.json"
    if manifest.exists():
        saved = read(manifest)
        if args.revision in ("main", saved["revision"]) and file_hash(dest / "data.json") == saved["sha256"]:
            print(f"Prepared dataset already exists: {dest} ({saved['revision']})", flush=True)
            return
        raise ValueError("Prepared data or revision mismatch. Use a separate data directory.")
    revision = HfApi().dataset_info(repo, revision=args.revision).sha
    rows = []
    for subset, name in files.items():
        path = Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision))
        if name.endswith(".jsonl"):
            with path.open(encoding="utf-8") as handle:
                data = [json.loads(line) for line in handle if line.strip()]
        else:
            data = read(path)
        rows.extend(normalize(row, subset, args.task) for row in data)
    if len(rows) != expected or len({r["id"] for r in rows}) != expected:
        raise ValueError(f"Expected {expected} unique official rows, got {len(rows)} rows")
    write(dest / "data.json", rows)
    write(manifest, dict(repo=repo, revision=revision, rows=len(rows), files=files,
                         sha256=file_hash(dest / "data.json")))
    print(f"{args.task}: pinned {revision}, {len(rows)} rows -> {dest}", flush=True)


def stratified(rows, count, seed, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    count = min(count, len(rows))
    quotas = {k: len(v) * count / len(rows) for k, v in groups.items()}
    taken = {k: int(q) for k, q in quotas.items()}
    order = sorted(groups, key=lambda k: (-(quotas[k] - taken[k]), str(k)))
    for k in order[:count - sum(taken.values())]:
        taken[k] += 1
    chosen = [r for k, v in groups.items()
              for r in sorted(v, key=lambda r: digest([seed, r["id"]]))[:taken[k]]]
    return sorted(chosen, key=lambda r: digest([seed+1, r["id"]]))


def process_tokens(tokenizer, steps):
    ranges, offset = [], 0
    for step in steps:
        ranges.append((offset, offset + len(step)))
        offset += len(step) + 2
    text = "\n\n".join(steps)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    tokens = encoded["input_ids"]
    if tokenizer.decode(tokens, clean_up_tokenization_spaces=False) != text:
        raise ValueError("Tokenization changed the published solution")
    mapping, step = [], 0
    for start, end in encoded["offset_mapping"]:
        # If a token spans two steps, only the later step may use that token.
        while step + 1 < len(ranges) and end > ranges[step+1][0]:
            step += 1
        lo, hi = ranges[step]
        mapping.append(step if max(lo, start) < min(hi, end) else None)
    if set(mapping) - {None} != set(range(len(steps))):
        raise ValueError("Token alignment lost a step")
    return tokens, mapping


def score_new(args):
    from ood_repro.cli import environment, load_model
    from ood_repro.engine import eos_ids, generate, replay
    from tqdm import tqdm
    from .scorer import DiagnosticScorer
    from .extension_analysis import parse_option

    datadir, out = independent_output(args.data / args.task, args.output)
    dataset = read(datadir / "manifest.json")
    if file_hash(datadir / "data.json") != dataset["sha256"]:
        raise ValueError("Prepared dataset checksum mismatch")
    rows = read(datadir / "data.json")
    process = args.task == "processbench"
    rows = stratified(rows, args.limit or (len(rows) if process else 1200), args.seed,
                      ("subset",) if process else ("discipline", "difficulty", "is_calculation"))
    checkpoint = Path(args.model).resolve()
    root = Path(__file__).resolve().parents[1]
    paths = (Path(__file__), Path(__file__).with_name("scorer.py"),
             Path(__file__).with_name("extension_analysis.py"),
             root/"ood_repro/engine.py", root/"ood_repro/scoring.py", root/"ood_repro/cli.py")
    plan = dict(task=args.task, dataset=dataset, ids=[r["id"] for r in rows], seed=args.seed,
                max_new_tokens=args.max_new_tokens if not process else None,
                checkpoint=str(checkpoint), dtype=args.dtype, attention=args.attention,
                revision=args.model_revision, tokenizer_revision=args.tokenizer_revision,
                checkpoint_config=file_hash(checkpoint/"config.json"),
                code={p.name: file_hash(p) for p in paths})
    manifest = dict(signature=digest(plan), settings=plan, count=len(rows), complete=False)
    manifest_path = out/"manifest.json"
    saved = read(manifest_path) if manifest_path.exists() else None
    if saved:
        if saved["signature"] != manifest["signature"]:
            raise ValueError("Source, checkpoint or settings changed. Use a new output directory.")
        if saved["complete"]:
            for _ in verified_rows(out, saved):
                pass
            print(f"Already complete: {out}", flush=True)
            return
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Nonempty output lacks a manifest")
    model, tokenizer = load_model(args)
    env = environment(model, tokenizer)
    for key in ("packages", "cuda", "gpu", "chat_template_hash"):
        if saved and env[key] != saved["environment"][key]:
            raise ValueError(f"Resume environment differs: {key}")
    manifest["environment"] = env
    out.mkdir(parents=True, exist_ok=True)
    write(manifest_path, manifest)
    (out/"rows").mkdir(exist_ok=True)
    stops = eos_ids(model, tokenizer)
    scorer = DiagnosticScorer(model, seed=args.seed)
    for row in tqdm(rows, desc=args.task):
        dest = out/"rows"/(digest(row["id"])+".json")
        if dest.exists():
            previous = read(dest)
            check_record(previous)
            if previous["source_hash"] != digest(row):
                raise ValueError("Saved row differs from prepared data")
            continue
        start = perf_counter()
        question = row["question"]
        text = question
        if not process:
            options = "\n".join(f"({chr(65+i)}) {v}" for i, v in enumerate(row["options"]))
            text += f"\n\n{options}\n\nReason step by step. End with a separate line: Final answer: X, where X is one option letter."
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                               tokenize=True, add_generation_prompt=True)
        scorer.begin_trajectory()
        group = question_key(question)
        record = dict(id=row["id"], source_hash=digest(row), question_hash=group)
        if process:
            tokens, mapping = process_tokens(tokenizer, row["steps"])
            if set(tokens) & set(tokenizer.all_special_ids) or len(prompt)+len(tokens) > model.config.max_position_embeddings:
                raise ValueError(f"Unsupported special token or context length: {row['id']}")
            result = replay(model, prompt, tokens, scorer, include_eos=True, save_tokens=True)
            trace = result["token_scores"]
            if [t["token_id"] for t in trace] != tokens:
                raise ValueError(f"Replay alignment changed: {row['id']}")
            for token, step in zip(trace, mapping, strict=True):
                token["step"] = step
            split = "dev" if int(digest([args.seed, group])[:12], 16) % 5 == 0 else "test"
            record.update(split=split, first_error=row["first_error"], n_steps=len(row["steps"]),
                          subset=row["subset"], generator=row["generator"], error=not row["final_correct"])
        else:
            if len(prompt)+args.max_new_tokens > model.config.max_position_embeddings:
                raise ValueError(f"Generation budget exceeds the context: {row['id']}")
            result = generate(model, prompt, scorer, stop_ids=stops, save_tokens=True,
                              max_new_tokens=args.max_new_tokens, seed=args.seed)
            tokens = result["generated_token_ids"]
            trace = result["token_scores"]
            response = tokenizer.decode(tokens, skip_special_tokens=True)
            answer = parse_option(response, len(row["options"]))
            record.update(split="test", discipline=row["discipline"], difficulty=row["difficulty"],
                          is_calculation=row["is_calculation"], gold=row["gold"], answer=answer,
                          error=answer != row["gold"], response=response, stop_reason=result["stop_reason"])
        record.update(token_scores=trace, prompt_tokens=len(prompt), generated_tokens=len(tokens),
                      elapsed_seconds=perf_counter()-start)
        if not trace:
            record["excluded"] = "no scored tokens"
        write(dest, seal_record(record))
    for _ in verified_rows(out, manifest):
        pass
    write(manifest_path, dict(manifest, complete=True))
    print(f"Complete: {out}", flush=True)


def main():
    from ood_repro.cli import model_args
    from .extension_analysis import analyze, freeze

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("task", choices=FILES)
    prep.add_argument("--data", type=Path, required=True)
    prep.add_argument("--revision", default="main")
    prep.set_defaults(action=prepare)
    score = sub.add_parser("score")
    score.add_argument("task", choices=FILES)
    score.add_argument("--data", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--limit", type=int, default=0)
    score.add_argument("--max-new-tokens", type=int, default=1024)
    score.add_argument("--seed", type=int, default=42)
    model_args(score)
    score.set_defaults(action=score_new, local_files_only=True)
    select = sub.add_parser("freeze")
    select.add_argument("--source-g1", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.set_defaults(action=freeze)
    report = sub.add_parser("analyze")
    report.add_argument("task", choices=("csqa", "aqua", *FILES))
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--policy", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--bootstrap", type=int, default=1000)
    report.set_defaults(action=analyze)
    args = parser.parse_args()
    if args.phase == "score" and (not args.model or args.limit < 0 or args.max_new_tokens < 1):
        parser.error("score requires --model, a nonnegative limit and a positive token budget")
    if args.phase == "analyze" and args.bootstrap < 0:
        parser.error("--bootstrap cannot be negative")
    args.action(args)


if __name__ == "__main__":
    main()
