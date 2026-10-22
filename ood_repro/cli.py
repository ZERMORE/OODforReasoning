from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .data import SPECS, digest, extract_answer, load_examples
from .evaluation import evaluate, read_jsonl, select_k
from .prompting import build_prompt, load_pack


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def model_args(parser):
    parser.add_argument("--model", help="HF model ID or local checkpoint directory")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")


def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not args.model:
        raise ValueError("--model is required (or provide it in --config)")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; select --device cpu --dtype float32 for a CPU smoke test")
    torch.manual_seed(getattr(args, "seed", 0))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(getattr(args, "seed", 0))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.tokenizer_revision or args.model_revision,
                                             local_files_only=args.local_files_only, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, torch_dtype=getattr(torch, args.dtype),
        attn_implementation=args.attention, local_files_only=args.local_files_only,
        trust_remote_code=False, use_safetensors=True,
    ).to(args.device).eval()
    if model.config.model_type not in {"llama", "qwen2"}:
        raise ValueError("This runner validates only Llama and Qwen2 dense linear heads")
    return model, tokenizer


def parse_ks(text):
    if isinstance(text, list):
        return tuple("all" if str(x).lower() == "all" else int(x) for x in text)
    return tuple("all" if x.strip().lower() == "all" else int(x) for x in text.split(","))


def environment(model, tokenizer):
    import torch
    packages = {}
    for name in ["torch", "transformers", "datasets", "accelerate", "numpy", "scikit-learn", "safetensors"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    device = model.get_input_embeddings().weight.device
    source_root = Path(__file__).resolve().parent
    implementation_hash = digest({path.relative_to(source_root).as_posix(): digest(path.read_text(encoding="utf-8"))
                                  for path in sorted(source_root.rglob("*.py"))})
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "git_commit": commit, "model_commit": getattr(model.config, "_commit_hash", None),
            "implementation_hash": implementation_hash,
            "tokenizer_commit": tokenizer.init_kwargs.get("_commit_hash"),
            "chat_template_hash": digest(tokenizer.chat_template)}


def run(args):
    from tqdm import tqdm
    from .engine import check_head, eos_ids, generate, replay
    from .scoring import GeometryScorer
    if args.task not in SPECS:
        raise ValueError("--task is required")
    if args.split == "train" and args.limit is None:
        raise ValueError("Use --limit for an explicit training/calibration subset")
    pack = load_pack(args.task, args.prompt_file)
    demo_questions = [item["question"].split("\nAnswer Choices:")[0] for item in pack["examples"]] if args.split == "train" else []
    examples, dataset_meta = load_examples(args.task, args.split, args.data_jsonl,
                                           args.dataset_revision, args.limit, args.seed, args.split == "train", demo_questions)
    if not examples:
        raise ValueError("No examples")
    ks = parse_ks(args.ks)
    selection = None
    if args.k_selection:
        selection = json.loads(Path(args.k_selection).read_text(encoding="utf-8"))
        calibration = selection["calibration_manifest"]
        if calibration["config"]["task"] != args.task or calibration["config"]["model"] != args.model:
            raise ValueError("Selected k belongs to a different task/model")
        for key in ["model_revision", "tokenizer_revision", "prompt_mode", "dtype", "attention", "temperature", "top_p", "include_eos", "max_new_tokens"]:
            if calibration["config"][key] != getattr(args, key):
                raise ValueError(f"Calibration/evaluation protocol mismatch: {key}")
        if calibration["pack_hash"] != digest(pack):
            raise ValueError("Calibration/evaluation prompt packs differ")
        if set(calibration["dataset"]["sample_ids"]) & set(dataset_meta["sample_ids"]):
            raise ValueError("Calibration and evaluation sample IDs overlap")
        if set(calibration["dataset"]["question_hashes"]) & set(dataset_meta["question_hashes"]):
            raise ValueError("Calibration and evaluation questions overlap")
        ks = (selection["selected_k"],)
    config = {k: v for k, v in vars(args).items() if k not in {"resume", "output", "config", "bootstrap", "command"}}
    signature = digest({"config": config, "dataset": dataset_meta, "pack": pack, "ks": ks, "selection": selection})
    root = Path(args.output)
    manifest_path, prediction_path = root / "manifest.json", root / "predictions.jsonl"
    done = set()
    old_manifest = None
    if root.exists() and any(root.iterdir()):
        if not args.resume or not manifest_path.exists():
            raise ValueError("Output directory is nonempty. Use a new directory or --resume")
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_manifest["signature"] != signature:
            raise ValueError("Resume configuration/data/prompt mismatch")
        previous = read_jsonl(prediction_path) if prediction_path.exists() else []
        done = {r["sample_id"] for r in previous}
        if len(done) != len(previous) or not done <= set(dataset_meta["sample_ids"]):
            raise ValueError("Invalid/duplicate previous predictions")
    root.mkdir(parents=True, exist_ok=True)
    if len(done) == len(examples):
        write_json(root / "metrics.json", evaluate(read_jsonl(prediction_path), args.bootstrap, args.seed))
        print(f"Already complete: {root}")
        return
    model, tokenizer = load_model(args)
    env = environment(model, tokenizer)
    if old_manifest:
        for key in ["implementation_hash", "model_commit", "tokenizer_commit", "chat_template_hash", "packages"]:
            if old_manifest["environment"][key] != env[key]:
                raise ValueError(f"Resume environment mismatch: {key}")
    head = model.get_output_embeddings()
    scorer = GeometryScorer(head.weight, getattr(head, "bias", None), ks, args.vocab_chunk)
    stop = eos_ids(model, tokenizer)
    first_text, first_ids, _ = build_prompt(tokenizer, examples[0], args.task, pack, args.prompt_mode)
    head_error = check_head(model, first_ids)
    manifest = {"created": datetime.now(timezone.utc).astimezone().isoformat(), "signature": signature,
                "config": config, "resolved_ks": ks, "dataset": dataset_meta, "pack_hash": digest(pack),
                "prompt_provenance": {k: v for k, v in pack.items() if k != "examples"},
                "environment": env, "head_reconstruction_max_abs_error": head_error,
                "eos_token_ids": sorted(stop), "k_selection": selection,
                "protocol": "independent implementation; full-output confidence; parse failures counted incorrect"}
    if not old_manifest:
        write_json(manifest_path, manifest)
        (root / "prompt-example.txt").write_text(first_text, encoding="utf-8")
    verified = 0
    with prediction_path.open("a", encoding="utf-8") as stream:
        for index, example in enumerate(tqdm(examples, desc=args.task)):
            if example["sample_id"] in done:
                continue
            text, ids, prompt_meta = build_prompt(tokenizer, example, args.task, pack, args.prompt_mode)
            context_limit = getattr(model.config, "max_position_embeddings", None)
            if context_limit and len(ids) + args.max_new_tokens > context_limit:
                raise ValueError(f"Prompt + generation exceeds context for {example['sample_id']}")
            generated = generate(model, ids, scorer if args.score_mode == "online" else None,
                                 max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                 top_p=args.top_p, seed=args.seed + index, stop_ids=stop,
                                 include_eos=args.include_eos, save_tokens=args.save_token_scores)
            verification = None
            if args.score_mode == "replay" or verified < args.verify_replay:
                replayed = replay(model, ids, generated["generated_token_ids"], scorer,
                                  chunk_size=args.replay_chunk, stop_ids=stop,
                                  include_eos=args.include_eos, save_tokens=args.save_token_scores)
                if args.score_mode == "online":
                    import math
                    if generated["scored_tokens"] != replayed["scored_tokens"]:
                        raise ValueError("Online/replay token counts differ")
                    differences = {k: abs(v - replayed["scores"][k]) for k, v in generated["scores"].items()}
                    if any(not math.isclose(v, replayed["scores"][k], rel_tol=args.verify_rtol, abs_tol=args.verify_atol)
                           for k, v in generated["scores"].items()):
                        raise ValueError(f"Online/replay scores differ: {differences}")
                    verification = differences
                    verified += 1
                else:
                    generated.update(replayed)
            response = tokenizer.decode(generated["generated_token_ids"], skip_special_tokens=True)
            parsed = extract_answer(response, args.task)
            row = {**example, **prompt_meta, **generated, "prompt_token_ids": ids,
                   "response": response, "parsed_answer": parsed, "is_correct": parsed == example["gold_answer"],
                   "sample_seed": args.seed + index, "online_replay_abs_diff": verification,
                   "timing_scope": "generation with scoring" if args.score_mode == "online" else "generation only; excludes replay"}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
    report = evaluate(read_jsonl(prediction_path), args.bootstrap, args.seed)
    write_json(root / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def benchmark(args):
    """Paired forced-token online benchmark; no offline replay counted as online."""
    import statistics
    from .engine import generate
    from .scoring import GeometryScorer
    if args.tokens < 1 or args.repeats < 1 or args.warmup < 1:
        raise ValueError("Benchmark tokens, repeats and warmup must be positive")
    target = Path(args.output)
    if target.exists():
        raise ValueError("Benchmark output exists; choose a new path")
    model, tokenizer = load_model(args)
    ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    reference = generate(model, ids, max_new_tokens=args.tokens, force_length=True)["generated_token_ids"]
    head = model.get_output_embeddings()
    scorers = {"standard": None,
               "nci": GeometryScorer(head.weight, getattr(head, "bias", None), ks=(), metrics=("nci",)),
               "fdbd": GeometryScorer(head.weight, getattr(head, "bias", None), ks=(args.k,), metrics=())}
    timings = {k: [] for k in scorers}
    for _ in range(args.warmup):
        for scorer in scorers.values():
            generate(model, ids, scorer, max_new_tokens=args.tokens, force_length=True, forced_tokens=reference)
    # Rotate execution order to reduce systematic thermal/order bias.
    names = list(scorers)
    for trial in range(args.repeats):
        order = names[trial % len(names):] + names[:trial % len(names)]
        for name in order:
            result = generate(model, ids, scorers[name], max_new_tokens=args.tokens,
                              force_length=True, forced_tokens=reference)
            timings[name].append(result["ms_per_generated_token"])
    medians = {k: statistics.median(v) for k, v in timings.items()}
    result = {"environment": environment(model, tokenizer), "config": vars(args), "raw_ms_per_token": timings,
              "median_ms_per_token": medians, "overhead_fraction": {k: v / medians["standard"] - 1 for k, v in medians.items()},
              "prompt_ids": ids, "generated_ids": reference,
              "scope": "synchronized end-to-end prefill+decode; batch=1; same forced trajectory; includes validation syncs; not paper A100 protocol"}
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, result)
    print(json.dumps(medians, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Independent geometric hallucination-detection reproduction")
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="Generate, score, label, and evaluate")
    model_args(runner)
    runner.add_argument("--config", type=str)
    runner.add_argument("--task", choices=list(SPECS))
    runner.add_argument("--split")
    runner.add_argument("--data-jsonl", help="Local raw task rows (same schema as HF dataset)")
    runner.add_argument("--dataset-revision")
    runner.add_argument("--limit", type=int)
    runner.add_argument("--prompt-file")
    runner.add_argument("--prompt-mode", choices=["chat", "plain"], default="chat")
    runner.add_argument("--output", default="runs/experiment")
    runner.add_argument("--resume", action="store_true")
    runner.add_argument("--ks", default="1000")
    runner.add_argument("--k-selection")
    runner.add_argument("--vocab-chunk", type=int, default=4096)
    runner.add_argument("--score-mode", choices=["online", "replay"], default="online")
    runner.add_argument("--max-new-tokens", type=int, default=512)
    runner.add_argument("--temperature", type=float, default=0.0)
    runner.add_argument("--top-p", type=float, default=1.0)
    runner.add_argument("--seed", type=int, default=0)
    runner.add_argument("--include-eos", action="store_true")
    runner.add_argument("--save-token-scores", action="store_true")
    runner.add_argument("--replay-chunk", type=int, default=16)
    runner.add_argument("--verify-replay", type=int, default=0)
    runner.add_argument("--verify-rtol", type=float, default=0.01)
    runner.add_argument("--verify-atol", type=float, default=0.001)
    runner.add_argument("--bootstrap", type=int, default=2000)
    evaluator = commands.add_parser("evaluate", help="Re-evaluate saved predictions; no model required")
    evaluator.add_argument("predictions")
    evaluator.add_argument("--output", required=True)
    evaluator.add_argument("--bootstrap", type=int, default=2000)
    evaluator.add_argument("--seed", type=int, default=0)
    selector = commands.add_parser("select-k", help="Choose k exclusively from a completed train run")
    selector.add_argument("run_dir")
    selector.add_argument("--output", required=True)
    bench = commands.add_parser("benchmark", help="Same-token online latency comparison")
    model_args(bench)
    bench.add_argument("--prompt", default="Q: What is 17 multiplied by 23? Explain.\nA:")
    bench.add_argument("--tokens", type=int, default=128)
    bench.add_argument("--k", type=int, default=1000)
    bench.add_argument("--repeats", type=int, default=5)
    bench.add_argument("--warmup", type=int, default=2)
    bench.add_argument("--output", default="runs/benchmark.json")
    args = parser.parse_args()
    if args.command == "run" and args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        allowed = set(vars(args)) - {"command", "config"}
        if set(config) - allowed:
            parser.error(f"Unknown configuration keys: {set(config) - allowed}")
        runner.set_defaults(**config)
        args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "evaluate":
        write_json(args.output, evaluate(read_jsonl(args.predictions), args.bootstrap, args.seed))
    elif args.command == "select-k":
        print(json.dumps(select_k(args.run_dir, args.output), ensure_ascii=False, indent=2))
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
