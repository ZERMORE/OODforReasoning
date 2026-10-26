"""Auditable, append-only experiment files shared by sequence-study commands."""
import hashlib
import json
from argparse import Namespace
from pathlib import Path

from .data import digest
from .evaluation import read_jsonl


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def independent_output(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Use an output directory separate from the source and its ancestors")
    return source, output


def source_run(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    rows = read_jsonl(path / "predictions.jsonl")
    ids = [r["sample_id"] for r in rows]
    if not rows or len(set(ids)) != len(ids) or set(ids) != set(manifest["dataset"]["sample_ids"]):
        raise ValueError("Source run is empty, incomplete or has duplicate sample IDs")
    return manifest, rows


def start_output(output, settings, resume=False):
    output = Path(output)
    signature = digest(settings)
    previous, saved = [], None
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise ValueError("Output nonempty; choose a new directory or --resume")
        saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if saved["signature"] != signature:
            raise ValueError("Resume source, implementation or settings mismatch")
        if (output / "predictions.jsonl").exists():
            previous = read_jsonl(output / "predictions.jsonl")
        for row in previous:
            check_record(row)
    output.mkdir(parents=True, exist_ok=True)
    return signature, previous, saved


def seal_record(row):
    return {**row, "record_hash": digest(row)}


def check_record(row):
    if row.get("record_hash") != digest({k: v for k, v in row.items() if k != "record_hash"}):
        raise ValueError("Record integrity check failed")


def source_model(args, manifest, previous_env=None):
    from .cli import environment, load_model
    config = dict(manifest["config"])
    config.update(model=args.model or config["model"], device=args.device, local_files_only=True)
    config.setdefault("seed", 0)
    model, tokenizer = load_model(Namespace(**config))
    env = environment(model, tokenizer)
    for key in ("model_commit", "tokenizer_commit", "chat_template_hash", "packages"):
        if manifest["environment"].get(key) != env.get(key):
            raise ValueError(f"Source/replay environment mismatch: {key}")
    if previous_env is not None and previous_env != env:
        raise ValueError("Resume environment mismatch")
    return model, tokenizer, env


def implementation_hash():
    root = Path(__file__).parent
    names = ("engine.py", "scoring.py", "deepconf.py", "study_io.py", "replay_scores.py", "interventions.py")
    return digest({name: file_hash(root / name) for name in names if (root / name).exists()})
