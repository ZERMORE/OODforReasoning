"""E1/E2 prerequisite: matched replay of existing responses, auditing ALL old scores."""
import argparse

import numpy as np

from .data import digest
from .study_io import (file_hash, implementation_hash, independent_output, seal_record,
                       source_model, source_run, start_output, write_json)


class StudyScorer:
    def __init__(self, model, ks):
        from .scoring import GeometryScorer
        from .deepconf import TopKConfidence
        head = model.get_output_embeddings()
        self.geometry = GeometryScorer(head.weight, getattr(head, "bias", None), ks=ks)
        self.confidence = TopKConfidence(20)

    def score(self, state, logits, token):
        scores = self.geometry.score(state, logits, token)
        scores["deepconf_top20"] = self.confidence.score(state, logits, token)["deepconf_mean"]
        return scores


def score_row(model, row, scorer, manifest):
    from .engine import replay
    result = replay(model, row["prompt_token_ids"], row["generated_token_ids"], scorer,
                    mode="matched", stop_ids=set(manifest["eos_token_ids"]),
                    include_eos=manifest["config"]["include_eos"], save_tokens=True)
    if result["scored_tokens"] != row["scored_tokens"] or not result["scored_tokens"]:
        raise ValueError(f"Scored token count mismatch/empty: {row['sample_id']}")
    config = manifest["config"]
    audit = {}
    for key, old in row["scores"].items():
        if key not in result["scores"]:
            raise ValueError(f"Unsupported source score {key}; use the ORIGINAL main run")
        new = result["scores"][key]
        audit[key] = {"old": old, "replayed": new, "abs_diff": abs(new - old)}
        if not np.isfinite(new) or not np.isclose(new, old, rtol=config.get("verify_rtol", .01),
                                                atol=config.get("verify_atol", .001)):
            raise ValueError(f"Replay score mismatch: {row['sample_id']}/{key}: {audit[key]}")
    return seal_record({**row, "source_row_hash": digest(row), "sequence_tokens": result["token_scores"],
                        "sequence_scores": result["scores"], "score_audit": audit})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", help="Same checkpoint at a new local path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, help="Smoke only; use a separate output directory")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source, output = independent_output(args.source, args.output)
    manifest, rows = source_run(source)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("limit must be positive")
        rows = rows[:args.limit]
    settings = {"schema": 1, "kind": "sequence-replay", "source": str(source),
                "source_manifest_sha256": file_hash(source / "manifest.json"),
                "source_predictions_sha256": file_hash(source / "predictions.jsonl"),
                "implementation_hash": implementation_hash(), "model_override": args.model,
                "limit": args.limit, "mode": "matched", "top_k": 20, "count": len(rows)}
    signature, previous, saved = start_output(output, settings, args.resume)
    if len(previous) > len(rows):
        raise ValueError("Resume has extra rows")
    for old, row in zip(previous, rows):
        original = {k: v for k, v in old.items() if k not in {
            "record_hash", "source_row_hash", "sequence_tokens", "sequence_scores", "score_audit", "token_pieces"}}
        if old["source_row_hash"] != digest(row) or original != row:
            raise ValueError("Resume rows do not match source prefix")
    if len(previous) < len(rows):
        from .cli import parse_ks
        from tqdm import tqdm
        model, tokenizer, env = source_model(args, manifest, saved.get("environment") if saved else None)
        ks = tuple(manifest.get("resolved_ks", parse_ks(manifest["config"]["ks"])))
        scorer = StudyScorer(model, ks)
        saved = {"signature": signature, "settings": settings, "source_manifest": manifest,
                 "environment": env, "complete": False, "dataset": {"sample_ids": [r["sample_id"] for r in rows]}}
        write_json(output / "manifest.json", saved)
        import json
        with (output / "predictions.jsonl").open("a", encoding="utf-8") as stream:
            for row in tqdm(rows[len(previous):], initial=len(previous), total=len(rows), desc="sequence replay"):
                record = score_row(model, row, scorer, manifest)
                if tokenizer is not None:
                    record.pop("record_hash")
                    record["token_pieces"] = tokenizer.convert_ids_to_tokens(row["generated_token_ids"])
                    record = seal_record(record)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
    saved["complete"] = True
    write_json(output / "manifest.json", saved)
    print(f"Complete: {len(rows)} audited trajectories -> {output}")


if __name__ == "__main__":
    main()
