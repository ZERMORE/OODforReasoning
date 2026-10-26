import json
from argparse import Namespace

import numpy as np
import pytest
import torch

from ood_repro.aggregation import aggregate, permutation_indices
from ood_repro.data import digest
from ood_repro.engine import generate
from ood_repro.error_positions import analyze_positions, validate_annotations
from ood_repro.interventions import paired_behavior, run_branch, validate_events
from ood_repro.replay_scores import StudyScorer, score_row
from ood_repro.sequence_analysis import load_replay, metrics_matrix, order_controls, summarize
from ood_repro.study_io import check_record, seal_record, write_json


def test_aggregation_math_short_traces_steps_and_order_invariants():
    x = np.arange(1., 13.)
    values = aggregate(x, widths=(2, 64), steps=[(0, 2), (2, 12)])
    assert values["mean"] == 6.5
    assert values["token_bottom10"] == 1.5
    assert values["lowest_w2"] == 1.5
    assert values["bottom10_w2"] == 2.
    assert values["lowest_w64"] == values["mean"]
    assert values["step_equal_mean"] == 4.5
    reversed_values = aggregate(x[::-1], widths=(2, 64))
    for key in ("mean", "min", "token_bottom10", "lowest_w2", "bottom10_w2"):
        assert values[key] == pytest.approx(reversed_values[key])
    assert values["tail_weighted"] > reversed_values["tail_weighted"]
    with pytest.raises(ValueError):
        aggregate(x, steps=[(0, 3), (4, 12)])
    with pytest.raises(ValueError):
        aggregate([float("nan")])
    for mode in ("shuffle", "phase", "block", "reverse"):
        idx = permutation_indices(len(x), np.random.default_rng(0), mode)
        assert sorted(idx) == list(range(len(x)))


def test_order_diagnostic_distinguishes_clustering_from_same_score_multiset():
    rows = []
    for i in range(8):
        correct = bool(i % 2)
        values = np.tile([0., 1., 1., 1.], 64) if correct else np.r_[np.zeros(64), np.ones(192)]
        values = np.roll(values, i * 3)
        rows.append({"sample_id": str(i), "is_correct": correct,
                     "sequence_tokens": [{"scores": {"fdbd_all": float(x)}} for x in values]})
    scores = metrics_matrix(rows, (64,))
    assert np.all(scores["fdbd_all/mean"] == .75)
    assert np.all(scores["fdbd_all/token_bottom10"] == 0.)
    result = order_controls(rows, "fdbd_all", repeats=20, bootstrap=30)
    assert result["shuffle"]["original_auroc"] == 1.
    assert result["shuffle"]["permuted_mean"] < .8
    assert result["reverse"]["delta"] == 0.


@pytest.fixture
def tiny_source(tmp_path):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from ood_repro.scoring import GeometryScorer
    torch.manual_seed(42)
    config = Qwen2Config(vocab_size=31, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=256)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval()
    rows = []
    for i in range(4):
        generation = generate(model, [1, 2, 3], GeometryScorer(model.get_output_embeddings().weight, ks=(2, "all")),
                              forced_tokens=[4 + i, 8, 9, 10, 30], max_new_tokens=5, stop_ids={30})
        rows.append({**generation, "sample_id": f"q{i}", "prompt_token_ids": [1, 2, 3], "question": f"question {i}",
                     "response": "The answer is 42", "gold_answer": "42", "parsed_answer": "42",
                     "is_correct": bool(i % 2)})
    manifest = {"config": {"model": "local", "model_revision": "pinned", "tokenizer_revision": "pinned",
                           "task": "gsm8k", "ks": "2,all", "dtype": "float32", "attention": "eager",
                           "include_eos": False, "seed": 0}, "resolved_ks": [2, "all"], "eos_token_ids": [30],
                "dataset": {"sample_ids": [r["sample_id"] for r in rows], "split": "test"}, "environment": {}}
    source = tmp_path / "source"
    source.mkdir()
    write_json(source / "manifest.json", manifest)
    (source / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return model, manifest, rows, source


def test_all_metric_replay_audits_original_scores_and_rejects_mismatch(tiny_source):
    model, manifest, rows, _ = tiny_source
    scorer = StudyScorer(model, (2, "all"))
    record = score_row(model, rows[0], scorer, manifest)
    check_record(record)
    assert record["scores"] == rows[0]["scores"]
    assert len(record["sequence_tokens"]) == 4
    assert "deepconf_top20" in record["sequence_scores"]
    assert set(record["score_audit"]) == set(rows[0]["scores"])
    bad = {**rows[0], "scores": {**rows[0]["scores"], "nci": 100}}
    with pytest.raises(ValueError, match="Replay score mismatch"):
        score_row(model, bad, scorer, manifest)


def test_replay_cli_complete_resume_integrity_and_analysis_cli(tiny_source, tmp_path, monkeypatch):
    import sys
    import ood_repro.replay_scores as module
    import ood_repro.sequence_analysis as analysis
    model, manifest, rows, source = tiny_source
    output, report = tmp_path / "replay", tmp_path / "analysis"
    monkeypatch.setattr(module, "source_model", lambda *a: (model, None, {}))
    argv = ["replay", "--source", str(source), "--output", str(output), "--resume"]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    before = (output / "predictions.jsonl").read_bytes()
    monkeypatch.setattr(module, "source_model", lambda *a: pytest.fail("Complete resume must not load model"))
    module.main()
    assert before == (output / "predictions.jsonl").read_bytes()
    saved_manifest, records = load_replay(output)
    assert saved_manifest["complete"]
    result = summarize(records, metrics_matrix(records, (64,)), bootstrap=20)
    assert len(result) > 10 and all(r["delta_vs_mean"] == 0 for r in result if r["aggregation"] == "mean")
    # Short traces have identical mean and one-window scores, for every order.
    controls = order_controls(records, "fdbd_all", repeats=3, bootstrap=20)
    assert controls["reverse"]["delta"] == pytest.approx(0)
    monkeypatch.setattr(sys, "argv", ["analysis", "--replay", str(output), "--output", str(report),
                                      "--bootstrap", "20", "--permutations", "3"])
    analysis.main()
    assert (report / "aggregation.csv").exists() and (report / "order_controls.png").exists()
    corrupted = dict(records[0])
    corrupted["sequence_tokens"][0]["scores"]["nci"] += 1
    with pytest.raises(ValueError, match="integrity"):
        check_record(corrupted)


def test_position_buckets_have_shared_correct_controls_and_reject_unreviewed(tiny_source):
    model, manifest, rows, _ = tiny_source
    records = [score_row(model, row, StudyScorer(model, (2, "all")), manifest) for row in rows]
    annotations = [{"sample_id": row["sample_id"], "reviewed": True, "process_valid": row["is_correct"],
                    "source_row_hash": record["source_row_hash"],
                    "first_error_position": None if row["is_correct"] else 1} for row, record in zip(rows, records)]
    lookup = validate_annotations(annotations, records)
    report = analyze_positions(records, records, lookup, "fdbd_all", "mean", bootstrap=20)
    assert all(r["n_controls"] == 2 for r in report["buckets"])
    assert sum(r["n_errors"] for r in report["buckets"]) == 2
    annotations[0]["reviewed"] = False
    with pytest.raises(ValueError, match="reviewed"):
        validate_annotations(annotations, records)


def test_intervention_fresh_cache_prefix_exclusion_and_paired_question_weights(tiny_source):
    model, manifest, rows, _ = tiny_source
    row = rows[1]
    event = {"event_id": "e1", "sample_id": row["sample_id"], "source_row_hash": digest(row),
             "reviewed": True, "original_process_valid": True, "start": 1, "end": 3,
             "error_step_text": "wrong", "sham_step_text": "equivalent"}
    validate_events([event], rows)
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [11, 12] if text == "wrong" else [13, 14]
        def decode(self, ids, skip_special_tokens=True):
            return "The answer is 42"
    seen_caches = []
    handle = model.register_forward_pre_hook(lambda m, a, kw: seen_caches.append(kw["past_key_values"]), with_kwargs=True)
    args = Namespace(max_new_tokens=3, temperature=.7)
    records = [run_branch(model, Tokenizer(), row, event, branch, 0, StudyScorer(model, (2, "all")), manifest, args)
               for branch in ("original", "error", "sham")]
    handle.remove()
    assert len({id(cache) for cache in seen_caches}) == 3
    assert all(len(r["token_scores"]) <= 3 and r["score_scope"].startswith("new continuation") for r in records)
    assert records[0]["edited_prefix_ids"] == row["generated_token_ids"][:3]
    assert records[1]["edited_prefix_ids"] == row["generated_token_ids"][:1] + [11, 12]
    synthetic = [{"sample_id": "a", "branch": b, "final_correct": b != "error", "stop_reason": "eos"}
                 for b in ("original", "error", "sham")]
    synthetic += [{"sample_id": "b", "branch": b, "final_correct": True, "stop_reason": "eos"}
                  for b in ("original", "error", "sham") for _ in range(10)]
    result = paired_behavior(synthetic, 20)
    assert result["paired_deltas"]["error-original"]["delta"] == .5  # Equal questions, not 1/11.
    event["reviewed"] = False
    with pytest.raises(ValueError):
        validate_events([event], rows)


def test_intervention_cli_resume_and_review_gate(tiny_source, tmp_path, monkeypatch):
    import sys
    import ood_repro.interventions as module
    model, manifest, rows, source = tiny_source
    row = rows[1]
    event = {"event_id": "e1", "sample_id": row["sample_id"], "source_row_hash": digest(row),
             "reviewed": True, "original_process_valid": True, "start": 1, "end": 3,
             "error_step_text": "wrong", "sham_step_text": "equivalent"}
    events, output = tmp_path / "events.jsonl", tmp_path / "interventions"
    events.write_text(json.dumps(event) + "\n", encoding="utf-8")
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [11, 12] if text == "wrong" else [13, 14]
        def decode(self, ids, skip_special_tokens=True):
            return "The answer is 42"
    monkeypatch.setattr(module, "source_model", lambda *a: (model, Tokenizer(), {}))
    argv = ["interventions", "--source", str(source), "--events", str(events), "--output", str(output),
            "--seeds", "0", "--max-new-tokens", "3", "--bootstrap", "20", "--resume"]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    saved = (output / "predictions.jsonl").read_bytes()
    assert json.loads((output / "behavior.json").read_text())["n_continuations"] == 3
    monkeypatch.setattr(module, "source_model", lambda *a: pytest.fail("Completed continuation should not reload"))
    module.main()
    assert (output / "predictions.jsonl").read_bytes() == saved
    event["reviewed"] = False
    events.write_text(json.dumps(event) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed"):
        module.main()


def test_prepare_interventions_selects_unique_reviewed_steps(tiny_source):
    from ood_repro.prepare_interventions import select_steps
    model, manifest, rows, _ = tiny_source
    records = [score_row(model, row, StudyScorer(model, (2, "all")), manifest) for row in rows]
    annotations = {r["sample_id"]: {"process_valid": True, "steps": [(0, 1), (1, 2), (2, 3), (3, 4)]} for r in rows}
    events, candidates = select_steps(records, annotations, count=1)
    assert candidates == 2 and len(events) == 2
    assert len({(e["start"], e["end"]) for e in events}) == 2
    assert all(e["reviewed"] is False and e["error_step_text"] is None for e in events)


def test_calibration_config_inherits_model_and_requires_train_limit(tiny_source, tmp_path, monkeypatch):
    import sys
    import ood_repro.calibration_config as module
    _, manifest, _, source = tiny_source
    output = tmp_path / "config"
    monkeypatch.setattr(sys, "argv", ["config", "--source", str(source), "--output", str(output),
                                      "--run-output", str(tmp_path / "raw"), "--limit", "200"])
    module.main()
    config = json.loads((output / "config.json").read_text())
    assert config["split"] == "train" and config["limit"] == 200 and config["bootstrap"] == 0
    assert config["model_revision"] == manifest["config"]["model_revision"]


def test_confidence_curves_pair_matching_events_and_seeds(tmp_path):
    from ood_repro.interventions import continuation_curves
    rows = []
    for branch, value in (("original", 2.), ("error", 1.), ("sham", 3.)):
        rows.append({"sample_id": "q", "event_id": "e", "seed": 0, "branch": branch,
                     "scores": {"nci": value}, "token_scores": [{"scores": {"nci": value}}] * 32})
    # Unmatched error seed must not contaminate the question-paired difference.
    rows.append({"sample_id": "q", "event_id": "e", "seed": 1, "branch": "error",
                 "scores": {"nci": 100.}, "token_scores": [{"scores": {"nci": 100.}}] * 32})
    continuation_curves(rows, tmp_path, bootstrap=20)
    report = json.loads((tmp_path / "confidence_curves.json").read_text())
    assert report["curves"]["nci"][0]["error-original"]["delta"] == -1.
    assert report["curves"]["nci"][2]["error-original"] is None
    assert (tmp_path / "continuation_confidence.png").exists()
