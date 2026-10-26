import json
import sys

import numpy as np
import pytest
import torch

from ood_repro.data import digest
from ood_repro.eval2_data import build_split, load_study, pooling
from ood_repro.eval2_fit import batch, normalization, summarize
from ood_repro.eval2_models import SequenceClassifier
from ood_repro.eval2_positions import analyze, match_controls
from ood_repro.study_io import file_hash, seal_record, write_json


@pytest.fixture
def dataset(tmp_path):
    def make(name, split, count):
        path = tmp_path / name
        path.mkdir()
        rows = []
        rng = np.random.default_rng(0)
        for i in range(count):
            correct = bool(i % 2)
            values = rng.normal(1. if correct else 0., .5, size=8 + i % 7)
            tokens = [{"position": j, "token_id": 1, "scores": {"fdbd_all": float(v), "nci": float(v+.1)}} for j, v in enumerate(values)]
            scores = {key: float(np.mean([t["scores"][key] for t in tokens])) for key in tokens[0]["scores"]}
            source = {"sample_id": f"{name}:{i}", "question": f"{name} question {i}", "is_correct": correct,
                      "generated_token_ids": [1]*len(tokens), "scored_tokens": len(tokens), "response": "answer", "gold_answer": "1",
                      "parsed_answer": "1", "stop_reason": "eos", "scores": scores}
            rows.append(seal_record({**source, "source_row_hash": digest(source), "sequence_tokens": tokens, "sequence_scores": scores}))
        manifest = {"complete": True, "dataset": {"sample_ids": [r["sample_id"] for r in rows]},
                    "source_manifest": {"config": {"task": "gsm8k", "model_revision": "pinned", "include_eos": False},
                                        "environment": {}, "dataset": {"split": split}, "eos_token_ids": [30]}}
        write_json(path / "manifest.json", manifest)
        (path / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return path, rows
    pool, pool_rows = make("pool", "train", 60)
    test, test_rows = make("test", "test", 8)
    split = tmp_path / "splits.json"
    write_json(split, {"ids": build_split(pool_rows, test_rows), "pool_sha256": file_hash(pool / "predictions.jsonl"),
                       "test_sha256": file_hash(test / "predictions.jsonl")})
    return pool, test, split, pool_rows, test_rows


def test_split_groups_and_rejects_content_overlap(dataset):
    pool, test, split, rows, test_rows = dataset
    partitions, _, _ = load_study(pool, test, split)
    assert sum(len(r) for r in partitions.values()) == 60
    altered = [dict(r) for r in rows]
    altered[0]["question"] = test_rows[0]["question"]
    with pytest.raises(ValueError, match="content overlaps"):
        build_split(altered, test_rows)


def test_quantile_is_not_bottom_tail_mean():
    values = pooling(np.arange(20.))
    assert values["quantile10"] == pytest.approx(1.9)
    assert values["token_bottom10"] == .5


@pytest.mark.parametrize("kind", ["deepsets", "gru", "cnn", "attention", "attention_nopos"])
def test_models_ignore_padding_and_set_models_ignore_order(kind):
    torch.manual_seed(2)
    model = SequenceClassifier(kind, hidden=4).eval()
    arrays = [np.arange(5, dtype=np.float32)[:, None], np.arange(9, dtype=np.float32)[:, None]]
    norm = normalization(arrays)
    with torch.no_grad():
        one = model(*batch(arrays, [0], norm, "cpu"))[0]
        together = model(*batch(arrays, [0, 1], norm, "cpu"))[0]
        assert one.item() == pytest.approx(together.item(), abs=1e-6)
        shuffled = model(*batch(arrays, [0], norm, "cpu", np.random.default_rng(4)))[0]
        if kind in {"deepsets", "attention_nopos"}:
            assert one.item() == pytest.approx(shuffled.item(), abs=1e-6)


def test_positions_require_all_error_reviews_and_use_length_matching(dataset):
    _, _, _, calibration, rows = dataset
    annotations = [{"sample_id": r["sample_id"], "source_row_hash": r["source_row_hash"], "reviewed": True,
                    "process_valid": r["is_correct"], "first_error_position": None if r["is_correct"] else (0 if i < 4 else len(r["generated_token_ids"])-1)}
                   for i, r in enumerate(rows)]
    result = analyze(rows, calibration, annotations, ["fdbd_all"], ["mean"], bootstrap=20)
    record = result["records"][0]
    assert "early_minus_late_tpr" in record
    assert all(b["n_errors"] == b["n_matched_controls"] for b in record["buckets"])
    with pytest.raises(ValueError, match="Review ALL"):
        analyze(rows, calibration, annotations[1:], ["fdbd_all"], ["mean"], bootstrap=20)
    annotations[0].update(process_valid=None, first_error_position=None, notes="Cannot determine the first invalid inference")
    unknown = analyze(rows, calibration, annotations, ["fdbd_all"], ["mean"], bootstrap=20)
    assert unknown["reviewed_undecidable_processes"] == 1 and unknown["unlocated_final_errors"] == 1
    selected, distances = match_controls(rows, [0, 2], [1, 3, 5, 7])
    assert len(set(selected)) == 2 and len(distances) == 2


def test_complete_fitting_all_models_and_resume(dataset, tmp_path, monkeypatch):
    import ood_repro.eval2_fit as module
    pool, test, split, _, _ = dataset
    output = tmp_path / "fit"
    args = ["fit", "--pool", str(pool), "--test", str(test), "--split", str(split), "--output", str(output),
            "--metrics", "fdbd_all", "--seeds", "0", "--epochs", "2", "--hidden", "4", "--batch-size", "16",
            "--bootstrap", "20", "--min-train-errors", "2", "--resume"]
    monkeypatch.setattr(sys, "argv", args)
    module.main()
    saved = json.loads((output / "experiment2.json").read_text())
    assert {"gru", "cnn", "attention", "attention_nopos", "deepsets", "gru_shuffled", "logreg"} <= {r["method"] for r in saved["records"]}
    assert (output / "experiment2.png").exists()
    monkeypatch.setattr(module, "train_sequence", lambda *a: pytest.fail("Completed jobs must be reused"))
    module.main()
    assert json.loads((output / "manifest.json").read_text())["complete"]


def test_preparation_and_experiment1_cli(dataset, tmp_path, monkeypatch):
    import ood_repro.eval2 as setup
    import ood_repro.eval2_positions as positions
    pool, test, split, _, rows = dataset
    template = tmp_path / "review"
    monkeypatch.setattr(sys, "argv", ["eval2", "prepare-annotations", "--test", str(test), "--output", str(template)])
    setup.main()
    annotations = [json.loads(s) for s in (template / "annotations.jsonl").read_text().splitlines()]
    assert len(annotations) == 8 and all(a["reviewed"] is False for a in annotations)
    assert not any("scores" in a for a in annotations)
    lookup = {r["sample_id"]: r for r in rows}
    for a in annotations:
        row = lookup[a["sample_id"]]
        a.update(reviewed=True, process_valid=row["is_correct"], first_error_position=None if row["is_correct"] else 0)
    reviewed = tmp_path / "reviewed.jsonl"
    reviewed.write_text("".join(json.dumps(a) + "\n" for a in annotations), encoding="utf-8")
    output = tmp_path / "positions"
    monkeypatch.setattr(sys, "argv", ["positions", "--pool", str(pool), "--test", str(test), "--split", str(split),
        "--annotations", str(reviewed), "--output", str(output), "--metrics", "fdbd_all", "--aggregations", "mean", "--bootstrap", "20"])
    positions.main()
    assert (output / "experiment1.png").exists() and (output / "positions.csv").exists()


def test_online_import_avoids_second_model_pass_and_validates_saved_tokens(dataset, tmp_path, monkeypatch):
    import ood_repro.eval2 as module
    from ood_repro.sequence_analysis import load_replay
    _, _, _, rows, _ = dataset
    raw, output = tmp_path / "raw", tmp_path / "imported"
    raw.mkdir()
    converted = []
    for row in rows:
        source = {k: v for k, v in row.items() if k not in {"source_row_hash", "record_hash", "sequence_tokens", "sequence_scores"}}
        converted.append({**source, "token_scores": row["sequence_tokens"]})
    manifest = {"config": {"include_eos": False}, "eos_token_ids": [30], "environment": {},
                "dataset": {"split": "train", "sample_ids": [r["sample_id"] for r in converted]}}
    write_json(raw / "manifest.json", manifest)
    (raw / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in converted), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["import", "import-online", "--test", str(raw), "--output", str(output), "--resume"])
    module.main()
    _, imported = load_replay(output)
    assert len(imported) == 60 and imported[0]["sequence_scores"] == rows[0]["sequence_scores"]
    before = (output / "predictions.jsonl").read_bytes()
    module.main()
    assert (output / "predictions.jsonl").read_bytes() == before
