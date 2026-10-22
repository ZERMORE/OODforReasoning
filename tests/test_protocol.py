import json

import pytest

from ood_repro.data import extract_answer, load_examples
from ood_repro.evaluation import evaluate, select_k
from ood_repro.prompting import load_pack


@pytest.mark.parametrize("task,text,expected", [
    ("csqa", "Reasoning. So the answer is (b).", "B"),
    ("csqa", "Maybe (a). The answer is (b).", None),
    ("aqua", "The answer is (E).", "E"),
    ("gsm8k", "We used 7. The answer is -1,200.50.", "-1200.5"),
    ("gsm8k", "The answer is 6. Then reconsider.", None),
    ("gsm8k", "The answer is 1/2.", None),
    ("gsm8k", "The answer is 1e3.", "1E+3"),
])
def test_strict_answer_parsing(task, text, expected):
    assert extract_answer(text, task) == expected


def rows():
    return [{"sample_id": str(i), "is_correct": good, "parsed_answer": "A",
             "stop_reason": "eos", "scores": {"mean_logprob": base, "nci": confidence}}
            for i, (good, base, confidence) in enumerate([(True, 1, 4), (False, 2, 1), (True, 3, 3), (False, 4, 2)])]


def test_error_orientation_paired_bootstrap_and_single_class():
    report = evaluate(rows(), bootstrap=100, seed=7)
    assert report["auroc"]["nci"] == 1
    assert report["paired_delta_vs_reference"]["nci"]["delta"] > 0
    assert report["bootstrap"]["accepted"] > 0
    assert "undefined_reason" in evaluate([r for r in rows() if r["is_correct"]])


def test_selection_refuses_evaluation_split(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"dataset": {"split": "validation"}}))
    with pytest.raises(ValueError, match="split=train"):
        select_k(tmp_path, tmp_path / "selection.json")


def test_selection_completed_train_run(tmp_path):
    records = rows()
    for record in records:
        record["scores"]["fdbd_k10"] = record["scores"]["nci"]
        record["scores"]["fdbd_all"] = record["scores"]["mean_logprob"]
    (tmp_path / "manifest.json").write_text(json.dumps({"dataset": {"split": "train", "sample_ids": [r["sample_id"] for r in records]}}))
    (tmp_path / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    result = select_k(tmp_path, tmp_path / "selection.json")
    assert result["selected_k"] == 10
    assert result["calibration_auroc"]["fdbd_k10"] == 1


def test_original_prompt_counts_and_provenance():
    for task, count in [("csqa", 7), ("gsm8k", 8), ("aqua", 4)]:
        pack = load_pack(task)
        assert len(pack["examples"]) == count
        assert len(pack["archive_sha256"]) == 64
    assert "fox" in load_pack("csqa")["examples"][2]["question"]


def test_train_and_eval_synthetic_ids_do_not_collide(tmp_path):
    data = tmp_path / "gsm.jsonl"
    data.write_text(json.dumps({"question": "2+2?", "answer": "#### 4"}) + "\n")
    train, _ = load_examples("gsm8k", "train", data)
    test, _ = load_examples("gsm8k", "test", data)
    assert train[0]["sample_id"] != test[0]["sample_id"]
    assert train[0]["question"] == test[0]["question"]
    excluded, meta = load_examples("gsm8k", "train", data, exclude_questions=["2+2?"])
    assert excluded == [] and meta["excluded_demo_questions"] == 1
