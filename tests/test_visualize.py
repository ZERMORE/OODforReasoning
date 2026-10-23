import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ood_repro.data import digest
from ood_repro.visualize import bootstrap_aucs, find_main, load_cell, risk_coverage, weighted_auc


def test_weighted_auc_matches_independent_sklearn_with_ties_and_duplicates():
    y = np.array([0, 1, 1, 0, 1])
    risk = np.array([.2, .2, .8, .4, .8])
    counts = np.array([2, 3, 0, 1, 2])
    idx = np.repeat(np.arange(len(y)), counts)
    assert weighted_auc(y, risk, counts) == pytest.approx(roc_auc_score(y[idx], risk[idx]), abs=1e-14)


def test_bootstrap_shared_indices_match_brute_force_reference():
    y = np.array([0, 1, 1, 0, 1, 0])
    scores = {"a": np.array([3, 2, 2, 1, 0, 4.]), "b": np.array([1, 0, 2, 4, 1, 3.])}
    fast = bootstrap_aucs(y, scores, repeats=100, seed=9)
    slow = {key: [] for key in scores}
    rng = np.random.default_rng(9)
    for _ in range(100):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        for key, values in scores.items():
            slow[key].append(roc_auc_score(y[idx], -values[idx]))
    for key in scores:
        np.testing.assert_allclose(fast[key], slow[key], atol=1e-14, rtol=0)


def test_risk_coverage_never_arbitrarily_splits_tied_scores():
    y = np.array([1, 0, 0, 1])
    scores = np.array([2, 2, 1, 1])
    coverage, risk = risk_coverage(y, scores)
    np.testing.assert_allclose(coverage, [.5, 1.])
    np.testing.assert_allclose(risk, [.5, .5])
    permutation = np.array([1, 0, 3, 2])
    np.testing.assert_allclose(risk_coverage(y[permutation], scores[permutation]), (coverage, risk))


def fixture_cell(path):
    path.mkdir(parents=True)
    rows = []
    for i, good in enumerate([True, False, True, False]):
        answer = "A" if good else "B"
        rows.append({"sample_id": str(i), "question": f"Question {i}?", "options": [], "gold_answer": "A",
                     "parsed_answer": answer, "is_correct": good, "response": f"So the answer is ({answer.lower()}).",
                     "stop_reason": "eos", "generated_token_ids": [i],
                     "scores": {key: float(4 - i) for key in ["mean_logprob", "nci", "fdbd_all", "fdbd_k100", "logit_margin"]}})
    sample_hash = digest([{key: row[key] for key in ("sample_id", "question", "options", "gold_answer")} for row in rows])
    manifest = {"config": {"task": "csqa", "verify_replay": 0}, "dataset": {"split": "validation", "sample_ids": [str(i) for i in range(4)], "sample_hash": sample_hash}}
    auc = roc_auc_score([0, 1, 0, 1], [-4, -3, -2, -1])
    metrics = {"n": 4, "n_scored": 4, "n_unscored": 0, "accuracy": .5, "parse_failure_rate": 0., "truncation_rate": 0.,
               "auroc": {key: auc for key in rows[0]["scores"]}}
    for filename, value in [("manifest.json", manifest), ("metrics.json", metrics)]:
        (path / filename).write_text(json.dumps(value), encoding="utf-8")
    (path / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return rows


def test_audit_accepts_complete_consistent_cell_and_rejects_label_tampering(tmp_path):
    rows = fixture_cell(tmp_path / "main/qwen_csqa")
    cell = load_cell(tmp_path / "main/qwen_csqa", "csqa", expected_count=4)
    assert cell["errors"].sum() == 2
    assert find_main(tmp_path) == tmp_path / "main"
    rows[0]["is_correct"] = False
    (tmp_path / "main/qwen_csqa/predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="label mismatch"):
        load_cell(tmp_path / "main/qwen_csqa", "csqa", expected_count=4)


def test_audit_rejects_stale_metrics(tmp_path):
    fixture_cell(tmp_path / "cell")
    path = tmp_path / "cell/metrics.json"
    metrics = json.loads(path.read_text())
    metrics["auroc"]["nci"] = .123
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="differs from metrics"):
        load_cell(tmp_path / "cell", "csqa", expected_count=4)


def test_ambiguous_main_directories_are_rejected(tmp_path):
    fixture_cell(tmp_path / "one/main/qwen_csqa")
    fixture_cell(tmp_path / "two/main/qwen_csqa")
    with pytest.raises(ValueError, match="exactly one"):
        find_main(tmp_path)
