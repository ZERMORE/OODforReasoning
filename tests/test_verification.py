from copy import deepcopy

from ood_repro.verification import compare_online_replay


def response():
    return {"scored_tokens": 2, "scores": {"nci": 2.}, "token_scores": [
        {"position": 0, "token_id": 5, "scores": {"nci": 1.}},
        {"position": 1, "token_id": 9, "scores": {"nci": 3.}}]}


def test_equal_trace_passes_without_relaxing_tolerance():
    report = compare_online_replay(response(), response(), rtol=0., atol=0.)
    assert report["passed"] and report["max_token_abs_diff"] == {"nci": 0.}


def test_equal_sequence_mean_cannot_hide_token_drift():
    online = response()
    other = deepcopy(online)
    other["token_scores"][0]["scores"]["nci"] += .5
    other["token_scores"][1]["scores"]["nci"] -= .5
    report = compare_online_replay(online, other, rtol=.01, atol=.001)
    assert not report["passed"]
    assert report["aggregate_abs_diff"]["nci"] == 0
    assert report["token_failure_count"] == 2


def test_misalignment_is_rejected_even_with_identical_scores():
    other = response()
    other["token_scores"][0]["position"] = 1
    report = compare_online_replay(response(), other, rtol=.01, atol=.001)
    assert not report["passed"] and "alignment" in report["first_token_failures"][0]["differences"]


def test_missing_score_or_missing_trace_is_not_silently_ignored():
    other = response()
    other["scores"] = {}
    other["token_scores"] = []
    report = compare_online_replay(response(), other, rtol=.01, atol=.001)
    assert not report["passed"]
    assert "score_keys" in report["aggregate_failures"]
    assert "trace_length" in report["aggregate_failures"]
