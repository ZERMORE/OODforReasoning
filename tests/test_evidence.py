import math

import numpy as np
import pytest
import torch

from ood_repro.data import extract_answer, grade_math, normalize
from ood_repro.evidence import detection_summary, evidence_drop
from ood_repro.scoring import GeometryScorer


def test_top20_entropy_excludes_rest_of_vocabulary():
    scorer = GeometryScorer(torch.ones(21, 2), ks=(), metrics=("shannon_top20",))
    logits = torch.tensor([0.] * 20 + [-1.])
    score = scorer.score(torch.ones(2), logits, 0)["shannon_top20"]
    assert score.item() == pytest.approx(-math.log(20))


def test_drop_risk_and_right_endpoint():
    risk, positions = evidence_drop([0., -3., 0.])
    assert risk == pytest.approx(1.) and positions == [1]
    assert evidence_drop([-1., -1., 0.]) == (0., [])


def test_wrong_answer_threshold_direction_and_ties():
    errors = np.array([1, 1, 1, 1, 0])
    confidence = np.array([0., 1., 1., 2., 3.])
    half = detection_summary(errors, confidence, errors, confidence)["operating_points"]["0.5"]
    assert half["calibration_wrong_acceptance"] == .25
    assert half["coverage"] == .4 and half["selective_accuracy"] == .5


def test_nested_box_and_symbolic_grade():
    answer = extract_answer(r"We first guessed \boxed{2}. The result is \boxed{\frac{1}{2}}.", "math")
    gold = normalize({"problem": "Half?", "solution": r"\boxed{0.5}"}, "math", 0)["gold_answer"]
    assert answer == r"\frac{1}{2}" and gold == "0.5"
    assert extract_answer(r"The answer is $\boxed 2$.", "math") == "2"
    assert extract_answer(r"We have \boxed{2}. Thus \boxed{", "math") is None
    pytest.importorskip("math_verify")
    assert grade_math(answer, gold) == (True, "symbolic")
    assert grade_math(answer, "2") == (False, "symbolic")
