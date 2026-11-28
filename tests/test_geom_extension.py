"""Regression checks for public-step alignment and first-error evaluation."""
import numpy as np

from geom_diagnostics.extension import SIGNALS, process_tokens
from geom_diagnostics.extension_analysis import alarm_stats, eligible, localize, step_features


class BoundaryTokenizer:
    def __call__(self, text, **kwargs):
        assert text == "a\n\nb"
        return {"input_ids": [10, 11, 12], "offset_mapping": [(0, 1), (1, 2), (2, 4)]}

    def decode(self, ids, **kwargs):
        assert ids == [10, 11, 12]
        return "a\n\nb"


def test_cross_boundary_token_is_not_available_to_previous_step():
    _, mapping = process_tokens(BoundaryTokenizer(), ["a", "b"])
    assert mapping == [0, None, 1]
    row = {"n_steps": 2}
    values = {key: np.array([1., 2., 9.]) for key in SIGNALS}
    signs = dict.fromkeys(SIGNALS, 1)
    scores = step_features(row, values, np.array([0, -1, 1]), signs)
    assert scores[0]["fisher_radial/trailing64"] == -1
    assert scores[1]["fisher_radial/trailing64"] == -4


def test_localization_random_baseline_and_first_error_labels():
    row = {"id": "test", "subset": "math", "first_error": 1}
    steps = np.array([0] * 8 + [1] * 4 + [2] * 8)
    risk = np.array([0.] * 8 + [10.] * 4 + [0.] * 8)
    values = {key: -risk for key in SIGNALS}
    entries = list(localize(row, values, steps, dict.fromkeys(SIGNALS, 1)))
    actual = next(item for item in entries if item["signal"] == "fisher_radial" and item["width"] == 8)
    assert actual["hit"]
    assert actual["random_hit"] == 11/13
    assert list(eligible(1, 3)) == [0, 1]
    assert list(eligible(-1, 3)) == [0, 1, 2]
    assert alarm_stats([1, -1], [1, -1])["official_f1"] == 1
