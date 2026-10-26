import csv

import numpy as np
import pytest

from ood_repro.main_table import METHODS, build_rows, export, rank, render, verify_all_scores
from ood_repro.qwen_suite import TASKS


def cells():
    return {task: {"metrics": {"auroc": {key: .7 + i * .001 for i, (key, _) in enumerate(METHODS)
                                        if key != "selected" and not (task == "gsm8k" and key == "fdbd_k100")}}}
            for task in TASKS}


def test_complete_table_keeps_missing_cell_and_resolves_aliases():
    result = build_rows(cells())
    assert len(result) == 10
    by_key = {row["method"]: row for row in result}
    assert by_key["fdbd_k100"]["tasks"]["gsm8k"]["auroc"] is None
    for task, expected in [("csqa", "fdbd_k100"), ("gsm8k", "fdbd_all"), ("aqua", "fdbd_k100")]:
        assert by_key["selected"]["tasks"][task] == by_key[expected]["tasks"][task]


def test_rank_uses_distinct_unrounded_values_and_shared_ties():
    values = [77.974, 77.974, 77.971, None, 70.]
    assert rank(77.974, values) == 1
    assert rank(77.971, values) == 2  # Both round to 77.97, but are different.
    assert rank(None, values) is None


def test_exports_include_all_rows_and_blank_missing_value(tmp_path):
    export(build_rows(cells()), tmp_path)
    with (tmp_path / "main_results.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    missing = next(row for row in rows if row["method"] == "fdbd_k100")
    assert missing["gsm8k_auroc_x100"] == ""
    latex = (tmp_path / "main_results.tex").read_text()
    assert "\\multirow{10}" in latex and "---" in latex


def test_complete_table_checks_diagnostic_scores_too():
    value = {"csqa": {"errors": np.array([0, 1]), "rows": [{"scores": {"nci_decoded": 1.}}, {"scores": {"nci_decoded": 0.}}],
                       "metrics": {"auroc": {"nci_decoded": 1.}}}}
    verify_all_scores(value)
    value["csqa"]["metrics"]["auroc"]["nci_decoded"] = .5
    with pytest.raises(ValueError, match="stale metric"):
        verify_all_scores(value)


def test_deepconf_rows_export_with_original_methods_and_window_note(tmp_path):
    value = cells()
    for cell in value.values():
        cell["metrics"]["auroc"].update(deepconf_mean=.73, deepconf_lowest=.74, deepconf_bottom10=.75)
    rows = build_rows(value)
    assert len(rows) == 13
    export(rows, tmp_path)
    assert "\\multirow{13}" in (tmp_path / "main_results.tex").read_text()
    note = "DeepConf: top-k = 20, window = 64, stride = 1; single-trace confidence only."
    render(rows, tmp_path, with_ci=True, deepconf_note=note)
    svg = (tmp_path / "main_results_with_ci.svg").read_text(encoding="utf-8")
    assert "bottom 10% windows" in svg and note in svg
