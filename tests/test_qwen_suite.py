import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ood_repro.qwen_suite import MODEL, PAPER, TASKS, build_command, experiment_dir, read_recipe, summarize


def recipe():
    return {"model_id": MODEL, "model_path": "/data/models/qwen/pinned-sha", "model_revision": "model-sha",
            "datasets": {task: {"revision": f"data-{task}", "expected_count": 2} for task in TASKS}}


def test_commands_pin_assets_use_only_qwen_and_keep_paper_k_in_main(tmp_path):
    for task in TASKS:
        command = build_command(recipe(), tmp_path, task, "main", 20, "bfloat16", 2000)
        assert command[command.index("--model-revision") + 1] == "model-sha"
        assert command[command.index("--dataset-revision") + 1] == f"data-{task}"
        assert "--limit" not in command and "--ks" not in command
        assert Path(command[command.index("--config") + 1]).name == f"qwen_{task}.json"
        config = json.loads(Path(command[command.index("--config") + 1]).read_text())
        assert config["model"] == MODEL
        assert config["ks"] == ("all" if task == "gsm8k" else "100,all")
        smoke = build_command(recipe(), tmp_path, task, "smoke", 20, "bfloat16", 2000)
        assert smoke[smoke.index("--ks") + 1] == "100"
        assert smoke[smoke.index("--limit") + 1] == "20"
        assert smoke[smoke.index("--output") + 1] != command[command.index("--output") + 1]


def make_cells(root, phase="main", incomplete=False):
    for task in TASKS:
        cell = root / phase / f"qwen_{task}"
        cell.mkdir(parents=True)
        manifest = {"config": {"model": recipe()["model_path"], "task": task, "dataset_revision": f"data-{task}"},
                    "dataset": {"sample_ids": ["a", "b"]}}
        metrics = {"n": 1 if incomplete else 2, "n_scored": 2, "accuracy": .5,
                   "parse_failure_rate": 0., "truncation_rate": 0.,
                   "auroc": {k: v / 100 for k, v in PAPER[task].items()}}
        (cell / "manifest.json").write_text(json.dumps(manifest))
        (cell / "metrics.json").write_text(json.dumps(metrics))


def test_summary_scaling_and_incomplete_result_refusal(tmp_path):
    make_cells(tmp_path)
    rows = summarize(tmp_path, recipe(), "main")
    assert len(rows) == 11
    assert all(row["delta_points"] == pytest.approx(0) for row in rows)
    assert (tmp_path / "main/summary.csv").is_file()
    assert "Measured" in (tmp_path / "main/summary.md").read_text()
    cell = tmp_path / "main/qwen_csqa/metrics.json"
    metrics = json.loads(cell.read_text())
    metrics["n"] = 1
    cell.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="Incomplete"):
        summarize(tmp_path, recipe(), "main")


def test_smoke_summary_does_not_compare_with_paper(tmp_path):
    make_cells(tmp_path, "smoke")
    rows = summarize(tmp_path, recipe(), "smoke")
    assert all(row["paper_auroc_x100"] is None and row["delta_points"] is None for row in rows)


def test_experiment_name_cannot_escape_data_root(tmp_path):
    assert experiment_dir(tmp_path, "qwen25-paper-v4").is_relative_to(tmp_path)
    for name in ["../other", ".", "", "a/b"]:
        with pytest.raises(ValueError):
            experiment_dir(tmp_path, name)


def test_recipe_checks_all_weight_shards(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "part1.safetensors").write_bytes(b"test fixture, not real weights")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "part1.safetensors", "b": "part2.safetensors"}}))
    value = {**recipe(), "model_path": str(model)}
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="missing shards"):
        read_recipe(path)


def test_prepare_downloads_public_assets_and_reuses_immutable_recipe(tmp_path, monkeypatch):
    import huggingface_hub
    from ood_repro import qwen_suite as suite
    requests = []

    class Api:
        def model_info(self, model_id, revision):
            requests.append((model_id, revision))
            return SimpleNamespace(sha="fixed-model-commit")

        def dataset_info(self, repo_id):
            return SimpleNamespace(sha=f"fixed-{repo_id}")

    def snapshot(model_id, revision, token, local_dir, allow_patterns):
        assert model_id == MODEL and revision == "fixed-model-commit" and token is False
        target = Path(local_dir)
        target.mkdir(parents=True)
        (target / "config.json").write_text("{}")
        (target / "weights.safetensors").write_bytes(b"synthetic test fixture")

    def examples(task, revision):
        expected = suite.SPECS[task][3]
        return [{}] * expected, {"sample_hash": f"hash-{task}"}

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token: Api())
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(suite, "load_examples", examples)
    first = suite.prepare(tmp_path, "run", "main")
    second = suite.prepare(tmp_path, "run", "main")
    assert first == second
    assert requests == [(MODEL, "main")]
    assert all(first["datasets"][task]["expected_count"] == suite.SPECS[task][3] for task in TASKS)


def test_orchestration_executes_all_three_cells_with_separate_outputs(tmp_path, monkeypatch):
    import torch
    from ood_repro import qwen_suite as suite
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "weights.safetensors").write_bytes(b"synthetic test fixture")
    value = {**recipe(), "model_path": str(model)}
    for task in TASKS:
        value["datasets"][task]["sample_hash"] = f"hash-{task}"
    root = experiment_dir(tmp_path, "run")
    suite.write_json(root / "recipe.json", value)
    calls = []

    def launch(command, cwd, check):
        assert check is True
        calls.append(command)
        config_path = Path(command[command.index("--config") + 1])
        task = json.loads(config_path.read_text())["task"]
        output = Path(command[command.index("--output") + 1])
        suite.write_json(output / "manifest.json", {"config": {"model": str(model), "task": task,
                         "dataset_revision": f"data-{task}", "model_revision": value["model_revision"],
                         "dtype": "bfloat16"}, "dataset": {"sample_ids": ["a", "b"]}})
        suite.write_json(output / "metrics.json", {"n": 2, "n_scored": 2, "accuracy": .5,
                         "parse_failure_rate": 0, "truncation_rate": 0, "auroc": {"nci": .7}})

    monkeypatch.setattr(suite, "load_examples", lambda task, revision: ([], {"sample_hash": f"hash-{task}"}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (24 * 2**30, 24 * 2**30))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "synthetic orchestration test GPU")
    monkeypatch.setattr(suite.subprocess, "run", launch)
    suite.run_cells(tmp_path, "run", "smoke", 20, "bfloat16", 0)
    suite.run_cells(tmp_path, "run", "main", 20, "bfloat16", 0)
    assert len(calls) == 6
    assert len({command[command.index("--output") + 1] for command in calls}) == 6
    assert (root / "main/summary.json").is_file()
