import pytest
import torch

from ood_repro.deepconf import TopKConfidence, augment_row, window_scores
from ood_repro.engine import generate
from ood_repro.scoring import GeometryScorer


def test_topk_uses_full_vocabulary_normalization_and_shift_invariance():
    logits = torch.tensor([3., 2., 1., -1.])
    scorer = TopKConfidence(2)
    value = scorer.score(None, logits, 0)["deepconf_mean"]
    expected = torch.logsumexp(logits, 0) - (3 + 2) / 2
    assert value.item() == pytest.approx(expected.item())
    assert scorer.score(None, logits + 100, 0)["deepconf_mean"].item() == pytest.approx(value.item(), abs=1e-5)
    assert value.item() != pytest.approx(-logits.topk(2).values.log_softmax(0).mean().item())
    with pytest.raises(ValueError):
        TopKConfidence(20).score(None, logits, 0)


def test_windows_short_trace_and_bottom_rounding():
    assert window_scores([2, 4], 64) == {"deepconf_mean": 3., "deepconf_lowest": 3., "deepconf_bottom10": 3.}
    values = list(range(12))
    result = window_scores(values, 2)  # 11 windows -> lowest ceil(1.1)=2
    assert result == {"deepconf_mean": 5.5, "deepconf_lowest": .5, "deepconf_bottom10": 1.}
    for values, window in [([], 3), ([float("nan")], 2), ([1], 0)]:
        with pytest.raises(ValueError):
            window_scores(values, window)


def test_teacher_forced_extension_preserves_original_answers_and_scores():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(22)
    config = Qwen2Config(vocab_size=31, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval()
    original = generate(model, [1, 3, 4], GeometryScorer(model.get_output_embeddings().weight, ks=()),
                        max_new_tokens=6, forced_tokens=[5, 6, 7, 8, 9, 30], stop_ids={30})
    row = {**original, "prompt_token_ids": [1, 3, 4], "sample_id": "test", "response": "unchanged"}
    result = augment_row(model, row, TopKConfidence(), {"include_eos": False}, {30}, 3)
    assert len(result["deepconf_token_confidence"]) == 5
    assert result["generated_token_ids"] == row["generated_token_ids"]
    assert result["response"] == row["response"]
    assert all(result["scores"][key] == value for key, value in row["scores"].items())
    broken = {**row, "scores": {**row["scores"], "mean_logprob": 100.}}
    with pytest.raises(ValueError, match="mean_logprob mismatch"):
        augment_row(model, broken, TopKConfidence(), {"include_eos": False}, {30}, 3)


def test_cli_inherits_protocol_writes_paired_metrics_and_resumes(tmp_path, monkeypatch):
    import json
    import sys
    import ood_repro.deepconf as module
    from transformers import Qwen2Config, Qwen2ForCausalLM
    config = Qwen2Config(vocab_size=31, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval()
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    rows = []
    for i in range(2):
        generated = generate(model, [1, 2], GeometryScorer(model.get_output_embeddings().weight, ks=()),
                             max_new_tokens=3, forced_tokens=[4 + i, 6, 30], stop_ids={30})
        rows.append({**generated, "sample_id": str(i), "prompt_token_ids": [1, 2],
                     "parsed_answer": "A", "is_correct": bool(i)})
    prediction = source / "predictions.jsonl"
    prediction.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    before = prediction.read_bytes()
    manifest = {"config": {"model": "local-checkpoint", "model_revision": "pinned", "tokenizer_revision": "pinned",
                           "dtype": "float32", "attention": "eager", "seed": 0, "include_eos": False, "task": "csqa"},
                "dataset": {"sample_ids": ["0", "1"]}, "environment": {"packages": {}}, "eos_token_ids": [30]}
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    def fake_load(args):
        assert args.model_revision == "pinned" and args.dtype == "float32" and args.attention == "eager"
        return model, None
    monkeypatch.setattr(module, "load_model", fake_load)
    monkeypatch.setattr(module, "environment", lambda *args: {"packages": {}})
    argv = ["deepconf", "--source", str(source), "--output", str(output), "--bootstrap", "8", "--resume"]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    report = json.loads((output / "metrics.json").read_text())
    assert "deepconf_bottom10" in report["auroc_ci95"]
    assert (output / "deepconf_vs_logit_margin.json").exists()
    saved = (output / "predictions.jsonl").read_bytes()
    monkeypatch.setattr(module, "load_model", lambda args: pytest.fail("Completed resume should not load model"))
    module.main()
    assert (output / "predictions.jsonl").read_bytes() == saved
    assert prediction.read_bytes() == before
    # A changed window cannot silently append into the old experiment.
    monkeypatch.setattr(sys, "argv", argv + ["--window", "128"])
    with pytest.raises(ValueError, match="Resume source/settings mismatch"):
        module.main()
