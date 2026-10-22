"""Offline end-to-end CLI smoke: local checkpoint, tokenizer and raw task rows."""
import json
import subprocess
import sys

import pytest


@pytest.fixture
def tiny_checkpoint(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    import torch
    torch.manual_seed(9)
    root = tmp_path / "model"
    vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "Q": 3, "A": 4,
             "The": 5, "answer": 6, "is": 7, "4": 8, ".": 9, ":": 10}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]")
    tokenizer.save_pretrained(root)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=8, intermediate_size=16,
                                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                       max_position_embeddings=2048, eos_token_id=2))
    model.save_pretrained(root)
    return root


def test_cli_run_replay_resume_and_config_override(tmp_path, tiny_checkpoint):
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"question": "What is 2 plus 2?", "answer": "#### 4"}) + "\n", encoding="utf-8")
    config = tmp_path / "config.json"
    output = tmp_path / "run"
    config.write_text(json.dumps({"model": str(tiny_checkpoint), "task": "gsm8k", "device": "cpu",
                                  "dtype": "float32", "prompt_mode": "plain", "max_new_tokens": 3,
                                  "data_jsonl": str(data), "ks": "1,all", "bootstrap": 0,
                                  "output": str(output), "verify_replay": 1}), encoding="utf-8")
    cmd = [sys.executable, "-m", "ood_repro", "run", "--config", str(config), "--max-new-tokens", "4", "--local-files-only"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["max_new_tokens"] == 4
    prediction = (output / "predictions.jsonl").read_text(encoding="utf-8")
    assert json.loads(prediction)["online_replay_abs_diff"] is not None
    result = subprocess.run(cmd + ["--resume"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (output / "predictions.jsonl").read_text(encoding="utf-8") == prediction
    result = subprocess.run(cmd + ["--resume", "--seed", "77"], capture_output=True, text=True)
    assert result.returncode != 0 and "mismatch" in result.stderr


def test_cli_benchmark_has_separate_detector_paths(tmp_path, tiny_checkpoint):
    output = tmp_path / "benchmark.json"
    cmd = [sys.executable, "-m", "ood_repro", "benchmark", "--model", str(tiny_checkpoint),
           "--device", "cpu", "--dtype", "float32", "--local-files-only", "--tokens", "3",
           "--k", "2", "--repeats", "1", "--warmup", "1", "--output", str(output)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert set(report["raw_ms_per_token"]) == {"standard", "nci", "fdbd"}
    assert len(report["generated_ids"]) == 3
