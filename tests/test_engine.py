"""Real randomly initialized tiny Llama/Qwen2; no model download needed."""
import pytest
import torch

from ood_repro.engine import check_head, generate, replay
from ood_repro.scoring import GeometryScorer


@pytest.mark.parametrize("architecture", ["llama", "qwen2"])
def test_online_chunked_replay_and_independent_full_forward(architecture):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(17)
    config_cls, model_cls = (LlamaConfig, LlamaForCausalLM) if architecture == "llama" else (Qwen2Config, Qwen2ForCausalLM)
    config = config_cls(vocab_size=31, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                        max_position_embeddings=128, attention_dropout=0., eos_token_id=30)
    config._attn_implementation = "eager"
    model = model_cls(config).eval()
    prompt = [1, 5, 9, 3]
    scorer = GeometryScorer(model.get_output_embeddings().weight, ks=(1, 5, "all"), vocab_chunk=7)
    assert check_head(model, prompt) < 1e-6
    online = generate(model, prompt, scorer, max_new_tokens=6, force_length=True, save_tokens=True)
    replayed = replay(model, prompt, online["generated_token_ids"], scorer, chunk_size=2, save_tokens=True)
    for k, value in online["scores"].items():
        assert value == pytest.approx(replayed["scores"][k], abs=1e-6, rel=1e-5)
    # Independent causal alignment check through a single full-sequence forward.
    captured = []
    hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    seq = prompt + online["generated_token_ids"]
    with torch.no_grad():
        output = model(torch.tensor([seq]), use_cache=False)
    hook.remove()
    z = captured[0]
    for t, token in enumerate(online["generated_token_ids"]):
        pos = len(prompt) - 1 + t
        expected = scorer.score(z[0, pos], output.logits[0, pos], token)
        for k, value in expected.items():
            assert value.item() == pytest.approx(online["token_scores"][t]["scores"][k], abs=1e-6, rel=1e-5)
    assert not model.get_output_embeddings()._forward_pre_hooks


def test_eos_exclusion_and_sample_reproducibility():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(3)
    config = LlamaConfig(vocab_size=11, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    model = LlamaForCausalLM(config).eval()
    scorer = GeometryScorer(model.get_output_embeddings().weight, ks=(2,))
    eos = generate(model, [1, 2], scorer, max_new_tokens=2, forced_tokens=[3, 4], stop_ids={3})
    assert eos["scored_tokens"] == 0 and eos["scores"] == {}
    assert eos["stop_reason"] == "eos"
    a = generate(model, [1, 2], max_new_tokens=4, temperature=.8, top_p=.9, seed=5)
    b = generate(model, [1, 2], max_new_tokens=4, temperature=.8, top_p=.9, seed=5)
    assert a["generated_token_ids"] == b["generated_token_ids"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("architecture", ["llama", "qwen2"])
def test_cuda_low_precision_head_and_replay(architecture):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(5)
    config_cls, model_cls = (LlamaConfig, LlamaForCausalLM) if architecture == "llama" else (Qwen2Config, Qwen2ForCausalLM)
    config = config_cls(vocab_size=31, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128)
    config._attn_implementation = "sdpa"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = model_cls(config).to(device="cuda", dtype=dtype).eval()
    prompt = [1, 4, 9, 8]
    assert check_head(model, prompt) < 1e-4
    scorer = GeometryScorer(model.get_output_embeddings().weight, ks=(3, "all"), vocab_chunk=7)
    online = generate(model, prompt, scorer, max_new_tokens=5, force_length=True)
    replayed = replay(model, prompt, online["generated_token_ids"], scorer, chunk_size=2)
    for key, value in online["scores"].items():
        assert value == pytest.approx(replayed["scores"][key], abs=.003, rel=.02)
