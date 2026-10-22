"""Batch-one causal decoding, actual LM-head input hooks, chunked teacher forcing."""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch

from .scoring import aggregate


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def head_inputs(model):
    state = {}
    handle = model.get_output_embeddings().register_forward_pre_hook(
        lambda module, args: state.update(z=args[0].detach())
    )
    try:
        yield state
    finally:
        handle.remove()


def eos_ids(model, tokenizer):
    value = getattr(model.generation_config, "eos_token_id", None)
    if value is None:
        value = tokenizer.eos_token_id
    return set(value if isinstance(value, (tuple, list)) else [value]) - {None}


def sample_token(logits, temperature, top_p, generator):
    if temperature == 0:
        return logits.argmax().item()
    scaled = logits.float() / temperature
    if top_p < 1:
        values, indices = scaled.sort(descending=True)
        cumulative = values.softmax(-1).cumsum(-1)
        remove = cumulative - values.softmax(-1) >= top_p
        values[remove] = -torch.inf
        draw = torch.multinomial(values.softmax(-1), 1, generator=generator)
        return indices[draw].item()
    return torch.multinomial(scaled.softmax(-1), 1, generator=generator).item()


@torch.inference_mode()
def generate(model, prompt_ids, scorer=None, *, max_new_tokens=512, temperature=0.0,
             top_p=1.0, seed=0, stop_ids=(), include_eos=False, force_length=False,
             save_tokens=False, forced_tokens=None):
    if not prompt_ids or max_new_tokens < 1 or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid generation arguments")
    device = model.get_input_embeddings().weight.device
    rng = torch.Generator(device=device).manual_seed(seed)
    tokens, steps, trace = [], [], []
    past = None
    current = torch.tensor([prompt_ids], device=device)
    synchronize(device)
    start = time.perf_counter()
    with head_inputs(model) as state:
        for t in range(max_new_tokens):
            out = model(input_ids=current, past_key_values=past, use_cache=True, return_dict=True)
            raw = out.logits[0, -1]
            token = forced_tokens[t] if forced_tokens is not None else sample_token(raw, temperature, top_p, rng)
            tokens.append(token)
            is_eos = token in stop_ids
            if scorer is not None and (include_eos or not is_eos):
                scores = scorer.score(state["z"][0, -1], raw, token)
                steps.append(scores)
                if save_tokens:
                    trace.append({"position": t, "token_id": token, "scores": scores})
            past = out.past_key_values
            # Release full prefill logits / head input instead of retaining them via out/state.
            state.clear()
            del out
            if is_eos and not force_length:
                break
            current = torch.tensor([[token]], device=device)
    synchronize(device)
    elapsed = time.perf_counter() - start
    for item in trace:
        item["scores"] = {k: v.item() for k, v in item["scores"].items()}
    return {"generated_token_ids": tokens, "scores": aggregate(steps) if steps else {},
            "scored_tokens": len(steps), "token_scores": trace,
            "stop_reason": "eos" if tokens[-1] in stop_ids and not force_length else "max_new_tokens",
            "elapsed_seconds": elapsed, "ms_per_generated_token": elapsed * 1000 / len(tokens)}


@torch.inference_mode()
def replay(model, prompt_ids, generated_ids, scorer, *, chunk_size=16, stop_ids=(),
           include_eos=False, save_tokens=False):
    """Feed prompt[:-1], then [prompt[-1], y1,...,y(T-1)] in chunks.

    Output position t therefore predicts yt, including the prompt's final state.
    This is for score correctness, not for measuring online latency.
    """
    if not prompt_ids or not generated_ids or chunk_size < 1:
        raise ValueError("Empty sequence or invalid replay chunk size")
    device = model.get_input_embeddings().weight.device
    past, steps, trace = None, [], []
    prediction_inputs = [prompt_ids[-1]] + generated_ids[:-1]
    with head_inputs(model) as state:
        if len(prompt_ids) > 1:
            # Bound prefill memory as well; no all-layer hidden state requests.
            for start in range(0, len(prompt_ids) - 1, chunk_size):
                ids = prompt_ids[start:min(start + chunk_size, len(prompt_ids) - 1)]
                out = model(input_ids=torch.tensor([ids], device=device), past_key_values=past,
                            use_cache=True, return_dict=True)
                past = out.past_key_values
                state.clear()
                del out
        for start in range(0, len(prediction_inputs), chunk_size):
            ids = prediction_inputs[start:start + chunk_size]
            out = model(input_ids=torch.tensor([ids], device=device), past_key_values=past,
                        use_cache=True, return_dict=True)
            for j in range(len(ids)):
                token = generated_ids[start + j]
                if not include_eos and token in stop_ids:
                    continue
                scores = scorer.score(state["z"][0, j], out.logits[0, j], token)
                steps.append(scores)
                if save_tokens:
                    trace.append({"position": start + j, "token_id": token, "scores": scores})
            past = out.past_key_values
            state.clear()
            del out
    for item in trace:
        item["scores"] = {k: v.item() for k, v in item["scores"].items()}
    return {"scores": aggregate(steps) if steps else {}, "scored_tokens": len(steps), "token_scores": trace}


@torch.inference_mode()
def check_head(model, prompt_ids):
    device = model.get_input_embeddings().weight.device
    with head_inputs(model) as state:
        out = model(input_ids=torch.tensor([prompt_ids], device=device), use_cache=False, return_dict=True)
        # Keep the original GEMM shape: single-position BF16 kernels can round
        # differently from the prefill GEMM even for a genuinely linear head.
        rebuilt = model.get_output_embeddings()(state["z"])[0, -1].float()
        actual = out.logits[0, -1].float()
        error = (rebuilt - actual).abs().max().item()
        if not torch.allclose(rebuilt, actual, atol=1e-4, rtol=1e-3):
            raise ValueError(f"LM head is not the assumed linear projection; max error={error}")
    return error
