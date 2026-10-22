"""Paper equations (2), (3), (4), (6), (7), with confidence-oriented output."""
from __future__ import annotations

import torch


@torch.no_grad()
def neutral_point(weight, bias=None, block_size=4096):
    """Minimum-norm logit-variance minimizer; never form a V x V matrix.

    Nonzero-bias heads require an O(d^2) matrix and an O(d^3) solve.
    Target Llama/Qwen heads have no bias, so this returns zero immediately.
    """
    if bias is None or torch.count_nonzero(bias).item() == 0:
        return torch.zeros(weight.shape[1], device=weight.device, dtype=torch.float32)
    d, n = weight.shape[1], weight.shape[0]
    gram = torch.zeros(d, d, device=weight.device, dtype=torch.float64)
    rhs = torch.zeros(d, device=weight.device, dtype=torch.float64)
    sum_w = torch.zeros_like(rhs)
    sum_b = bias.double().sum()
    for start in range(0, n, block_size):
        w = weight[start:start + block_size].double()
        b = bias[start:start + block_size].double()
        gram.add_(w.T @ w)
        rhs.add_(w.T @ b)
        sum_w.add_(w.sum(0))
    gram -= torch.outer(sum_w, sum_w) / n
    rhs -= sum_w * (sum_b / n)
    return (-torch.linalg.pinv(gram, hermitian=True) @ rhs).float()


class GeometryScorer:
    def __init__(self, weight, bias=None, ks=(1000,), vocab_chunk=4096, center=None, metrics=None):
        if weight.ndim != 2 or weight.shape[0] < 2:
            raise ValueError("A linear vocabulary head with >= 2 rows is required")
        self.weight, self.bias = weight.detach(), None if bias is None else bias.detach()
        self.ks = tuple(dict.fromkeys(ks))
        self.vocab_chunk = vocab_chunk
        self.metrics = set(metrics) if metrics is not None else {
            "nci", "nci_decoded", "mean_logprob", "max_probability", "negative_entropy", "logit_margin"}
        if vocab_chunk < 1:
            raise ValueError("vocab_chunk must be positive")
        for k in self.ks:
            if k != "all" and (not isinstance(k, int) or not 1 <= k < weight.shape[0]):
                raise ValueError(f"Invalid k={k}; require 1 <= k < vocabulary size")
        self.center = neutral_point(weight, bias) if center is None else center.to(weight.device).float()

    @torch.no_grad()
    def score(self, z, logits, actual_token):
        """Score one prediction state using RAW, unwarped logits.

        Returns GPU scalar tensors. CPU conversion is deferred until sequence end.
        Exact duplicate head rows / zero features raise rather than silently clamp.
        """
        z, logits = z.float(), logits.float()
        centered = z - self.center
        norm = centered.norm()
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("Non-finite or zero centered feature norm")
        winner = logits.argmax()
        w0 = self.weight[winner].float()
        out = {}
        if "nci" in self.metrics:
            out["nci"] = (w0 @ centered) / norm
        if "nci_decoded" in self.metrics:
            out["nci_decoded"] = (self.weight[actual_token].float() @ centered) / norm
        if self.metrics & {"mean_logprob", "max_probability", "negative_entropy"}:
            logp = logits.log_softmax(-1)
            if "mean_logprob" in self.metrics:
                out["mean_logprob"] = logp[actual_token]
            if "max_probability" in self.metrics:
                out["max_probability"] = logp.max().exp()
            if "negative_entropy" in self.metrics:
                out["negative_entropy"] = (logp.exp() * logp).sum()
        if "logit_margin" in self.metrics:
            out["logit_margin"] = logits.topk(2).values.diff().neg().squeeze()
        finite_ks = [k for k in self.ks if k != "all"]
        if finite_ks:
            # Mask the winner explicitly, including ties, before selecting alternatives.
            alternatives = logits.clone()
            alternatives[winner] = -torch.inf
            values, indices = alternatives.topk(max(finite_ks), sorted=True)
            distances = []
            for start in range(0, indices.numel(), self.vocab_chunk):
                idx = indices[start:start + self.vocab_chunk]
                denominators = (w0 - self.weight[idx].float()).norm(dim=-1)
                self._check_denominator(denominators)
                distances.append((logits[winner] - values[start:start + idx.numel()]) / denominators / norm)
            cumulative = torch.cat(distances).cumsum(0)
            for k in finite_ks:
                out[f"fdbd_k{k}"] = cumulative[k - 1] / k
        if "all" in self.ks:
            total = torch.zeros((), device=z.device)
            for start in range(0, self.weight.shape[0], self.vocab_chunk):
                end = min(start + self.vocab_chunk, self.weight.shape[0])
                indices = torch.arange(start, end, device=z.device)
                keep = indices != winner
                denominators = (w0 - self.weight[start:end].float()).norm(dim=-1)[keep]
                self._check_denominator(denominators)
                total += ((logits[winner] - logits[start:end][keep]) / denominators / norm).sum()
            out["fdbd_all"] = total / (self.weight.shape[0] - 1)
        if not all(torch.isfinite(value) for value in out.values()):
            raise ValueError("Non-finite detection score")
        return out

    @staticmethod
    def _check_denominator(value):
        if not torch.isfinite(value).all() or (value <= 0).any():
            raise ValueError("Non-finite / duplicate head rows: boundary distance undefined")


def aggregate(steps):
    """Average over scored generated tokens; log-domain PPL avoids overflow."""
    if not steps:
        raise ValueError("No scored tokens (e.g. EOS-only output with EOS excluded)")
    result = {k: torch.stack([step[k] for step in steps]).mean().item() for k in steps[0]}
    if "mean_logprob" in result:
        result["log_predictive_probability"] = result["mean_logprob"] * len(steps)
    # mean_logprob is a monotone equivalent of -PPL for AUROC, not probability/T.
    return result
