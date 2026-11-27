"""Output-head geometry in FP32, using the raw next-token distribution."""
import torch

from ood_repro.scoring import GeometryScorer


def fisher_std(weight, logits, directions):
    """Each column of directions is a unit vector in head-input coordinates."""
    response = weight @ directions
    p = logits.float().softmax(-1)[:, None]
    mean = (p * response).sum(0)
    return (p * (response - mean).square()).sum(0).sqrt()


class DiagnosticScorer:
    def __init__(self, model, *, cached_baselines=False, seed=42):
        head = model.get_output_embeddings()
        self.weight = head.weight.detach().float()
        keys = {"nci", "mean_logprob", "logit_margin", "negative_entropy", "shannon_top20"}
        self.baselines = GeometryScorer(head.weight, head.bias, ks=() if cached_baselines else ("all",), metrics=keys)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        direction = torch.randn(self.weight.shape[1], generator=gen)
        self.random = (direction / direction.norm()).to(self.weight.device)
        self.step_directions = None
        self.token_steps = {}
        self.position = 0

    @torch.no_grad()
    def score(self, z, logits, token):
        logits = logits.float()
        centered = z.float() - self.baselines.center
        norm = centered.norm()
        values, ids = logits.topk(2)
        normal = self.weight[ids[0]] - self.weight[ids[1]]
        scale = normal.norm()
        if not bool(torch.isfinite(norm + scale) & (norm > 0) & (scale > 0)):
            raise ValueError("Undefined angular geometry: zero or nonfinite vector")
        normal, radial = normal / scale, centered / norm
        sensitivity = fisher_std(self.weight, logits, torch.stack((normal, radial, self.random), dim=1))
        result = self.baselines.score(z, logits, token)
        result.update(norm=norm, angle_boundary=1 - (radial @ normal).clamp(-1, 1),
                      distance_top2=(values[0] - values[1]) / scale,
                      fisher_normal=sensitivity[0], fisher_radial=sensitivity[1],
                      fisher_random=sensitivity[2], winner_id=ids[0], competitor_id=ids[1])
        if self.step_directions is not None:
            step = self.token_steps.get(self.position, 0)
            direction = self.step_directions.get(step, self.step_directions[0])
            result["angle_step_ref"] = 1 - (radial @ direction).clamp(-1, 1)
        self.position += 1
        return result
