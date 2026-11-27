"""Regression for G1 top-2 integer metadata with the shared replay engine."""
from types import SimpleNamespace

import torch

from geom_diagnostics.scorer import DiagnosticScorer
from ood_repro import engine


class TinyHeadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 3)
        self.head = torch.nn.Linear(3, 8, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, past_key_values, **kwargs):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)), past_key_values=past_key_values)


def test_matched_replay_preserves_top2_metadata_without_averaging_it(monkeypatch):
    torch.manual_seed(7)
    monkeypatch.setattr(engine, "new_cache", lambda: None)
    model = TinyHeadModel().eval()
    # Exercise the full scoring route, including fDBD-All.
    scorer = DiagnosticScorer(model)
    scorer.step_directions = {0: torch.tensor([1., 0., 0.]), 2: torch.tensor([0., 1., 0.])}
    for prompt, generated, positions, predecessors in (
        ([1, 2], [3, 0, 4, 0], [0, 2], [2, 0]),
        ([2, 4], [1, 0], [0], [4]),
    ):
        mapping = {i: 2 for i in positions}
        scorer.begin_trajectory(positions, mapping)
        result = engine.replay(model, prompt, generated, scorer, stop_ids={0},
                               include_eos=False, save_tokens=True, mode="matched")
        trace = result["token_scores"]
        scorer.attach_metadata(trace)
        assert "winner_id" not in result["scores"] and "competitor_id" not in result["scores"]
        assert "angle_step_ref" in result["scores"] and "fdbd_all" in result["scores"]
        expected_ids = model.head(model.embedding(torch.tensor(predecessors))).topk(2).indices.tolist()
        assert [[t["winner_id"], t["competitor_id"]] for t in trace] == expected_ids
        assert [t["position"] for t in trace] == positions
        assert all(t["step"] == 2 and "winner_id" not in t["scores"] for t in trace)
