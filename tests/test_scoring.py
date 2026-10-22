import pytest
import torch

from ood_repro.scoring import GeometryScorer, neutral_point


def test_boundary_distances_against_independent_analytic_geometry():
    # Winner is x-axis class: distances to y-axis and negative x-axis are known.
    w = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    z = torch.tensor([2., 1.])
    score = GeometryScorer(w, ks=(1, 2, "all"), vocab_chunk=1).score(z, w @ z, 0)
    norm = 5 ** .5
    near = 1 / (2 ** .5) / norm
    far = 2 / norm
    assert score["nci"].item() == pytest.approx(2 / norm)
    assert score["fdbd_k1"].item() == pytest.approx(near)
    assert score["fdbd_k2"].item() == pytest.approx((near + far) / 2)
    assert score["fdbd_all"].item() == pytest.approx((near + far) / 2)


def test_center_minimizes_logit_variance_without_vocabulary_projector():
    torch.manual_seed(3)
    w, b = torch.randn(11, 4), torch.randn(11)
    center = neutral_point(w, b, block_size=3)
    wc, bc = w - w.mean(0), b - b.mean()
    expected = torch.linalg.lstsq(wc.double(), -bc.double()).solution.float()
    assert torch.allclose(center, expected, atol=1e-5)
    assert torch.allclose(wc.T @ (wc @ center + bc), torch.zeros(4), atol=1e-5)


def test_bias_is_not_in_nci_numerator_and_sampling_uses_argmax_for_geometry():
    w = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    b = torch.tensor([0., 5., 0.])
    z = torch.tensor([2., 1.])
    scorer = GeometryScorer(w, b, ks=(1,), center=torch.zeros(2))
    result = scorer.score(z, w @ z + b, 0)
    assert result["nci"].item() == pytest.approx(1 / 5 ** .5)
    assert result["nci_decoded"].item() == pytest.approx(2 / 5 ** .5)


def test_common_head_shift_preserves_fdbd_but_changes_nci():
    w = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    z = torch.tensor([2., 1.])
    shifted = w + torch.tensor([3., 2.])
    a = GeometryScorer(w, ks=(2,)).score(z, w @ z, 0)
    b = GeometryScorer(shifted, ks=(2,)).score(z, shifted @ z, 0)
    assert a["fdbd_k2"].item() == pytest.approx(b["fdbd_k2"].item())
    assert a["nci"].item() != pytest.approx(b["nci"].item())


def test_degenerate_geometry_fails_explicitly():
    with pytest.raises(ValueError, match="feature norm"):
        GeometryScorer(torch.eye(2), ks=(1,)).score(torch.zeros(2), torch.zeros(2), 0)
    with pytest.raises(ValueError, match="duplicate"):
        GeometryScorer(torch.ones(2, 2), ks=(1,)).score(torch.ones(2), torch.ones(2), 0)
    with pytest.raises(ValueError, match="Invalid k"):
        GeometryScorer(torch.eye(2), ks=(2,))


def test_tied_logits_exclude_exact_winner():
    w = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    z = torch.tensor([1., 1.])
    result = GeometryScorer(w, ks=(1,)).score(z, w @ z, 0)
    assert result["fdbd_k1"].item() == 0
