import numpy as np
import pytest
import torch
from scipy import ndimage as ndi

from scripts.comparators import components as C


def official_one_hot2dist(seg_onehot):
    """Transcription of LIVIAETS/boundary-loss utils.one_hot2dist @171c32d (resolution=None)."""
    res = np.zeros_like(seg_onehot, dtype=np.float32)
    for k in range(len(seg_onehot)):
        posmask = seg_onehot[k].astype(bool)
        if posmask.any():
            negmask = ~posmask
            res[k] = (
                ndi.distance_transform_edt(negmask) * negmask
                - (ndi.distance_transform_edt(posmask) - 1) * posmask
            )
    return res


def test_kervadec_dist_matches_official_foreground_channels():
    rng = np.random.default_rng(0)
    for _ in range(5):
        mask = np.zeros((40, 48), dtype=np.int64)
        mask[5:20, 6:30] = 1
        mask[25:35, 10:18] = 2
        mask[rng.integers(0, 40), rng.integers(0, 48)] = 3
        onehot = np.stack([mask == k for k in range(5)])  # class 4 absent
        ref = official_one_hot2dist(onehot)[1:]
        got = C.kervadec_dist_maps(mask, 5)
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-5)
        assert np.all(got[3] == 0)  # absent class left blank as upstream


def test_kervadec_dist_known_values():
    mask = np.zeros((1, 7), dtype=np.int64)
    mask[0, 2:5] = 1
    got = C.kervadec_dist_maps(mask, 2)[0, 0]
    np.testing.assert_allclose(got, [2, 1, 0, -1, 0, 1, 2])


def test_steal_weight_schedule():
    assert C.kervadec_weights(1) == (1.0, 0.01)
    r, b = C.kervadec_weights(2)
    assert r == pytest.approx(0.99) and b == pytest.approx(0.02)
    r, b = C.kervadec_weights(91)
    assert r == pytest.approx(0.1) and b == pytest.approx(0.91)
    r, b = C.kervadec_weights(150)
    assert r == pytest.approx(0.1) and b == pytest.approx(1.50)


def test_kervadec_objective_penalises_outside_mass():
    mask = torch.zeros(1, 16, 16, dtype=torch.long)
    mask[:, 4:10, 4:10] = 1
    dist = torch.from_numpy(C.kervadec_dist_maps(mask[0].numpy(), 2))[None]
    objective = C.KervadecBoundaryObjective(2)
    objective.set_epoch(100)
    logits = torch.zeros(1, 2, 16, 16, requires_grad=True)
    loss, scalars = objective(dict(seg_logits=logits), dict(mask=mask, kervadec_dist=dist))
    loss.backward()
    # raising foreground logit far outside the lesion must increase the loss
    assert logits.grad[0, 1, 15, 15] > 0
    assert logits.grad[0, 1, 6, 6] < logits.grad[0, 1, 15, 15]
    assert scalars["boundary_weight"] == pytest.approx(1.0)


@pytest.mark.parametrize("channels,classes", [(3, 2), (1, 4), (1, 9)])
def test_lbunet_logits_match_official_sigmoid_output(channels, classes):
    torch.manual_seed(0)
    model = C.LBUNetK(channels, classes).eval()
    x = torch.randn(2, channels, 224, 224)
    with torch.no_grad():
        out = model(x, return_boundary=True)
        _, _, official = model.net(x)
    assert out["seg_logits"].shape == (2, classes, 224, 224)
    torch.testing.assert_close(torch.sigmoid(out["seg_logits"]), official)
    assert len(out["lb_gt_pre"]) == 5 and len(out["lb_boundary_pre"]) == 3
    assert all(p.shape == (2, 1, 224, 224) for p in out["lb_gt_pre"] + out["lb_boundary_pre"])
    deploy = model(x[:1], return_boundary=False)
    assert set(deploy) == {"seg_logits"}


def test_lbunet_objective_backward():
    torch.manual_seed(0)
    model = C.LBUNetK(1, 4).train()
    x = torch.randn(2, 1, 224, 224)
    mask = torch.randint(0, 4, (2, 224, 224))
    boundary = torch.randint(0, 2, (2, 1, 224, 224)).float()
    loss, scalars = C.LBUNetObjective()(model(x), dict(mask=mask, boundary=boundary))
    loss.backward()
    assert torch.isfinite(loss) and {"deep_gt", "deep_boundary"} <= set(scalars)
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_phase16_scipy_dtm_matches_tracked_torch_dtm():
    from bds_lite.losses.gsl import distance_transform_maps_torch

    target = torch.zeros(2, 20, 24, dtype=torch.long)
    target[0, 3:9, 4:12] = 1
    target[1, 10:18, 2:6] = 2
    np.testing.assert_allclose(
        C.distance_transform_maps_scipy(target, 3).numpy(),
        distance_transform_maps_torch(target, 3).numpy(),
        atol=1e-4,
    )
