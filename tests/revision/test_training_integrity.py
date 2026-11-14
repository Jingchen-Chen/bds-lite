"""Engineering invariants; synthetic runs are not scientific experiments."""

import copy
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from bds_lite.training import BDSLiteLoss
from bds_lite.training.losses import LossWeights
from scripts.revision.run_queue import complete
from scripts.revision.train_run import epoch_train
from scripts.revision.training_support import (
    atomic_torch_save,
    gradient_probe,
    loss_components,
    restore_rng,
    rng_state,
)


class TinyProbeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 3, 1), nn.BatchNorm2d(3), nn.Dropout2d(0.2))
        self.seg = nn.Conv2d(3, 2, 1)
        self.boundary = nn.Conv2d(3, 1, 1)
        self.projection = nn.Conv2d(3, 2, 1)

    def forward(self, image):
        features = self.encoder(image)
        return dict(
            seg_logits=self.seg(features),
            boundary_logits=self.boundary(features),
            seg_boundary_features=self.projection(features),
            boundary_features=features[:, :2],
        )


def batch():
    image = torch.arange(128, dtype=torch.float32).reshape(2, 1, 8, 8) / 128
    mask = (image[:, 0] > 0.3).long()
    return dict(image=image, mask=mask, boundary=mask[:, None].float(), sdf=(0.5 - image).float())


def criterion():
    return BDSLiteLoss(LossWeights(boundary=0.05, distill=0.05, surface=1))


def test_probe_preserves_rng_buffers_existing_gradients_and_next_update():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    base = TinyProbeModel().train()
    reference = copy.deepcopy(base)
    for p in base.parameters():
        p.grad = torch.ones_like(p)
    before = copy.deepcopy(base.state_dict())
    rng = rng_state()
    result = gradient_probe(base, batch(), criterion())
    after = rng_state()
    assert torch.equal(rng["torch"], after["torch"])
    assert rng["python"] == after["python"]
    assert np.array_equal(rng["numpy"][1], after["numpy"][1])
    assert all(torch.equal(before[k], v) for k, v in base.state_dict().items())
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in base.parameters())
    assert set(result) >= {"sdf_cosine", "boundary_cosine", "distill_cosine", "aux_norm_ratio"}
    for model in [base, reference]:
        restore_rng(rng)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = criterion()(model(batch()["image"]), batch())
        loss.backward()
        optimizer.step()
    assert all(torch.equal(v, reference.state_dict()[k]) for k, v in base.state_dict().items())


def test_component_sum_matches_active_objective_and_detached_target():
    model = TinyProbeModel().train()
    outputs = model(batch()["image"])
    terms = loss_components(outputs, batch(), criterion())
    expected, _ = criterion()(outputs, batch())
    assert torch.allclose(terms["seg"] + terms["aux"], expected)
    target_gradient = torch.autograd.grad(
        terms["distill"], outputs["boundary_features"], allow_unused=True, retain_graph=True
    )[0]
    assert target_gradient is None


def test_epoch_resume_matches_uninterrupted_rng_optimizer_scheduler_and_shuffle(tmp_path):
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    model = TinyProbeModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    generator = torch.Generator().manual_seed(7)
    dataset = [{k: v[i] for k, v in batch().items()} for i in range(2)] * 5
    loader = DataLoader(dataset, batch_size=2, shuffle=True, generator=generator)
    step, probes = 45, []  # include an actual every-50-step probe
    _, step = epoch_train(model, loader, criterion(), optimizer, scheduler, 1, step, probes, True)
    saved = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        rng=rng_state(),
        loader=generator.get_state(),
        probes=probes,
        step=step,
    )
    atomic_torch_save(saved, tmp_path / "last.pt")
    for epoch in [2, 3]:
        _, step = epoch_train(
            model, loader, criterion(), optimizer, scheduler, epoch, step, probes, True
        )
    expected_model = copy.deepcopy(model.state_dict())
    expected_opt = copy.deepcopy(optimizer.state_dict())
    expected_rng = rng_state()
    expected_scheduler = scheduler.state_dict()
    restored = torch.load(tmp_path / "last.pt", weights_only=False)
    model2 = TinyProbeModel()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=150)
    model2.load_state_dict(restored["model"])
    optimizer2.load_state_dict(restored["optimizer"])
    scheduler2.load_state_dict(restored["scheduler"])
    generator.set_state(restored["loader"])
    restore_rng(restored["rng"])
    step = restored["step"]
    for epoch in [2, 3]:
        _, step = epoch_train(
            model2,
            loader,
            criterion(),
            optimizer2,
            scheduler2,
            epoch,
            step,
            restored["probes"],
            True,
        )
    assert all(torch.equal(v, model2.state_dict()[k]) for k, v in expected_model.items())
    assert scheduler2.state_dict() == expected_scheduler
    assert torch.equal(torch.get_rng_state(), expected_rng["torch"])
    for key, state in expected_opt["state"].items():
        for name, value in state.items():
            assert torch.equal(value, optimizer2.state_dict()["state"][key][name])
    assert not (tmp_path / "last.pt.tmp").exists()


def test_completion_refuses_partial_or_corrupted_artifacts(tmp_path):
    import json

    marker = tmp_path / "final_metrics.json"
    marker.write_text(json.dumps(dict(config_sha256="x", epochs=150, artifacts={"best.pt": "x"})))
    with pytest.raises(ValueError, match="lacks"):
        complete(tmp_path, "x")
    for name in ["best.pt", "config.json", "preds_val.npz", "preds_test.npz"]:
        (tmp_path / name).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        complete(tmp_path, "x")


def test_all_empty_sdf_has_zero_gradient_and_explicit_undefined_cosine():
    data = batch()
    data["sdf"] = torch.zeros_like(data["sdf"])
    model = TinyProbeModel().train()
    result = gradient_probe(model, data, criterion())
    assert result["sdf_norm"] == 0
    assert result["sdf_norm_ratio"] == 0
    assert result["sdf_cosine"] is None
