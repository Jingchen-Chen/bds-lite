"""Unit tests for the ACDC/Synapse KD study (revision/KD_MULTICLASS_PROTOCOL.md)."""

import copy

import numpy as np
import pytest
import torch
from scipy import ndimage as ndi

from bds_lite.training import BDSLiteLoss, build_model
from bds_lite.training.losses import weights_from_config
from bds_lite.utils import seed_everything
from scripts.kd.kd_loss import kd_loss as binary_kd_loss
from scripts.kd_mc.common import DATASETS, require_protocol, template
from scripts.kd_mc.loss import KDCriterionMC, class_band, kd_loss_mc
from scripts.kd_mc.register_mc import student_config, teacher_config


def test_two_class_case_reduces_exactly_to_the_isic_loss():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 2, 32, 32, generator=g)
    p = torch.rand(3, 32, 32, generator=g)
    p[0, 5, 5] = 0.5  # tie: background under both rules
    two = torch.stack([1 - p, p], 1)
    assert torch.allclose(kd_loss_mc(logits, two), binary_kd_loss(logits, p, "band"), rtol=1e-5)


def test_band_is_union_of_foreground_class_gradients_with_square_element():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 4, size=(2, 30, 30))
    labels[1] = 0
    labels[1, 10:20, 0:15] = 2  # one class touching the border
    prob = torch.nn.functional.one_hot(torch.from_numpy(labels), 4).permute(0, 3, 1, 2).float()
    band = class_band(prob * 0.7 + 0.075).numpy().astype(bool)
    square = np.ones((3, 3), bool)
    for i in range(2):
        expected = np.zeros((30, 30), bool)
        for c in range(1, 4):
            m = labels[i] == c
            expected |= ndi.binary_dilation(m, square) & ~ndi.binary_erosion(
                m, square, border_value=0
            )
        assert np.array_equal(band[i], expected)


def test_empty_band_gives_zero_loss_and_gradient():
    logits = torch.randn(2, 9, 16, 16, requires_grad=True)
    prob = torch.zeros(2, 9, 16, 16)
    prob[:, 0] = 1.0  # background everywhere
    loss = kd_loss_mc(logits, prob)
    assert loss.item() == 0.0
    assert torch.count_nonzero(torch.autograd.grad(loss, logits)[0]) == 0


def test_soft_cross_entropy_matches_manual_band_average():
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(2, 4, 20, 20, generator=g)
    prob = torch.softmax(torch.randn(2, 4, 20, 20, generator=g) * 3, 1)
    band = class_band(prob)
    p_s = torch.softmax(logits, 1).clamp(1e-6, 1 - 1e-6)
    ce = -(prob.clamp(1e-6, 1 - 1e-6) * torch.log(p_s)).sum(1)
    expected = (band * ce).sum() / (band.sum() + 1e-6)
    assert torch.allclose(kd_loss_mc(logits, prob), expected, rtol=1e-6)


@pytest.mark.parametrize("dataset", list(DATASETS))
def test_configs_equal_decoder_recipe_and_lambda_zero_reproduces_its_loss(dataset):
    base = template(dataset)
    config = student_config(dataset, 2, "0" * 64)
    require_protocol(config)
    for block in ["model", "loss", "trainer", "data", "dataset"]:
        assert config[block] == base[block]
    broken = copy.deepcopy(config)
    broken["trainer"]["batch_size"] = 8
    with pytest.raises(ValueError):
        require_protocol(broken)
    teacher = teacher_config(dataset, "0" * 64)
    require_protocol(teacher)
    assert teacher["trainer"] == base["trainer"] and teacher["loss"]["boundary_weight"] == 0.0
    k = base["dataset"]["num_classes"]
    loss = BDSLiteLoss(weights_from_config(base))
    outputs = dict(seg_logits=torch.randn(2, k, 16, 16), boundary_logits=torch.randn(2, 1, 16, 16))
    batch = dict(
        mask=torch.randint(0, k, (2, 16, 16)),
        boundary=(torch.rand(2, 1, 16, 16) > 0.8).float(),
        teacher_prob=torch.softmax(torch.randn(2, k, 16, 16), 1),
    )
    reference, _ = loss(outputs, batch)
    assert torch.equal(KDCriterionMC(loss, 0.0)(outputs, batch)[0], reference)
    assert "kd" in KDCriterionMC(loss, 1.0)(outputs, batch)[1]


@pytest.mark.parametrize("dataset", list(DATASETS))
def test_same_seed_gives_decoder_only_initial_weights(dataset):
    def init(config):
        seed_everything(config["project"]["seed"])
        return build_model(config).state_dict()

    a, b = init(template(dataset)), init(student_config(dataset, 1, "0" * 64))
    assert all(torch.equal(a[key], b[key]) for key in a)


def test_teacher_input_adapter():
    pytest.importorskip("segmentation_models_pytorch")
    from scripts.kd_mc.teacher import TeacherNetMC

    for norm in ["none", "imagenet"]:
        model = TeacherNetMC(4, norm, pretrained=False).eval()
        out = model(torch.rand(2, 1, 64, 64))["seg_logits"]
        assert out.shape == (2, 4, 64, 64)
