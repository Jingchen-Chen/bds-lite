"""Unit tests required by revision/KD_COMPARATOR_PROTOCOL.md 9.1, plus engineering invariants."""

import copy
import json

import numpy as np
import pytest
import torch
from scipy import ndimage as ndi

from bds_lite.training import BDSLiteLoss, build_model
from bds_lite.training.losses import weights_from_config
from bds_lite.utils import seed_everything
from scripts.kd.common import KD_SPEC, decoder_template, require_protocol
from scripts.kd.data import CACHE_NAME, IDS_NAME, TeacherCacheDataset
from scripts.kd.kd_loss import KDCriterion, kd_loss, teacher_band
from scripts.kd.register_kd import student_config, teacher_config


def _bce(p_s, p_t, clamp=1e-6):
    p_s, p_t = p_s.clamp(clamp, 1 - clamp), p_t.clamp(clamp, 1 - clamp)
    return -(p_t * torch.log(p_s) + (1 - p_t) * torch.log(1 - p_s))


def _batch(n=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, 2, 32, 32, generator=g, requires_grad=True)
    teacher = torch.rand(n, 32, 32, generator=g)
    return logits, teacher


def test_band_matches_scipy_square_element_with_zero_border():
    rng = np.random.default_rng(3)
    masks = rng.random((6, 40, 40)) > 0.6
    masks[0] = True  # region covering the whole image: band along the frame only
    masks[1] = False  # empty mask: empty band
    masks[2, :20, :] = True  # region touching the border
    prob = torch.from_numpy(masks.astype(np.float32) * 0.9 + 0.05)
    band = teacher_band(prob).numpy().astype(bool)
    square = np.ones((3, 3), bool)
    for i, m in enumerate(masks):
        expected = ndi.binary_dilation(m, square) & ~ndi.binary_erosion(m, square, border_value=0)
        assert np.array_equal(band[i], expected)
    assert not band[1].any()


def test_threshold_is_strictly_greater_than_half():
    prob = torch.full((1, 5, 5), 0.5)
    assert teacher_band(prob).sum() == 0
    prob[0, 2, 2] = 0.5001
    assert teacher_band(prob).sum() == 9


def test_full_mode_reduces_to_image_bce():
    logits, teacher = _batch()
    p_s = torch.softmax(logits.float(), 1)[:, 1]
    expected = _bce(p_s, teacher).sum() / (teacher.numel() + KD_SPEC["eps"])
    assert torch.allclose(kd_loss(logits, teacher, "full"), expected, rtol=1e-6, atol=0)


def test_band_mode_is_batch_normalised_over_band_pixels():
    logits, teacher = _batch()
    band = teacher_band(teacher)
    p_s = torch.softmax(logits.float(), 1)[:, 1]
    expected = (band * _bce(p_s, teacher)).sum() / (band.sum() + KD_SPEC["eps"])
    assert torch.allclose(kd_loss(logits, teacher, "band"), expected, rtol=1e-6, atol=0)
    # B = 1 everywhere gives the full-image BCE
    ones = band * 0 + 1
    assert torch.allclose(
        (ones * _bce(p_s, teacher)).sum() / (ones.sum() + KD_SPEC["eps"]),
        kd_loss(logits, teacher, "full"),
    )


def test_empty_band_gives_zero_loss_and_zero_gradient():
    logits, _ = _batch()
    teacher = torch.full((3, 32, 32), 0.2)  # no pixel above threshold: empty mask, empty band
    loss = kd_loss(logits, teacher, "band")
    assert loss.item() == 0.0
    (grad,) = torch.autograd.grad(loss, logits)
    assert torch.count_nonzero(grad) == 0


def test_lambda_zero_reproduces_decoder_only_loss_exactly():
    config = decoder_template()
    base = BDSLiteLoss(weights_from_config(config))
    g = torch.Generator().manual_seed(1)
    outputs = dict(
        seg_logits=torch.randn(2, 2, 16, 16, generator=g),
        boundary_logits=torch.randn(2, 1, 16, 16, generator=g),
        seg_boundary_features=torch.randn(2, 16, 16, 16, generator=g),
        boundary_features=torch.randn(2, 16, 16, 16, generator=g),
    )
    batch = dict(
        mask=(torch.rand(2, 16, 16, generator=g) > 0.5).long(),
        boundary=(torch.rand(2, 1, 16, 16, generator=g) > 0.8).float(),
        teacher_prob=torch.rand(2, 16, 16, generator=g),
    )
    reference, ref_scalars = base(outputs, batch)
    total, scalars = KDCriterion(base, "band", lambda_kd=0.0)(outputs, batch)
    assert torch.equal(total, reference) and scalars == ref_scalars
    with_kd, kd_scalars = KDCriterion(base, "band", lambda_kd=1.0)(outputs, batch)
    assert "kd" in kd_scalars and with_kd > reference


def test_validation_batches_without_cache_skip_the_term():
    config = decoder_template()
    base = BDSLiteLoss(weights_from_config(config))
    outputs = dict(seg_logits=torch.randn(2, 2, 8, 8))
    batch = dict(mask=torch.zeros(2, 8, 8, dtype=torch.long))
    assert torch.equal(KDCriterion(base, "band")(outputs, batch)[0], base(outputs, batch)[0])


def test_student_configs_equal_decoder_recipe_and_reject_changes():
    template = decoder_template()
    config = student_config(4, "band", "0" * 64)
    require_protocol(config)
    for block in ["model", "loss", "trainer", "data", "dataset"]:
        assert config[block] == template[block]
    for change in [("trainer", "lr", 1e-4), ("loss", "boundary_weight", 0.1)]:
        broken = copy.deepcopy(config)
        broken[change[0]][change[1]] = change[2]
        with pytest.raises(ValueError):
            require_protocol(broken)
    broken = copy.deepcopy(config)
    broken["kd"]["band_kernel"] = 5
    with pytest.raises(ValueError):
        require_protocol(broken)
    teacher = teacher_config("0" * 64)
    require_protocol(teacher)
    assert teacher["trainer"]["lr"] == 1e-4 and teacher["trainer"]["weight_decay"] == 1e-5
    assert teacher["trainer"]["batch_size"] == 8


def test_same_seed_gives_decoder_only_initial_weights():
    decoder = decoder_template()
    student = student_config(4, "band", "0" * 64)

    def init(config):
        seed_everything(config["project"]["seed"])
        return build_model(config).state_dict()

    a, b = init(decoder), init(student)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_cache_dataset_aligns_with_manifest(tmp_path, monkeypatch):
    ids = ["a", "b", "c"]
    cache = np.stack([np.full((224, 224), v, np.float16) for v in (0.1, 0.5, 0.9)])
    np.save(tmp_path / CACHE_NAME, cache)
    (tmp_path / IDS_NAME).write_text(json.dumps(ids))

    def fake_init(self, root, split, manifest, classes, channels, boundary=False, sdf=False):
        self.files = []

    def fake_get(self, index):
        return {"image": torch.zeros(3, 224, 224), "mask": torch.zeros(224, 224, dtype=torch.long)}

    from scripts.revision import training_support

    monkeypatch.setattr(training_support.ManifestDataset, "__init__", fake_init)
    monkeypatch.setattr(training_support.ManifestDataset, "__getitem__", fake_get)
    data = TeacherCacheDataset(None, "train", {"sample_ids": ids}, 2, 3, True, False, tmp_path)
    assert data[2]["teacher_prob"].dtype == torch.float16
    assert torch.allclose(data[2]["teacher_prob"].float(), torch.tensor(0.9).half().float())
    with pytest.raises(ValueError):
        TeacherCacheDataset(
            None, "train", {"sample_ids": ["a", "c", "b"]}, 2, 3, True, False, tmp_path
        )


def test_kd_supervisor_resumes_only_verified_thermal_stops(tmp_path, monkeypatch):
    from scripts.kd import supervise_kd

    monkeypatch.setattr(supervise_kd, "RUNS", tmp_path)
    monkeypatch.setattr(supervise_kd, "STOP", tmp_path / "STOP_KD_TRAINING")
    monkeypatch.setattr(supervise_kd, "ROOT", tmp_path)
    monkeypatch.setattr(
        supervise_kd.shutil, "disk_usage", lambda path: type("D", (), {"free": 100 * 1024**3})
    )
    run = tmp_path / "isic2018_kd_band_seed1"
    run.mkdir()
    (run / "gpu_telemetry.csv").write_text("timestamp, 75, rest\n")
    (run / "last.pt").write_bytes(b"checkpoint")
    (run / "progress.json").write_text(json.dumps(dict(epoch=24)))
    text = "run failed: isic2018_kd_band_seed1: 130"
    assert supervise_kd.thermal_evidence(text) is None  # not hot
    (run / "gpu_telemetry.csv").write_text("timestamp, 85, rest\n")
    assert supervise_kd.thermal_evidence(text)["saved_progress_epoch"] == 24
    assert supervise_kd.thermal_evidence("run failed: isic2018_kd_band_seed1: 1") is None
    (tmp_path / "STOP_KD_TRAINING").write_text("stop")
    assert supervise_kd.thermal_evidence(text) is None  # explicit stop is never resumed
    assert supervise_kd.thermal_evidence("GPU not idle/cool: 81 C") == "cool_launch_guard"


def test_queue_completion_refuses_test_outputs(tmp_path):
    from scripts.kd.run_queue_kd import complete

    entry = dict(kind="train", marker="final_metrics.json")
    assert complete(tmp_path, entry, "x") is False
    (tmp_path / "per_case_test.csv").write_text("id\n")
    with pytest.raises(ValueError):
        complete(tmp_path, entry, "x")
