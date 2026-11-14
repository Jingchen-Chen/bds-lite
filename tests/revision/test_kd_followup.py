"""Unit tests for the exploratory follow-up controls (revision/followup_registration.md)."""

import copy

import pytest
import torch

from scripts.acdc_confirm.common import make_config as stage3_config
from scripts.kd_followup import launch
from scripts.kd_followup.common import (
    ARMS,
    E3_SEEDS,
    PEER_CACHE_RUN,
    SEEDS,
    make_config,
    queue_runs,
    validate,
)
from scripts.kd_mc import common as kd_common
from scripts.kd_mc import loss as kd_loss
from scripts.kd_mc import train_mc

SHA = "0" * 64


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


@pytest.fixture
def restore_modules():
    saved = dict(
        seeds=kd_common.STUDENT_SEEDS,
        spec=dict(kd_common.KD_SPEC),
        band=kd_loss.class_band,
        require=train_mc.require_protocol,
        cache=train_mc.verified_cache,
        dataset=train_mc.TeacherCacheDatasetMC,
    )
    yield
    kd_common.STUDENT_SEEDS = saved["seeds"]
    kd_common.KD_SPEC = saved["spec"]
    kd_loss.class_band = saved["band"]
    train_mc.require_protocol = saved["require"]
    train_mc.verified_cache = saved["cache"]
    train_mc.TeacherCacheDatasetMC = saved["dataset"]


def test_queue_order_and_seeds():
    e1e2 = queue_runs("e1e2")
    assert [r[0] for r in e1e2[:17]] == ["kd_full"] * 17
    assert e1e2[17:19] == [("e2_teacher", 100, "train"), ("e2_teacher", 100, "cache")]
    assert [r[1] for r in e1e2[19:]] == list(SEEDS) and len(e1e2) == 36
    assert len(queue_runs("e3")) == 20 and E3_SEEDS == tuple(range(201, 211))


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_configs_differ_from_stage3_only_in_permitted_keys(arm):
    seed = 100 if arm == "e2_teacher" else 205
    parent = ARMS[arm][0]
    new, old = flat(make_config(arm, seed, SHA)), flat(stage3_config(parent, seed, SHA))
    changed = {k for k in new if new[k] != old.get(k)} | (set(old) - set(new))
    permitted = {
        "revision.purpose",
        "revision.protocol",
        "revision.arm",
        "revision.train_augmentation",
        "kd.mode",
        "kd.cache_run",
    }
    assert changed <= permitted
    for block in ["model", "loss", "trainer", "data", "dataset"]:
        assert make_config(arm, seed, SHA)[block] == stage3_config(parent, seed, SHA)[block]
    assert new["kd.lambda_kd"] == (0.0 if parent == "decoder" else 1.0)


def test_validate_rejects_other_changes_and_seeds():
    config = make_config("kd_full", 203, SHA)
    validate(config, check_registration_file=False)
    for path, value in [
        (("trainer", "lr"), 3e-4),
        (("kd", "lambda_kd"), 0.5),
        (("kd", "clamp"), 0),
    ]:
        bad = copy.deepcopy(config)
        bad[path[0]][path[1]] = value
        with pytest.raises(ValueError):
            validate(bad, check_registration_file=False)
    with pytest.raises(ValueError):
        validate(make_config("kd_full", 1, SHA), check_registration_file=False)
    with pytest.raises(ValueError):
        validate(make_config("kd_band_aug", 211, SHA), check_registration_file=False)


def test_kd_full_is_the_same_soft_cross_entropy_over_every_pixel(restore_modules):
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 16, 16)
    labels = torch.zeros(2, 16, 16, dtype=torch.long)
    labels[:, 4:12, 4:12] = 1  # one object: the band is its two-pixel rim, not the whole image
    teacher = 0.7 * torch.nn.functional.one_hot(labels, 4).permute(0, 3, 1, 2) + 0.075
    assert 0 < kd_loss.class_band(teacher).mean() < 0.5
    band_value = kd_loss.kd_loss_mc(logits, teacher)
    launch.admit(make_config("kd_full", 201, SHA), check_registration_file=False)
    assert kd_loss.class_band is launch.full_region
    full_value = kd_loss.kd_loss_mc(logits, teacher)
    p_s = torch.softmax(logits, 1).clamp(1e-6, 1 - 1e-6)
    p_t = teacher.clamp(1e-6, 1 - 1e-6)
    expected = (-(p_t * torch.log(p_s)).sum(1)).sum() / (2 * 16 * 16 + 1e-6)
    assert torch.allclose(full_value, expected, rtol=1e-6)
    assert not torch.allclose(full_value, band_value)
    train_mc.require_protocol(make_config("kd_full", 201, SHA))  # frozen checks still pass


def test_band_arm_keeps_the_frozen_band(restore_modules):
    original = kd_loss.class_band
    launch.admit(make_config("kd_band_peer", 207, SHA), check_registration_file=False)
    assert kd_loss.class_band is original
    assert train_mc.verified_cache is launch.peer_cache
    assert make_config("kd_band_peer", 207, SHA)["kd"]["cache_run"] == PEER_CACHE_RUN
    assert kd_common.STUDENT_SEEDS == (207,)


def test_d4_moves_inputs_targets_and_teacher_together():
    torch.manual_seed(1)
    mask = torch.randint(0, 4, (24, 24))
    teacher = torch.nn.functional.one_hot(mask, 4).permute(2, 0, 1).float()
    for code in range(8):
        m, t = launch.d4(mask, code), launch.d4(teacher, code)
        assert torch.equal(t.argmax(0), m)
        band_after = kd_loss.class_band(t[None])[0]
        band_before = launch.d4(kd_loss.class_band(teacher[None])[0], code)
        assert torch.equal(band_after, band_before)
    assert len({launch.d4(mask, c).numpy().tobytes() for c in range(8)}) == 8


def test_augmented_dataset_class_is_picklable_for_forkserver_workers(restore_modules):
    import pickle

    launch.admit(make_config("kd_band_aug", 201, SHA), check_registration_file=False)
    assert train_mc.TeacherCacheDatasetMC is launch.AugmentedTeacherCacheDatasetMC
    assert (
        pickle.loads(pickle.dumps(train_mc.TeacherCacheDatasetMC)) is train_mc.TeacherCacheDatasetMC
    )
