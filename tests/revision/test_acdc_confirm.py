"""Unit tests for the pre-registered ACDC replication (preregistration/acdc_confirmation.md)."""

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from bds_lite.training import BDSLiteLoss, build_model
from bds_lite.training.losses import weights_from_config
from bds_lite.utils import seed_everything
from scripts.acdc_confirm import common
from scripts.acdc_confirm.analyze import decision, paired
from scripts.acdc_confirm.common import LAMBDA, SEEDS, make_config, registered_runs, validate
from scripts.acdc_confirm.official_test import stage1
from scripts.kd.common import sha
from scripts.kd_mc.common import KD_SPEC, template
from scripts.kd_mc.loss import KDCriterionMC
from scripts.kd_mc.register_mc import student_config

ROOT = Path(__file__).resolve().parents[2]
SHA = "0" * 64


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def test_seeds_are_fresh_and_alternate_by_arm():
    assert len(SEEDS) == 17 and SEEDS == tuple(range(201, 218))
    assert not set(SEEDS) & (set(range(1, 11)) | {100})
    order = registered_runs()
    assert len(order) == 34 and order[:2] == [("decoder", 201), ("kd_band", 201)]


@pytest.mark.parametrize("arm", ["decoder", "kd_band"])
def test_configs_differ_from_frozen_student_only_in_permitted_keys(arm):
    new = flat(make_config(arm, 205, SHA))
    frozen = flat(student_config("acdc", 1, SHA))
    changed = {k for k in new if new[k] != frozen.get(k)} | (set(frozen) - set(new))
    permitted = {"project.seed", "revision.purpose", "revision.protocol", "revision.arm"}
    if arm == "decoder":
        permitted.add("kd.lambda_kd")
    assert changed <= permitted
    assert new["kd.lambda_kd"] == LAMBDA[arm]
    for block in ["model", "loss", "trainer", "data", "dataset"]:
        assert make_config(arm, 205, SHA)[block] == template("acdc")[block]


def test_validate_rejects_any_other_change():
    config = make_config("kd_band", 203, SHA)
    validate(config, check_protocol_file=False)
    for path, value in [
        (("trainer", "batch_size"), 8),
        (("kd", "lambda_kd"), 0.5),
        (("kd", "band_kernel"), 5),
        (("project", "seed"), 5),
    ]:
        broken = copy.deepcopy(config)
        broken[path[0]][path[1]] = value
        with pytest.raises(ValueError):
            validate(broken, check_protocol_file=False)


def test_same_seed_gives_identical_initial_weights_in_both_arms():
    def init(config):
        seed_everything(config["project"]["seed"])
        return build_model(config).state_dict()

    a, b = init(make_config("decoder", 207, SHA)), init(make_config("kd_band", 207, SHA))
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_decoder_arm_criterion_is_the_decoder_only_loss_bitwise():
    config = make_config("decoder", 201, SHA)
    base = BDSLiteLoss(weights_from_config(config))
    outputs = dict(seg_logits=torch.randn(2, 4, 16, 16), boundary_logits=torch.randn(2, 1, 16, 16))
    batch = dict(
        mask=torch.randint(0, 4, (2, 16, 16)),
        boundary=(torch.rand(2, 1, 16, 16) > 0.8).float(),
        teacher_prob=torch.softmax(torch.randn(2, 4, 16, 16), 1),
    )
    reference, _ = base(outputs, batch)
    total, scalars = KDCriterionMC(base, config["kd"]["lambda_kd"])(outputs, batch)
    assert torch.equal(total, reference) and "kd" not in scalars


def test_launcher_changes_no_file_and_only_the_two_constants(monkeypatch):
    from scripts.acdc_confirm.launch import admit
    from scripts.kd_mc import common as kd_common

    frozen = [
        p
        for f in ["scripts/kd_mc", "scripts/kd", "scripts/revision"]
        for p in (ROOT / f).rglob("*.py")
    ]
    before = {p: sha(p) for p in frozen}
    monkeypatch.setattr(kd_common, "STUDENT_SEEDS", kd_common.STUDENT_SEEDS)
    monkeypatch.setattr(kd_common, "KD_SPEC", kd_common.KD_SPEC)
    config = make_config("decoder", 209, SHA)
    admit(config, check_protocol_file=False)
    assert kd_common.STUDENT_SEEDS == (209,)
    assert kd_common.KD_SPEC == dict(KD_SPEC, lambda_kd=0.0)
    kd_common.require_protocol(config)
    assert before == {p: sha(p) for p in frozen}  # fmt: skip


def test_supervisor_copy_differs_only_in_registered_substitutions():
    original = (ROOT / "scripts/kd_mc/supervise_mc.py").read_text()
    copy_ = (ROOT / "scripts/acdc_confirm/supervise.py").read_text()
    header_end = copy_.index("ACDC/Synapse copy of scripts/kd/supervise_kd.py")
    restored = '"""' + copy_[header_end:]
    for new, old in [
        (
            "from scripts.kd_mc.common import ROOT, RUNS, STOP\n\n"
            'WORKSPACE = ROOT / "revision/acdc_confirm"\n',
            "from scripts.kd_mc.common import ROOT, RUNS, STOP, WORKSPACE\n",
        ),
        ('"scripts/acdc_confirm/run_queue.py",', '"scripts/kd_mc/run_queue_mc.py",'),
        ('log("acdc_confirm_supervisor_started"', 'log("kd_mc_supervisor_started"'),
    ]:
        assert restored.count(new) == 1
        restored = restored.replace(new, old)
    assert restored == original


def test_stage1_clips_at_99th_percentile_then_z_scores():
    v = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
    out = stage1(v)
    clipped = np.clip(v, None, np.percentile(v, 99))
    assert np.allclose(out, (clipped - clipped.mean()) / clipped.std(), atol=1e-6)
    assert abs(float(out.mean())) < 1e-5 and abs(float(out.std()) - 1) < 1e-5


def test_decision_rules():
    assert decision(dict(low=0.001, high=0.03))["outcome"] == "success"
    assert decision(dict(low=0.012, high=0.03))["size"] == "above the smallest effect of interest"
    assert decision(dict(low=-0.02, high=0.009))["outcome"] == "failure"
    assert decision(dict(low=-0.002, high=0.02))["outcome"] == "inconclusive"


def test_paired_sensitivity_matches_manual():
    rng = np.random.default_rng(0)
    a = [
        {"boundary_f1": {f"u{u}": float(rng.normal(0.82, 0.03)) for u in range(5)}}
        for _ in range(4)
    ]
    b = [
        {"boundary_f1": {f"u{u}": float(rng.normal(0.80, 0.03)) for u in range(5)}}
        for _ in range(4)
    ]
    r = paired(a, b, "boundary_f1")
    ya = np.array([[x["boundary_f1"][f"u{u}"] for u in range(5)] for x in a])
    yb = np.array([[x["boundary_f1"][f"u{u}"] for u in range(5)] for x in b])
    d = ya.mean(1) - yb.mean(1)
    se_unit = (ya.mean(0) - yb.mean(0)).std(ddof=1) / math.sqrt(5)
    assert r["delta"] == pytest.approx(d.mean())
    assert r["se"] == pytest.approx(math.hypot(d.std(ddof=1) / 2, se_unit)) and r["df"] == 3


def test_estimator_reproduces_the_exploratory_acdc_contrast():
    import expansion_statistics as ex

    paths = {
        arm: [common.RUNS / n / "per_case_test.csv" for n in common.EXPLORATORY[arm]]
        for arm in ("kd_band", "decoder")
    }
    if not all(p.exists() for v in paths.values() for p in v):
        pytest.skip("exploratory per-case files are local-only")
    kd = [ex.unit_values(p, "acdc") for p in paths["kd_band"]]
    de = [ex.unit_values(p, "acdc") for p in paths["decoder"]]
    c = ex.contrast(kd, de, "boundary_f1")
    assert c["delta"] == pytest.approx(0.023873198328970946, abs=1e-12)
    assert c["p"] == pytest.approx(0.029263289569158298, abs=1e-12)


def test_failure_evidence_rules():
    from scripts.acdc_confirm.declare_failed import evidence

    assert evidence("... FloatingPointError: nonfinite training loss epoch=3", "diverged")
    assert not evidence("RuntimeError: CUDA error", "diverged")
    tb = "Traceback (most recent call last)\n"
    assert evidence(tb * 3, "cannot_complete") and not evidence(tb * 2, "cannot_complete")


def test_queue_skips_only_declared_failed_runs(monkeypatch):
    from scripts.acdc_confirm import run_queue

    monkeypatch.setattr(run_queue, "failed_runs", lambda: {"acdc_confirm_kd_band_seed205": {}})
    monkeypatch.setattr(run_queue, "_complete", lambda run, entry, h: False)
    assert run_queue.complete(None, {"run_id": "acdc_confirm_kd_band_seed205"}, "")
    assert not run_queue.complete(None, {"run_id": "acdc_confirm_decoder_seed205"}, "")
    assert run_queue.run_queue_mc.complete is run_queue.complete


def test_achieved_power_matches_the_planned_power():
    from scripts.acdc_confirm.analyze import achieved_power

    planned = achieved_power(17, 17, 50, reps=100000)
    assert planned["delta_0.015_sd_bound_90"] == pytest.approx(0.803, abs=0.01)
    assert achieved_power(15, 17, 50, reps=20000)["delta_0.015_sd_bound_90"] < 0.80
