# ruff: noqa: E501
"""Check the final-revision keys, tables and figure inputs against the evidence freeze.

The expected values below are typed from revision/BDS_LITE_EVIDENCE_FREEZE.md v1.0.0 (sections
named in each entry) and revision/plan/claim_ledger.md, not read from the generator, so that a
generator error cannot pass. Also checks that the per-seed values plotted in Fig. 3 reproduce the
arm means of the contrast files and that every generated table contains its ledger cells.
Exits 1 on any mismatch.

    PYTHONPATH=src:. .venv/bin/python analysis/revision/verify_final_assets.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "revision/manuscript/generated"
M = "$-$"

# key -> expected printed value (freeze section)
EXPECTED_KEYS = {
    "TE.prop.expected": "+0.0056",
    "TE.prop.upper": "+0.0044",
    "TE.prop.share": "18",
    # 10.4 / C13 / L-03, L-11, L-12
    "E3.E3.boundary_f1.delta": f"{M}0.0037",
    "E3.E3.boundary_f1.ci": f"[{M}0.0118, +0.0044]",
    "E3.E3.boundary_f1.p": "0.346",
    "E3.E3.boundary_f1.mean.a": "0.8432",
    "E3.E3.boundary_f1.mean.b": "0.8469",
    "E3.E3.dsc.ci": f"[{M}0.0142, +0.0000]",
    "E3.E3.iou.ci": f"[{M}0.0149, {M}0.0002]",
    "E3.E3.hd95.delta": "+0.17",
    "E3.E3.assd.delta": "+0.06",
    "E3.E3a.boundary_f1.delta": "+0.0672",
    "E3.E3a.boundary_f1.ci": "[+0.0500, +0.0844]",
    "E3.E3b.boundary_f1.delta": "+0.0457",
    "E3.E3a.dsc.mean.a": "0.7964",
    "E3.E3a.dsc.mean.b": "0.7349",
    # 10.1 / C03 / L-07
    "KDO.isic2018.boundary_f1.delta": "+0.0376",
    "KDO.isic2018.boundary_f1.ci": f"[{M}0.0565, +0.1317]",
    "KDO.isic2018.boundary_f1.holm": "0.496",
    "KDO.acdc.boundary_f1.delta": "+0.0239",
    "KDO.acdc.boundary_f1.ci": "[+0.0040, +0.0438]",
    "KDO.acdc.boundary_f1.holm": "0.088",
    "KDO.synapse.boundary_f1.delta": "+0.0150",
    "KDO.synapse.boundary_f1.ci": f"[{M}0.0423, +0.0724]",
    "KDO.synapse.boundary_f1.holm": "0.500",
    "KDO.isic2018.boundary_f1.nb": "10",
    # 9 / C01 / L-02, L-09, L-25
    "IN.nosurf.pooled.boundary_f1.est": "+0.0092",
    "IN.nosurf.pooled.boundary_f1.ci": f"[{M}0.0010, +0.0194]",
    "IN.surf.pooled.boundary_f1.est": "+0.0011",
    "IN.surf.pooled.boundary_f1.ci": f"[{M}0.0104, +0.0125]",
    "IN.nosurf.isic2018.boundary_f1.est": f"{M}0.0048",
    "IN.nosurf.acdc.boundary_f1.est": "+0.0106",
    "IN.nosurf.synapse.boundary_f1.est": "+0.0215",
    "IN.dec.pooled.boundary_f1.est": "+0.0159",
    "IN.dec.pooled.boundary_f1.ci": "[+0.0053, +0.0265]",
    "IN.s.nodec.pooled.boundary_f1.ci": f"[{M}0.0084, +0.0174]",
    "IN.s.dec.pooled.boundary_f1.ci": f"[{M}0.0152, +0.0073]",
    # 8.2 / C02 / L-04, L-08, L-09
    "TEN.full.decoder.boundary_f1.delta": f"{M}0.0321",
    "TEN.full.decoder.boundary_f1.ci": f"[{M}0.0518, {M}0.0124]",
    "TEN.full.decoder.boundary_f1.fresh": f"{M}0.0405",
    "TEN.full.decoder.boundary_f1.holm": "0.031",
    "TEN.gsl.decoder.boundary_f1.delta": f"{M}0.0308",
    "TEN.gsl.decoder.boundary_f1.holm": "0.014",
    "TEN.full.unet.boundary_f1.ci": f"[{M}0.0323, +0.0095]",
    "TEN.gsl.unet.boundary_f1.ci": f"[{M}0.0326, +0.0124]",
    "TEN.gsl.full.boundary_f1.ci": f"[{M}0.0216, +0.0242]",
    "TEN.decoder.unet.boundary_f1.delta": "+0.0207",
    "TEN.decoder.unet.boundary_f1.ci": "[+0.0014, +0.0399]",
    "TEN.decoder.unet.boundary_f1.holm": "0.260",
    "TEN.full.decoder.boundary_f1.mean.a": "0.3620",
    "TEN.full.decoder.boundary_f1.mean.b": "0.3940",
    "TENM.unet.hd95": "28.18",
    "TENM.decoder.hd95": "27.65",
    "TENM.full.hd95": "28.21",
    "TENM.gsl.hd95": "27.58",
    # 8.1 (matched absolute)
    "MT.acdc.M0.boundary_f1": "0.7828",
    "MT.acdc.M6.boundary_f1": "0.8006",
    "MT.synapse.GSL.dsc": "0.5325",
    "MT.isic2018.C4.hd95": "19.69",
    # 12 / C14 / L-14
    "RF.params.extra": "1,072",
    "RF.gpu.b1.pct": "8.3",
    "RF.gpu.b16.pct": "15.0",
    "RF.kernels.u": "104",
    "RF.kernels.g": "115",
    "RF.cache.sec": "27.3",
    "RF.teacher.hours": "0.243",
    "RF.s3.hours.kd": "7.63",
    "RF.s3.hours.dec": "7.52",
    # 11 gate identity / L-14
    "GID.isic2018.bf1": f"{M}0.001410",
    "GID.acdc.bf1": "+0.000438",
    "GID.synapse.bf1": f"{M}0.000094",
    "GID.maxabs": "0.0014",
    # 3 / L-22
    "FIN.runs.total": "237",
    "FIN.slot.lbunet": "9",
    "FIN.teachers": "4",
    "FIN.cf.abs": "0.0178",
    "FIN.cf.low": "0.0098",
    "FIN.cf.text": "0.0098 to 0.0258",
    "FIN.cf.overlap": "0.0107",
    "FIN.full.decoder.abs": "0.0321",
    "SPL.isic2018.train": "2,075",
    "SPL.acdc.train": "1,350",
    "SPL.synapse.test": "1,568",
    # 10.3 readings (C06: unresolved after Holm)
    "FX.C1.boundary_f1.reading": "unresolved",
    "FX.C4.boundary_f1.reading": "unresolved",
    "FX.C2.boundary_f1.reading": "gain, size unclear",
    "FX.C3.boundary_f1.reading": "gain, size unclear",
}

# keys from the existing generated files that the ledger uses (10.2, 10.3)
EXPECTED_EXISTING = {
    # freeze sections 3, 4, 5, 10.2, 12, 15 (values used in the final main text and letter)
    "numbers_restructure.tex": {
        "RS.KDC.ratio.deployed": "12.7",
        "RS.KDC.student.deployed": "1,926,532",
        "RS.s3.adj.ci": "[+0.0104, +0.0253]",
        "RS.tr.nosurf.adj.k05.ci": "[+0.0011, +0.0181]",
        "RS.tr.nosurf.adj.k15.ci": f"[{M}0.0010, +0.0195]",
    },
    "numbers_expansion.tex": {
        "XM.decoder.boundary_f1.sd": "0.0186",
        "XM.gsl.boundary_f1.sd": "0.0257",
    },
    "numbers_kd.tex": {
        "CF.delta": "+0.0178",
        "CF.n": "17",
        "CF.seed.first": "201",
        "CF.seed.last": "217",
        "CF.p": "$7.8\\times10^{-5}$",
        "CF.kd.sd": "0.0069",
        "CF.dec.sd": "0.0083",
        "CF.dsc.delta": "+0.0136",
        "CF.iou.delta": "+0.0183",
        "CF.assd.delta": "+0.39",
        "CF.hd95.delta": "+1.22",
        "A1.hist.delta": "+0.0067",
        "A1.hist.ci": f"[{M}0.0082, +0.0216]",
        "DS.isic2018.test.n": "1,000",
        "DS.acdc.train.units": "70",
        "DS.acdc.val.units": "10",
        "DS.acdc.test.units": "20",
        "DS.synapse.train.units": "14",
        "DS.synapse.val.units": "4",
        "DS.synapse.test.units": "12",
        "OT.patients": "50",
        "OT.slices": "1,076",
        "EXC.completed": "14",
        "EXC.registered": "70",
        "KDC.acdc.cache.slices": "1,350",
        "KDC.acdc.teacher.params": "24,436,804",
        "PAR.unet.m": "1.93",
        "PRE.gate.slices": "1,536",
        "PRE.gate.diff": "0.0148",
        "PRE.impact.bf1": "0.00011",
        "CF.ci": "[+0.0098, +0.0258]",
        "CF.low": "+0.0098",
        "CF.kd.mean": "0.7975",
        "CF.dec.mean": "0.7797",
        "CF.overlap": f"{M}0.0107",
        "CF.above": "17",
        "CF.dsc.ci": "[+0.0061, +0.0211]",
        "CF.iou.ci": "[+0.0105, +0.0262]",
        "CF.assd.ci": "[+0.05, +0.72]",
        "CF.hd95.ci": f"[{M}0.09, +2.54]",
        "A1.nosurf.boundary_f1.ci": f"[{M}0.0010, +0.0194]",
    },
    "numbers_followup.tex": {
        "PW.nosurf.acdc": "84",
        "FU.C2.boundary_f1.ci": "[+0.0034, +0.0177]",
        "FU.C3.boundary_f1.ci": "[+0.0037, +0.0200]",
        "FU.C2.boundary_f1.holm": "0.020",
        "FU.C1.boundary_f1.ci": f"[{M}0.0005, +0.0150]",
        "FU.C4.boundary_f1.ci": "[+0.0001, +0.0119]",
        "FU.C1.boundary_f1.holm": "0.095",
        "FU.S3.RV.boundary_f1.delta": "+0.0263",
        "FU.S3.myocardium.boundary_f1.delta": "+0.0193",
        "FU.S3.LV.boundary_f1.delta": "+0.0109",
        "FU.C1.hd95.ci": "[+0.34, +2.38]",
        "FU.C1.assd.ci": "[+0.10, +0.64]",
    },
}

# table file -> strings that must occur
EXPECTED_CELLS = {
    # S-column input since the primary/secondary relayout (Delta, low, high, p in separate
    # cells); same values as the earlier one-cell "Delta [CI]" strings.
    "tab_acdc_official_main.tex": [
        "& 17 & +0.0178 & +0.0098 & +0.0258 & {\\num{7.8e-5}} &",
        "& 17 & +0.0106 & +0.0034 & +0.0177 & 0.020 &",
        "& 17 & +0.0118 & +0.0037 & +0.0200 & 0.020 &",
        "& 17 & +0.0073 & -0.0005 & +0.0150 & 0.095 & \\multirow[t]{2}{=}{unresolved}",
        "& 17 & +0.0060 & +0.0001 & +0.0119 & 0.095 & \\multirow[t]{2}{=}{unresolved\\tnote{b}}",
        "& 10 & -0.0037 & -0.0118 & +0.0044 & 0.346 & \\multirow[t]{2}{=}{undetermined}",
    ],
    # Two-panel layout: A puts the CI on the line below the estimate; B uses S-column input
    # (Delta, low, high in separate cells). Same values as the earlier one-tabular layout.
    "tab_innetwork_main.tex": [
        "& +0.0092\\tnote{a} &",
        f"{{[{M}0.0010, +0.0194]}}",
        "& +0.0011 &",
        f"{{[{M}0.0104, +0.0125]}}",
        # fresh-seed interval added from expansion_contrasts.csv (fresh_low, fresh_high)
        "BDS-Lite${}-{}$D & -0.0321 & -0.0518 & -0.0124 & -0.0405 & -0.0668 & -0.0142 & 0.031 & \\textbf{confirmed}",
        "U-Net+GSL${}-{}$D & -0.0308 & -0.0522 & -0.0093 & -0.0459 & -0.0713 & -0.0204 & 0.014 & \\textbf{confirmed}",
        "D${}-{}$U & +0.0207 & +0.0014 & +0.0399 & +0.0226 & -0.0017 & +0.0470 & 0.260 & supported, not confirmed",
    ],
    # main-text resource and published-result tables (values of resource_profile.csv and of
    # panel B of tab_lit_main.tex, unchanged)
    "tab_resource_text.tex": [
        "U-Net & 1,927,042 & 15.970 & 1.66 & 36.75 & 129.9",
        "& 1,928,114 & 16.079 & 1.79 & 42.26 & 134.8",
        "& 1,927,042 & 15.970 & 1.66 & 36.84 & 131.1",
    ],
    "tab_published_text.tex": ["& 0.8946 &", "& 0.8954 &", "& 0.7748 & 31.69", "& 0.7913 & 21.55"],
    "tab_resource_final.tex": ["1,928,114", "1.66 & 36.75", "1.79 & 42.26", "129.9", "134.8"],
    # siunitx S-column input since the Table 3 relayout; the three ACDC SDs of 0.0080 were
    # recomputed from the per-run files (0.008009, 0.007989, 0.008010) and are distinct values.
    "tab_matched_main.tex": [
        "\\quad U & 0.7420 +- 0.0080",
        "\\quad D & 0.7583 +- 0.0080 & 0.7993 +- 0.0080",
    ],
    # adapted LB-UNet rows moved from main Table 3 to Supplementary S7
    "lb_adapted_scores.tex": ["$0.5098 \\pm 0.0255$", "$0.2137 \\pm 0.0330$"],
    # post hoc teacher evaluation (results/teacher_eval/REPORT.md values)
    "tab_s_teacher_scores.tex": ["& 0.8778 &", "& 0.7735 &", "24,436,804"],
    "tab_s_teacher_groups.tex": [
        "+0.0309 [+0.0158, +0.0454]",
        f"{M}0.0062 [{M}0.0160, +0.0037]",
        "0.8469 (0.8376--0.8575) & 0",
    ],
    "tab_s_e3_all.tex": ["+0.0672 [+0.0500, +0.0844]", "+0.0457 [+0.0309, +0.0604]"],
    # Table 1: runs per arm (seeds) recounted from the training records; the two confirmatory
    # studies (2 and 4). Replaces the run strings of the earlier seven-column layout.
    "tab_studies.tex": [
        "& 3 (1--3) &",
        "& 10 (1--10)\\tnote{b} & ISIC2018 test & \\textbf{Confirmatory}",
        "& 17 (201--217) & \\R{OT.patients} official ACDC test patients & \\textbf{Confirmatory}",
        "& 17; 10 & as study 4 & Exploratory",
        "seeds 4--10 are new, and the decision rule was fixed before they were trained",
        "10 runs each (seeds 201--210)",
    ],
    # Supplementary S15 design count (moved from the Table 1 note): 237 = 249 listed - 12 reused
    "tab_s_design_count.tex": [
        "including 9 runs of the adapted LB-UNet & 36 & 0 & 36",
        "& 40 & 12 & 28",
        "Total & & 249 & 12 & 237",
    ],
}


def keys_from(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if line.startswith("\\expandafter\\def\\csname R@"):
            key = line.split("R@", 1)[1].split("\\endcsname", 1)[0]
            out[key] = line.split("\\endcsname{", 1)[1][:-1]
    return out


def read_csv(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    errors: list[str] = []
    final = json.loads((GEN / "keys_final.json").read_text())
    for key, want in EXPECTED_KEYS.items():
        got = final.get(key)
        if got != want:
            errors.append(f"{key}: expected {want!r}, got {got!r}")
    for name, expected in EXPECTED_EXISTING.items():
        keys = keys_from(GEN / name)
        for key, want in expected.items():
            if keys.get(key) != want:
                errors.append(f"{name}:{key}: expected {want!r}, got {keys.get(key)!r}")
    for name, cells in EXPECTED_CELLS.items():
        text = (GEN / name).read_text()
        for cell in cells:
            if cell not in text:
                errors.append(f"{name}: missing {cell!r}")
    # Fig. 3 inputs: per-seed run means reproduce the contrast-file arm means
    fu = read_csv(ROOT / "revision/followup/results/contrasts.csv")
    seeds_fu = read_csv(ROOT / "revision/followup/results/per_seed.csv")
    e3 = read_csv(ROOT / "revision/evidence_freeze/final_e3_analysis.csv")
    seeds_e3 = read_csv(ROOT / "revision/evidence_freeze/final_e3_per_seed.csv")
    arm_rows = {
        "decoder": ("C2", "arm_b_mean", fu),
        "kd_full": ("C2", "arm_a_mean", fu),
        "kd_band_peer": ("C3", "arm_a_mean", fu),
        "kd_band": ("C1", "arm_a_mean", fu),
    }
    for arm, (cid, column, rows) in arm_rows.items():
        values = [
            float(r["value"]) for r in seeds_fu if r["arm"] == arm and r["metric"] == "boundary_f1"
        ]
        want = float(
            next(r[column] for r in rows if r["id"] == cid and r["metric"] == "boundary_f1")
        )
        if len(values) != 17 or abs(np.mean(values) - want) > 1e-10:
            errors.append(
                f"Fig. 3 per-seed {arm}: n={len(values)}, mean {np.mean(values)} vs {want}"
            )
    for arm, column in (("kd_band_aug", "arm_a_mean"), ("decoder_aug", "arm_b_mean")):
        values = [
            float(r["value"]) for r in seeds_e3 if r["arm"] == arm and r["metric"] == "boundary_f1"
        ]
        want = float(
            next(r[column] for r in e3 if r["id"] == "E3" and r["metric"] == "boundary_f1")
        )
        if len(values) != 10 or abs(np.mean(values) - want) > 1e-10:
            errors.append(
                f"Fig. 3 per-seed {arm}: n={len(values)}, mean {np.mean(values)} vs {want}"
            )
    for error in errors:
        print("MISMATCH", error)
    checked = (
        len(EXPECTED_KEYS)
        + sum(map(len, EXPECTED_EXISTING.values()))
        + sum(map(len, EXPECTED_CELLS.values()))
        + 6
    )
    print(f"{checked} checks, {len(errors)} mismatches")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
