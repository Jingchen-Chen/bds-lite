# ruff: noqa: E501, E741
import glob
import os

import pandas as pd

pd.set_option(
    "display.width", 220, "display.max_columns", 30, "display.float_format", lambda x: f"{x:.4f}"
)
print(os.listdir("revision/extra_cohorts/famO_unet/acdc_seed1/test")[:10])
for ds in ["acdc", "synapse"]:
    for arm, leg in [("famO_unet", "unet"), ("famO_full", "bds_lite_full")]:
        allc = []
        for s in (1, 2, 3):
            a = pd.read_csv(f"outputs/evaluations/{ds}_{leg}_seed{s}_test.csv")
            a = a[a.class_id > 0]
            f = glob.glob(f"revision/extra_cohorts/{arm}/{ds}_seed{s}/test/*.csv")
            b = pd.read_csv(
                [x for x in f if "per_case" in x or "per_unit" in x or "metrics" in x][0]
            )
            if s == 1 and arm == "famO_unet" and ds == "acdc":
                print(b.columns.tolist()[:12])
                print(a.columns.tolist())
            idc = "id" if "id" in b else "sample_id"
            lab = "label" if "label" in b else "class_id"
            m = a.merge(
                b, left_on=["sample_id", "class_id"], right_on=[idc, lab], suffixes=("_leg", "_cur")
            )
            m = m[~((m.dsc_leg >= 0.999999) & (m.hd95_leg.fillna(0) == 0))]
            allc.append(m)
        m = pd.concat(allc)
        print(
            ds,
            arm,
            "n",
            len(m),
            f"DSC maxdiff {(m.dsc_leg - m.dsc_cur).abs().max():.5f}",
            f"BF1 leg {m.boundary_f1_leg.mean():.4f} cur {m.boundary_f1_cur.mean():.4f} corr {m.boundary_f1_leg.corr(m.boundary_f1_cur):.4f} mean|d| {(m.boundary_f1_leg - m.boundary_f1_cur).abs().mean():.4f}",
            "HD ratio %.3f" % (m.hd95_cur.mean() / m.hd95_leg.mean()),
        )
