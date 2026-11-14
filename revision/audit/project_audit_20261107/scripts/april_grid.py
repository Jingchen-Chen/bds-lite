# ruff: noqa: E501, E741
import re

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option(
    "display.width",
    240,
    "display.max_columns",
    40,
    "display.max_rows",
    300,
    "display.float_format",
    lambda x: f"{x:.4f}",
)
SP = "revision/audit/project_audit_20261107/"
arms = [
    "unet",
    "unet_surface_loss",
    "bds_lite_boundary_only",
    "bds_lite_boundary_distill",
    "bds_lite_boundary_gate",
    "bds_lite_gate_surface",
    "bds_lite_full",
    "unet_deep_supervision",
]


def unit(ds, sid):
    if ds == "isic2018":
        return sid
    m = re.match(r"(patient\d+|case\d+)", str(sid))
    return m.group(1) if m else str(sid)


rows = []
for ds in ["isic2018", "acdc", "synapse"]:
    for a in arms:
        for s in (1, 2, 3):
            d = pd.read_csv(f"outputs/evaluations/{ds}_{a}_seed{s}_test.csv")
            d = d[d.class_id > 0].copy()
            both = (d.dsc >= 0.999999) & (d.hd95.fillna(0) == 0) & (d.assd.fillna(0) == 0)
            d = d[~both]
            img = d.groupby("sample_id")[["dsc", "iou", "boundary_f1", "hd95", "assd"]].mean()
            img["unit"] = [unit(ds, i) for i in img.index]
            u = img.groupby("unit")[["dsc", "iou", "boundary_f1", "hd95", "assd"]].mean()
            rows.append(
                dict(
                    ds=ds,
                    arm=a,
                    seed=s,
                    units=len(u),
                    dsc=u.dsc.mean(),
                    bf1=u.boundary_f1.mean(),
                    hd95=u.hd95.mean(),
                    assd=u.assd.mean(),
                )
            )
R = pd.DataFrame(rows)
R.to_csv(SP + "april_grid_runs_current_convention.csv", index=False)
print(R.groupby(["ds", "arm"])[["units", "dsc", "bf1", "hd95", "assd"]].mean().to_string())
print(R.groupby(["ds", "arm"])[["bf1"]].std().unstack(0).to_string())


def con(ds, a, b, m):
    x = R[(R.ds == ds) & (R.arm == a)][m].values
    y = R[(R.ds == ds) & (R.arm == b)][m].values
    d = x.mean() - y.mean()
    vx = x.var(ddof=1) / 3
    vy = y.var(ddof=1) / 3
    se = np.sqrt(vx + vy)
    df = se**4 / (vx**2 / 2 + vy**2 / 2)
    t = stats.t.ppf(0.975, df)
    return d, d - t * se, d + t * se, se


C = [
    ("bds_lite_boundary_only", "unet", "decoder - U-Net"),
    ("bds_lite_boundary_distill", "bds_lite_boundary_only", "distill(no gate) - decoder"),
    ("bds_lite_boundary_gate", "bds_lite_boundary_only", "gate(no distill) - decoder"),
    ("bds_lite_boundary_distill", "bds_lite_boundary_gate", "distill - gate"),
    ("bds_lite_full", "bds_lite_gate_surface", "distill within full"),
    ("bds_lite_full", "bds_lite_boundary_only", "full - decoder"),
    ("bds_lite_full", "unet", "full - U-Net"),
    ("unet_surface_loss", "unet", "surface - U-Net"),
    ("bds_lite_gate_surface", "bds_lite_boundary_gate", "surface | dec+gate"),
]
out = []
for a, b, l in C:
    r = {"contrast": l}
    ests = []
    ses = []
    for ds in ["isic2018", "acdc", "synapse"]:
        d, lo, hi, se = con(ds, a, b, "bf1")
        r[ds] = f"{d:+.4f} [{lo:+.4f},{hi:+.4f}]"
        ests.append(d)
        ses.append(se)
    w = 1 / np.array(ses) ** 2
    p = (w * np.array(ests)).sum() / w.sum()
    sp = 1 / np.sqrt(w.sum())
    r["pooled"] = f"{p:+.4f} [{p - 1.96 * sp:+.4f},{p + 1.96 * sp:+.4f}]"
    out.append(r)
print(pd.DataFrame(out).to_string())
