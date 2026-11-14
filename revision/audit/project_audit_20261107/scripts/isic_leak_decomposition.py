# ruff: noqa: E501, E741
import json
import os

import numpy as np
import pandas as pd

pd.set_option(
    "display.width", 220, "display.max_columns", 30, "display.float_format", lambda x: f"{x:.4f}"
)
p = pd.read_csv("results/midterm_q2_rescue/analysis/per_case_metrics.csv")
p = p[p.dataset == "isic2018"].copy()
s = json.load(open("splits/isic2018_split.json"))


def ids(x):
    return set(os.path.splitext(os.path.basename(str(i)))[0] for i in x)


tr, va, te = ids(s["train"]), ids(s["val"]), ids(s["test"])
p["member"] = p.sample_id.map(
    lambda i: (
        "train(1815)"
        if i in tr
        else "val(389)"
        if i in va
        else "heldout(test390)"
        if i in te
        else "?"
    )
)
print(p.member.value_counts().to_dict())
rng = np.random.default_rng(20261107)
rows = []
for g, d in list(p.groupby("member")) + [("ALL519", p)]:
    r = dict(subset=g, n=len(d))
    for m, sg in (("dsc", 1), ("boundary_f1", 1), ("hd95", -1), ("assd", -1)):
        x = sg * (d[f"bds_lite_full_{m}"] - d[f"unet_{m}"]).dropna().values
        b = [rng.choice(x, len(x)).mean() for _ in range(4000)]
        r[m] = f"{x.mean():+.4f} [{np.percentile(b, 2.5):+.4f},{np.percentile(b, 97.5):+.4f}]"
        r["u_" + m] = d[f"unet_{m}"].mean()
    rows.append(r)
print(pd.DataFrame(rows)[["subset", "n", "dsc", "boundary_f1", "hd95", "assd"]].to_string())
print(pd.DataFrame(rows)[["subset", "u_dsc", "u_boundary_f1", "u_hd95", "u_assd"]].to_string())
# the 390 clean test, per-case CSV (legacy BF1), seed-level
A = {}
for a in ("unet", "bds_lite_full"):
    fr = []
    for k in (1, 2, 3):
        d = pd.read_csv(f"outputs/evaluations/isic2018_{a}_seed{k}_test.csv")
        d = d[d.class_id > 0].set_index("sample_id")[["dsc", "iou", "boundary_f1", "hd95", "assd"]]
        fr.append(d)
    A[a] = sum(fr) / 3
d = A["bds_lite_full"] - A["unet"]
print(
    "clean test390 seed-averaged paired delta",
    d.mean().round(4).to_dict(),
    "U means",
    A["unet"].mean().round(4).to_dict(),
)
b = [d.sample(len(d), replace=True, random_state=i).mean() for i in range(2000)]
B = pd.DataFrame(b)
print(B.quantile([0.025, 0.975]).round(4).to_string())
