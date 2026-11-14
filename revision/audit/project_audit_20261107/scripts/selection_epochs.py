# ruff: noqa: E501, E741
import json

import numpy as np
import pandas as pd

pd.set_option(
    "display.width", 220, "display.max_columns", 30, "display.float_format", lambda x: f"{x:.4f}"
)
rows = []
for ds in ["acdc", "synapse", "isic2018"]:
    for s in (1, 2, 3):
        for arm, famO, famP in [
            ("U", f"outputs/runs/{ds}_unet_seed{s}", f"results/runs/{ds}_p1b_unet_seed{s}"),
            (
                "F",
                f"outputs/runs/{ds}_bds_lite_full_seed{s}",
                f"results/runs/{ds}_p1b_bdslite_full_current_seed{s}",
            ),
        ]:
            v = pd.read_csv(famO + "/val_metrics_epoch.csv")
            col = "val_dsc_foreground_macro"
            e = v.loc[v[col].idxmax()]
            ep = [json.loads(l) for l in open(famP + "/metrics.jsonl")]
            dsc = np.array([x["val"]["val_dsc"] for x in ep])
            rows.append(
                dict(
                    ds=ds,
                    arm=arm,
                    seed=s,
                    O_ep=int(e["epoch"]),
                    O_best=float(e[col]),
                    O_last=float(v[col].iloc[-1]),
                    P_ep=int(ep[int(dsc.argmax())]["epoch"]),
                    P_best=float(dsc.max()),
                    P_last=float(dsc[-1]),
                    P_n=len(ep),
                )
            )
R = pd.DataFrame(rows)
print(R.to_string())
print(
    R.groupby(["ds", "arm"])[["O_ep", "O_best", "O_last", "P_ep", "P_best", "P_last"]]
    .mean()
    .to_string()
)
