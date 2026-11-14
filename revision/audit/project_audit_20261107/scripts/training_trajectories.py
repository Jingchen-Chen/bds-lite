# ruff: noqa: E501, E741
import json

import pandas as pd
import yaml

pd.set_option(
    "display.width", 220, "display.max_columns", 30, "display.float_format", lambda x: f"{x:.4f}"
)
rows = []
for ds in ["acdc", "synapse", "isic2018"]:
    for s in (1, 2, 3):
        t = pd.read_csv(f"outputs/runs/{ds}_unet_seed{s}/train_log.csv")
        v = pd.read_csv(f"outputs/runs/{ds}_unet_seed{s}/val_metrics_epoch.csv")
        ep = [json.loads(l) for l in open(f"results/runs/{ds}_p1b_unet_seed{s}/metrics.jsonl")]
        rows.append(
            dict(
                ds=ds,
                seed=s,
                O_tr_dice50=t.seg_dice.iloc[49],
                O_tr_dice150=t.seg_dice.iloc[-1],
                O_tr_ce150=t.seg_ce.iloc[-1],
                P_tr_dice50=ep[49]["train"]["dice"],
                P_tr_dice150=ep[-1]["train"]["dice"],
                P_tr_ce150=ep[-1]["train"]["ce"],
                O_val_dice_loss150=v.val_loss_seg_dice.iloc[-1],
                P_val_dice_loss150=ep[-1]["val"]["dice"],
            )
        )
print(pd.DataFrame(rows).groupby("ds").mean(numeric_only=True).drop(columns="seed").to_string())
c = yaml.safe_load(open("outputs/runs/synapse_unet_seed1/config_resolved.yaml"))
print({k: c[k] for k in c if k not in ("model", "loss", "dataset", "metrics")})
print(open("outputs/runs/synapse_unet_seed1/config_source.yaml").read()[:400])
c2 = json.load(open("results/runs/synapse_p1b_unet_seed1/config.json"))
print(
    json.dumps({k: c2[k] for k in c2 if k in ("data", "trainer", "project", "augmentation")})[:700]
)
