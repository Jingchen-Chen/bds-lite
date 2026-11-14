# ruff: noqa: E501, E741
# Independent re-derivation of the frozen Stage 1 ISIC contrasts from raw per-image files.
import numpy as np
import pandas as pd
import yaml
from scipy import stats

s2 = set(r["run_id"] for r in yaml.safe_load(open("revision/queue_confirmation.yaml"))["runs"])


def load(arm, seed):
    if arm in ("U", "F") and seed <= 3:
        p = f"revision/official_isic/test/{arm}_seed{seed}/per_case.csv"
    else:
        rid = {"U": "m0_unet", "F": "m6_full", "D": "m2_aux_boundary_only", "G": "gsl"}[arm]
        rid = f"isic2018_{rid}_seed{seed}"
        assert rid not in s2 and seed <= 10
        p = f"results/revision_runs/{rid}/per_case_test.csv"
    d = pd.read_csv(p)
    d = d[d["label"] > 0] if "label" in d else d
    return d.groupby("id")[["dsc", "iou", "boundary_f1", "hd95", "assd"]].mean()


arms = {a: {s: load(a, s) for s in range(1, 11)} for a in "UDFG"}


def con(a, b, m, seeds, sign=1):
    A = pd.concat([arms[a][s][m] for s in seeds], axis=1).mean(axis=1)
    B = pd.concat([arms[b][s][m] for s in seeds], axis=1).mean(axis=1)
    ra = np.array([arms[a][s][m].mean() for s in seeds])
    rb = np.array([arms[b][s][m].mean() for s in seeds])
    d = sign * (ra.mean() - rb.mean())
    va = ra.var(ddof=1) / len(ra)
    vb = rb.var(ddof=1) / len(rb)
    diff = (A - B).dropna()
    su = diff.std(ddof=1) / np.sqrt(len(diff))
    se = np.sqrt(va + vb + su**2)
    df = (va + vb) ** 2 / (va**2 / (len(ra) - 1) + vb**2 / (len(rb) - 1))
    t = stats.t.ppf(0.975, df)
    p = 2 * stats.t.sf(abs(d / se), df)
    return d, d - t * se, d + t * se, p


F = pd.read_csv("revision/tables/families/expansion_contrasts.csv")
mp = {"decoder": "D", "unet": "U", "full": "F", "gsl": "G"}
mx = 0
for _, r in F[F.metric.isin(["boundary_f1", "dsc", "hd95"])].iterrows():
    sg = -1 if r.metric == "hd95" else 1
    a, b = mp[r["first"]], mp[r["second"]]
    d, lo, hi, p = con(a, b, r.metric, range(1, 11), sg)
    d2, lo2, hi2, p2 = con(a, b, r.metric, range(4, 11), sg)
    e = max(abs(d - r.all_delta), abs(d2 - r.fresh_delta))
    mx = max(mx, e)
    print(r.metric, r["first"], r["second"], round(e, 5)) if e > 1e-6 else None
    if r.metric == "boundary_f1":
        print(
            f"{r['first']}-{r['second']}: all {d:+.4f} [{lo:+.4f},{hi:+.4f}] frozen {r.all_delta:+.4f}[{r.all_low:+.4f},{r.all_high:+.4f}] | fresh {d2:+.4f} p={p2:.4f} frozen {r.fresh_delta:+.4f} p={r.fresh_p:.4f}"
        )
print("max |delta| discrepancy over 18 contrast rows x 2 sets:", mx)
