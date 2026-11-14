# ruff: noqa: E501, E741, UP031

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option(
    "display.width",
    230,
    "display.max_columns",
    40,
    "display.max_rows",
    300,
    "display.float_format",
    lambda x: f"{x:.4f}",
)
SP = "revision/audit/project_audit_20261107/"


def welch(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    d = x.mean() - y.mean()
    vx = x.var(ddof=1) / len(x)
    vy = y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy)
    df = se**4 / (vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1))
    t = stats.t.ppf(0.975, df)
    p = 2 * stats.t.sf(abs(d / se), df)
    return d, d - t * se, d + t * se, p


# ---- ISIC val/test ten-run
R = pd.read_csv(SP + "isic_val_test_runs.csv")
U = pd.concat([R[(R.src == "oct") & (R.arm == "U")], R[(R.src == "nov") & (R.arm == "m0_unet")]])
Fu = pd.concat([R[(R.src == "oct") & (R.arm == "F")], R[(R.src == "nov") & (R.arm == "m6_full")]])
D = R[(R.src == "nov") & (R.arm == "m2_aux_boundary_only")]
G = R[(R.src == "nov") & (R.arm == "gsl")]
print("n", len(U), len(Fu), len(D), len(G))
print("--- ISIC VALIDATION (519), 10 runs per arm, run-level Welch")
for n, a, b in [
    ("decoder-U", D, U),
    ("full-U", Fu, U),
    ("full-decoder", Fu, D),
    ("gsl-U", G, U),
    ("gsl-full", G, Fu),
    ("gsl-decoder", G, D),
]:
    print(
        n.ljust(14),
        "BF1 {:.4f} [{:.4f},{:.4f}] p={:.3f}".format(*welch(a.valbf1, b.valbf1)),
        " DSC {:.4f} [{:.4f},{:.4f}] p={:.3f}".format(*welch(a.valdsc, b.valdsc)),
        " HD95 {:.3f} [{:.3f},{:.3f}] p={:.3f}".format(*welch(a.valhd, b.valhd)),
    )
print(
    f"run SD val BF1  U {U.valbf1.std():.4f} D {D.valbf1.std():.4f} F {Fu.valbf1.std():.4f} G {G.valbf1.std():.4f}"
)
# val vs test across runs with both (seeds 4-10 + decoder/gsl 1-10)
B = R[
    (R.src == "nov")
    & R.testbf1.notna()
    & R.arm.isin(["m0_unet", "m2_aux_boundary_only", "m6_full", "gsl"])
]
print(
    "corr val BF1 vs test BF1 across %d runs: %.3f ; within-arm:"
    % (len(B), B.valbf1.corr(B.testbf1)),
    B.groupby("arm").apply(lambda g: round(g.valbf1.corr(g.testbf1), 3)).to_dict(),
)
# Official-test SDs
print("test BF1 sd", B.groupby("arm").testbf1.std().round(4).to_dict())
# ---- October dev contrasts run level
O = R[R.src == "oct"]
print("--- OCT dev ISIC val, run-level (3 v 3)")


def g(a):
    return O[O.arm == a].valbf1


for n, a, b in [
    ("B-C distill|dec,gate", "B", "C"),
    ("C-G boundary|gate", "C", "G"),
    ("G-U gate/proj", "G", "U"),
    ("D-G match w/o bnd", "D", "G"),
    ("B-D bnd|match", "B", "D"),
    ("B-U", "B", "U"),
    ("F-B surface|path", "F", "B"),
    ("S-U surface", "S", "U"),
    ("F-S path|surface", "F", "S"),
    ("F-U", "F", "U"),
    ("UA-U D4 aug", "UA", "U"),
]:
    print(n.ljust(24), "{:.4f} [{:.4f},{:.4f}] p={:.3f}".format(*welch(g(a), g(b))))
M2 = R[(R.src == "nov") & (R.arm == "m2_aux_boundary_only") & (R.seed <= 3)].valbf1
print(
    "M2 val s1-3 %.4f ; M2-C(no gate vs gate, both no distill) %.4f [%.4f,%.4f]"
    % ((M2.mean(),) + welch(M2, g("C"))[:3])
)
print("B - M2(s1-3) on val {:.4f} [{:.4f},{:.4f}]".format(*welch(g("B"), M2)[:3]))
