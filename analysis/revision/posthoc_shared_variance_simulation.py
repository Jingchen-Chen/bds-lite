# ruff: noqa: E501
"""Simulation check for posthoc_shared_variance.py (post hoc, 2026-11-09).

Under y = run + unit + arm-by-unit + residual with common test units, compare the empirical
Var(Delta) with the mean of the recipe-level SE^2 and of SE^2 minus the shared run-by-unit term.
Parameters are illustrative scales, not fitted values.
"""

import numpy as np

rng = np.random.default_rng(1)


def sim(nA, nB, U, sr, sg, se, reps=20000):
    d = []
    se2 = []
    sh = []
    for _ in range(reps):
        ups = rng.normal(0, 1, U)
        gA = rng.normal(0, sg, U)
        gB = rng.normal(0, sg, U)
        yA = ups + gA + rng.normal(0, sr, (nA, 1)) + rng.normal(0, se, (nA, U))
        yB = ups + gB + rng.normal(0, sr, (nB, 1)) + rng.normal(0, se, (nB, U))
        ra, rb = yA.mean(1), yB.mean(1)
        d.append(ra.mean() - rb.mean())
        s = ra.var(ddof=1) / nA + rb.var(ddof=1) / nB + np.var(yA.mean(0) - yB.mean(0), ddof=1) / U
        se2.append(s)

        def ms(y):
            r = y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + y.mean()
            return (r**2).sum() / ((y.shape[0] - 1) * (y.shape[1] - 1))

        sh.append(ms(yA) / (nA * U) + ms(yB) / (nB * U))
    true = np.var(d)
    theo = sr**2 * (1 / nA + 1 / nB) + 2 * sg**2 / U + se**2 * (1 / nA + 1 / nB) / U
    print(
        f"n={nA},{nB} U={U}: Var(Δ) emp {true:.3e} theory {theo:.3e} | mean SE² {np.mean(se2):.3e} | mean SE²-shared {np.mean(np.array(se2) - np.array(sh)):.3e}"
    )


sim(3, 3, 20, 0.003, 0.005, 0.03)
sim(17, 17, 50, 0.008, 0.01, 0.03)
sim(3, 3, 12, 0.01, 0.01, 0.05)
