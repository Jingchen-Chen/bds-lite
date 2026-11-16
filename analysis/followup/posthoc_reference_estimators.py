"""POST HOC reference analyses of the run-level estimator (registration section 10.2).

For the Stage 3 primary contrast and, when available, the follow-up comparisons C1-C4 (test
Boundary F1 and DSC, patients 101-150), three estimates of the same Delta are set side by side:
1. the pre-specified estimator (expansion_statistics.contrast);
2. a crossed random-effects model fitted by the method of moments (balanced ANOVA):
       y_aru = mu_a + r_ar + p_u + (ap)_au + e_aru,
   runs r nested in arm, patients p crossed with arms, run-by-patient residual e. With one value
   per run and patient, within each arm E[MS_run] = s_e^2 + U s_r^2 and E[MS_res] = s_e^2, and the
   arm-difference profile d_u has variance 2 s_ap^2 + s_e^2 (1/n_A + 1/n_B). Hence
       Var(Delta) = MS_run,A/(n_A U) + MS_run,B/(n_B U) + max(0, s_d^2 - s_e^2 (1/n_A + 1/n_B))/U,
   with a Satterthwaite df. This removes the residual term that the pre-specified SE counts in
   both components (Supplementary S20), so it equals the S20 adjustment at k = 1 when untruncated;
3. a hierarchical bootstrap (10,000 draws, numpy default_rng(20261109)): patients resampled with
   replacement (one draw shared by both arms), runs resampled within each arm; percentile 95%
   interval. Resampling both levels is known to overstate the variance of a crossed design, so
   this interval is expected to be conservative.
Patients with a defined value in every run of both arms are used (as in the frozen estimator).

    PYTHONPATH=src:. .venv/bin/python analysis/followup/posthoc_reference_estimators.py
Outputs: results/followup/results/posthoc_reference_estimators.{csv,json}
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
sys.path.insert(0, str(ROOT / "analysis/followup"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import expansion_statistics as ex  # noqa: E402
from analyze import CONTEXT, FAMILY, arm_runs  # noqa: E402

OUT = ROOT / "results/followup/results"
DRAWS = 10_000


def matrices(a: list[dict], b: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray]:
    sign = -1.0 if metric in ("hd95", "assd") else 1.0
    units = sorted(set.intersection(*[set(r[metric]) for r in a + b]))
    ya = sign * np.array([[r[metric][u] for u in units] for r in a])
    yb = sign * np.array([[r[metric][u] for u in units] for r in b])
    return ya, yb


def anova_arm(y: np.ndarray) -> tuple[float, float, int, int]:
    """MS_run and MS_res of a runs x patients table without replication, with their df."""
    n, u = y.shape
    grand = y.mean()
    run_effect = y.mean(1) - grand
    patient_effect = y.mean(0) - grand
    residual = y - grand - run_effect[:, None] - patient_effect[None, :]
    ms_run = u * float((run_effect**2).sum()) / (n - 1)
    ms_res = float((residual**2).sum()) / ((n - 1) * (u - 1))
    return ms_run, ms_res, n - 1, (n - 1) * (u - 1)


def mixed(ya: np.ndarray, yb: np.ndarray) -> dict:
    na, u = ya.shape
    nb = yb.shape[0]
    ms_a, res_a, df_a, dfr_a = anova_arm(ya)
    ms_b, res_b, df_b, dfr_b = anova_arm(yb)
    s_e2 = (res_a * dfr_a + res_b * dfr_b) / (dfr_a + dfr_b)
    d = ya.mean(0) - yb.mean(0)
    s_d2 = float(d.var(ddof=1))
    interaction = s_d2 - s_e2 * (1 / na + 1 / nb)
    terms = [ms_a / (na * u), ms_b / (nb * u), max(0.0, interaction) / u]
    dfs = [df_a, df_b, u - 1]
    var = sum(terms)
    df = var**2 / sum(t**2 / f for t, f in zip(terms, dfs, strict=True) if t > 0)
    delta = float(ya.mean() - yb.mean())
    se = math.sqrt(var)
    t = stats.t.ppf(0.975, df)
    return dict(
        delta=delta,
        se=se,
        df=df,
        low=delta - t * se,
        high=delta + t * se,
        p=float(2 * stats.t.sf(abs(delta / se), df)),
        sigma_run2_a=max(0.0, (ms_a - res_a) / u),
        sigma_run2_b=max(0.0, (ms_b - res_b) / u),
        sigma_residual2=s_e2,
        sigma_arm_patient2=max(0.0, interaction / 2),
        interaction_truncated=interaction < 0,
    )


def bootstrap(ya: np.ndarray, yb: np.ndarray, rng: np.random.Generator) -> dict:
    na, u = ya.shape
    nb = yb.shape[0]
    out = np.empty(DRAWS)
    for i in range(DRAWS):
        cols = rng.integers(0, u, u)
        ra, rb = rng.integers(0, na, na), rng.integers(0, nb, nb)
        out[i] = ya[np.ix_(ra, cols)].mean() - yb[np.ix_(rb, cols)].mean()
    low, high = np.percentile(out, [2.5, 97.5])
    return dict(
        low=float(low),
        high=float(high),
        se=float(out.std(ddof=1)),
        share_le_zero=float((out <= 0).mean()),
    )


def main() -> None:
    arms = arm_runs()
    values = {arm: [ex.unit_values(p, "acdc") for p in r.values()] for arm, r in arms.items()}
    rows = []
    rng = np.random.default_rng(20261109)
    for cid, first, second, _question in CONTEXT + FAMILY:
        if first not in values or second not in values:
            continue
        for metric in ("boundary_f1", "dsc"):
            frozen = ex.contrast(values[first], values[second], metric)
            ya, yb = matrices(values[first], values[second], metric)
            m = mixed(ya, yb)
            b = bootstrap(ya, yb, rng)
            if abs(m["delta"] - frozen["delta"]) > 1e-12:
                raise ValueError("estimators disagree on the point estimate")
            rows.append(
                dict(
                    id=cid,
                    first=first,
                    second=second,
                    metric=metric,
                    delta=frozen["delta"],
                    prespecified_low=frozen["low"],
                    prespecified_high=frozen["high"],
                    prespecified_se=math.hypot(frozen["se_run"], frozen["se_unit"]),
                    prespecified_df=frozen["df"],
                    mixed_low=m["low"],
                    mixed_high=m["high"],
                    mixed_se=m["se"],
                    mixed_df=m["df"],
                    mixed_p=m["p"],
                    sigma_run2_a=m["sigma_run2_a"],
                    sigma_run2_b=m["sigma_run2_b"],
                    sigma_residual2=m["sigma_residual2"],
                    sigma_arm_patient2=m["sigma_arm_patient2"],
                    interaction_truncated=m["interaction_truncated"],
                    bootstrap_low=b["low"],
                    bootstrap_high=b["high"],
                    bootstrap_se=b["se"],
                    bootstrap_share_le_zero=b["share_le_zero"],
                    patients=ya.shape[1],
                )
            )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "posthoc_reference_estimators.json").write_text(
        json.dumps(dict(label="post hoc", draws=DRAWS, seed=20261109, rows=rows), indent=2) + "\n"
    )
    with (OUT / "posthoc_reference_estimators.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(
            f"{r['id']} {r['metric']:12s} {r['delta']:+.4f} pre [{r['prespecified_low']:+.4f}, "
            f"{r['prespecified_high']:+.4f}] mixed [{r['mixed_low']:+.4f}, {r['mixed_high']:+.4f}] "
            f"boot [{r['bootstrap_low']:+.4f}, {r['bootstrap_high']:+.4f}]"
        )


if __name__ == "__main__":
    main()
