# ruff: noqa: E501
"""WP5a numerical audit of the separate-versus-combined gradient residual.

At every probe the diagnostic run computes, on the shared encoder parameters, the gradient of the
segmentation loss (g_s) and, in separate backward passes, of each auxiliary term (SDF, boundary,
distillation) and of their weighted sum (aux). Linearity implies the scalar identity
    <g_aux, g_s>/|g_s| = sum_k <g_k, g_s>/|g_s|,   i.e.  |g_aux| cos_aux = sum_k |g_k| cos_k.
The logged values satisfy it only approximately (max absolute residual ~1e-5, DEVIATIONS.md),
because independent FP32 backward passes with cuDNN TF32 convolutions accumulate in different
orders. This audit quantifies the residual relative to the quantities that are reported:

  abs_residual        |sum_k |g_k| cos_k - |g_aux| cos_aux|
  rel_residual        abs_residual / max(sum_k ||g_k| cos_k|, |g_aux| |cos_aux|)
  implied_cos_error   abs_residual / |g_aux|  (upper bound on the error of the reported aux cosine
                      if the whole residual were attributed to it)

Acceptance (fixed here, before inspecting the per-dataset summary): the residual is immaterial if
the 99th percentile of implied_cos_error is below 0.005 (half the last reported digit of the
two-decimal cosines) and its maximum is below the smallest between-dataset difference of the
reported mean aux cosines. Probes with undefined cosines (zero norm) are skipped.

Output: revision/validation/gradient_residual_audit.csv (per dataset) and .json (verdict).
"""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = ("sdf", "boundary", "distill")


def main() -> None:
    rows, means = [], {}
    for dataset in ("isic2018", "acdc", "synapse"):
        probes = json.loads(
            (
                ROOT / f"results/revision_runs/{dataset}_wp5_full_seed1/gradient_probes.json"
            ).read_text()
        )
        abs_res, rel_res, cos_err, aux_cos = [], [], [], []
        for p in probes:
            if p["aux_cosine"] is None or p["aux_norm"] == 0:
                continue
            parts = [p[f"{k}_norm"] * (p[f"{k}_cosine"] or 0.0) for k in COMPONENTS]
            aux = p["aux_norm"] * p["aux_cosine"]
            r = abs(sum(parts) - aux)
            scale = max(sum(abs(x) for x in parts), abs(aux))
            abs_res.append(r)
            rel_res.append(r / scale if scale > 0 else 0.0)
            cos_err.append(r / p["aux_norm"])
            aux_cos.append(p["aux_cosine"])
        cos_err_arr = np.array(cos_err)
        means[dataset] = float(np.mean(aux_cos))
        rows.append(
            dict(
                dataset=dataset,
                probes=len(probes),
                probes_checked=len(abs_res),
                abs_residual_max=float(np.max(abs_res)),
                rel_residual_median=float(np.median(rel_res)),
                rel_residual_max=float(np.max(rel_res)),
                implied_cos_error_p99=float(np.quantile(cos_err_arr, 0.99)),
                implied_cos_error_max=float(cos_err_arr.max()),
                aux_cosine_mean=means[dataset],
            )
        )
    min_gap = min(abs(means[a] - means[b]) for a, b in combinations(means, 2))
    p99 = max(r["implied_cos_error_p99"] for r in rows)
    worst = max(r["implied_cos_error_max"] for r in rows)
    verdict = dict(
        criterion="p99 implied cosine error < 0.005 and max < smallest between-dataset gap of mean aux cosine",
        p99_implied_cos_error=p99,
        max_implied_cos_error=worst,
        smallest_between_dataset_gap=min_gap,
        immaterial=bool(p99 < 0.005 and worst < min_gap),
        note="Exact separate/combined equivalence is not claimed; FP32 with cuDNN TF32 convolutions.",
    )
    out = ROOT / "revision/validation/gradient_residual_audit"
    with out.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    out.with_suffix(".json").write_text(
        json.dumps(dict(datasets=rows, verdict=verdict), indent=2) + "\n"
    )
    for r in rows:
        print({k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(verdict)


if __name__ == "__main__":
    main()
