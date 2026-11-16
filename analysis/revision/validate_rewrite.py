# ruff: noqa: E501
"""Checks for the reconstructed manuscript, supplement and response letter.

1. Every \\R{key} used is defined; no red placeholder remains in the compiled PDFs.
2. Every citation and label resolves; every table and figure is referenced in the main text.
3. Key numbers are re-derived from raw per-case CSVs and manifests with code that shares nothing
   with family_statistics.py or make_rewrite_assets.py, then compared with the rendered keys.
4. Claim-discipline scan of main.tex (AGENTS.md phrase list), reported for manual review.
5. Figures referenced by the sources exist.

Output: results/validation/rewrite_validation.json. Exit code 1 on any failed check.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MS = ROOT / "paper"
LETTER = ROOT / "revision/response_letter_v2.tex"
RELEASE_SPLITS = Path("/home/jc/bds-lite-release-v2/splits/isic2018")


def keys() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("numbers.tex", "numbers_extra.tex"):
        text = (MS / "generated" / name).read_text()
        out.update(dict(re.findall(r"\\csname R@([^\\]+)\\endcsname\{(.*)\}", text)))
    return out


def value(text: str) -> float:
    return float(text.replace("$-$", "-").replace("+", ""))


def samples(cohort_path: str, metric: str) -> dict[str, float]:
    """Seed-averaged, label-macro value per sample from raw per-case CSVs, written afresh."""
    per: dict[tuple[str, str], list[float]] = defaultdict(list)
    for seed in (1, 2, 3):
        with (ROOT / cohort_path.format(seed=seed)).open() as stream:
            for row in csv.DictReader(stream):
                if row.get("empty_prediction") == "True" and row.get("empty_target") == "True":
                    continue
                if row.get(metric, "") in ("", "None", "nan"):
                    continue
                per[(row["id"], row.get("label", "1"))].append(float(row[metric]))
    sample: dict[str, list[float]] = defaultdict(list)
    for (sid, _), vals in per.items():
        sample[sid].append(float(np.mean(vals)))
    return {sid: float(np.mean(vals)) for sid, vals in sample.items()}


def delta(a: str, b: str, metric: str, cluster: str | None = None) -> float:
    """Paired on samples where the metric is defined in both arms, then averaged within units."""
    sa, sb = samples(a, metric), samples(b, metric)
    grouped: dict[str, list[float]] = defaultdict(list)
    for sid in sorted(set(sa) & set(sb)):
        unit = re.match(cluster, sid).group(1) if cluster else sid
        grouped[unit].append(sa[sid] - sb[sid])
    diff = np.array([np.mean(v) for v in grouped.values()])
    return float(diff.mean() * (-1 if metric in ("hd95", "assd") else 1))


def numeric_checks(k: dict[str, str]) -> list[dict]:
    so, sn = (
        "results/extra_cohorts/famO_full/{ds}_seed{{seed}}/test/per_case.csv",
        "results/extra_cohorts/famO_unet/{ds}_seed{{seed}}/test/per_case.csv",
    )
    m0 = "results/official_isic/test/U_seed{seed}/per_case.csv"
    m6 = "results/official_isic/test/F_seed{seed}/per_case.csv"
    run = "results/revision_runs/{arm}_seed{{seed}}/per_case_test.csv"
    syn = r"^(case\d+)"
    acdc = r"^(patient\d+)"
    cases = [
        (
            "P.isic2018.S6.S0.boundary_f1.delta",
            so.format(ds="isic2018"),
            sn.format(ds="isic2018"),
            "boundary_f1",
            None,
            1e-4,
        ),
        (
            "P.isic2018.S6.S0.hd95.delta",
            so.format(ds="isic2018"),
            sn.format(ds="isic2018"),
            "hd95",
            None,
            6e-3,
        ),
        ("P-matched.isic2018.M6.M0.boundary_f1.delta", m6, m0, "boundary_f1", None, 1e-4),
        (
            "A0.isic2018.M2.M0.boundary_f1.delta",
            run.format(arm="isic2018_m2_aux_boundary_only"),
            m0,
            "boundary_f1",
            None,
            1e-4,
        ),
        (
            "C.isic2018.GSL.M0.boundary_f1.delta",
            run.format(arm="isic2018_gsl"),
            m0,
            "boundary_f1",
            None,
            1e-4,
        ),
        (
            "C.isic2018.C4.M0.hd95.delta",
            run.format(arm="isic2018_c4_lbunet"),
            m0,
            "hd95",
            None,
            6e-3,
        ),
        ("P.acdc.S6.S0.hd95.delta", so.format(ds="acdc"), sn.format(ds="acdc"), "hd95", acdc, 6e-3),
        (
            "P.synapse.S6.S0.dsc.delta",
            so.format(ds="synapse"),
            sn.format(ds="synapse"),
            "dsc",
            syn,
            1e-4,
        ),
        (
            "C-vs-M6.acdc.C1.M6.hd95.delta",
            run.format(arm="acdc_c1_boundary_loss"),
            run.format(arm="acdc_m6_full_unweighted"),
            "hd95",
            acdc,
            6e-3,
        ),
    ]
    rows = []
    for key, a, b, metric, cluster, tol in cases:
        computed = delta(a, b, metric, cluster)
        rendered = value(k[key])
        rows.append(
            dict(
                key=key, rendered=rendered, recomputed=computed, ok=abs(rendered - computed) <= tol
            )
        )
    return rows


def partition_checks() -> list[dict]:
    def ids(obj: object) -> set[str]:
        return set(re.findall(r"ISIC_\d+", json.dumps(obj)))

    doc_val = ids(json.loads((RELEASE_SPLITS / "val.json").read_text()))
    seed1 = json.loads((ROOT / "splits/isic2018_split.json").read_text())
    train, val, test = ids(seed1["train"]), ids(seed1["val"]), ids(seed1["test"])
    return [
        dict(
            key="519 images in stale training set",
            rendered=358,
            recomputed=len(doc_val & train),
            ok=len(doc_val & train) == 358,
        ),
        dict(
            key="519 images in stale validation set",
            rendered=84,
            recomputed=len(doc_val & val),
            ok=len(doc_val & val) == 84,
        ),
        dict(
            key="519 images in neither",
            rendered=77,
            recomputed=len(doc_val & test),
            ok=len(doc_val & test) == 77,
        ),
    ]


def pdf_text(path: Path) -> str:
    return subprocess.run(
        ["pdftotext", str(path), "-"], capture_output=True, text=True, check=True
    ).stdout


def main() -> int:
    k = keys()
    report: dict = {}
    sources = {"main": MS / "main.tex", "supplement": MS / "supplement.tex", "letter": LETTER}
    used = {
        n: set(re.findall(r"\\R\{([^{}]+)\}", p.read_text())) - {"key"} for n, p in sources.items()
    }
    report["keys_used"] = {n: len(v) for n, v in used.items()}
    report["undefined_keys"] = sorted(x for v in used.values() for x in v if x not in k)

    pdfs = {
        "main": MS / "main.pdf",
        "supplement": MS / "supplement.pdf",
        "letter": ROOT / "revision/response_letter_v2.pdf",
    }
    report["pdf_placeholders"] = {}
    for name, path in pdfs.items():
        text = pdf_text(path)
        hits = re.findall(r"\[[A-Z][A-Za-z0-9-]*\.[a-z0-9_.-]+\]|PENDING|\?\?", text)
        report["pdf_placeholders"][name] = hits
        report.setdefault("pdf_pages", {})[name] = int(
            re.search(
                r"Pages:\s+(\d+)",
                subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout,
            ).group(1)
        )

    main_tex = sources["main"].read_text()
    labels = set(re.findall(r"\\label\{([^}]+)\}", main_tex))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", main_tex))
    report["unresolved_refs"] = sorted(refs - labels)
    report["unreferenced_floats"] = sorted(
        x for x in labels - refs if x.startswith(("tab:", "fig:"))
    )
    bib = set(re.findall(r"@\w+\{([^,]+),", (MS / "references_zotero.bib").read_text()))
    generated = (MS / "generated/tab_lit_compact.tex").read_text()
    cites = {
        c.strip()
        for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", main_tex + generated)
        for c in group.split(",")
    }
    report["missing_citations"] = sorted(cites - bib)
    report["citations_used"] = len(cites)

    figures = re.findall(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", main_tex + sources["supplement"].read_text()
    )
    missing = []
    for fig in figures:
        path = MS / fig
        if not any(
            path.with_suffix(s).exists() or Path(str(path) + s).exists()
            for s in ("", ".pdf", ".png")
        ):
            missing.append(fig)
    report["missing_figures"] = missing

    report["numeric_recomputation"] = numeric_checks(k) + partition_checks()

    body = main_tex[
        main_tex.index("\\begin{abstract}") : main_tex.index("\\section*{Acknowledgments}")
    ]
    phrases = [
        "novel",
        "state-of-the-art",
        "SOTA",
        "superior",
        "outperform",
        "best",
        "robust",
        "generaliz",
        "deployment-ready",
        "comprehensive",
        "nuanced",
        "taken together",
        "characterization",
        "P1",
        "P2",
        "P3",
        "P4",
    ]
    report["phrase_scan"] = {
        p: len(re.findall(rf"\b{p}", body, re.I if p[0] != "P" else 0)) for p in phrases
    }

    failed = bool(
        report["undefined_keys"]
        or any(report["pdf_placeholders"].values())
        or report["unresolved_refs"]
        or report["unreferenced_floats"]
        or report["missing_citations"]
        or report["missing_figures"]
        or not all(r["ok"] for r in report["numeric_recomputation"])
    )
    report["status"] = "FAILED" if failed else "PASSED"
    out = ROOT / "results/validation/rewrite_validation.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({x: report[x] for x in report if x != "numeric_recomputation"}, indent=1))
    for row in report["numeric_recomputation"]:
        print(
            f"{'ok ' if row['ok'] else 'BAD'} {row['key']}: rendered {row['rendered']} recomputed {row['recomputed']:.5f}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
