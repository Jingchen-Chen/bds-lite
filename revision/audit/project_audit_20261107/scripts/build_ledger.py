# ruff: noqa: E501, E741
"""Build a run-level experiment ledger from the filesystem. Stage 2 runs: existence only, no metric file is opened."""

import collections
import glob
import os
import re
import sys

import pandas as pd
import yaml

ROOT = "/home/jc/bds-lite"
os.chdir(ROOT)
OUT = sys.argv[1]


def q(f):
    return [r["run_id"] for r in yaml.safe_load(open("revision/" + f))["runs"]]


QA, QB, QT, QE, QC, QW = [
    q(x)
    for x in (
        "queue.yaml",
        "queue_b.yaml",
        "queue_tierb.yaml",
        "queue_expansion.yaml",
        "queue_confirmation.yaml",
        "queue_withdrawn.yaml",
    )
]
# duplicate groups
dup = pd.read_csv("revision/validation/checkpoint_state_duplicates.csv")


def key(p):
    return p.replace("/best.pt", "")


state = {key(r.path): r.state_sha256 for r in dup.itertuples()}
groups = collections.defaultdict(list)
for k, v in state.items():
    groups[v].append(k)
rows = []


def add(**k):
    base = dict(
        run_id="",
        path="",
        era="",
        family="",
        dataset="",
        arm="",
        seed="",
        epochs="",
        status="",
        trained="yes",
        duplicate_of="",
        train_partition="",
        selection_set="",
        evaluation="",
        evidence_class="",
        in_current_manuscript="no",
        notes="",
    )
    base.update(k)
    rows.append(base)


def ds_of(n):
    for d in ("isic2018", "acdc", "synapse"):
        if n.startswith(d):
            return d
    return ""


def nlines(p):
    try:
        with open(p) as f:
            return sum(1 for l in f if l.strip())
    except OSError:
        return ""


def dupof(path):
    s = state.get(path)
    if not s:
        return ""
    g = sorted(groups[s])
    # canonical = first by preference: outputs/runs main arms, then followup, then p1b full
    pref = [
        x
        for x in g
        if re.search(
            r"(bds_lite_boundary_only|bds_lite_gate_surface_seed|followup_20261018_v2|damped_surface(_class13_mild)?$|p1b_bdslite_full_current)",
            x,
        )
    ]
    c = (pref or g)[0]
    return "" if c == path else c


# --- results/runs (April single-seed exploration, p1, p1b, p2, p2b, e1)
for d in sorted(glob.glob("results/runs/*")):
    n = os.path.basename(d)
    ds = ds_of(n)
    ck = f"results/checkpoints/{n}"
    m = re.search(r"_seed(\d+)", n)
    seed = m.group(1) if m else "2026"
    ep = nlines(d + "/metrics.jsonl")
    ev = "ISIC val519 / ACDC+Synapse val at the time"
    if "_p1b_" in n:
        era, fam = "E3 May 2026", "p1b gate-free multi-seed"
        part = {"isic2018": "locked 2075/519"}.get(ds, "locked")
        evc = "retrained reference / replication"
        status = "complete"
        trained = (
            "no (evaluation variant of p1b full)"
            if ("gate_identity" in n or "gate_removed" in n)
            else "yes"
        )
    elif "_p1_" in n:
        era, fam = "E3 May 2026", "p1 gate pilot (single seed)"
        part = "locked"
        evc = "exploratory pilot"
        status = "complete"
        trained = "yes"
        if "gate_identity" in n or "gate_removed" in n or "full_current" in n:
            trained = "no (alias of April run or inference variant)"
    elif "_p2b_" in n:
        era, fam = "E3 May 2026", "p2b BDS-Lite(no gate)+GSL"
        part = "locked"
        evc = "comparator, legacy lineage, unused"
        status = "complete"
        trained = "yes"
    elif "_p2_" in n:
        era, fam = "E3 May 2026", "p2 recent-loss baselines (GSL, Boundary DoU)"
        part = "locked"
        evc = "comparator, legacy lineage, superseded"
        status = "complete"
        trained = "yes"
        if "rerun_step3" in n:
            status = "verification (10 epochs)"
            evc = "engineering check"
    elif "_e1_" in n:
        era, fam = "E3 May 2026", "E1 long-schedule GSL"
        part = "locked"
        evc = "invalid (GSL term not active); stopped"
        status = "aborted"
        trained = "yes"
    else:
        era, fam = "E1 Apr 26-27 2026", "single-seed exploration"
        part = "locked 2075/519" if ds == "isic2018" else "locked"
        evc = "exploratory, validation-driven tuning"
        status = "complete"
        trained = "yes"
    add(
        run_id=n,
        path=d,
        era=era,
        family=fam,
        dataset=ds,
        arm=re.sub(r"^(isic2018|acdc|synapse)_", "", re.sub(r"_seed\d+.*$", "", n)),
        seed=seed,
        epochs=ep,
        status=status,
        trained=trained,
        duplicate_of=dupof(ck),
        train_partition=part,
        selection_set="val",
        evaluation="validation (ISIC), val/test (ACDC, Synapse)",
        evidence_class=evc,
        in_current_manuscript="yes (M0/M6/R0/R6)"
        if ("p1b_unet" in n or "p1b_bdslite_full" in n)
        else "no",
    )
# --- outputs/runs April grid
for d in sorted(glob.glob("outputs/runs/*")):
    n = os.path.basename(d)
    ds = ds_of(n)
    m = re.search(r"_seed(\d+)$", n)
    seed = m.group(1) if m else ""
    ep = nlines(d + "/train_log.csv")
    ep = ep - 1 if isinstance(ep, int) and ep else ep
    smoke = n.startswith("t0") or n.startswith("t1")
    arm = re.sub(r"^(isic2018|acdc|synapse)_", "", re.sub(r"_seed\d+$", "", n))
    main = arm in ("unet", "bds_lite_full")
    add(
        run_id=n,
        path=d,
        era="E2 Apr 2026 (commit 5914d35)",
        family="smoke" if smoke else "family O reproducibility grid",
        dataset=ds,
        arm=arm,
        seed=seed,
        epochs=ep,
        status="smoke" if smoke else "complete",
        duplicate_of=dupof(d),
        train_partition={"isic2018": "split seed 1: 1815/389/390"}.get(ds, "locked"),
        selection_set="val389" if ds == "isic2018" else "val",
        evaluation="internal test390 (all arms); U-Net/full also val519 (contaminated) + official test"
        if ds == "isic2018"
        else "locked test",
        evidence_class="engineering"
        if smoke
        else ("submitted models" if main else "completed, unreleased trainer, never reported"),
        in_current_manuscript="yes (S0/S6)" if main else "no",
    )
# --- EGE
for d in sorted(glob.glob("outputs/checkpoints/*")):
    n = os.path.basename(d)
    ds = ds_of(n)
    seed = re.search(r"_seed(\d+)", n).group(1)
    add(
        run_id=n,
        path=d,
        era="E3 May 2026",
        family="EGE-UNet compact baseline",
        dataset=ds,
        arm="ege_unet",
        seed=seed,
        epochs=150,
        status="complete",
        train_partition="see config",
        selection_set="val",
        evaluation="val519(ISIC)/test",
        evidence_class="comparator, dropped before submission",
        notes="rows filtered out of the public release results",
    )
# --- Phase 16 GSL
for d in sorted(glob.glob("recovered/followup/results/gsl/runs/*")):
    n = os.path.basename(d)
    ds = [x for x in ("isic2018", "acdc", "synapse") if x in n][0]
    seed = n[-1]
    add(
        run_id=n,
        path=d,
        era="E4 Jun 2026",
        family="Phase 16 matched GSL",
        dataset=ds,
        arm="unet_gsl",
        seed=seed,
        epochs=nlines(d + "/metrics.jsonl"),
        status="complete; checkpoints absent",
        train_partition="locked 2075/519" if ds == "isic2018" else "locked",
        selection_set="val",
        evaluation="val519(ISIC, selection set)/test",
        evidence_class="submitted comparator; reproduced exactly by revision retrain",
        in_current_manuscript="via retrain",
    )
# --- October
for root, arms, fam in [
    ("followup_20261018_v2", "USBF", "ISIC U/S/B/F factorial"),
    ("pathway_controls_20261018", "GC", "ISIC pathway controls"),
    ("boundary_specificity_20261018", "D", "ISIC boundary-specificity control"),
    ("region_baseline_20261018", ["UA"], "ISIC D4 augmentation baseline"),
]:
    for a in arms:
        for s in (1, 2, 3):
            d = f"recovered/followup/results/{root}/isic2018_{a}_seed{s}"
            add(
                run_id=f"oct_isic2018_{a}_seed{s}",
                path=d,
                era="E5 Oct 2026",
                family=fam,
                dataset="isic2018",
                arm=a,
                seed=s,
                epochs=nlines(d + "/metrics.jsonl"),
                status="complete",
                duplicate_of=dupof(d),
                train_partition="locked 2075/519",
                selection_set="val519",
                evaluation="val519 (development); U/S/B/F later on official test",
                evidence_class="development study, independently re-inferred",
                in_current_manuscript="yes (reused as M0/M1/M3/M6)" if a in "USBF" else "no",
            )
for root, pat, fam in [
    ("acdc_index_baseline_20261019", "acdc_*_seed*", "ACDC index-grid region vs paper-GSL"),
    ("acdc_context_baseline_20261019", "acdc_context_seed*", "ACDC three-index context control"),
    ("acdc_phase_reduction_baseline_20261019", "acdc_*_seed*", "ACDC phase-reduction control"),
]:
    for d in sorted(glob.glob(f"recovered/followup/results/{root}/{pat}")):
        n = os.path.basename(d)
        s = n[-1]
        add(
            run_id="oct_" + n + ("_" + root.split("_")[1] if False else ""),
            path=d,
            era="E5 Oct 2026",
            family=fam,
            dataset="acdc (recovered 80-case dev data)",
            arm=re.sub(r"_seed\d$", "", n),
            seed=s,
            epochs=150,
            status="complete",
            train_partition="70 train / 10 val patients; different preparation",
            selection_set="val patients",
            evaluation="10 val patients only; test not accessed",
            evidence_class="development study, closed by frozen screen",
            in_current_manuscript="no",
        )
add(
    run_id="followup_20261017_U_seed1",
    path="recovered/followup/results/followup_20261017/isic2018_U_seed1",
    era="E5 Oct 2026",
    family="failed launch",
    dataset="isic2018",
    arm="U",
    seed=1,
    epochs=1,
    status="failed (AMP overflow check)",
    train_partition="locked",
    selection_set="",
    evaluation="none",
    evidence_class="failed attempt, fixed by amendment",
)
# --- November
stage = {}
for r in QA:
    stage[r] = "WP2/WP4 queue A"
for r in QB:
    stage[r] = "WP4 queue B"
for r in QT:
    stage[r] = "WP2/WP3 Tier B"
for r in QE:
    stage[r] = "Stage 1 expansion"
for r in QC:
    stage[r] = "Stage 2 Experiment C"
for r, fam in stage.items():
    d = f"results/revision_runs/{r}"
    ds = ds_of(r)
    m = re.search(r"_seed(\d+)$", r)
    seed = m.group(1)
    arm = re.sub(r"^(isic2018|acdc|synapse)_", "", re.sub(r"_seed\d+$", "", r))
    ex = os.path.isdir(d)
    done = os.path.exists(d + "/final_metrics.json")
    s2 = fam.startswith("Stage 2")
    status = (
        ("finished, BLIND" if done else ("running, BLIND" if ex else "queued"))
        if s2
        else ("complete" if done else "incomplete")
    )
    evc = (
        "confirmatory, blind"
        if s2
        else {"Stage 1 expansion": "frozen protocol, validated"}.get(fam, "registered, validated")
    )
    note = ""
    if r.startswith("acdc_m2_aux_boundary_only_seed"):
        evc = "superseded (class-weighted by mistake)"
        note = "replaced by *_unweighted"
    if r == "acdc_m1_sdf_only_seed1":
        evc = "sensitivity only (class-weighted)"
    add(
        run_id=r,
        path=d,
        era="E6 Nov 2026",
        family=fam,
        dataset=ds,
        arm=arm,
        seed=seed,
        epochs="" if s2 else 150,
        status=status,
        train_partition="locked 2075/519" if ds == "isic2018" else "locked",
        selection_set="val",
        evaluation="BLIND" if s2 else "official test1000 (ISIC) / locked test",
        evidence_class=evc,
        in_current_manuscript="no (blind)" if s2 else "yes",
        notes=note,
    )
for r in QW:
    add(
        run_id=r,
        path="",
        era="E6 Nov 2026",
        family="withdrawn registration",
        dataset=ds_of(r),
        arm=re.sub(r"^(isic2018|acdc|synapse)_", "", re.sub(r"_seed\d+$", "", r)),
        seed=re.search(r"_seed(\d+)$", r).group(1),
        status="never run",
        trained="no",
        evidence_class="withdrawn before start",
        notes="revision/queue_withdrawn.yaml",
    )
for d in sorted(glob.glob("results/revision_runs/gate_*")):
    n = os.path.basename(d)
    add(
        run_id=n,
        path=d,
        era="E6 Nov 2026",
        family="reproduction gate",
        dataset="",
        arm=n,
        seed=1,
        epochs=4,
        status="gate (no test evaluation)",
        trained="partial",
        evidence_class="engineering check",
    )
L = pd.DataFrame(rows)
L.to_csv(OUT, index=False)
print(len(L))
print(L.groupby(["era", "family", "status"]).size().to_string())
