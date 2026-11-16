"""Check presentation outputs against saved source bytes independently of table generators."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def rows(data: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(data.decode())))


def keys(name: str) -> dict[str, str]:
    text = (ROOT / "paper/generated" / name).read_text()
    return dict(re.findall(r"\\csname R@(.+?)\\endcsname\{(.*)\}$", text, re.M))


def main() -> None:
    source = rows((ROOT / "results/tables/resource_profile.csv").read_bytes())
    profile = {r["model"]: r for r in source if r["dataset"] == "isic2018"}
    final = json.loads((ROOT / "paper/generated/keys_final.json").read_text())
    a, b = profile["U-Net"], profile["BDS-Lite (deployed)"]
    ratio = 100 * (float(b["gflops_b1"]) / float(a["gflops_b1"]) - 1)
    if final["RF.gflops.pct"] != f"{ratio:.2f}":
        raise ValueError("GFLOPs percentage differs from saved FlopCounterMode values")
    for filename in ("tab_resource_text.tex", "tab_resource_supp.tex", "tab_resource_final.tex"):
        text = (ROOT / "paper/generated" / filename).read_text()
        for model in ("U-Net", "BDS-Lite (deployed)"):
            row = profile[model]
            if f"{int(row['params_deployed']):,} & {float(row['gflops_b1']):.3f}" not in text:
                raise ValueError(f"{filename}: resource row does not match saved profile")
    receipt = json.loads((ROOT / "results/gate_identity_unweighted_acdc/source.json").read_text())
    archive = ROOT / receipt["archive"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != receipt["archive_sha256"]:
        raise ValueError("Gate source archive changed")
    with tarfile.open(archive) as stream:
        data = stream.extractfile(receipt["member"]).read()
    if hashlib.sha256(data).hexdigest() != receipt["member_sha256"]:
        raise ValueError("Gate archive member differs from the source receipt")
    if (
        data
        != (ROOT / "results/gate_identity_unweighted_acdc/gate_identity_effects.csv").read_bytes()
    ):
        raise ValueError("Extracted gate result differs from archive bytes")
    gate = next(r for r in rows(data) if r["metric"] == "boundary_f1")
    values = keys("numbers_gate_addendum.tex")

    def signed(x: str) -> str:
        return ("$-$" if float(x) < 0 else "+") + f"{abs(float(x)):.6f}"

    if values["GIU.acdc.bf1"] != signed(gate["mean_delta_identity_minus_full"]):
        raise ValueError("Unweighted gate estimate differs from the archive")
    interval = f"[{signed(gate['ci95_low'])}, {signed(gate['ci95_high'])}]"
    if values["GIU.acdc.bf1.ci"] != interval:
        raise ValueError("Unweighted gate interval differs from the archive")
    old = next(
        r
        for r in rows((ROOT / "results/tables/gate_identity_effects.csv").read_bytes())
        if r["dataset"] == "acdc" and r["split"] == "test" and r["metric"] == "boundary_f1"
    )
    if final["GID.acdc.bf1"] != signed(old["mean_delta_identity_minus_full"]):
        raise ValueError("Class-weighted gate key was overwritten")
    literature = json.loads((ROOT / "paper/literature_context.json").read_text())["CTO"]["values"]
    published = (ROOT / "paper/generated/tab_published_text.tex").read_text()
    for dataset, record in literature.items():
        value = f"{record['DSC']:.4f}"
        hd = f"{record['HD']:.2f}" if "HD" in record else "--"
        if f"CTO \\citep{{Lin2025Rethinking}} & {value} & {hd}" not in published:
            raise ValueError(f"CTO {dataset}: exact source value absent")
    print("Saved-profile, archive-member, model-group and exact literature source checks passed")


if __name__ == "__main__":
    main()
