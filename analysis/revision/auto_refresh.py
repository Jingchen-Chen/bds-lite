# ruff: noqa: E501
"""Regenerate statistics, tables, figures and PDFs whenever new results complete.

Polls every 10 minutes. When the set of completed runs/cohorts changes, it runs:
family_statistics -> gradient_analysis -> make_manuscript_assets -> compile manuscript,
supplement and response letter; error_bands and figure_qualitative only when the
active training run is not GSL (GSL uses CPU distance transforms; plan section 0.3).
Never starts, stops or signals training. If no supervisor process is alive while
registered runs are pending, it logs an ALERT line (no automatic restart, so the
supervisor's halt-on-unrecognised-failure safety is preserved).
Log: revision/auto_refresh.log (git-ignored).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "revision/auto_refresh.log"
PY = sys.executable
TECTONIC = Path.home() / ".local/bin/tectonic"


def log(message: str) -> None:
    with LOG.open("a") as stream:
        stream.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}\n")


def signature() -> tuple:
    done = sorted(
        str(p.parent.name) for p in (ROOT / "results/revision_runs").glob("*/final_metrics.json")
    )
    extra = sorted(
        str(p.relative_to(ROOT))
        for p in (ROOT / "revision/extra_cohorts").glob("*/*/test/provenance.json")
    )
    return tuple(done + extra)


def active_run() -> str:
    runs = list((ROOT / "results/revision_runs").glob("*/progress.json"))
    if not runs:
        return ""
    latest = max(runs, key=lambda p: p.stat().st_mtime)
    return latest.parent.name if time.time() - latest.stat().st_mtime < 900 else ""


def step(args: list[str], cwd: Path = ROOT) -> bool:
    env = os.environ | {"PYTHONPATH": f"{ROOT / 'src'}:{ROOT}", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        ["nice", "-n", "15", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=7200
    )
    if result.returncode:
        log(f"FAILED {' '.join(args[-2:])}: {result.stderr.strip().splitlines()[-1:]}")
        return False
    return True


def supervisor_alive() -> bool:
    out = subprocess.run(["pgrep", "-f", "supervise_"], capture_output=True, text=True).stdout
    return bool(out.strip())


def refresh() -> None:
    gsl_active = "_gsl_" in active_run()
    step([PY, "analysis/revision/family_statistics.py"])
    step([PY, "analysis/revision/gradient_analysis.py"])
    if not gsl_active:
        step([PY, "analysis/revision/error_bands.py"])
        step([PY, "analysis/revision/figure_qualitative.py"])
        step([PY, "analysis/revision/native_jaccard.py"])
    step([PY, "analysis/revision/make_manuscript_assets.py"])
    manuscript = ROOT / "revision/manuscript"
    for tex, cwd in [
        ("main.tex", manuscript),
        ("supplement.tex", manuscript),
        ("response_letter_v2.tex", ROOT / "revision"),
    ]:
        if TECTONIC.exists():
            step([str(TECTONIC), "-X", "compile", tex], cwd=cwd)
    log(f"refreshed (gsl_active={gsl_active})")


def main() -> None:
    last = None
    while True:
        try:
            current = signature()
            if current != last:
                log(f"change detected: {len(current)} completed items")
                refresh()
                last = current
            pending = [
                p
                for p in (ROOT / "configs/revision").glob("*_seed[123].yaml")
                if not (ROOT / "results/revision_runs" / p.stem / "final_metrics.json").exists()
            ]
            if pending and not supervisor_alive():
                log(f"ALERT: no supervisor alive; {len(pending)} configs without final metrics")
        except Exception as error:  # keep polling; errors are logged
            log(f"ERROR {type(error).__name__}: {error}")
        time.sleep(600)


if __name__ == "__main__":
    main()
