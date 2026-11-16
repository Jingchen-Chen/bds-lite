"""Regenerate public presentation assets using saved evidence, without training or inference."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))


def main() -> None:
    from evidence.restore import restore

    restore(groups=("results", "protocols", "provenance"))

    import expansion_statistics as es
    import family_statistics as fs
    import make_comparison_table as comparison
    import make_distillation_assets as distillation
    import make_final_assets as final
    import make_gate_addendum_assets as gate
    import make_main_round3 as round3
    import make_review_assets as review
    import make_supplement_figures as figures
    import make_supplement_labels as labels
    import verify_final_assets as verify
    import verify_paper_sources as sources

    from analysis.revision import family_statistics as comparison_statistics
    from analysis.revision.paper_archive_inputs import CACHE, prepare, verify_config_weights

    prepare()
    verify_config_weights()
    # The statistical modules are imported only for their saved-file locators.
    # Their analysis entry points are never called and their source files stay unchanged.
    # Frozen generators import the locator both as a package and as a direct module.
    fs.RUNS = comparison_statistics.RUNS = es.RUNS = CACHE / "results/revision_runs"
    with tempfile.TemporaryDirectory(prefix="bds-paper-checks-") as directory:
        comparison.VALIDATION = Path(directory)
        distillation.main()
        final.main()
        review.main()
        gate.main()
        round3.main()
        labels.main()
        figures.main()
        sources.main()
        if verify.main():
            raise SystemExit("Paper asset validation failed")


if __name__ == "__main__":
    main()
