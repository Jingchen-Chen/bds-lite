from pathlib import Path

from scripts.prepare_cross_dataset_report import build_report, metric_status


def test_metric_status_handles_direction_and_ties() -> None:
    assert metric_status(0.01) == "improved"
    assert metric_status(-0.01) == "worse"
    assert metric_status(-0.01, lower_is_better=True) == "improved"
    assert metric_status(0.0001) == "tie"


def test_build_report_marks_partial_support_for_distance_failure(tmp_path: Path) -> None:
    cross_dataset = tmp_path / "cross.csv"
    cost_ablation = tmp_path / "cost.csv"
    output_csv = tmp_path / "report.csv"
    output_md = tmp_path / "report.md"

    cross_dataset.write_text(
        "\n".join(
            [
                "dataset,method_variant,baseline_dsc,bds_dsc,delta_dsc,baseline_iou,bds_iou,"
                "delta_iou,baseline_hd95,bds_hd95,delta_hd95,baseline_assd,bds_assd,"
                "delta_assd,baseline_boundary_f1,bds_boundary_f1,delta_boundary_f1,"
                "claim_status,notes",
                "Synapse,test,0.8,0.81,0.01,0.7,0.71,0.01,10,11,1,3,4,1,"
                "0.8,0.82,0.02,mixed,Distance failure",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cost_ablation.write_text(
        "\n".join(
            [
                "variant,boundary_branch,active_params,flops,fps,latency_ms,peak_memory_mb,"
                "active_params_over_baseline_pct,flops_over_baseline_pct,"
                "latency_over_disabled_pct",
                "unet_baseline,False,100,1000,10,1,10,,,",
                "bdslite_boundary_disabled,False,100.5,1005,10,1,10,0.5,0.5,",
                "bdslite_boundary_retained,True,120,1300,8,1.4,12,20,30,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_report(cross_dataset, cost_ablation, output_csv, output_md)

    assert rows[0]["report_decision"] == "partial_support"
    assert rows[0]["hd95_status"] == "worse"
    assert output_csv.exists()
    assert "Ready for a full cross-dataset boundary-distance claim: no" in output_md.read_text(
        encoding="utf-8"
    )


def test_build_report_requires_all_datasets_for_full_claim(tmp_path: Path) -> None:
    cross_dataset = tmp_path / "cross.csv"
    cost_ablation = tmp_path / "cost.csv"
    output_csv = tmp_path / "report.csv"
    output_md = tmp_path / "report.md"

    header = (
        "dataset,method_variant,baseline_dsc,bds_dsc,delta_dsc,baseline_iou,bds_iou,"
        "delta_iou,baseline_hd95,bds_hd95,delta_hd95,baseline_assd,bds_assd,"
        "delta_assd,baseline_boundary_f1,bds_boundary_f1,delta_boundary_f1,"
        "claim_status,notes"
    )
    cross_dataset.write_text(
        "\n".join(
            [
                header,
                "A,test,0.8,0.81,0.01,0.7,0.71,0.01,10,9,-1,3,2,-1,"
                "0.8,0.82,0.02,positive,Supported",
                "B,test,0.8,0.81,0.01,0.7,0.71,0.01,10,9,-1,3,2,-1,"
                "0.8,0.82,0.02,positive,Supported",
                "C,test,0.8,0.81,0.01,0.7,0.71,0.01,10,11,1,3,4,1,0.8,0.82,0.02,mixed,Partial",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cost_ablation.write_text(
        "\n".join(
            [
                "variant,boundary_branch,active_params,flops,fps,latency_ms,peak_memory_mb,"
                "active_params_over_baseline_pct,flops_over_baseline_pct,"
                "latency_over_disabled_pct",
                "unet_baseline,False,100,1000,10,1,10,,,",
                "bdslite_boundary_disabled,False,100.5,1005,10,1,10,0.5,0.5,",
                "bdslite_boundary_retained,True,120,1300,8,1.4,12,20,30,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    build_report(cross_dataset, cost_ablation, output_csv, output_md)

    assert "Ready for a full cross-dataset boundary-distance claim: no" in output_md.read_text(
        encoding="utf-8"
    )
