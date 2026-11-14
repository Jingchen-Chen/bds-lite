from pathlib import Path

from scripts.prepare_final_ablation_report import build_report, read_mean_metrics


def test_read_mean_metrics_extracts_mean_row(tmp_path: Path) -> None:
    path = tmp_path / "eval.csv"
    path.write_text(
        "label,dsc,iou,hd95,assd,boundary_f1\n1,0.8,0.7,10,3,0.6\nmean,0.9,0.8,5,2,0.7\n",
        encoding="utf-8",
    )

    assert read_mean_metrics(path) == {
        "dsc": 0.9,
        "iou": 0.8,
        "hd95": 5.0,
        "assd": 2.0,
        "boundary_f1": 0.7,
    }


def test_build_report_writes_status_rows(tmp_path: Path) -> None:
    month4 = tmp_path / "month4.csv"
    cost = tmp_path / "cost.csv"
    unet = tmp_path / "unet.csv"
    bds = tmp_path / "bds.csv"
    unet_lcc = tmp_path / "unet_lcc.csv"
    bds_lcc = tmp_path / "bds_lcc.csv"
    output_csv = tmp_path / "status.csv"
    output_md = tmp_path / "status.md"

    month4.write_text(
        "dataset,best_variant,claim_status,dsc_delta,iou_delta,hd95_delta,assd_delta,"
        "boundary_f1_delta,inference_flops_overhead_pct,deploy_params_overhead_pct,decision\n"
        "A,full,positive,0.1,0.1,-1,-1,0.1,0.5,0.5,accept\n"
        "B,full,mixed,0.1,0.1,1,1,0.1,0.5,0.5,mixed\n",
        encoding="utf-8",
    )
    cost.write_text(
        "variant,boundary_branch,active_params,flops,fps,latency_ms,peak_memory_mb,"
        "active_params_over_baseline_pct,flops_over_baseline_pct,latency_over_disabled_pct\n"
        "unet_baseline,False,100,1000,10,1,10,,,\n"
        "bdslite_boundary_disabled,False,100.5,1005,10,1,10,0.5,0.5,\n"
        "bdslite_boundary_retained,True,120,1300,8,1.4,12,20,30,40\n",
        encoding="utf-8",
    )
    unet.write_text(
        "label,dsc,iou,hd95,assd,boundary_f1\nmean,0.8,0.7,10,3,0.6\n",
        encoding="utf-8",
    )
    bds.write_text(
        "label,dsc,iou,hd95,assd,boundary_f1\nmean,0.81,0.71,11,4,0.62\n",
        encoding="utf-8",
    )
    unet_lcc.write_text(
        "label,dsc,iou,hd95,assd,boundary_f1\nmean,0.8,0.7,9,2,0.6\n",
        encoding="utf-8",
    )
    bds_lcc.write_text(
        "label,dsc,iou,hd95,assd,boundary_f1\nmean,0.81,0.71,10,3,0.62\n",
        encoding="utf-8",
    )

    rows = build_report(month4, cost, unet, bds, unet_lcc, bds_lcc, output_csv, output_md)

    assert [row["category"] for row in rows] == [
        "module_ablation",
        "loss_ablation",
        "inference_cost_ablation",
        "targeted_synapse_postprocess_ablation",
        "paper_grade_repeats",
        "synapse_current_delta",
    ]
    assert output_csv.exists()
    assert "Final Ablation Status Report" in output_md.read_text(encoding="utf-8")
