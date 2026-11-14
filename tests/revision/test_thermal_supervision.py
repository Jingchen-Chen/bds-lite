"""Automatic continuation must not turn arbitrary failures into retries."""

import json

from analysis.revision import supervise_gradient_queue as supervisor


def test_only_current_run_with_recent_temperature_and_checkpoint_can_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    old = tmp_path / "results/revision_runs/old"
    old.mkdir(parents=True)
    (old / "gpu_telemetry.csv").write_text("timestamp, 90, rest\n")
    assert supervisor.thermal_evidence("run failed: new: 130") is None
    new = tmp_path / "results/revision_runs/new"
    new.mkdir()
    (new / "gpu_telemetry.csv").write_text("timestamp, 75, rest\n")
    (new / "last.pt").write_bytes(b"checkpoint")
    (new / "progress.json").write_text(json.dumps(dict(epoch=24)))
    assert supervisor.thermal_evidence("run failed: new: 130") is None
    (new / "gpu_telemetry.csv").write_text("timestamp, 85, rest\n")
    result = supervisor.thermal_evidence("run failed: new: 130")
    assert result["run"] == "new" and result["saved_progress_epoch"] == 24
    assert supervisor.thermal_evidence("run failed: new: 1") is None
    assert supervisor.thermal_evidence("nonfinite training loss") is None
