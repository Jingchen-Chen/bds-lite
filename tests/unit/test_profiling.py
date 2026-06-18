import torch

from bds_lite.evaluation import count_parameters, profile_model
from bds_lite.training import build_model


def test_count_parameters_matches_trainable_for_unfrozen_model() -> None:
    model = build_model(
        {
            "dataset": {"in_channels": 1, "num_classes": 2},
            "model": {"name": "unet", "base_channels": 4},
        }
    )

    params, trainable_params = count_parameters(model)

    assert params > 0
    assert trainable_params == params


def test_profile_model_reports_inference_resources() -> None:
    model = build_model(
        {
            "dataset": {"in_channels": 1, "num_classes": 2},
            "model": {"name": "unet", "base_channels": 4},
        }
    )

    report = profile_model(
        model,
        input_shape=(1, 1, 32, 32),
        device=torch.device("cpu"),
        warmup=0,
        iterations=1,
    )

    assert report.params > 0
    assert report.trainable_params == report.params
    assert report.fps > 0
    assert report.latency_ms > 0
    assert report.peak_memory_mb is None
    assert report.input_shape == (1, 1, 32, 32)
