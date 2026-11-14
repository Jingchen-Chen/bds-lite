from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


@dataclass(frozen=True)
class ResourceProfile:
    params: int
    trainable_params: int
    flops: int | None
    fps: float
    latency_ms: float
    peak_memory_mb: float | None
    device: str
    batch_size: int
    input_shape: tuple[int, int, int, int]
    boundary_branch: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    params = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return params, trainable


def _forward(
    model: nn.Module,
    inputs: torch.Tensor,
    include_boundary: bool,
) -> Any:
    if include_boundary:
        try:
            return model(inputs, return_boundary=True)
        except TypeError:
            pass
    return model(inputs)


def estimate_flops(
    model: nn.Module,
    inputs: torch.Tensor,
    include_boundary: bool = False,
) -> int | None:
    flops: list[int] = []
    handles: list[RemovableHandle] = []

    def conv_hook(module: nn.Conv2d, _module_input: Any, module_output: torch.Tensor) -> None:
        output_elements = module_output.numel()
        kernel_ops = module.in_channels // module.groups
        kernel_ops *= module.kernel_size[0] * module.kernel_size[1]
        flops.append(int(output_elements * kernel_ops * 2))

    def conv_transpose_hook(
        module: nn.ConvTranspose2d,
        _module_input: Any,
        module_output: torch.Tensor,
    ) -> None:
        output_elements = module_output.numel()
        kernel_ops = module.in_channels // module.groups
        kernel_ops *= module.kernel_size[0] * module.kernel_size[1]
        flops.append(int(output_elements * kernel_ops * 2))

    def linear_hook(module: nn.Linear, _module_input: Any, module_output: torch.Tensor) -> None:
        flops.append(int(module_output.numel() * module.in_features * 2))

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.ConvTranspose2d):
            handles.append(module.register_forward_hook(conv_transpose_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    try:
        with torch.inference_mode():
            _forward(model, inputs, include_boundary)
            if inputs.is_cuda:
                torch.cuda.synchronize(inputs.device)
    finally:
        for handle in handles:
            handle.remove()

    total = sum(flops)
    return total if total > 0 else None


def measure_inference_speed(
    model: nn.Module,
    inputs: torch.Tensor,
    warmup: int = 10,
    iterations: int = 50,
    include_boundary: bool = False,
) -> tuple[float, float, float | None]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    device = inputs.device
    if inputs.is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(warmup):
            _forward(model, inputs, include_boundary)
        if inputs.is_cuda:
            torch.cuda.synchronize(device)

        started_at = time.perf_counter()
        for _ in range(iterations):
            _forward(model, inputs, include_boundary)
        if inputs.is_cuda:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started_at

    batch_size = int(inputs.shape[0])
    latency_ms = elapsed * 1000.0 / iterations
    fps = batch_size * iterations / elapsed
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if inputs.is_cuda else None
    return fps, latency_ms, peak_memory_mb


def profile_model(
    model: nn.Module,
    input_shape: tuple[int, int, int, int],
    device: torch.device,
    warmup: int = 10,
    iterations: int = 50,
    include_boundary: bool = False,
) -> ResourceProfile:
    model = model.to(device)
    model.eval()
    inputs = torch.zeros(input_shape, dtype=torch.float32, device=device)

    params, trainable_params = count_parameters(model)
    flops = estimate_flops(model, inputs, include_boundary=include_boundary)
    fps, latency_ms, peak_memory_mb = measure_inference_speed(
        model,
        inputs,
        warmup=warmup,
        iterations=iterations,
        include_boundary=include_boundary,
    )

    return ResourceProfile(
        params=params,
        trainable_params=trainable_params,
        flops=flops,
        fps=fps,
        latency_ms=latency_ms,
        peak_memory_mb=peak_memory_mb,
        device=str(device),
        batch_size=input_shape[0],
        input_shape=input_shape,
        boundary_branch=include_boundary,
    )
