"""Exact conditional flow-matching objective from CoVeTwin Eqs. (17)--(18).

The production decoder already uses the TRELLIS controlled flow model.  This
small standalone objective documents the paper equation explicitly and can be
used when retraining that decoder without changing the original trainer.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _broadcast_time(t: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if t.ndim != 1 or t.shape[0] != reference.shape[0]:
        raise ValueError("t must have shape (batch,) and match x0's batch size")
    return t.reshape(t.shape[0], *([1] * (reference.ndim - 1)))


def interpolate_flow_path(
    x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
) -> torch.Tensor:
    """Return ``x_t = (1-t) x_0 + t epsilon`` (paper Eq. 17)."""

    if x0.shape != noise.shape:
        raise ValueError("x0 and noise must have identical shapes")
    time = _broadcast_time(t, x0).to(device=x0.device, dtype=x0.dtype)
    return (1.0 - time) * x0 + time * noise


def target_velocity(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Return the constant path velocity ``epsilon - x_0``."""

    if x0.shape != noise.shape:
        raise ValueError("x0 and noise must have identical shapes")
    return noise - x0


def covetwin_flow_matching_loss(
    denoiser,
    x0: torch.Tensor,
    image_condition: Any,
    coarse_voxel_condition: Any,
    *,
    t: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    time_scale: float = 1000.0,
) -> dict[str, torch.Tensor]:
    """Evaluate Eq. (18) for a controlled TRELLIS-style denoiser.

    The expected model signature is
    ``denoiser(x_t, scaled_t, image_condition, coarse_voxel_condition)``.
    Returning the sampled tensors makes experiment logging and objective tests
    reproducible without changing the trained model.
    """

    if x0.ndim < 2:
        raise ValueError("x0 must include batch and feature/spatial dimensions")
    if noise is None:
        noise = torch.randn_like(x0)
    if t is None:
        t = torch.rand(x0.shape[0], device=x0.device, dtype=torch.float32)
    else:
        t = t.to(device=x0.device, dtype=torch.float32)
    xt = interpolate_flow_path(x0, t, noise)
    prediction = denoiser(
        xt,
        t.to(dtype=x0.dtype) * time_scale,
        image_condition,
        coarse_voxel_condition,
    )
    target = target_velocity(x0, noise)
    if prediction.shape != target.shape:
        raise ValueError(
            f"denoiser returned {tuple(prediction.shape)}, expected {tuple(target.shape)}"
        )
    loss = F.mse_loss(prediction, target)
    return {
        "loss": loss,
        "prediction": prediction,
        "target_velocity": target,
        "x_t": xt,
        "t": t,
    }
