"""Losses corresponding to equations (2)-(4), (6), (11), (15)-(17)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import math


def infonce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """In-batch InfoNCE over two views of the same samples (Eq. 2)."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if z1.ndim != 2 or z1.shape != z2.shape or z1.size(0) == 0:
        raise ValueError("InfoNCE requires matching non-empty [batch, features] tensors")
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


def stance_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Ignore unknown stances, including batches with no stance annotations."""
    valid = labels != -100
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], labels[valid])


def l2_regularization(parameters, lambda_reg: float) -> torch.Tensor:
    total = None
    for p in parameters:
        term = p.pow(2).sum()
        total = term if total is None else total + term
    if total is None:
        return torch.tensor(0.0)
    return lambda_reg * total


def total_loss(
    cl: torch.Tensor,
    isr: torch.Tensor,
    cls: torch.Tensor,
    reg: torch.Tensor,
    alpha_cl: float,
    alpha_isr: float,
    alpha_cls: float,
) -> torch.Tensor:
    """Eq. (16): alpha1 L_CL + alpha2 L_ISR + alpha3 L_cls + L_reg."""
    return alpha_cl * cl + alpha_isr * isr + alpha_cls * cls + reg
