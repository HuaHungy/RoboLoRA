"""Small DDPM/DDIM utilities shared by training and deployment."""

from __future__ import annotations

import numpy as np


def cosine_schedule(num_steps: int, offset: float = 0.008):
    if num_steps < 2:
        raise ValueError("diffusion num_steps must be >= 2")
    x = np.linspace(0, num_steps, num_steps + 1, dtype=np.float64)
    alpha_bar = np.cos(((x / num_steps) + offset) / (1 + offset) * np.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    betas = np.clip(betas, 1e-5, 0.999).astype(np.float32)
    alphas = (1.0 - betas).astype(np.float32)
    alpha_bars = np.cumprod(alphas).astype(np.float32)
    return betas, alphas, alpha_bars
