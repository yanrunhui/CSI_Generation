from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn


def cosine_beta_schedule(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal, clipped for numerical safety."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / timesteps) + offset) / (1 + offset) * math.pi / 2).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    selected = values.gather(0, timesteps)
    return selected.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    """Epsilon-prediction DDPM/DDIM for arbitrary-rank real tensors."""

    def __init__(self, timesteps: int = 1000, cosine_offset: float = 0.008) -> None:
        super().__init__()
        if timesteps < 2:
            raise ValueError("timesteps must be at least two")
        betas = cosine_beta_schedule(timesteps, cosine_offset)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
        posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        buffers = {
            "betas": betas,
            "alphas": alphas,
            "alpha_bar": alpha_bar,
            "alpha_bar_prev": alpha_bar_prev,
            "sqrt_alpha_bar": alpha_bar.sqrt(),
            "sqrt_one_minus_alpha_bar": (1.0 - alpha_bar).sqrt(),
            "sqrt_recip_alpha_bar": alpha_bar.rsqrt(),
            "sqrt_recipm1_alpha_bar": (1.0 / alpha_bar - 1.0).sqrt(),
            "posterior_variance": posterior_variance,
            "posterior_log_variance": posterior_variance.clamp_min(1e-20).log(),
            "posterior_mean_coef1": betas * alpha_bar_prev.sqrt() / (1.0 - alpha_bar),
            "posterior_mean_coef2": (1.0 - alpha_bar_prev) * alphas.sqrt() / (1.0 - alpha_bar),
        }
        for name, value in buffers.items():
            self.register_buffer(name, value.float())

    @property
    def timesteps(self) -> int:
        return int(self.betas.shape[0])

    def add_noise(
        self, clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean) if noise is None else noise
        noisy = _extract(self.sqrt_alpha_bar, timesteps, clean.shape) * clean
        noisy = noisy + _extract(self.sqrt_one_minus_alpha_bar, timesteps, clean.shape) * noise
        return noisy, noise

    def training_loss(
        self,
        clean: torch.Tensor,
        predict_noise: Callable[[torch.Tensor, torch.Tensor, bool], torch.Tensor],
        *,
        element_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = clean.shape[0]
        timestep = torch.randint(0, self.timesteps, (batch,), device=clean.device)
        noise = torch.randn_like(clean)
        if element_mask is not None:
            mask = element_mask.to(dtype=clean.dtype, device=clean.device)
            clean = clean * mask
            noise = noise * mask
        noisy, target = self.add_noise(clean, timestep, noise)
        prediction = predict_noise(noisy, timestep, False)
        error = (prediction - target).square()
        if element_mask is None:
            return error.flatten(1).mean(dim=1).mean()
        error = error * mask
        denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0)
        return (error.flatten(1).sum(dim=1) / denominator).mean()

    def _guided_prediction(
        self,
        predict_noise: Callable[[torch.Tensor, torch.Tensor, bool], torch.Tensor],
        sample: torch.Tensor,
        timestep: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        conditional = predict_noise(sample, timestep, False)
        if guidance_scale == 1.0:
            return conditional
        unconditional = predict_noise(sample, timestep, True)
        return unconditional + guidance_scale * (conditional - unconditional)

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        predict_noise: Callable[[torch.Tensor, torch.Tensor, bool], torch.Tensor],
        *,
        device: torch.device,
        sampler: str = "ddim",
        sampling_steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        clip_denoised: float | None = 5.0,
        element_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if clip_denoised is not None and clip_denoised <= 0:
            raise ValueError("clip_denoised must be positive or None")
        sample = torch.randn(shape, device=device, generator=generator)
        mask = None if element_mask is None else element_mask.to(device=device, dtype=sample.dtype)
        if mask is not None:
            sample = sample * mask
        if sampler == "ddpm":
            sequence = list(range(self.timesteps - 1, -1, -1))
            for index in sequence:
                timestep = torch.full((shape[0],), index, device=device, dtype=torch.long)
                predicted_noise = self._guided_prediction(
                    predict_noise, sample, timestep, guidance_scale
                )
                clean = _extract(self.sqrt_recip_alpha_bar, timestep, sample.shape) * sample
                clean = clean - _extract(self.sqrt_recipm1_alpha_bar, timestep, sample.shape) * predicted_noise
                if clip_denoised is not None:
                    clean = clean.clamp(-clip_denoised, clip_denoised)
                mean = _extract(self.posterior_mean_coef1, timestep, sample.shape) * clean
                mean = mean + _extract(self.posterior_mean_coef2, timestep, sample.shape) * sample
                if index > 0:
                    noise = torch.randn(shape, device=device, generator=generator)
                    sample = mean + (0.5 * _extract(self.posterior_log_variance, timestep, sample.shape)).exp() * noise
                else:
                    sample = mean
                if mask is not None:
                    sample = sample * mask
                if not torch.isfinite(sample).all():
                    raise FloatingPointError(
                        f"non-finite diffusion sample produced at timestep {index}"
                    )
            return sample
        if sampler != "ddim":
            raise ValueError("sampler must be 'ddpm' or 'ddim'")
        count = max(1, min(int(sampling_steps), self.timesteps))
        times = torch.linspace(self.timesteps - 1, 0, count, device=device).long().unique_consecutive()
        for position, index_tensor in enumerate(times):
            index = int(index_tensor.item())
            next_index = int(times[position + 1].item()) if position + 1 < len(times) else -1
            timestep = torch.full((shape[0],), index, device=device, dtype=torch.long)
            predicted_noise = self._guided_prediction(
                predict_noise, sample, timestep, guidance_scale
            )
            alpha = self.alpha_bar[index]
            alpha_next = self.alpha_bar[next_index] if next_index >= 0 else torch.tensor(1.0, device=device)
            clean = (sample - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
            if clip_denoised is not None:
                clean = clean.clamp(-clip_denoised, clip_denoised)
            sigma = eta * ((1.0 - alpha_next) / (1.0 - alpha) * (1.0 - alpha / alpha_next)).clamp_min(0.0).sqrt()
            direction = (1.0 - alpha_next - sigma.square()).clamp_min(0.0).sqrt() * predicted_noise
            noise = torch.randn(shape, device=device, generator=generator) if next_index >= 0 else torch.zeros_like(sample)
            sample = alpha_next.sqrt() * clean + direction + sigma * noise
            if mask is not None:
                sample = sample * mask
            if not torch.isfinite(sample).all():
                raise FloatingPointError(
                    f"non-finite diffusion sample produced at timestep {index}"
                )
        return sample

