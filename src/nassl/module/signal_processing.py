# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Tuple

import torch


def mix_batch_random_snr(
    wave_s: torch.Tensor, wave_n: torch.Tensor, snr_range: Tuple[float, float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # wave_s, wave_n: [batch_size, length]
    assert wave_s.ndim == 2 and wave_n.ndim == 2
    assert wave_s.shape == wave_n.shape
    batch_size = wave_s.shape[0]
    snr_db = torch.empty(batch_size, device=wave_s.device, dtype=wave_s.dtype).uniform_(snr_range[0], snr_range[1])
    wave_x = mix_batch_snr(wave_s=wave_s, wave_n=wave_n, snr=snr_db)
    return wave_x, snr_db


def mix_batch_snr(wave_s: torch.Tensor, wave_n: torch.Tensor, snr: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    # wave_s, wave_n: [batch_size, length]
    assert wave_s.ndim == 2 and wave_n.ndim == 2
    assert wave_s.shape == wave_n.shape
    assert snr.ndim == 1 and snr.shape[0] == wave_s.shape[0]
    eps = 1.0e-8
    power_s = torch.mean(wave_s**2, dim=1)  # [batch_size]
    power_n = torch.mean(wave_n**2, dim=1)  # [batch_size]
    target_noise_power = power_s / (10.0 ** (snr / 10.0))
    scale = torch.sqrt(target_noise_power / (power_n + eps))  # [batch_size]
    wave_x = wave_s + scale[:, None] * wave_n
    if clamp:
        wave_x = torch.clamp(wave_x, min=-1.0, max=1.0)
    return wave_x
