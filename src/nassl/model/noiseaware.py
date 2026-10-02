# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class NoiseAwareModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def validate_wave_pair(self, wave_x: torch.Tensor, wave_n_ref: torch.Tensor) -> None:
        if wave_x.ndim != 2:
            raise ValueError(f"wave_x must have shape (B, T), got {wave_x.shape}")
        if wave_n_ref.ndim != 2:
            raise ValueError(f"wave_n_ref must have shape (B, T), got {wave_n_ref.shape}")
        if wave_x.shape[0] != wave_n_ref.shape[0]:
            raise ValueError(
                f"wave_x and wave_n_ref must have the same batch size, got {wave_x.shape[0]} and {wave_n_ref.shape[0]}"
            )

    def forward(self, wave_x: torch.Tensor, wave_n_ref: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
