# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from abc import ABC

import torch
from torch import nn


class BaseExtractor(nn.Module, ABC):
    """Base class for feature extraction models."""

    def __init__(self):
        super().__init__()

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
