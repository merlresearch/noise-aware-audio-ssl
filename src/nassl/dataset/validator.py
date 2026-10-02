# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch


class BaseValidator(ABC):
    @abstractmethod
    def validate(self, sample_dict: Dict[str, Any]) -> bool:
        raise NotImplementedError


class TrueValidator(BaseValidator):
    def validate(self, sample_dict: Dict[str, Any]) -> bool:
        return True


def _has_nonzero_waveform(value: torch.Tensor) -> bool:
    if value.ndim <= 1:
        return bool(value.abs().sum() != 0)

    value_flat = value.reshape(-1, value.shape[-1])
    return bool(torch.all(value_flat.abs().sum(dim=1) != 0))


class NoisePowerValidator(BaseValidator):
    def __init__(self, eps: float = 1e-7, noise_keys: Optional[List[str]] = None) -> None:
        self.eps = eps
        self.noise_keys = noise_keys if noise_keys is not None else ["wave_n", "wave_n_ref"]

    def validate(self, sample_dict: Dict[str, Any]) -> bool:
        for key, value in sample_dict.items():
            if key in self.noise_keys:
                if not _has_nonzero_waveform(value=value):
                    return False
        return True
