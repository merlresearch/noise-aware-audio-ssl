# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch

from nassl.module import mix_batch_random_snr
from nassl.utils.hydra import instantiate

logger = logging.getLogger(__name__)


class TensorCollator:
    def __init__(self, shuffle: bool = False) -> None:
        del shuffle  # dummy

    def __call__(self, batch_list: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
        dict_of_list: dict[str, list] = defaultdict(list)
        for sample in batch_list:
            for key, value in sample.items():
                dict_of_list[key].append(value)

        output_dict: dict[str, Any] = {}
        for key, data_list in dict_of_list.items():
            if isinstance(data_list[0], torch.Tensor):
                output_dict[key] = torch.stack(data_list, dim=0)
            else:
                output_dict[key] = data_list
        return output_dict


class PairGeneratorCollator(TensorCollator):
    def __init__(
        self,
        pair_generator_cfg: dict[str, Any],
        shuffle: bool = False,
    ) -> None:
        super().__init__(shuffle=shuffle)
        self.pair_generator = instantiate({**pair_generator_cfg, "shuffle": shuffle})

    def __call__(self, batch_list: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
        output_dict = super().__call__(batch_list)
        output_dict = self.pair_generator(output_dict)
        return output_dict


class RandomSNRMixCollator(TensorCollator):
    def __init__(
        self,
        snr_range: list[float],
        shuffle: bool = False,
        return_snr: bool = False,
    ) -> None:
        super().__init__(shuffle=shuffle)
        self.snr_range = (snr_range[0], snr_range[1])
        self.return_snr = return_snr
        assert shuffle
        assert self.snr_range[0] <= self.snr_range[1]

    def __call__(self, batch_list: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
        output_dict = super().__call__(batch_list)
        assert "wave_x" not in output_dict
        wave_x, snr = mix_batch_random_snr(
            wave_s=output_dict["wave_s"],
            wave_n=output_dict["wave_n"],
            snr_range=self.snr_range,
        )
        output_dict["wave_x"] = wave_x
        if self.return_snr:
            output_dict["snr"] = snr
        return output_dict
