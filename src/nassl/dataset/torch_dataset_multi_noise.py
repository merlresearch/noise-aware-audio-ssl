# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch

from nassl.dataset.torch_dataset import PadOrTrimSegment
from nassl.dataset.torch_dataset_denoising import DenoisingDataset
from nassl.dataset.utils import _avoid_zero_signal

logger = logging.getLogger(__name__)


class TargetMultiNoisePairDataset(DenoisingDataset):
    def __init__(
        self,
        target_path_selector_list: List[str],
        noise_path_selector_dict: Dict[str, List[str]],
        sec_s: Union[str, float],
        validator_cfg: Dict[str, Any],
        num_noises: int = 4,
        sample_rate: int = 16000,
        return_path: bool = False,
        shuffle: bool = False,
        max_retries: int = 100,
        local_cache_root: Union[str, None] = None,
    ) -> None:
        super().__init__(
            target_path_selector_list=target_path_selector_list,
            noise_path_selector_dict=noise_path_selector_dict,
            sec_s=sec_s,
            validator_cfg=validator_cfg,
            sample_rate=sample_rate,
            return_path=return_path,
            shuffle=shuffle,
            max_retries=max_retries,
            local_cache_root=local_cache_root,
        )
        self.num_noises = num_noises
        assert self.num_noises > 0
        all_noise_paths = {
            noise_path for noise_path_list in self.noise_path_dict_of_list.values() for noise_path in noise_path_list
        }
        assert len(all_noise_paths) >= self.num_noises

    def _get_noise_path(self) -> Path:
        noise_key = random.choice(list(self.noise_path_dict_of_list.keys()))
        noise_path = random.choice(self.noise_path_dict_of_list[noise_key])
        return noise_path

    def _load_wave_s(self, idx: int) -> Tuple[torch.Tensor, PadOrTrimSegment, Path, Optional[int]]:
        target_path = self.target_path_list[idx]
        assert isinstance(self.length_s, int)

        # Load target signal
        wave_s, segment_s = self._load_wave_segment(path=target_path, target_length=self.length_s)
        wave_s = _avoid_zero_signal(wave_s=wave_s, source_path=target_path)
        return wave_s, segment_s, target_path, self.length_s

    def _load_noise_group(
        self, target_length: Optional[int]
    ) -> Tuple[torch.Tensor, List[Path], List[PadOrTrimSegment]]:
        wave_n_list: List[torch.Tensor] = []
        noise_path_list: List[Path] = []
        segment_n_list: List[PadOrTrimSegment] = []
        used_noise_path_set: Set[Path] = set()
        max_trials = self.max_retries * self.num_noises

        for _ in range(max_trials):
            if len(wave_n_list) == self.num_noises:
                break

            noise_path = self._get_noise_path()
            if noise_path in used_noise_path_set:
                continue

            wave_n, segment_n = self._load_wave_segment(path=noise_path, target_length=target_length)
            used_noise_path_set.add(noise_path)
            wave_n_list.append(wave_n)
            noise_path_list.append(noise_path)
            segment_n_list.append(segment_n)

        assert len(wave_n_list) == self.num_noises
        return torch.stack(wave_n_list, dim=0), noise_path_list, segment_n_list

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        wave_s, segment_s, target_path, target_length_s = self._load_wave_s(idx=idx)

        for _ in range(self.max_retries):
            wave_n, noise_path_list, segment_n_list = self._load_noise_group(target_length=target_length_s)

            sample_dict: Dict[str, Any] = {
                "wave_s": wave_s.clone(),
                "wave_n": wave_n.clone(),
            }
            if self.return_path:
                sample_dict["wave_s_path"] = str(target_path)
                sample_dict["wave_n_path"] = [str(noise_path) for noise_path in noise_path_list]
                sample_dict["segment_s"] = segment_s
                sample_dict["segment_n"] = segment_n_list

            if self.validator.validate(sample_dict=sample_dict):
                return sample_dict
            logger.warning(f"Validation failed for the sampled noise group, retrying... (target_path: {target_path})")

        raise ValueError(
            f"Failed to get a valid multi-noise sample after {self.max_retries} retries (target_path: {target_path})"
        )
