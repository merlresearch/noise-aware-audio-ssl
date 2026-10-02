# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from nassl.dataset.torch_dataset import (
    PadOrTrimSegment,
    _get_length_from_wavpath,
    _load_pad_and_trim_wave,
    _pad_and_trim_segment,
)
from nassl.dataset.utils import (
    _avoid_zero_signal,
    _stage_manifest_item_paths,
    build_feature_cache_path,
    collect_paths,
    stage_paths_to_local_data,
    torch_mono_wav_load,
)
from nassl.dataset.validator import BaseValidator
from nassl.utils.hydra import instantiate
from nassl.utils.io import read_json

logger = logging.getLogger(__name__)


class DenoisingDataset(Dataset):
    def __init__(
        self,
        target_path_selector_list: list[str],
        noise_path_selector_dict: dict[str, list[str]],
        sec_s: str | float,
        validator_cfg: dict[str, Any],
        sample_rate: int = 16000,
        return_path: bool = False,
        shuffle: bool = False,
        max_retries: int = 100,
        local_cache_root: str | None = None,
    ) -> None:
        super().__init__()
        self.target_path_list = stage_paths_to_local_data(
            path_list=collect_paths(path_selector_list=target_path_selector_list),
            local_cache_root=local_cache_root,
        )
        self.noise_path_dict_of_list = {}
        for key, path_list in noise_path_selector_dict.items():
            self.noise_path_dict_of_list[key] = stage_paths_to_local_data(
                path_list=collect_paths(path_selector_list=path_list),
                local_cache_root=local_cache_root,
            )

        self.validator: BaseValidator = instantiate(validator_cfg)
        self.sample_rate = sample_rate
        self.return_path = return_path
        self.shuffle = shuffle
        self.length_s = self._sec_to_length(sec=sec_s)
        self.max_retries = max_retries

    def __len__(self) -> int:
        return len(self.target_path_list)

    def _sec_to_length(self, sec: float | str | None) -> int | str | None:
        if isinstance(sec, (int, float)):
            length = int(sec * self.sample_rate)
            assert length >= 0
            return length
        else:
            raise ValueError(f"Unsupported type for sec: {type(sec)} (value: {sec})")

    def _load_wave_segment(self, path: Path, target_length: int | None) -> tuple[torch.Tensor, PadOrTrimSegment]:
        original_length = _get_length_from_wavpath(path=path, sample_rate=self.sample_rate)
        segment = _pad_and_trim_segment(
            original_length=original_length,
            target_length=target_length,
            shuffle=self.shuffle,
        )
        wave = _load_pad_and_trim_wave(wave_path=path, segment=segment, sample_rate=self.sample_rate)
        return wave, segment

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Get target signal
        target_path = self.target_path_list[idx]
        if self.length_s == "all":
            target_length_s = None
        else:
            assert isinstance(self.length_s, int)
            target_length_s = self.length_s

        # Load target signal
        wave_s, segment_s = self._load_wave_segment(path=target_path, target_length=target_length_s)
        wave_s = _avoid_zero_signal(wave_s=wave_s, source_path=target_path)

        # Noise sampling with validation
        for _ in range(self.max_retries):
            # Get noise signal
            noise_key = random.choice(list(self.noise_path_dict_of_list.keys()))
            noise_path = random.choice(self.noise_path_dict_of_list[noise_key])
            wave_n, segment_n = self._load_wave_segment(path=noise_path, target_length=target_length_s)

            # Build sample dict
            sample_dict: dict[str, Any] = {
                "wave_s": wave_s.clone(),
                "wave_n": wave_n.clone(),
            }
            if self.return_path:
                sample_dict["wave_s_path"] = str(target_path)
                sample_dict["wave_n_path"] = str(noise_path)
                sample_dict["segment_s"] = segment_s
                sample_dict["segment_n"] = segment_n

            # Validate sample dict
            if self.validator.validate(sample_dict=sample_dict):
                return sample_dict
            else:
                logger.warning(
                    f"Validation failed for the sampled noise pair, retrying... (target_path: {target_path})"
                )

        raise ValueError(
            f"Failed to get a valid noise pair sample after {self.max_retries} retries (target_path: {target_path})"
        )


class DenoisingTestDataset(Dataset):
    def __init__(
        self,
        manifest_json_path: str,
        sample_rate: int,
        return_wave_s: bool = True,
        return_rep_s: bool = False,
        feature_dir: str | Path | None = None,
        local_cache_root: str | None = None,
    ) -> None:
        super().__init__()
        self.local_cache_root = local_cache_root
        self.item_list: list[dict[str, Any]] = read_json(json_path=manifest_json_path)  # type: ignore
        self.item_list = _stage_manifest_item_paths(item_list=self.item_list, local_cache_root=self.local_cache_root)
        self.sample_rate = sample_rate
        self.return_wave_s = return_wave_s
        self.return_rep_s = return_rep_s
        self.feature_dir = feature_dir
        if self.return_rep_s:
            assert self.feature_dir is not None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.item_list[idx]

        wave_x = torch_mono_wav_load(path=item["wave_x_path"], sample_rate=self.sample_rate)
        if self.return_wave_s:
            wave_s = _load_pad_and_trim_wave(
                wave_path=item["wave_s_path"],
                segment=item["segment_s"],
                sample_rate=self.sample_rate,
            )
            wave_s = _avoid_zero_signal(wave_s=wave_s, source_path=item["wave_s_path"])

        output_dict = {
            "wave_x": wave_x,
            "wave_x_path": item["wave_x_path"],
        }
        if self.return_wave_s:
            output_dict["wave_s"] = wave_s

        if self.return_rep_s:
            assert self.feature_dir is not None
            wave_s_path = Path(item["wave_s_path"])
            if self.local_cache_root is not None:
                wave_s_path = wave_s_path.relative_to(f"{self.local_cache_root}/files")
                wave_s_path = Path(f"/{wave_s_path}")
            rep_s_path = build_feature_cache_path(
                wave_path=wave_s_path,
                feature_dir=Path(self.feature_dir),
                feature_ext=".pt",
                segment=item["segment_s"],
            )
            output_dict["rep_s_path"] = str(rep_s_path)
            output_dict["rep_s"] = torch.load(rep_s_path)  # type: ignore

        return output_dict

    def __len__(self) -> int:
        return len(self.item_list)
