# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Tuple, Union

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

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

Segment = Tuple[int, int]
PadOrTrimSegment = Tuple[Union[str, int], Union[str, int]]
NoisePairSample = Tuple[Path, Path, torch.Tensor, torch.Tensor, Segment, Segment]


def _get_length_from_wavpath(path: Path, sample_rate: int) -> int:
    info = sf.info(str(path))
    assert info.samplerate == sample_rate
    assert info.channels == 1
    return int(info.frames)


def _pad_and_trim_segment(original_length: int, target_length: int | None, shuffle: bool = False) -> PadOrTrimSegment:
    if target_length is None:
        return (0, original_length)

    if original_length < target_length:
        total_pad = target_length - original_length
        if shuffle:
            left_pad = random.randint(0, total_pad)
        else:
            left_pad = 0
        right_pad = total_pad - left_pad
        segment = (f"zeropad_{left_pad}", f"zeropad_{right_pad}")
    else:
        if shuffle:
            start = random.randint(0, original_length - target_length)
        else:
            start = 0
        segment = (start, start + target_length)

    return segment


def _get_segment_type(segment: PadOrTrimSegment) -> str:
    assert len(segment) == 2
    if isinstance(segment[0], str):
        for x in segment:
            assert isinstance(x, str)
            assert x.startswith("zeropad_")
            assert len(x.split("_")) == 2
        return "pad"
    else:
        for x in segment:
            assert isinstance(x, int)
        return "trim"


def _load_pad_and_trim_wave(wave_path: Path, segment: PadOrTrimSegment, sample_rate: int) -> torch.Tensor:
    segment_type = _get_segment_type(segment=segment)
    if segment_type == "pad":
        assert isinstance(segment[0], str) and isinstance(segment[1], str)
        left_pad = int(segment[0].split("_")[1])
        right_pad = int(segment[1].split("_")[1])
        wave, sr = torchaudio.load(wave_path)
        wave = torch.nn.functional.pad(wave, pad=(left_pad, right_pad), mode="constant", value=0.0)
    else:
        assert isinstance(segment[0], int) and isinstance(segment[1], int)
        start = segment[0]
        end = segment[1]
        wave, sr = torchaudio.load(wave_path, frame_offset=start, num_frames=end - start)

    assert sr == sample_rate and wave.shape[0] == 1
    return wave[0]


class BaseNoisePairShiftDataset(Dataset):
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
            assert isinstance(sec, str) or (sec is None)
            return sec

    def _get_noise_pair(self, wave_s: torch.Tensor) -> NoisePairSample:
        raise NotImplementedError

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
            # Get noise pair
            noise_path, noise_ref_path, wave_n, wave_n_ref, segment_n, segment_n_ref = self._get_noise_pair(
                wave_s=wave_s
            )

            # Build sample dict
            sample_dict: dict[str, Any] = {
                "wave_s": wave_s.clone(),
                "wave_n": wave_n.clone(),
                "wave_n_ref": wave_n_ref.clone(),
            }
            if self.return_path:
                sample_dict["wave_s_path"] = str(target_path)
                sample_dict["wave_n_path"] = str(noise_path)
                sample_dict["wave_n_ref_path"] = str(noise_ref_path)
                sample_dict["segment_s"] = segment_s
                sample_dict["segment_n"] = segment_n
                sample_dict["segment_n_ref"] = segment_n_ref

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


class NoisePairShiftLeftDataset(BaseNoisePairShiftDataset):
    """
    Base dataset for same-recording noise pairs where wave_n_ref is left-shifted from wave_n.
    Subclasses define whether the shift anchor is wave_n start or wave_n end.
    """

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
        min_start_sec_n: float | None = None,
        sec_n_ref: float | str = "same_as_s",
        sec_n_ref_shift: float = 0.0,
    ) -> None:
        """
        Args:
            min_start_sec_n: Set this if you want to use the same segment with different shifts.
            When enabled, the dataset can sample the same wave_n while applying different wave_n_ref shifts.
            This value must be greater than the subclass-specific lower-bound formula.
        """
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
        self.min_start_idx_n: int | None = self._sec_to_length(sec=min_start_sec_n)  # type: ignore
        self.length_n_ref = self._sec_to_length(sec=sec_n_ref)
        self.length_n_ref_shift: int = self._sec_to_length(sec=sec_n_ref_shift)  # type: ignore
        assert isinstance(self.length_n_ref_shift, int)
        assert self.length_n_ref_shift >= 0

    def _get_min_start_idx_n_lower_bound(self, wave_s: torch.Tensor, length_n_ref: int) -> int:
        raise NotImplementedError

    def _get_ref_segment(self, start_idx_n: int, end_idx_n: int, length_n_ref: int, shift_length: int) -> Segment:
        raise NotImplementedError

    def _get_noise_pair(self, wave_s: torch.Tensor) -> NoisePairSample:
        # Load the base noise segment
        noise_key = random.choice(list(self.noise_path_dict_of_list.keys()))
        noise_path = random.choice(self.noise_path_dict_of_list[noise_key])
        noise_ref_path = noise_path
        length_n_base = _get_length_from_wavpath(path=noise_path, sample_rate=self.sample_rate)

        # Determine the segment for wave_n and wave_n_ref
        # Calculate min_start_idx_n_lower_bound
        if self.length_n_ref == "same_as_s":
            length_n_ref = len(wave_s)
        else:
            assert isinstance(self.length_n_ref, int)
            assert self.length_n_ref > 0
            length_n_ref = self.length_n_ref
        min_start_idx_n_lower_bound = self._get_min_start_idx_n_lower_bound(wave_s=wave_s, length_n_ref=length_n_ref)
        assert min_start_idx_n_lower_bound >= 0
        # If min_start_idx_n is provided, validate it.
        if self.min_start_idx_n is not None:
            min_start_idx_n = self.min_start_idx_n
            assert min_start_idx_n >= min_start_idx_n_lower_bound
        else:
            min_start_idx_n = min_start_idx_n_lower_bound
        assert length_n_base >= min_start_idx_n + len(wave_s)
        # Randomly sample the start and end index for wave_n
        start_idx_n = random.randint(min_start_idx_n, length_n_base - len(wave_s))
        end_idx_n = start_idx_n + len(wave_s)
        segment_n = (start_idx_n, end_idx_n)
        start_idx_n_ref, end_idx_n_ref = self._get_ref_segment(
            start_idx_n=start_idx_n,
            end_idx_n=end_idx_n,
            length_n_ref=length_n_ref,
            shift_length=self.length_n_ref_shift,
        )
        segment_n_ref = (start_idx_n_ref, end_idx_n_ref)
        assert start_idx_n_ref >= 0
        assert end_idx_n_ref <= length_n_base
        wave_n = _load_pad_and_trim_wave(
            wave_path=noise_path,
            segment=segment_n,
            sample_rate=self.sample_rate,
        )
        wave_n_ref = _load_pad_and_trim_wave(
            wave_path=noise_ref_path,
            segment=segment_n_ref,
            sample_rate=self.sample_rate,
        )

        return noise_path, noise_ref_path, wave_n, wave_n_ref, segment_n, segment_n_ref


class NoisePairShiftLeftFromStartDataset(NoisePairShiftLeftDataset):
    """
    Obtain the reference noise by shifting the segment left from the start of the ground-truth noise segment.
    |<-------------------------------wave_n_base------------------------------->|
    |<--------------min_start_sec_n---------------->|
    |<--length_n_ref-->|<--length_n_ref_shift-->|
    Example
    |              |<--wave_n_ref-->|<------shift------->|<----wave_n---->|     |
    """

    def _get_min_start_idx_n_lower_bound(self, wave_s: torch.Tensor, length_n_ref: int) -> int:
        return length_n_ref + self.length_n_ref_shift

    def _get_ref_segment(self, start_idx_n: int, end_idx_n: int, length_n_ref: int, shift_length: int) -> Segment:
        end_idx_n_ref = start_idx_n - shift_length
        start_idx_n_ref = end_idx_n_ref - length_n_ref
        return start_idx_n_ref, end_idx_n_ref


class NoisePairShiftLeftFromEndDataset(NoisePairShiftLeftDataset):
    """
    Obtain the reference noise by shifting the segment left from the end of the ground-truth noise segment.
    |<-------------------------------wave_n_base------------------------------->|
    |<--length_n_ref-->|<--length_n_ref_shift-->|
    |<--------min_start_sec_n--------->|<--wave_n-->|
    Example
    |                                      |<----wave_n---->|                   |
    |       |<--length_n_ref-->|<--length_n_ref_shift-->|                   |
    """

    def _get_min_start_idx_n_lower_bound(self, wave_s: torch.Tensor, length_n_ref: int) -> int:
        return max(0, length_n_ref + self.length_n_ref_shift - len(wave_s))

    def _get_ref_segment(self, start_idx_n: int, end_idx_n: int, length_n_ref: int, shift_length: int) -> Segment:
        end_idx_n_ref = end_idx_n - shift_length
        start_idx_n_ref = end_idx_n_ref - length_n_ref
        return start_idx_n_ref, end_idx_n_ref


class TestDataset(Dataset):
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
        wave_n_ref = _load_pad_and_trim_wave(
            wave_path=item["wave_n_ref_path"],
            segment=item["segment_n_ref"],
            sample_rate=self.sample_rate,
        )
        if self.return_wave_s:
            wave_s = _load_pad_and_trim_wave(
                wave_path=item["wave_s_path"],
                segment=item["segment_s"],
                sample_rate=self.sample_rate,
            )
            wave_s = _avoid_zero_signal(wave_s=wave_s, source_path=item["wave_s_path"])

        output_dict = {
            "wave_x": wave_x,
            "wave_n_ref": wave_n_ref,
            "wave_x_path": item["wave_x_path"],
            "wave_n_ref_path": item["wave_n_ref_path"],
            "segment_n_ref": item["segment_n_ref"],
        }
        if self.return_wave_s:
            output_dict["wave_s"] = wave_s
            output_dict["wave_s_path"] = item["wave_s_path"]
            output_dict["segment_s"] = item["segment_s"]

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
