# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import soundfile as sf
from tqdm import tqdm

from nassl.utils.io import read_json, write_json


class TestManifestModificator(Protocol):
    def modify_item(self, item: dict[str, Any], sample_rate: int) -> dict[str, Any]: ...


def get_length(path: Path | str, sample_rate: int) -> int:
    info = sf.info(path)
    assert info.samplerate == sample_rate
    return info.frames


class ShiftSegmentNRefModificator:
    def __init__(self, shift_sec: int | float | str, shift_direction: str) -> None:
        self.shift_sec = shift_sec
        self.shift_direction = shift_direction

    def modify_item(self, item: dict[str, Any], sample_rate: int) -> dict[str, Any]:
        wave_n_length = get_length(path=item["wave_n_ref_path"], sample_rate=sample_rate)
        shift_length = self._get_shift_length(sample_rate=sample_rate)
        item["segment_n_ref"] = self._shift_segment_n_ref(
            segment_n_ref=item["segment_n_ref"],
            shift_length=shift_length,
            shift_direction=self.shift_direction,
            wave_n_length=wave_n_length,
            wave_s_path=item["wave_s_path"],
        )
        return item

    def _get_shift_length(self, sample_rate: int) -> int:
        assert isinstance(self.shift_sec, (int, float))
        return int(self.shift_sec * sample_rate)

    @staticmethod
    def _shift_segment_n_ref(
        segment_n_ref: list[int],
        shift_length: int,
        shift_direction: str,
        wave_n_length: int,
        wave_s_path: Path | str,
    ) -> list[int]:
        start_idx, end_idx = int(segment_n_ref[0]), int(segment_n_ref[1])
        if shift_direction == "left":
            if start_idx - shift_length < 0:
                raise ValueError(
                    f"Cannot shift left by {shift_length} samples for {wave_s_path} "
                    f"because it would result in a negative start index."
                )
            return [start_idx - shift_length, end_idx - shift_length]
        if shift_direction == "right":
            if end_idx + shift_length > wave_n_length:
                raise ValueError(
                    f"Cannot shift right by {shift_length} samples for {wave_s_path} because "
                    f"it would result in an end index that exceeds the length of the noisy waveform."
                )
            return [start_idx + shift_length, end_idx + shift_length]
        raise ValueError(f"shift_direction must be 'left' or 'right', but got: {shift_direction}")


def modify_manifest_test(
    manifest_test_path: Path,
    manifest_output_path: Path,
    sample_rate: int,
    modificator: TestManifestModificator,
) -> Path:
    manifest_item_list: list[dict[str, Any]] = read_json(json_path=manifest_test_path)  # type: ignore
    assert isinstance(manifest_item_list, list)
    for item in tqdm(manifest_item_list):
        item = modificator.modify_item(item=item, sample_rate=sample_rate)
    manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_output_path.exists():
        raise FileExistsError(f"File already exists: {manifest_output_path}")
    write_json(json_path=manifest_output_path, data=manifest_item_list)
    return manifest_output_path
