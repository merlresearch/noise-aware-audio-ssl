# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import librosa
import soundfile as sf
from tqdm import tqdm

from nassl.preprocessing.split import random_split_path
from nassl.utils.io import read_sf, write_json, write_sf, write_txt
from nassl.utils.round import myround

SPLIT_NAMES: List[str] = ["train", "valid", "test"]
THRESHOLD_SEC = 60
# Maximum segment length in the downstream task is 30 sec
# Assuming that the reference noise is extracted from the same recording, I set the threshold to 60 sec.


def create_output_dir(script_path: Path) -> Path:
    dataname = script_path.stem
    output_dir = Path(f"scratch/dataset/processed/{dataname}")
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Created output directory: {output_dir.resolve()}")
    return output_dir


def split_units_random(unit_list: List[Any], split_count: List[int], seed: int = 0) -> Dict[str, List[Any]]:
    train_units, valid_units, test_units = random_split_path(
        path_list=unit_list, train_valid_test_split_count=split_count, seed=seed
    )
    return {"train": train_units, "valid": valid_units, "test": test_units}


def get_path_duration_sec(
    path: Union[Path, str],
    path_to_duration_sec: Union[None, Dict[str, float]] = None,
) -> float:
    resolved_path_str = str(Path(path).resolve())
    if path_to_duration_sec is not None:
        assert resolved_path_str in path_to_duration_sec
        return path_to_duration_sec[resolved_path_str]
    return sf.info(str(path)).duration


def filter_path_list_by_duration(
    path_list: List[Path],
    path_to_duration_sec: Union[None, Dict[str, float]] = None,
) -> List[Path]:
    filtered_path_list: List[Path] = []
    for path in tqdm(path_list):
        duration_sec = get_path_duration_sec(path=path, path_to_duration_sec=path_to_duration_sec)
        if duration_sec > THRESHOLD_SEC:
            filtered_path_list.append(path)
    return filtered_path_list


def filter_unit_list_by_duration(
    unit_list: List[Any],
    unit_to_path_list_func: Callable[..., List[str]],
    path_to_duration_sec: Union[None, Dict[str, float]] = None,
) -> List[Any]:
    filtered_unit_list: List[Any] = []
    for unit in tqdm(unit_list):
        path_list = unit_to_path_list_func(unit=unit)
        assert len(path_list) > 0
        duration_sec_list = [
            get_path_duration_sec(path=path, path_to_duration_sec=path_to_duration_sec) for path in path_list
        ]
        assert len(set(duration_sec_list)) == 1
        duration_sec = duration_sec_list[0]
        if duration_sec > THRESHOLD_SEC:
            filtered_unit_list.append(unit)
    return filtered_unit_list


def build_channel_output_path(src_path: Path, data_dir: Path, output_dir: Path, channel_idx: int) -> Path:
    relative_path = src_path.relative_to(data_dir)
    return output_dir / "audio" / relative_path.parent / f"{relative_path.stem}.CH{channel_idx+1}.wav"


def src_path_to_channel_paths(src_path: Path, data_dir: Path, output_dir: Path, channels: int) -> List[str]:
    path_list = []
    for channel_idx in range(channels):
        output_path = build_channel_output_path(
            src_path=src_path,
            data_dir=data_dir,
            output_dir=output_dir,
            channel_idx=channel_idx,
        )
        path_list.append(str(output_path.resolve()))
    return path_list


def preprocess_multichannel_audio(
    src_path_list: List[Path],
    data_dir: Path,
    output_dir: Path,
    source_sample_rate: int,
    target_sample_rate: int,
    channels: int,
) -> None:
    (output_dir / "audio").mkdir(parents=False, exist_ok=False)

    for src_path in tqdm(src_path_list):
        wav, original_sample_rate, subtype = read_sf(wav_path=src_path)
        assert wav.ndim == 2
        assert wav.shape[1] == channels
        assert original_sample_rate == source_sample_rate
        wav = librosa.resample(wav, orig_sr=source_sample_rate, target_sr=target_sample_rate, axis=0)
        for channel_idx in range(channels):
            output_path = build_channel_output_path(
                src_path=src_path,
                data_dir=data_dir,
                output_dir=output_dir,
                channel_idx=channel_idx,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_sf(
                wav_path=output_path,
                audio=wav[:, channel_idx],
                sr=target_sample_rate,
                subtype=subtype,
            )


def convert_split_units_to_paths(
    split_to_unit_list: Dict[str, List[Any]],
    unit_to_path_list_func: Callable[..., List[str]],
) -> Dict[str, List[str]]:
    split_to_path_list: Dict[str, List[str]] = {}
    for split_name in SPLIT_NAMES:
        path_list: List[str] = []
        for unit in tqdm(split_to_unit_list[split_name]):
            path_list.extend(unit_to_path_list_func(unit=unit))
        path_list = [str(Path(p).resolve()) for p in path_list]
        split_to_path_list[split_name] = path_list
    return split_to_path_list


def flatten_split_paths(split_to_path_list: Dict[str, List[str]]) -> List[str]:
    all_path_list: List[str] = []
    for split_name in SPLIT_NAMES:
        all_path_list.extend(split_to_path_list[split_name])
    return all_path_list


def validate_audio_paths(path_list: List[str], expected_channels: int, expected_sample_rate: int) -> None:
    for path_str in tqdm(path_list):
        path = Path(path_str)
        assert path.exists()
        info = sf.info(str(path))
        assert info.channels == expected_channels
        assert info.samplerate == expected_sample_rate


def save_split_paths(output_dir: Path, split_to_path_list: Dict[str, List[str]]) -> None:
    path_dir = output_dir / "path"
    path_dir.mkdir(parents=False, exist_ok=False)
    for split_name in SPLIT_NAMES:
        write_json(
            json_path=path_dir / f"{split_name}.json",
            data=split_to_path_list[split_name],
        )


def get_total_duration(path_list: List[str]) -> float:
    total_duration = 0.0
    for file in tqdm(path_list):
        total_duration += sf.info(file).duration
    total_duration /= 3600.0  # convert to hours
    return total_duration


def write_split_info(
    output_dir: Path,
    split_to_unit_list: Dict[str, List[Any]],
    split_to_path_list: Dict[str, List[str]],
) -> None:
    train_duration = get_total_duration(path_list=split_to_path_list["train"])
    valid_duration = get_total_duration(path_list=split_to_path_list["valid"])
    test_duration = get_total_duration(path_list=split_to_path_list["test"])
    total_units = len(split_to_unit_list["train"]) + len(split_to_unit_list["valid"]) + len(split_to_unit_list["test"])
    total_paths = len(split_to_path_list["train"]) + len(split_to_path_list["valid"]) + len(split_to_path_list["test"])
    info = (
        f"Total unit: {total_units}\n"
        f"Train unit: {len(split_to_unit_list['train'])}, "
        f"Valid unit: {len(split_to_unit_list['valid'])}, "
        f"Test unit: {len(split_to_unit_list['test'])}\n"
        f"Total path: {total_paths}\n"
        f"Train path: {len(split_to_path_list['train'])}, "
        f"Valid path: {len(split_to_path_list['valid'])}, "
        f"Test path: {len(split_to_path_list['test'])}\n"
        f"Total duration: {myround(train_duration + valid_duration + test_duration)} hours\n"
        f"Train duration: {myround(train_duration)} hours, "
        f"Valid duration: {myround(valid_duration)} hours, "
        f"Test duration: {myround(test_duration)} hours"
    )
    write_txt(txt_path=output_dir / "info.txt", data=info)


def run_split_manifest_pipeline(
    output_dir: Path,
    split_to_unit_list: Dict[str, List[Any]],
    unit_to_path_list_func: Callable[..., List[str]],
    expected_total_path_count: int,
    expected_channels: int,
    expected_sample_rate: int,
    generate_path_list: bool = True,
) -> None:
    if not generate_path_list:
        return

    split_to_path_list = convert_split_units_to_paths(
        split_to_unit_list=split_to_unit_list,
        unit_to_path_list_func=unit_to_path_list_func,
    )
    all_path_list = flatten_split_paths(split_to_path_list=split_to_path_list)
    assert len(all_path_list) == expected_total_path_count
    validate_audio_paths(
        path_list=all_path_list,
        expected_channels=expected_channels,
        expected_sample_rate=expected_sample_rate,
    )
    save_split_paths(output_dir=output_dir, split_to_path_list=split_to_path_list)
    write_split_info(
        output_dir=output_dir,
        split_to_unit_list=split_to_unit_list,
        split_to_path_list=split_to_path_list,
    )
