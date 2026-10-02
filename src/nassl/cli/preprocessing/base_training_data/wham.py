# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from nassl.cli.preprocessing.base_training_data.dataset_pipeline import (
    create_output_dir,
    filter_path_list_by_duration,
    preprocess_multichannel_audio,
    run_split_manifest_pipeline,
    src_path_to_channel_paths,
)

SAMPLE_RATE = 16000
SOURCE_SAMPLE_RATE = 48000
CH = 2


def check_data_directory(data_dir: Path) -> None:
    wham_dir = data_dir / "high_res_wham"
    if not wham_dir.is_dir():
        raise FileNotFoundError(f"high_res_wham directory does not exist: {wham_dir}")


def build_path_to_duration_sec(metadata_df: pd.DataFrame, data_dir: Path) -> Dict[str, float]:
    path_to_duration_sec: Dict[str, float] = {}
    for filename, duration_sec in zip(metadata_df["Filename"].values, metadata_df["File Length (sec)"].values):
        src_path = data_dir / "high_res_wham/audio" / filename
        path_to_duration_sec[str(src_path.resolve())] = float(duration_sec)
    return path_to_duration_sec


def wham_split_path(path_list: List[Path], metadata_df: pd.DataFrame, data_dir: Path) -> Dict[str, List[Path]]:
    path_set = set(path_list)
    filename_dict: Dict[str, List[Path]] = {}
    for split in ["train", "valid", "test"]:
        split_idx = metadata_df["WHAM! Split"] == split.capitalize()
        split_path_list = [data_dir / "high_res_wham/audio" / p for p in metadata_df[split_idx]["Filename"].values]
        filename_dict[split] = [path for path in split_path_list if path in path_set]
    return filename_dict


def main(data_dir: Path, generate_path_list: bool = True) -> None:
    check_data_directory(data_dir=data_dir)
    output_dir = create_output_dir(script_path=Path(__file__))
    metadata_df = pd.read_csv(data_dir / "high_res_wham/high_res_metadata.csv")
    path_to_duration_sec = build_path_to_duration_sec(metadata_df=metadata_df, data_dir=data_dir)

    src_path_list = [data_dir / "high_res_wham/audio" / filename for filename in metadata_df["Filename"].values]
    src_path_list = filter_path_list_by_duration(
        path_list=src_path_list,
        path_to_duration_sec=path_to_duration_sec,
    )

    preprocess_multichannel_audio(
        src_path_list=src_path_list,
        data_dir=data_dir,
        output_dir=output_dir,
        source_sample_rate=SOURCE_SAMPLE_RATE,
        target_sample_rate=SAMPLE_RATE,
        channels=CH,
    )

    split_to_src_path_list = wham_split_path(path_list=src_path_list, metadata_df=metadata_df, data_dir=data_dir)

    def convert_to_path(unit: Path) -> List[str]:
        return src_path_to_channel_paths(src_path=unit, data_dir=data_dir, output_dir=output_dir, channels=CH)

    run_split_manifest_pipeline(
        output_dir=output_dir,
        split_to_unit_list=split_to_src_path_list,
        unit_to_path_list_func=convert_to_path,
        expected_total_path_count=len(src_path_list) * CH,
        expected_channels=1,
        expected_sample_rate=SAMPLE_RATE,
        generate_path_list=generate_path_list,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", type=Path)
    argparser.add_argument("--skip_path_list", action="store_true")
    args = argparser.parse_args()
    main(data_dir=args.data_dir, generate_path_list=not args.skip_path_list)
