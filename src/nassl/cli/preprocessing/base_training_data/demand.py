# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
from pathlib import Path
from typing import List

from nassl.cli.preprocessing.base_training_data.dataset_pipeline import (
    create_output_dir,
    filter_unit_list_by_duration,
    run_split_manifest_pipeline,
    split_units_random,
)

CH = 16
SAMPLE_RATE = 16000


def check_data_directory(data_dir: Path) -> None:
    sub_dir_list = sorted([sub_dir for sub_dir in data_dir.glob("*") if sub_dir.is_dir()])
    if len(sub_dir_list) == 0:
        raise FileNotFoundError(f"No DEMAND recording directories found in {data_dir}")

    for sub_dir in sub_dir_list:
        missing_path_list = [
            sub_dir / f"ch{ch+1:02d}.wav" for ch in range(CH) if not (sub_dir / f"ch{ch+1:02d}.wav").is_file()
        ]
        if len(missing_path_list) > 0:
            raise FileNotFoundError(f"Missing DEMAND channel files in {sub_dir}: {missing_path_list}")


def convert_to_path(unit: Path) -> List[str]:
    path_list = []
    for ch in range(CH):
        path_list.append(str(unit / f"ch{ch+1:02d}.wav"))
    return path_list


def main(data_dir: Path, generate_path_list: bool = True) -> None:
    check_data_directory(data_dir=data_dir)
    output_dir = create_output_dir(script_path=Path(__file__))
    sub_dir_list = sorted([sub_dir for sub_dir in data_dir.glob("*") if sub_dir.is_dir()])
    sub_dir_list = filter_unit_list_by_duration(
        unit_list=sub_dir_list,
        unit_to_path_list_func=convert_to_path,
    )

    split_count = [len(sub_dir_list), 0, 0]
    split_to_sub_dir_list = split_units_random(unit_list=sub_dir_list, split_count=split_count, seed=0)
    run_split_manifest_pipeline(
        output_dir=output_dir,
        split_to_unit_list=split_to_sub_dir_list,
        unit_to_path_list_func=convert_to_path,
        expected_total_path_count=len(sub_dir_list) * CH,
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
