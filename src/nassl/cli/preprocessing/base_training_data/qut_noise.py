# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
from pathlib import Path
from typing import List

from nassl.cli.preprocessing.base_training_data.dataset_pipeline import (
    create_output_dir,
    filter_path_list_by_duration,
    preprocess_multichannel_audio,
    run_split_manifest_pipeline,
    split_units_random,
    src_path_to_channel_paths,
)

SAMPLE_RATE = 16000
SOURCE_SAMPLE_RATE = 48000
CH = 2


def check_data_directory(data_dir: Path) -> None:
    qut_noise_dir = data_dir / "QUT-NOISE"
    if not qut_noise_dir.is_dir():
        raise FileNotFoundError(f"QUT-NOISE directory does not exist: {qut_noise_dir}")


def main(data_dir: Path, generate_path_list: bool = True) -> None:
    check_data_directory(data_dir=data_dir)
    output_dir = create_output_dir(script_path=Path(__file__))
    src_path_list = sorted([p for p in data_dir.glob("QUT-NOISE/QUT-NOISE/*.wav")])
    src_path_list = filter_path_list_by_duration(path_list=src_path_list)

    preprocess_multichannel_audio(
        src_path_list=src_path_list,
        data_dir=data_dir,
        output_dir=output_dir,
        source_sample_rate=SOURCE_SAMPLE_RATE,
        target_sample_rate=SAMPLE_RATE,
        channels=CH,
    )

    split_count = [len(src_path_list), 0, 0]
    split_to_src_path_list = split_units_random(unit_list=src_path_list, split_count=split_count, seed=0)

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
