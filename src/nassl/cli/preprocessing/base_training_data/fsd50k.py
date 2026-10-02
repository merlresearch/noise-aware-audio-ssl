# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import librosa
import pandas as pd
from tqdm import tqdm

from nassl.cli.preprocessing.base_training_data.dataset_pipeline import create_output_dir, run_split_manifest_pipeline
from nassl.utils.io import read_sf, write_sf

SAMPLE_RATE = 16000
SOURCE_SAMPLE_RATE = 44100


def check_data_directory(data_dir: Path) -> None:
    required_dir_list = [
        data_dir / "FSD50K.dev_audio",
        data_dir / "FSD50K.eval_audio",
        data_dir / "FSD50K.ground_truth",
    ]
    missing_dir_list = [required_dir for required_dir in required_dir_list if not required_dir.is_dir()]
    if len(missing_dir_list) > 0:
        raise FileNotFoundError(f"Missing FSD50K directories under {data_dir}: {missing_dir_list}")


def split_dev_paths(output_dir: Path, fsd50k_ground_truth_dir: Path) -> tuple[list[Path], list[Path]]:
    dev_df = pd.read_csv(fsd50k_ground_truth_dir / "dev.csv")
    split_to_fname_list: dict[str, list[str]] = {
        split_name: dev_df.loc[dev_df["split"] == split_name, "fname"].tolist() for split_name in ["train", "val"]
    }
    assert sum(len(fname_list) for fname_list in split_to_fname_list.values()) == 40966

    split_to_source_paths: dict[str, list[Path]] = defaultdict(list)
    for split_name in ["train", "val"]:
        for fname in split_to_fname_list[split_name]:
            source_path = output_dir / "audio/FSD50K.dev_audio" / f"{fname}.wav"
            assert source_path.is_file()
            split_to_source_paths[split_name].append(source_path)
    return split_to_source_paths["train"], split_to_source_paths["val"]


def convert_to_path(unit: Path) -> list[str]:
    return [str(unit.resolve())]


def resample_fsd50k_audio(
    data_dir: Path,
    output_dir: Path,
    source_sample_rate: int,
    target_sample_rate: int,
) -> None:
    audio_output_dir = output_dir / "audio"
    audio_output_dir.mkdir(parents=False, exist_ok=False)
    audio_dir_list = [data_dir / "FSD50K.dev_audio", data_dir / "FSD50K.eval_audio"]
    src_path_list: list[Path] = []
    for audio_dir in audio_dir_list:
        src_path_list.extend(sorted(audio_dir.glob("*.wav")))
    assert len(src_path_list) == 51197

    for src_path in tqdm(src_path_list):
        wav, original_sample_rate, subtype = read_sf(wav_path=src_path)
        assert wav.ndim == 1
        assert original_sample_rate == source_sample_rate
        wav = librosa.resample(wav, orig_sr=source_sample_rate, target_sr=target_sample_rate, axis=0)
        relative_path = src_path.relative_to(data_dir)
        output_path = audio_output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_sf(
            wav_path=output_path,
            audio=wav,
            sr=target_sample_rate,
            subtype=subtype,
        )


def main(data_dir: Path, generate_path_list: bool = True) -> None:
    check_data_directory(data_dir=data_dir)
    output_dir = create_output_dir(script_path=Path(__file__))
    fsd50k_ground_truth_dir = data_dir / "FSD50K.ground_truth"
    resample_fsd50k_audio(
        data_dir=data_dir,
        output_dir=output_dir,
        source_sample_rate=SOURCE_SAMPLE_RATE,
        target_sample_rate=SAMPLE_RATE,
    )

    train_source_paths, valid_source_paths = split_dev_paths(
        output_dir=output_dir, fsd50k_ground_truth_dir=fsd50k_ground_truth_dir
    )
    eval_source_paths = sorted((output_dir / "audio/FSD50K.eval_audio").glob("*.wav"))
    assert len(eval_source_paths) == 10231

    split_to_path_list = {
        "train": train_source_paths,
        "valid": valid_source_paths,
        "test": eval_source_paths,
    }
    run_split_manifest_pipeline(
        output_dir=output_dir,
        split_to_unit_list=split_to_path_list,
        unit_to_path_list_func=convert_to_path,
        expected_total_path_count=51197,
        expected_channels=1,
        expected_sample_rate=SAMPLE_RATE,
        generate_path_list=generate_path_list,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", type=Path, required=True)
    argparser.add_argument("--skip_path_list", action="store_true")
    args = argparser.parse_args()
    main(data_dir=args.data_dir, generate_path_list=not args.skip_path_list)
