#!/bin/bash -l
# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

set -eu
cd "$(git rev-parse --show-toplevel)"


mkdir -p scratch/dataset/processed

# DEMAND
# Expected directory structure: scratch/dataset/original/demand/<recording>/ch01.wav ... ch16.wav
uv run python -m nassl.cli.preprocessing.base_training_data.demand \
--data_dir "scratch/dataset/original/demand"

# Expected directory structure: scratch/dataset/original/qut_noise/QUT-NOISE/QUT-NOISE/*.wav
uv run python -m nassl.cli.preprocessing.base_training_data.qut_noise \
--data_dir "scratch/dataset/original/qut_noise"

# Expected directory structure: scratch/dataset/original/wham/high_res_wham/{audio,high_res_metadata.csv}
uv run python -m nassl.cli.preprocessing.base_training_data.wham \
--data_dir "scratch/dataset/original/wham"

# Expected directory structure: scratch/dataset/original/fsd50k/{FSD50K.dev_audio,FSD50K.eval_audio,FSD50K.ground_truth}
uv run python -m nassl.cli.preprocessing.base_training_data.fsd50k \
--data_dir "scratch/dataset/original/fsd50k"
