#!/bin/bash -l
# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

set -eu
cd "$(git rev-parse --show-toplevel)"

# IR ##################################
config_name="random_rir"
uv run python -m nassl.cli.preprocessing.dcase2026.generate_ir \
--config-name ${config_name} \
n_ir=1000 name="${config_name}/valid" seed=1

uv run python -m nassl.cli.preprocessing.dcase2026.generate_ir \
--config-name ${config_name} \
n_ir=10000 name="${config_name}/train" seed=0

# Noisy ###############################
config_name="${config_name}_snr_-10_10_4noise"
uv run python -m nassl.cli.preprocessing.dcase2026.generate_noisy \
experiment="${config_name}" name="${config_name}"


# Clean feature extraction ############
split="valid"
wave_key="s"
for model in beats_iter3 eat dasheng; do
uv run python -m nassl.cli.preprocessing.feature_extraction \
model="${model}" \
output_dir="scratch/dcase2026/feature/${config_name}/${split}/${wave_key}_${model}" \
manifest_test_path="scratch/dcase2026/noisy/${config_name}/${split}/manifest_test.json" \
wave_key="${wave_key}"
done
