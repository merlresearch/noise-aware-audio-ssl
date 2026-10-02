#!/bin/bash -l
# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

set -eu
cd "$(git rev-parse --show-toplevel)"


target_name="fsd50k"
noise_name="wham_snr_-5_10"
split=valid

# Generate noisy signals
uv run python -m nassl.cli.preprocessing.iwaenc2026.generate_noisy \
experiment="${target_name}_${noise_name}" split="${split}"


# Extract clean features for validation
model="beats_iter3"
wave_key="s"

uv run python -m nassl.cli.preprocessing.feature_extraction \
model="${model}" \
output_dir="scratch/iwaenc2026/feature/${target_name}_${noise_name}/${split}/${wave_key}_${model}" \
manifest_test_path="scratch/iwaenc2026/noisy/${target_name}/${noise_name}/${split}_noise/manifest_test.json" \
wave_key="${wave_key}"
