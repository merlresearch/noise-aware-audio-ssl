#!/bin/bash -l
# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

set -eu

exp_list=(
    "dcase2026/nabeats"
    "dcase2026/naeat"
    "dcase2026/nadasheng"
)

cd "$(git rev-parse --show-toplevel)"

for seed in 0 1 2; do
for exp in "${exp_list[@]}"; do
uv run python -m nassl.cli.train experiment="${exp}" name="${exp}" seed="${seed}"
done
done
