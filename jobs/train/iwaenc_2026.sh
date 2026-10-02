#!/bin/bash -l
# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

set -eu

exp_list=(
    "iwaenc2026/nabeats_film"
    "iwaenc2026/nabeats_ca"
    "iwaenc2026/dbeats"
)

cd "$(git rev-parse --show-toplevel)"

for exp in "${exp_list[@]}"
do
uv run python -m nassl.cli.train experiment="${exp}" name="${exp}"
done
