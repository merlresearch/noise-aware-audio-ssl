# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# http://wham.whisper.ai/
# WHAM!48kHz noise dataset.

cd "$(git rev-parse --show-toplevel)"
mkdir -p scratch/dataset/original/wham
cd scratch/dataset/original/wham

wget -c --tries=0 --read-timeout=30 --timeout=30 \
https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/high_res_wham.zip
unzip -o high_res_wham.zip
