# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# https://zenodo.org/records/4060432
# FSD50K

cd "$(git rev-parse --show-toplevel)"
mkdir -p scratch/dataset/original/fsd50k
cd scratch/dataset/original/fsd50k

for filename in \
FSD50K.dev_audio.z01 \
FSD50K.dev_audio.z02 \
FSD50K.dev_audio.z03 \
FSD50K.dev_audio.z04 \
FSD50K.dev_audio.z05 \
FSD50K.dev_audio.zip \
FSD50K.eval_audio.z01 \
FSD50K.eval_audio.zip \
FSD50K.ground_truth.zip \
FSD50K.metadata.zip \
FSD50K.doc.zip
do
wget -c --tries=0 --read-timeout=30 --timeout=30 \
https://zenodo.org/records/4060432/files/${filename}
done

zip -s 0 FSD50K.dev_audio.zip --out FSD50K.dev_audio.unsplit.zip
unzip -o FSD50K.dev_audio.unsplit.zip

zip -s 0 FSD50K.eval_audio.zip --out FSD50K.eval_audio.unsplit.zip
unzip -o FSD50K.eval_audio.unsplit.zip

unzip -o FSD50K.ground_truth.zip
unzip -o FSD50K.metadata.zip
unzip -o FSD50K.doc.zip
