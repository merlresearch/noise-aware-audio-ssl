# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# https://research.qut.edu.au/saivt/databases/qut-noise-databases-and-protocols/
# https://github.com/qutsaivt/QUT-NOISE/tree/master?tab=readme-ov-file

cd "$(git rev-parse --show-toplevel)"
mkdir -p scratch/dataset/original/qut_noise
cd scratch/dataset/original/qut_noise

wget -c --tries=0 --read-timeout=30 --timeout=30 \
https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/8342a090-89e7-4402-961e-1851da11e1aa/download/qutnoise.zip
unzip -o qutnoise.zip

wget -c --tries=0 --read-timeout=30 --timeout=30 \
https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/9b0f10ed-e3f5-40e7-b503-73c2943abfb1/download/qutnoisecafe.zip
unzip -o qutnoisecafe.zip

wget -c --tries=0 --read-timeout=30 --timeout=30 \
https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/7412452a-92e9-4612-9d9a-6b00f167dc15/download/qutnoisecar.zip
unzip -o qutnoisecar.zip

wget -c --tries=0 --read-timeout=30 --timeout=30 \
https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/35cd737a-e6ad-4173-9aee-a1768e864532/download/qutnoisehome.zip
unzip -o qutnoisehome.zip

wget -c --tries=0 --read-timeout=30 --timeout=30 \
https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/164d38a5-c08e-4e20-8272-793534eb10c7/download/qutnoisereverb.zip
unzip -o qutnoisereverb.zip

wget -c --tries=0 --read-timeout=30 --timeout=30 \
https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/10eeceae-9f0c-4556-b33a-dcf35c4f4db9/download/qutnoisestreet.zip
unzip -o qutnoisestreet.zip
