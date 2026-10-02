# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# https://zenodo.org/records/1227121
# DEMAND 16 kHz files.

cd "$(git rev-parse --show-toplevel)"
mkdir -p scratch/dataset/original/demand
cd scratch/dataset/original/demand

for filename in \
DKITCHEN_16k.zip \
DLIVING_16k.zip \
DWASHING_16k.zip \
NFIELD_16k.zip \
NPARK_16k.zip \
NRIVER_16k.zip \
OOFFICE_16k.zip \
OHALLWAY_16k.zip \
OMEETING_16k.zip \
PCAFETER_16k.zip \
PRESTO_16k.zip \
PSTATION_16k.zip \
SPSQUARE_16k.zip \
STRAFFIC_16k.zip \
TBUS_16k.zip \
TCAR_16k.zip \
TMETRO_16k.zip
do
wget -c --tries=0 --read-timeout=30 --timeout=30 \
https://zenodo.org/records/1227121/files/${filename}
unzip -o ${filename}
done
