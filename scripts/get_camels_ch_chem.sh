#!/bin/bash
# Download CAMELS-CH-Chem (Nascimento et al. 2025, CC BY 4.0) into data/camels-ch-chem.
# About 170 MB download. Run from the repo root:  bash scripts/get_camels_ch_chem.sh
set -euo pipefail
DEST="${1:-data/camels-ch-chem}"
URL="https://zenodo.org/api/records/14980027/files/camels-ch-chem.zip/content"
if [ -d "$DEST/stream_water_chemistry/timeseries/daily" ]; then
    echo "CAMELS-CH-Chem already in $DEST ($(ls "$DEST/stream_water_chemistry/timeseries/daily" | wc -l) daily files)"
    exit 0
fi
TMP="$(mktemp -d "${TMPDIR:-/tmp}/camels.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
echo "Downloading CAMELS-CH-Chem from Zenodo..."
wget -q --show-progress -O "$TMP/camels.zip" "$URL"
echo "Unzipping..."
unzip -q "$TMP/camels.zip" -d "$TMP/x"
# the zip has one top-level folder; find the folder that contains stream_water_chemistry
SRC="$(dirname "$(find "$TMP/x" -type d -name stream_water_chemistry | head -1)")"
mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST"
mv "$SRC" "$DEST"
echo "Done: $DEST ($(ls "$DEST/stream_water_chemistry/timeseries/daily" | wc -l) daily gauge files)"
