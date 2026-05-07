#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT_DIR/data/raw"
mkdir -p "$RAW_DIR"

base="https://zenodo.org/api/records/15293562/files"

echo "Downloading DMS_substitutions.csv"
curl -L "$base/DMS_substitutions.csv/content" -o "$RAW_DIR/DMS_substitutions.csv"

echo "Downloading DMS_ProteinGym_substitutions.zip"
curl -L "$base/DMS_ProteinGym_substitutions.zip/content" -o "$RAW_DIR/DMS_ProteinGym_substitutions.zip"

echo "Downloading ProteinGym_AF2_structures.zip"
curl -L "$base/ProteinGym_AF2_structures.zip/content" -o "$RAW_DIR/ProteinGym_AF2_structures.zip"

echo "Done. Files saved to $RAW_DIR"
