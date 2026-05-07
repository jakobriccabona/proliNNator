# ProLinnator

ProLinnator predicts per-residue probability that a site is mutable to Proline.

Given mutation data with columns like `WT`, `POS`, `MUT_TO_PRO`, `DDG` (plus protein/structure identifiers), it trains a graph neural network on residue graphs built from PDB structures.

## What this repository includes

- GraphSAGE per-residue classifier with sigmoid probability output.
- Flexible CSV normalization for common column name variants.
- PDB residue graph construction (sequence edges + 3D contact edges).
- Masked loss so only experimentally measured positions contribute to training.
- Helper script to fetch ProteinGym metadata and AF2 structures from Zenodo.

## Software requirements

The project was set up and run on macOS.

Required software:
- Python 3.10 or newer (tested with Python 3.13)
- pip (inside a virtual environment)
- unzip
- curl

Python packages are listed in `requirements.txt` and include:
- numpy
- pandas
- scikit-learn
- matplotlib
- biopython
- torch
- torch-geometric
- tqdm

## Installation and run guide (exact workflow used)

All commands below were run from the project root directory.

1. Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Upgrade pip and install dependencies.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Fetch ProteinGym metadata and AlphaFold2 structures.

```bash
mkdir -p data/raw
curl -L "https://zenodo.org/api/records/15293562/files/DMS_substitutions.csv/content" -o data/raw/DMS_substitutions.csv
curl -L "https://zenodo.org/api/records/15293562/files/DMS_ProteinGym_substitutions.zip/content" -o data/raw/DMS_ProteinGym_substitutions.zip
curl -L "https://zenodo.org/api/records/15293562/files/ProteinGym_AF2_structures.zip/content" -o data/raw/ProteinGym_AF2_structures.zip
```

4. Extract downloaded data.

```bash
unzip -q -o data/raw/DMS_ProteinGym_substitutions.zip -d data/raw
mkdir -p data/raw/structures
unzip -q -o data/raw/ProteinGym_AF2_structures.zip -d data/raw/structures
```

5. Build a Proline-only training table from ProteinGym substitutions.

```bash
python scripts/build_proline_dataset_from_proteingym.py \
  --metadata-csv data/raw/DMS_substitutions.csv \
  --dms-dir data/raw/DMS_ProteinGym_substitutions \
  --out data/processed/proline_training_from_proteingym.csv
```

6. Train and evaluate the model (train/val/test split with plot generation).

```bash
PYTHONPATH=src python -m prolinnator.train \
  --csv data/processed/proline_training_from_proteingym.csv \
  --structures-dir data/raw/structures/ProteinGym_AF2_structures \
  --loss focal \
  --focal-gamma 2.0 \
  --epochs 20 \
  --batch-size 8 \
  --hidden-dim 128 \
  --layers 3 \
  --dropout 0.2 \
  --early-stopping-monitor val_pr_auc \
  --early-stopping-patience 30 \
  --early-stopping-min-delta 0.001 \
  --output-dir outputs/run_2026-05-04_02
```

Notes:
- Default loss is now focal loss (`--loss focal`).
- Add `--use-pos-weight` to additionally apply class-ratio weighting.
- You can switch back to BCE with `--loss bce`.

7. Check generated outputs.

Main artifacts:
- `outputs/run_2026-05-04_02/history.csv` (epoch-wise train and val loss)
- `outputs/run_2026-05-04_02/test_summary.json`
- `outputs/run_2026-05-04_02/metrics/*.json`
- `outputs/run_2026-05-04_02/plots/*.png`

The plots include:
- loss curves
- precision-recall curves
- logarithmic ROC curves
- normalized confusion matrices (with counts)

## Tabular baselines

If you want to test whether local residue features are already sufficient, you can train graph-free baselines on the same residue-level features used by the GNN.

Implemented baselines:
- random forest
- linear SVM
- small MLP
- XGBoost
- LightGBM
- CatBoost

These models use the same per-residue feature vector as the GNN node input:
- amino-acid one-hot
- backbone torsion features
- optional pretrained residue embeddings

Example:

```bash
PYTHONPATH=src python -m prolinnator.train_tabular \
  --csv data/processed/proline_training_from_proteingym_aligned.csv \
  --structures-dir data/raw/structures/ProteinGym_AF2_structures \
  --embeddings-dir data/processed/embeddings_esm2_t33 \
  --embedding-strict \
  --models random_forest linear_svm mlp xgboost lightgbm catboost \
  --output-dir outputs/tabular_baselines_esm2
```

For honest comparison to the GNN, keep the same grouped protein split fractions.

## Early stopping

Early stopping is built into `prolinnator.train`.

Useful arguments:
- `--early-stopping-monitor {val_pr_auc,val_loss}`
- `--early-stopping-patience <int>`
- `--early-stopping-min-delta <float>`

Recommended defaults for this task:
- monitor: `val_pr_auc`
- patience: `20` to `40`
- min-delta: `0.001`

Set `--early-stopping-patience 0` to disable early stopping.

## Pretrained residue embeddings (optional)

You can add pretrained embeddings to each residue node.

How this is used:
- Base node features (amino-acid one-hot) are concatenated with embedding vectors.
- Final node feature per residue is `[one_hot || embedding]`.

Enable this with:

```bash
PYTHONPATH=src python -m prolinnator.train \
  --csv data/processed/proline_training_from_proteingym.csv \
  --structures-dir data/raw/structures/ProteinGym_AF2_structures \
  --embeddings-dir data/processed/embeddings \
  --embedding-strict \
  --epochs 100 \
  --early-stopping-monitor val_pr_auc \
  --early-stopping-patience 30 \
  --output-dir outputs/run_with_embeddings
```

Supported embedding files in `--embeddings-dir`:
- `<protein_id>.npy`
- `<protein_id>.npz`
- `<pdb_stem>.npy`
- `<pdb_stem>.npz`

Expected formats:
- `.npy`: shape `[num_residues, emb_dim]` (must match residue order and length)
- `.npz` with key `embeddings`: either
  - shape `[num_residues, emb_dim]` (same order as structure residues), or
  - shape `[N, emb_dim]` plus key `residue_numbers` for explicit mapping by PDB residue number

If `--embedding-strict` is set, missing or malformed embedding files raise an error.

## Quick start (custom data)

1. Create/activate your Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Put your mutation CSV in `data/raw/`.
4. Put your PDBs in `data/raw/structures/`.
5. Train:

```bash
python -m prolinnator.train \
  --csv data/raw/your_megascale.csv \
  --structures-dir data/raw/structures \
  --single-pdb your_protein.pdb \
  --ddg-threshold 1.0 \
  --epochs 30
```

For multi-protein training, include `protein_id`, `pdb_file`, and optionally `chain_id` columns in your CSV, then omit `--single-pdb`.

## Expected CSV columns

Required mutation columns:
- `WT` (wild-type AA)
- `POS` (1-based residue position, mapped to PDB residue number)
- `MUT_TO_PRO` or equivalent (`MUT`, `MUTANT`, etc.)
- `DDG`

Recommended grouping columns for multi-protein training:
- `protein_id`
- `pdb_file`
- `chain_id` (optional, default `A`)

Notes:
- The pipeline keeps only rows where mutant AA is Proline (`P`).
- Label can come from either:
  - `DDG` with thresholding (`1` if `DDG <= ddg_threshold`, else `0`)
  - `LABEL` if pre-binarized labels are already present

## Fetch ProteinGym assets

```bash
bash scripts/fetch_proteingym_assets.sh
```

This downloads:
- `data/raw/DMS_substitutions.csv`
- `data/raw/DMS_ProteinGym_substitutions.zip`
- `data/raw/ProteinGym_AF2_structures.zip`

You can unzip AF2 structures with:

```bash
unzip -q data/raw/ProteinGym_AF2_structures.zip -d data/raw/structures
```
