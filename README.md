# Proline Probability Graph Pipeline

End-to-end utilities to label proline residues in protein structures, train a graph neural network, and annotate new PDBs with predicted proline probabilities.

## Components

- `graph_generation.py`: converts structures (PDB/mmCIF) into residue graphs with torsion/sidechain features, peptide-bond edges, and binary proline labels. Outputs JSONL (one graph per structure).
- `graph_visualization.py`: renders a single graph (3D scatter) to inspect nodes/edges and labels.
- `train_gcn.py`: trains a graph attention network (GAT+MLP) on the generated graphs with class-weighted BCE, early stopping, and diagnostic plots (loss curves, confusion matrix, PR curve).
- `predict_proline_probabilities.py`: loads a trained model, infers probabilities on a PDB, and writes them to the B-factor column for all atoms in each residue.

## Requirements

- Python 3.10+
- [Biopython](https://biopython.org/) (structure parsing)
- PyTorch + PyTorch Geometric
- NumPy, Matplotlib, scikit-learn

Install (example):

```bash
pip install biopython torch torchvision torchaudio pyg-lib torch_geometric numpy matplotlib scikit-learn
```

## Usage

### 1. Generate Graphs

```bash
python graph_generation.py \
  --dataset-dir /media/data/jri/cath_S40/whole-dataset \
  --output-path graphs.jsonl
```

Key features per residue:

- Cα coordinates
- Backbone torsions (φ, ψ, ω) encoded as cos/sin pairs
- Sidechain heavy-atom counts
- Binary label (`1` if residue is `PRO`, else `0`)

### 2. Visualize a Graph (Optional)

```bash
python graph_visualization.py \
  --graph-file graphs.jsonl \
  --structure-id 2g3aA01 \
  --output-path example.png
```

### 3. Train the GAT Model

```bash
python train_gcn.py \
  --graph-file graphs.jsonl \
  --hidden-dim 32 \
  --epochs 500 \
  --batch-size 8 \
  --device cuda \
  --model-out models/proline_gat.pt \
  --report-path reports/training_report.png \
  --early-stop-patience 30
```

Notes:

- Class imbalance is handled via automatic `pos_weight` computation.
- Early stopping monitors validation loss.
- The optional report PNG includes loss curves, test confusion matrix, and precision-recall curve.

### 4. Predict on New Structures

```bash
python predict_proline_probabilities.py \
  --model-path models/proline_gat.pt \
  --pdb-path /media/data/jri/cath_S40/whole-dataset/2g3aA01.pdb \
  --output-path annotated/2g3aA01_probs.pdb \
  --hidden-dim 32 \
  --device cuda
```

The output PDB stores per-residue proline probabilities in the B-factor column (identical value for all atoms in a residue).

## Tips

- If probabilities are uniformly low, consider increasing `pos_weight`, widening the hidden dimension, or standardizing features before training.
- Use `--max-structures` in `graph_generation.py` for rapid prototyping.
- Run `graph_visualization.py` to verify that torsions, edges, and labels look as expected before training.

